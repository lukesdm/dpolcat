import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import os
    from pathlib import Path

    import dask
    import dask.distributed
    import geopandas as gpd
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray
    import numpy as np
    import odc.stac
    import pandas as pd
    import pystac_client
    import planetary_computer
    import rioxarray
    import shapely
    import shapely.geometry as sg
    import xarray
    from xhistogram.xarray import histogram
    import zarr

    from dpolcat.pc_rio_driver import PCRioDriver

    hv.extension("bokeh")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # dpolcat EDA - Scaling revisited
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Helpers
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### STAC Search
    """)
    return


@app.function
def find_stac_items(aoi: gpd.GeoDataFrame, date_range: str):
    """
    Find Sentinel-1 RTC scenes, grouped by orbit/track. Uses Microsoft Planetary Computer.
    - `aoi` is a single-row GeoDataFrame.
    - `date_range` is a string with format `{from_date}/{to_date}`, e.g., "2020-01-01/2020-03-31".
    """
    STAC_COLLECTION = "sentinel-1-rtc"

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )

    if len(aoi.geometry) != 1:
        raise ValueError(
            "Expected only a single geometry for the given AoI."
        )

    search = catalog.search(
        collections=[STAC_COLLECTION],
        datetime=date_range,
        # intersects=search_centroid,
        intersects=aoi.to_crs("EPSG:4326").geometry[0],
    )

    items = search.item_collection()

    stac_gdf = gpd.GeoDataFrame.from_features(items)

    # Group items by ascending/descending & relative orbit pairs.
    orbits = {
        k: g
        for k, g in stac_gdf.groupby(
            ["sat:orbit_state", "sat:relative_orbit"]
        )
    }

    grouped_items = {
        orbit: [items[i] for i in orbits[orbit].index] for orbit in orbits
    }

    return grouped_items


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Data loading
    """)
    return


@app.function
def get_epsg(stac_items):
    _epsgs = set(
        [int(item.properties["proj:code"][5:]) for item in stac_items]
    )
    if len(_epsgs) != 1:
        raise ValueError(
            "Items are of different CRSs, this is not supported."
        )

    epsg = _epsgs.pop()
    return epsg


@app.function
def prep_ds(stac_items, aoi, resolution):
    """
    Prepare lazy Dask-backed XArray dataset
    """
    epsg = get_epsg(stac_items)
    ds = odc.stac.load(
        stac_items,
        bands=["vv", "vh"],
        bbox=aoi.to_crs(4326).total_bounds,
        crs=f"EPSG:{epsg}",
        resolution=resolution,
        chunks={"time": 10, "x": 1024, "y": 1024},
        nodata=np.nan,  # override STAC-provided
        groupby="solar_day",  # datatake_id would be more appropriate (and is used subsequently)
        driver=PCRioDriver(),
    )
    return ds


@app.function
def get_zarr_group(zarr_store, zarr_path):
    try:
        g = zarr.open_group(zarr_store, path=zarr_path, mode="r")
    except FileNotFoundError:
        return None
    return g


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analysis
    """)
    return


@app.cell
def _():
    DB_BINS = np.concat(
        [
            np.array([-np.inf]),
            np.arange(-40.0, 10.0, 0.5),
            np.array([np.inf]),
        ]
    )
    return (DB_BINS,)


@app.function
def db_scale(x):
    return 10 * np.log10(x)


@app.cell
def _(DB_BINS):
    def calc_histograms(backscatter_db: xarray.Dataset) -> xarray.Dataset:
        result = xarray.Dataset(
            {
                "vv": histogram(
                    backscatter_db["vv"], bins=DB_BINS, dim=["y", "x"]
                ),
                "vh": histogram(
                    backscatter_db["vh"], bins=DB_BINS, dim=["y", "x"]
                ),
            }
        )
        return result

    return (calc_histograms,)


@app.function
def calc_nodata(ds: xarray.Dataset):
    _sizes = ds.sizes
    px_count = _sizes["x"] * _sizes["y"]
    nan_count = np.isnan(ds).sum(dim=["y", "x"])
    return nan_count, px_count


@app.function
def filter_incomplete(
    stac_items, aoi, resolution, max_nodata_fraction=0.01
):
    backscatter_lin = prep_ds(stac_items, aoi, resolution)
    nodata_count, px_count = calc_nodata(backscatter_lin)
    nodata_fraction = (nodata_count / px_count).compute()

    return nodata_fraction.where(
        nodata_fraction < max_nodata_fraction, drop=True
    )


@app.cell
def _(calc_histograms):
    def perform_analysis(stac_items, aoi, resolution, time_index):
        backscatter_lin = prep_ds(stac_items, aoi, resolution)
        backscatter_lin = backscatter_lin.sel(time=time_index)
        nodata_count, px_count = calc_nodata(backscatter_lin)
        nodata_fraction = nodata_count / px_count
        backscatter_db = db_scale(backscatter_lin)
        result = calc_histograms(backscatter_db)
        result["px_count"] = px_count
        result["vv_nodata_fraction"] = nodata_fraction["vv"]
        result["vh_nodata_fraction"] = nodata_fraction["vh"]
        return result

    return (perform_analysis,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analysis

    - Mosaic scenes for the selected AoI based on orbit direction and relative orbit #.
    - Calculate backscatter histograms, pixel counts and NoData percentages.
    """)
    return


@app.cell
def _():
    # Get/start local dask client with threaded scheduler
    try:
        dask_client = dask.distributed.get_client()
    except ValueError:
        dask_client = dask.distributed.Client(processes=False)
    dask_client
    return


@app.cell
def _():
    aoi_files = {
        "gran_canaria": "gis/aois/gran_canaria_4326.gpkg",
        "inverness": "gis/aois/inverness_4326.gpkg",
        "titicaca": "gis/aois/lapaz_titicaca_4326.gpkg",
        "salzburg": "gis/aois/salzburg_south_4326.gpkg",
    }
    return (aoi_files,)


@app.cell
def _():
    # date_range = "2020-08-01/2020-08-31"  # COULDDO: Parameterize
    date_range = "2025-01-01/2025-12-31"
    return (date_range,)


@app.cell
def _():
    resolution = 10
    preview_resolution=1000
    return preview_resolution, resolution


@app.cell
def _(aoi_files):
    aois = {
        aoi_name: gpd.read_file(aoi_files[aoi_name]) for aoi_name in aoi_files
    }
    return (aois,)


@app.cell
def _(aois, date_range):
    stac_groups = {
        aoi_name: find_stac_items(aois[aoi_name], date_range)
        for aoi_name in aois
    }
    return (stac_groups,)


@app.cell
def _(stac_groups):
    df_inputs = pd.DataFrame(
        [(aoi, direction, orbit) for aoi, groups in stac_groups.items() for direction, orbit in groups],
        columns=["aoi_name", "direction", "relative_orbit"],
    )
    return (df_inputs,)


@app.cell
def _(df_inputs):
    ui_input_selector = mo.ui.table(df_inputs, page_size=30)
    ui_input_selector
    return (ui_input_selector,)


@app.cell
def _(ui_input_selector):
    mo.stop(len(ui_input_selector.value) == 0)

    selected_inputs = ui_input_selector.value
    return (selected_inputs,)


@app.cell
def _(ui_input_selector):
    ui_begin_analysis = mo.ui.run_button(
        label="Begin analysis", disabled=len(ui_input_selector.value) == 0
    )
    ui_begin_analysis
    return (ui_begin_analysis,)


@app.cell
def _():
    result_store = "data/histograms.zarr"
    return (result_store,)


@app.cell
def _(
    aois,
    date_range,
    perform_analysis,
    preview_resolution,
    resolution,
    result_store,
    selected_inputs,
    stac_groups,
    ui_begin_analysis,
):
    mo.stop(not ui_begin_analysis.value, "Click 👆 to begin.")

    zarr_root = zarr.open_group(result_store, mode="a")

    results_new = {}
    for _idx, _row in selected_inputs.iterrows():
        _aoi_name = _row["aoi_name"]
        _direction = _row["direction"]
        _relative_orbit = _row["relative_orbit"]
        _aoi = aois[_aoi_name]
        _date_range = date_range.replace(
            "/", "_"
        )  # for zarr path compatibility
        _now = pd.Timestamp.now("UTC")

        _zarr_path = (
            f"{_aoi_name}/{_date_range}_{resolution}m/{_direction}_{_relative_orbit}"
        )

        if _existing_group := get_zarr_group(result_store, _zarr_path):
            print(f"Found existing results at {_zarr_path}, skipping work.")
            # Load already-processed results instead.
            results_new[(_aoi_name, _direction, _relative_orbit)] = (
                xarray.open_zarr(result_store, _zarr_path).load()
            )
            continue

        print(
            f"{_now}: Processing for {_aoi_name}, {_direction}, rel. orbit {_relative_orbit}"
        )

        _stac_items = stac_groups[_aoi_name][(_direction, _relative_orbit)]
        _to_process = filter_incomplete(
            _stac_items,
            _aoi,
            resolution=preview_resolution,
            max_nodata_fraction=0.01,
        )

        if len(_to_process.time) == 0:
            print("No complete scenes, skipping.")
            continue
        else:
            print(f"{len(_to_process.time)} timesteps to process.")

        _result = perform_analysis(
            stac_items=_stac_items,
            aoi=aois[_aoi_name],
            resolution=resolution,
            time_index=_to_process.time,
        )

        _result = dask.compute(_result)[0]

        _result["aoi_name"] = _aoi_name
        _result["direction"] = _direction
        _result["relative_orbit"] = _relative_orbit
        _result["resolution"] = resolution

        # _result = _result.assign_attrs(
        #     aoi_name=_aoi_name,
        #     direction=_direction,
        #     relative_orbit=_relative_orbit,
        # )

        _result.to_zarr(result_store, group=_zarr_path, mode="w")

        results_new[(_aoi_name, _direction, _relative_orbit)] = _result
    return (results_new,)


@app.cell
def _(result_store, results_new):
    # Add depdency to force re-running this cell when new results are available.
    # If you just want to open existing results, comment this out:
    _ = results_new

    results_tree = xarray.open_datatree(result_store, engine="zarr")
    return (results_tree,)


@app.cell
def _(results_tree):
    # Get flattened results
    results = {}
    for _aoi_name, _aoi_group in results_tree.children.items():
        _merged = xarray.merge(_aoi_group.children.values())
        for _g, _ds in _merged.items():
            _direction, _relative_orbit = _g.split("_")
            results[(_aoi_name, _direction, _relative_orbit)] = _ds.load()
    return (results,)


@app.cell
def _():
    # Snippet for deleting a zarr group
    # del zarr_root["xxx/yyy"]
    # zarr.consolidate_metadata(result_store)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualization
    """)
    return


@app.cell
def _():
    ui_viz_enabled = mo.ui.switch(label="Viz enabled")
    ui_viz_enabled
    return (ui_viz_enabled,)


@app.cell
def _(DB_BINS):
    def plot_hist(hist_ds, pol):
        # Histogram plot
        # This is complicated because it mixes hvplot features and
        # features from the outer holoviews.
        # Also colours the overflow (inf) bins red.
        # Mostly Clauded (Opus 5.5)

        bin_dim = f"{pol}_bin"
        lo, hi = DB_BINS[1] - 0.5, DB_BINS[-2] + 0.5
        xs = hist_ds[bin_dim].values.copy()
        xs[[0, -1]] = lo, hi

        return (
            hist_ds[pol]
            .assign_coords({bin_dim: xs})
            .hvplot.line(x=bin_dim)
            .apply(hv.Histogram)
            .opts(
                width=450,
                tools=["hover"],
                fill_color=hv.dim(bin_dim).categorize(
                    {lo: "red", hi: "red"}, default="#30a2da"
                ),
            )
        )

    return (plot_hist,)


@app.cell
def _(results, ui_viz_enabled):
    mo.stop(len(results) == 0 or not ui_viz_enabled.value)

    _options = list(results.keys())
    ui_result_acq = mo.ui.dropdown(options=_options, value=_options[0])
    ui_result_acq
    return (ui_result_acq,)


@app.cell
def _(plot_hist, results, ui_result_acq):
    sel_result_acq = ui_result_acq.value

    plot_hist(results[sel_result_acq], "vv") + plot_hist(results[sel_result_acq], "vh")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Test
    """)
    return


@app.cell
def _():
    # test_aoi_file = "gis/aois/inverness_4326.gpkg"
    # test_aoi_file = "gis/aois/salzburg_south_4326.gpkg"
    # test_aoi_file = "gis/aois/gran_canaria_4326.gpkg"
    test_aoi_file = "gis/aois/lapaz_titicaca_4326.gpkg"
    test_aoi = gpd.read_file(test_aoi_file)
    return (test_aoi,)


@app.cell
def _(test_aoi):
    # test_items_grouped = find_stac_items(test_aoi, "2020-08-01/2020-08-31")
    test_items_grouped = find_stac_items(test_aoi, "2025-05-01/2025-05-31")
    return (test_items_grouped,)


@app.cell
def _(test_items_grouped):
    test_orbits = list(test_items_grouped.keys())
    ui_sel_test_orbit = mo.ui.dropdown(options=test_orbits, value=test_orbits[0])
    ui_sel_test_orbit
    return (ui_sel_test_orbit,)


@app.cell
def _(test_items_grouped, ui_sel_test_orbit):
    sel_test_orbit = ui_sel_test_orbit.value
    test_items = test_items_grouped[sel_test_orbit]
    return (test_items,)


@app.cell
def _(test_items):
    test_items_gdf = gpd.GeoDataFrame.from_features(test_items, crs="EPSG:4326")
    test_items_gdf["id"] = [item.id for item in test_items]
    test_items_gdf["datetime"] = pd.to_datetime(test_items_gdf["datetime"])
    return (test_items_gdf,)


@app.cell
def _(test_aoi, test_items_gdf):
    test_intersection = test_items_gdf.dissolve(
        ["sat:orbit_state", "sat:relative_orbit", "s1:datatake_id"]
    ).intersection(test_aoi.geometry[0])
    return (test_intersection,)


@app.cell
def _(test_intersection):
    test_intersection.to_file("data/test_intersection.gpkg")
    return


@app.cell
def _(test_aoi, test_items):
    test_resolution=500
    test_ds = prep_ds(test_items, test_aoi, resolution=test_resolution)
    return (test_ds,)


@app.cell
def _(test_ds):
    test_nan_count, test_px_count = calc_nodata(test_ds)
    (test_nan_count / test_px_count).compute()
    return


@app.cell
def _(test_aoi, test_items):
    filter_incomplete(test_items, test_aoi, resolution=1000, max_nodata_fraction=0.5)
    return


@app.cell
def _(test_aoi, test_ds, test_items):
    test_ds_filtered = test_ds.sel(
        time=filter_incomplete(
            test_items, test_aoi, resolution=1000, max_nodata_fraction=0.5
        ).time
    )
    test_ds_filtered
    return (test_ds_filtered,)


@app.cell
def _(test_ds_filtered):
    test_ds_db = db_scale(test_ds_filtered)
    return (test_ds_db,)


@app.cell
def _(test_ds_db):
    def plot_backscatter(da):
        return da.hvplot.image(x="x", y="y", cmap="gray", aspect=1.0)

    plot_backscatter(test_ds_db["vv"]).opts(xrotation=30) + plot_backscatter(test_ds_db["vh"]).opts(xrotation=30)
    return


@app.cell
def _(calc_histograms, test_ds_db):
    test_hists = calc_histograms(test_ds_db)
    test_hists = test_hists.compute()
    return (test_hists,)


@app.cell
def _(plot_hist, test_hists):
    plot_hist(test_hists, "vv") + plot_hist(test_hists, "vh")
    return


if __name__ == "__main__":
    app.run()
