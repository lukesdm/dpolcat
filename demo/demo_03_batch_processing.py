import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from pathlib import Path

    import geopandas as gpd
    import geoviews
    import hvplot.xarray
    import hvplot.pandas

    import dpolcat.utils
    import dpolcat.processor


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## *dpolcat* Demo 03: Batch Processing

    This notebook demonstrates creating a *dpolcat* datacube in [ZARR format](https://zarr.dev/) for a given area of interest (AoI) and time period.

    We will build a *dpolcat* datacube for Graz and it's surroundings, for the month of January 2020.
    """)
    return


@app.cell
def _():
    aoi = gpd.read_file("gis/graz/graz_4326.gpkg")

    date_range = "2020-01-01/2020-01-31"
    return aoi, date_range


@app.cell
def _(aoi):
    aoi.hvplot(geo=True, tiles=True, fill_color=None)

    # Due to a version incompability bug, this plot currently shows the AoI in the wrong place.
    # TODO: update versions when fixed.
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    One of the key differences of Sentinel-1 SAR and Sentinel-2/3 optical imagery is the sensitivity to sensor geometry when capturing the same area across multiple acquisitions. The same area can appear vastly different between captures during ascending vs descending orbits. So, for like-for-like comparisons in a time series, it's best to stick to a single relative orbit.

    There are some helper functions for determining which orbits best cover an AoI, giving the number of fullly-covered scenes and partial scenes. Select an orbit below.
    """)
    return


@app.cell
def _(aoi, date_range):
    stac_items_gdf = dpolcat.utils.find_stac_items(aoi, date_range=date_range)
    df_coverages = dpolcat.utils.calc_coverages(stac_items_gdf, aoi=aoi)
    return df_coverages, stac_items_gdf


@app.cell
def _(df_coverages):
    ui_orbit_selector = mo.ui.table(df_coverages.reset_index(), selection="single")
    ui_orbit_selector
    return (ui_orbit_selector,)


@app.cell
def _(ui_orbit_selector):
    mo.stop(
        len(ui_orbit_selector.value) == 0
        or ui_orbit_selector.value["full_scenes"].values[0] == 0,
        "Selected an entry with at least 1 full scene.",
    )
    sat_direction = ui_orbit_selector.value["sat:orbit_state"].values[0]
    sat_rel_orbit = ui_orbit_selector.value["sat:relative_orbit"].values[0]
    return sat_direction, sat_rel_orbit


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Show the STAC items for the chosen orbit.
    """)
    return


@app.cell
def _(sat_direction, sat_rel_orbit, stac_items_gdf):
    sel_items_gdf = dpolcat.utils.select_by_orbit(
        stac_items_gdf,
        orbit_direction=sat_direction,
        relative_orbit=sat_rel_orbit,
    )
    sel_items = sel_items_gdf["_item"].values
    return (sel_items_gdf,)


@app.cell
def _(sel_items_gdf):
    sel_items_gdf
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Visualize the AoI and item footprints.

    Note: AoI geometry may appear a little off due to a reprojection dependency compatibility issue (see https://github.com/oceanmodeling/stofs-event-dashboard/pull/36).
    """)
    return


@app.cell
def _(aoi, sel_items_gdf, stac_items_gdf):
    (
        stac_items_gdf.hvplot(geo=True, color=None, alpha=0.5, label="Available items")
        * sel_items_gdf.hvplot(
            geo=True,
            alpha=0.2,
            color="orange",
            label="Selected items",
        )
        * aoi.hvplot(
            geo=True, tiles=True, alpha=0.5, color="blue", label="AoI"
        )
    ).opts(tools=["wheel_zoom"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `dpolcat.processor` accepts several parameters. See its [intro](http://localhost:2718/?file=dpolcat.processor.py#dpolcat-processor) for details.

    Define the parameters for our data cube.
    """)
    return


@app.cell
def _():
    out_zarr_path = "data/dpolcat_graz_small.zarr"
    return (out_zarr_path,)


@app.cell
def _(aoi, date_range, out_zarr_path, sat_direction, sat_rel_orbit):
    processor_params = dict(
        p_headless=True,
        p_aoi=aoi,
        p_date_range=date_range,
        p_sat_direction=sat_direction,
        p_sat_rel_orbit=sat_rel_orbit,
        p_speckle_filter="Frost",
        p_speckle_filter_window=5,
        p_out_zarr_path=out_zarr_path,
        p_write_intermediate_data=False,
    )
    return (processor_params,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Click below to start processing. Note the progress bar in the following cell.
    """)
    return


@app.cell
def _():
    ui_start = mo.ui.run_button(label="Start processing")
    ui_start
    return (ui_start,)


@app.cell
def _(out_zarr_path, processor_params, ui_start):
    mo.stop(not ui_start.value, "Click 👆 to start.")

    assert not Path(out_zarr_path).exists(), f"Datacube {out_zarr_path} already exists, please remove and try again."

    _out_cells, result_vars = dpolcat.processor.app.run(
        processor_params
    )
    return (result_vars,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When the child notebook has been run, we can access the variables it created. From the `result` variable we retrieve an Xarray Dataset object from the Zarr store we just made. The data is stored in the Zarr under a group for the orbit.
    """)
    return


@app.cell
def _(result_vars, sat_direction, sat_rel_orbit):
    zarr_group = f"/{sat_direction}_{sat_rel_orbit}/dpolcat"
    ds_created = result_vars["result"][zarr_group]
    return ds_created, zarr_group


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Show the timesteps of the created datacube. Depending on the AoI this may be smaller than the number of input STAC items, because items from the same datatake are mosiacked to try and fully cover the AoI.
    """)
    return


@app.cell
def _(ds_created):
    ds_created.time
    return


@app.cell
def _(ds_created, sel_items_gdf):
    f"{len(sel_items_gdf)} selected STAC items -> {len(ds_created.time)} scenes in the datacube."
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You may have noticed that we didn't pass in our selected STAC items, just the orbit identifiers. The processor performs a STAC search itself. You can extract the source STAC items as follows:
    """)
    return


@app.cell
def _(result_vars):
    result_vars["sel_items_gdf"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appending

    We will now expand the temporal coverage of the data cube to include data for Feb 2020. We'll use the processor params as before but with a new date range.

    /// warning
    After appending (especially retrospectively), when data is loaded, e.g., in *QGIS* or with `xarray.open_groups(...)`, bands may not appear in chronological order. Check the metadata to confirm which bands map to which dates.
    """)
    return


@app.cell
def _():
    date_range_append = "2020-02-01/2020-02-29"
    return (date_range_append,)


@app.cell
def _(date_range_append, processor_params):
    processor_params_append = processor_params | dict(
        p_date_range=date_range_append,
    )
    processor_params_append
    return (processor_params_append,)


@app.cell
def _():
    ui_append = mo.ui.run_button(label="Append to datacube")
    ui_append
    return (ui_append,)


@app.cell
def _(processor_params_append, ui_append):
    mo.stop(not ui_append.value, "Click 👆 to start.")

    _out_cells, append_result_vars = dpolcat.processor.app.run(
        processor_params_append
    )
    return (append_result_vars,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As before, we can inspect the results of our append operation.
    """)
    return


@app.cell
def _(append_result_vars):
    append_result_vars["sel_items_gdf"]
    return


@app.cell
def _(append_result_vars, zarr_group):
    ds_appended = append_result_vars["result"][zarr_group]
    return (ds_appended,)


@app.cell
def _(ds_appended):
    ds_appended.time
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note: this gives us the timesteps for the entire cube (not just the added items).
    """)
    return


@app.cell
def _(append_result_vars, ds_appended):
    f"Appended {len(append_result_vars['sel_items_gdf'])} new STAC items -> {len(ds_appended.time)} total datacube scenes."
    return


if __name__ == "__main__":
    app.run()
