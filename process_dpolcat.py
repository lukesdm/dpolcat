import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pystac_client
    import planetary_computer
    import shapely.geometry as sg
    import stackstac
    import xarray
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray
    import rioxarray

    import dpolcat as dp
    import fast_frost as ff

    hv.extension("bokeh")
    return (
        dp,
        ff,
        gpd,
        hv,
        mo,
        np,
        pd,
        planetary_computer,
        pystac_client,
        sg,
        stackstac,
        xarray,
    )


@app.cell
def _(mo):
    mo.md(r"""## Search and Select Sentinel-1 RTC data""")
    return


@app.cell
def _(mo):
    # AoI bounding boxes
    aois ={
        "Salzburg": [13.02, 47.76, 13.09, 47.83],
        "Tenerife": [-16.96, 27.92, -16.05, 28.61]
    }

    ui_aoi = mo.ui.dropdown(options=aois.keys(), value="Salzburg")
    ui_date_range = mo.ui.date_range(start="2015-01-01", value=("2022-06-01", "2022-06-15"))

    mo.md(f"""
    Area of interest: {ui_aoi} <br>
    Search dates: {ui_date_range} <br>
    """)
    return aois, ui_aoi, ui_date_range


@app.cell
def _(aois, ui_aoi, ui_date_range):
    bbox = aois[ui_aoi.value]

    _d1, _d2 = ui_date_range.value
    date_range = f"{_d1.year}-{_d1.month:02}-{_d1.day:02}/{_d2.year}-{_d2.month:02}-{_d2.day:02}"
    return bbox, date_range


@app.cell
def _(bbox, date_range, mo, pd, planetary_computer, pystac_client):
    def summarize(item_collection):
        """Summarize catalog search results"""
        data = []
        for item in item_collection:
            a_or_d = item.properties["sat:orbit_state"]
            item_epsg = item.properties["proj:code"]
            item_bbox_proj = item.properties["proj:bbox"]
            item_bbox_lonlat = item.bbox

            data.append({
                "id": item.id,
                "date": item.properties["start_datetime"],
                "A/D": a_or_d,
                "epsg": item_epsg,
                "bbox_proj": item_bbox_proj,
                "bbox_lonlat": item_bbox_lonlat
            })

        df = pd.DataFrame(data)
        return df

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-1-rtc"], bbox=bbox, datetime=date_range
    )
    items = search.item_collection()
    print(f"Found {len(items)} items")
    _df = summarize(items)

    ui_items = mo.ui.table(_df)
    ui_items
    return items, ui_items


@app.cell
def _(items, mo, ui_items):
    sel_items = [items[i] for i in ui_items.value.index]
    mo.stop(len(sel_items) == 0, "No items selected")

    _epsgs = set([int(item.properties["proj:code"][5:]) for item in sel_items])
    assert len(_epsgs) == 1, "items are of different CRSs, this is not supported."
    epsg_num = int(_epsgs.pop())
    return epsg_num, sel_items


@app.cell
def _(bbox, gpd, sel_items, sg):
    _aoi = gpd.GeoDataFrame({"geometry": [sg.box(bbox[0], bbox[1], bbox[2], bbox[3])]}, crs=4236)
    _footprints = gpd.GeoDataFrame({"geometry": [sg.shape(item.geometry) for item in sel_items]},crs=4326)

    (
        _aoi.hvplot(geo=True, tiles=True, alpha=0.2, color="blue")
        * _footprints.hvplot(geo=True, alpha=0.2, color="orange")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Processing""")
    return


@app.cell
def _(mo):
    ui_speckle_filter = mo.ui.dropdown(options=["Unfiltered", "Frost"], value="Frost")
    ui_run = mo.ui.run_button(label="⚙️ Perform processing")

    mo.md(f"""
    Speckle filter: {ui_speckle_filter} <br><br>
    {ui_run}
    """)
    return ui_run, ui_speckle_filter


@app.cell
def _(ff, xarray):
    def frost_xr(da: xarray.DataArray, nanny=True, damping_factor=2.0, win_size=5) -> xarray.DataArray:
        """Apply 2D (y, x) Frost filter to 3D (y, x, time) xarray DataArray; See underlying filter function for damping_factor and win_size."""
        filtered = xarray.apply_ufunc(
            ff.frost_filter_nanny if nanny else ff.frost_filter_fast,
            da,
            input_core_dims=[['y', 'x']],
            output_core_dims=[['y', 'x']],
            vectorize=True,
            kwargs=dict(damping_factor=damping_factor, win_size=win_size),
            dask="parallelized",
            output_dtypes=[da.dtype],
        )
        return filtered
    return (frost_xr,)


@app.cell
def _(bbox, epsg_num, mo, sel_items, stackstac, ui_run):
    mo.stop(not ui_run.value, "Click the button above to start processing.")

    # Load data
    resolution = 10 # Native to data source
    ds = stackstac.stack(sel_items, bounds_latlon=bbox, epsg=epsg_num, resolution=resolution)
    vv_lin = ds.sel(band="vv")#.compute()
    vh_lin = ds.sel(band="vh")#.compute()
    return vh_lin, vv_lin


@app.cell
def _(dp, frost_xr, ui_speckle_filter, vh_lin, vv_lin):
    # Apply speckle filtering and scaling
    _filter = ui_speckle_filter.value
    if _filter == "Frost":
        vv_filt = frost_xr(vv_lin.fillna(-1), nanny=False)
        vh_filt = frost_xr(vh_lin.fillna(-1), nanny=False)
        vv_sn = dp.scale_nice(vv_filt)
        vh_sn = dp.scale_nice(vh_filt)
    else:
        vv_sn = dp.scale_nice(vv_lin)
        vh_sn = dp.scale_nice(vh_lin)
    return vh_sn, vv_sn


@app.cell
def _():
    return


@app.cell(disabled=True)
def _(item, vh_sn, vv_sn):
    # TODO: Fix this viz

    _id = item.id

    (vv_sn.hvplot(x="x", y="y", data_aspect=1.0, cmap="gray", title=f"VV (scaled) [{_id}]") +
     vh_sn.hvplot(x="x", y="y", data_aspect=1.0, cmap="gray", title=f"VH (scaled) [{_id}]")
    ).cols(2)
    return


@app.cell
def _(mo):
    mo.md(r"""### Categorization""")
    return


@app.cell
def _(dp, mo):
    @mo.persistent_cache()
    def categorize(vv_sn, vh_sn):
        return dp.xr_categorize(vv_sn, vh_sn).compute()
    return (categorize,)


@app.cell
def _(categorize, epsg_num, vh_sn, vv_sn):
    cat_result = categorize(vv_sn, vh_sn)
    cat_result = cat_result.rio.write_crs(epsg_num)
    return (cat_result,)


@app.cell
def _(dp, hv, np, xarray):
    cat_cmap = {str(i): list(dp.color_list[i]) for i in range(len(dp.color_list))}

    def make_legend(size=350):
        n_steps = 100
        steps   = np.linspace(-0.05, 1, n_steps)
        vv_ramp = np.vstack([steps] * n_steps)
        vh_ramp = vv_ramp.T

        cat_values = xarray.DataArray(
            dp.categorize_np(vv_ramp, vh_ramp),
            coords=[steps, steps], dims=["VH", "VV"]
        )

        label_data = []
        for i in range(1, dp.NUM_CATEGORIES):
            box       = cat_values.where(cat_values == i, drop=True)
            center_x  = float(box["VV"].mean())
            center_y  = float(box["VH"].mean())
            label_data.append((center_x, center_y, str(i)))

        labels = hv.Labels(label_data, kdims=["VV", "VH"], vdims=["text"]).opts(text_color="black")

        return (
            cat_values.astype("str").hvplot(cmap=cat_cmap).opts(height=size, width=size)
            * labels
            * hv.Text(-0.02, -0.02, "0").opts(text_color="white")
        )

    # make_legend(250)
    return cat_cmap, make_legend


@app.cell
def _(cat_cmap, cat_result, make_legend):
    # Result preview. Just a subset if it's not small.
    _sx = cat_result.sizes["x"]
    _sy = cat_result.sizes["y"]
    _window_size = 500
    _offset_x = 0
    _offset_y = 0
    if _sx > 250 and _sy > _window_size:
        _offset_x = int(_sx / 2 - _window_size / 2)
        _offset_y = int(_sy / 2 - _window_size / 2)

    _windowed = cat_result[:, _offset_y:_offset_y + _window_size, _offset_x:_offset_x + _window_size]

    (
        _windowed.astype("str").hvplot(x="x", y="y", data_aspect=1.0, cmap=cat_cmap, frame_width=350)
        + make_legend(size=250)
    )
    return


@app.cell
def _(mo):
    ui_export = mo.ui.run_button(label="Export")
    ui_export
    return (ui_export,)


@app.cell
def _(cat_result, mo, np, ui_aoi, ui_export):
    mo.stop(not ui_export.value, "Click the button above to export.")

    for timeslice in cat_result:
        _date = str(timeslice["time"].values)[:10]
        _filename = f"data/dpolcat-{ui_aoi.value}-{_date}.tif"
        timeslice.astype(np.uint8).rename("dpolcat").rio.to_raster(_filename)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
