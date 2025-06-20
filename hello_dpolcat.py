import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import pystac_client
    import planetary_computer
    import stackstac
    import xarray
    import holoviews as hv
    import hvplot.xarray

    import dpolcat as dp
    return dp, hv, mo, np, planetary_computer, pystac_client, stackstac, xarray


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
    # epsg_num = 32633 # UTM zone 33N
    bbox = aois[ui_aoi.value]

    _d1, _d2 = ui_date_range.value
    # _d1.year
    # date_range =  "2022-06-01/2022-06-15"
    date_range = f"{_d1.year}-{_d1.month:02}-{_d1.day:02}/{_d2.year}-{_d2.month:02}-{_d2.day:02}"
    return bbox, date_range


@app.cell
def _(bbox, date_range, mo, planetary_computer, pystac_client):
    # item_i = 1
    resolution = 10

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-1-rtc"], bbox=bbox, datetime=date_range
    )
    items = search.item_collection()

    print(f"Found {len(items)} items")

    _options = {items[i].id: i for i in range(len(items))}
    ui_items = mo.ui.multiselect(_options)
    ui_items
    return items, resolution, ui_items


@app.cell
def _(items, mo, ui_items):
    sel_items = [items[i] for i in ui_items.value]
    mo.stop(len(sel_items) == 0, "No items selected")
 
    _epsgs = set([int(item.properties["proj:code"][5:]) for item in sel_items])
    assert len(_epsgs) == 1, "items are of different CRSs, this is not supported."
    epsg_num = int(_epsgs.pop())
    return epsg_num, sel_items


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
    return (ui_run,)


@app.cell
def _(bbox, epsg_num, mo, resolution, sel_items, stackstac, ui_run):
    # Load data

    # Wait for button click
    mo.stop(not ui_run.value, "Click the button to start processing.")

    ds = stackstac.stack(sel_items, bounds_latlon=bbox, epsg=epsg_num, resolution=resolution)
    vv_lin = ds.sel(band="vv").compute()
    vh_lin = ds.sel(band="vh").compute()
    return vh_lin, vv_lin


@app.cell
def _(dp, vh_lin, vv_lin):
    vv_sn = dp.scale_nice(vv_lin)
    vh_sn = dp.scale_nice(vh_lin)
    return vh_sn, vv_sn


@app.cell
def _(item, vh_sn, vv_sn):
    _id = item.id

    (vv_sn.hvplot(x="x", y="y", data_aspect=1.0, cmap="gray", title=f"VV (scaled) [{_id}]") +
     vh_sn.hvplot(x="x", y="y", data_aspect=1.0, cmap="gray", title=f"VH (scaled) [{_id}]")
    ).cols(2)
    return


@app.cell
def _(mo):
    mo.md(r"""## Categorization""")
    return


@app.cell
def _(dp, mo):
    @mo.persistent_cache()
    def categorize(vv_sn, vh_sn):
        return dp.xr_categorize(vv_sn, vh_sn)
    return (categorize,)


@app.cell
def _(categorize, vh_sn, vv_sn):
    cat_result = categorize(vv_sn, vh_sn)
    return (cat_result,)


@app.cell
def _(dp, hv, np, xarray):
    cat_cmap = {str(i): list(dp.color_list[i]) for i in range(len(dp.color_list))}

    def make_legend(size=350):
        n_steps = 100
        steps = np.linspace(-0.05, 1, n_steps)
        vv_ramp = np.vstack([steps] * n_steps)
        vh_ramp = vv_ramp.transpose()
        cat_values = xarray.DataArray(dp.categorize_np(vv_ramp, vh_ramp), coords=[steps, steps], dims=["VH", "VV"])

        labels = {}
        for i in range(1, dp.NUM_CATEGORIES):
            box = cat_values.where(cat_values == i,drop=True)
            min_vv = float(box["VV"].min())
            max_vv = float(box["VV"].max())
            min_vh = float(box["VH"].min())
            max_vh = float(box["VH"].max())
            center_x = (min_vv + max_vv) / 2
            center_y = (min_vh + max_vh) / 2
            labels[i] = hv.Text(center_x, center_y, str(i))


        return (
            cat_values.astype("str").hvplot(cmap=cat_cmap).opts(height=size, width=size)
            * hv.NdOverlay(labels).opts(show_legend=False)
            * hv.Text(-0.02,-0.02, "0").opts(text_color="white")
        )

    # make_legend(300)
    return cat_cmap, make_legend


@app.cell
def _(cat_cmap, cat_result, make_legend):
    cat_result.astype("str").hvplot(x="x", y="y", data_aspect=1.0, cmap=cat_cmap, frame_width=350) + make_legend(size=250)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
