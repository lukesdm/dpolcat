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
    mo.md(r"""## Load Sentinel-1 RTC data""")
    return


@app.cell
def _(planetary_computer, pystac_client, stackstac):
    # Salzburg
    epsg_num = 32633 # UTM zone 33N
    bbox = [13.02, 47.76, 13.09, 47.83]

    date_range = "2022-06-01/2022-06-15"

    item_i = 1
    resolution = 10

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-1-rtc"], bbox=bbox, datetime=date_range
    )
    items = search.item_collection()
    item = items[item_i]
    print(f"Found {len(items)} items, selected {item.id}.")

    ds = stackstac.stack(item, bounds_latlon=bbox, epsg=epsg_num, resolution=resolution)

    vv_lin = ds.sel(band="vv")[0].compute()
    vh_lin = ds.sel(band="vh")[0].compute()
    return item, vh_lin, vv_lin


@app.cell
def _(mo):
    mo.md(r"""## Amplitude scaling""")
    return


@app.cell
def _(dp, vh_lin, vv_lin):
    vv_sn = dp.scale_nice(vv_lin)
    vh_sn = dp.scale_nice(vh_lin)
    return vh_sn, vv_sn


@app.cell
def _(item, vh_sn, vv_sn):
    _id = item.id

    (vv_sn.hvplot(cmap="gray", title=f"VV (scaled) [{_id}]") +
     vh_sn.hvplot(cmap="gray", title=f"VH (scaled) [{_id}]")
    ).cols(1)
    return


@app.cell
def _(mo):
    mo.md(r"""## Categorization""")
    return


@app.cell
def _(dp, vh_sn, vv_sn):
    cat_result = dp.xr_categorize(vv_sn, vh_sn)
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
    cat_result.astype("str").hvplot(cmap=cat_cmap) + make_legend(size=250)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
