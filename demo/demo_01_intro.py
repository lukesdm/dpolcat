import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import bokeh.models
    import numpy as np
    import odc.stac
    import pandas as pd
    from pyproj import Transformer
    import pystac
    import pystac_client
    import planetary_computer
    import requests
    import rioxarray
    import holoviews as hv
    import hvplot.xarray
    import hvplot.pandas
    import xarray

    import dpolcat
    import dpolcat.utils


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Luke McQuade, orig. June 2023, updated October 2026.

    # *dpolcat* Demo 01: Introduction

    This notebook introduces the *dpolcat* module - a tool for the automated categorization of dual-polarimetric Sentinel-1 radar imagery. It is inspired by Satellite Image Automatic Mapper (SIAM) and its concept of spectral categories [1].

    This notebook uses the [Sentinel-1 radiometrically-terrain-corrected (RTC) dataset](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc) from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) via STAC, as in [2].
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Helpers

    A few functions used later.
    """)
    return


@app.function
def plot_hist(da, pol, unit):
    """Create a backscatter histogram plot"""
    n_hist_bins = 40

    return da.rename(unit).hvplot.hist(
        bins=n_hist_bins,
        bin_range=(0, 250) if unit == "linear" else (-30, 20),
        title=f"{pol} ({unit})",
        xlabel="Value",
        ylabel="Pixel count",
        width=300,
        height=250,
    )


@app.function
def plot_image(da, pol, unit):
    """Create a single-band grayscale plot"""
    return (
        da.rename(unit)
        .hvplot.image(
            x="x",
            y="y",
            data_aspect=1.0,
            cmap="gray",
            clim=(0, 100) if unit == "linear" else (-30, 20),
            title=f"{pol} ({unit})",
            xlabel="",
            xformatter="%d",
            ylabel="",
            yformatter="%d",
        )
        .opts(xrotation=30, frame_width=300)
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Scaling Function

    Raw (linear) backscatter data is typically very heavily positively skewed, which makes hierarchical thresholding (as in a decision tree) very sensitive. The power ratio (dB) is more suitable.

    Let's load an image and compare the distributions of the different scaling approaches.
    """)
    return


@app.cell
def _():
    # Load image
    aoi = "salzburg"
    # aoi = "tenerife"
    if aoi == "salzburg":
        epsg_num = 32633  # UTM zone 33N
        bbox = [13.02, 47.76, 13.09, 47.83]
        item_i = 1
        resolution = 10
    elif aoi == "tenerife":
        epsg_num = 32628  # UTM zone 28N
        bbox = [-16.96, 27.92, -16.05, 28.61]
        item_i = 2
        resolution = 10
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    _search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime="2022-06-01/2022-06-15",
    )
    _items = _search.item_collection()
    item = _items[item_i]
    print(f"Found {len(_items)} items, selected {item.id}.")
    _ds = odc.stac.load(
        [item], bbox=bbox, epsg=epsg_num, resolution=resolution
    )
    vv_lin = _ds["vv"].compute().isel(time=0)
    vh_lin = _ds["vh"].compute().isel(time=0)
    return aoi, epsg_num, item, vh_lin, vv_lin


@app.cell
def _(vh_lin, vv_lin):
    vv_db = dpolcat.linear_to_db(vv_lin)
    vh_db = dpolcat.linear_to_db(vh_lin)
    return vh_db, vv_db


@app.cell
def _(aoi, epsg_num, item, vh_db, vh_lin, vv_db, vv_lin):
    (
        plot_image(vv_lin, "VV", "linear")
        + plot_image(vh_lin, "VH", "linear")
        + plot_image(vv_db, "VV", "dB")
        + plot_image(vh_db, "VH", "dB")
    ).cols(2).opts(
        title=f"Visualization of {aoi}, {str(item.datetime)[:19]}, EPSG:{epsg_num}",
        fontsize={"title": 12},
    )
    return


@app.cell
def _(aoi, item, vh_db, vh_lin, vv_db, vv_lin):
    (
        plot_hist(vv_lin, "VV", "linear")
        + plot_hist(vh_lin, "VH", "linear")
        + plot_hist(vv_db, "VV", "dB")
        + plot_hist(vh_db, "VH", "dB")
    ).cols(2).opts(
        title=f"Backscatter distributions for '{aoi}', {str(item.datetime)[:19]}",
        fontsize={"title": 12},
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Polarimetric Categorization

    The main feature of dpolcat is to assign categories to combinations of VV and VH values. These categories are based on the physical scattering properties of surfaces and objects on the ground: specular, volume, double/triple bounce, and combinations of these. This is achieved with a decision tree algorithm; see `dpolcat.categorize` for details of this.

    The categories are visualized below for the typical range of VV and VH values (in dB) expected in an image.

    A rough guide to the category numbers:
    - Mostly surface scattering: 1–5
    - Mostly volume scattering: 6–12
    - Double/triple-bounce: 13–18
    - Ill-defined (physically): 19
    - NoData/invalid: 0
    """)
    return


@app.cell
def _():
    dpolcat.utils.make_legend()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Demo

    Below, we perform polarimetric categorization of the sample image we selected earlier.
    """)
    return


@app.cell
def _(vh_db, vv_db):
    cat_result = dpolcat.xr_categorize(vv_db, vh_db)
    return (cat_result,)


@app.cell
def _(cat_result, vh_db, vv_db):
    # Visualize the result

    # First, we need to convert to str for categorical colourmap to work.
    _da = cat_result.astype("str")

    cat_cmap = dpolcat.utils.make_cmap()

    # Setup linked crosshairs,
    # via https://discourse.holoviz.org/t/linked-crosshair/4326/2
    linked_crosshair = bokeh.models.CrosshairTool(
        overlay=[
            bokeh.models.Span(dimension="width"),
            bokeh.models.Span(dimension="height"),
        ]
    )

    (
        plot_image(_da, "dpolcat", "category").opts(
            cmap=cat_cmap, frame_width=350, tools=[linked_crosshair, "hover"]
        )
        + dpolcat.utils.make_legend(size=250).opts(width=300)
        + plot_image(vv_db, "VV", "dB").opts(
            frame_width=350, tools=[linked_crosshair, "hover"]
        )
        + plot_image(vh_db, "VH", "dB").opts(
            frame_width=350, tools=[linked_crosshair, "hover"]
        )
    ).cols(2)
    return (cat_cmap,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Speckle filtering
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Doesn't the image seem rather busy and chaotic? Radar speckle plays a big part here. So, let's apply a speckle filter (a classic Frost filter [3]) and try again.
    """)
    return


@app.cell
def _(vh_lin, vv_lin):
    vv_filt = dpolcat.utils.frost_xr(vv_lin)
    vh_filt = dpolcat.utils.frost_xr(vh_lin)
    vv_db_filt = dpolcat.linear_to_db(vv_filt)
    vh_db_filt = dpolcat.linear_to_db(vh_filt)
    return vh_db_filt, vv_db_filt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below you can see the effect on the backscatter.
    """)
    return


@app.cell
def _(vv_db, vv_db_filt):
    plot_image(vv_db, "VV", "dB") + plot_image(
        vv_db_filt, "VV", "dB"
    ).opts(title="Frost-filtered VV (dB)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And here the effect on the final categorization.
    """)
    return


@app.cell
def _(vh_db_filt, vv_db_filt):
    cat_result_filt = dpolcat.xr_categorize(vv_db_filt, vh_db_filt)
    return (cat_result_filt,)


@app.cell
def _(cat_cmap, cat_result, cat_result_filt):
    _cat_orig = cat_result.astype(str)
    _cat_filt = cat_result_filt.astype(str)

    (
        plot_image(_cat_orig, "dpolcat", "category").opts(
            cmap=cat_cmap, frame_width=350
        )
        + plot_image(_cat_filt, "dpolcat", "category").opts(
            cmap=cat_cmap,
            frame_width=350,
            title="Frost-filtered: dpolcat (category)",
        )
        + dpolcat.utils.make_legend(size=250)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The distribution of categories are as follows:
    """)
    return


@app.cell
def _(cat_result_filt):
    cat_counts = (
        cat_result_filt.to_dataframe().groupby("dpolcat").size().to_frame("count")
    )
    cat_counts["%"] = 100 * cat_counts["count"] / cat_result_filt.size
    return (cat_counts,)


@app.cell
def _(cat_counts):
    def plot_cat_counts(cat_counts):
        _df = cat_counts.copy()
        _df["color"] = _color = dpolcat.utils.get_hex_colors(cat_counts.index)
        _df
        return (
            _df.reset_index()
            .astype({"dpolcat": str})
            .hvplot.bar(
                x="dpolcat",
                y="count",
                color="color",
                yformatter="%d",
                xlabel="Category",
                ylabel="Count",
                width=400,
                height=550,
            )
        )

    _cat_count_table = mo.ui.table(
        cat_counts,
        page_size=20,
        show_column_summaries=False,
        show_search=False,
        format_mapping={"%": "{:.1f}"},
    )

    _cat_count_plot = plot_cat_counts(cat_counts)

    mo.hstack(
        [
            _cat_count_table,
            _cat_count_plot,
        ],
        justify="start",
        align="end",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusion

    The core concepts of *dpolcat* have now been introduced. If you have in mind the question, "OK, but what can I do with it?", check out the next demo notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    [1]: *GEO-CEOS stage 4 validation of the Satellite Image Automatic Mapper lightweight computer program for ESA Earth observation level 2 product generation - Part 1: Theory* (Andrea Baraldi, Michael Laurence Humber, Dirk Tiede & Stefan Lang) | Louis-Noel Moresi (Reviewing Editor) (2018), Cogent Geoscience, 4:1, DOI: [10.1080/23312041.2018.1467357](https://doi.org/10.1080/23312041.2018.1467357)

    [2]: *Accessing Sentinel-1 RTC data with the Planetary Computer STAC API* (Jupyter Notebook, Microsoft Planetary Computer, accessed June 2023) ([link](https://pccompute.westeurope.cloudapp.azure.com/compute/hub/user-redirect/lab/tree/PlanetaryComputerExamples/datasets/sentinel-1-rtc/sentinel-1-rtc-example.ipynb))

    [3]: *A model for radar images and its application to adaptive digital filtering of multiplicative noise* (V. S. Frost, J. A. Stiles, K. S. Shanmugan, and J. C. Holtzman, IEEE Trans. Pattern Anal. Mach. Intell., vol. PAMI-4, no. 2, pp. 157–165, Mar. 1982, https://doi.org/10.1109/TPAMI.1982.4767223)
    """)
    return


if __name__ == "__main__":
    app.run()
