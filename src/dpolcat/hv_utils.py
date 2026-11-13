import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import geoviews as gv
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray
    import numpy as np
    import pandas as pd
    import xarray

    import dpolcat.core as dpolcat


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Luke McQuade, November 2026

    # `dpolcat hv_utils`

    Utility/helper functions etc. for *dpolcat*-related visualization with [HoloViz](https://holoviz.org/) tools.

    /// admonition | Info

    This notebook is designed around the concept of [marimo reusable functions](https://docs.marimo.io/guides/reusing_functions/). Be careful changing things!

    ///
    """)
    return


@app.function
def make_cmap():
    """Make a HoloViews-compatible categorical colormap (where a string key is needed)."""
    return {
        str(i): list(dpolcat.color_list[i])
        for i in range(len(dpolcat.color_list))
    }


@app.function
def get_hex_colors(cat_indexes = range(dpolcat.NUM_CATEGORIES)):
    """Make a hex-string-format colour list for the given categories."""
    return [
        f"#{r:02x}{g:02x}{b:02x}"
        for r, g, b in dpolcat.color_list[cat_indexes]
    ]


@app.function
def make_legend(size=350):
    """Create a legend (in the form of a Holoviews plot)."""
    n_steps = 100
    steps = np.linspace(-30.5, 10.5, n_steps)
    vv_ramp = np.vstack([steps] * n_steps)
    vh_ramp = vv_ramp.T

    cat_values = xarray.DataArray(
        dpolcat.categorize_np(vv_ramp, vh_ramp),
        coords=[steps, steps],
        dims=["VH", "VV"],
    )

    label_data = []
    for i in range(1, dpolcat.NUM_CATEGORIES):
        box = cat_values.where(cat_values == i, drop=True)
        center_x = float(box["VV"].mean())
        center_y = float(box["VH"].mean())
        label_data.append((center_x, center_y, str(i)))

    labels = hv.Labels(
        label_data, kdims=["VV", "VH"], vdims=["text"]
    ).opts(text_color="black")

    return (
        cat_values.astype("str")
        .hvplot(cmap=make_cmap())
        .opts(
            width=size,
            height=size,
            title="NoData (0) is black",
            xlabel="VV (dB)",
            ylabel="VH (dB)",
        )
        * labels
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Test/demo
    """)
    return


@app.cell
def _():
    make_legend()
    return


@app.cell
def _():
    _n = dpolcat.NUM_CATEGORIES
    _color = get_hex_colors(range(_n))
    pd.DataFrame(
        {
            "category": range(_n),
            "count": [10] * _n,
            "color": _color,
        }
    ).hvplot.bar(x="category", y="count", color="color")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is an example map plot, with simple legend. Note: GeoViews can be slow to reproject rasters, so it's often best to get them into EPSG:3857 first with, e.g., `_da = _da.rio.to_crs(3857)`.
    """)
    return


@app.cell
def _():
    # Dummy data extent, EPSG:3857
    _x1 = 1710000
    _x2 = 1728000
    _y1 = 5944000
    _y2 = 5962000

    _s = 20

    _coords = np.linspace(start=[_y1, _x1], stop=[_y2, _x2], num=_s)
    _data = np.random.randint(
        low=0, high=dpolcat.NUM_CATEGORIES, size=(_s, _s)
    )
    _cats = xarray.DataArray(_data, dims=["y", "x"], coords=_coords.T, name="category")

    _cats = _cats.rio.write_crs(3857)

    _cat_cmap = get_hex_colors()

    cat_plot = gv.tile_sources.OSM * _cats.hvplot(x="x", y="y").opts(
        cmap=_cat_cmap,
        clim=(-0.5, dpolcat.NUM_CATEGORIES - 0.5),
        cticks=list(range(0, dpolcat.NUM_CATEGORIES)),
        title="dpoclat categories",
        active_tools=["wheel_zoom"],
        xlim=(_x1-10000, _x2+10000),
        ylim=(_y1-10000, _y2+10000),
        frame_width=400,
        frame_height=400,
    )

    cat_plot
    return


if __name__ == "__main__":
    app.run()
