import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import geopandas as gpd
    import geoviews as gv
    import holoviews as hv
    import hvplot.xarray
    import hvplot.pandas
    import numpy as np
    import odc.stac
    import pandas as pd
    import planetary_computer
    import pystac_client
    import rasterio.features
    import rioxarray
    import shapely.geometry as sg
    import shapely.ops as so
    import xarray

    import dpolcat
    import dpolcat.utils
    import dpolcat.hv_utils

    # Some extra config for visualization
    hv.extension("bokeh")
    _hatch = ["hatch_pattern", "hatch_color", "hatch_alpha", "hatch_scale", "hatch_weight"]
    hv.Store.add_style_opts(gv.Polygons, _hatch, "bokeh")
    hv.Store.add_style_opts(hv.Polygons, _hatch, "bokeh")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Luke McQuade, orig. June 2023, updated October 2026.

    # *dpolcat* Demo 02: Flood Mapping

    The previous notebook in the series (`demo_01_intro.py`) introduced the core concepts of the *dpolcat* module. Here we demonstrate a use case - flood mapping.

    This notebook uses the [Sentinel-1 radiometrically-terrain-corrected (RTC) dataset](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc) from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) via STAC.
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
def get_item_by_id(item_collection, id: str):
    """Look up STAC item in collection by its ID."""
    return [item for item in item_collection if item.id == id][0]


@app.function
def plot_image(da):
    """Create a single-band grayscale plot"""
    return da.hvplot.image(
        x="x",
        y="y",
        data_aspect=1.0,
        cmap="gray",
        xlabel="",
        xformatter="%d",
        ylabel="",
        yformatter="%d",
    ).opts(xrotation=30, frame_width=300)


@app.function
def create_fcc(vv_db, vh_db):
    """Prepare a VV-VH false-colour composite DataArray ready for visualization."""
    _vv_db = dpolcat.utils.normalize_db(vv_db)

    # The VH band tends to be lower on average than VV, so compensate by multiplying by 2.
    _vh_db = (dpolcat.utils.normalize_db(vh_db) * 2).clip(0.0, 1.0)

    _red = (_vv_db + _vh_db) / 2
    _green = _vv_db
    _blue = _vh_db

    _ds = xarray.Dataset({"red": _red, "green": _green, "blue": _blue})
    _da = _ds.to_array("band")
    return _da


@app.function
def plot_composite(da):
    """Create a 3-band RGB composite plot"""
    return da.hvplot.rgb(
        x="x",
        y="y",
        bands="band",
        data_aspect=1.0,
        xlabel="",
        ylabel="",
        xformatter="%d",
        yformatter="%d",
    ).opts(xrotation=30)


@app.function
def plot_binary(da):
    return plot_image(da).opts(
        cmap="viridis",
        color_levels=2,
        clim=(-0.5, 1.5),
        colorbar=True,
        cticks=[0, 1],
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analysis & Modelling

    There was widespread flooding in Western Europe in the Summer of 2021. One of the affected areas was around the city of Duisburg in western Germany.

    Let's develop a simple model to try and delineate the flooded area using polarimetric categories.

    From inspection of images captured between the dates around the time of the flooding (2021-07-08 to 2021-07-21), the following images were selected for comparison:
    * Pre-flood: S1A_IW_GRDH_1SDV_20210713T171647_20210713T171712_038762_0492ED_rtc
    * Mid-flood: S1B_IW_GRDH_1SDV_20210716T054127_20210716T054152_027815_0351B0_rtc
    """)
    return


@app.cell
def _():
    bbox = [6.695, 51.407, 6.847, 51.462]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )

    _search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime="2021-07-08/2021-07-21",
    )
    _items = _search.item_collection()

    epsg_num = 32632
    resolution = 10
    pre = get_item_by_id(
        _items,
        "S1A_IW_GRDH_1SDV_20210713T171647_20210713T171712_038762_0492ED_rtc",
    )
    post = get_item_by_id(
        _items,
        "S1B_IW_GRDH_1SDV_20210716T054127_20210716T054152_027815_0351B0_rtc",
    )

    print(f"Found {len(_items)} items, using {pre.id=} and {post.id=}")
    return bbox, epsg_num, post, pre, resolution


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Load the data, apply Frost speckle filtering, and perform the categorization.
    """)
    return


@app.cell
def _(bbox, epsg_num, post, pre, resolution):
    _ds = odc.stac.load(
        [pre, post],
        bbox=bbox,
        epsg=epsg_num,
        resolution=resolution,
        patch_url=planetary_computer.sign,
    )
    vv_lin = _ds.vv.compute()
    vh_lin = _ds.vh.compute()
    vv_db = dpolcat.linear_to_db(vv_lin)
    vv_db = dpolcat.utils.frost_xr(vv_db)
    vh_db = dpolcat.linear_to_db(vh_lin)
    vh_db = dpolcat.utils.frost_xr(vh_db)
    cat_result = dpolcat.xr_categorize(vv_db, vh_db)
    return cat_result, vh_db, vv_db


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Visualize the source and categorized images. The source images are visualized with a false-colour composite, in the manner:

    * Red channel: average VV and VH
    * Green channel: VV
    * Blue channel: VH

    Note: the VH band is typically lower than VV, so is amplified to compensate for this.
    """)
    return


@app.cell
def _(cat_result, post, pre, vh_db, vv_db):
    _pre_vv, _pre_vh = vv_db[0], vh_db[0]
    _post_vv, _post_vh = vv_db[1], vh_db[1]
    _pre_cat, _post_cat = (
        cat_result[0].astype(str),
        cat_result[1].astype(str),
    )

    _pre_comp = create_fcc(_pre_vv, _pre_vh)
    _post_comp = create_fcc(_post_vv, _post_vh)

    _pre_date = pre.datetime.strftime("%d %b %Y")
    _post_date = post.datetime.strftime("%d %b %Y")

    _width = 450

    cat_cmap = dpolcat.hv_utils.make_cmap()

    (
        plot_composite(_pre_comp).opts(frame_width=_width, title=f"VV-VH false-colour composite, {_pre_date}")
        + plot_composite(_post_comp).opts(frame_width=_width, title=f"VV-VH false-colour composite, {_post_date}")
        + plot_image(_pre_cat).opts(
            cmap=cat_cmap, frame_width=_width, title=f"dpolcat categories, {_pre_date}"
        )
        + plot_image(_post_cat).opts(
            cmap=cat_cmap, frame_width=_width, title=f"dpolcat categories, {_post_date}"
        ) + dpolcat.hv_utils.make_legend(size=300)
    ).cols(2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Hopefully, you can make out the structures that make up the urban area of Duisburg, as well as the confluence of two rivers — the Rhine and the Ruhr.
    Comparing the earlier and later categorized images, notice the widening of the lilac/cyan regions. These are areas of category 1-3, which correspond to smooth to moderately smooth surface scattering, typical of most water. For our model, let's assume that pixels that have newly changed to category 1-3 are flooded areas. `dpolcat_modelling` provides some convenient classes for working with these categories in a more semantic way.
    """)
    return


@app.cell
def _():
    from dpolcat.modelling import Surface

    return (Surface,)


@app.cell
def _(Surface, cat_result):
    # Calculate new likely water pixels.
    _pre_cat = cat_result[0]
    _post_cat = cat_result[1]
    _water_cats = [Surface.Smooth, Surface.SlightlyRough, Surface.MediumRough]
    flooded = (
        _post_cat.isin(_water_cats).astype(np.int8)
        - _pre_cat.isin(_water_cats)
    ).astype(np.int8) == 1
    return (flooded,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Visualize the result.
    """)
    return


@app.cell
def _(flooded):
    plot_binary(flooded).opts(
        clabel="Flooded",
        title="Duisburg: Flood estimate",
        frame_width=500,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can see some solid regions of new cat. 1-3 cells near the rivers, which are likely flooded areas, but there are also many small, disparate areas. Let's apply some spatial filtering to reduce these effects.
    """)
    return


@app.cell
def _(flooded):
    from scipy.ndimage import gaussian_filter

    sigma = 1.5
    flooded_gaussian = xarray.apply_ufunc(
        gaussian_filter, flooded.astype(np.float32), sigma
    )
    return (flooded_gaussian,)


@app.cell
def _(flooded_gaussian):
    plot_image(flooded_gaussian).opts(
        clabel="Flooded",
        cmap="viridis",
        title="Duisburg: Flood area estimate with Gaussian filtering",
        frame_width=500,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The filtering shows areas of high concentrations of estimated flood cells, versus low concentrations. Let's apply a threshold, keeping only the highly-concentrated areas.
    """)
    return


@app.cell
def _():
    ui_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.6,
        step=0.05,
        label="Threshold",
        show_value=True,
        debounce=True,
    )
    ui_threshold
    return (ui_threshold,)


@app.cell
def _(flooded_gaussian, ui_threshold):
    flooded_thres = flooded_gaussian > ui_threshold.value
    return (flooded_thres,)


@app.cell
def _(flooded_thres):
    plot_binary(flooded_thres).opts(
        clabel="Flooded",
        title="Duisburg: Filtered and thresholded flood estimate",
        frame_width=500,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This seems to have eliminated many of the disparate regions away from the rivers (likely false positives).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation

    The Copernicus Emergency Mapping Service was activated for this, and we can use these maps as a reference [1]. Read through the steps in this section, then press the button below to perform the evaluation.
    """)
    return


@app.cell
def _():
    btn_evaluate = mo.ui.run_button(label="Perform evaluation")
    btn_evaluate
    return (btn_evaluate,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Polygonize our estimated flooded area. We end up with a MultiPolygon-containing GeoDataFrame.
    """)
    return


@app.cell
def _(btn_evaluate, epsg_num, flooded_thres):
    mo.stop(not btn_evaluate.value, "Click 👆 to start.")

    _da = flooded_thres.astype(np.uint8)
    _polys = rasterio.features.shapes(_da, transform=_da.rio.transform())

    _polys = so.unary_union(
        [sg.shape(poly) for poly, value in _polys if value == 1]
    )

    flooded_estimated = gpd.GeoDataFrame(geometry=[_polys], crs=epsg_num)
    return (flooded_estimated,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Load the reference map's flooded area polygons. Again, we end up with a MultiPolygon-containing GeoDataFrame.
    """)
    return


@app.cell
def _(epsg_num):
    _flooded_reference = gpd.read_file(
        "gis/duisburg/EMSR517_AOI06_DEL_MONIT01_observedEventA_r1_v1_extract2.gpkg"
    ).to_crs(epsg_num)
    flooded_reference = gpd.GeoDataFrame(
        geometry=[_flooded_reference.union_all()], crs=epsg_num
    )
    return (flooded_reference,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare the results on a map.
    """)
    return


@app.cell
def _(epsg_num, flooded_estimated, flooded_reference, flooded_thres):
    # Calculate extent from our DataArray, rather than bbox, as there's a discrepancy.
    extent = gpd.GeoDataFrame(
        geometry=[sg.box(*flooded_thres.rio.bounds())], crs=epsg_num
    )

    (
        gv.tile_sources.StamenTerrain
        * flooded_estimated.to_crs(3857)
        .hvplot(
            fill_alpha=0.5,
            label="Estimated",
        )
        .opts(hatch_pattern="/")
        * flooded_reference.to_crs(3857).hvplot(
            fill_alpha=0.5, label="Reference", hatch_pattern="\\"
        )
        * extent.to_crs(3857).hvplot(fill_color=0, label="Study area")
    ).opts(
        frame_width=600,
        frame_height=450,
        title="Duisburg: Flood area",
        tools=["wheel_zoom"],
    )
    return (extent,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Perform spatial vector arithmetic to determine the regions of:
    - True positives (TP): `intersect(estimated, reference)`
    - False positives (FP): `estimated - reference`
    - False negatives (FN): `reference - estimated`
    """)
    return


@app.cell
def _(flooded_estimated, flooded_reference):
    tp = flooded_estimated.intersection(flooded_reference)
    fp = flooded_estimated.difference(flooded_reference)
    fn = flooded_reference.difference(flooded_estimated)
    return fn, fp, tp


@app.cell
def _(extent, fn, fp, tp):
    (
        gv.tile_sources.StamenTerrain
        * gpd.GeoDataFrame({"TP": tp, "FP": fp, "FN": fn})
        .melt()
        .set_geometry("value")
        .to_crs(3857)
        .hvplot(color="variable")
        .opts()
        * extent.to_crs(3857).hvplot(fill_color=0, label="Study area")
    ).opts(
        frame_width=600,
        frame_height=450,
        title="Duisburg: Flood area",
        tools=["wheel_zoom"],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using these, an accuracy calculation can be made. Due to the sparse occurence of flooded areas in the study area, the [F1 score](https://www.v7labs.com/blog/f1-score-guide) is a more appropriate measure than overall accuracy in this case.
    """)
    return


@app.cell
def _(fn, fp, tp):
    f1 = tp.area / (tp.area + 0.5 * (fp.area + fn.area))

    mo.md(f"**Final F1 score: {f1.values[0]:.3f}**")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A QGIS project has been created for further inspection of the outputs - see `📂 gis/duisburg`. Click the button below to export the outputs to this folder.
    """)
    return


@app.cell
def _():
    btn_save = mo.ui.run_button(label="Save results")
    btn_save
    return (btn_save,)


@app.cell
def _(
    btn_save,
    cat_result,
    epsg_num,
    flooded_estimated,
    flooded_thres,
    fn,
    fp,
    tp,
):
    mo.stop(not btn_save.value, "Click 👆 to save the results.")

    # Raster layers
    _cat_result = cat_result.rio.write_crs(epsg_num)
    _cat_result = _cat_result.rio.write_nodata(0)
    _cat_result[0].astype(np.uint8).rio.to_raster("gis/duisburg/duisburg_pre_dpolcat.tif")
    _cat_result[1].astype(np.uint8).rio.to_raster("gis/duisburg/duisburg_post_dpolcat.tif")
    _flooded_thres = flooded_thres.rio.write_crs(epsg_num)
    _flooded_thres = _flooded_thres.astype(np.uint8)
    _flooded_thres = _flooded_thres.rio.write_nodata(255)
    _flooded_thres.rio.to_raster("gis/duisburg/duisburg_flood_estimate.tif")

    # Vector layers
    flooded_estimated.to_file("gis/duisburg/flood_estimate_vec.gpkg")
    tp.to_file("gis/duisburg/true_positive.gpkg")
    fp.to_file("gis/duisburg/false_positive.gpkg")
    fn.to_file("gis/duisburg/false_negative.gpkg")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion & Conclusion

    You should have seen that our model achieved an F1 score of approx. 0.6-0.7, depending on the post-filter threshold. It was somewhat effective, but far from state of the art. However, it was a very simple model, and was hopefully easily understood.

    In this urban study area, the land cover differences mapped well to differences in the polarimetric scattering types. This is often not the case in other environments. For example, large areas of tarmac and fallow agricultural fields in winter will also exhibit surface scattering behaviour that is difficult to distinguish from water. So, the model will likely not generalize well in its current form.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    [1]: Activation EMSR517: "Flood in Western Germany", Copernicus Emergency Management, 13 July 2021.  https://mapping.emergency.copernicus.eu/activations/EMSR517/
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
