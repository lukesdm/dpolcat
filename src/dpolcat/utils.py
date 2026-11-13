import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from collections import defaultdict
    from collections.abc import Iterable
    from functools import partial
    from pathlib import Path
    from typing import Literal
    import xml.etree.ElementTree as ET

    import bokeh.models
    import dask
    import geopandas as gpd
    import numpy as np
    import odc.stac
    import pandas as pd
    import pystac
    import pystac_client
    import shapely.geometry as sg
    import xarray
    import rioxarray
    import zarr

    import dpolcat.core as dpolcat
    import dpolcat.fast_frost as ff
    from dpolcat.pc_rio_driver import PCRioDriver

    # Want total coverage, with some wiggle room due to reprojection
    MIN_COVERAGE_FRACTION = 0.99


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Luke McQuade, October 2026

    # `dpolcat utils`

    Utility/helper functions etc. for working with *dpolcat* in various settings.

    /// admonition | Info

    This notebook is designed around the concept of [marimo reusable functions](https://docs.marimo.io/guides/reusing_functions/). Be careful changing things!

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Search

    Finding Sentinel-1 STAC items that work well together for a *dpolcat* time series is a little more involved than for optical imagery. For consistent comparisons over time, especially in hilly/mountainous areas of interest (AoIs), the sensing geometry should be as similar as possible. I.e., items should should have the same orbit direction and relative orbit number. However, items covering the AoI with the same `datatake_id` (continuous take over a day's orbit), can be safely combined (which are then referred to as 'scenes'). These are the rules governing the search functions here.
    """)
    return


@app.function
def find_stac_items(
    aoi: gpd.GeoDataFrame, date_range: str
) -> gpd.GeoDataFrame:
    """
    Find Sentinel-1 RTC scenes from the Microsoft Planetary Computer catalog intersecting the `aoi`.
    - `aoi` is a single-row GeoDataFrame.
    - `date_range` is a string with format `{from_date}/{to_date}`, e.g., "2020-01-01/2020-03-31".

    Returns a GeoDataFrame representing the STAC items, ordered by datetime. The original pystac.Item is stashed in the _item field.
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
        intersects=aoi.to_crs("EPSG:4326").geometry[0],
    )

    items = search.item_collection()

    items_gdf = gpd.GeoDataFrame.from_features(items, crs="EPSG:4326")

    # Add the item ID, which isn't included in the above.
    items_gdf["id"] = [item.id for item in items]

    # datetime is originally a string, so convert it.
    items_gdf["datetime"] = pd.to_datetime(items_gdf["datetime"])

    # Stash the original item (a shallow copy) for later.
    items_gdf["_item"] = items

    return items_gdf.sort_values("datetime")


@app.function
def calc_coverages(
    items_gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Calculate full vs partial scene coverages by orbit for the given AoI, grouped by `datatake_id`, i.e.,
    approx. same time.
    - `items_gdf` is a GeoDataFrame representing the STAC items.
    - `aoi` is a single-row GeoDataFrame.

    Returns a DataFrame containing the orbit, and number of `full_scenes` and `partial_scenes` covered,
    ordered by number of `full_scenes` (highest first).
    """

    # Reproject to UTM for area calculation, which is planar.
    utm_crs = items_gdf.estimate_utm_crs()
    items_gdf = items_gdf.to_crs(utm_crs)
    aoi = aoi.to_crs(utm_crs)

    intersection = items_gdf.dissolve(
        by=["sat:orbit_state", "sat:relative_orbit", "s1:datatake_id"]
    ).intersection(aoi.geometry[0])
    overlap_fraction = (intersection.area / aoi.area[0]).rename(
        "overlap_fraction"
    )

    full_scenes_count = (
        overlap_fraction[overlap_fraction >= MIN_COVERAGE_FRACTION]
        .groupby(["sat:orbit_state", "sat:relative_orbit"])
        .count()
    )
    partial_scenes_count = (
        overlap_fraction[overlap_fraction < MIN_COVERAGE_FRACTION]
        .groupby(["sat:orbit_state", "sat:relative_orbit"])
        .count()
    )

    df_coverages = pd.DataFrame(
        {
            "full_scenes": full_scenes_count,
            "partial_scenes": partial_scenes_count,
        }
    ).fillna(0)

    return df_coverages.sort_values("full_scenes", ascending=False)


@app.function
def select_by_orbit(
    items_gdf,
    orbit_direction: Literal["ascending", "descending"],
    relative_orbit: int,
) -> Iterable[pystac.Item]:
    """
    Selects the STAC items for the given orbit direction and relative orbit.
    - `items_gdf` is a GeoDataFrame representing the source STAC items.
    - `orbit_direction` is either "ascending" or "descending".
    - `relative_orbit` is the relative orbit number.
    Returns a filtered collection of `pystac.Item`s.
    """
    return items_gdf.loc[
        (items_gdf["sat:orbit_state"] == orbit_direction)
        & (items_gdf["sat:relative_orbit"] == relative_orbit)
    ]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test/demo
    """)
    return


@app.cell
def _():
    test_aoi = gpd.read_file("gis/aois/salzburg_south_4326.gpkg")
    test_date_range = "2025-03-01/2025-03-31"
    return test_aoi, test_date_range


@app.cell
def _(test_aoi, test_date_range):
    test_items = find_stac_items(aoi=test_aoi, date_range=test_date_range)
    return (test_items,)


@app.cell
def _(test_items):
    test_items
    return


@app.cell
def _(test_aoi, test_items):
    test_scene_coverages = calc_coverages(test_items, aoi=test_aoi)
    test_scene_coverages
    return (test_scene_coverages,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Find the orbit with the highest number of fully-covering scenes.
    """)
    return


@app.cell
def _(test_scene_coverages):
    _best_match = test_scene_coverages.reset_index().iloc[0]
    test_best_match = _best_match["sat:orbit_state"], _best_match["sat:relative_orbit"]
    test_best_match
    return (test_best_match,)


@app.cell
def _(test_best_match, test_items):
    test_sel_items = select_by_orbit(test_items,*test_best_match)
    test_sel_items["_item"].values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analysis
    """)
    return


@app.function
def frost_xr(
    da: xarray.DataArray, nanny=True, damping_factor=2.0, win_size=5
) -> xarray.DataArray:
    """Apply 2D (y, x) Frost filter to 3D (y, x, time) xarray DataArray; See underlying filter function for damping_factor and win_size."""
    # Mostly Claude (Opus 5.5)
    f = ff.frost_filter_nanny if nanny else ff.frost_filter_fast
    g = np.vectorize(
        partial(f, damping_factor=damping_factor, win_size=win_size),
        signature="(m,n)->(m,n)",
    )

    # In case da holds a materialized np array, operate directly...
    if da.chunks is None:
        return da.copy(data=g(da.data))


    # ...otherwise, set up parallel calc with overlapping windows
    # (because it's a neighbourhood operation).
    h = win_size // 2
    data = da.data.map_overlap(
        g,
        depth=(0, h, h),
        boundary="none",
        meta=np.array((), dtype=da.dtype),
    )
    return da.copy(data=data)


@app.function
def normalize_db(arr, min=-30, max=10):
    """Shift, scale and clip the input array in dB to the range 0 to 1"""
    # E.g. range -30 to +10 -> 0.0 to 1.0

    return ((arr - min) / (max - min)).clip(0.0, 1.0)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### QGIS Colour Map

    Create a QGIS Color Map file that can be used for dpolcat layer symbology.
    """)
    return


@app.function
def write_qgis_colormap():
    lines = []
    
    for i, c in enumerate(dpolcat.color_list):
        label = i
        a = 255
        r, g, b = c
        lines.append(f"{i} {r} {g} {b} {a} {label}\n")
    
    f = open("dpolcat-qgis-cmap.clr", "w")
    f.writelines(lines)
    f.close()


if __name__ == "__main__":
    app.run()
