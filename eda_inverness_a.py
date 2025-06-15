import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# dpolcat EDA - Inverness - Part A""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prepare the environment""")
    return


@app.cell
def _():
    import geopandas as gpd
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray
    import numpy as np
    import pandas as pd
    import pystac_client
    import planetary_computer
    import rioxarray
    import stackstac
    import xarray

    hv.extension("bokeh")
    return (
        gpd,
        hv,
        np,
        pd,
        planetary_computer,
        pystac_client,
        stackstac,
        xarray,
    )


@app.cell
def _():
    import dpolcat
    return (dpolcat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Find Sentinel-1 RTC scenes""")
    return


@app.cell
def _():
    # Interior rectangle of desired Sentinel 1 slice
    # - An around Inverness, Scotland.
    search_poly_coords = [
        [-4.115193, 58.318314],
        [-0.335202, 58.350820],
        [-0.344490, 57.310626],
        [-4.079339, 57.308856],
        [-4.115193, 58.318314]
    ]

    # Summer
    search_start = "2020-06-01"
    search_end = "2020-08-31"
    return search_end, search_poly_coords, search_start


@app.cell
def _(
    planetary_computer,
    pystac_client,
    search_end,
    search_poly_coords,
    search_start,
):
    STAC_COLLECTION = "sentinel-1-rtc"

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    cql2_contains = {
        "op": "s_contains",
        "args": [
            {"property": "geometry"},
            {
                "type": "Polygon",
                "coordinates": [search_poly_coords]
            }
        ]
    }

    search = catalog.search(
        collections=[STAC_COLLECTION], datetime=f"{search_start}/{search_end}", filter=cql2_contains
    )

    items = search.item_collection()
    print(f"Found {len(items)} items")
    return STAC_COLLECTION, catalog, items


@app.cell
def _(items, pd):
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

    summarize(items)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Select S1 scene""")
    return


@app.cell
def _(items, mo):
    DEFAULT_ITEM_ID = "S1B_IW_GRDH_1SDV_20200825T175046_20200825T175111_023083_02BD3A_rtc"
    item_dd = mo.ui.dropdown([item.id for item in items], value=DEFAULT_ITEM_ID)
    item_dd
    return (item_dd,)


@app.cell
def _(STAC_COLLECTION, catalog, item_dd):
    s1_item_id = item_dd.value
    item = catalog.search(collections=[STAC_COLLECTION], ids=[s1_item_id]).item_collection()[0]
    # Get the item's CRS EPSG. This will be used in subsequent operations.
    epsg = int(item.properties["proj:code"][5:])
    print(f"Item CRS = EPSG:{epsg}")
    item
    return epsg, item, s1_item_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load our sample polygons and bounding AoI""")
    return


@app.cell
def _(epsg, gpd):
    sample_polys = gpd.read_file("eda_inverness/inverness-samples.gpkg").to_crs(epsg)
    aoi = gpd.read_file("eda_inverness/inverness_aoi.gpkg").to_crs(epsg)

    # A quick plot
    aoi.hvplot(data_aspect=1, title="Sample polygons and area bounds") * sample_polys.hvplot()
    return aoi, sample_polys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load S1 data""")
    return


@app.cell
def _(aoi, epsg, item, stackstac):
    RESOLUTION = 10

    ds = stackstac.stack(item, bounds=tuple(aoi.total_bounds), epsg=epsg, resolution=RESOLUTION, properties=False, band_coords=False)
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Perform dpolcat scaling""")
    return


@app.cell
def _(dpolcat, ds, epsg, xarray):
    vv_lin = ds.sel(band="vv")[0]
    vh_lin = ds.sel(band="vh")[0]

    vv_sn = dpolcat.xr_scale_nice(vv_lin)
    vh_sn = dpolcat.xr_scale_nice(vh_lin)

    dpds = xarray.Dataset({"vv": vv_sn.compute().drop_vars("band"), "vh": vh_sn.compute().drop_vars("band")})
    dpds.rio.write_crs(epsg, inplace=True)
    return (dpds,)


@app.cell
def _():
    # Export (+ dpolcat)
    # dpds["vv"].rio.to_raster("inverness-vv.tif")
    # dpcats = dpolcat.xr_categorize(vv_sn, vh_sn).compute()
    # dpcats.astype(np.uint8).rio.to_raster("inverness-dpolcat.tif")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Explore VV and VH relationships""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Sample the VV and VH band data according to our polygons

    Create a dataframe with the band data at each point in the sampling polygon.
    """
    )
    return


@app.cell
def _(dpds, pd, sample_polys):
    sdfs = []

    for idx, sample_poly in sample_polys.iterrows():
        sampled = dpds.rio.clip(geometries=[sample_poly.geometry])
        sdf = sampled[["vv", "vh"]] \
          .stack(point=("x", "y")).reset_index("point") \
          .to_dataframe().drop(columns=["epsg", "spatial_ref"]).dropna()
        sdf["sample_name"] = sample_poly["Name"]
        sdfs.append(sdf)

    df_samples = pd.concat(sdfs)
    return (df_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Perform subsampling

    To reduce the data size for visualization, perform subsampling - 10% of the original sample.
    """
    )
    return


@app.cell
def _(df_samples):
    # For inter-day analysis, bump up to 50% subsample. (perhaps we can validate on the remaining, later)
    df_subsampled = df_samples.groupby("sample_name").sample(frac=0.5, random_state=1000)

    # df_subsampled = df_samples.groupby("sample_name").sample(frac=0.1, random_state=2000)

    # For set sample size, use n=...
    # df_subsampled = df_samples.groupby("sample_name").sample(n=10, random_state=4000)
    return (df_subsampled,)


@app.cell
def _(df_subsampled):
    # Show sample counts (pixels)
    subsample_counts = df_subsampled.groupby("sample_name")["vv"].count().rename("count")
    subsample_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visualize the result""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Scatter plot.""")
    return


@app.cell
def _():
    # Not so useful, so skip this by default
    # df_subsampled.hvplot.scatter(x="vv", y="vh", by="sample_name",marker="x", data_aspect=1, frame_height=450)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Density plot.""")
    return


@app.cell
def _(df_subsampled, hv):
    hv.extension('bokeh')
    cvars = {'Agri-1': 'greens', 'Downtown-1': 'oranges', 'Lake-1': 'greys', 'Woodland-1': 'purples', 'Sea-1': 'blues'}
    ps = []
    for sample_name, cmap in cvars.items():
        d = df_subsampled.loc[df_subsampled['sample_name'] == sample_name]
        _p = d.hvplot.bivariate(x='vv', y='vh', cmap=cmap, data_aspect=1, frame_height=450, legend=False, colorbar=False, xlim=(0, 1), ylim=(0, 1))
        _p = _p.opts(axiswise=True)
        ps.append(_p)
    hv.Overlay(ps)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculate summary statistics.""")
    return


@app.cell
def _(df_subsampled):
    subsampled_stats = df_subsampled[["sample_name", "vv", "vh"]].groupby("sample_name").describe()

    # Flatten columns (needed for plots)
    subsampled_stats.columns = ['_'.join(col).strip() for col in subsampled_stats.columns.values]
    return (subsampled_stats,)


@app.cell
def _(s1_item_id, subsampled_stats):
    short_id = s1_item_id[17:32]
    subsampled_stats.to_csv(f"data/{short_id}.csv")
    return (short_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot medians.""")
    return


@app.cell
def _(short_id, subsampled_stats):
    ptitle = f'medians_{short_id}'
    _p = subsampled_stats[['vv_50%', 'vh_50%']].hvplot.scatter(x='vv_50%', y='vh_50%', by='sample_name', xlim=(0, 0.7), ylim=(0, 0.5), data_aspect=1, frame_width=570, xlabel='vv_median', ylabel='vh_median', title=ptitle)
    _p
    return (ptitle,)


@app.cell
def _(ptitle):
    print(ptitle)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Time-series analysis

    Using the results we exported before, build a multi-temporal dataframe.
    """
    )
    return


@app.cell
def _():
    from pathlib import Path
    return (Path,)


@app.cell
def _(Path):
    stat_csvs = list(Path("data").glob("*.csv"))
    return (stat_csvs,)


@app.cell
def _(np, pd, stat_csvs):
    _dfs = []
    for _stat_csv in stat_csvs:
        _fr = pd.read_csv(_stat_csv)
        _datestr = _stat_csv.name[:8]
        _fr["date"] = np.datetime64(f"{_datestr[:4]}-{_datestr[4:6]}-{_datestr[6:8]}")
        _fr = _fr.set_index(["date", "sample_name"])
        _dfs.append(_fr)

    df_stats_mt = pd.concat(_dfs)
    return (df_stats_mt,)


@app.cell
def _(df_stats_mt):
    # Calculate interquartile range
    df_stats_mt["vv_iqr"] = df_stats_mt["vv_75%"] - df_stats_mt["vv_25%"]
    df_stats_mt["vh_iqr"] = df_stats_mt["vh_75%"] - df_stats_mt["vh_25%"]
    return


@app.cell
def _(df_stats_mt):
    # A rough line plot of all sample medians. Is cluttered, so we go more selective soon.
    _df_stats_mt = df_stats_mt.reset_index() # Need this otherwise plotting is messed up
    (
        _df_stats_mt.hvplot(x="date", y="vv_50%", by="sample_name", legend=False, height=475)
        + _df_stats_mt.hvplot(x="date", y="vh_50%", by="sample_name", height=475)
    ).cols(1).opts(toolbar="below")
    return


@app.cell
def _(df_stats_mt):
    df_stats_mt.hvplot.scatter(
        x="vv_50%", y="vh_50%", by="sample_name",
        xlabel="vv_median", ylabel="vh_median",
        hover_cols=["date"], marker="x", cmap="glasbey_warm",
        height=550
        # xlim=(0,0.7), ylim=(0,0.5),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Selective plotting.""")
    return


@app.cell
def _(df_stats_mt):
    sel_samples = ["Lake-1", "Sea-1", "Agri-1", "Woodland-1", "Downtown-1"]
    df_stats_selected = df_stats_mt.loc[(slice(None), sel_samples), :]
    return (df_stats_selected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot medians and IQR time-series.""")
    return


@app.cell
def _(df_stats_selected):
    # For simplicity, keep this alphabetical
    colors = {
        "Agri-1": "green",
        "Downtown-1": "orangered",
        "Lake-1": "grey",
        "Sea-1": "blue",
        "Woodland-1": "purple",
    }

    clist = list(colors.values())

    _dfs = df_stats_selected.reset_index()
    (
        _dfs.hvplot(x="date", y="vv_50%", by="sample_name", color=clist) *
          _dfs.hvplot.area(x="date", y="vv_25%", y2="vv_75%", by="sample_name", stacked=False, color=clist)
        +
        _dfs.hvplot(x="date", y="vh_50%", by="sample_name", color=clist) *
          _dfs.hvplot.area(x="date", y="vh_25%", y2="vh_75%", by="sample_name", stacked=False, color=clist)
    ).cols(1)
    return


if __name__ == "__main__":
    app.run()
