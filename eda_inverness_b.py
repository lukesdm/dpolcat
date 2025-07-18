import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# dpolcat EDA - Inverness - Part B""")
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
    return gpd, hv, pd, planetary_computer, pystac_client, stackstac, xarray


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
    return (items,)


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
    mo.md(r"""Set the CRS for our workflow from the raster data.""")
    return


@app.cell
def _(items):
    _epsgs = set([int(item.properties["proj:code"][5:]) for item in items])
    assert len(_epsgs) == 1, "items are of different CRSs, this is not supported."
    epsg = _epsgs.pop()
    print(f"CRS = EPSG:{epsg}")
    return (epsg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load our sample polygons and bounding AoI""")
    return


@app.cell
def _(epsg, gpd):
    sample_polys = gpd.read_file("eda_inverness/inverness-samples.gpkg").to_crs(epsg)
    n_sample_feats = sample_polys.shape[0]
    aoi = gpd.read_file("eda_inverness/inverness_aoi.gpkg").to_crs(epsg)

    # A quick plot
    aoi.hvplot(data_aspect=1, title="Sample polygons and area bounds") * sample_polys.hvplot()
    return aoi, n_sample_feats, sample_polys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load S1 data""")
    return


@app.cell
def _(aoi, epsg, items, stackstac):
    # 10m resolution, matches the dataset.
    RESOLUTION = 10

    ds = stackstac.stack(items, bounds=tuple(aoi.total_bounds), epsg=epsg, resolution=RESOLUTION,
                         properties=False, band_coords=False)

    n_scenes = len(items)
    return ds, n_scenes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Perform dpolcat scaling and categorization""")
    return


@app.cell
def _(dpolcat, ds, epsg, xarray):
    vv_lin = ds.sel(band="vv")
    vh_lin = ds.sel(band="vh")

    vv_sn = dpolcat.xr_scale_nice(vv_lin)
    vh_sn = dpolcat.xr_scale_nice(vh_lin)

    dpds = xarray.Dataset({"vv": vv_sn.compute().drop_vars("band"), "vh": vh_sn.compute().drop_vars("band")})
    dpds = dpds.rio.write_crs(epsg)
    return dpds, vh_sn, vv_sn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Perform polarimetric categorization with dpolcat.""")
    return


@app.cell
def _(dpds, dpolcat, vh_sn, vv_sn):
    dpcats = dpolcat.xr_categorize(vv_sn, vh_sn).compute()
    dpds["dpolcat"] = dpcats
    return (dpcats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot with a corner subset. Note: Have to convert categories to strings for color map to work correctly.""")
    return


@app.cell
def _(dpolcat):
    _colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in dpolcat.color_list]
    # String keys needed
    dp_cmap = {str(k): _colors[k] for k in range(dpolcat.NUM_CATEGORIES)}
    # dp_cmap = dict(zip(range(dpolcat.NUM_CATEGORIES), _colors))
    #dp_cmap = {str(k): dp_cmap[k] for k in dp_cmap}
    return (dp_cmap,)


@app.cell
def _(dp_cmap, dpcats):
    _ss1 = dpcats[0][0:100, 0:100]
    _ss1.astype(str).hvplot(cmap=dp_cmap, data_aspect=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Explore VV, VH and category relationships""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Sample the band and category data according to our polygons

    Create a dataframe with the band data at each point in the sampling polygon.
    """
    )
    return


@app.cell
def _(dpds, pd, sample_polys):
    sdfs = []

    for idx, sample_poly in sample_polys.iterrows():
        sampled = dpds.rio.clip(geometries=[sample_poly.geometry])
        sdf = sampled[["vv", "vh", "dpolcat"]].to_dataframe() \
          .drop(columns=["epsg", "spatial_ref", "id"]).dropna()
        sdf["sample_name"] = sample_poly["Name"]
        sdfs.append(sdf)

    df_samples = pd.concat(sdfs)
    return (df_samples,)


@app.cell
def _():
    # TODO: Investigate invalid values. (there are 17 warnings... number of samples)
    # They are probably NaNs outside of the bounds of the image.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Perform subsampling

    Pick *n* pixels of each sample; apply this to all timesteps.
    """
    )
    return


@app.cell
def _(df_samples):
    N_SUBSAMPLE = 10 # Pixels per class
    # Select pixels based on first scene... we assume other scenes have matching grid.
    # Note: order of y and x here reflect indexing of the upstream xarray
    t0, _y, _x = df_samples.index[0]
    subsample_points = df_samples.loc[t0].groupby("sample_name").sample(n=N_SUBSAMPLE, random_state=1000)[["sample_name"]]

    # Generate matching indices for all timesteps.
    _subsample_indices = [(t, y, x) for t in df_samples.index.levels[0] for (y, x) in subsample_points.index]

    df_subsampled = df_samples.loc[_subsample_indices]
    return N_SUBSAMPLE, df_subsampled, t0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Re-index the dataframe: time, sample_name and a subsample_pixel_index (effectively a row number, 0..9).""")
    return


@app.cell
def _(N_SUBSAMPLE, df_subsampled, n_sample_feats, n_scenes):
    df_subsampled['px_idx'] = n_scenes * n_sample_feats * list(range(N_SUBSAMPLE))
    df_subsampled_1 = df_subsampled.reset_index(['y', 'x']).set_index(['sample_name', 'px_idx'], append=True)
    return (df_subsampled_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This allows some convenient time-series querying, e.g.:""")
    return


@app.cell
def _(df_subsampled_1):
    df_subsampled_1.loc[:, 'Agri-1', 0][['vv', 'vh']]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A quick plot showing subsample locations.""")
    return


@app.cell
def _(df_subsampled_1, t0):
    df_subsampled_1.loc[t0].hvplot.scatter(x='x', y='y', marker='x', by='sample_name')
    return


@app.cell
def _(df_subsampled_1):
    subsample_counts = df_subsampled_1.groupby(['time', 'sample_name'])['vv'].count().rename('count')
    subsample_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Individual pixel changes over time

    How much individual pixels change over time, rather than as a group, determines use of pixel-wise categorization.

    The concept of 'categorical stability'.
    """
    )
    return


@app.cell
def _(df_subsampled_1, dp_cmap, hv):
    def plot_dpolcat_timeseries(sel_sample):
        _dfs = df_subsampled_1.loc[:, sel_sample, :]
        _dfs = _dfs.copy()
        _dfs['dpolcat'] = _dfs['dpolcat'].astype('str')
        _hvds = hv.Dataset(data=_dfs, kdims=['time', 'px_idx'], vdims=['dpolcat'])
        _hm = hv.HeatMap(_hvds).opts(cmap=dp_cmap)
        _p = _hm * hv.Labels(_hm).opts(title=f'Pol. Cats.: {sel_sample}')
        return _p
    plot_dpolcat_timeseries('Agri-1')
    return (plot_dpolcat_timeseries,)


@app.cell
def _(hv, plot_dpolcat_timeseries):
    _ps = [plot_dpolcat_timeseries(_sel_sample) for _sel_sample in ['Agri-1', 'Agri-2', 'Moorland-1', 'Woodland-1', 'Woodland-2', 'Sea-1', 'Lake-1', 'Downtown-1', 'Industrial-1']]
    hv.Layout(_ps).cols(3)
    return


@app.cell
def _(df_subsampled_1):
    _sel_sample = 'Agri-1'
    _dfs = df_subsampled_1.loc[:, _sel_sample, :]
    (_dfs.hvplot(x='time', y='vv', by='px_idx', title=f'VV, {_sel_sample}') + _dfs.hvplot(x='time', y='vh', by='px_idx', title=f'VH, {_sel_sample}')).cols(1)
    return


@app.cell
def _(df_subsampled_1):
    _sel_sample = 'Sea-1'
    _dfs = df_subsampled_1.loc[:, _sel_sample, :]
    (_dfs.hvplot(x='time', y='vv', by='px_idx', title=f'VV, {_sel_sample}') + _dfs.hvplot(x='time', y='vh', by='px_idx', title=f'VH, {_sel_sample}')).cols(1)
    return


@app.cell
def _(df_subsampled_1):
    _sel_sample = 'Downtown-1'
    _dfs = df_subsampled_1.loc[:, _sel_sample, :]
    (_dfs.hvplot(x='time', y='vv', by='px_idx', title=f'VV, {_sel_sample}') + _dfs.hvplot(x='time', y='vh', by='px_idx', title=f'VH, {_sel_sample}')).cols(1)
    return


@app.cell
def _(df_subsampled_1):
    _sel_sample = 'Woodland-1'
    _dfs = df_subsampled_1.loc[:, _sel_sample, :]
    (_dfs.hvplot(x='time', y='vv', by='px_idx', title=f'VV, {_sel_sample}') + _dfs.hvplot(x='time', y='vh', by='px_idx', title=f'VH, {_sel_sample}')).cols(1)
    return


@app.cell
def _(df_subsampled_1):
    grp = df_subsampled_1[['vv', 'vh']].groupby(['sample_name', 'px_idx'])
    ranges = grp.max() - grp.min()
    ranges = ranges.rename(columns={'vv': 'vv_range', 'vh': 'vh_range'})
    return grp, ranges


@app.cell
def _(ranges):
    ranges.hvplot.scatter(x="vv_range", y="vh_range", by=["sample_name"], title="Subsample pixel range over time", marker="x", height=470, hover_cols=["px_idx"])
    return


@app.cell
def _(ranges):
    ranges.groupby("sample_name").mean().hvplot.scatter(x="vv_range", y="vh_range", by=["sample_name"], title="Mean of subsample pixels' ranges over time", height=470)
    return


@app.cell
def _(grp):
    means = grp.mean().rename(columns={"vv": "vv_mean", "vh": "vh_mean"})
    return (means,)


@app.cell
def _(means, pd, ranges):
    combined = pd.concat([ranges, means], axis=1)
    return (combined,)


@app.cell
def _(combined):
    _cagg = combined.groupby("sample_name").mean()

    (
        _cagg.hvplot.scatter(x="vv_mean", y="vv_range", by="sample_name", title="VV: Mean vs range", height=470) +
        _cagg.hvplot.scatter(x="vh_mean", y="vh_range", by="sample_name", title="VH: Mean vs range", height=470)
    ).cols(1)
    return


if __name__ == "__main__":
    app.run()
