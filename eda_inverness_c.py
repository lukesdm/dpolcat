import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # dpolcat EDA - Inverness - Part C

    Experiments with speckle filtering.
    """
    )
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
def _(mo):
    load_button = mo.ui.run_button(label="Click to start workflow")
    load_button
    return (load_button,)


@app.cell
def _(
    load_button,
    mo,
    planetary_computer,
    pystac_client,
    search_end,
    search_poly_coords,
    search_start,
):
    mo.stop(not load_button.value)

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


@app.cell
def _(ds):
    # Load raw (linear) input data. compute() here to avoid issues due to MSPC tokens expiring.
    vv_lin = ds.sel(band="vv").compute()
    vh_lin = ds.sel(band="vh").compute()
    return vh_lin, vv_lin


@app.cell
def _(xarray):
    from fast_frost import frost_filter_fast, frost_filter_nanny

    def frost_xr(da: xarray.DataArray, nanny=True, damping_factor=2.0, win_size=5) -> xarray.DataArray:
        """Apply 2D (y, x) Frost filter to 3D (y, x, time) xarray DataArray; See underlying filter function for damping_factor and win_size."""
        filtered = xarray.apply_ufunc(
            frost_filter_nanny if nanny else frost_filter_fast,
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
def _(frost_xr, vh_lin, vv_lin):
    vv_filt = frost_xr(vv_lin.fillna(-1), nanny=False)
    vh_filt = frost_xr(vh_lin.fillna(-1), nanny=False)
    return vh_filt, vv_filt


@app.cell
def _(mo):
    filter_toggle = mo.ui.radio(options=["Unfiltered", "Filtered"], value="Filtered")
    filter_toggle
    return (filter_toggle,)


@app.cell
def _(filter_toggle):
    print(filter_toggle.value)
    return


@app.cell
def _(dpolcat, filter_toggle, vh_filt, vh_lin, vv_filt, vv_lin):
    if filter_toggle.value == "Filtered":
        vv_sn = dpolcat.xr_scale_nice(vv_filt)
        vh_sn = dpolcat.xr_scale_nice(vh_filt)
    else:
        vv_sn = dpolcat.xr_scale_nice(vv_lin)
        vh_sn = dpolcat.xr_scale_nice(vh_lin)
    return vh_sn, vv_sn


@app.cell
def _(dpolcat, vv_filt, vv_lin):
    # A little demo of the filtering
    _ox, _oy = 300, 0
    _unfilt = dpolcat.xr_scale_nice(vv_lin[0][_oy:_oy+1100, _ox:_ox+1000])
    _filt = dpolcat.xr_scale_nice(vv_filt[0][_oy:_oy+1100, _ox:_ox+1000])
    _unfilt.hvplot(data_aspect=1.0) + _filt.hvplot(data_aspect=1.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Perform dpolcat scaling and categorization""")
    return


@app.cell
def _(dpolcat, epsg, vh_sn, vv_sn, xarray):
    dpds = xarray.Dataset({"vv": vv_sn.drop_vars("band"), "vh": vh_sn.drop_vars("band")})
    dpds = dpds.rio.write_crs(epsg)
    dpcats = dpolcat.xr_categorize(vv_sn, vh_sn).compute()
    dpds["dpolcat"] = dpcats
    return dpcats, dpds


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
    _ss1 = dpcats[0][0:200, 0:200]
    _ss1.astype(str).hvplot(cmap=dp_cmap, data_aspect=1)
    return


@app.cell
def _():
    # Export
    # dpds["vv"].rio.to_raster("data/inverness2-vv.tif")
    # dpcats = dpolcat.xr_categorize(vv_sn, vh_sn).compute()
    # dpcats.astype(np.uint8).rio.to_raster("inverness-dpolcat.tif")
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
    # plot_dpolcat_timeseries('Agri-1')
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


@app.cell
def _(mo):
    mo.md(r"""Bivariate VV-VH distribution plotting for a given date (or over all dates, if none selected).""")
    return


@app.cell(hide_code=True)
def _(df_subsampled, mo):
    def date_from_ts(timestamp):
        return str(timestamp)[:10]

    _available_dates = list(df_subsampled.index.unique(level="time"))
    _options = {date_from_ts(d): d for d in _available_dates}
    _default = date_from_ts(_available_dates[0])
    # date_dropdown = mo.ui.dropdown(options=_options, value=_default)
    date_dropdown = mo.ui.dropdown(options=_options)
    date_dropdown
    return date_dropdown, date_from_ts


@app.cell(hide_code=True)
def _(date_dropdown, date_from_ts, df_subsampled, filter_toggle, hv):
    _sel_date = date_dropdown.value
    _date_label = date_from_ts(_sel_date) if _sel_date is not None else "All dates"

    cvars = {'Agri-1': 'greens', 'Downtown-1': 'oranges', 'Lake-1': 'greys', 'Woodland-1': 'purples', 'Sea-1': 'blues'}
    ps = []
    _df = df_subsampled.loc[_sel_date] if _sel_date is not None else df_subsampled 
    for sample_name, cmap in cvars.items():
        d = _df.loc[_df['sample_name'] == sample_name]
        _p = d.hvplot.bivariate(x='vv', y='vh', cmap=cmap, data_aspect=1, frame_height=450, legend=False, colorbar=False, xlim=(0, 1), ylim=(0, 1))
        _p = _p.opts(axiswise=True)
        ps.append(_p)
    biv_plot = hv.Overlay(ps).opts(
        title=f"VV-VH distribution, {_date_label}, {filter_toggle.value}")
    return biv_plot, cvars


@app.cell(hide_code=True)
def _(cvars, hv):
    # Create a simple legend plot using rectangles. Mainly generated with Claude Sonnet 4.

    colormap_colors = {
        "greens": "#2d8f2d",
        "oranges": "#ff8c00",
        "greys": "#696969",
        "purples": "#8a2be2",
        "blues": "#1e90ff",
    }

    def make_biv_legend():
        legend_elements = []

        # Create rectangles and text for each sample
        for i, (sample_name, cmap_name) in enumerate(cvars.items()):
            color = colormap_colors[cmap_name]

            # Create a rectangle for each sample
            rect = hv.Rectangles([(0, i - 0.2, 0.2, i + 0.2)]).opts(
                color=color, line_color="black", line_width=1
            )

            # Add text label
            text = hv.Text(0.3, i, sample_name).opts(
                text_align="left", text_font_size="10pt"
            )

            legend_elements.extend([rect, text])

        legend_plot = hv.Overlay(legend_elements).opts(
            xlim=(-0.1, 2),
            ylim=(-0.5, len(cvars) - 0.5),
            width=150,
            height=150,
            title="",
            xlabel="",
            ylabel="",
            xaxis=None,
            yaxis=None,
            show_grid=False,
        )

        return legend_plot


    # Create the legend
    legend = make_biv_legend()
    return (legend,)


@app.cell
def _(biv_plot, legend):
    biv_plot + legend
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
