import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from pathlib import Path

    import geopandas as gpd

    import dpolcat.utils
    import dpolcat.processor


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Script: Prepare Graz Datacube
    """)
    return


@app.cell
def _():
    aoi = gpd.read_file("gis/aois/graz_4326.gpkg")

    date_range = "2020-01-01/2025-12-31"
    return aoi, date_range


@app.cell
def _():
    out_zarr_path = "data/dpolcat_graz_full.zarr"

    assert not Path(out_zarr_path).exists(), f"Datacube '{out_zarr_path}' already exists, remove it and try again."
    return (out_zarr_path,)


@app.cell
def _():
    sat_direction = "descending"
    sat_rel_orbit = 22
    return sat_direction, sat_rel_orbit


@app.cell
def _(aoi, date_range, out_zarr_path, sat_direction, sat_rel_orbit):
    processor_params = dict(
        p_headless=True,
        p_aoi=aoi,
        p_date_range=date_range,
        p_sat_direction=sat_direction,
        p_sat_rel_orbit=sat_rel_orbit,
        p_speckle_filter="Frost",
        p_speckle_filter_window=5,
        p_out_zarr_path=out_zarr_path,
        p_write_intermediate_data=False,
    )
    return (processor_params,)


@app.cell
def _(processor_params):
    _, out_vars = dpolcat.processor.app.run(processor_params)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
