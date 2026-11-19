# An implementation of GeoZarr multiscale/overview-building which supports
# appending to existing stores (which GDAL's doesn't yet). Nearest-neighbour only.
#
# Claude Opus 5.5

import math

import dask
import numpy as np
import rioxarray  # noqa: F401
import xarray
import zarr

MULTISCALES_UUID = "d35379db-88df-4056-af3a-620245f8e347"
_INHERITED_ENCODING = (
    "compressors",
    "filters",
    "serializer",
    "fill_value",
    "dtype",
    "_FillValue",
    "scale_factor",
    "add_offset",
    "units",
    "calendar",
    "grid_mapping",
)


def build_overviews(store, group, name, factors=(2, 4, 8, 16), resampling="nearest"):
    """Create or extend nearest-neighbour overviews of `{group}/{name}/{name}` in a Zarr v3 store.

    Only time steps missing from the overviews are computed. Overview groups that don't
    match the base array (wrong levels, shape or times, e.g. GDAL-built) are rebuilt.
    Call after every append to the base array: xarray's appends drop the group's
    multiscales attributes, and this re-writes them.
    """
    base_path = f"{group}/{name}"
    root = zarr.open_group(store, path=base_path, mode="a", use_consolidated=False)
    base = xarray.open_zarr(
        store, group=base_path, consolidated=False, decode_coords="all"
    )
    da = base[name]
    levels = overview_pyramid(da, factors)
    n_done = _steps_done(root, name, levels)

    writes = []
    for asset, ovr in levels.items():
        new = ovr.isel(time=slice(n_done, None)).drop_encoding()
        if not new.sizes["time"]:
            continue
        encoding = (
            None
            if n_done
            else {
                name: _inherit(da.encoding)
                | {
                    "chunks": tuple(
                        min(c, s) for c, s in zip(root[name].chunks, ovr.shape)
                    )
                },
                "time": _inherit(base.time.encoding),
            }
        )
        writes.append(
            new.to_dataset(name=name).to_zarr(
                store,
                group=f"{base_path}/{asset}",
                mode="a",
                append_dim="time" if n_done else None,
                zarr_format=3,
                consolidated=False,
                encoding=encoding,
                compute=False,
                align_chunks=True,
            )
        )
    dask.compute(*writes)
    root.attrs.update(multiscales_attrs(name, da, levels, resampling))


def overview_pyramid(da, factors):
    """Lazy GDAL-equivalent nearest overviews (each level sampled from the previous one)."""
    x0, dx = float(da.x[0]), float(da.x[1] - da.x[0])
    y0, dy = float(da.y[0]), float(da.y[1] - da.y[0])
    levels, prev = {}, da
    for f in sorted(factors):
        nx, ny = math.ceil(da.sizes["x"] / f), math.ceil(da.sizes["y"] / f)
        ovr = prev.isel(
            x=_gdal_nearest_idx(prev.sizes["x"], nx),
            y=_gdal_nearest_idx(prev.sizes["y"], ny),
        ).assign_coords(  # extent-preserving, as GDAL georeferences overviews
            x=x0 - dx / 2 + (np.arange(nx) + 0.5) * dx * da.sizes["x"] / nx,
            y=y0 - dy / 2 + (np.arange(ny) + 0.5) * dy * da.sizes["y"] / ny,
        )
        levels[f"ovr_{f}x"] = prev = ovr.rio.write_transform(
            ovr.rio.transform(recalc=True)
        )
    return levels


def multiscales_attrs(name, base, levels, resampling="nearest"):
    layout = [{"asset": name, "transform": {"scale": [1.0] * base.ndim}}]
    prev_name, prev = name, base
    for asset, ovr in levels.items():
        layout.append(
            {
                "asset": asset,
                "derived_from": prev_name,
                "transform": {
                    "scale": [prev.sizes[d] / ovr.sizes[d] for d in ovr.dims],
                    "translation": [0.0] * ovr.ndim,
                },
                "resampling_method": resampling,
            }
        )
        prev_name, prev = asset, ovr
    return {
        "zarr_conventions": [
            {
                "uuid": MULTISCALES_UUID,
                "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
                "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
                "name": "multiscales",
                "description": "Multiscale layout of zarr datasets",
            }
        ],
        "multiscales": {"layout": layout},
    }


def _gdal_nearest_idx(n_src, n_dst):
    # Same pixel selection as GDALResampleChunk_Near: int(0.5 + i * src/dst)
    return np.minimum((0.5 + np.arange(n_dst) * (n_src / n_dst)).astype(int), n_src - 1)


def _inherit(encoding):
    return {k: v for k, v in encoding.items() if k in _INHERITED_ENCODING}


def _steps_done(root, name, levels):
    """Time steps already in every overview level; 0 (after clearing them) if inconsistent."""
    existing = {k for k in root.group_keys() if k.startswith("ovr_")}
    try:
        base_time = root["time"][:]
        done = set()
        for asset, ovr in levels.items():
            arr, t = root[f"{asset}/{name}"], root[f"{asset}/time"][:]
            spatial = {
                d: s
                for d, s in zip(arr.metadata.dimension_names, arr.shape)
                if d != "time"
            }
            if (
                existing != set(levels)
                or spatial != {d: ovr.sizes[d] for d in ovr.dims if d != "time"}
                or not np.array_equal(t, base_time[: len(t)])
            ):
                raise ValueError
            done.add(len(t))
        if len(done) == 1:
            return done.pop()
    except (KeyError, ValueError):
        pass
    for k in existing:
        del root[k]
    return 0
