FROM ghcr.io/marimo-team/marimo:0.13.15

# Install GDAL and Python.
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    python3-dev

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1
ENV GDAL_CONFIG=/usr/bin/gdal-config

# GDAL Python bindings - has to match GDAL version installed by apt-get
RUN uv pip install gdal==3.6.2

# Install other Python libraries
RUN uv pip install rioxarray geopandas shapely planetary-computer stackstac dask[complete] plotly holoviews hvplot scipy numba geoviews
