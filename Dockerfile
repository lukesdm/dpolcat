FROM ghcr.io/prefix-dev/pixi:0.84.1-noble

# Prevent environment (dependencies) folder being created in app folder, which kills perf on WSL.
RUN pixi config set --global detached-environments true

# Install dependencies
WORKDIR /app
COPY pyproject.toml pixi.lock* README.md LICENSE ./
COPY src/dpolcat/__init__.py ./src/dpolcat/__init__.py
RUN pixi install

# No need to copy anything else - it will be volume mounted.
# COPY . .

ENV MARIMO_SKIP_UPDATE_CHECK=1

EXPOSE 8080
CMD ["pixi", "run", "marimo", "edit", "--host", "0.0.0.0", "--port", "8080", "--no-token"]
