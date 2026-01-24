FROM python:3.11-slim

# Install system spatial dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev libproj-dev libgeos-dev proj-bin \
    && rm -rf /var/lib/apt/lists/*

# Install uv and project dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy source code and input data
COPY src/ ./src/
COPY input_collector/ ./input_collector/

# Set Python path to ensure 'src' is treated as a package
ENV PYTHONPATH="/app"

# No ENTRYPOINT here; the Pipeline will specify which script to run