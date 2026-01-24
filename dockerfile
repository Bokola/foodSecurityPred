FROM python:3.11-slim

# Install system spatial dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgdal-dev libproj-dev libgeos-dev proj-bin \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into the SYSTEM python environment
# This ensures any 'python' command finds the packages
RUN uv pip install --system -r pyproject.toml

# Copy source code and input data
COPY src/ ./src/
COPY input_collector/ ./input_collector/

# Set Python path
ENV PYTHONPATH="/app"