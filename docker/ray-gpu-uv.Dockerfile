FROM rayproject/ray:2.34.0-gpu

# Install uv and minimal tooling (CUDA base already present in Ray GPU image)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app
# Copy project files for dependency resolution first (better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv venv && . ./.venv/bin/activate && uv sync --frozen || true

# Copy the rest of the repo and set PYTHONPATH
COPY . .
ENV PYTHONPATH=/app

# Default command
CMD ["bash", "-lc", "python -V && nvidia-smi || true && ray --help"]

