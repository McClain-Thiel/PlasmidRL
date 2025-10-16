FROM rayproject/ray:2.34.0-py310

# Install uv and minimal tooling
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

# Default command does nothing; Ray will override for head/worker/driver
CMD ["bash", "-lc", "python -V && ray --help"]

