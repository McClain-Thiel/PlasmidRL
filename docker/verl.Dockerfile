# Start from the official VERL app image (already has CUDA/PyTorch/vLLM/etc.)
FROM verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

# Install git (needed if you want to clone)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone VERL source and install only the verl package (no deps; base image has them)
RUN git clone --depth 1 https://github.com/volcengine/verl /opt/verl && \
    pip3 install --no-deps -e /opt/verl
