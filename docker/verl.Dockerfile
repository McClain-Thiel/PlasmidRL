# Start from the official VERL app image (already has CUDA/PyTorch/vLLM/transformers/etc.)
FROM verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

# Install git and patch utility
RUN apt-get update && apt-get install -y git patch && rm -rf /var/lib/apt/lists/*

# Clone VERL source and install only the verl package (no deps; base image has them)
RUN git clone --depth 1 https://github.com/volcengine/verl /opt/verl && \
    pip3 install --no-deps -e /opt/verl

# Copy and apply monkey patch for GPT-2 compatibility
COPY docker/verl-monkey.patch /tmp/verl-monkey.patch
RUN cd /opt/verl && patch -p1 < /tmp/verl-monkey.patch && rm /tmp/verl-monkey.patch

# Install plasmidkit from git
RUN pip3 install git+https://github.com/McClain-Thiel/plasmid-kit.git

WORKDIR /mcclain
