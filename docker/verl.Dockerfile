# Start from the official VERL app image (already has CUDA/PyTorch/vLLM/etc.)
FROM verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.7.4

# Install git and patch utility
RUN apt-get update && apt-get install -y git patch && rm -rf /var/lib/apt/lists/*

# Clone VERL source and install only the verl package (no deps; base image has them)
RUN git clone --depth 1 https://github.com/volcengine/verl /opt/verl && \
    pip3 install --no-deps -e /opt/verl

# Copy and apply monkey patch for GPT-2 compatibility
COPY docker/verl-monkey.patch /tmp/verl-monkey.patch
RUN cd /opt/verl && patch -p1 < /tmp/verl-monkey.patch && rm /tmp/verl-monkey.patch
