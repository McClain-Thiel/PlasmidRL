
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Avoid interactive tzdata prompts during apt installs
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# --- Minimal OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential bash tini \
      openssh-client tzdata && \
    rm -rf /var/lib/apt/lists/*

# --- Non-root user (write-friendly on bind mounts) ---
ARG USERNAME=app
RUN useradd -m ${USERNAME}
USER ${USERNAME}
WORKDIR /workspace
SHELL ["/bin/bash", "-lc"]

# --- Install uv for fast Python dep management inside container ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
ENV PATH=/home/${USERNAME}/.cargo/bin:/home/${USERNAME}/.local/bin:$PATH

# --- Good runtime defaults ---
ENV HF_HOME=/mcclain/.cache/huggingface
ENV HF_DATASETS_CACHE=/mcclain/.cache/huggingface/datasets
ENV HF_HUB_CACHE=/mcclain/.cache/huggingface/hub
ENV TRANSFORMERS_OFFLINE=0

# NCCL / multi-GPU niceties (tweak as needed)
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=0
ENV NCCL_SOCKET_IFNAME=^lo,docker0
ENV OMP_NUM_THREADS=1
