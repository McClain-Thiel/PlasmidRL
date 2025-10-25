
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Avoid interactive tzdata prompts during apt installs
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# --- Minimal OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential bash tini \
      openssh-client tzdata && \
    rm -rf /var/lib/apt/lists/*

# --- Run as root for consistency with VERL and simpler permissions ---
WORKDIR /mcclain
SHELL ["/bin/bash", "-lc"]

# --- Install micromamba (for bio tools) ---
RUN curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | \
    tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    echo 'export MAMBA_ROOT_PREFIX=/opt/mamba' >> ~/.bashrc && \
    echo 'export PATH="/opt/mamba/envs/bio/bin:$PATH"' >> ~/.bashrc

ENV MAMBA_ROOT_PREFIX=/opt/mamba
ENV PATH=/usr/local/bin:/opt/mamba/envs/bio/bin:$PATH

# --- Create a micromamba env for bio binaries ONLY (no need to mix Pythons) ---
RUN /usr/local/bin/micromamba create -y -n bio -c conda-forge -c bioconda \
      blast \
      diamond \
      infernal \
    && /usr/local/bin/micromamba clean -a -y

# --- Install uv for fast Python dep management inside container ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
ENV PATH=/root/.local/bin:$PATH

# --- Good runtime defaults ---
ENV BLASTDB=/db/blast
ENV HF_HOME=/mcclain/.cache/huggingface
ENV HF_DATASETS_CACHE=/mcclain/.cache/huggingface/datasets
ENV HF_HUB_CACHE=/mcclain/.cache/huggingface/hub
ENV TRANSFORMERS_OFFLINE=0

# NCCL / multi-GPU niceties (tweak as needed)
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=0
ENV NCCL_SOCKET_IFNAME=^lo,docker0
ENV OMP_NUM_THREADS=1
