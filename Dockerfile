
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Avoid interactive tzdata prompts during apt installs
ENV TZ=Etc/UTC

# --- Minimal OS deps ---
RUN apt-get update && TZ=${TZ} DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential bash tini \
      openssh-client tzdata && \
    rm -rf /var/lib/apt/lists/*

# --- Non-root user (write-friendly on bind mounts) ---
ARG USERNAME=app
RUN useradd -m ${USERNAME}
USER ${USERNAME}
WORKDIR /workspace
SHELL ["/bin/bash", "-lc"]

# --- Install micromamba (for bio tools) ---
RUN curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | \
    tar -xj -C $HOME --strip-components=1 bin/micromamba && \
    echo 'export MAMBA_ROOT_PREFIX=$HOME/mamba' >> ~/.bashrc && \
    echo 'export PATH="$MAMBA_ROOT_PREFIX/envs/bio/bin:$PATH"' >> ~/.bashrc

ENV MAMBA_ROOT_PREFIX=/home/${USERNAME}/mamba
ENV PATH=/home/${USERNAME}:$PATH

# --- Create a micromamba env for bio binaries ONLY (no need to mix Pythons) ---
RUN micromamba create -y -n bio -c conda-forge -c bioconda \
      blast \
      diamond \
      infernal \
    && micromamba clean -a -y

# --- Install uv for fast Python dep management inside container ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
ENV PATH=/home/${USERNAME}/.cargo/bin:$PATH

# --- Good runtime defaults ---
ENV BLASTDB=/db/blast
ENV HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0

# NCCL / multi-GPU niceties (tweak as needed)
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=0
ENV NCCL_SOCKET_IFNAME=^lo,docker0
ENV OMP_NUM_THREADS=1
