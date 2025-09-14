# 🚀 Running with Docker Compose (GPU)

This project ships a GPU-ready Docker/Compose setup that builds a single deep learning + bioinformatics base image and exposes multiple services (`dev`, `train`, `eval`, `notebook`, `tensorboard`). Each service maps to a Python entrypoint so you can run long-lived training, evaluations, or interactive sessions with the same reproducible environment. Databases and outputs are mounted as volumes; GPUs are managed via `CUDA_VISIBLE_DEVICES` in `.env`. Build once (`docker build -t plasmidrl:cuda12.1 .`), then run tasks with `docker compose run --rm train`, `docker compose run --rm eval`, or spin up visualization with `docker compose up tensorboard`.

---

## 📦 Services & Examples

### 🐚 Development Shell

Interactive dev shell with GPUs and project files mounted:

```bash
docker compose run --rm dev
```

### 🏋️ Training (long-running)

Launch a multi-GPU training job (uses `torchrun` under the hood):

```bash
docker compose run --rm train
```

* Number of GPUs per node = `NPROC_PER_NODE` (from `.env`).
* Checkpoints → `./checkpoints`
* Logs → `./logs`
* Resume by setting `RESUME=true` in `.env`.

### 📊 Evaluation

Evaluate a saved checkpoint:

```bash
docker compose run --rm eval
```

* Expects `./checkpoints/latest.pt` (or adjust args in `docker-compose.yml`).
* Writes outputs to `./logs`.

### 📓 Jupyter Lab

Start Jupyter at [http://localhost:8888](http://localhost:8888):

```bash
docker compose up notebook
```

* Mounts `./notebooks` for persistence.
* No token/password by default (adjust in compose if needed).

### 📈 TensorBoard

Visualize logs at [http://localhost:6006](http://localhost:6006):

```bash
docker compose up tensorboard
```

### 🔍 Logs & Monitoring

Follow live logs for any service:

```bash
docker compose logs -f train
```

Stop a running service:

```bash
docker compose stop train
```

Attach an interactive shell to a running container:

```bash
docker compose exec train bash
```

---

## ⚙️ Configuration

All runtime configuration is controlled via `.env`. Copy and edit:

```bash
cp .env.example .env
```

* **`CUDA_VISIBLE_DEVICES`**: comma-separated GPU IDs to expose to the container.
* **`NPROC_PER_NODE`**: number of GPUs used by `torchrun` (1 process/GPU).
* **`RESUME`**: `true|false` to auto-resume from latest checkpoint.
* **`UID`/`GID`**: host user/group IDs so files written from the container are owned by you on the host.

---

## 🧠 Advanced Usage

### Run multiple jobs concurrently

Duplicate services or override at the command line:

```bash
# Example: launch two training jobs on different GPU sets
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 docker compose run --rm train
CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 docker compose run --rm train
```

### Background (detached) jobs

```bash
# Run training in the background and keep it alive across disconnects
CUDA_VISIBLE_DEVICES=0,1 docker compose up -d train
# Tail logs
docker compose logs -f train
# Stop when done
docker compose stop train
```


## 📄 `.env.example`

```dotenv
# GPU selection (comma-separated indices)
CUDA_VISIBLE_DEVICES=0,1

# Number of GPUs (processes) per node for torchrun
NPROC_PER_NODE=2

# Resume training from last checkpoint (true/false)
RESUME=false

# Host UID/GID for file ownership on bind mounts
UID=1000
GID=1000
```

---

## 🔗 FastAPI server integration (required)

- Required by `train`, `eval`, and `notebook`.
- Clone the server repo next to this project or set a custom path:

```bash
git clone https://github.com/ucl-cssb/Plasmid-Informatics-Server ../Plasmid-Informatics-Server
```

- Minimal `.env` entries:

```dotenv
PLASMID_SERVER_PATH=../Plasmid-Informatics-Server
PLASMID_API_URL=http://server:8000
```

- Start backend and services:

```bash
docker compose up -d server
# then
docker compose run --rm train     # or: docker compose run --rm eval
# for notebooks
docker compose up -d notebook     # open http://localhost:8888
```

- Inside containers, call the API via `PLASMID_API_URL` (defaults to `http://server:8000`).
