SFT scripts for BioNeMo Evo2

Prereqs
- NVIDIA Docker with GPU access
- Pulled image set in `sft/config.env` (default: nvcr.io/nvidia/clara/bionemo-framework:2.6.3)
- Large host storage mounted at `/mcclain`

Setup
1) Edit `sft/config.env` to adjust paths and options (image, caches, dataset/model paths, W&B)
2) Ensure caches/output dirs exist (scripts will create them as needed)

Convert Evo2 HF → NeMo2
```bash
/mcclain/projects/PlasmidRL/sft/convert_model.sh
```
Outputs to `OUTPUT_NEMO_DIR` (default `/mcclain/models/nemo2_evo2_1b_8k`).

Preprocess FASTA → indexed dataset
1) Put your FASTA at `DATA_FASTA` (default `/mcclain/projects/data/sequences.fasta`)
2) Adjust `sft/preprocess_evo2.json` if needed (datapaths, output_dir, splits)
3) Run:
```bash
/mcclain/projects/PlasmidRL/sft/preprocess_fasta.sh
# or specify a different JSON
/mcclain/projects/PlasmidRL/sft/preprocess_fasta.sh sft/other.json
```
Outputs `.bin/.idx` under `PREPROC_OUTPUT_DIR` (default `/mcclain/datasets/evo2_preprocessed/sequences`).

Train / Fine-tune Evo2 (with W&B)
1) Set your W&B credentials:
```bash
export WANDB_API_KEY=YOUR_KEY
```
2) Optionally set `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_TAGS` in `sft/config.env`
3) Adjust training knobs in `sft/config.env` (e.g., `SEQ_LENGTH`, `MODEL_SIZE`, `MICRO_BATCH_SIZE`, `GRAD_ACC_BATCHES`, `CKPT_DIR`)
4) Ensure `sft/train_evo2.yaml` points to your preprocessed dataset prefixes
5) Run:
```bash
/mcclain/projects/PlasmidRL/sft/train_evo2.sh
```

Notes
- All scripts run inside the BioNeMo container with GPU and mount `/mcclain`; caches and temps are kept on `/mcclain`.
- Reduce OOM risk by lowering `SEQ_LENGTH` (e.g., 1024), increasing `GRAD_ACC_BATCHES`, enabling activation checkpointing, and optionally `FP8=true` in `sft/config.env`.

References
- BioNeMo container usage: https://docs.nvidia.com/bionemo-framework/latest/main/getting-started/access-startup/#running-the-container-on-a-local-machine
- Evo2 data preprocessing: https://github.com/NVIDIA/bionemo-framework/blob/main/sub-packages/bionemo-evo2/src/bionemo/evo2/data/README.md

