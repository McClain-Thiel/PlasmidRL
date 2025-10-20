from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import boto3
from src.config import get_config
import torch
import os
import glob
import shutil
import tempfile
from transformers import AutoConfig


config = get_config()

s3_client = boto3.client("s3", config.region_name)
api = HfApi(token=config.huggingface_token)



def s3_download(client, bucket: str, prefix: str, dest: str):
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):].lstrip("/")
                local = os.path.join(dest, rel)
                if key.endswith("/"):
                    os.makedirs(local, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(local), exist_ok=True)
                client.download_file(bucket, key, local)


def checkpoint_to_huggingface(s3_client, checkpoint_path: str, huggingface_path: str):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = os.path.join(tmp, "raw")
        hf_dir  = os.path.join(tmp, "hf_out")
        os.makedirs(raw_dir, exist_ok=True)

        s3_parts = checkpoint_path.replace("s3://", "").split("/", 1)
        s3_bucket = s3_parts[0]
        s3_prefix = s3_parts[1] if len(s3_parts) > 1 else ""
        s3_download(s3_client, s3_bucket, s3_prefix, raw_dir)


        actor_dir = os.path.join(raw_dir, "actor")
        hf_assets = os.path.join(actor_dir, "huggingface")


        ckpt = sorted(glob.glob(os.path.join(actor_dir, "model_world_size_*_rank_0.pt")))
        if not ckpt:
            raise FileNotFoundError("Could not find model_world_size_*_rank_0.pt under actor/")
        ckpt_path = ckpt[0]

        model_config = AutoConfig.from_pretrained(hf_assets)
        model = AutoModelForCausalLM.from_config(model_config)

        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
        for pfx in ("module.", "model."):
            if any(k.startswith(pfx) for k in sd): sd = { (k[len(pfx):] if k.startswith(pfx) else k): v for k,v in sd.items() }

        model.load_state_dict(sd, strict=False)
        model.save_pretrained(hf_dir, safe_serialization=True)

        tok = AutoTokenizer.from_pretrained(hf_assets, use_fast=True)
        tok.save_pretrained(hf_dir)

        for fname in ("generation_config.json","special_tokens_map.json","tokenizer_config.json","tokenizer.json","vocab.json","merges.txt"):
            src = os.path.join(hf_assets, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(hf_dir, fname))

        api.create_repo(repo_id=huggingface_path, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=huggingface_path,
            repo_type="model",
            folder_path=hf_dir,
            commit_message="Convert VERL/GRPO actor checkpoint to HF format",
            token=config.huggingface_token,
        )

    return f"https://huggingface.co/{huggingface_path}"

            


