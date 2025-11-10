from vllm import LLM, SamplingParams
from src.config import get_config
import datetime
import boto3
import io
import pandas as pd
import os
from typing import Optional


config = get_config()

def process_outputs(df: pd.DataFrame, folder_name: Optional[str] = None):
    if not folder_name:
        folder_name = config.sample_model

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = (config.s3_bucket.rstrip("/") + "/" +
            config.infered_path.strip("/") + "/" +
            folder_name + "_" + ts + "/")

    # parse s3://bucket/prefix -> bucket, prefix
    s3_uri = base if base.startswith("s3://") else "s3://" + base
    no_scheme = s3_uri.replace("s3://", "")
    bucket, key_prefix = no_scheme.split("/", 1)

    s3 = boto3.client("s3")

    # -- CSV --
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    s3.put_object(Bucket=bucket, Key=key_prefix + "outputs.csv", Body=csv_buf.getvalue().encode("utf-8"))

    # -- FASTA (save each sequence to its own file) --
    if "full" not in df.columns:
        raise ValueError("DataFrame must include a 'full' column.")
    id_col = "id" if "id" in df.columns else None

    for i, row in df.iterrows():
        header = f">{row[id_col]}" if id_col else f">record_{i}"
        fasta_body = f"{header}\n{str(row['full']).strip()}\n"
        
        # Create a unique filename for each sequence
        fasta_filename = f"sequence_{i}.fasta" if not id_col else f"{row[id_col]}.fasta"
        
        s3.put_object(Bucket=bucket, Key=key_prefix + fasta_filename, Body=fasta_body.encode("utf-8"))

    print(f"Saved to {s3_uri}")


def main():
    prompts = [config.default_query, "ATG"] * 50 #strong prompt and weak prompt

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.8,
        top_p=0.95,
        top_k=0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop_token_ids=[2] #be careful here, this is hard coded to the [SEP] token for the GPT2 model
    )
    
    # Normalize the model path to ensure it's recognized as a local path
    model_path = os.path.normpath(config.sample_model.rstrip("/"))
    original_path = model_path
    
    # Translate host path to Docker container path if needed
    # Host: /mnt/s3/phd-research-storage-1758274488/... -> Container: /s3/...
    container_path = None
    if model_path.startswith("/mnt/s3/phd-research-storage-1758274488"):
        container_path = model_path.replace("/mnt/s3/phd-research-storage-1758274488", "/s3", 1)
        if os.path.exists(container_path):
            model_path = container_path
    
    # Verify the path exists and has config.json
    if not os.path.exists(model_path):
        checked_paths = [original_path]
        if container_path:
            checked_paths.append(container_path)
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}\n"
            f"Checked paths:\n" + "\n".join(f"  - {p}" for p in checked_paths)
        )
    
    config_json = os.path.join(model_path, "config.json")
    if not os.path.exists(config_json):
        raise FileNotFoundError(f"config.json not found at: {config_json}")
    
    print(f"Loading model from: {model_path}")
    # Reduce GPU memory utilization to work with limited available memory
    # Default is 0.9 (90%), but we need to use less when other processes are using GPU
    llm = LLM(model=model_path, gpu_memory_utilization=0.12)
    outputs = llm.generate(prompts, sampling_params)

    records = []

    for output in outputs:
        records.append({
            "prompt": output.prompt,
            "completion": output.outputs[0].text.replace(" ", ""),
            "full":  output.prompt + output.outputs[0].text.replace(" ", ""),
            "length": len(output.outputs[0].text),
        })

    df = pd.DataFrame(records)
    print(f"Number of records: {len(df)}")
    df = df.drop_duplicates(subset="full")
    print(f"Number of unique records: {len(df)}")
    process_outputs(df)
    return df


if __name__ == "__main__":
    df = main()
    print(df.head())