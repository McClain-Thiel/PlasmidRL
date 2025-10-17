from vllm import LLM, SamplingParams
from src.config import get_config
import datetime
import boto3
import io
import pandas as pd
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
    prompts = [config.default_query, "ATG"] * 100 #strong prompt and weak prompt

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.8,
        top_p=0.95,
        top_k=0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    llm = LLM(model=config.sample_model)
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