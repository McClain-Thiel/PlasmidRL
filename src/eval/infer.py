import boto3
import ray
from vllm import LLM, SamplingParams
import pandas as pd
import datetime

from src.config import get_config


config = get_config()

def main():
        

    ds = ray.data.read_parquet("s3://anonymous@air-example-data/prompts.txt")

    gfp = ray.data.read_text("s3://phd-research-storage-1758274488/prompts/GFP_cassette.fasta")
    gfp_cassette = "".join([x['text'].upper() for x in ds.iter_rows()][1:])

    prompts = [
        gfp_cassette,
        "ATG"
    ] * 50

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

    llm = LLM(model=config.model, tokenizer=config.tokenizer)

    results = llm.generate(prompts, sampling_params)

    records = []
    for output in results:
        records.append({
            "prompt": output.prompt,
            "response": output.outputs[0].text,
            "full": output.prompt + output.outputs[0].text,
            "length": len(output.prompt + output.outputs[0].text),
        })

    df = pd.DataFrame(records)
    id = f"{config.model}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    df.to_parquet(config.storage_bucket + "infer.parquet")


if __name__ == "__main__":
    main()