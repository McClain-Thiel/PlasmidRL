import httpx
from typing import Dict

async def get_prodigal_output(seq: str) -> Dict:
    return {"output": "test"}

async def get_amrfinder_output(seq: str) -> Dict:
    return {"output": "test"}

def plannotate_output(seq: str) -> Dict:
    return {"output": "test"}
