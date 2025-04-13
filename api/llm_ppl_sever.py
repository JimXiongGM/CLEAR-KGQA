import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)


class SentenceInput(BaseModel):
    sentence: str


def load_model_and_tokenizer(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def calculate_perplexity_sync(sentence, model, tokenizer):
    device = next(model.parameters()).device

    if "-Instruct" in model.config._name_or_path:
        # Convert to messages format first
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": sentence},
        ]
        # Apply chat template
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        chat_text = sentence

    inputs = tokenizer(chat_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    num_tokens = input_ids.size(1)

    start_time = time.time()

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    end_time = time.time()
    processing_time = end_time - start_time
    tokens_per_second = num_tokens / processing_time

    logger.info(
        f"Get {num_tokens} tokens in {processing_time:.2f} seconds, average speed is {tokens_per_second:.2f} tokens/s"
    )

    return torch.exp(loss).item()


@app.post("/perplexity")
async def calculate_perplexity(input_data: SentenceInput):
    try:
        loop = asyncio.get_event_loop()
        perplexity = await loop.run_in_executor(
            executor, calculate_perplexity_sync, input_data.sentence, model, tokenizer
        )
        return {"perplexity": perplexity}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a FastAPI server for perplexity calculation.")
    parser.add_argument("--model", type=str, required=True, help="Name of the language model to use.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on.")
    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)

    if torch.cuda.is_available():
        model = model.cuda()

    # 不使用 workers 参数
    uvicorn.run(app, host="0.0.0.0", port=args.port)

"""
Test:
CUDA_VISIBLE_DEVICES=0 python api/llm_ppl_sever.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 26000

curl -X POST "https://iqftajrymhae4hd9snow.deepln.com/perplexity" -H "Content-Type: application/json" -d '{"sentence": "The quick brown fox jumps over the lazy dog."}'
"""
