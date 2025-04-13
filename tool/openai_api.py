import json
import os
import pickle
from typing import List

import httpx
import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

_real_k = os.environ.get("OPENAI_API_KEY", None)
assert _real_k is not None, "OPENAI_API_KEY is not set."
_http_client = httpx.Client(proxy="http://127.0.0.1:7890")
logger.warning("KBQA_API_IP not set. It is the server of APIs.")
DEFAULT_CLIENT = openai.OpenAI(api_key=_real_k, http_client=_http_client, timeout=30)

# You can setup the Redis server.
# import redis
# redis_client_emb = redis.StrictRedis(host="localhost", port=16379, db=1)

redis_client_emb = None


OPENAI_EMBEDDING_MODELS = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]


@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def chatgpt(
    prompt="Hello!",
    system_content="You are an AI assistant.",
    messages=None,
    model=None,
    temperature=0,
    top_p=1,
    n=1,
    stop=None,  # ["\n"],
    max_completion_tokens=512,
    presence_penalty=0,
    frequency_penalty=0,
    seed=42,
    logit_bias={},
    logprobs=False,
    url=None,
    key=None,
    response_format=None,  # usage: response_format={"type": "json_object"}
) -> dict:
    """
    role:
        The role of the author of this message. One of `system`, `user`, or `assistant`.
    temperature:
        What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        We generally recommend altering this or `top_p` but not both.
    top_p:
        An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both.

    messages as history usage:
        history = [{"role": "system", "content": "You are an AI assistant."}]

        inp = "Hello!"
        history.append({"role": "user", "content": inp})
        response = chatgpt(messages=history)
        out = response["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": out})
        print(json.dumps(history,ensure_ascii=False,indent=4))
    """
    assert model is not None, "model name is None"

    messages = (
        messages
        if messages
        else [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
    )

    if "gpt-" in model or model.startswith("o1-") or "claude" in model:
        client = DEFAULT_CLIENT
    else:
        assert url is not None, f"You must provide a url for non-OpenAI model: {model}"
        client = openai.OpenAI(api_key=key, base_url=url, timeout=30)

    if model.startswith("o1-"):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            n=n,
            max_completion_tokens=max_completion_tokens,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            max_tokens=max_completion_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            response_format=response_format,
            # seed=seed,
        )
    # content = response["choices"][0]["message"]["content"]
    response = json.loads(response.model_dump_json())
    return response


@retry(wait=wait_fixed(2), stop=stop_after_attempt(10))
def get_embedding_batch(
    texts: List[str],
    model="text-embedding-ada-002",
) -> list[float]:
    if redis_client_emb is None:
        req = DEFAULT_CLIENT.embeddings.create(input=texts, model=model)
        return [i.embedding for i in req.data]

    unseen_texts: List[str] = []
    for text in texts:
        cache = redis_client_emb.get(model + text)
        if cache is None:
            unseen_texts.append(text)

    if unseen_texts:
        req = DEFAULT_CLIENT.embeddings.create(input=unseen_texts, model=model)
        vec_batch = [i.embedding for i in req.data]
        assert len(vec_batch) == len(unseen_texts)
        for unseen_text, vec in zip(unseen_texts, vec_batch):
            redis_client_emb.set(model + unseen_text, pickle.dumps(vec))

    res = [pickle.loads(redis_client_emb.get(model + text)) for text in texts]
    assert None not in res
    return res


def get_embedding(
    text: str,
    model="text-embedding-ada-002",
) -> list[float]:
    return get_embedding_batch([text], model)[0]
