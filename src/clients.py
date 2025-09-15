from dotenv import load_dotenv
import google.genai as genai
from utilites import load_yaml_config
from groq import Groq
from openai import OpenAI
from datetime import datetime
import requests
import logger
import os
import sys

import logging

# Import logger here to avoid circular imports
try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("clients")
except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("clients")

# Load environment variables
load_dotenv()

config = load_yaml_config("rag_config.yaml")



## Groq
GROQ_URL = os.environ["GROQ_URL"]
GROQ_API_TOKEN= os.environ["GROQ_API_TOKEN"]


## Deepinfra
DEEPINFRA_API_KEY = os.environ["DEEPINFRA_API_KEY"]
DEEPINFRA_URL = os.environ["DEEPINFRA_URL"]
DEEPINFRA_EMBEDDING_URL = os.environ["DEEPINFRA_EMBEDDING_URL"]
DEEPINFRA_RERANK_URL = os.environ["DEEPINFRA_RERANK_URL"]



# def qwen_generate_content(prompt: str) -> str:
#     """Streaming chat completion with Qwen on SiliconFlow via OpenAI client."""
#     if not (os.environ['SILICONFLOW_URL'] and os.environ['SILICONFLOW_API_KEY']):
#         logger.error("SILICONFLOW_URL or SILICONFLOW_API_KEY not configured.")
#         return ""
        
#     client = OpenAI(base_url=os.environ['SILICONFLOW_URL'], api_key=os.environ['SILICONFLOW_API_KEY'])
#     logger.info("Calling Qwen/Qwen3-30B-Instruct for generation...")
#     output = ""
#     logger.info(f"{config['apis_models']['silicon_flow']['qwen']['chat3_30b']}")
#     response = client.chat.completions.create(
#         model=config["apis_models"]["silicon_flow"]["qwen"]["chat3_30b"],
#         messages=[{"role": "user", "content": prompt}],
#         stream=True)

#     for chunk in response:
#         if not getattr(chunk, "choices", None):
#             continue
#         delta = chunk.choices[0].delta
#         if getattr(delta, "content", None):
#             output += delta.content

#         # if hasattr(delta, "reasoning_content") and delta.reasoning_content:
#         #     output += delta.reasoning_content

#     logger.info("Successfully generated content with Qwen")
#     return output.strip()



def groq_qwen_generate_content(prompt: str) -> str:
    """Streaming chat completion with Qwen on SiliconFlow via OpenAI client."""
    if not (GROQ_URL and GROQ_API_TOKEN):
        logger.error("GROQ_URL or GROQ_API_TOKEN not configured.")
        return ""
        
    client = OpenAI(base_url=GROQ_URL, api_key=GROQ_API_TOKEN)
    if client is None:
        logger.error("Failed to initialize Groq client.")
        return ""
    else:
        logger.info("Successfully initialized Groq client.")
    # logger.info("Calling Qwen/Qwen3-32B for generation from Groq...")
    logger.info("Calling openai/gpt-oss-120b for generation from Groq")

    output = ""
    response = client.chat.completions.create(
        # model=config["apis_models"]["groq"]["qwen"]["chat3_32b"],
        model = config["apis_models"]["groq"]["openai"]["gpt_oss"],
        messages=[{"role": "user", "content": prompt}],
        stream=True,)
        # reasoning_effort="none")
    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            output += delta.content
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            output += delta.reasoning_content
    logger.info("Successfully generated content with Qwen")
    return output.strip()




def siliconflow_qwen_generate_content(prompt: str) -> str:
    """Streaming chat completion with Qwen on SiliconFlow via OpenAI client."""
    if not (os.environ['SILICONFLOW_URL'] and os.environ['SILICONFLOW_API_KEY']):
        logger.error("SILICONFLOW_URL or SILICONFLOW_API_KEY not configured.")
        return ""
    client = OpenAI(base_url=os.environ['SILICONFLOW_URL'], api_key=os.environ['SILICONFLOW_API_KEY'])
    if client is None:
        logger.error("Failed to initialize SiliconFlow client.")
        return ""
    else:
        logger.info("Successfully initialized SiliconFlow client.")
    logger.info("Calling Qwen/Qwen3-30B-Instruct for generation...")
    output = ""
    logger.info(f"{config['apis_models']['silicon_flow']['qwen']['chat3_30b']}")
    response = client.chat.completions.create(
        model=config["apis_models"]["silicon_flow"]["qwen"]["chat3_30b"],
        messages=[{"role": "user", "content": prompt}],
        stream=True)
    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            output += delta.content
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            output += delta.reasoning_content
    logger.info("Successfully generated content with Qwen")
    return output.strip()



def deepinfra_qwen_generate_content(prompt: str) -> str:
    """Streaming chat completion with Qwen on SiliconFlow via OpenAI client."""
    if not (DEEPINFRA_URL and DEEPINFRA_API_KEY):
        logger.error("GROQ_URL or GROQ_API_TOKEN not configured.")
        return ""
        
    client = OpenAI(base_url=DEEPINFRA_URL, api_key=DEEPINFRA_API_KEY)
    if client is None:
        logger.error("Failed to initialize Groq client.")
        return ""
    else:
        logger.info("Successfully initialized Groq client.")
    # logger.info("Calling Qwen/Qwen3-32B for generation from DeepInfra...")
    logger.info("Calling openai gpt-oss-120b for generation from DeepInfra...")
    output = ""
    response = client.chat.completions.create(
        # model=config["apis_models"]["groq"]["qwen"]["chat3_32b"],
        model = config["apis_models"]["groq"]["openai"]["gpt_oss"],
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="low",
        stream=True,
        tools=[{"type":"browser_search"}])

        # reasoning_effort="none")
    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            output += delta.content
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            output += delta.reasoning_content
    logger.info("Successfully generated content with Qwen")
    return output.strip()



def deepinfra_embedding(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    all_embeddings = []
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": config["apis_models"]["deepinfra"]["qwen"]["embed"],
            "input": batch}
        try:
            response = requests.post(
                DEEPINFRA_EMBEDDING_URL, json=payload, headers=headers)
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"DeepInfra API error {response.status_code}: {response.text}")
                # Return empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])
                continue
            data = response.json()
            # Check for API error in response
            if "detail" in data and "error" in data["detail"]:
                logger.error(f"DeepInfra API error: {data['detail']['error']}")
                # Return empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])
                continue
            if "data" not in data:
                logger.error(f"Invalid response format: {data}")
                # Return empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])
                continue
            batch_embs = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embs)
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            # Return empty embeddings for failed batch
            all_embeddings.extend([[] for _ in batch])
    return all_embeddings



def deepinfra_rerank(batch: list[str], items_to_rerank: list[str]) -> list[str]:
    payload = {
        "model": config["apis_models"]["deepinfra"]["qwen"]["rerank"],
        "input": batch}
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"}
    r = requests.post(
        DEEPINFRA_RERANK_URL,
        json=payload,
        headers=headers,
        timeout=60,)
    if r.ok:
        rerank_data = r.json()
        ranked_docs = sorted(
            zip(rerank_data.get("results", []), items_to_rerank),
            key=lambda x: x[0].get("relevance_score", 0),
            reverse=True)
        reranked = ranked_docs[0][1] if ranked_docs else batch
        return reranked
    else:
        return batch

def deepinfra_client():
   return OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url=os.environ["DEEPINFRA_URL"],)


def qwen_generate(prompt: str) -> str:
    """Streaming chat completion with Qwen on SiliconFlow and Groq via OpenAI client."""
    if config["apis_models"]["num"] == 1:
        return siliconflow_qwen_generate_content(prompt)
    else:
        return groq_qwen_generate_content(prompt)



if __name__ == "__main__":
    # client = init_weaviate_client()
    # if client is None:
    #     logger.error(f"api of weaviate is not working")
    # client.close()
    
    gen_prompt = Prompt_template_LLM_Generation.format(
        new_query="what is autism")
    logger.info(f"groq qwen generate.....: {groq_qwen_generate_content(gen_prompt)}")
    
    print(f"=" * 50)
    response = siliconflow_qwen_generate_content("what is autism")
    logger.info(f"siliconflow qwen response: {response}")

    print(f"=" * 50)
    
    # Try DeepInfra first, then fallback to Groq
    response = deepinfra_embedding(["what is autism"], 1)
    if response and response[0]:  # Check if we got valid embeddings
        logger.info(f"deepinfra embedding response: {response}")
    else:
        raise ValueError("Empty embeddings returned")


    print(f"=" * 50)
    response = deepinfra_rerank(["what is autism"], ["what is autism"])
    logger.info(f"deepinfra rerank response: {response}")