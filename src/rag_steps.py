import logging
from groq import Client
import requests
import os
from weaviate.classes.init import Auth
import pypdf
import docx
from clients import get_weaviate_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from src.utilites import load_yaml_config
load_dotenv()

# Load SiliconFlow API key
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_EMBEDDING_URL = os.getenv("SILICONFLOW_EMBEDDING_URL")

# load config from yaml
config = load_yaml_config("config.yaml")

try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("rags_steps")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("rag_steps")


# ─── Utility: Extract raw text ──────────────────────────────────────────────────
def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
    return text



# ─── Chunker & Embed ──────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunking"]["chunk_size"],
    chunk_overlap=config["chunking"]["chunk_size"],
    separators=config["chunking"]["chunk_size"],)


def embed_texts(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    all_embeddings = []
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        logger.info(f"Embedding Model: {config['apis_models']['silicon_flow']['qwen']['embed']}")
        payload = {
            "model": config["apis_models"]["silicon_flow"]["qwen"]["embed"],
            "input": batch}
        response = requests.post(
            SILICONFLOW_EMBEDDING_URL, json=payload, headers=headers)
        # response.raise_for_status()
        data = response.json()
        if "data" not in data:
            raise ValueError(f"Invalid response format: {data}")
        batch_embs = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embs)
        # Fill with empty embeddings in case of failure
        all_embeddings.extend([[] for _ in batch])
    return all_embeddings





# ─── Ingest & Index ───────────────────────────────────────────────────────────
def ingest_file(file_path: str) -> str:
    raw = extract_text(file_path)
    docs = splitter.split_text(raw)
    texts = [chunk for chunk in docs]
    vectors = embed_texts(texts)
    client = get_weaviate_client()
    if client is None:
        logger.info("Weaviate client not connected. Please check your WEAVIATE_URL and WEAVIATE_API_KEY.")
    else:
        logger.info("Weaviate client connected (startup checks skipped).")
    documents = client.collections.get(config["rag"]["weavaite_collection"])
    with client.batch.dynamic() as batch:
        for txt, vec in zip(texts, vectors):
            batch.add_object(
                collection=documents,
                properties={"text": txt},
                vector=vec)
    return f"Ingested {len(texts)} chunks from {os.path.basename(file_path)}"

# client.close()
if __name__=="__main__":
    logger.info(f"Test Pdf: ")
    pdf_text = extract_text("tests/Computational Requirements for Embed.pdf")
    logger.info(f"Extracted text from Pdf: {pdf_text}")

    logger.info(f"Test txt: ")
    txt_text = extract_text("assets/RAG_Documents/Autism_Books_1.txt")
    logger.info(f"Extracted text from Txt file: {pdf_text}")

    logger.info(f"Test docs: ")
    docs_text = extract_text("tests/Computational Requirements for Embed.docx")
    logger.info(f"Extracted text from Docs: {pdf_text}")




