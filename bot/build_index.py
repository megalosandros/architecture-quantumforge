# build_index.py
import os

# --- MUST be set BEFORE any Chroma or LangChain import ---
os.environ["CHROMA_TELEMETRY"] = "off"
os.environ["ANONYMIZED_TELEMETRY"] = "false"  # additional safety

# Optional: disable posthog explicitly (used by Chroma internally)
os.environ["POSTHOG_DEBUG"] = "0"
os.environ["POSTHOG_DISABLED"] = "1"
# ----------------------------------------------------------


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path
import time
import uuid


PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
CHROMADB_DIR = PROJECT_ROOT / "chromadb"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

assert KNOWLEDGE_BASE_DIR.exists(), f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 800    # BGE рекомендует до 512 токенов, но ~800 символов ≈ 512 токенов для англ.
CHUNK_OVERLAP = 80
COLLECTION_NAME = "yandex_rag_bot"

def split_doc(doc: str):
    source = None
    title = None
    lines = doc.splitlines()
    body_start = 0

    for i, line in enumerate(lines[:5]):
        if line.startswith("# Source:"):
            source = line.replace("# Source:", "").strip()
        elif line.startswith("# Title:"):
            title = line.replace("# Title:", "").strip()
        else:
            body_start = i
            break
    else:
        body_start = len(lines)

    body = "\n".join(lines[body_start:]).strip()
    return source, title, body

def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 2,
        },
    )

def main():
    # Check if index already exists
    if (CHROMADB_DIR / "chroma.sqlite3").exists():
        print(f"ERROR: Vector index already exists in '{CHROMADB_DIR}'.")
        print("To rebuild, delete the 'chromadb' folder manually and run this script again.")
        exit(1)

    print(f"Loading knowledge base from: {KNOWLEDGE_BASE_DIR}")
    txt_files = list(KNOWLEDGE_BASE_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {KNOWLEDGE_BASE_DIR}")

    print(f"Found {len(txt_files)} document(s)")

    docs = []
    for file_path in txt_files:
        try:
            raw_text = file_path.read_text(encoding="utf-8")
            source_url, doc_title, body = split_doc(raw_text)
            doc = {
                "page_context": body,
                "metadata": {
                    "source": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "file_stem": file_path.stem,
                    "source_url": source_url or "unknown",
                    "document_title": doc_title or file_path.stem.replace("_", " "),
                }
            }
            docs.append(doc)
        except Exception as e:
            print(f"Error loading '{file_path.name}': {e}")

    print(f"Loaded {len(docs)} document(s)")

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = []
    for doc in docs:
        texts = splitter.split_text(doc["page_context"])
        for i, text in enumerate(texts):
            chunks.append(Document(
                page_content=text.strip(),
                metadata={
                    **doc["metadata"],
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "total_chunks": len(texts),
                }
            ))

    print(f"Generated {len(chunks)} chunks")

    # Model loading: Hugging Face uses cache automatically
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (using cache if available)...")

    start_load = time.time()
    embeddings = load_embeddings()
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")

    print("Creating ChromaDB vector index...")
    start_index = time.time()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMADB_DIR),
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist() #   Since Chroma 0.4.x the manual persistence method is no longer supported

    index_time = time.time() - start_index
    print(f"Vector index saved to '{CHROMADB_DIR}' in {index_time:.2f} seconds")
    print("Indexing completed successfully.")

if __name__ == "__main__":
    main()