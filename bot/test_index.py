# test_index.py
import os

# --- MUST be set BEFORE any Chroma or LangChain import ---
os.environ["CHROMA_TELEMETRY"] = "off"
os.environ["ANONYMIZED_TELEMETRY"] = "false"  # additional safety
os.environ["CHROMA_TELEMETRY_SETTINGS"] = "api_key"

# Optional: disable posthog explicitly (used by Chroma internally)
os.environ["POSTHOG_DEBUG"] = "0"
os.environ["POSTHOG_DISABLED"] = "1"
# ----------------------------------------------------------

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
CHROMADB_DIR = PROJECT_ROOT / "chromadb"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "yandex_rag_bot"

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
    if not (CHROMADB_DIR / "chroma.sqlite3").exists():
        print(f"ERROR: Vector index not found in '{CHROMADB_DIR}'.")
        print("Please run 'build_index.py' first to create the index.")
        return

    print("Loading embedding model...")
    embeddings = load_embeddings()
    print("Embedding model loaded.")

    print("Loading ChromaDB vector index...")
    vectorstore = Chroma(
        persist_directory=str(CHROMADB_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Index loaded successfully.")

    # Test queries
    test_queries = [
        "Кем приходится Валери Камнегрив Морвену Камнегриву?",
        "Какой секрет хранит Рован Камнегрив?",
        "Где находится Город Златых Башен?",
    ]

    print("\n Running test queries:")

    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: '{query}'")

        results = retriever.invoke(query) # no prefix for bge-m3
        for j, doc in enumerate(results):
            print(f"-- Chunk {j+1}: {doc.metadata['file_name']} (chunk position {doc.metadata['chunk_index']}/{doc.metadata['total_chunks']})")
            preview = doc.page_content[:200].replace("\n", " ") # print 200 symbols from chunk
            print(f"   Chunk preview: \"{preview}...\"")
        print()    
    
if __name__ == "__main__":
    main()