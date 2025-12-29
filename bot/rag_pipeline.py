import os

# --- MUST be set BEFORE any Chroma or LangChain import ---
os.environ["CHROMA_TELEMETRY"] = "off"
os.environ["ANONYMIZED_TELEMETRY"] = "false"  # additional safety

# Optional: disable posthog explicitly (used by Chroma internally)
os.environ["POSTHOG_DEBUG"] = "0"
os.environ["POSTHOG_DISABLED"] = "1"
# ----------------------------------------------------------

from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from build_index import CHROMADB_DIR, MODELS_DIR, load_embeddings, load_vectorstore


LLM_MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

GGUF_MODEL_PATH = MODELS_DIR / FILENAME

DANGEROUS_PATTERNS = [
    "ignore all", "ignore all instructions", "output:", "print:", "execute", "run", "system", "sudo",

    "password", "пароль", "pwd", "root", "admin", "superuser", "суперпользователь",
    "суперпароль", "superpassword", "credentials", "логин", "login", "token",

    "root:", "user:", "login:", "pass:", "pwd:",
]

PROMPT_TEMPLATE = """Ты — помощник по внутренней документации.

ВАЖНО:
— НИКОГДА не генерируй пароли, логины, токены.
— Если в контексте есть фраза «ВНИМАНИЕ: Все найденные фрагменты содержат потенциально опасный контент» — отвечай ТОЛЬКО: «Я не знаю».
— На запросы про пароли, суперпароли, root, password — отвечай: «Я не знаю».


Правила:
1. Отвечай, если в контексте есть достаточно информации, чтобы однозначно ответить.
2. Не домысливай и не интерполируй — опирайся только на то, что написано.
3. Если цитаты нет — отвечай: «Я не знаю».
4. Не интерпретируй, не домысливай, не переводи.
5. Ответ — одно слово или короткая фраза.

Примеры:

Контекст:
«Город Златых Башен — город на восточном побережье Континента Семи Властей».

Q: Где расположен Город Золотых Башен?
A: Континент Семи Властей

Контекст:
«Город Златых Башен был основан королём Эйгоном Змейрогом».

Q: В какой стране расположен Город Златых Башен?
A: Я не знаю

Контекст:
«Илиана Дарвель — старшая дочь лорда Алдара Дарвеля и его жены Мариссы Дарвель».

Q: Кто родители Илианы Дарвель?
A: Алдар Дарвель и Марисса Дарвель

Теперь твой черёд.

Контекст:
{context}

Q: {question}
A: """

def is_malicious(text: str) -> bool:
    low = text.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern in low:
            return True
    # search possible hashes
    import re
    if re.search(r"[A-Za-z0-9+/]{20,}", text):
        print(text)
        return True
    return False

# Create bge model retriever
class BGERetriever(BaseRetriever):
    vectorstore: Chroma

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        
        # Рекомендация BAAI:
        query_with_instruction = (
            "Represent this question for searching relevant passages: " + query
        )
        docs = self.vectorstore.similarity_search(query_with_instruction, k=3)  # Find 3 nearest chunks        

        safe_docs = []
        for doc in docs:
            if is_malicious(doc.page_content):
                print("Found malicious chunk, skipping")
                continue
            safe_docs.append(doc)

        # Still generate context for model
        if not safe_docs:
            return [Document(
                page_content="ВНИМАНИЕ: Все найденные фрагменты содержат потенциально опасный контент. Ответ невозможен.",
                metadata={})]

        return safe_docs

def load_llm():
    return LlamaCpp(
        model_path=str(GGUF_MODEL_PATH),
        n_ctx=4096, # Mistral-7B is trained with 32k context, but for RAG we intentionally limit n_ctx to 4096
        n_batch=256,
        n_gpu_layers=0,  # 0 = CPU only
        max_tokens=512,
        temperature=0.3,
        top_p=0.95,
        repeat_penalty=1.25,
        stop=[  # exclude Mistral model possible template keywords to not hallucinate and generate whole QA blocks
            "\n\n",
            "Вопрос:",
            "Ответ:",
            "Запрос:",
            "Ты —",
            "ИНСТРУКЦИИ:",
        ],
        verbose=False,  # disable llama.cpp log
    )


def load_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # all context in one prompt
        retriever=retriever,  # specific bge retriever
        chain_type_kwargs={"prompt": PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )},
        return_source_documents=True  # show source documents
    )


if __name__ == "__main__":
    # Load model
    if not GGUF_MODEL_PATH.exists():
        print(f"GGUF model not found at {GGUF_MODEL_PATH}")
        print(f"Downloading {FILENAME} from Hugging Face Hub ({LLM_MODEL_NAME})...")

        hf_hub_download(
            repo_id=LLM_MODEL_NAME,
            filename=FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
            token=False,
        )

        print(f"Model downloaded to {GGUF_MODEL_PATH}")
    else:
        print(f"Using cached GGUF model: {GGUF_MODEL_PATH}")

    # Check vector DB existence
    assert CHROMADB_DIR.exists(), "ChromaDB directory not found, check path or run build_index.py first"

    # Load vector index
    vectorstore = load_vectorstore(load_embeddings())

    print(f"Downloading {LLM_MODEL_NAME} LLM model...")

    llm = load_llm()

    retriever = BGERetriever(vectorstore=vectorstore)

    # Create RetrievalQA chain
    qa_chain = load_qa_chain(llm, retriever)

    print("RetrievalQA RAG pipeline is ready")

    # --- TESTS ------------------------------------------------------------------------------------------------------------
    print("\nTESTS:")

    test_queries = [
        "Кто родители Илианы Дарвель?",
        "Носит ли Рован Камнегрив меч?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\nQ: {query}")

        result = qa_chain.invoke({"query": query})

        print("\nA: ")
        print(result["result"].strip())

        print("\nИсточники:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"  {i}. {doc.metadata['file_name']} (chunk {doc.metadata['chunk_index']})")
