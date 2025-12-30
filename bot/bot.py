import sys
import os

from build_index import load_embeddings, load_vectorstore
from rag_pipeline import BGERetriever, load_llm, load_qa_chain, DANGEROUS_PATTERNS

CONTENT_LENGTH = 128

BANNER_TEXT = (
    "Привет! Я — интеллектуальный RAG-бот, "
    "помощник по внутренней документации "
    "(выход: 'quit' / 'exit')."
)

BANNER_LINE = "=" * len(BANNER_TEXT)


def extract_final_answer(text: str) -> str:
    lines = text.strip().split("\n")
    for line in lines:
        if line.startswith("Итог:"):
            return line[len("Итог:"):].strip()
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("1.", "2.", "3.", "A:", "Q:", "→")):
            return stripped
    return "Я не знаю."

def is_dangerous_query(query: str) -> bool:
    low = query.lower()
    return any(kw in low for kw in DANGEROUS_PATTERNS)

def configure_stdin():
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        # Python < 3.7 или альтернативные окружения
        pass

def clear_stdin():
    if not sys.stdin.isatty():
        return

    if os.name == "posix":
        # Linux / macOS
        try:
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass

    elif os.name == "nt":
        # Windows
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        except Exception:
            pass


def run_console_bot():
    configure_stdin()
    print(f"\n{BANNER_LINE}")
    print(BANNER_TEXT)
    print(f"{BANNER_LINE}\n")
    
    qa_chain = load_qa_chain(load_llm(), BGERetriever(vectorstore=load_vectorstore(load_embeddings())))

    while True:
        try:
            clear_stdin()
            query = input("Q: ").strip()
            if not query:
                continue

            if query.lower() in {"quit", "exit"}:
                print("Выход...")
                break

            # do not call model for dangerous queries
            if is_dangerous_query(query):
                print("\nA: Я не знаю.")
                continue

            # Run query
            result = qa_chain.invoke({"query": query})

            # Get answer
            raw_answer = result["result"].strip()
            final_answer = extract_final_answer(raw_answer)

            print(f"\nA: {final_answer}\n")

            # Optional: show source documents
            clear_stdin()
            show_sources = input("Показать источники? (y/N): ").strip().lower()
            if show_sources in {"y", "yes"}:
                print("\nИсточники:")
                for i, doc in enumerate(result["source_documents"], 1):
                    content_preview = doc.page_content[:CONTENT_LENGTH].replace("\n", " ").strip()
                    if len(doc.page_content) > CONTENT_LENGTH:
                        content_preview += "…"
                    print(f"  {i}. {doc.metadata.get('file_name', 'unknown')} "
                          f"(chunk {doc.metadata.get('chunk_index', '?')})")
                    print(f" → {content_preview}")
                print()

        except KeyboardInterrupt:
            print("\n\nВыход...")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")
            import traceback
            traceback.print_exc()

# Запуск REPL после инициализации
if __name__ == "__main__":
    run_console_bot()