import asyncio
import os
from pathlib import Path

from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

from rag.indexer import build_index
from rag.retriever import retrieve

# ──────────────────────────────────────────────
# Configuración inicial
# ──────────────────────────────────────────────
load_dotenv()

client = AsyncOpenAI(
    base_url=os.environ["OLLAMA_BASE_URL"],
    api_key=os.environ["OLLAMA_API_KEY"],
)
set_default_openai_client(client)
set_tracing_disabled(True)

MODEL = os.environ["OLLAMA_MODEL"]

# Ruta al PDF del libro
PDF_PATH = Path(__file__).parent / "docs" / "Bachelard_Gaston_El_agua_y_los_suenos.pdf"

# ──────────────────────────────────────────────
# Definición del agente RAG
# ──────────────────────────────────────────────
agent = Agent(
    name="Bachelard RAG Agent",
    instructions=(
        "Eres un asistente experto en el libro 'El agua y los sueños' de Gastón Bachelard. "
        "Responde ÚNICAMENTE basándote en los fragmentos del libro que se te proporcionan como contexto. "
        "Si el contexto no contiene información suficiente para responder, dilo claramente. "
        "Cita o parafrasea el libro cuando sea pertinente. "
        "Responde siempre en español con un lenguaje claro y reflexivo."
    ),
    model=MODEL,
)


async def answer_question(question: str, collection) -> str:
    """
    Orquesta el pipeline RAG completo:
      1. Recupera los fragmentos más relevantes del libro.
      2. Los inyecta como contexto en el prompt del agente.
      3. Devuelve la respuesta del LLM.
    """
    chunks = await retrieve(question, collection, client, MODEL)

    context_blocks = "\n\n---\n\n".join(
        f"[Fragmento {i+1}]\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )

    prompt = (
        f"Pregunta: {question}\n\n"
        f"Contexto extraído del libro:\n\n{context_blocks}"
    )

    result = await Runner.run(agent, input=prompt)
    return result.final_output


async def main() -> None:
    # Construir (o cargar) el índice vectorial del libro
    collection = build_index(PDF_PATH)

    print("\n" + "═" * 60)
    print("  Agente RAG — El agua y los sueños · Gastón Bachelard")
    print("  Escribe tu pregunta o 'salir' para terminar.")
    print("═" * 60 + "\n")

    while True:
        try:
            question = input("Tu pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        print("\nProcesando...\n")
        answer = await answer_question(question, collection)
        print(f"Respuesta:\n{answer}\n")
        print("─" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
