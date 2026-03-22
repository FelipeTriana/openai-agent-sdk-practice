"""
retriever.py — Pipeline de recuperación con Query Expansion + Re-ranking (Cross-encoder).

Flujo completo:
  1. Query Expansion: el LLM genera variaciones de la pregunta original.
  2. Bi-encoder retrieval: ChromaDB busca los fragmentos más similares para cada variación.
  3. Deduplicación: elimina duplicados entre resultados de las distintas queries.
  4. Cross-encoder re-ranking: puntúa cada fragmento contra la pregunta original y reordena.
  5. Devuelve los top-K fragmentos más relevantes como contexto para el LLM.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

# Modelos
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Parámetros de recuperación
N_EXPANDED_QUERIES = 4     # Cuántas variaciones de la pregunta genera el LLM
N_CANDIDATES_PER_QUERY = 10 # Fragmentos que recupera ChromaDB por cada variación
TOP_K_FINAL = 5             # Fragmentos finales que llegan al LLM tras el re-ranking


@dataclass
class RetrievedChunk:
    text: str
    pages: str
    score: float  # Puntuación del cross-encoder (más alto = más relevante)


# ──────────────────────────────────────────────────────────────
# Carga de modelos (se hace una sola vez al importar el módulo)
# ──────────────────────────────────────────────────────────────
print("[Retriever] Cargando modelos de lenguaje (solo la primera vez tarda)...")
_bi_encoder = SentenceTransformer(EMBED_MODEL)
_cross_encoder = CrossEncoder(RERANK_MODEL)
print("[Retriever] Modelos listos.")


async def _expand_query(question: str, openai_client) -> list[str]:
    """
    Usa el LLM para generar N_EXPANDED_QUERIES variaciones de la pregunta original.
    Cada variación busca el mismo significado con distintas palabras,
    aumentando la probabilidad de encontrar fragmentos relevantes en el índice.
    """
    prompt = (
        f"Eres un asistente especializado en el libro 'El agua y los sueños' de Gastón Bachelard.\n"
        f"Genera exactamente {N_EXPANDED_QUERIES} reformulaciones diferentes de la siguiente pregunta, "
        f"cada una en una línea separada, sin numeración ni viñetas. "
        f"Usa sinónimos, cambia el orden de las ideas o enfoca distintos aspectos.\n\n"
        f"Pregunta original: {question}"
    )

    response = await openai_client.chat.completions.create(
        model=None,  # Se sobreescribe en agent.py con set_default_openai_client
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    expansions = [line.strip() for line in raw.splitlines() if line.strip()]
    # Incluimos siempre la pregunta original junto con las expansiones
    return [question] + expansions[:N_EXPANDED_QUERIES]


def _retrieve_candidates(
    queries: list[str],
    collection: chromadb.Collection,
) -> list[str]:
    """
    Para cada query, busca los N_CANDIDATES_PER_QUERY fragmentos más similares en ChromaDB.
    Devuelve una lista deduplicada de textos candidatos.
    """
    query_embeddings = _bi_encoder.encode(queries).tolist()
    seen_ids: set[str] = set()
    candidates: list[str] = []

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=N_CANDIDATES_PER_QUERY,
        include=["documents"],
    )

    for doc_list, id_list in zip(results["documents"], results["ids"]):
        for doc, doc_id in zip(doc_list, id_list):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                candidates.append(doc)

    return candidates


def _rerank(question: str, candidates: list[str]) -> list[RetrievedChunk]:
    """
    El cross-encoder lee la pregunta y cada candidato juntos para asignar
    una puntuación de relevancia más precisa. Devuelve los TOP_K_FINAL mejores.
    """
    pairs = [[question, candidate] for candidate in candidates]
    scores = _cross_encoder.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        RetrievedChunk(text=text, pages="", score=float(score))
        for score, text in ranked[:TOP_K_FINAL]
    ]


async def retrieve(
    question: str,
    collection: chromadb.Collection,
    openai_client,
    model: str,
) -> list[RetrievedChunk]:
    """
    Pipeline completo: query expansion → retrieval → re-ranking.
    Devuelve los TOP_K_FINAL fragmentos más relevantes.
    """
    print(f"\n[Retriever] Expandiendo query: '{question}'")

    # Parcheamos el modelo en el cliente para la llamada de expansión
    original_model = None
    try:
        # La expansión usa el mismo cliente Ollama configurado en agent.py
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Genera exactamente {N_EXPANDED_QUERIES} reformulaciones diferentes "
                        f"de la siguiente pregunta sobre el libro 'El agua y los sueños' de "
                        f"Gastón Bachelard. Cada reformulación en una línea, sin numeración.\n\n"
                        f"Pregunta: {question}"
                    ),
                }
            ],
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        expansions = [line.strip() for line in raw.splitlines() if line.strip()]
        queries = [question] + expansions[:N_EXPANDED_QUERIES]
        print(f"[Retriever] Queries generadas: {len(queries)}")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
    except Exception as e:
        print(f"[Retriever] Error en query expansion: {e}. Usando solo la query original.")
        queries = [question]

    print(f"[Retriever] Buscando candidatos en el índice...")
    candidates = _retrieve_candidates(queries, collection)
    print(f"[Retriever] Candidatos únicos encontrados: {len(candidates)}")

    print(f"[Retriever] Re-ranking con cross-encoder...")
    top_chunks = _rerank(question, candidates)
    print(f"[Retriever] Top {TOP_K_FINAL} fragmentos seleccionados.\n")

    return top_chunks
