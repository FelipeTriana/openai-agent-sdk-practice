"""
indexer.py — Carga el PDF, lo divide en fragmentos (chunks) y los indexa en ChromaDB.

Solo necesita ejecutarse UNA VEZ. Si el índice ya existe en disco, no hace nada.
"""

import hashlib
import os
from pathlib import Path

import chromadb
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# Configuración del chunking
# ──────────────────────────────────────────────
CHUNK_SIZE = 400       # Número máximo de palabras por fragmento
CHUNK_OVERLAP = 50     # Palabras que se repiten entre fragmentos consecutivos
                       # para no perder contexto en los bordes de cada chunk

# Directorio donde ChromaDB guardará el índice en disco
DB_DIR = Path(__file__).parent.parent / "chroma_db"

# Nombre de la colección dentro de ChromaDB
COLLECTION_NAME = "bachelard"

# Modelo bi-encoder para generar embeddings
EMBED_MODEL = "all-MiniLM-L6-v2"


def _extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extrae el texto del PDF página por página."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


def _split_into_chunks(pages: list[dict]) -> list[dict]:
    """
    Divide el texto de todas las páginas en fragmentos de tamaño fijo
    con superposición (overlap) para no perder contexto entre fragmentos.
    """
    # Juntamos todo el texto con sus metadatos de página
    all_words: list[tuple[str, int]] = []
    for page in pages:
        words = page["text"].split()
        for word in words:
            all_words.append((word, page["page"]))

    chunks = []
    start = 0
    while start < len(all_words):
        end = min(start + CHUNK_SIZE, len(all_words))
        chunk_words = [w for w, _ in all_words[start:end]]
        chunk_pages = list({p for _, p in all_words[start:end]})
        chunk_text = " ".join(chunk_words)

        # ID único basado en el contenido del chunk
        chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()

        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "pages": sorted(chunk_pages),
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def build_index(pdf_path: Path) -> chromadb.Collection:
    """
    Construye el índice vectorial a partir del PDF.
    Si el índice ya existe con datos, lo devuelve directamente sin reindexar.
    """
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Si ya tiene documentos indexados, no hace nada
    if collection.count() > 0:
        print(f"[Indexer] Índice ya existente ({collection.count()} fragmentos). Listo.")
        return collection

    print(f"[Indexer] Procesando PDF: {pdf_path.name}")
    pages = _extract_text_from_pdf(pdf_path)
    print(f"[Indexer] Páginas extraídas: {len(pages)}")

    chunks = _split_into_chunks(pages)
    print(f"[Indexer] Fragmentos generados: {len(chunks)}")

    print(f"[Indexer] Cargando modelo de embeddings '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    print("[Indexer] Generando embeddings (esto puede tardar unos minutos la primera vez)...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Guardamos en lotes de 500 para no saturar memoria
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings[i : i + batch_size],
            metadatas=[{"pages": str(c["pages"])} for c in batch],
        )
    print(f"[Indexer] {len(chunks)} fragmentos indexados correctamente.")
    return collection
