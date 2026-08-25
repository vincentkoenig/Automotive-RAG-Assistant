"""
ingest.py
Liest die internen Dokumente ein, zerlegt sie in Chunks,
erzeugt Embeddings und speichert sie in ChromaDB.
Dieser Prozess läuft einmalig (bzw. bei neuen/geänderten Dokumenten),
getrennt vom eigentlichen Retrieval zur Laufzeit.
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()  # lädt OPENAI_API_KEY aus der .env-Datei in die Umgebungsvariablen

# Pfad zum Dokumentenordner, relativ zu dieser Datei
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "documents")

# Pfad, unter dem ChromaDB seine Daten persistent speichert (lokal, dateibasiert)
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")

COLLECTION_NAME = "musterhaus_mobility_docs"

def load_documents() -> list[dict]:
    """
    Liest alle .txt-Dateien aus DOCUMENTS_DIR ein.

    Rückgabe: Liste von Dicts mit 'filename' und 'content',
    damit wir später wissen, aus welchem Dokument ein Chunk stammt
    (wichtig für die Quellenangabe im Chat).
    """
    documents = []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "filename": filename,
                "content": content
            })

    return documents


def chunk_document(filename: str, content: str) -> list[dict]:
    """
    Zerlegt ein Dokument in Chunks anhand der nummerierten Abschnitte
    (z.B. "1. Arbeitszeiten", "2. Urlaubsregelung").

    Strukturbasiertes Chunking statt fixer Zeichen-Grenzen, weil unsere
    Dokumente bewusst in geschlossene Sinneinheiten gegliedert sind -
    so bleibt jeder Chunk inhaltlich vollständig und nicht mitten im
    Satz abgeschnitten.

    Rückgabe: Liste von Dicts mit 'text', 'source' (Dateiname) und
    'chunk_index' (Position im Dokument, für Nachvollziehbarkeit).
    """
    # Nur Abschnitte behalten, die tatsächlich mit "Zahl. " beginnen -
    # der Teil vor der ersten Nummerierung (Titel/Header) enthält keine
    # eigenständige inhaltliche Aussage und wird bewusst verworfen.
    SECTION_PATTERN = re.compile(r"^\d+\.\s")

    pattern = r"(?m)^(?=\d+\.\s)"
    raw_sections = re.split(pattern, content)

    chunks = []
    for i, section in enumerate(raw_sections):
        section = section.strip()
        if not section:
            continue  # leere Abschnitte überspringen

        if not SECTION_PATTERN.match(section):
            continue  # kein nummerierter Abschnitt (z.B. Titelzeile) -> verwerfen

        chunks.append({
            "text": section,
            "source": filename,
            "chunk_index": i
        })

    return chunks


def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """Wendet chunk_document auf alle geladenen Dokumente an."""
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc["filename"], doc["content"])
        all_chunks.extend(doc_chunks)
    return all_chunks


def get_chroma_collection():
    """
    Erstellt (oder öffnet, falls schon vorhanden) eine persistente ChromaDB-Collection.

    Eine Collection ist konzeptionell vergleichbar mit einer SQL-Tabelle:
    sie enthält viele "Einträge", hier: Chunk-Text + Vektor + Metadaten.

    Wir übergeben eine OpenAI-Embedding-Funktion, damit ChromaDB automatisch
    weiß, wie es aus Text einen Vektor erzeugt (sowohl beim Speichern als
    auch später bei der Suche mit einer neuen Nutzerfrage).
    """
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )

    return collection


def store_chunks(chunks: list[dict]) -> None:
    """
    Speichert die Chunks in ChromaDB. ChromaDB erzeugt dabei automatisch
    (über die hinterlegte embedding_function) die Embedding-Vektoren -
    wir müssen die OpenAI Embeddings API also nicht manuell aufrufen.

    Nutzt upsert statt add: dadurch ist die Funktion idempotent - ein
    erneuter Lauf (z.B. nach Änderung eines Dokuments) aktualisiert
    bestehende Chunks anhand ihrer ID, statt einen Fehler wegen
    doppelter IDs zu werfen oder Duplikate anzulegen.
    """
    collection = get_chroma_collection()

    ids = [f"{chunk['source']}_{chunk['chunk_index']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "chunk_index": chunk["chunk_index"]} for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"{len(chunks)} Chunks in ChromaDB gespeichert/aktualisiert (Collection: '{COLLECTION_NAME}').")


def test_query(query: str, n_results: int = 3):
    """Testet die Ähnlichkeitssuche isoliert, ohne LLM-Antwort - nur zur Kontrolle."""
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    print(f"\nSuche: '{query}'\n")
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        distance = results["distances"][0][i]
        print(f"[{i+1}] (Quelle: {source}, Distanz: {distance:.4f})")
        print(f"    {doc[:100]}...\n")


if __name__ == "__main__":
    docs = load_documents()
    print(f"{len(docs)} Dokumente gefunden.")

    chunks = chunk_all_documents(docs)
    print(f"{len(chunks)} Chunks erzeugt.")

    store_chunks(chunks)

    test_query("Was passiert bei einem Motorschaden?")