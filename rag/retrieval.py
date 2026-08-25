"""
retrieval.py
Läuft zur Laufzeit bei jeder Nutzeranfrage: holt die relevantesten
Chunks aus ChromaDB und lässt GPT-4o-mini daraus eine Antwort formulieren.

Getrennt von ingest.py, weil Ingestion (Indexierung) und Retrieval
(Abfrage) unterschiedliche Lebenszyklen haben - Ingestion läuft selten,
Retrieval bei jeder einzelnen Chat-Nachricht.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Wir importieren die Collection-Funktion aus ingest.py wieder,
# damit wir die Chroma-Verbindungslogik nicht duplizieren
from rag.ingest import get_chroma_collection

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def retrieve_relevant_chunks(query: str, n_results: int = 4) -> list[dict]:
    """
    Sucht die n_results ähnlichsten Chunks zur Nutzerfrage in ChromaDB.

    Rückgabe: Liste von Dicts mit 'text', 'source', 'distance' -
    'distance' brauchen wir später ggf. für Debugging/Transparenz.
    """
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    chunks = []
    for i, doc_text in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc_text,
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })

    return chunks


if __name__ == "__main__":
    chunks = retrieve_relevant_chunks("Wie lange dauert die Herstellergarantie?")
    for chunk in chunks:
        print(f"[{chunk['source']}] {chunk['text'][:80]}...")