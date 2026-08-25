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


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Baut den Prompt für GPT-4o-mini: die abgerufenen Chunks werden als
    Kontext eingebettet, zusammen mit klaren Anweisungen, wie das Modell
    damit umgehen soll (u.a. ehrlich sagen, wenn die Antwort nicht im
    Kontext steht - wichtig gegen Halluzinationen).
    """
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Quelle: {chunk['source']}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Du bist ein interner Wissensassistent für Mitarbeitende der Musterhaus Mobility GmbH.
Beantworte die folgende Frage AUSSCHLIESSLICH auf Basis des unten stehenden Kontexts aus internen Dokumenten.

Wichtige Regeln:
- Wenn die Antwort nicht eindeutig aus dem Kontext hervorgeht, sage das ehrlich - erfinde keine Informationen.
- Nenne am Ende deiner Antwort, aus welchem Dokument (Quelle) die Information stammt.
- Antworte präzise und auf Deutsch.

Kontext aus internen Dokumenten:
{context}

Frage: {query}

Antwort:"""

    return prompt


def generate_answer(query: str) -> dict:
    """
    Führt den kompletten RAG-Ablauf aus: Retrieval + Generation.

    Rückgabe: Dict mit 'answer' (GPT-Antwort) und 'sources'
    (Liste der verwendeten Dokumente, für Transparenz im Chat-UI).
    """
    chunks = retrieve_relevant_chunks(query)
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2  # niedrig, weil wir faktische, konsistente Antworten wollen - nicht kreativ
    )

    answer = response.choices[0].message.content

    # Eindeutige Quellen (ohne Duplikate) für die Anzeige im Chat sammeln
    sources = list(set(chunk["source"] for chunk in chunks))

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    result = generate_answer("Wie lange dauert die Herstellergarantie?")
    print("Antwort:", result["answer"])
    print("Quellen:", result["sources"])