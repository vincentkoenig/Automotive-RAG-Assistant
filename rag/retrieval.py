"""
retrieval.py
Läuft zur Laufzeit bei jeder Nutzeranfrage: holt die relevantesten
Chunks aus ChromaDB und lässt GPT-4o-mini daraus eine Antwort formulieren.

Getrennt von ingest.py, weil Ingestion (Indexierung) und Retrieval
(Abfrage) unterschiedliche Lebenszyklen haben - Ingestion läuft selten,
Retrieval bei jeder einzelnen Chat-Nachricht.
"""

import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Wir importieren die Collection-Funktion aus ingest.py wieder,
# damit wir die Chroma-Verbindungslogik nicht duplizieren
from rag.ingest import get_chroma_collection

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def retrieve_relevant_chunks(query: str, n_results: int = 5) -> list[dict]:
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
- Wenn die Antwort nicht eindeutig aus dem Kontext hervorgeht, sage das ehrlich - erfinde keine Informationen. Nenne in diesem Fall am Ende "[Quelle: keine]".
- Wenn du die Frage beantworten kannst, nenne am Ende deiner Antwort in eckigen Klammern die verwendete(n) Quelle(n), im Format [Quelle: dateiname.txt]. Nenne nur Dokumente, die du tatsächlich für die Antwort verwendet hast.
- Antworte präzise und auf Deutsch.

Kontext aus internen Dokumenten:
{context}

Frage: {query}

Antwort:"""

    return prompt


def generate_answer(query: str) -> dict:
    """
    Führt den kompletten RAG-Ablauf aus: Retrieval + Generation.

    Die Quellenliste wird aus der GPT-Antwort selbst extrahiert
    (per Regex auf "[Quelle: ...]"), nicht aus den rohen Retrieval-
    Ergebnissen - so zeigen wir nur, was GPT tatsächlich verwendet hat,
    nicht alle abgerufenen (aber evtl. ungenutzten) Chunks.
    """
    chunks = retrieve_relevant_chunks(query)
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    # Alle "[Quelle: dateiname.txt]"-Markierungen aus der Antwort extrahieren
    found_sources = re.findall(r"\[Quelle:\s*([^\]]+)\]", answer)
    sources = list(set(
        s.strip() for s in found_sources
        if s.strip().lower() != "keine"
    ))

    # Markierung aus dem sichtbaren Antworttext entfernen, da die Quelle(n)
    # bereits separat als eigene Liste zurückgegeben werden - vermeidet
    # doppelte Anzeige im Chat (einmal inline, einmal als eigene Zeile)
    answer = re.sub(r"\s*\[Quelle:\s*[^\]]+\]", "", answer).strip()

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    result = generate_answer("Wie lange dauert die Herstellergarantie?")
    print("Antwort:", result["answer"])
    print("Quellen:", result["sources"])