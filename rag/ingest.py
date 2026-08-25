"""
ingest.py
Liest die internen Dokumente ein, zerlegt sie in Chunks,
erzeugt Embeddings und speichert sie in ChromaDB.
Dieser Prozess läuft einmalig (bzw. bei neuen/geänderten Dokumenten),
getrennt vom eigentlichen Retrieval zur Laufzeit.
"""

import os
import re

# Pfad zum Dokumentenordner, relativ zu dieser Datei
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "documents")


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


# Kleiner manueller Test: Datei direkt ausführen, um zu prüfen ob das Einlesen klappt
if __name__ == "__main__":
    docs = load_documents()
    print(f"{len(docs)} Dokumente gefunden:")
    for doc in docs:
        print(f"- {doc['filename']} ({len(doc['content'])} Zeichen)")

    chunks = chunk_all_documents(docs)
    print(f"\n{len(chunks)} Chunks erzeugt:\n")
    for chunk in chunks:
        preview = chunk["text"][:60].replace("\n", " ")
        print(f"[{chunk['source']} #{chunk['chunk_index']}] {preview}...")