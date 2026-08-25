"""
ingest.py
Liest die internen Dokumente ein, zerlegt sie in Chunks,
erzeugt Embeddings und speichert sie in ChromaDB.
Dieser Prozess läuft einmalig (bzw. bei neuen/geänderten Dokumenten),
getrennt vom eigentlichen Retrieval zur Laufzeit.
"""

import os

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


# Kleiner manueller Test: Datei direkt ausführen, um zu prüfen ob das Einlesen klappt
if __name__ == "__main__":
    docs = load_documents()
    print(f"{len(docs)} Dokumente gefunden:")
    for doc in docs:
        print(f"- {doc['filename']} ({len(doc['content'])} Zeichen)")