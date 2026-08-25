"""
app.py
Flask-Entrypoint für den internen KI-Wissensassistenten.
Stellt eine einfache Chat-Oberfläche bereit, die Nutzerfragen
an die RAG-Pipeline (rag/retrieval.py) weiterleitet.
"""

from flask import Flask, render_template, request, jsonify
from rag.retrieval import generate_answer
from rag.ingest import load_documents

app = Flask(__name__)


@app.route("/")
def index():
    """Zeigt die Chat-Oberfläche an."""
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Nimmt eine Nutzerfrage per POST entgegen und gibt die RAG-Antwort
    inkl. Quellen als JSON zurück.
    """
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Keine Frage übermittelt."}), 400

    try:
        result = generate_answer(query)
    except Exception as e:
        # Fängt u.a. den Fall ab, dass die ChromaDB-Collection leer/nicht
        # initialisiert ist (z.B. wenn rag/ingest.py noch nicht gelaufen ist)
        app.logger.error(f"Fehler bei der Antwortgenerierung: {e}")
        return jsonify({
            "error": "Die Wissensbasis konnte nicht durchsucht werden. "
                     "Bitte sicherstellen, dass 'python rag/ingest.py' ausgeführt wurde."
        }), 500

    return jsonify({
        "answer": result["answer"],
        "sources": result["sources"]
    })


@app.route("/documents")
def documents():
    """Zeigt eine Übersicht aller internen Dokumente, die der Assistent kennt."""
    docs = load_documents()
    return render_template("documents.html", documents=docs)


@app.route("/documents/<filename>")
def view_document(filename):
    """
    Zeigt den Volltext eines einzelnen Dokuments.

    Sicherheitshinweis: filename kommt aus der URL (Nutzereingabe) - wir
    greifen aber NICHT direkt mit diesem Wert auf das Dateisystem zu.
    Stattdessen gleichen wir ihn gegen die Liste der tatsächlich über
    load_documents() geladenen, bekannten Dateien ab. Das verhindert
    Path-Traversal-Angriffe (z.B. filename="../../.env").
    """
    docs = load_documents()
    doc = next((d for d in docs if d["filename"] == filename), None)

    if doc is None:
        return "Dokument nicht gefunden.", 404

    return render_template("document_detail.html", document=doc)

if __name__ == "__main__":
    app.run(debug=True, port=5000)