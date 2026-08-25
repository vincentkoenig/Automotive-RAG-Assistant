"""
app.py
Flask-Entrypoint für den internen KI-Wissensassistenten.
Stellt eine einfache Chat-Oberfläche bereit, die Nutzerfragen
an die RAG-Pipeline (rag/retrieval.py) weiterleitet.
"""

from flask import Flask, render_template, request, jsonify
from rag.retrieval import generate_answer

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


if __name__ == "__main__":
    app.run(debug=True, port=5000)