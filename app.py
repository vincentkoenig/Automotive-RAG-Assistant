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

    result = generate_answer(query)

    return jsonify({
        "answer": result["answer"],
        "sources": result["sources"]
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)