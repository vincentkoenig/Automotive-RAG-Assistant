# Automotive RAG Assistant

Ein interner KI-Wissensassistent für Mitarbeitende eines Autohauses, der Fragen
zu Garantiebedingungen, Leasingkonditionen, Werkstattabläufen und internen
Prozessen auf Basis firmeninterner Dokumente beantwortet.

**Hinweis:** Alle verwendeten Dokumente und die fiktive Firma "Musterhaus
Mobility GmbH" sind zu Demonstrationszwecken frei erfunden.

## Warum dieses Projekt?

Klassische Suche in internen Dokumenten (Strg+F, Ordnerstruktur durchklicken)
ist langsam und setzt voraus, dass man die richtigen Suchbegriffe kennt.
Dieses Projekt zeigt, wie sich mit Retrieval-Augmented Generation (RAG) eine
natürlichsprachliche Abfrage über interne Wissensdokumente umsetzen lässt -
ohne das Sprachmodell selbst mit sensiblen Firmendaten zu trainieren oder
fine-zutunen.

## Was ist RAG?

Retrieval-Augmented Generation kombiniert zwei Schritte:

1. **Retrieval**: Zu einer Nutzerfrage werden per semantischer Ähnlichkeitssuche
   (Vektor-Embeddings) die relevantesten Textausschnitte aus einer
   Wissensdatenbank ermittelt - nicht per Keyword-Matching, sondern basierend
   auf inhaltlicher Bedeutung.
2. **Generation**: Ein Sprachmodell (hier: GPT-4o-mini) erhält diese
   Textausschnitte als Kontext und formuliert daraus eine Antwort.

Der Vorteil: Das Sprachmodell "kennt" die internen Dokumente nicht auswendig,
sondern bekommt sie situativ zur Anfrage mitgeliefert. Das hält die
Wissensbasis pflegbar (neue Dokumente hinzufügen, ohne das Modell neu zu
trainieren) und ermöglicht Quellenangaben.

## Architektur

```
Nutzerfrage
    ↓
Embedding der Frage (OpenAI text-embedding-3-small)
    ↓
Ähnlichkeitssuche in ChromaDB (lokaler Vektor-Store)
    ↓
Top-k relevante Dokument-Chunks
    ↓
Prompt mit Chunks als Kontext an GPT-4o-mini
    ↓
Antwort inkl. Quellenangabe
```

**Tech-Stack:**
- **Backend**: Python, Flask
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vektor-Datenbank**: ChromaDB (lokal, persistent)
- **Frontend**: Vanilla HTML/CSS/JavaScript (bewusst schlicht gehalten - der
  Fokus liegt auf der RAG-Pipeline, nicht auf UI-Politur)

### Warum diese Architektur-Entscheidungen?

- **Strukturbasiertes Chunking** (statt fixer Zeichen-Grenzen): Die
  Beispieldokumente sind bewusst in nummerierte, thematisch geschlossene
  Abschnitte gegliedert. Das Chunking folgt dieser Struktur, sodass jeder
  Chunk eine vollständige Sinneinheit bleibt statt mitten im Satz
  abgeschnitten zu werden.
- **Lokaler Vektor-Store (ChromaDB)**: Kein externer Server nötig, kostenlos,
  ausreichend für eine überschaubare Wissensbasis. Bei größerem Datenvolumen
  oder Multi-User-Betrieb wäre ein gehosteter Vektor-Store (z.B. Pinecone,
  Weaviate Cloud) die konsequente Weiterentwicklung.
- **Quellenangabe aus der Antwort extrahiert**: Statt alle per Vektorsuche
  gefundenen Chunks als "Quelle" anzuzeigen (auch wenn nicht alle tatsächlich
  zur Antwort beigetragen haben), wird die Quellenangabe aus der vom Modell
  selbst genannten Referenz extrahiert. Details dazu in `NOTES.md`.
- **Niedrige Temperature (0.2)**: Für einen Wissensassistenten sind
  konsistente, faktennahe Antworten wichtiger als kreative Variation.

## Setup

1. Repository klonen und virtuelle Umgebung erstellen:
   ```bash
   git clone https://github.com/vincentkoenig/automotive-rag-assistant.git
   cd automotive-rag-assistant
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```

2. `.env`-Datei im Projekt-Root anlegen:
   ```
   OPENAI_API_KEY=dein-api-key
   ```

3. Dokumente indexieren (einmalig, oder nach Änderung an den Dokumenten in
   `data/documents/`):
   ```bash
   python rag/ingest.py
   ```

4. Anwendung starten:
   ```bash
   python app.py
   ```
   Anschließend im Browser: `http://127.0.0.1:5000`

## Retrieval-Qualität

Die Retrieval-Qualität wurde anhand von 7 Testfragen unterschiedlicher
Schwierigkeit systematisch geprüft (leicht, mittel formuliert, indirekt
formuliert, Cross-Dokument-Fragen, Fragen außerhalb der Wissensbasis).
Details, inklusive einer dokumentierten Schwäche bei stark umschriebenen
Fragen und den daraus gezogenen Erkenntnissen, siehe [`NOTES.md`](NOTES.md).

## Was fehlt bewusst (Zeitrahmen: 3 Tage MVP)

- Kein Reranking oder Hybrid-Suche (Vektor + Keyword/BM25)
- Keine Nutzerauthentifizierung / Multi-User-Unterstützung
- Keine automatisierte Evaluation (z.B. RAGAS) - nur manuelle Testfragen
- Keine Chat-History über Sessions hinweg
- Kein Cloud-Deployment (lokale Demo für dieses MVP ausreichend)

## Datenschutz

Alle Dokumente sind fiktiv. In einem echten Einsatz mit tatsächlichen
Kundendaten wären zusätzlich erforderlich: Zugriffskontrolle je nach
Mitarbeiterrolle, Auftragsverarbeitungsvertrag mit dem LLM-Anbieter (oder
Einsatz eines lokal gehosteten Modells), sowie eine Prüfung, welche
Dokumentinhalte überhaupt für eine KI-Verarbeitung geeignet sind.