# Automotive RAG Assistant 🚗💬

Ein interner KI-Wissensassistent für Mitarbeitende eines Autohauses, gebaut mit **Flask**, **OpenAI** und **ChromaDB**. Beantwortet Fragen zu Garantiebedingungen, Leasingkonditionen, Werkstattabläufen und internen Prozessen in natürlicher Sprache — auf Basis firmeninterner Dokumente, mit Quellenangabe und direkter Einsicht in die Originaldokumente.

**Hinweis:** Alle verwendeten Dokumente und die fiktive Firma „Musterhaus Mobility GmbH" sind zu Demonstrationszwecken frei erfunden.

## Warum dieses Projekt?

Klassische Suche in internen Dokumenten (Strg+F, Ordnerstruktur durchklicken) ist langsam und setzt voraus, dass man die richtigen Suchbegriffe kennt. Dieses Projekt zeigt, wie sich mit Retrieval-Augmented Generation (RAG) eine natürlichsprachliche Abfrage über interne Wissensdokumente umsetzen lässt — ohne das Sprachmodell selbst mit sensiblen Firmendaten zu trainieren oder fine-zutunen.

## Features

### 💬 Chat-Wissensassistent
- Natürlichsprachliche Fragen zu internen Dokumenten, beantwortet über GPT-4o-mini auf Basis semantisch abgerufener Textausschnitte
- Antworten inklusive Quellenangabe — direkt aus der Modellantwort extrahiert, sodass nur tatsächlich verwendete Dokumente angezeigt werden, nicht alle per Vektorsuche gefundenen (aber ggf. ungenutzten) Chunks
- Ehrliches Verhalten bei nicht beantwortbaren Fragen: Das Modell wird über den Prompt explizit angewiesen, keine Informationen zu erfinden, wenn die Antwort nicht im gelieferten Kontext steht
- Chatverlauf bleibt bei einem Seiten-Reload erhalten (clientseitig über `sessionStorage`), inklusive Button zum gezielten Zurücksetzen des Verlaufs
- Tipp-Indikator (animierte Punkte) während die Antwort generiert wird, statt eines stillen Wartens

### 📄 Dokumentenansicht
- Übersicht aller internen Dokumente, die der Assistent durchsucht, als Karten-Grid
- Volltextansicht jedes einzelnen Dokuments
- Quellenangaben im Chat sind direkt anklickbar und öffnen das jeweilige Dokument in einem neuen Tab — erhöht Nachvollziehbarkeit und Vertrauenswürdigkeit der Antworten
- Dateizugriff ausschließlich über eine kontrollierte, bekannte Liste geladener Dokumente (kein direkter Dateisystemzugriff über Nutzereingaben) — schützt gegen Path-Traversal-Angriffe

### 🔍 RAG-Pipeline
- **Strukturbasiertes Chunking**: Dokumente werden entlang ihrer nummerierten Abschnitte zerlegt, statt an fixen Zeichen-Grenzen — jeder Chunk bleibt eine geschlossene Sinneinheit
- **Semantische Ähnlichkeitssuche** über ChromaDB (lokaler, persistenter Vektor-Store) statt reinem Keyword-Matching
- **Idempotente Ingestion**: Ein erneuter Indexierungslauf (z. B. nach Änderung eines Dokuments) nutzt `upsert` statt `add` und führt nicht zu doppelten Einträgen oder Fehlern
- Robuste Fehlerbehandlung, falls die Wissensbasis noch nicht indexiert wurde (klare Nutzermeldung statt unspezifischem Server-Fehler)

### 🎨 UI / Design
- Eigenständiges Farbschema, angelehnt an ein Autohaus-Corporate-Design (Rot/Anthrazit), umgesetzt über CSS Custom Properties für zentrale, leicht anpassbare Theme-Werte
- Google-Font „Inter" für eine klare, moderne Typografie
- Empty-State mit Hinweistext, bevor die erste Frage gestellt wurde, statt eines leeren Chat-Fensters
- Vollständig eigenständiges Vanilla-HTML/CSS/JavaScript-Frontend, bewusst ohne Framework — der Fokus liegt auf der RAG-Pipeline, nicht auf UI-Komplexität

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_API-412991?style=flat&logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=flat&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)

- **Flask** — Backend, Routing und HTML-Seiten-Rendering
- **OpenAI API** — `text-embedding-3-small` für Embeddings, `gpt-4o-mini` für die Antwortgenerierung
- **ChromaDB** — lokaler, persistenter Vektor-Store für die semantische Ähnlichkeitssuche
- **python-dotenv** — sichere Verwaltung des API-Keys über Umgebungsvariablen
- **Vanilla JavaScript** — Chat-Interaktion, `fetch`-basierte API-Kommunikation, `sessionStorage` für clientseitige Verlaufspersistenz
- **Google Fonts (Inter)** — Typografie

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
Antwort inkl. Quellenangabe (klickbar zur Originalquelle)
```

### Warum diese Architektur-Entscheidungen?

- **Strukturbasiertes Chunking statt fixer Zeichen-Grenzen**: Die Beispieldokumente sind bewusst in nummerierte, thematisch geschlossene Abschnitte gegliedert. Das Chunking folgt dieser Struktur, sodass jeder Chunk eine vollständige Sinneinheit bleibt statt mitten im Satz abgeschnitten zu werden.
- **Lokaler Vektor-Store (ChromaDB)**: Kein externer Server nötig, kostenlos, ausreichend für eine überschaubare Wissensbasis. Bei größerem Datenvolumen oder Multi-User-Betrieb wäre ein gehosteter Vektor-Store (z. B. Pinecone, Weaviate Cloud) die konsequente Weiterentwicklung.
- **Quellenangabe aus der Antwort extrahiert**: Statt alle per Vektorsuche gefundenen Chunks als „Quelle" anzuzeigen, wird die Quellenangabe aus der vom Modell selbst genannten Referenz extrahiert. Details dazu in [`NOTES.md`](NOTES.md).
- **Niedrige Temperature (0.2)**: Für einen Wissensassistenten sind konsistente, faktennahe Antworten wichtiger als kreative Variation.
- **Getrennte Ingestion- und Retrieval-Module**: `rag/ingest.py` (Indexierung, läuft selten) ist bewusst von `rag/retrieval.py` (läuft bei jeder Chat-Anfrage) getrennt — unterschiedliche Lebenszyklen, unterschiedliche Verantwortlichkeiten.

## Projektstruktur

```
automotive-rag-assistant/
├── app.py                        # Flask-Routen (Chat-API, Dokumentenansicht)
├── rag/
│   ├── __init__.py
│   ├── ingest.py                   # Dokumente laden, chunken, embedden, in ChromaDB speichern (upsert)
│   └── retrieval.py                # Ähnlichkeitssuche, Prompt-Bau, GPT-4o-mini-Antwortgenerierung
├── data/
│   └── documents/                  # Interne Beispieldokumente (.txt)
├── chroma_db/                      # Lokaler, persistenter Vektor-Store (generiert, nicht versioniert)
├── templates/
│   ├── chat.html                     # Chat-Oberfläche
│   ├── documents.html                 # Dokumentenübersicht (Karten-Grid)
│   └── document_detail.html            # Volltextansicht eines Dokuments
├── static/
│   └── style.css                    # Corporate-Design, Chat-Bubbles, Karten, Animationen
├── requirements.txt
├── .env                             # OPENAI_API_KEY (nicht versioniert)
├── README.md
└── NOTES.md                         # Architektur-Entscheidungen, verworfene Ansätze, Retrieval-Qualitätstests
```

## API-Übersicht

| Methode | Route | Beschreibung |
|--------|-------|-------------|
| `GET` | `/` | Chat-Oberfläche |
| `POST` | `/api/chat` | Nimmt eine Nutzerfrage entgegen, gibt Antwort inkl. Quellen als JSON zurück |
| `GET` | `/documents` | Übersicht aller internen Dokumente |
| `GET` | `/documents/<filename>` | Volltextansicht eines einzelnen Dokuments |

## Erste Schritte

**1. Repository klonen und virtuelle Umgebung einrichten**
```bash
git clone https://github.com/vincentkoenig/automotive-rag-assistant.git
cd automotive-rag-assistant
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**2. `.env`-Datei anlegen**
```
OPENAI_API_KEY=dein_openai_api_key
```
> API-Key gibt es unter [platform.openai.com](https://platform.openai.com/api-keys)

**3. Dokumente indexieren**

Einmalig, oder erneut nach Änderungen an den Dokumenten in `data/documents/`:
```bash
python rag/ingest.py
```
Dank `upsert` ist dieser Schritt idempotent — ein erneuter Lauf erzeugt keine Duplikate.

**4. Anwendung starten**
```bash
python app.py
```

**5. Im Browser öffnen**
```
http://127.0.0.1:5000
```

## Retrieval-Qualität

Die Retrieval-Qualität wurde anhand von sieben Testfragen unterschiedlicher Schwierigkeit systematisch geprüft (leicht, mittel formuliert, indirekt formuliert, Cross-Dokument-Fragen, Fragen außerhalb der Wissensbasis). Details, inklusive einer dokumentierten und später behobenen Schwäche bei stark umschriebenen Fragen sowie den daraus gezogenen Erkenntnissen, siehe [`NOTES.md`](NOTES.md).

## Was fehlt bewusst (Zeitrahmen: 3 Tage MVP)

- Kein Reranking oder Hybrid-Suche (Vektor + Keyword/BM25)
- Keine Nutzerauthentifizierung / Multi-User-Unterstützung
- Keine automatisierte Evaluation (z. B. RAGAS) — nur systematische, manuelle Testfragen
- Chatverlauf nur clientseitig persistiert (sessionStorage), nicht serverseitig über mehrere Geräte hinweg
- Kein Cloud-Deployment (lokale Demo für dieses MVP ausreichend)

## Datenschutz

Alle Dokumente sind fiktiv. In einem echten Einsatz mit tatsächlichen Kundendaten wären zusätzlich erforderlich: Zugriffskontrolle je nach Mitarbeiterrolle, ein Auftragsverarbeitungsvertrag mit dem LLM-Anbieter (oder Einsatz eines lokal gehosteten Modells), sowie eine bewusste Prüfung, welche Dokumentinhalte überhaupt für eine KI-Verarbeitung geeignet sind.

## Was ich dabei gelernt habe

- Aufbau einer vollständigen RAG-Pipeline von Grund auf — Chunking, Embedding, Vektorsuche und Antwortgenerierung selbst implementiert statt über ein Framework wie LangChain abstrahiert, um jeden Schritt im Detail zu verstehen und erklären zu können
- Strukturbasiertes Chunking anhand der Dokumentstruktur statt fixer Zeichen-Grenzen, inklusive Debugging eines fehlerhaften Filters gegen inhaltsleere Titel-Chunks (Längenfilter griff zu kurz, ein musterbasierter Filter auf die Abschnittsnummerierung löste es sauber)
- Systematisches Testen der Retrieval-Qualität mit einer bewusst gestuften Testfragen-Matrix (leicht/mittel/schwer/Cross-Dokument/Out-of-Scope), statt sich auf einzelne Ad-hoc-Fragen zu verlassen
- Analyse und gezielte Behebung einer realen Retrieval-Schwäche: Eine indirekt formulierte Frage fand zunächst nicht den inhaltlich präzisesten Chunk — durch Anpassung von `n_results` konnte das gezielt nachgewiesen und behoben werden
- Vermeidung von Halluzinationen über eine klare Prompt-Anweisung, verifiziert durch gezielte Out-of-Scope-Testfragen
- Idempotente Datenverarbeitung: Umstellung von `add` auf `upsert` in ChromaDB, damit wiederholte Indexierungsläufe keine Duplikate oder Fehler erzeugen
- Iteratives Debugging eines fehlgeschlagenen JSON-Mode-Ansatzes zur strukturierten Antwortausgabe — Rückkehr zu einer robusteren, einfacheren Regex-Extraktion aus der Fließtext-Antwort, samt Dokumentation der verworfenen Alternative
- Absicherung gegen Path-Traversal-Angriffe bei der Dokumentenansicht durch Abgleich von Nutzereingaben gegen eine kontrollierte, bekannte Dateiliste statt direktem Dateisystemzugriff
- Umsetzung eines eigenen, an eine reale Marke angelehnten Corporate-Designs über CSS Custom Properties, inklusive Debugging eines CSS-Spezifitätskonflikts zwischen zwei Selektoren
- Clientseitige Zustandspersistenz über `sessionStorage`, um den Chatverlauf bei einem versehentlichen Seiten-Reload zu erhalten — bewusst als pragmatische MVP-Lösung anstelle einer serverseitigen Session-Verwaltung
- Durchgängige, saubere Trennung von Ingestion- und Retrieval-Logik in eigene Module, mit klar dokumentierten Design-Entscheidungen und verworfenen Ansätzen in einer begleitenden `NOTES.md`
