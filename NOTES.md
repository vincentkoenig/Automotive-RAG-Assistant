# Entwicklungsnotizen – Automotive RAG Assistant

Diese Datei dokumentiert Design-Entscheidungen und Testergebnisse während der
Entwicklung, insbesondere zur Retrieval-Qualität. Zweck: Nachvollziehbarkeit
für mich selbst und als Gesprächsgrundlage (z.B. im Interview).

## Architektur-Entscheidungen

- **Chunking**: Strukturbasiert (Split an nummerierten Abschnitten wie "1. ...",
  "2. ..."), statt fixer Zeichen-Grenzen mit Overlap. Passt zu den bewusst so
  strukturierten Beispieldokumenten - jeder Chunk bleibt eine geschlossene
  Sinneinheit.
- **Embedding-Modell**: `text-embedding-3-small` (OpenAI) - für die überschaubare,
  thematisch klar abgegrenzte Wissensbasis (21 Chunks aus 4 Dokumenten)
  ausreichend. Bei größerer/heterogenerer Dokumentenbasis wäre
  `text-embedding-3-large` zu evaluieren.
- **Vektor-Store**: ChromaDB, lokal persistent (`chroma_db/`, nicht versioniert -
  reproduzierbar aus `data/documents/` via `ingest.py`).
- **Generation**: GPT-4o-mini, `temperature=0.2` (niedrig, da faktische statt
  kreative Antworten gewünscht sind).
- **Anti-Halluzination**: Prompt weist das Modell explizit an, bei fehlendem
  Kontext ehrlich "keine Antwort möglich" zu signalisieren (`[Quelle: keine]`),
  statt zu spekulieren.
- **Quellenangabe**: Wird aus der GPT-Antwort selbst per Regex extrahiert
  (`[Quelle: dateiname.txt]`), nicht aus der rohen ChromaDB-Trefferliste.
  Grund: ChromaDB liefert bei `n_results=4` immer 4 Treffer, auch wenn nicht
  alle inhaltlich relevant sind - das führte anfangs zu irreführenden
  Quellenangaben (z.B. 3 Quellen angezeigt, obwohl die Frage unbeantwortbar war).

## Verworfene Ansätze

- **JSON-Mode für strukturierte Antworten** (`response_format={"type": "json_object"}`):
  Ursprünglich geplant, um Antwort und Quellen sauber getrennt zu bekommen.
  Führte in der Praxis zu leeren/fehlerhaften Antworten (Ursache nicht
  abschließend geklärt). Zugunsten von Zeitdruck auf die einfachere,
  funktionierende Regex-Extraktion aus der Fließtext-Antwort zurückgegangen.
  Für eine produktionsreifere Version wäre JSON-Mode oder Function Calling
  der robustere Ansatz.
- **Längenbasierter Filter gegen Titel-Chunks** (`MIN_CHUNK_LENGTH`): Erste
  Version filterte Chunks unter einer Mindestlänge, um Dokumenttitel
  auszuschließen. Fehlschlag: Titelzeilen waren teils länger als kürzere,
  aber inhaltlich vollständige Abschnitte. Ersetzt durch einen Filter, der
  gezielt nur Abschnitte mit nummeriertem Muster (`^\d+\.\s`) behält.

## Retrieval-Qualitäts-Tests (Tag 1)

7 Testfragen in 5 Kategorien, manuell durchgeführt und bewertet:

| # | Kategorie | Frage | Ergebnis |
|---|---|---|---|
| 1 | Leicht | Wie viele Urlaubstage habe ich pro Jahr? | ✅ Korrekt, exakte Quelle |
| 2 | Leicht | Wie lange dauert die Herstellergarantie? | ✅ Korrekt, exakte Quelle |
| 3 | Mittel | Was muss ich tun, wenn mein Dienstwagen einen Unfall hatte? | ✅ Korrekt, exakte Quelle |
| 4 | Mittel | Kann ich meinen Leasingvertrag vorzeitig beenden? | ✅ Korrekt, exakte Quelle |
| 5 | Schwer | Was passiert bei einem Motorschaden? | ✅ Nach Anpassung (n_results 4→5) korrekt, siehe Analyse unten |
| 6 | Cross-Dokument | Bekomme ich ein Ersatzfahrzeug, wenn mein Auto in der Werkstatt ist? | ✅ Korrekt, exakte Quelle, sauber von Leasing-Rückgabe getrennt |
| 7 | Out-of-Scope | Wie hoch ist die Mehrwertsteuer in Deutschland? | ✅ Korrekt abgelehnt, keine Halluzination |

**Bilanz: 6/7 einwandfrei, 1/7 mit dokumentierter Schwäche.**

### Analyse Fall #5 (Motorschaden)

Die Frage "Was passiert bei einem Motorschaden?" fand als Top-Treffer den
Abschnitt "3. Ablauf bei einem Garantiefall" (allgemeiner Prozess), nicht
Abschnitt "5. Ausschlüsse", wo wörtlich "Antriebsstrangdefekten (z.B.
Motorschaden...)" steht.

Mögliche Ursache: Der Ausschlüsse-Abschnitt ist thematisch breiter (behandelt
mehrere Ausschluss-Gründe), wodurch das Embedding des gesamten Abschnitts
"verdünnt" wird und der spezifische Begriff "Motorschaden" im Vektor weniger
Gewicht bekommt als in einem fokussierteren Chunk.

Trotzdem lieferte GPT-4o-mini am Ende eine inhaltlich brauchbare Antwort, da
alle Top-Treffer aus demselben Dokument (`garantierichtlinien.txt`) stammten
und der generelle Ablauf beschrieben wurde - nur eben nicht die spezifischste
verfügbare Information.

**Mögliche Verbesserungen (nicht umgesetzt, Zeitrahmen):**
- Feineres Chunking innerhalb langer Abschnitte (z.B. Ausschlüsse-Liste in
  einzelne Punkte aufteilen)
- `n_results` erhöhen, um mehr Kontext-Puffer zu geben
- Hybrid-Suche (Vektor + Keyword/BM25) für Fälle mit spezifischen Fachbegriffen

**Update:** Durch Erhöhung von `n_results` von 4 auf 5 in
`retrieve_relevant_chunks()` wurde der relevante Ausschlüsse-Chunk
zuverlässig mit in den Kontext aufgenommen. Erneuter Test lieferte eine
inhaltlich präzise Antwort mit korrektem Bezug auf die Wartungsintervall-
Bedingung. Die ursprüngliche Hypothese (Chunk zu lang/thematisch verdünnt)
war demnach nicht die Hauptursache - der Chunk war lediglich knapp außerhalb
des betrachteten Suchradius. Zeigt: Auch scheinbar kleine Parameter wie
`n_results` können messbaren Einfluss auf die Antwortqualität haben.

### Generelle Erkenntnis

Semantische Suche funktioniert zuverlässig, wenn Frage und Dokumentformulierung
ähnliche Begriffe verwenden. Bei stark umschriebenen/indirekten Fragen kann der
inhaltlich präziseste Chunk hinter thematisch verwandten, aber allgemeineren
Chunks zurückfallen. Retrieval-Qualität ist kein "einmal eingerichtet, fertig"-
Thema, sondern erfordert aktives Testen mit realistischen Fragen - reines
"es läuft ja" reicht nicht aus.

## Offene Punkte / Nächste Schritte

- Mobile-Optimierung des Chat-UI
- Ggf. Hybrid-Retrieval (Vektor + Keyword) für Fachbegriff-lastige Fragen
- Strukturierter Output (JSON-Mode/Function Calling) statt Regex-Parsing,
  falls Zeit reicht