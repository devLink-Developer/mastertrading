# Knowledge Exports

This folder contains machine-readable context exports generated from:

- `C:\Users\rortigoza\Documents\messages.md`

## Files

- `messages_context.jsonl`
  - One entry per line (best for RAG ingestion and embeddings).
- `messages_knowledge.jsonld`
  - Semantic knowledge graph representation (best for linked data pipelines).
- `messages_knowledge.graphml`
  - Graph format for visualization and graph tools (Gephi, NetworkX, Neo4j import workflows).

## Regenerate

Run from repo root:

```powershell
python scripts/export_messages_knowledge.py --input "C:\Users\rortigoza\Documents\messages.md" --output-dir knowledge
```

## Notes

- The exporter extracts entries, tags, settings, symbols, and file mentions.
- JSON-LD and GraphML share the same base entities (`Project`, `ContextEntry`, `Setting`, `Symbol`, `File`).
