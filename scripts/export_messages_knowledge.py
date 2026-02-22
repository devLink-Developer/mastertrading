#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape as xml_escape


SYMBOL_RE = re.compile(r"\b([A-Z]{3,12}USDT)\b")
SETTING_RE = re.compile(r"\b([A-Z][A-Z0-9_]{4,})\b")
FILE_RE = re.compile(r"`([^`]*\.(?:py|md|json|yml|yaml|env|txt|sql|csv|graphml))`")
TIMESTAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?|\d{2}-[A-Za-z]{3}-\d{4})"
)
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
DELIM_RE = re.compile(r"^\s*---+\s*$", re.MULTILINE)

KNOWN_SETTINGS = {
    "MARKET_REGIME_ADX_MIN",
    "SIGNAL_FLIP_MIN_AGE_ENABLED",
    "SIGNAL_FLIP_MIN_AGE_MINUTES",
    "ALLOCATOR_MIN_MODULES_ACTIVE",
    "PER_INSTRUMENT_RISK",
    "MAX_DAILY_TRADES_LOW_ADX",
    "MODULE_ADX_TREND_MIN",
    "REGIMEFILTERCONFIG",
}

TAG_MAP: dict[str, list[str]] = {
    "risk": ["risk", "drawdown", "sl", "stop", "ruin", "volatility", "regime"],
    "execution": ["execution", "flip", "order", "position", "exchange_close", "cooldown"],
    "signals": ["signal", "trend", "meanrev", "allocator", "smc", "adx"],
    "backtest": ["backtest", "walk-forward", "oos", "purged", "monte carlo"],
    "strategy": ["expectancy", "pnl", "win rate", "profit factor", "edge"],
    "assets": ["btc", "eth", "alts", "symbol"],
    "ops": ["docker", "runtime", "test", "migration", "live"],
}


@dataclass
class Entry:
    entry_id: str
    index: int
    title: str
    actor: str
    category: str
    timestamp: str
    text: str
    summary: str
    tags: list[str]
    settings: list[str]
    symbols: list[str]
    files: list[str]


def normalize_title(raw: str) -> str:
    return " ".join(raw.replace("\ufeff", "").split()).strip()


def detect_actor(title: str, text: str) -> str:
    blob = f"{title}\n{text}".lower()
    if "supervisi" in blob:
        return "supervisor"
    if "supervisor" in blob:
        return "supervisor"
    if "agente" in blob or "agent" in blob:
        return "agent"
    if "investig" in blob:
        return "research"
    return "log"


def detect_category(title: str, text: str) -> str:
    blob = f"{title}\n{text}".lower()
    if "evaluacion del pdf" in blob or "investig" in blob:
        return "analysis"
    if "observacion" in blob:
        return "review"
    if "status de implementacion" in blob or "live" in blob:
        return "implementation"
    if "plan de accion" in blob or "prioridad" in blob:
        return "plan"
    return "note"


def infer_tags(text: str) -> list[str]:
    blob = text.lower()
    tags: list[str] = []
    for tag, words in TAG_MAP.items():
        if any(word in blob for word in words):
            tags.append(tag)
    return tags


def extract_summary(text: str, limit: int = 260) -> str:
    line = " ".join([ln.strip() for ln in text.splitlines() if ln.strip()])
    return line[:limit].strip()


def safe_node_id(prefix: str, raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_")
    if not cleaned:
        cleaned = "unknown"
    return f"{prefix}:{cleaned}"


def extract_entries(markdown_text: str) -> list[Entry]:
    parts = [p.strip() for p in DELIM_RE.split(markdown_text) if p.strip()]
    entries: list[Entry] = []
    for idx, chunk in enumerate(parts, start=1):
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        heading = ""
        for ln in lines:
            m = HEADING_RE.match(ln)
            if m:
                heading = normalize_title(m.group(1))
                break
        if not heading:
            heading = normalize_title(lines[0][:120])
        timestamp_match = TIMESTAMP_RE.search(chunk)
        timestamp = timestamp_match.group(1) if timestamp_match else ""
        settings_raw = {
            s
            for s in SETTING_RE.findall(chunk)
            if s in KNOWN_SETTINGS or ("_" in s and len(s) >= 8)
        }
        settings = sorted(
            {
                s
                for s in settings_raw
                if not s.endswith("_")
                and "*" not in s
                and not s.startswith("_")
                and len(s.strip("_")) >= 6
            }
        )
        symbols = sorted(set(SYMBOL_RE.findall(chunk)))
        files = sorted(set(FILE_RE.findall(chunk)))
        summary = extract_summary(chunk)
        entry = Entry(
            entry_id=f"entry:{idx}",
            index=idx,
            title=heading,
            actor=detect_actor(heading, chunk),
            category=detect_category(heading, chunk),
            timestamp=timestamp,
            text=chunk.strip(),
            summary=summary,
            tags=infer_tags(chunk),
            settings=settings,
            symbols=symbols,
            files=files,
        )
        entries.append(entry)
    return entries


def build_jsonl(entries: Iterable[Entry], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fh:
        for e in entries:
            row = {
                "id": e.entry_id,
                "index": e.index,
                "title": e.title,
                "actor": e.actor,
                "category": e.category,
                "timestamp": e.timestamp,
                "summary": e.summary,
                "tags": e.tags,
                "settings": e.settings,
                "symbols": e.symbols,
                "files": e.files,
                "text": e.text,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_jsonld(entries: list[Entry], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    graph: list[dict] = []

    project_node = {
        "@id": "project:mastertrading",
        "@type": "Project",
        "name": "mastertrading",
        "hasEntry": [{"@id": e.entry_id} for e in entries],
    }
    graph.append(project_node)

    setting_ids: set[str] = set()
    symbol_ids: set[str] = set()
    file_ids: set[str] = set()

    for e in entries:
        entry_node = {
            "@id": e.entry_id,
            "@type": "ContextEntry",
            "title": e.title,
            "actor": e.actor,
            "category": e.category,
            "timestamp": e.timestamp,
            "summary": e.summary,
            "tags": e.tags,
            "mentionsSetting": [{"@id": safe_node_id("setting", s)} for s in e.settings],
            "mentionsSymbol": [{"@id": safe_node_id("symbol", s)} for s in e.symbols],
            "mentionsFile": [{"@id": safe_node_id("file", s)} for s in e.files],
        }
        graph.append(entry_node)
        for s in e.settings:
            setting_ids.add(s)
        for s in e.symbols:
            symbol_ids.add(s)
        for s in e.files:
            file_ids.add(s)

    for s in sorted(setting_ids):
        graph.append(
            {
                "@id": safe_node_id("setting", s),
                "@type": "Setting",
                "name": s,
            }
        )
    for s in sorted(symbol_ids):
        graph.append(
            {
                "@id": safe_node_id("symbol", s),
                "@type": "Symbol",
                "name": s,
            }
        )
    for s in sorted(file_ids):
        graph.append(
            {
                "@id": safe_node_id("file", s),
                "@type": "File",
                "path": s,
            }
        )

    payload = {
        "@context": {
            "@vocab": "https://example.org/mt#",
            "name": "http://schema.org/name",
            "path": "http://schema.org/path",
            "summary": "http://schema.org/description",
            "timestamp": "http://schema.org/dateCreated",
        },
        "@graph": graph,
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _graphml_key(key_id: str, attr_name: str, attr_type: str) -> str:
    return (
        f'  <key id="{xml_escape(key_id)}" for="node" '
        f'attr.name="{xml_escape(attr_name)}" attr.type="{xml_escape(attr_type)}"/>\n'
    )


def build_graphml(entries: list[Entry], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    nodes: dict[str, dict[str, str]] = {}
    edges: set[tuple[str, str, str]] = set()
    project_id = "project:mastertrading"
    nodes[project_id] = {"label": "mastertrading", "type": "Project", "actor": "", "timestamp": ""}

    for i, e in enumerate(entries):
        nodes[e.entry_id] = {
            "label": e.title,
            "type": "ContextEntry",
            "actor": e.actor,
            "timestamp": e.timestamp,
        }
        edges.add((project_id, e.entry_id, "HAS_ENTRY"))
        if i > 0:
            edges.add((entries[i - 1].entry_id, e.entry_id, "NEXT"))
        for s in e.settings:
            sid = safe_node_id("setting", s)
            nodes.setdefault(sid, {"label": s, "type": "Setting", "actor": "", "timestamp": ""})
            edges.add((e.entry_id, sid, "MENTIONS_SETTING"))
        for s in e.symbols:
            sid = safe_node_id("symbol", s)
            nodes.setdefault(sid, {"label": s, "type": "Symbol", "actor": "", "timestamp": ""})
            edges.add((e.entry_id, sid, "MENTIONS_SYMBOL"))
        for s in e.files:
            fid = safe_node_id("file", s)
            nodes.setdefault(fid, {"label": s, "type": "File", "actor": "", "timestamp": ""})
            edges.add((e.entry_id, fid, "MENTIONS_FILE"))

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>\n')
    lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
    lines.append(_graphml_key("d0", "label", "string"))
    lines.append(_graphml_key("d1", "type", "string"))
    lines.append(_graphml_key("d2", "actor", "string"))
    lines.append(_graphml_key("d3", "timestamp", "string"))
    lines.append('  <graph edgedefault="directed">\n')

    for node_id, data in nodes.items():
        lines.append(f'    <node id="{xml_escape(node_id)}">\n')
        lines.append(f'      <data key="d0">{xml_escape(data["label"])}</data>\n')
        lines.append(f'      <data key="d1">{xml_escape(data["type"])}</data>\n')
        lines.append(f'      <data key="d2">{xml_escape(data["actor"])}</data>\n')
        lines.append(f'      <data key="d3">{xml_escape(data["timestamp"])}</data>\n')
        lines.append("    </node>\n")

    for edge_idx, (source, target, label) in enumerate(sorted(edges), start=1):
        lines.append(
            f'    <edge id="e{edge_idx}" source="{xml_escape(source)}" target="{xml_escape(target)}">\n'
        )
        lines.append(f'      <data key="d0">{xml_escape(label)}</data>\n')
        lines.append("    </edge>\n")

    lines.append("  </graph>\n")
    lines.append("</graphml>\n")
    output_file.write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export messages.md knowledge into JSONL + JSON-LD + GraphML."
    )
    parser.add_argument(
        "--input",
        default=r"C:\Users\rortigoza\Documents\messages.md",
        help="Absolute path to messages markdown file.",
    )
    parser.add_argument(
        "--output-dir",
        default="knowledge",
        help="Output directory for generated files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    markdown = input_path.read_text(encoding="utf-8", errors="replace")
    entries = extract_entries(markdown)
    if not entries:
        raise ValueError("No entries detected in input markdown.")

    jsonl_path = output_dir / "messages_context.jsonl"
    jsonld_path = output_dir / "messages_knowledge.jsonld"
    graphml_path = output_dir / "messages_knowledge.graphml"

    build_jsonl(entries, jsonl_path)
    build_jsonld(entries, jsonld_path)
    build_graphml(entries, graphml_path)

    print(f"entries={len(entries)}")
    print(f"jsonl={jsonl_path}")
    print(f"jsonld={jsonld_path}")
    print(f"graphml={graphml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
