from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from yt_dlp import YoutubeDL


TAG_PATTERNS: dict[str, list[str]] = {
    "smc_liquidity": [
        r"\bliquidez\b",
        r"\bliquidity\b",
        r"\bsweep\b",
        r"\bbarrid(?:o|a)s?\b",
        r"\bssl\b",
        r"\bbsl\b",
        r"\bstop\s*hunt\b",
        r"\btoma\s+de\s+liquidez\b",
    ],
    "market_structure": [
        r"\bestructura\b",
        r"\bbos\b",
        r"\bchoch\b",
        r"\btendencia\b",
        r"\bhigher\s+high\b",
        r"\blower\s+low\b",
        r"\bcambio\s+de\s+caracter\b",
    ],
    "fvg_orderblock": [
        r"\bfvg\b",
        r"\bfair\s+value\s+gap\b",
        r"\border\s*block\b",
        r"\borderblock\b",
        r"\bbloque\s+de\s+orden(?:es)?\b",
        r"\bimbalanc(?:e|es)\b",
    ],
    "session_open_wallstreet": [
        r"\bwall\s*street\b",
        r"\bapertura\b",
        r"\bsesion\b",
        r"\blondon\b",
        r"\bnew\s*york\b",
        r"\bny\b",
    ],
    "risk_management": [
        r"\briesgo\b",
        r"\bstop\s*loss\b",
        r"\btake\s*profit\b",
        r"\brisk\s*reward\b",
        r"\bdrawdown\b",
        r"\bgestion\s+de\s+riesgo\b",
    ],
    "macro_news": [
        r"\bcpi\b",
        r"\bnfp\b",
        r"\badp\b",
        r"\bpmi\b",
        r"\binflacion\b",
        r"\bfed\b",
        r"\btasa(?:s)?\b",
        r"\bempleo\b",
        r"\bnomina(?:s)?\b",
    ],
    "volatility_regime": [
        r"\bvolatil(?:idad)?\b",
        r"\brango\b",
        r"\bromp(?:e|io|imiento|imientos|iendo)?\b",
        r"\bimpulso\b",
        r"\bdisplacement\b",
        r"\bexpansion\b",
    ],
    "execution_timing": [
        r"\bentrada\b",
        r"\bconfirmacion\b",
        r"\besperar\b",
        r"\bretest\b",
        r"\brechazo\b",
        r"\btiming\b",
    ],
    "psychology": [
        r"\bpsicologia\b",
        r"\bdisciplina\b",
        r"\bemocion(?:es)?\b",
        r"\bestres\b",
        r"\bpaciencia\b",
    ],
    "funding_leverage": [
        r"\bapalanc(?:amiento|ado)?\b",
        r"\bfunding\b",
        r"\bfondeo\b",
    ],
}


@dataclass
class VideoCatalogRow:
    video_id: str
    title: str
    url: str
    upload_date: str
    upload_datetime_utc: str
    duration_sec: int
    view_count: int
    transcript_lang: str
    transcript_lines: int
    transcript_words: int
    tags: list[str]
    tag_counts: dict[str, int]
    btc_day_open: float | None
    btc_day_high: float | None
    btc_day_low: float | None
    btc_day_close: float | None
    btc_return_1d_pct: float | None
    btc_return_3d_pct: float | None


def _load_json(path: Path) -> Any:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return json.loads(raw.decode("utf-16", errors="ignore"))
    return json.loads(raw.decode("utf-8", errors="ignore"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean_vtt_to_lines(vtt_text: str) -> list[str]:
    lines: list[str] = []
    last = ""
    for raw in vtt_text.splitlines():
        txt = raw.strip()
        if not txt:
            continue
        if txt.startswith("WEBVTT"):
            continue
        if "-->" in txt:
            continue
        if txt.startswith("Kind:"):
            continue
        if txt.startswith("Language:"):
            continue
        if re.match(r"^\d+$", txt):
            continue
        txt = re.sub(r"<[^>]+>", "", txt)
        txt = txt.replace("&nbsp;", " ").strip()
        if not txt or txt == last:
            continue
        lines.append(txt)
        last = txt
    return lines


def _normalize_for_match(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _read_text_best_effort(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            decoded = raw.decode(enc)
            break
        except UnicodeDecodeError:
            decoded = ""
    else:
        decoded = raw.decode("latin-1", errors="ignore")

    if "Ã" in decoded or "Â" in decoded:
        try:
            repaired = decoded.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
            bad_src = decoded.count("Ã") + decoded.count("Â")
            bad_fix = repaired.count("Ã") + repaired.count("Â")
            if bad_fix < bad_src:
                decoded = repaired
        except UnicodeError:
            pass
    return decoded


def _extract_tag_counts(text: str) -> dict[str, int]:
    normalized = _normalize_for_match(text)
    out: dict[str, int] = {}
    for tag, patterns in TAG_PATTERNS.items():
        cnt = 0
        for pattern in patterns:
            cnt += len(re.findall(pattern, normalized, flags=re.IGNORECASE))
        out[tag] = cnt
    return out


def _select_tags(tag_counts: dict[str, int]) -> list[str]:
    ranked = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    return [tag for tag, cnt in ranked if cnt >= 2][:8]


def _dt_from_upload(upload_date: str | None, ts: int | None) -> datetime | None:
    if ts:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    if upload_date and len(upload_date) == 8:
        return datetime(
            int(upload_date[0:4]),
            int(upload_date[4:6]),
            int(upload_date[6:8]),
            tzinfo=timezone.utc,
        )
    return None


def _fetch_btc_daily_map(min_d: date, max_d: date) -> dict[date, dict[str, float]]:
    start = datetime(min_d.year, min_d.month, min_d.day, tzinfo=timezone.utc) - timedelta(days=2)
    end = datetime(max_d.year, max_d.month, max_d.day, tzinfo=timezone.utc) + timedelta(days=4)
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 1000,
    }
    resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=30)
    resp.raise_for_status()
    rows = resp.json()
    out: dict[date, dict[str, float]] = {}
    for r in rows:
        d = datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).date()
        out[d] = {
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
        }
    return out


def _btc_context_for_day(d: date, btc_map: dict[date, dict[str, float]]) -> dict[str, float | None]:
    row = btc_map.get(d)
    if not row:
        return {
            "day_open": None,
            "day_high": None,
            "day_low": None,
            "day_close": None,
            "ret_1d_pct": None,
            "ret_3d_pct": None,
        }
    close_0 = row["close"]
    row_1 = btc_map.get(d + timedelta(days=1))
    row_3 = btc_map.get(d + timedelta(days=3))
    ret_1 = ((row_1["close"] - close_0) / close_0 * 100.0) if row_1 else None
    ret_3 = ((row_3["close"] - close_0) / close_0 * 100.0) if row_3 else None
    return {
        "day_open": row["open"],
        "day_high": row["high"],
        "day_low": row["low"],
        "day_close": row["close"],
        "ret_1d_pct": ret_1,
        "ret_3d_pct": ret_3,
    }


def _find_best_vtt_file(raw_dir: Path, video_id: str) -> tuple[Path | None, str]:
    files = sorted(raw_dir.glob(f"{video_id}*.vtt"))
    if not files:
        return None, ""
    preferred = [f"{video_id}.es-orig.vtt", f"{video_id}.es.vtt"]
    by_name = {f.name: f for f in files}
    for name in preferred:
        if name in by_name:
            lang = "es-orig" if name.endswith(".es-orig.vtt") else "es"
            return by_name[name], lang
    sel = files[0]
    lang_match = re.search(rf"{re.escape(video_id)}\.(.+?)\.vtt$", sel.name)
    lang = lang_match.group(1) if lang_match else "unknown"
    return sel, lang


def build_catalog(channel_url: str, out_dir: Path, max_videos: int = 0) -> None:
    metadata_dir = out_dir / "metadata"
    raw_dir = out_dir / "transcripts_raw"
    clean_dir = out_dir / "transcripts_clean"
    report_dir = out_dir / "catalog"
    for d in (metadata_dir, raw_dir, clean_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    ydl_base_opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    with YoutubeDL({**ydl_base_opts, "extract_flat": True}) as ydl:
        channel_info = ydl.extract_info(channel_url, download=False)

    entries = channel_info.get("entries") or []
    if max_videos > 0:
        entries = entries[:max_videos]
    ids = [str(e.get("id")) for e in entries if e and e.get("id")]
    if not ids:
        raise RuntimeError("No se encontraron videos en el canal.")

    videos_meta: list[dict[str, Any]] = []
    all_dates: list[date] = []

    # 1) metadata + subtitles per video
    for idx, vid in enumerate(ids, start=1):
        url = f"https://www.youtube.com/watch?v={vid}"
        meta_path = metadata_dir / f"{vid}.json"
        if meta_path.exists():
            info = _load_json(meta_path)
            print(f"[{idx}/{len(ids)}] metadata {vid} (cached)")
        else:
            print(f"[{idx}/{len(ids)}] metadata {vid}")
            with YoutubeDL(ydl_base_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            if not info:
                continue
            _save_json(meta_path, info)
        videos_meta.append(info)

        ud = info.get("upload_date")
        ts = info.get("timestamp")
        dtu = _dt_from_upload(ud, ts)
        if dtu:
            all_dates.append(dtu.date())

        if any(raw_dir.glob(f"{vid}*.vtt")):
            print(f"[{idx}/{len(ids)}] subtitles {vid} (cached)")
        else:
            print(f"[{idx}/{len(ids)}] subtitles {vid}")
            sub_opts = {
                **ydl_base_opts,
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["es.*", "es", "en.*", "en"],
                "subtitlesformat": "vtt",
                "outtmpl": str(raw_dir / "%(id)s.%(ext)s"),
            }
            try:
                with YoutubeDL(sub_opts) as ydl:
                    ydl.extract_info(url, download=True)
            except Exception:
                pass

    if not videos_meta:
        raise RuntimeError("No se pudo extraer metadata de videos.")

    min_d = min(all_dates) if all_dates else date.today()
    max_d = max(all_dates) if all_dates else date.today()
    btc_map = _fetch_btc_daily_map(min_d, max_d)

    rows: list[VideoCatalogRow] = []
    global_tag_counter: Counter[str] = Counter()
    monthly_tag_counter: dict[str, Counter[str]] = defaultdict(Counter)

    # 2) clean transcript + tags + context
    for info in videos_meta:
        vid = str(info.get("id") or "")
        title = str(info.get("title") or "")
        url = str(info.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}")
        dtu = _dt_from_upload(info.get("upload_date"), info.get("timestamp"))
        upload_iso = dtu.date().isoformat() if dtu else ""
        upload_dt_iso = dtu.isoformat() if dtu else ""
        duration = int(info.get("duration") or 0)
        views = int(info.get("view_count") or 0)

        vtt_file, transcript_lang = _find_best_vtt_file(raw_dir, vid)
        clean_lines: list[str] = []
        if vtt_file and vtt_file.exists():
            vtt_text = _read_text_best_effort(vtt_file)
            clean_lines = _clean_vtt_to_lines(vtt_text)
            (clean_dir / f"{vid}.txt").write_text("\n".join(clean_lines), encoding="utf-8")

        full_text = "\n".join(clean_lines)
        words = len(re.findall(r"\w+", full_text, flags=re.UNICODE))
        tag_counts = _extract_tag_counts(full_text) if full_text else {}
        tags = _select_tags(tag_counts) if tag_counts else []

        for t in tags:
            global_tag_counter[t] += 1
            if upload_iso:
                monthly_tag_counter[upload_iso[:7]][t] += 1

        btc_ctx = _btc_context_for_day(dtu.date(), btc_map) if dtu else {
            "day_open": None,
            "day_high": None,
            "day_low": None,
            "day_close": None,
            "ret_1d_pct": None,
            "ret_3d_pct": None,
        }

        rows.append(
            VideoCatalogRow(
                video_id=vid,
                title=title,
                url=url,
                upload_date=upload_iso,
                upload_datetime_utc=upload_dt_iso,
                duration_sec=duration,
                view_count=views,
                transcript_lang=transcript_lang,
                transcript_lines=len(clean_lines),
                transcript_words=words,
                tags=tags,
                tag_counts=tag_counts,
                btc_day_open=btc_ctx["day_open"],
                btc_day_high=btc_ctx["day_high"],
                btc_day_low=btc_ctx["day_low"],
                btc_day_close=btc_ctx["day_close"],
                btc_return_1d_pct=btc_ctx["ret_1d_pct"],
                btc_return_3d_pct=btc_ctx["ret_3d_pct"],
            )
        )

    # 3) write catalog files
    rows_sorted = sorted(rows, key=lambda r: (r.upload_date, r.video_id))
    csv_path = report_dir / "videos_catalog.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "video_id",
                "upload_date",
                "upload_datetime_utc",
                "title",
                "url",
                "duration_sec",
                "view_count",
                "transcript_lang",
                "transcript_lines",
                "transcript_words",
                "tags",
                "tag_counts_json",
                "btc_day_open",
                "btc_day_high",
                "btc_day_low",
                "btc_day_close",
                "btc_return_1d_pct",
                "btc_return_3d_pct",
            ]
        )
        for r in rows_sorted:
            writer.writerow(
                [
                    r.video_id,
                    r.upload_date,
                    r.upload_datetime_utc,
                    r.title,
                    r.url,
                    r.duration_sec,
                    r.view_count,
                    r.transcript_lang,
                    r.transcript_lines,
                    r.transcript_words,
                    ";".join(r.tags),
                    json.dumps(r.tag_counts, ensure_ascii=False),
                    r.btc_day_open,
                    r.btc_day_high,
                    r.btc_day_low,
                    r.btc_day_close,
                    r.btc_return_1d_pct,
                    r.btc_return_3d_pct,
                ]
            )

    json_path = report_dir / "videos_catalog.json"
    _save_json(json_path, [r.__dict__ for r in rows_sorted])

    # 4) aggregate markdown report
    report_lines: list[str] = []
    report_lines.append("# Alvaro Smart Money - Catalogo y Aprendizajes")
    report_lines.append("")
    report_lines.append(f"- Canal: {channel_url}")
    report_lines.append(f"- Videos procesados: {len(rows_sorted)}")
    if rows_sorted:
        report_lines.append(f"- Rango fechas: {rows_sorted[0].upload_date} -> {rows_sorted[-1].upload_date}")
    report_lines.append("")
    videos_with_transcript = sum(1 for r in rows_sorted if r.transcript_words > 0)
    coverage_pct = ((videos_with_transcript / len(rows_sorted)) * 100.0) if rows_sorted else 0.0
    report_lines.append(f"- Cobertura transcripcion: {videos_with_transcript}/{len(rows_sorted)} ({coverage_pct:.1f}%)")
    report_lines.append("")
    report_lines.append("## Tags mas frecuentes")
    for tag, cnt in global_tag_counter.most_common(12):
        report_lines.append(f"- {tag}: {cnt}")
    report_lines.append("")
    report_lines.append("## Evolucion mensual (tags top)")
    for month in sorted(monthly_tag_counter.keys()):
        top = ", ".join([f"{t}:{c}" for t, c in monthly_tag_counter[month].most_common(6)])
        report_lines.append(f"- {month}: {top}")
    report_lines.append("")
    report_lines.append("## Observaciones contextuales")
    report_lines.append("- El catalogo incluye contexto BTC por fecha de publicacion (OHLC del dia y retorno +1d/+3d).")
    report_lines.append("- Interpretar cada video en su contexto temporal evita sobre-ajustar reglas a un unico regimen.")
    report_lines.append("")
    report_lines.append("## Reglas reutilizables sugeridas")
    report_lines.append("- Priorizar confirmacion tras toma de liquidez y evitar anticipar suelo/techo.")
    report_lines.append("- En desplazamientos fuertes, esperar retest antes de nuevas entradas en la misma direccion.")
    report_lines.append("- Ajustar agresividad de TP/trailing cuando ATR esta en regime alto.")
    report_lines.append("- Combinar sesgo estructural (trend) con filtro de ejecucion (spread/volumen/cooldown).")
    report_lines.append("")
    report_lines.append("## Timeline por video (fecha + contexto)")
    for r in rows_sorted:
        tags_text = ", ".join(r.tags[:4]) if r.tags else "-"
        r1 = f"{r.btc_return_1d_pct:.2f}%" if r.btc_return_1d_pct is not None else "n/a"
        r3 = f"{r.btc_return_3d_pct:.2f}%" if r.btc_return_3d_pct is not None else "n/a"
        report_lines.append(
            f"- {r.upload_date} | {r.title} | tags: {tags_text} | BTC+1d: {r1} | BTC+3d: {r3}"
        )

    md_path = report_dir / "learning_report.md"
    md_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "videos_total": len(rows_sorted),
        "date_from": rows_sorted[0].upload_date if rows_sorted else "",
        "date_to": rows_sorted[-1].upload_date if rows_sorted else "",
        "videos_with_transcript": videos_with_transcript,
        "transcript_coverage_pct": round(coverage_pct, 2),
        "top_tags": global_tag_counter.most_common(12),
        "catalog_csv": str(csv_path),
        "catalog_json": str(json_path),
        "learning_report": str(md_path),
        "transcripts_clean_dir": str(clean_dir),
    }
    _save_json(report_dir / "summary.json", summary)
    print("Done:", json.dumps(summary, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-url", type=str, default="https://www.youtube.com/@AlvaroSmartMoney/videos")
    parser.add_argument("--out-dir", type=str, default="tmp/alvaro_smc")
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    build_catalog(
        channel_url=args.channel_url,
        out_dir=Path(args.out_dir),
        max_videos=max(0, int(args.max_videos)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
