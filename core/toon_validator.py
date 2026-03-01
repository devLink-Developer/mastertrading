from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToonValidationResult:
    path: str
    valid: bool
    errors: list[str]


def _is_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and set(s) <= {"=", "-", "_"}


def _is_narrative_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _is_separator(s):
        return False
    if s.startswith("#"):
        return False
    if s.startswith("- "):
        return False
    if s.startswith("IF ") or s.startswith("ELSE"):
        return False
    if "=" in s:
        return False
    if ":" in s:
        return False
    if "->" in s or "→" in s or "∈" in s:
        return False
    if "_" in s and len(s.split()) <= 8:
        return False
    if s.upper() == s and len(s.split()) <= 8:
        return False
    return True


def validate_toon_text(text: str, *, path: str = "<memory>") -> ToonValidationResult:
    errors: list[str] = []
    lines = [ln.rstrip("\n") for ln in str(text or "").splitlines()]
    joined = "\n".join(lines)

    if "FORMAT: TOON" not in joined:
        errors.append("missing FORMAT: TOON")
    if "END_OF_TOON_CONTEXT" not in joined:
        errors.append("missing END_OF_TOON_CONTEXT")
    if "SECTION: HARD_CONSTRAINTS" not in joined:
        errors.append("missing SECTION: HARD_CONSTRAINTS")

    # Enforce "no paragraph > 3 lines": consecutive narrative-like lines >3.
    run = 0
    max_run = 0
    for line in lines:
        if _is_narrative_line(line):
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    if max_run > 3:
        errors.append(f"narrative block too long (max consecutive narrative lines={max_run})")

    return ToonValidationResult(path=path, valid=(len(errors) == 0), errors=errors)


def validate_toon_file(path: Path) -> ToonValidationResult:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return ToonValidationResult(path=str(path), valid=False, errors=[f"read_error:{type(exc).__name__}"])
    return validate_toon_text(content, path=str(path))
