# TOON — Token Optimized Operational Notation
Version: 2026-03
Purpose: Deterministic AI-Readable Structured Context Format
Audience: LLM Systems + Quant Engineers

---

# 1. What is TOON

TOON = Token Optimized Operational Notation

It is a structured, low-narrative, high-signal format designed for:

- LLM ingestion
- Operational rule encoding
- Deterministic constraint declaration
- Quant system definition
- Token efficiency
- Reduced hallucination surface

TOON is NOT prose.
TOON is NOT documentation narrative.
TOON is NOT conceptual explanation.

TOON is structured operational context.

---

# 2. Design Principles

TOON follows these rules:

1. No paragraphs.
2. No storytelling.
3. No redundant explanation.
4. Use declarative blocks.
5. Use bounded numeric intervals.
6. Use explicit constraints.
7. Use deterministic mappings.
8. Use uppercase section headers.
9. Avoid natural language ambiguity.
10. Avoid emotional or persuasive phrasing.

---

# 3. Structural Rules

Every TOON document must contain:

HEADER
SECTION blocks
END marker

Structure template:

# NAME
FORMAT: TOON
MODE: TOKEN_OPTIMIZED
INTENT: short_description

============================================================
SECTION: SECTION_NAME
============================================================

KEY: VALUE
RULE: CONDITION → ACTION
BOUND: variable ∈ [min,max]
FORMULA: expression
NO:
ALWAYS:

============================================================
END_OF_TOON_CONTEXT
============================================================

---

# 4. Allowed Syntax Elements

Allowed constructs:

- KEY: VALUE
- IF condition → result
- VARIABLE ∈ [min,max]
- clamp(x,min,max)
- normalize(weights)
- NO:
- ALWAYS:
- INPUTS:
- OUTPUT:
- FORMULA:
- TARGET:
- ALERT IF:
- CAP:
- FAILSAFE:
- ACCEPT_IF:
- REJECT_IF:
- STEP:
- PHASE:

Not allowed:

- Narrative explanation
- Long paragraphs
- Marketing tone
- Subjective adjectives
- Metaphors
- Future speculation
- "We think"
- "This is better because"

---

# 5. Section Design Pattern

Common section types:

HARD_CONSTRAINTS
OBJECTIVE
REGIME_ENGINE
RISK_MODEL
TP_SL_MODEL
TRAILING_MODEL
META_ALLOCATOR
MONTE_CARLO
PORTFOLIO_RULES
MONITORING
DEPLOY_SEQUENCE
FAILSAFE
VALIDATION

Each section must:

- Be atomic.
- Be logically independent.
- Avoid cross-referencing prose.
- Avoid hidden assumptions.

---

# 6. Numeric Discipline Rules

All multipliers must be bounded:

Example:

risk_mult ∈ [0.50,1.00]
long_penalty ∈ [0.00,0.30]

Never allow:

unbounded growth
implicit scaling
dynamic base risk modification

---

# 7. Deterministic Mapping Rule

Every IF must have:

- clear input
- single output
- no ambiguity

Bad:

IF volatility high → reduce risk

Good:

IF ATR_PERCENTILE ≥ 80 → risk_mult=0.75

---

# 8. AI Safety Guard

TOON must encode:

- Non-negotiable constraints
- Fallback state
- Failure handling
- Risk caps
- Deployment order

LLM must not propose changes outside declared bounds.

---

# 9. Validation Rule for TOON Documents

A TOON document is valid if:

- No paragraph exceeds 3 lines.
- No section contains narrative explanation.
- All multipliers bounded.
- At least one HARD_CONSTRAINTS section exists.
- Has explicit END marker.
- No rhetorical sentences.

---

# 10. Token Optimization Guidelines

To reduce tokens:

- Prefer short variable names.
- Avoid repeated words.
- Use symbolic mapping.
- Use enumeration instead of prose.
- Remove articles (the, a, an).
- Remove filler transitions.

---

# 11. Example Minimal TOON Block

============================================================
SECTION: EXAMPLE
============================================================

INPUTS:
- ATR_PCTL
- TREND

IF TREND=bear AND ATR_PCTL≥70 → BEAR_EXPANSION
IF TREND=bull AND ATR_PCTL≤35 → BULL_COMPRESSION

risk_mult ∈ [0.50,1.00]
long_penalty ∈ [0.00,0.30]

NO:
- BASE_RISK_INCREASE

ALWAYS:
- DD_THROTTLE_APPLIED

============================================================
END_OF_TOON_CONTEXT
============================================================

---

# 12. When to Use TOON

Use TOON when:

- Encoding system rules
- Encoding quant models
- Feeding LLM persistent context
- Defining guardrails
- Defining regime maps
- Defining allocator logic
- Defining validation criteria

Do NOT use TOON for:

- Human training material
- Marketing documentation
- Tutorials
- Conceptual explanation

---

# 13. Summary

TOON is:

- Operational
- Deterministic
- Bounded
- Token-efficient
- AI-aligned
- Risk-aware

It reduces hallucination by:

- Eliminating narrative
- Encoding hard constraints
- Bounding parameters
- Removing ambiguity

---

END_OF_TOON_FORMAT_SPECIFICATION