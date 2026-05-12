---
type: canonical
source: none
sync: none
sla: none
---

# Fallax Website Redesign — Design Spec

**Date:** 2026-05-12

---

## Goal

Replace the current `website/index.html` with a Research Terminal redesign: monospace font,
dark background, pure monochrome (no color accent), narrow centered column. The aesthetic is
a terminal session or research paper landing page — maximum credibility, minimum decoration.

---

## Design Language

| Property | Value |
|---|---|
| Font | `'Courier New', Courier, monospace` (system stack, no web font load) |
| Background | `#0a0a0a` |
| Primary text | `#e8e8e8` |
| Secondary text | `#888` |
| Muted / labels | `#555` |
| Dim / comments | `#333` |
| Borders | `#1e1e1e` (hairline), `#2a2a2a` (interactive) |
| Accent color | **None** — pure monochrome |
| Max content width | `640px`, centered |
| Font sizes | `14px` base; section labels `.65rem`; body `.78–.85rem`; h1 `1.75rem` |

---

## Layout Structure

Single-page, single column. All sections share the same `640px` centered container.
Sections are divided by `1px solid #1e1e1e` horizontal rules (`border-bottom` on each section).

```
┌─────────────────────────────────────┐
│  header (sticky)                    │
│  fallax                    docs github MIT │
├─────────────────────────────────────┤
│  hero                               │
│  LLM REASONING BENCHMARK            │
│  Adversarial Reasoning              │
│  Evaluation for LLMs                │
│  [meta row: 25 templates · 100 …]   │
│  [$ uv sync … ▌ click to copy]      │
├─────────────────────────────────────┤
│  BENCHMARK V1 RESULTS ──────────    │
│  model | score | fail% | captured   │
│  claude-sonnet-4-6  —   —   —       │
│  gpt-4o-mini        —   —   —       │
│  (pending note)                     │
├─────────────────────────────────────┤
│  FAILURE TAXONOMY ──────────────    │
│  logic_error    │ assumption_error  │
│  constraint_… │ generalization_…   │
│  ambiguity_…  │ multi_step_break   │
├─────────────────────────────────────┤
│  QUICK START ───────────────────    │
│  # install                          │
│  uv sync                            │
│  # run evaluation                   │
│  uv run python -m fallax run …      │
│  # benchmark                        │
│  uv run python -m fallax baseline … │
├─────────────────────────────────────┤
│  footer                             │
│  fallax — github.com/… — MIT  links │
└─────────────────────────────────────┘
```

---

## Sections

### Header (sticky)

- `position: sticky; top: 0` with `backdrop-filter: blur(6px)` and `background: rgba(10,10,10,.95)`
- Left: `fallax` in bold `.85rem`
- Right nav: `docs`, `github` links (`.75rem`, color `#555`) + `MIT` tag (border `#2a2a2a`)
- Bottom border: `1px solid #1e1e1e`

### Hero

- Eyebrow: `LLM REASONING BENCHMARK` in `.7rem` uppercase, letter-spacing `.14em`, color `#555`
- `h1`: `1.75rem`, bold, color `#e8e8e8`, line-height `1.2`
  - Text: "Adversarial Reasoning\nEvaluation for LLMs"
- Description paragraph: `.85rem`, color `#888`, max-width `520px`
  - "Fallax surfaces failure modes that single-turn benchmarks miss. Step-level correctness
    scoring across 25 adversarial prompt templates — not just final-answer accuracy."
- Meta row: `25 templates · 100 benchmark prompts · 6 failure categories · Python 3.12+ · MIT`
  - Each item `.72rem`, label `#555`, value `#888`
- Install command block: full-width, `background: #111`, `border: 1px solid #2a2a2a`
  - Shows: `$ uv sync && uv run python -m fallax --help`
  - Right-aligned "click to copy" hint in `#333`
  - JS copies text to clipboard on click; hint briefly changes to "copied!"

### Benchmark v1 Results

- Section label: `BENCHMARK V1 RESULTS` with hairline extending to right edge
- Table: `width: 100%`, columns: Model | Overall Score | Failure Rate | Captured
- Header row: `.65rem` uppercase, color `#555`, `font-weight: 400`
- Data rows: model name `#e8e8e8`; score/rate/captured `#555` (dashes until populated)
- Pending note below table: `.68rem`, italic, `#333`
  - "Baselines pending — run `fallax baseline capture --version v1` to populate."
- When baselines are captured: replace `—` with real values; remove pending note

### Failure Taxonomy

- Section label: `FAILURE TAXONOMY`
- 2-column grid, 6 cells (one per failure category)
- Each cell: category name in `#888`, sub-types in `#333` `.65rem`
- Odd cells: `border-right: 1px solid #1e1e1e`, `padding-right: 1.5rem`
- Even cells: `padding-left: 1.5rem`
- All cells: `border-bottom: 1px solid #1e1e1e`

Content:

| Category | Types |
|---|---|
| `logic_error` | `contradiction · invalid_inference` |
| `assumption_error` | `unstated_assumption · unjustified_assumption` |
| `constraint_violation` | `ignored_constraint · partial_satisfaction` |
| `generalization_error` | `overgeneralization · pattern_misapplication` |
| `ambiguity_failure` | `ambiguity_failure` |
| `multi_step_break` | `multi_step_break` |

### Quick Start

- Section label: `QUICK START`
- Single code block: `background: #111`, `border: 1px solid #2a2a2a`, `padding: 1.25rem`
- `.78rem` monospace, color `#888`, line-height `1.8`
- Comments in `#333`

```
# install
uv sync

# run evaluation
uv run python -m fallax run \
  --models claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001 \
  --output results.jsonl

# benchmark against v1
uv run python -m fallax baseline capture \
  --version v1 \
  --model claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001

# analyze results
uv run python -m fallax analyze results.jsonl
```

### Footer

- `padding: 1.75rem 0`, `border-top: 1px solid #1e1e1e`
- Left: `fallax — github.com/alawein/fallax — MIT` in `.72rem`, color `#555`
- Right: `github`, `meshal.ai`, `contact` links in `.72rem`, color `#333`

---

## Performance

- No web fonts loaded — system monospace stack only
- All CSS inlined in `<style>` tag — zero external stylesheets
- No JavaScript frameworks — one small inline `<script>` for copy-to-clipboard only
- No images except favicon (existing)
- Total page size target: under 8 KB

---

## Interaction

- **Copy button:** clicking the install command copies `uv sync && uv run python -m fallax --help`
  to clipboard. The "click to copy" hint briefly reads "copied!" then reverts after 1.5s.
  Implemented with ~5 lines of vanilla JS inline at bottom of `<body>`.
- **Hover states:** nav links `#555 → #888`; footer links `#333 → #555`
- No animations, no scroll effects, no transitions beyond `color .15s`

---

## Meta / SEO

Preserve existing tags verbatim:
- `<title>`, `<meta name="description">`, Open Graph, Twitter Card, canonical link

Update Python badge from `3.10+` to `3.12+` in the `<title>` and any badge references.

---

## Files Changed

| File | Change |
|---|---|
| `website/index.html` | Full rewrite — all CSS inline, single `<script>` block |

No other files change. The `website/vercel.json` (if present) stays as-is.

---

## Out of Scope

- Dashboard frontend (separate spec)
- Any backend changes
- Dark/light mode toggle
- Animation or scroll-triggered effects
- External CSS frameworks
