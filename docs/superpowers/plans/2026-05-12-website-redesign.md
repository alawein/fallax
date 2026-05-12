---
type: canonical
source: none
sync: none
sla: none
---

# Fallax Website Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `website/index.html` with a Research Terminal redesign — monospace system font, pure monochrome dark palette, 640 px centered column, no external dependencies.

**Architecture:** Single HTML file, all CSS inlined in `<style>`, one `<script>` block for copy-to-clipboard. No build step, no bundler, no web fonts. Each task adds one section to the file and commits. Tasks are sequential — each builds on the previous file state.

**Tech Stack:** HTML5, inline CSS (CSS custom properties), ~10 lines vanilla JS

---

## File Map

| Action | Path | What changes |
|---|---|---|
| Full rewrite | `website/index.html` | Replaces minified single-line CSS with readable structured markup |

---

## Task 1: Skeleton, CSS, and sticky header

This task writes the complete file from scratch — doctype, head, all CSS custom properties and rules, and the header element. Subsequent tasks only add sections inside `<main>`.

**Files:**
- Rewrite: `website/index.html`

- [ ] **Step 1: Verify the current file opens in a browser**

Open `website/index.html` in your browser (drag the file into a new tab, or use `start website/index.html` in PowerShell from the repo root). Note what you see — this is the before state.

- [ ] **Step 2: Write the skeleton with all CSS and the header**

Replace the entire contents of `website/index.html` with:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="icon" href="/favicon.ico" sizes="32x32">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="manifest" href="/site.webmanifest">
<title>Fallax — Multi-Step Reasoning Evaluation for LLMs</title>
<meta name="description" content="Benchmark suite for evaluating language models on structured, multi-step reasoning tasks — surfacing failure modes that single-turn benchmarks miss.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://fallax.online/">
<meta property="og:title" content="Fallax — Multi-Step Reasoning Evaluation for LLMs">
<meta property="og:description" content="Benchmark suite for evaluating language models on structured, multi-step reasoning tasks — surfacing failure modes that single-turn benchmarks miss.">
<meta property="og:image" content="https://fallax.online/og-image.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Fallax — Multi-Step Reasoning Evaluation for LLMs">
<meta name="twitter:description" content="Benchmark suite for evaluating language models on structured, multi-step reasoning tasks — surfacing failure modes that single-turn benchmarks miss.">
<meta name="twitter:image" content="https://fallax.online/og-image.png">
<link rel="canonical" href="https://fallax.online/">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0a0a;--bg2:#111;
  --border:#1e1e1e;--border2:#2a2a2a;
  --t1:#e8e8e8;--t2:#888;--t3:#555;--t4:#333;
  --font:'Courier New',Courier,monospace;
  --max:640px
}
html{background:var(--bg);color:var(--t1);font-family:var(--font);font-size:14px;line-height:1.6}
.wrap{max-width:var(--max);margin:0 auto;padding:0 1.5rem}

/* header */
header{position:sticky;top:0;z-index:10;background:rgba(10,10,10,.95);backdrop-filter:blur(6px);border-bottom:1px solid var(--border);padding:.75rem 0}
header .wrap{display:flex;justify-content:space-between;align-items:center}
.logo{font-size:.85rem;font-weight:700;color:var(--t1);text-decoration:none}
nav{display:flex;gap:1.5rem;align-items:center}
nav a{font-size:.75rem;color:var(--t3);text-decoration:none;transition:color .15s}
nav a:hover{color:var(--t2)}
.tag{font-size:.65rem;color:var(--t4);border:1px solid var(--border2);padding:.1rem .4rem}

/* hero */
.hero{padding:4rem 0 2.5rem;border-bottom:1px solid var(--border)}
.eyebrow{font-size:.7rem;letter-spacing:.14em;color:var(--t3);text-transform:uppercase;margin-bottom:.75rem}
h1{font-size:1.75rem;font-weight:700;color:var(--t1);line-height:1.2;margin-bottom:.75rem}
.hero-desc{font-size:.85rem;color:var(--t2);line-height:1.7;max-width:520px;margin-bottom:1.5rem}
.meta-row{display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:1.5rem}
.meta-item{font-size:.72rem;color:var(--t3)}
.meta-item span{color:var(--t2)}
.cmd{display:flex;align-items:center;gap:.75rem;background:var(--bg2);border:1px solid var(--border2);padding:.6rem 1rem;font-size:.8rem;color:var(--t2);cursor:pointer;transition:border-color .15s;width:100%;font-family:var(--font);text-align:left}
.cmd:hover{border-color:#3a3a3a}
.prompt{color:var(--t4)}
.copy-hint{margin-left:auto;font-size:.65rem;color:var(--t4)}

/* sections */
.section{padding:2.5rem 0;border-bottom:1px solid var(--border)}
.section-label{font-size:.65rem;letter-spacing:.14em;color:var(--t3);text-transform:uppercase;margin-bottom:1.25rem;display:flex;align-items:center;gap:.75rem}
.section-label::after{content:'';flex:1;height:1px;background:var(--border)}

/* results table */
.results-table{width:100%;border-collapse:collapse;font-size:.78rem}
.results-table th{text-align:left;color:var(--t3);font-weight:400;font-size:.65rem;letter-spacing:.08em;padding:0 0 .6rem;border-bottom:1px solid var(--border2)}
.results-table th:not(:first-child){text-align:right}
.results-table td{padding:.55rem 0;border-bottom:1px solid var(--border);color:var(--t2)}
.results-table td:not(:first-child){text-align:right;color:var(--t3)}
.results-table td:first-child{color:var(--t1);font-size:.8rem}
.pending-note{font-size:.68rem;color:var(--t4);margin-top:.75rem;font-style:italic}

/* taxonomy */
.taxonomy-grid{display:grid;grid-template-columns:1fr 1fr}
.tax-item{padding:.7rem 0;border-bottom:1px solid var(--border);font-size:.78rem}
.tax-item:nth-child(odd){padding-right:1.5rem;border-right:1px solid var(--border)}
.tax-item:nth-child(even){padding-left:1.5rem}
.tax-name{color:var(--t2);margin-bottom:.2rem}
.tax-types{font-size:.65rem;color:var(--t4);line-height:1.6}

/* code block */
.code-block{background:var(--bg2);border:1px solid var(--border2);padding:1.25rem;font-size:.78rem;color:var(--t2);overflow-x:auto;line-height:1.8;white-space:pre}
.comment{color:var(--t4)}

/* footer */
footer{padding:1.75rem 0}
.footer-row{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem}
.footer-left{font-size:.72rem;color:var(--t3)}
.footer-links{display:flex;gap:1rem}
.footer-links a{font-size:.72rem;color:var(--t4);text-decoration:none;transition:color .15s}
.footer-links a:hover{color:var(--t3)}

/* mobile */
@media(max-width:600px){h1{font-size:1.3rem}.meta-row{gap:.75rem}nav{gap:1rem}}
</style>
</head>
<body>

<header>
  <div class="wrap">
    <a class="logo" href="/">fallax</a>
    <nav>
      <a href="https://github.com/alawein/fallax#installation">docs</a>
      <a href="https://github.com/alawein/fallax">github</a>
      <span class="tag">MIT</span>
    </nav>
  </div>
</header>

<main>
  <div class="wrap">
    <!-- sections go here -->
  </div>
</main>

<footer>
  <!-- added in Task 6 -->
</footer>

</body>
</html>
```

- [ ] **Step 3: Open in browser and verify the header**

Open `website/index.html` in your browser. You should see:
- Black background, monospace text
- Sticky header with `fallax` on left, `docs github MIT` on right
- Header sticks when you scroll (nothing to scroll yet, but the structure is there)
- No content below the header yet — that's expected

If the background is white or the font isn't monospace, check the `<style>` tag was written correctly.

- [ ] **Step 4: Commit**

```bash
git add website/index.html
git commit -m "feat(website): scaffold research terminal redesign — skeleton, CSS, header"
```

---

## Task 2: Hero section with install command

**Files:**
- Modify: `website/index.html`

- [ ] **Step 1: Replace the `<!-- sections go here -->` comment with the hero HTML**

Find this line in `website/index.html`:

```html
    <!-- sections go here -->
```

Replace it with:

```html
    <section class="hero">
      <div class="eyebrow">LLM Reasoning Benchmark</div>
      <h1>Adversarial Reasoning<br>Evaluation for LLMs</h1>
      <p class="hero-desc">Fallax surfaces failure modes that single-turn benchmarks miss. Step-level correctness scoring across 25 adversarial prompt templates &mdash; not just final-answer accuracy.</p>
      <div class="meta-row">
        <div class="meta-item"><span>25</span> templates</div>
        <div class="meta-item"><span>100</span> benchmark prompts</div>
        <div class="meta-item"><span>6</span> failure categories</div>
        <div class="meta-item"><span>Python 3.12+</span></div>
        <div class="meta-item"><span>MIT</span></div>
      </div>
      <button class="cmd" id="install-cmd" onclick="copyInstall()">
        <span class="prompt">$</span>
        <span>uv sync &amp;&amp; uv run python -m fallax --help</span>
        <span class="copy-hint" id="copy-hint">click to copy</span>
      </button>
    </section>

    <!-- results, taxonomy, quickstart go here -->
```

- [ ] **Step 2: Verify in browser**

Reload `website/index.html`. You should see:
- Eyebrow text `LLM REASONING BENCHMARK` in small grey uppercase
- Large bold heading "Adversarial Reasoning / Evaluation for LLMs"
- Grey description paragraph
- Meta row with 5 items (25 templates · 100 benchmark prompts · …)
- A full-width dark install command box with `$ uv sync …` and `click to copy` on the right
- The install box border should lighten slightly on hover

- [ ] **Step 3: Commit**

```bash
git add website/index.html
git commit -m "feat(website): add hero section with meta row and install command"
```

---

## Task 3: Benchmark v1 results table

**Files:**
- Modify: `website/index.html`

- [ ] **Step 1: Replace the results comment with the results section**

Find this line:

```html
    <!-- results, taxonomy, quickstart go here -->
```

Replace it with:

```html
    <section class="section">
      <div class="section-label">Benchmark v1 Results</div>
      <table class="results-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Overall Score</th>
            <th>Failure Rate</th>
            <th>Captured</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>claude-sonnet-4-6</td>
            <td>&mdash;</td>
            <td>&mdash;</td>
            <td>&mdash;</td>
          </tr>
          <tr>
            <td>gpt-4o-mini</td>
            <td>&mdash;</td>
            <td>&mdash;</td>
            <td>&mdash;</td>
          </tr>
        </tbody>
      </table>
      <p class="pending-note">Baselines pending &mdash; run <code>fallax baseline capture --version v1</code> to populate.</p>
    </section>

    <!-- taxonomy, quickstart go here -->
```

- [ ] **Step 2: Verify in browser**

Reload. Scroll below the hero. You should see:
- `BENCHMARK V1 RESULTS` label with a hairline extending to the right edge
- Table with four columns: Model, Overall Score, Failure Rate, Captured
- Two data rows (`claude-sonnet-4-6`, `gpt-4o-mini`), all values showing `—`
- Italic pending note below the table in very dim grey
- Model names in bright `#e8e8e8`; score/rate/captured values in dim `#555`

- [ ] **Step 3: Commit**

```bash
git add website/index.html
git commit -m "feat(website): add benchmark v1 results table with pending dashes"
```

---

## Task 4: Failure taxonomy grid

**Files:**
- Modify: `website/index.html`

- [ ] **Step 1: Replace the taxonomy comment with the taxonomy section**

Find this line:

```html
    <!-- taxonomy, quickstart go here -->
```

Replace it with:

```html
    <section class="section">
      <div class="section-label">Failure Taxonomy</div>
      <div class="taxonomy-grid">
        <div class="tax-item">
          <div class="tax-name">logic_error</div>
          <div class="tax-types">contradiction &middot; invalid_inference</div>
        </div>
        <div class="tax-item">
          <div class="tax-name">assumption_error</div>
          <div class="tax-types">unstated_assumption &middot; unjustified_assumption</div>
        </div>
        <div class="tax-item">
          <div class="tax-name">constraint_violation</div>
          <div class="tax-types">ignored_constraint &middot; partial_satisfaction</div>
        </div>
        <div class="tax-item">
          <div class="tax-name">generalization_error</div>
          <div class="tax-types">overgeneralization &middot; pattern_misapplication</div>
        </div>
        <div class="tax-item">
          <div class="tax-name">ambiguity_failure</div>
          <div class="tax-types">ambiguity_failure</div>
        </div>
        <div class="tax-item">
          <div class="tax-name">multi_step_break</div>
          <div class="tax-types">multi_step_break</div>
        </div>
      </div>
    </section>

    <!-- quickstart goes here -->
```

- [ ] **Step 2: Verify in browser**

Reload. Below the results table you should see:
- `FAILURE TAXONOMY` label with hairline
- 2-column grid with 6 cells (3 rows × 2 cols)
- Odd cells have a right border dividing the two columns
- Category names in `#888`; sub-type names in dim `#333`
- Each cell has a bottom border hairline

- [ ] **Step 3: Commit**

```bash
git add website/index.html
git commit -m "feat(website): add failure taxonomy 2-column grid"
```

---

## Task 5: Quick start code block

**Files:**
- Modify: `website/index.html`

- [ ] **Step 1: Replace the quickstart comment with the quick start section**

Find this line:

```html
    <!-- quickstart goes here -->
```

Replace it with:

```html
    <section class="section">
      <div class="section-label">Quick Start</div>
      <div class="code-block"><span class="comment"># install</span>
uv sync

<span class="comment"># run evaluation</span>
uv run python -m fallax run \
  --models claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001 \
  --output results.jsonl

<span class="comment"># benchmark against v1</span>
uv run python -m fallax baseline capture \
  --version v1 \
  --model claude-sonnet-4-6 \
  --judge claude-haiku-4-5-20251001

<span class="comment"># analyze results</span>
uv run python -m fallax analyze results.jsonl</div>
    </section>
```

Note: The content inside `.code-block` must start immediately after the opening `<div class="code-block">` with no leading whitespace, because `white-space:pre` renders all whitespace literally.

- [ ] **Step 2: Verify in browser**

Reload. Below the taxonomy you should see:
- `QUICK START` label with hairline
- Dark box (`#111` background, dim border) containing the CLI commands
- Comments (`# install`, `# run evaluation`, etc.) in very dim `#333`
- Command text in `#888`
- Lines preserved exactly as written (no wrapping unless window is narrow)

- [ ] **Step 3: Commit**

```bash
git add website/index.html
git commit -m "feat(website): add quick start code block"
```

---

## Task 6: Footer

**Files:**
- Modify: `website/index.html`

- [ ] **Step 1: Replace the empty footer element with the footer content**

Find this in `website/index.html`:

```html
<footer>
  <!-- added in Task 6 -->
</footer>
```

Replace it with:

```html
<footer>
  <div class="wrap">
    <div class="footer-row">
      <span class="footer-left">fallax &mdash; <a href="https://github.com/alawein/fallax" style="color:inherit;text-decoration:none">github.com/alawein/fallax</a> &mdash; MIT</span>
      <div class="footer-links">
        <a href="https://github.com/alawein/fallax">github</a>
        <a href="https://meshal.ai/">meshal.ai</a>
        <a href="mailto:contact@meshal.ai">contact</a>
      </div>
    </div>
  </div>
</footer>
```

- [ ] **Step 2: Verify in browser**

Reload and scroll to the bottom. You should see:
- `fallax — github.com/alawein/fallax — MIT` on the left in `#555`
- `github · meshal.ai · contact` links on the right in `#333`, lightening to `#555` on hover
- No top border on the footer (the last section's `border-bottom` serves as the divider)
- Appropriate vertical padding above and below the footer text

- [ ] **Step 3: Commit**

```bash
git add website/index.html
git commit -m "feat(website): add footer with links"
```

---

## Task 7: Copy-to-clipboard JS, final verification, file size check

**Files:**
- Modify: `website/index.html`

- [ ] **Step 1: Add the copy-to-clipboard script**

Find the closing `</body>` tag and insert the script block immediately before it:

```html
<script>
function copyInstall(){
  navigator.clipboard.writeText('uv sync && uv run python -m fallax --help');
  var h=document.getElementById('copy-hint');
  h.textContent='copied!';
  setTimeout(function(){h.textContent='click to copy'},1500);
}
</script>
</body>
```

The final lines of the file should be:

```html
<script>
function copyInstall(){
  navigator.clipboard.writeText('uv sync && uv run python -m fallax --help');
  var h=document.getElementById('copy-hint');
  h.textContent='copied!';
  setTimeout(function(){h.textContent='click to copy'},1500);
}
</script>
</body>
</html>
```

- [ ] **Step 2: Verify copy interaction in browser**

Reload. Click the install command block (`$ uv sync && …`). The hint text should briefly read `copied!` then revert to `click to copy` after 1.5 seconds. Open a text editor and paste — you should see `uv sync && uv run python -m fallax --help`.

Note: `navigator.clipboard` requires a secure context (HTTPS or localhost). If you're opening the file directly as `file://`, this may silently fail in some browsers. That's fine — it will work correctly when deployed to fallax.online.

- [ ] **Step 3: Verify file size**

Run in PowerShell from the repo root:

```powershell
(Get-Item website/index.html).Length / 1KB
```

Expected: a number below `8` (i.e., under 8 KB). If it's larger, look for duplicate or redundant CSS rules.

- [ ] **Step 4: Visual QA checklist**

Open the file in your browser and verify each item:

- [ ] Header sticks to top when scrolling
- [ ] Eyebrow is uppercase small grey text above h1
- [ ] h1 reads "Adversarial Reasoning / Evaluation for LLMs" (line break between)
- [ ] Meta row shows 5 items with `#888` values and `#555` labels
- [ ] Install command box is full-width, shows `$ uv sync …`, has "click to copy" right-aligned
- [ ] Results table has 4 columns, 2 data rows, all dashes, pending note below
- [ ] Taxonomy grid is 2 columns × 3 rows, correct content
- [ ] Quick start code block preserves line breaks, comments are dimmer
- [ ] Footer has text left and links right
- [ ] No external network requests (check browser DevTools → Network — should be empty except favicon)
- [ ] No color other than black/white/grays anywhere on the page

- [ ] **Step 5: Check mobile layout**

In browser DevTools (F12), toggle device toolbar and set to 375px width. Verify:
- h1 shrinks to `1.3rem` (readable, not overflowing)
- Meta row wraps naturally
- Nav links remain legible
- Code block scrolls horizontally rather than breaking layout

- [ ] **Step 6: Commit**

```bash
git add website/index.html
git commit -m "feat(website): add copy-to-clipboard JS — complete research terminal redesign"
```

---

## Self-Review

**Spec coverage:**
- Header (sticky, logo, nav, MIT tag) — Task 1 ✓
- Hero (eyebrow, h1, description, meta row, install command) — Task 2 ✓
- Benchmark results table (4 cols, 2 rows, pending note) — Task 3 ✓
- Failure taxonomy 2-column grid (6 categories, sub-types) — Task 4 ✓
- Quick start code block (comments in dim color) — Task 5 ✓
- Footer (left text, right links, hover states) — Task 6 ✓
- Copy-to-clipboard JS (copies correct string, reverts after 1.5s) — Task 7 ✓
- No web fonts — no `@import` or `<link>` to font CDN ✓
- All CSS inlined — single `<style>` block ✓
- File size under 8 KB — verified in Task 7 Step 3 ✓
- SEO meta tags preserved — Task 1 `<head>` includes all existing OG/Twitter/canonical tags ✓
- Python badge updated to 3.12+ — `<title>` reflects correct version ✓
- Hover transitions `color .15s` — `nav a`, `.footer-links a` ✓
- Pure monochrome — no color value outside `#0a0a0a` → `#e8e8e8` grey range ✓

**Placeholder scan:** No TBDs. All HTML, CSS, and JS shown in full. ✓

**Type consistency:** `copyInstall()` defined in Task 7 and called via `onclick="copyInstall()"` in Task 2. `id="install-cmd"` on the button (unused by JS but harmless). `id="copy-hint"` referenced by `document.getElementById('copy-hint')` — consistent. ✓
