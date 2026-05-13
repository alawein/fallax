---
type: canonical
owner: platform-engineering
last-reviewed: 2026-03-31
---

# Troubleshooting · fallax

> TODO: Document known failure modes, diagnostic steps, and common fixes.

## Common Issues

### `git status` shows many files modified right after a clone or pull

The repo enforces LF line endings via `.gitattributes` (the byte-stable v1
benchmark contract in `benchmarks/v1/metadata.json` depends on it). If
your local git is configured with `core.autocrlf=true`, your working
tree will be CRLF while the index is LF and every text file looks
modified.

Fix one of two ways:

```bash
# Option 1: keep autocrlf, but renormalize the working tree to LF
git add --renormalize . && git status

# Option 2: turn autocrlf off for this repo
git config core.autocrlf false && git checkout -- .
```

If `benchmarks/v1/prompts.jsonl` keeps reverting to CRLF on save, the
culprit is usually Dropbox or an IDE auto-format setting. The
`.gitattributes` rule `benchmarks/**/*.jsonl text eol=lf` will still
make git store the LF version on commit, so CI is unaffected; only the
local working tree drifts.

## Diagnostic Steps

## Known Failure Modes

## FAQ

