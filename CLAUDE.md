# Blog — poname.github.io

Personal tech blog. Hugo + PaperMod theme, deployed via GitHub Pages.

## Stack

- **Hugo** with [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme (dark mode default)
- Theme is NOT installed locally — cloned in CI (`deploy.yml`)
- Config: `hugo.yaml`

## Creating a New Post

Use Hugo **page bundles** — each post is a directory:

```
content/posts/my-post-slug/
  index.md          # post content
  diagram1.png      # images co-located
  diagram2.png
  source.d2         # D2 source files (optional, for re-rendering)
```

### Frontmatter template

```yaml
---
title: "Post Title Here"
date: 2026-01-15
draft: false
tags: ["Tag1", "Tag2"]
categories: ["Technical"]
summary: "One-sentence hook for the listing page."
cover:
    image: "cover-image.png"
    alt: "Description of cover image"
    relative: true
---
```

### Image references

Images are **relative** since they live in the same directory as `index.md`:

```markdown
![Alt text](diagram.png)
```

NOT `![Alt text](/static/images/diagram.png)` or absolute paths.

## Diagrams

Render diagrams with [D2](https://d2lang.com/) using **Theme 1 (Neutral Grey)** for a clean, professional look.

```bash
d2 --theme 1 --sketch --scale 2 --pad 40 source.d2 output.png
```

### Style conventions

- **Theme 1** (Neutral Grey) + **`--sketch`** — professional with hand-drawn warmth. Proper-case text, not ALL-CAPS.
- **`--scale 2`** for crisp text at blog width (~720px in PaperMod)
- **Aspect ratios:** Target 1:1 to 3:1. Avoid anything wider than 4:1 (unreadable when scaled) or taller than 1:2 (excessive scroll).
- **Color palette** — consistent semantic colors across all posts:
  - Green (`#dcfce7` fill, `#16a34a` stroke) — positive/correct/success
  - Red (`#fee2e2` fill, `#dc2626` stroke) — negative/wrong/danger
  - Blue (`#dbeafe` fill, `#2563eb` stroke) — key elements, primary
  - Purple (`#f3e8ff` fill, `#7c3aed` stroke) — ML/classifier, secondary
  - Amber (`#fef3c7` fill, `#d97706` stroke) — callouts/highlights
- **Font sizes:** 14-16 for main labels, 12-13 for details. No titles in diagrams — use the markdown caption instead.
- **Layout:** Use `direction: down` for flow diagrams, `direction: right` for side-by-side comparisons with `direction: down` inside each container.
- **Grid:** Use `grid-rows: 1` for horizontal groups inside vertical flows. Use `grid-columns: N` for side-by-side comparison at the top level.
- Keep `.d2` source files in the post bundle so diagrams can be re-rendered

## Writing voice

- **Technical but approachable** — clear explanations, some personality, not academic
- First person singular ("I observed", "I considered") not plural "we"
- Concrete examples throughout — no abstract hand-waving
- Pick a fun, relatable domain for running examples (movies, games, recipes — not enterprise software)
- Keep examples consistent within a post — one domain, one story, all the way through

## Structure conventions

- Start with a *Prerequisites* note if the post assumes prior knowledge
- When building on existing techniques, credit prior art inline (not a standalone section — keeps the reader moving)
- End with **Where to Go From Here** pointing to the natural next topic
- Add a **References** section with numbered links to papers/benchmarks cited
- Use inline citations linking to individual references: `[[1]](#ref-1)`
- Each reference needs an anchor: `<span id="ref-1">**[1]**</span>` (requires `markup.goldmark.renderer.unsafe: true` in hugo.yaml)

## Existing posts

- `churn-reduction.md` — business case study (Strategy, SQL)
- `time-series-forecasting.md` — technical deep dive (Python, ML)
- `dual-index-rag/` — page bundle with D2 diagrams (RAG, LLM)

### LLM Audit & Guard series (all `draft: true`)

Six posts covering LLM runtime security and audit infrastructure. Code examples are simplified from a production system — real patterns, reshaped names. No company-specific details.

| Post | Slug | Topic |
|------|------|-------|
| Part 1 | `zero-touch-llm-guarding` | Callback architecture, context gating, five defense layers, middleware alternative |
| Part 2 | `llm-guard-patterns` | Regex patterns (intent not keywords), base64 decode, XML delimiters, output scanning |
| Part 3 | `llm-guard-classifiers` | Meta Prompt Guard 2, online/offline deployment, red teaming (PromptFoo, Garak) |
| Standalone | `template-method-ml` | Template method pattern for ML infrastructure lifecycle enforcement |
| Standalone | `llm-security-healthcare` | Healthcare-specific LLM security inversions (6 problems) |
| Standalone | `zero-touch-llm-auditing` | Automatic audit logging, invocation correlation, two-tier storage |

**Key constraints for this series:**
- No company names, repo paths, or internal system names (FHIR, Verily, GKE, CPR)
- Code examples use reshaped names — never copy-paste from production code
- The author does NOT want to be positioned as a healthcare-only specialist — the series targets broad ML/AI engineering audience
- Healthcare post is domain expertise, not the series identity
- Each post should be somewhat self-contained — brief references to other posts, not re-explanations
- "Simplified from production" framing is stated once in Part 1

## Tooling

```bash
# Preview locally (requires Hugo installed + theme cloned to themes/PaperMod)
hugo server -D

# Render a D2 diagram
d2 --theme 1 --sketch --scale 2 --pad 40 input.d2 output.png

# Deploy — just push to main, GitHub Actions handles the rest
git push origin main
```
