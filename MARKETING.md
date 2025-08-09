# OmegaOMG — High‑precision Entity Extraction at Scale

OmegaOMG is a compact, production‑ready grammar and engine for turning unstructured text into clean, canonical entities. It combines a simple DSL with an optimized evaluator and a robust resolver pipeline to deliver accurate, deduplicated results you can trust.

> Define rules once. Extract with confidence. Ship faster.

Built on top of [OmegaMatch](https://github.com/scholarsmate/omega-match) for the list matching engine.

---

## Why OmegaOMG

- Precision by design: longest, left‑most, non‑overlapping anchors keep noise out.
- Built for scale: fast, byte‑level evaluation with smart indexing and caching.
- Clean output: canonicalization, parent/child linking, and sentence/paragraph metadata.
- Easy to author: expressive DSL (v1.0) with imports, quantifiers, dot/char classes, named captures.
- Works anywhere: Python runtime, simple CLI, and a VS Code extension for authoring.

---

## What makes it different

- Anchored matching: Uses curated token lists (via OmegaMatch) to pre‑anchor rules and minimize false positives.
- Deterministic resolution: A staged, auditable pipeline to deduplicate parents, attach children, and enrich with boundaries.
- Practical performance: Offset‑indexed lookups, greedy quantifier chaining, and adaptive sampling reduce scanning cost.
- Clear boundaries: No unbounded regex; bounded quantifiers only. Production behavior is predictable and fast.

---

## Ideal for

- PII and sensitive data detection (names, phones, emails, IDs)
- Compliance & eDiscovery workflows
- Threat intel and OSINT enrichment
- Resume/CV and profile parsing
- Log and support ticket mining

---

## Core capabilities

- DSL v1.0: literals, escapes (\\d \\s \\w, etc.), dot `.`, character classes `[...]`, grouping, alternation, named captures, bounded quantifiers, and list matches `[[alias]]`.
- Imports with flags: `ignore-case`, `ignore-punctuation`, `elide-whitespace`, `word-boundary`, `word-prefix`, `word-suffix`, `line-start`, `line-end`.
- Resolver methods: `exact` and `fuzzy(threshold=...)`, optional tokens, and case/punctuation normalization.
- Parent/child modeling: Dotted rules (e.g., `person.surname`) with automatic parent boundary enrichment.
- JSON output: Clean, line‑delimited JSON or pretty JSON for easy pipelines.
 - HTML visualization: interactive highlighted HTML via `highlighter.py` for fast review and demos.

---

## How it works (high level)

1) Author rules in OMG DSL referencing curated token lists.  
2) Evaluate over byte inputs using an optimized engine with pre‑anchored matches.  
3) Apply entity resolution:
   - Overlap removal with deterministic tie‑breaking
   - Horizontal canonicalization (dedupe parents)
   - Vertical child→parent linking
   - Boundary metadata enrichment (sentence, paragraph)

Detailed algorithm: see `RESOLUTION.md`.

---

## Quick start

- Try the demo rules on a sample document:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python omg.py --output matches.json .\demo\demo.omg .\demo\CIA_Briefings_of_Presidential_Candidates_1952-1992.txt
python highlighter.py .\demo\CIA_Briefings_of_Presidential_Candidates_1952-1992.txt matches.json CIA_demo.html
```

Open `CIA_demo.html` in your browser to inspect the results.

---

## Example rule (DSL)

```dsl
version 1.0
import "names.txt" as given_name with word-boundary, ignore-case
import "surnames.txt" as surname with word-boundary, ignore-case

resolver default uses exact with ignore-case

person = [[given_name]] ( \s{1,4} [[given_name]] ){0,2} ( \s{1,4} \w | \s{1,4} \w "." )? \s{1,4} [[surname]]
```

---

## Integrations

- CLI: `omg.py` for batch extraction and JSON output
- Python API: programmatic parsing and evaluation (`dsl.omg_parser`, `dsl.omg_evaluator`)
- VS Code extension: [OMG Language Support](https://github.com/scholarsmate/omega-omg-vscode) for syntax highlighting & IntelliSense
- List matching engine: powered by [OmegaMatch](https://github.com/scholarsmate/omega-match)
 - Highlighter: render interactive HTML from matches (`highlighter.py`)

---

## Performance mindset

- Byte‑accurate matching with pre‑compiled list anchors
- Binary‑searched offset maps and caching across nodes
- Greedy, adjacency‑enforced ListMatch quantifiers
- Adaptive start‑offset sampling for complex patterns

Performance depends on rule complexity and input size—designed to be efficient for large files and corpora.

---

## Governance & licensing

- Open source under Apache 2.0 (`LICENSE`)
- Tested with a comprehensive pytest suite
- Minimal, pinned dependencies for stability

---

## Learn more

- README: features, usage, and CLI options
- RESOLUTION.md: deep dive into the resolver pipeline
- Demo: `demo/demo.omg` and sample lists / texts

> Get started today—write a rule, run the demo, and ship accurate entity extraction with OmegaOMG.
