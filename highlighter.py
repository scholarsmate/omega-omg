import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import List, Optional


@dataclass
class Match:
    offset: int
    length: int
    rule: str
    match: str
    reference: Optional[int] = None
    sentence_end: Optional[int] = None
    paragraph_end: Optional[int] = None


def highlight_text_byte_aligned(text, matches):
    # encode text to bytes
    btext = text.encode("utf-8")
    # build parent text map using efficient dict comprehension
    parent_texts = {pm.offset: pm.match for pm in matches if pm.reference is None}
    # build and dedupe spans (matches are already non-overlapping)
    unique_spans = {}
    for m in matches:
        start = m.offset
        end = start + m.length
        key = (start, end, m.rule)
        if key not in unique_spans:
            unique_spans[key] = m
    # sort spans by start offset
    sorted_matches = sorted(unique_spans.values(), key=lambda m: m.offset)
    # filter empty content only (no overlap filtering needed)
    filtered = []
    for m in sorted_matches:
        start = m.offset
        end = start + m.length
        raw_text = btext[start:end].decode("utf-8")
        raw = escape(raw_text)
        if not raw.strip():
            continue
        filtered.append((start, end, m, raw))
    # build highlighted HTML
    result = bytearray()
    last = 0
    for start, end, m, raw in filtered:
        # append preceding text
        segment = btext[last:start].decode("utf-8")
        result.extend(escape(segment).encode("utf-8"))
        # prepare span attributes
        rule = m.rule
        reference = m.reference
        # build hover title: include JSON match text, and parent info if reference
        match_text = m.match
        if reference is not None:
            parent_raw = parent_texts.get(reference, "")
            title = f"{rule}: {match_text} (ref: {reference} -> {parent_raw})"
        else:
            title = f"{rule}: {match_text}"
        # sanitize title for HTML attribute: escape &, <, >, ' and ", replace newlines
        safe_title = escape(title, quote=True).replace('"', "&quot;").replace("\n", " ")
        attrs = f'class="highlight" data-rule="{rule}" data-sentence-end="{m.sentence_end}" data-paragraph-end="{m.paragraph_end}"'
        if reference is None:
            attrs += f' id="match-{start}"'
        else:
            attrs += f' data-reference="{reference}" onclick="gotoRef({reference})"'
        # handle multi-line matches by splitting on newline
        if "\n" in raw:
            parts = raw.split("\n")
            for idx, part in enumerate(parts):
                # wrap each line part
                part_span = f'<span {attrs} title="{safe_title}"><span class="inner">{part}</span></span>'
                result.extend(part_span.encode("utf-8"))
                # reinsert newline between parts
                if idx < len(parts) - 1:
                    result.extend(b"\n")
        else:
            # single-line match
            part_span = f'<span {attrs} title="{safe_title}"><span class="inner">{raw}</span></span>'
            result.extend(part_span.encode("utf-8"))
        last = end
    # append remaining text
    tail = btext[last:].decode("utf-8")
    result.extend(escape(tail).encode("utf-8"))
    return result.decode("utf-8")


def generate_css_for_rules(rule_types):
    palette = [
        "#5fa8d3",
        "#72b69d",
        "#bfa75c",
        "#c87f7f",
        "#999ca1",
        "#7fcad3",
        "#cb8b8b",
        "#9b9ea1",
        "#88c39d",
        "#c5ae6d",
    ]
    css = []
    for i, rule in enumerate(sorted(rule_types)):
        color = palette[i % len(palette)]
        # use attribute selector to handle dots and special chars in rule names
        css.append(
            f".highlight[data-rule='{rule}'] .inner {{ background-color: {color}; }}"
        )
    return "\n        ".join(css)


def generate_html(highlighted_text, rule_counts, show_line_numbers=True):
    lines = highlighted_text.splitlines(keepends=True)
    numbered_text = "".join(
        f"<span class='line'><span class='lineno'>{i:4}</span> {ln}</span>"
        for i, ln in enumerate(lines, start=1)
    )
    # Prepare palette and toggles with colored squares for each rule
    rule_types = sorted(rule_counts.keys())
    # Generate toggles with colored checkboxes via accent-color
    palette = [
        "#5fa8d3",
        "#72b69d",
        "#bfa75c",
        "#c87f7f",
        "#999ca1",
        "#7fcad3",
        "#cb8b8b",
        "#9b9ea1",
        "#88c39d",
        "#c5ae6d",
    ]
    toggles = "\n".join(
        f"<label><input type='checkbox' checked style='accent-color: {palette[i % len(palette)]}' onchange=\"toggleRule('{rule}')\"> {rule} ({rule_counts[rule]})</label>"
        for i, rule in enumerate(rule_types)
    )
    rule_styles = generate_css_for_rules(rule_types)
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset=\"UTF-8\">
    <style>
        body {{ font-family: monospace; background: #121212; color: #e0e0e0; padding: 1em; margin: 0; }}
        .highlight {{ border-bottom: 2px dotted #888; cursor: help; }}
        /* clickable reference */
        .highlight[data-reference] {{ cursor: pointer; }}
        .line {{ display: block; }}
        .lineno {{ display: inline-block; width: 3em; text-align: right; margin-right: 1em; color: #888; }}
        {rule_styles}
        pre {{ white-space: pre-wrap; word-wrap: break-word; line-height: 1.4; margin-top: 6em; }}
        .controls {{ position: fixed; top: 0; left: 0; right: 0; background: #1e1e1e; padding: 1em; z-index: 1000; border-bottom: 1px solid #444; }}
        .controls-content {{ margin-top: 0.5em; }}
        label {{ margin-right: 1em; }}
    </style>
    <script>
    // toggle visibility of the controls content
    function toggleControls() {{
        const content = document.getElementById('controls-content');
        const btn = document.getElementById('controls-toggle');
        const expanded = btn.getAttribute('aria-expanded') === 'true';
        // toggle visibility and aria attributes
        content.style.display = expanded ? 'none' : 'block';
        content.setAttribute('aria-hidden', expanded);
        btn.setAttribute('aria-expanded', !expanded);
        btn.textContent = expanded ? 'Show controls' : 'Hide controls';
    }}
    function toggleRule(rule) {{
        document.querySelectorAll(`[data-rule='${{rule}}'] > .inner`).forEach(el => {{
            el.style.backgroundColor = (el.style.backgroundColor === 'transparent') ? '' : 'transparent';
        }});
    }}
    function toggleLineNumbers() {{
        document.querySelectorAll('.lineno').forEach(el => {{
            el.style.display = (el.style.display === 'none') ? 'inline-block' : 'none';
        }});
    }}
    document.addEventListener('DOMContentLoaded', () => {{
        if (!{ 'true' if show_line_numbers else 'false' }) toggleLineNumbers();
    }});
    // Initialize keyboard navigation for highlighted matches
    let matchElements = [], currentMatch = -1;
    document.addEventListener('DOMContentLoaded', () => {{
        matchElements = Array.from(document.querySelectorAll('.highlight > .inner'));
        document.addEventListener('keydown', e => {{
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === 'n') nextMatch();
            else if (e.key === 'p') prevMatch();
        }});
    }});
    function nextMatch() {{
        if (!matchElements.length) return;
        currentMatch = (currentMatch + 1) % matchElements.length;
        matchElements[currentMatch].scrollIntoView({{behavior:'smooth', block:'center'}});
        flashMatch(matchElements[currentMatch]);
    }}
    function prevMatch() {{
        if (!matchElements.length) return;
        currentMatch = (currentMatch - 1 + matchElements.length) % matchElements.length;
        matchElements[currentMatch].scrollIntoView({{behavior:'smooth', block:'center'}});
        flashMatch(matchElements[currentMatch]);
    }}
    function flashMatch(el) {{
        const orig = el.style.backgroundColor;
        el.style.backgroundColor = 'yellow';
        setTimeout(() => el.style.backgroundColor = orig, 500);
    }}
    // Jump to parent match when clicking a reference span
    function gotoRef(offset) {{
        const el = document.getElementById('match-' + offset);
        if (el) {{
            el.scrollIntoView({{behavior:'smooth', block:'center'}});
            const inner = el.querySelector('.inner');
            if (inner) setTimeout(() => flashMatch(inner), 800);
        }}
    }}
    </script>
</head>
<body>
<div class="controls">
    <button id="controls-toggle" aria-expanded="true" aria-controls="controls-content" onclick="toggleControls()">Hide controls</button>
    <div id="controls-content" class="controls-content" aria-hidden="false">
        <strong>Toggle highlights:</strong><br>
        {toggles}
        <br><label><input type='checkbox' { 'checked' if show_line_numbers else '' } onchange="toggleLineNumbers()"> Show line numbers</label>
    </div>
</div>
<pre>{numbered_text}</pre>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Highlight matches in a text file based on JSON annotations from omg.py."
    )
    parser.add_argument("text_file", type=Path, help="Path to the input text file")
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the JSON file with enriched match data from omg.py",
    )
    parser.add_argument(
        "output_file", type=Path, help="Path to save the output HTML file"
    )
    parser.add_argument(
        "--no-line-numbers",
        action="store_true",
        help="Disable line numbers by default in the HTML",
    )
    args = parser.parse_args()

    # load single input set
    text = args.text_file.read_text(encoding="utf-8")
    with args.json_file.open(encoding="utf-8") as f:
        raw_matches = [json.loads(line) for line in f]

    # convert dicts to Match objects - omg.py now provides enriched data with reference and boundaries
    matches: List[Match] = []
    seen = set()
    for m in raw_matches:
        key = (m["offset"], m["length"], m["rule"])
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            Match(
                offset=m["offset"],
                length=m["length"],
                rule=m["rule"],
                match=m.get("match", ""),
                reference=m.get("reference"),
                sentence_end=m.get("sentence_end"),
                paragraph_end=m.get("paragraph_end"),
            )
        )

    # drop zero-length
    matches = [m for m in matches if m.length > 0]

    # highlight (matches are already non-overlapping from omg.py)
    t0 = time.time()
    highlighted = highlight_text_byte_aligned(text, matches)
    t1 = time.time()

    # print performance metrics
    print(f"Rendering: {t1-t0:.3f}s, Total matches: {len(matches)}")
    rule_counts = Counter(m.rule for m in matches)
    html = generate_html(
        highlighted, rule_counts, show_line_numbers=not args.no_line_numbers
    )
    args.output_file.write_text(html, encoding="utf-8")
    print(f"HTML file with highlights saved to: {args.output_file}")


if __name__ == "__main__":
    main()
