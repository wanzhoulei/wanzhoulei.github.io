#!/usr/bin/env python3
"""Build the Notes section: notes/src/*.md -> notes/<slug>.html plus notes/index.html.

Usage:  python3 tools/build_notes.py
Needs pandoc (brew install pandoc). Math is rendered in the browser by KaTeX (pinned version below).

Each source file starts with YAML front matter (title, subtitle, order, updated). Special blocks:
  ::: qa                     collapsible practice question (first paragraph = question)
  ::: {.qa asked="..."}      same, marked as reported in a real interview
  ::: key / ::: pitfall      callout boxes
"""
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notes" / "src"
OUT = ROOT / "notes"
TEMPLATE = ROOT / "tools" / "notes" / "note.template.html"
FILTER = ROOT / "tools" / "notes" / "notes.lua"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/"

QA_RE = re.compile(r"^:::+\s*(?:qa\b|\{[^}]*\.qa\b)", re.M)
ASKED_RE = re.compile(r"^:::+\s*\{[^}]*\.qa\b[^}]*asked=", re.M)


def front_matter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    meta = {}
    if not m:
        return meta, text
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep:
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            meta[k.strip()] = v
    return meta, text[m.end():]


def load_notes():
    notes = []
    for path in sorted(SRC.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, body = front_matter(text)
        prose = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$|```.*?```", " ", body, flags=re.S)
        words = len(re.findall(r"[A-Za-z0-9']+", prose))
        notes.append({
            "slug": path.stem,
            "path": path,
            "title": meta.get("title", path.stem),
            "subtitle": meta.get("subtitle", ""),
            "order": float(meta.get("order", 99)),
            "updated": meta.get("updated", ""),
            "qcount": len(QA_RE.findall(body)),
            "asked": len(ASKED_RE.findall(body)),
            "minutes": max(1, round(words / 220)),
        })
    notes.sort(key=lambda n: (n["order"], n["slug"]))
    return notes


def build_page(note, prev, nxt):
    cmd = [
        "pandoc", str(note["path"]),
        "-f", "markdown", "-t", "html5", "--wrap=none",
        "--template", str(TEMPLATE),
        "--lua-filter", str(FILTER),
        f"--katex={KATEX}",
        "--toc", "--toc-depth=2",
        "-V", f"qcount={note['qcount']}",
        "-V", f"minutes={note['minutes']}",
        "-o", str(OUT / f"{note['slug']}.html"),
    ]
    if prev:
        cmd += ["-V", f"prev_slug={prev['slug']}", "-V", f"prev_title={prev['title']}"]
    if nxt:
        cmd += ["-V", f"next_slug={nxt['slug']}", "-V", f"next_title={nxt['title']}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"pandoc failed on {note['path'].name}:\n{res.stderr}")
    if res.stderr.strip():
        print(f"[{note['slug']}] {res.stderr.strip()}")
    # The contents list appears twice (side rail and phone dropdown); drop the ids in the second copy.
    out = OUT / f"{note['slug']}.html"
    page = out.read_text(encoding="utf-8")
    start = page.find('<details class="toc-mobile">')
    if start >= 0:
        end = page.find("</details>", start)
        page = page[:start] + re.sub(r' id="toc-[^"]*"', "", page[start:end]) + page[end:]
        out.write_text(page, encoding="utf-8")


def card(n, featured=False, num=None):
    e = html.escape
    stats = f"{n['qcount']} questions · {n['minutes']} min read"
    cls = "note-card featured" if featured else "note-card"
    num = "" if featured else f'<span class="note-num">{num:02d}</span>'
    return (f'<a class="{cls}" href="{e(n["slug"])}.html">{num}'
            f'<span class="note-card-title">{e(n["title"])}</span>'
            f'<span class="note-card-sub">{e(n["subtitle"])}</span>'
            f'<span class="note-card-stats">{e(stats)}</span></a>')


def build_index(notes):
    bank = [n for n in notes if n["slug"] == "interview-question-bank"]
    topics = [n for n in notes if n["slug"] != "interview-question-bank"]
    topic_q = sum(n["qcount"] for n in topics)
    bank_q = sum(n["qcount"] for n in bank)
    bank_asked = sum(n["asked"] for n in bank)
    updated = max((n["updated"] for n in notes), default="")
    featured = card(bank[0], featured=True) if bank else ""
    cards = "\n      ".join(card(n, num=i) for i, n in enumerate(topics, 1))
    page = INDEX.format(featured=featured, cards=cards, topic_q=topic_q, bank_q=bank_q, bank_asked=bank_asked,
                        n_topics=len(topics), updated=html.escape(updated))
    (OUT / "index.html").write_text(page, encoding="utf-8")


INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Notes · Wanzhou Lei</title>
  <meta name="description" content="Notes on generative modeling, computer vision and deep learning, with derivations and practice questions.">
  <link rel="icon" type="image/png" href="../assets/favicon.png">
  <link rel="apple-touch-icon" href="../assets/favicon.png">
  <link rel="stylesheet" href="../style.css">
  <link rel="stylesheet" href="notes.css">
</head>
<body class="note-page">

<header class="site-header">
  <div class="container bar">
    <a class="brand" href="../index.html">Wanzhou Lei</a>
    <nav class="site-nav" aria-label="Sections">
      <a class="navlink" href="../index.html#about">About</a>
      <a class="navlink" href="../index.html#research">Research</a>
      <a class="navlink" href="../index.html#experience">Experience</a>
      <a class="navlink" href="../index.html#news">News</a>
      <a class="navlink" href="../index.html#publications">Publications</a>
      <a class="notes-btn is-current" href="./">Notes</a>
      <a class="cv-btn" href="../assets/Wanzhou_Lei_CV.pdf" target="_blank" rel="noopener">CV</a>
    </nav>
  </div>
</header>

<main class="container">
  <div class="notes-head">
    <h1 class="page-title">Notes</h1>
    <p class="page-lede">Notes on generative modeling, computer vision and deep learning. Each page builds the core ideas from first principles, derives the key equations, and ends with practice questions you can expand one at a time.</p>
    <p class="note-meta"><span>{n_topics} topics</span><span>{topic_q} practice questions in the topic notes</span><span>{bank_q} interview-bank questions, {bank_asked} with a cited source</span><span>Updated {updated}</span></p>
  </div>
  {featured}
  <div class="note-grid">
      {cards}
  </div>
</main>

<footer class="site-footer">
  <div class="container">&copy; 2026 Wanzhou Lei &middot; <a href="../index.html">Home</a></div>
</footer>
</body>
</html>
"""


def main():
    notes = load_notes()
    if not notes:
        sys.exit("No notes found in notes/src/")
    for i, n in enumerate(notes):
        build_page(n, notes[i - 1] if i > 0 else None, notes[i + 1] if i + 1 < len(notes) else None)
    build_index(notes)
    for n in notes:
        print(f"{n['slug']:32s} {n['qcount']:3d} q ({n['asked']} asked)  {n['minutes']:3d} min")


if __name__ == "__main__":
    main()
