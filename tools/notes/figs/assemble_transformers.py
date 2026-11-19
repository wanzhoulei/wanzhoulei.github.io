#!/usr/bin/env python3
"""Assemble notes/src/transformers.md from the template and the generated figures.

Usage: python3 tools/notes/figs/assemble_transformers.py
Then:  python3 tools/build_notes.py

The template is tools/notes/figs/transformers.template.md; a line `{{fig:NAME}}` is replaced by the
inline SVG figure NAME from transformers_figs.py, numbered in order of appearance.
"""
import re, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from transformers_figs import FIGS

ROOT = HERE.parents[2]
tmpl = (HERE / "transformers.template.md").read_text()
out = ROOT / "notes" / "src" / "transformers.md"

n = 0
def sub(m):
    global n
    name = m.group(1)
    if name not in FIGS:
        raise SystemExit(f"unknown figure {name}")
    n += 1
    fn, cap = FIGS[name]
    return "```{=html}\n" + fn().svg(cap, n) + "\n```"

text = re.sub(r"^\{\{fig:([a-z0-9_-]+)\}\}\s*$", sub, tmpl, flags=re.M)
unused = [k for k in FIGS if f"{{{{fig:{k}}}}}" not in tmpl]
out.write_text(text)
print(f"wrote {out} with {n} figures; unused: {unused}")
