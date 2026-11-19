"""Tiny inline-SVG helper for the notes diagrams.

Colors come from the site's CSS variables (style.css), so the figures match the page.
Text labels support `_{sub}` and `^{sup}` (rendered with tspans), and `**bold**`.
"""
import html
import re

COL = {
    "ink": "var(--ink)", "muted": "var(--muted)", "line": "var(--line)", "panel": "var(--panel)",
    "blue": "var(--blue)", "gold": "var(--gold)", "link": "var(--link)", "white": "#ffffff",
    "goldsoft": "#fff3d6", "bluesoft": "#e7eef7", "greysoft": "#eceff4", "red": "#b3261e",
    "green": "#2e7d32", "greensoft": "#e6f2e8", "redsoft": "#fbe9e7", "none": "none",
}


def c(name):
    return COL.get(name, name)


def _rich(s, fs):
    """Convert `_{..}`, `^{..}` and `**..**` into tspans."""
    out = []
    pos = 0
    pat = re.compile(r"(_\{([^}]*)\}|\^\{([^}]*)\}|\*\*([^*]+)\*\*)")
    small = round(fs * 0.72, 1)
    dy = round(fs * 0.3, 1)
    for m in pat.finditer(s):
        if m.start() > pos:
            out.append(html.escape(s[pos:m.start()]))
        if m.group(2) is not None:
            out.append(f'<tspan dy="{dy}" font-size="{small}">{html.escape(m.group(2))}</tspan><tspan dy="{-dy}">​</tspan>')
        elif m.group(3) is not None:
            out.append(f'<tspan dy="{-dy}" font-size="{small}">{html.escape(m.group(3))}</tspan><tspan dy="{dy}">​</tspan>')
        else:
            out.append(f'<tspan font-weight="700">{html.escape(m.group(4))}</tspan>')
        pos = m.end()
    if pos < len(s):
        out.append(html.escape(s[pos:]))
    return "".join(out)


class Fig:
    def __init__(self, name, w, h, label):
        self.name, self.w, self.h, self.label = name, w, h, label
        self.el = []
        self.markers = set()

    # ---- primitives -------------------------------------------------
    def text(self, x, y, s, fs=12, anchor="middle", fill="ink", italic=False, bold=False, rotate=None, family=None):
        st = f"fill:{c(fill)};font-size:{fs}px"
        if italic:
            st += ";font-style:italic"
        if bold:
            st += ";font-weight:700"
        if family == "mono":
            st += ";font-family:ui-monospace,Menlo,Consolas,monospace"
        tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.el.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" style="{st}"{tr}>{_rich(s, fs)}</text>')

    def lines(self, x, y, rows, fs=12, anchor="middle", fill="ink", lh=None, bold_first=False, italic=False):
        lh = lh or round(fs * 1.25)
        y0 = y - lh * (len(rows) - 1) / 2
        for i, r in enumerate(rows):
            self.text(x, y0 + i * lh, r, fs, anchor, fill, bold=(bold_first and i == 0), italic=italic)

    def rect(self, x, y, w, h, fill="panel", stroke="blue", rx=6, sw=1.2, dash=False, opacity=None):
        st = f"fill:{c(fill)};stroke:{c(stroke)};stroke-width:{sw}"
        if dash:
            st += ";stroke-dasharray:5 4"
        if opacity is not None:
            st += f";opacity:{opacity}"
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" style="{st}"/>')

    def box(self, x, y, w, h, label="", sub=None, fill="panel", stroke="blue", rx=6, fs=12, bold=False, dash=False, tc="ink", sw=1.2, subfs=None, subcolor="muted"):
        self.rect(x, y, w, h, fill, stroke, rx, sw, dash)
        cx, cy = x + w / 2, y + h / 2
        if sub:
            subfs = subfs or max(9, fs - 2)
            self.text(cx, cy - subfs * 0.65, label, fs, fill=tc, bold=bold)
            self.text(cx, cy + fs * 0.7, sub, subfs, fill=subcolor)
        elif label:
            if isinstance(label, (list, tuple)):
                self.lines(cx, cy, label, fs, fill=tc, bold_first=bold)
            else:
                self.text(cx, cy, label, fs, fill=tc, bold=bold)
        return (x, y, w, h)

    def circle(self, x, y, r, label="", fill="white", stroke="blue", fs=13, tc="ink"):
        self.el.append(f'<circle cx="{x}" cy="{y}" r="{r}" style="fill:{c(fill)};stroke:{c(stroke)};stroke-width:1.2"/>')
        if label:
            self.text(x, y, label, fs, fill=tc)

    def _marker(self, color):
        mid = f"ah-{self.name}-{re.sub(r'[^a-z0-9]', '', color.lower())}"
        if mid not in self.markers:
            self.markers.add(mid)
        return mid

    def line(self, x1, y1, x2, y2, color="ink", sw=1.3, dash=False, head=False, tail=False):
        st = f"stroke:{c(color)};stroke-width:{sw};fill:none"
        if dash:
            st += ";stroke-dasharray:5 4"
        extra = ""
        if head:
            extra += f' marker-end="url(#{self._marker(color)})"'
        if tail:
            extra += f' marker-start="url(#{self._marker(color)}-s)"'
        self.el.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="{st}"{extra}/>')

    def arrow(self, x1, y1, x2, y2, label=None, lx=None, ly=None, color="ink", dash=False, sw=1.3, fs=11, anchor="middle", lcolor="muted", italic=False):
        self.line(x1, y1, x2, y2, color, sw, dash, head=True)
        if label:
            if lx is None:
                lx = (x1 + x2) / 2
            if ly is None:
                ly = (y1 + y2) / 2 - 8
            self.text(lx, ly, label, fs, anchor, lcolor, italic=italic)

    def poly(self, pts, color="ink", sw=1.3, dash=False, head=True, label=None, lx=None, ly=None, fs=11, anchor="middle", lcolor="muted"):
        st = f"stroke:{c(color)};stroke-width:{sw};fill:none;stroke-linejoin:round"
        if dash:
            st += ";stroke-dasharray:5 4"
        extra = f' marker-end="url(#{self._marker(color)})"' if head else ""
        d = " ".join(f"{x},{y}" for x, y in pts)
        self.el.append(f'<polyline points="{d}" style="{st}"{extra}/>')
        if label:
            self.text(lx, ly, label, fs, anchor, lcolor)

    def matrix(self, x, y, w, h, label, dims=None, fill="bluesoft", stroke="blue", fs=12, rows=None, cols=None, shade=None, dimpos="below"):
        """A matrix drawn as a rectangle, optionally with a grid and shaded cells (shade(i,j)->color or None)."""
        self.rect(x, y, w, h, fill, stroke, rx=2)
        if rows and cols:
            cw, ch = w / cols, h / rows
            for i in range(rows):
                for j in range(cols):
                    col = shade(i, j) if shade else None
                    if col:
                        self.el.append(f'<rect x="{x + j * cw}" y="{y + i * ch}" width="{cw}" height="{ch}" style="fill:{c(col)};stroke:none"/>')
            for i in range(1, rows):
                self.line(x, y + i * ch, x + w, y + i * ch, "line", 0.8)
            for j in range(1, cols):
                self.line(x + j * cw, y, x + j * cw, y + h, "line", 0.8)
            self.rect(x, y, w, h, "none", stroke, rx=2)
        if label:
            self.text(x + w / 2, y + h / 2, label, fs, bold=True)
        if dims:
            if dimpos == "below":
                self.text(x + w / 2, y + h + 11, dims, 10.5, fill="muted")
            else:
                self.text(x + w / 2, y - 10, dims, 10.5, fill="muted")

    def brace_label(self, x, y1, y2, label, side="right", fs=11):
        """A thin bracket with a label, to mark 'x L' repeats etc."""
        xx = x + 6 if side == "right" else x - 6
        self.el.append(f'<path d="M{x},{y1} H{xx} V{y2} H{x}" style="fill:none;stroke:{c("muted")};stroke-width:1"/>')
        self.text(xx + (8 if side == "right" else -8), (y1 + y2) / 2, label, fs, "start" if side == "right" else "end", "muted")

    def dot(self, x, y, r=3, color="ink"):
        self.el.append(f'<circle cx="{x}" cy="{y}" r="{r}" style="fill:{c(color)}"/>')

    # ---- output -----------------------------------------------------
    def svg(self, caption=None, num=None):
        defs = []
        for mid in sorted(self.markers):
            color = mid.split("-")[-1]
            # recover the color token from the id (lowercased letters) -> look up in COL by stripped key
            colv = next((v for k, v in COL.items() if re.sub(r'[^a-z0-9]', '', k.lower()) == color), None) or c(color)
            defs.append(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:{colv}"/></marker>')
        body = "\n".join(self.el)
        svg = (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{html.escape(self.label)}" '
               f'style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:{self.w}px;margin:0 auto">'
               f'<defs>{"".join(defs)}</defs>\n{body}\n</svg>')
        cap = ""
        if caption:
            n = f"<strong>Figure {num}.</strong> " if num else ""
            cap = f"<figcaption>{n}{caption}</figcaption>"
        return f'<figure class="fig" id="fig-{self.name}">{svg}{cap}</figure>'
