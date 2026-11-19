"""Figures for notes/src/transformers.md. Run: python3 tools/notes/figs/transformers_figs.py [--sheet out.html]
Each function returns a Fig; FIGS maps name -> (fig, caption)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svgfig import Fig

FIGS = {}


def reg(name, caption):
    def deco(fn):
        FIGS[name] = (fn, caption)
        return fn
    return deco


# ---------------------------------------------------------------- helpers
def residual_block(f, xs, y, w=250, causal=True, label_attn=None, gap=22, bh=34):
    """Draw one pre-LN block: residual stream at x=xs, sublayers to the right. Returns y after the block."""
    bx = xs + 60
    label_attn = label_attn or ("Masked multi-head" if causal else "Multi-head")
    # attention sublayer
    y1 = y
    f.box(bx, y1, 60, bh, "LN", fs=12)
    f.box(bx + 80, y1, w - 80, bh, label_attn, sub="self-attention", fs=12)
    f.arrow(bx + 60, y1 + bh / 2, bx + 80, y1 + bh / 2)
    yplus = y1 + bh + gap + 10
    f.line(xs, y1 - 14, xs, yplus - 10)
    f.line(xs, y1 + bh / 2, bx, y1 + bh / 2, head=True)
    f.poly([(bx + w, y1 + bh / 2), (bx + w + 14, y1 + bh / 2), (bx + w + 14, yplus), (xs + 10, yplus)], head=True)
    f.circle(xs, yplus, 10, "+")
    # mlp sublayer
    y2 = yplus + gap + 4
    f.box(bx, y2, 60, bh, "LN", fs=12)
    f.box(bx + 80, y2, w - 80, bh, "MLP", sub="D → 4D → D", fs=12)
    f.arrow(bx + 60, y2 + bh / 2, bx + 80, y2 + bh / 2)
    yplus2 = y2 + bh + gap + 10
    f.line(xs, yplus + 10, xs, yplus2 - 10)
    f.line(xs, y2 + bh / 2, bx, y2 + bh / 2, head=True)
    f.poly([(bx + w, y2 + bh / 2), (bx + w + 14, y2 + bh / 2), (bx + w + 14, yplus2), (xs + 10, yplus2)], head=True)
    f.circle(xs, yplus2, 10, "+")
    return yplus2


# ---------------------------------------------------------------- 1 overview
@reg("overview", "A decoder-only transformer from token IDs to next-token probabilities. The residual stream (vertical line) carries one D-vector per token through L identical blocks; each block adds the output of an attention sub-layer and of an MLP sub-layer to it. Shapes on the right are for a batch of B sequences of N tokens.")
def overview():
    f = Fig("overview", 780, 620, "Whole decoder-only transformer: embedding, L pre-LN blocks of attention and MLP on a residual stream, final norm, output head")
    xs = 150
    f.box(xs - 70, 14, 140, 34, "Token IDs", fs=12)
    f.text(660, 31, "(B, N) integers", 11.5, "middle", "muted")
    f.arrow(xs, 48, xs, 70)
    f.box(xs - 130, 70, 260, 44, "Embedding lookup E[t] + position", sub="(learned or sinusoidal; RoPE acts inside attention)", fs=12, subfs=9.5)
    f.text(660, 92, "(B, N, D)", 11.5, "middle", "muted")
    f.arrow(xs, 114, xs, 150)
    # block group
    f.rect(xs - 40, 140, 420, 236, fill="none", stroke="muted", dash=True, rx=10)
    f.text(xs - 30, 152, "one block (pre-LN)", 10.5, "start", "muted", italic=True)
    yend = residual_block(f, xs, 168, w=250, causal=True)
    f.brace_label(xs + 390, 140, 376, "× L", side="right", fs=13)
    f.text(660, 160, "(B, N, D) in", 11.5, "middle", "muted")
    f.text(660, 372, "(B, N, D) out", 11.5, "middle", "muted")
    f.arrow(xs, yend + 10, xs, 404)
    f.box(xs - 50, 404, 100, 32, "Final LN", fs=12)
    f.arrow(xs, 436, xs, 462)
    f.box(xs - 110, 462, 220, 44, "Output head W_{U}", sub="D → V logits (often tied to E)", fs=12, subfs=9.5)
    f.text(660, 484, "(B, N, V)", 11.5, "middle", "muted")
    f.arrow(xs, 506, xs, 536)
    f.box(xs - 130, 536, 260, 44, "softmax → cross-entropy", sub="target at position t is token t+1", fs=12, subfs=9.5)
    f.text(660, 558, "N predictions per pass", 11.5, "middle", "muted")
    # side note on attention being the only mixing op
    f.lines(660, 262, ["attention is the only place", "tokens exchange information;", "LN, MLP and + act per token"], 11, fill="muted", italic=True)
    return f


# ---------------------------------------------------------------- 2 families
@reg("families", "The same block in three arrangements. Encoder-only stacks use bidirectional attention and read out a representation; decoder-only stacks use causal attention and predict the next token; encoder–decoder models add a cross-attention sub-layer in every decoder block whose keys and values come from the encoder's output.")
def families():
    f = Fig("families", 760, 330, "Encoder-only, decoder-only and encoder-decoder transformers side by side")
    def stack(x, title, sub, layers, top, bottom, color="panel"):
        f.text(x + 90, 18, title, 13, bold=True)
        f.text(x + 90, 36, sub, 10.5, fill="muted", italic=True)
        f.box(x + 20, 262, 140, 30, bottom, fs=11)
        yy = 230
        for i, (lab, s) in enumerate(reversed(layers)):
            pass
        y = 232
        for lab, s in layers:
            y -= 44
            f.box(x, y, 180, 38, lab, sub=s, fill=color, fs=11.5, subfs=9.5)
        f.arrow(x + 90, 262, x + 90, 232)
        f.brace_label(x + 186, y, 232, "× L", fs=11)
        f.arrow(x + 90, y, x + 90, 60)
        f.box(x + 10, 44, 160, 30, top, fs=11)
        return y
    stack(20, "Encoder-only", "BERT, ViT, DiT, text encoders", [("MLP", None), ("Self-attention", "bidirectional: every token sees all")], "representation per token", "input tokens")
    stack(290, "Decoder-only", "GPT, Llama", [("MLP", None), ("Self-attention", "causal: token t sees ≤ t")], "next-token logits", "prefix tokens")
    y = stack(560, "Encoder–decoder", "original transformer, T5", [("MLP", None), ("Cross-attention", "Q from decoder, K,V from encoder"), ("Self-attention", "causal")], "target tokens", "target prefix")
    # encoder feeding cross-attention
    f.box(560, 296, 180, 28, "encoder output (M × D)", fs=10.5, fill="bluesoft")
    f.poly([(740, 310), (752, 310), (752, y + 44 + 19), (740, y + 44 + 19)], color="link", head=True)
    return f


# ---------------------------------------------------------------- 3 tokenize
@reg("tokenize", "Three inputs, one interface. Text becomes subword IDs and an embedding lookup; an image is cut into P×P patches and each flattened patch is projected by one linear map; a VAE latent is patchified the same way with a smaller patch. Everything downstream sees an (N, D) matrix.")
def tokenize():
    f = Fig("tokenize", 780, 350, "Text tokens, image patches and latent patches all become an N by D matrix of token vectors")
    # text row
    f.text(14, 48, "Text", 13, "start", bold=True)
    f.box(70, 32, 120, 32, '"the cat sat"', fs=11.5, fill="white")
    f.arrow(190, 48, 250, 48, "tokenizer (BPE)", ly=18, fs=10)
    f.box(250, 32, 120, 32, "[464, 3797, 3332]", fs=10.5, fill="white")
    f.arrow(370, 48, 430, 48, "row lookup E[t]", ly=18, fs=10)
    f.matrix(430, 26, 60, 44, "", dims="N × D", rows=3, cols=6)
    f.text(560, 48, "N tokens of D numbers each", 10.5, "start", "muted")
    # image row
    f.text(14, 160, "Image", 13, "start", bold=True)
    f.rect(70, 110, 100, 100, fill="greysoft", stroke="muted", rx=2)
    for i in range(1, 4):
        f.line(70 + 25 * i, 110, 70 + 25 * i, 210, "muted", 0.8)
        f.line(70, 110 + 25 * i, 170, 110 + 25 * i, "muted", 0.8)
    f.rect(95, 135, 25, 25, fill="goldsoft", stroke="gold", rx=0, sw=1.6)
    f.text(120, 222, "H × W × C, patch size P", 10.5, fill="muted")
    f.arrow(172, 160, 250, 160, "flatten one patch", ly=130, fs=10)
    f.box(250, 144, 70, 32, "P²C", fs=11.5, fill="goldsoft", stroke="gold")
    f.arrow(320, 160, 430, 160, "linear map P²C → D", ly=130, fs=10)
    f.text(375, 192, "(= conv, kernel P, stride P)", 9.5, fill="muted")
    f.box(430, 144, 46, 32, "D", fs=12, fill="bluesoft")
    f.text(453, 206, "one token", 10, fill="muted")
    f.arrow(476, 160, 530, 160, "all patches", ly=130, fs=10)
    f.matrix(530, 130, 60, 60, "", dims="N × D, N = HW/P²", rows=4, cols=6)
    f.lines(690, 150, ["ViT-B/16 at 224²:", "N = 14 × 14 = 196", "(+1 class token)", "P²C = 768 → D = 768"], 10.5, "middle", "muted", lh=14)
    # latent row
    f.text(14, 290, "Latent", 13, "start", bold=True)
    f.rect(70, 260, 60, 60, fill="greysoft", stroke="muted", rx=2)
    for i in range(1, 4):
        f.line(70 + 15 * i, 260, 70 + 15 * i, 320, "muted", 0.8)
        f.line(70, 260 + 15 * i, 130, 260 + 15 * i, "muted", 0.8)
    f.text(100, 333, "h × w × c (VAE)", 10.5, fill="muted")
    f.arrow(132, 290, 250, 290, "patch size p (DiT: p = 2)", ly=258, fs=10)
    f.box(250, 274, 70, 32, "p²c", fs=11.5, fill="goldsoft", stroke="gold")
    f.arrow(320, 290, 430, 290, "linear p²c → D", ly=258, fs=10)
    f.box(430, 274, 46, 32, "D", fs=12, fill="bluesoft")
    f.arrow(476, 290, 530, 290)
    f.matrix(530, 262, 60, 56, "", dims="N × D, N = hw/p²", rows=4, cols=6)
    f.lines(690, 284, ["DiT-XL/2 at 256²:", "latent 32 × 32 × 4,", "N = 16 × 16 = 256,", "p²c = 16 → D = 1152"], 10.5, "middle", "muted", lh=14)
    return f


# ---------------------------------------------------------------- 4 block
@reg("block", "One pre-LN block with shapes. Each sub-layer reads a normalized copy of the stream, computes something, and adds the result back. The MLP widens each token to 4D and back; attention is the only step whose output at one position depends on other positions.")
def block():
    f = Fig("block", 760, 300, "A pre-LN transformer block with tensor shapes")
    xs = 70
    f.text(xs, 24, "X  (N × D)", 12, bold=True)
    f.line(xs, 36, xs, 60)
    bh = 40
    # attention
    y1 = 60
    f.box(130, y1, 64, bh, "LN", fs=12)
    f.box(214, y1, 230, bh, "multi-head self-attention", sub="Q, K, V = XW_{Q}, XW_{K}, XW_{V}; out · W_{O}", fs=12, subfs=9)
    f.line(xs, y1 + bh / 2, 130, y1 + bh / 2, head=True)
    f.arrow(194, y1 + bh / 2, 214, y1 + bh / 2)
    f.text(470, y1 + bh / 2 - 11, "N × D", 10.5, "start", "muted")
    yp = y1 + bh + 26
    f.poly([(444, y1 + bh / 2), (530, y1 + bh / 2), (530, yp), (xs + 10, yp)], head=True)
    f.line(xs, 60, xs, yp - 10)
    f.circle(xs, yp, 10, "+")
    # mlp
    y2 = yp + 26
    f.box(130, y2, 64, bh, "LN", fs=12)
    f.box(214, y2, 70, bh, "Linear", sub="D → 4D", fs=11, subfs=9)
    f.box(296, y2, 60, bh, "GELU", fs=11)
    f.box(368, y2, 76, bh, "Linear", sub="4D → D", fs=11, subfs=9)
    f.line(xs, y2 + bh / 2, 130, y2 + bh / 2, head=True)
    f.arrow(194, y2 + bh / 2, 214, y2 + bh / 2)
    f.arrow(284, y2 + bh / 2, 296, y2 + bh / 2)
    f.arrow(356, y2 + bh / 2, 368, y2 + bh / 2)
    f.text(320, y2 - 8, "N × 4D", 10.5, "middle", "muted")
    f.text(470, y2 + bh / 2 - 11, "N × D", 10.5, "start", "muted")
    yp2 = y2 + bh + 26
    f.poly([(444, y2 + bh / 2), (530, y2 + bh / 2), (530, yp2), (xs + 10, yp2)], head=True)
    f.line(xs, yp + 10, xs, yp2 - 10)
    f.circle(xs, yp2, 10, "+")
    f.line(xs, yp2 + 10, xs, yp2 + 30, head=True)
    f.text(xs, yp2 + 44, "X'  (N × D)", 12, bold=True)
    # per-token note
    f.lines(640, 150, ["per token:", "LN, MLP, +", "across tokens:", "attention only"], 11, fill="muted", italic=True)
    f.text(640, 230, "params per block ≈ 12D²", 11, fill="muted")
    f.text(640, 248, "(4D² attention + 8D² MLP)", 10.5, fill="muted")
    return f


# ---------------------------------------------------------------- 5 pre vs post LN
@reg("prepost", "Where the normalization sits. Post-LN (original transformer) normalizes after each residual addition, so every gradient path passes through a LayerNorm at every layer. Pre-LN (GPT-2 onward) normalizes the sub-layer input and leaves an identity path from the loss straight down the stream; it needs a final LN before the head.")
def prepost():
    f = Fig("prepost", 760, 250, "Post-LN versus pre-LN placement of layer normalization in a block")
    def chain(x, title, pre):
        f.text(x + 150, 20, title, 13, bold=True)
        xs = x + 30
        y = 50
        f.text(xs, y - 8, "x", 12, bold=True)
        f.line(xs, y, xs, y + 24)
        for i, name in enumerate(["sub-layer F (attn or MLP)", "sub-layer F (attn or MLP)"][:1]):
            pass
        # one sublayer drawn
        if pre:
            f.box(xs + 40, y + 20, 50, 30, "LN", fs=11)
            f.box(xs + 100, y + 20, 150, 30, "F  (attention / MLP)", fs=11)
            f.line(xs, y + 35, xs + 40, y + 35, head=True)
            f.arrow(xs + 90, y + 35, xs + 100, y + 35)
            f.poly([(xs + 250, y + 35), (xs + 268, y + 35), (xs + 268, y + 90), (xs + 10, y + 90)], head=True)
            f.line(xs, y + 24, xs, y + 80)
            f.circle(xs, y + 90, 10, "+")
            f.line(xs, y + 100, xs, y + 150, head=True)
            f.text(xs, y + 162, "x' = x + F(LN(x))", 11.5, bold=True)
            f.text(xs + 150, y + 125, "identity path: ∂x'/∂x = I + ∂F/∂x", 10.5, "middle", "muted", italic=True)
        else:
            f.box(xs + 40, y + 20, 150, 30, "F  (attention / MLP)", fs=11)
            f.line(xs, y + 35, xs + 40, y + 35, head=True)
            f.poly([(xs + 190, y + 35), (xs + 208, y + 35), (xs + 208, y + 90), (xs + 10, y + 90)], head=True)
            f.line(xs, y + 24, xs, y + 80)
            f.circle(xs, y + 90, 10, "+")
            f.arrow(xs, y + 100, xs, y + 116)
            f.box(xs - 25, y + 116, 50, 28, "LN", fs=11)
            f.line(xs, y + 144, xs, y + 150, head=True)
            f.text(xs, y + 162, "x' = LN(x + F(x))", 11.5, bold=True)
            f.text(xs + 150, y + 130, "every path crosses a LN Jacobian", 10.5, "middle", "muted", italic=True)
    chain(20, "Post-LN (2017)", False)
    chain(400, "Pre-LN (GPT-2, Llama, ViT, DiT)", True)
    return f


# ---------------------------------------------------------------- 6 attention
@reg("attention", "Scaled dot-product attention as matrices. Three projections of the same input give queries, keys and values; the score matrix compares every query with every key; a row-wise softmax turns each row into weights that sum to one; the output of each query is the weighted average of the value rows.")
def attention():
    f = Fig("attention", 760, 330, "Attention computed as matrix products: projections, scores, softmax, weighted sum of values")
    f.matrix(20, 110, 60, 90, "X", dims="N × D", rows=5, cols=4)
    # projections
    f.arrow(82, 155, 120, 60, "W_{Q}", lx=92, ly=96, fs=11, lcolor="ink")
    f.arrow(82, 155, 120, 150, "W_{K}", lx=100, ly=170, fs=11, lcolor="ink")
    f.arrow(82, 155, 120, 240, "W_{V}", lx=92, ly=214, fs=11, lcolor="ink")
    f.matrix(120, 20, 50, 60, "Q", dims="N × d", rows=5, cols=3, fill="goldsoft", stroke="gold")
    f.matrix(120, 110, 50, 60, "K", dims="M × d", rows=5, cols=3)
    f.matrix(120, 200, 50, 60, "V", dims="M × d_{v}", rows=5, cols=3, fill="greensoft", stroke="green")
    # scores
    f.arrow(172, 50, 236, 95)
    f.arrow(172, 140, 236, 110)
    f.matrix(236, 70, 70, 70, "S", dims="N × M", rows=5, cols=5)
    f.text(271, 56, "QKᵀ / √d", 11, fill="ink")
    f.arrow(308, 105, 352, 105)
    f.text(330, 60, "softmax", 11, fill="ink")
    f.text(330, 74, "per row", 9.5, fill="muted")
    f.matrix(352, 70, 70, 70, "A", dims="N × M, rows sum to 1", rows=5, cols=5,
             shade=lambda i, j: "goldsoft" if i == 1 else None)
    f.arrow(424, 105, 470, 105)
    f.text(447, 60, "A · V", 11, fill="ink")
    f.arrow(172, 230, 470, 230)
    f.poly([(470, 230), (490, 230), (490, 140)], head=True)
    f.matrix(470, 70, 50, 70, "out", dims="N × d_{v}", rows=5, cols=3, fill="greensoft", stroke="green")
    f.arrow(522, 105, 560, 105)
    f.text(541, 60, "W_{O}", 11, fill="ink")
    f.matrix(560, 70, 50, 70, "", dims="N × D", rows=5, cols=4)
    # row explanation
    f.lines(655, 215, ["self-attention: M = N, K, V from X", "cross-attention: K, V from a context C", "", "row i of A: weights of query i over keys", "row i of out = Σ_{j} a_{ij} v_{j}", "", "cost N·M·d (S) + N·M·d_{v} (A·V);", "N·M scores per head in memory"], 10, "middle", "muted", lh=14)
    return f


# ---------------------------------------------------------------- 7 heads
@reg("heads", "Multi-head attention. The D channels of Q, K and V are split into h blocks of width d_h = D/h; each block runs its own attention with its own N×N weights; the h outputs are concatenated along channels and mixed by W_O. Parameters and FLOPs do not depend on h.")
def heads():
    f = Fig("heads", 760, 300, "Multi-head attention: split channels into heads, attend separately, concatenate, project")
    f.matrix(20, 110, 60, 90, "X", dims="N × D", rows=5, cols=4)
    f.arrow(82, 155, 120, 155)
    f.text(100, 172, "W_{Q,K,V}", 10.5, fill="ink")
    # QKV with colored column blocks for heads
    def shade(i, j):
        return ["goldsoft", "bluesoft", "greensoft", "redsoft"][j // 2]
    f.matrix(120, 95, 96, 120, "", dims="Q, K, V each N × D", rows=5, cols=8, shade=shade)
    f.text(168, 84, "column blocks = heads", 10.5, fill="muted")
    f.text(168, 246, "view(B,N,h,d_{h}).transpose(1,2)", 9, fill="muted", family="mono")
    # per head attention boxes
    cols = ["goldsoft", "bluesoft", "greensoft", "redsoft"]
    strokes = ["gold", "blue", "green", "red"]
    for k in range(4):
        y = 40 + k * 60
        f.arrow(218, 110 + k * 25, 270, y + 18)
        f.box(270, y, 150, 36, f"head {k + 1}: Attn(Q_{{{k + 1}}}, K_{{{k + 1}}}, V_{{{k + 1}}})", sub=f"N × N weights, out N × d_{{h}}", fill=cols[k], stroke=strokes[k], fs=10.5, subfs=9)
        f.arrow(420, y + 18, 470, 110 + k * 25)
    f.matrix(470, 95, 96, 120, "", dims="concat: N × D", rows=5, cols=8, shade=shade)
    f.arrow(568, 155, 610, 155, "W_{O}", ly=141, fs=11, lcolor="ink")
    f.matrix(610, 110, 60, 90, "out", dims="N × D", rows=5, cols=4)
    f.lines(380, 276, ["h = 12, d_{h} = 64 in GPT-2 / BERT / ViT-B;  h = 32, d_{h} = 128 in Llama 2 7B",
                       "W_{O} lets each head write into all D channels; the heads' contributions add"], 10.5, "middle", "muted")
    return f


# ---------------------------------------------------------------- 8 uses
@reg("uses", "The three uses of the same operation differ only in where Q, K, V come from and which scores are allowed (white = allowed, grey = set to −∞ before the softmax). Self-attention: all from X, all allowed. Causal: all from X, query i sees keys j ≤ i. Cross-attention: Q from the sequence being updated, K and V from another sequence (text, encoder output), no mask needed.")
def uses():
    f = Fig("uses", 760, 290, "Self-attention, causal self-attention and cross-attention: sources of Q, K, V and the allowed score pattern")
    def panel(x, title, qsrc, kvsrc, rows, cols, masked, note):
        f.text(x + 115, 20, title, 13, bold=True)
        f.box(x, 40, 100, 30, qsrc, fs=10.5, fill="goldsoft", stroke="gold")
        f.box(x + 130, 40, 100, 30, kvsrc, fs=10.5, fill="bluesoft")
        f.text(x + 50, 84, "Q", 11, fill="muted")
        f.text(x + 180, 84, "K, V", 11, fill="muted")
        gx, gy, cell = x + 65, 100, 20
        f.arrow(x + 50, 70, x + 50, 100 + rows * cell / 2 - 2)
        f.arrow(x + 180, 70, gx + cols * cell - 6, 98)
        f.matrix(gx, gy, cols * cell, rows * cell, "", rows=rows, cols=cols, fill="white",
                 shade=lambda i, j: "greysoft" if masked(i, j) else None)
        f.text(gx + cols * cell / 2, gy + rows * cell + 12, f"scores {rows} × {cols}", 10, fill="muted")
        f.text(gx - 8, gy + rows * cell / 2, "queries", 9.5, "end", "muted", rotate=-90)
        f.text(gx + 2, gy - 6, "keys →", 9.5, "start", "muted")
        f.lines(x + 115, 252, note, 10, fill="muted", italic=True, lh=13)
    panel(10, "Self (encoder)", "X  (N × D)", "X  (N × D)", 6, 6, lambda i, j: False,
          ["BERT, ViT, DiT: every token", "reads every token"])
    panel(265, "Causal (decoder)", "X  (N × D)", "X  (N × D)", 6, 6, lambda i, j: j > i,
          ["GPT: token i reads only j ≤ i;", "one pass trains N predictions"])
    panel(520, "Cross", "X  (N × D)", "C  (M × D_{c})", 6, 4, lambda i, j: False,
          ["SD U-Net: image queries,", "77 text keys and values"])
    return f


# ---------------------------------------------------------------- 9 rope
@reg("rope", "RoPE rotates each two-channel pair of a query by its position angle mθ and of a key by nθ. Because rotations compose by adding angles, the dot product of the rotated pair depends only on the offset n − m: the score is relative while the inputs stay absolute.")
def rope():
    f = Fig("rope", 760, 250, "Rotary position embedding: query and key pairs rotated by their position angles; the score depends on the angle difference")
    import math
    def axes(cx, cy, r, title):
        f.line(cx - r - 10, cy, cx + r + 10, cy, "line", 1)
        f.line(cx, cy + r + 10, cx, cy - r - 10, "line", 1)
        f.el.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" style="fill:none;stroke:var(--line);stroke-width:1;stroke-dasharray:3 3"/>')
        f.text(cx, cy - r - 30, title, 12, bold=True)
    def vec(cx, cy, r, ang, label, color, lab_off=14):
        x2, y2 = cx + r * math.cos(ang), cy - r * math.sin(ang)
        f.line(cx, cy, x2, y2, color, 2, head=True)
        f.text(cx + (r + lab_off) * math.cos(ang), cy - (r + lab_off) * math.sin(ang), label, 11, fill=color)
    # panel 1: raw q, k
    axes(110, 130, 60, "pair (q_{1}, q_{2}), (k_{1}, k_{2})")
    vec(110, 130, 60, 0.35, "q", "gold")
    vec(110, 130, 60, 1.1, "k", "blue")
    # panel 2: rotated
    axes(380, 130, 60, "rotate by position: q by mθ, k by nθ")
    vec(380, 130, 60, 0.35 + 0.9, "R_{mθ} q", "gold", 20)
    vec(380, 130, 60, 1.1 + 1.5, "R_{nθ} k", "blue", 20)
    f.el.append(f'<path d="M{380 + 40*math.cos(0.35)},{130 - 40*math.sin(0.35)} A40,40 0 0,0 {380 + 40*math.cos(1.25)},{130 - 40*math.sin(1.25)}" style="fill:none;stroke:var(--gold);stroke-width:1;stroke-dasharray:3 2"/>')
    f.el.append(f'<path d="M{380 + 30*math.cos(1.1)},{130 - 30*math.sin(1.1)} A30,30 0 0,0 {380 + 30*math.cos(2.6)},{130 - 30*math.sin(2.6)}" style="fill:none;stroke:var(--blue);stroke-width:1;stroke-dasharray:3 2"/>')
    # panel 3: formula
    f.lines(630, 110, ["⟨R_{mθ} q, R_{nθ} k⟩ = qᵀ R_{(n−m)θ} k", "", "angle between them:", "(n − m)θ + (original angle)", "", "same offset ⇒ same score"], 11.5, fill="ink", lh=17)
    f.lines(630, 205, ["d/2 pairs per head, θ_{i} = 10000^{−2i/d}:", "fast pairs resolve near offsets, slow pairs far ones"], 10, fill="muted", lh=13)
    return f


# ---------------------------------------------------------------- 10 kv cache
@reg("kvcache", "Decoding with a KV cache. Under a causal mask the keys and values of earlier positions never change, so each layer stores them; a new token computes only its own q, k, v, appends k and v, and attends its single query over the whole cache. Prefill fills the cache from the prompt in one pass.")
def kvcache():
    f = Fig("kvcache", 760, 250, "KV cache during decoding: cached keys and values grow by one row per step; the new query attends over all of them")
    f.text(130, 22, "step t: new token only", 12.5, bold=True)
    f.box(40, 40, 180, 32, "x_{t}  (1 × D)", fs=11.5, fill="goldsoft", stroke="gold")
    f.arrow(130, 72, 130, 96, "W_{Q}, W_{K}, W_{V}", lx=200, ly=84, fs=10.5, lcolor="ink")
    f.box(40, 96, 50, 30, "q_{t}", fs=11.5, fill="goldsoft", stroke="gold")
    f.box(105, 96, 50, 30, "k_{t}", fs=11.5, fill="bluesoft")
    f.box(170, 96, 50, 30, "v_{t}", fs=11.5, fill="greensoft", stroke="green")
    # cache
    f.text(430, 22, "cache per layer (B, h_{kv}, t, d_{h})", 12.5, bold=True)
    f.matrix(330, 40, 90, 120, "K", dims="keys 1..t−1", rows=6, cols=3)
    f.matrix(450, 40, 90, 120, "V", dims="values 1..t−1", rows=6, cols=3, fill="greensoft", stroke="green")
    f.rect(330, 160, 90, 20, fill="bluesoft", stroke="blue", rx=2, dash=True)
    f.rect(450, 160, 90, 20, fill="greensoft", stroke="green", rx=2, dash=True)
    f.text(375, 170, "k_{t} appended", 9.5)
    f.text(495, 170, "v_{t} appended", 9.5)
    f.poly([(155, 111), (260, 111), (260, 170), (328, 170)], head=True, color="blue")
    f.poly([(220, 111), (240, 111), (240, 190), (448, 190), (448, 170)], head=True, color="green")
    # attention of q over cache
    f.poly([(65, 126), (65, 215), (600, 215)], head=True, color="gold")
    f.text(330, 228, "q_{t} · [K; k_{t}]ᵀ → softmax → · [V; v_{t}]   (1 × t scores, no mask needed)", 10.5, fill="ink")
    f.box(600, 196, 140, 38, "output for token t", sub="then next layer", fs=11, subfs=9)
    f.lines(650, 100, ["why exact: with a causal mask,", "k_{j}, v_{j} depend only on", "tokens ≤ j, so they never change", "", "size: 2 · L · h_{kv} · d_{h} · t numbers"], 10.5, fill="muted", lh=15)
    return f



# ---------------------------------------------------------------- 11 gqa
@reg("gqa", "Standard multi-head attention gives every query head its own key and value head. Grouped-query attention shares one K/V head among a group of query heads; multi-query attention shares a single K/V head among all of them. Only the K and V side shrinks, so the cache and the K/V projections shrink by h_kv / h while the query side is unchanged.")
def gqa():
    f = Fig("gqa", 760, 225, "Multi-head, grouped-query and multi-query attention: how query heads map to key-value heads")
    def panel(x, title, hq, hkv, note):
        f.text(x + 115, 20, title, 13, bold=True)
        f.text(x + 115, 38, note, 10.5, fill="muted", italic=True)
        cw = 220 / hq
        for i in range(hq):
            f.rect(x + i * cw + 2, 60, cw - 4, 30, fill="goldsoft", stroke="gold", rx=3)
        f.text(x + 115, 52, "query heads (h = 8)", 10, fill="muted")
        kw = 220 / hkv
        for j in range(hkv):
            f.rect(x + j * kw + 2, 140, kw - 4, 30, fill="bluesoft", stroke="blue", rx=3)
        f.text(x + 115, 182, f"K, V heads (h_{{kv}} = {hkv})", 10, fill="muted")
        for i in range(hq):
            j = i * hkv // hq
            f.line(x + i * cw + cw / 2, 90, x + j * kw + kw / 2, 140, "muted", 1)
        f.text(x + 115, 206, f"KV cache = {hkv}/8 of MHA", 11, fill="ink")
    panel(10, "MHA", 8, 8, "Llama 2 7B")
    panel(270, "GQA", 8, 2, "Llama 2 70B, Llama 3, Mistral (groups of 4 or 8)")
    panel(530, "MQA", 8, 1, "PaLM, Falcon")
    return f


# ---------------------------------------------------------------- 12 flash
@reg("flash", "FlashAttention never materializes the N×N score matrix. Standard attention writes S and P to GPU memory (HBM) and reads them back, which is memory-bound. FlashAttention loads a block of queries and loops over blocks of keys and values inside fast on-chip SRAM, keeping a running max, running normalizer and running output per query row (online softmax), and writes only the final output. Same result, O(N) extra memory instead of O(N²), far less traffic; FLOPs unchanged.")
def flash():
    f = Fig("flash", 760, 330, "Standard attention versus FlashAttention: what is written to HBM and what stays in SRAM")
    # left: standard
    f.text(180, 20, "Standard attention", 13, bold=True)
    f.rect(20, 36, 320, 262, fill="greysoft", stroke="muted", rx=8, dash=True)
    f.text(40, 50, "HBM (GPU memory, ~2 TB/s)", 10.5, "start", "muted", italic=True)
    f.matrix(40, 70, 40, 60, "Q", rows=4, cols=2, fill="goldsoft", stroke="gold")
    f.matrix(90, 70, 40, 60, "K", rows=4, cols=2)
    f.matrix(140, 70, 40, 60, "V", rows=4, cols=2, fill="greensoft", stroke="green")
    f.matrix(210, 60, 80, 80, "S = QKᵀ", dims="N × N, written", rows=5, cols=5, fill="redsoft", stroke="red", fs=11)
    f.matrix(210, 180, 80, 80, "P = softmax", dims="N × N, written and read", rows=5, cols=5, fill="redsoft", stroke="red", fs=10.5)
    f.matrix(40, 190, 60, 60, "O = PV", rows=4, cols=2, fill="greensoft", stroke="green", fs=10.5)
    f.arrow(182, 100, 208, 100)
    f.arrow(250, 152, 250, 178)
    f.arrow(208, 220, 102, 220)
    f.text(180, 286, "4 passes over N² numbers: memory-bound", 10.5, fill="red")
    # right: flash
    f.text(560, 20, "FlashAttention", 13, bold=True)
    f.rect(380, 36, 360, 100, fill="greysoft", stroke="muted", rx=8, dash=True)
    f.text(400, 50, "HBM", 10.5, "start", "muted", italic=True)
    f.matrix(400, 66, 40, 60, "Q", rows=4, cols=2, fill="goldsoft", stroke="gold")
    f.matrix(450, 66, 40, 60, "K", rows=4, cols=2)
    f.matrix(500, 66, 40, 60, "V", rows=4, cols=2, fill="greensoft", stroke="green")
    f.matrix(660, 66, 60, 60, "O", rows=4, cols=2, fill="greensoft", stroke="green")
    f.text(690, 56, "only O is written", 9.5, fill="muted")
    f.rect(380, 172, 360, 126, fill="white", stroke="blue", rx=8)
    f.text(400, 186, "SRAM (on-chip, ~19 TB/s, ~192 KB per SM)", 10.5, "start", "blue", italic=True)
    f.matrix(400, 202, 40, 30, "Q_{i}", rows=2, cols=2, fill="goldsoft", stroke="gold", fs=10)
    f.matrix(450, 202, 40, 30, "K_{j}", rows=2, cols=2, fs=10)
    f.matrix(500, 202, 40, 30, "V_{j}", rows=2, cols=2, fill="greensoft", stroke="green", fs=10)
    f.matrix(560, 202, 40, 30, "S_{ij}", rows=2, cols=2, fill="bluesoft", fs=10)
    f.lines(670, 217, ["per query row: m, ℓ, o", "rescaled by e^{m−m'}"], 10, fill="ink", lh=13)
    f.text(560, 256, "loop j over K/V blocks; Q_{i} rows never leave SRAM", 10, fill="muted")
    f.text(560, 274, "backward: recompute S_{ij} from Q, K and the saved log-sum-exp", 10, fill="muted")
    f.arrow(420, 128, 420, 200, dash=True)
    f.arrow(470, 128, 470, 200, dash=True)
    f.text(495, 152, "block j of K, V", 9.5, "start", "muted")
    f.arrow(520, 128, 520, 200, dash=True)
    f.arrow(690, 200, 690, 128, dash=True)
    f.text(560, 316, "no N × N tensor anywhere: memory O(N), FLOPs still O(N²d)", 10.5, fill="blue")
    return f


# ---------------------------------------------------------------- 13 vit
@reg("vit", "The Vision Transformer. Patchify, embed, prepend a learned class token, add learned position embeddings, run L standard pre-LN encoder blocks with bidirectional attention, and classify from the class token (or from the mean of the patch tokens). Nothing in the blocks is image-specific.")
def vit():
    f = Fig("vit", 780, 300, "Vision Transformer: image to patches to tokens, class token and positions, encoder blocks, classification head")
    f.rect(20, 60, 90, 90, fill="greysoft", stroke="muted", rx=2)
    for i in range(1, 3):
        f.line(20 + 30 * i, 60, 20 + 30 * i, 150, "muted", 0.8)
        f.line(20, 60 + 30 * i, 110, 60 + 30 * i, "muted", 0.8)
    f.text(65, 164, "224 × 224 × 3", 10, fill="muted")
    f.arrow(112, 105, 150, 105, "P = 16", ly=92, fs=10)
    # patch tokens row
    f.text(150, 46, "flatten + linear (768 → D)", 10, "start", "muted")
    for k in range(9):
        f.rect(150 + k * 26, 90, 22, 30, fill="bluesoft", stroke="blue", rx=3)
    f.rect(150 + 9 * 26 + 6, 90, 22, 30, fill="goldsoft", stroke="gold", rx=3)
    f.text(150 + 9 * 26 + 17, 105, "cls", 9)
    f.text(285, 150, "196 patch tokens + 1 class token = 197 × D", 10, "start", "muted")
    # position
    f.text(285, 180, "+ learned position embedding (197 × D)", 10, "start", "muted")
    f.circle(265, 180, 9, "+", fs=12)
    f.arrow(265, 120, 265, 170)
    f.arrow(265, 189, 265, 210)
    # encoder
    f.box(150, 210, 230, 40, "encoder block × L", sub="LN → MHA (bidirectional) → + ; LN → MLP → +", fs=12, subfs=9)
    f.arrow(380, 230, 430, 230, "197 × D", ly=216, fs=10)
    f.box(430, 212, 90, 36, "take cls", sub="or mean-pool", fs=11, subfs=9, fill="goldsoft", stroke="gold")
    f.arrow(520, 230, 560, 230, "D", ly=216, fs=10)
    f.box(560, 212, 90, 36, "LN + MLP head", fs=10.5)
    f.arrow(650, 230, 690, 230)
    f.box(690, 212, 80, 36, "1000 classes", fs=10.5)
    f.lines(640, 90, ["ViT-B/16: L = 12, D = 768, h = 12", "≈ 86M parameters, 197 tokens", "ViT-L/16: L = 24, D = 1024, 304M", "", "no locality built in: needs", "large pretraining or strong augmentation"], 10.5, fill="muted", lh=14)
    return f


# ---------------------------------------------------------------- 14 dit
@reg("dit", "The Diffusion Transformer. Top: the whole model, a ViT-style encoder on patchified VAE latents whose output is unpatchified back into a latent-shaped prediction. Bottom: one DiT block with adaLN-Zero. The conditioning vector c (timestep + class, or pooled text) is mapped by a zero-initialized linear layer to a shift, scale and gate for each sub-layer; at initialization all six are zero, so every block is the identity and the network starts as a no-op.")
def dit():
    f = Fig("dit", 780, 470, "DiT: patchified noisy latent through adaLN-Zero blocks to an unpatchified noise prediction; block detail with shift, scale and gate from the conditioning vector")
    # top: whole model
    y = 40
    f.box(20, y, 110, 44, "noisy latent z_{t}", sub="32 × 32 × 4", fs=11, subfs=9.5)
    f.arrow(130, y + 22, 170, y + 22, "patchify p = 2", ly=y - 14, fs=9.5)
    f.box(170, y, 110, 44, "256 tokens × D", sub="+ 2D sin-cos pos", fs=11, subfs=9.5, fill="bluesoft")
    f.arrow(280, y + 22, 320, y + 22)
    f.box(320, y - 6, 150, 56, "DiT block × L", sub="(28 for XL, D = 1152)", fs=12, subfs=9.5)
    f.arrow(470, y + 22, 510, y + 22)
    f.box(510, y, 120, 44, "LN + linear", sub="D → p·p·2c per token", fs=11, subfs=9.5)
    f.arrow(630, y + 22, 668, y + 22, "unpatchify", ly=y - 14, fs=9.5)
    f.box(668, y, 100, 44, "ε̂ (and Σ)", sub="32 × 32 × 4 (× 2)", fs=11, subfs=9.5)
    # conditioning
    f.box(300, 120, 190, 36, "c = emb(t) + emb(y)", sub="timestep + class (or pooled text)", fs=11, subfs=9, fill="goldsoft", stroke="gold")
    f.arrow(395, 120, 395, 86, color="gold")
    f.text(410, 104, "to every block", 9.5, "start", "muted", italic=True)
    f.text(250, 138, "bidirectional attention; no KV cache across steps", 9.5, "end", "muted", italic=True)
    # bottom: block detail
    f.line(20, 176, 760, 176, "line", 1)
    f.text(20, 192, "one DiT block (adaLN-Zero)", 12.5, "start", bold=True)
    xs = 70
    f.text(xs, 214, "x", 12, bold=True)
    f.line(xs, 222, xs, 236)
    bh = 34
    y1 = 236
    f.box(120, y1, 48, bh, "LN", fs=11)
    f.box(178, y1, 96, bh, "(1+γ_{1}) ⊙ · + β_{1}", fs=10.5, fill="goldsoft", stroke="gold")
    f.box(284, y1, 110, bh, "multi-head attn", fs=11)
    f.box(404, y1, 50, bh, "⊙ α_{1}", fs=11, fill="goldsoft", stroke="gold")
    for a, b in [(168, 178), (274, 284), (394, 404)]:
        f.arrow(a, y1 + bh / 2, b, y1 + bh / 2)
    f.line(xs, y1 + bh / 2, 120, y1 + bh / 2, head=True)
    yp = y1 + bh + 22
    f.poly([(454, y1 + bh / 2), (480, y1 + bh / 2), (480, yp), (xs + 10, yp)], head=True)
    f.line(xs, 236, xs, yp - 10)
    f.circle(xs, yp, 10, "+")
    y2 = yp + 22
    f.box(120, y2, 48, bh, "LN", fs=11)
    f.box(178, y2, 96, bh, "(1+γ_{2}) ⊙ · + β_{2}", fs=10.5, fill="goldsoft", stroke="gold")
    f.box(284, y2, 110, bh, "MLP (4D)", fs=11)
    f.box(404, y2, 50, bh, "⊙ α_{2}", fs=11, fill="goldsoft", stroke="gold")
    for a, b in [(168, 178), (274, 284), (394, 404)]:
        f.arrow(a, y2 + bh / 2, b, y2 + bh / 2)
    f.line(xs, y2 + bh / 2, 120, y2 + bh / 2, head=True)
    yp2 = y2 + bh + 22
    f.poly([(454, y2 + bh / 2), (480, y2 + bh / 2), (480, yp2), (xs + 10, yp2)], head=True)
    f.line(xs, yp + 10, xs, yp2 - 10)
    f.circle(xs, yp2, 10, "+")
    f.line(xs, yp2 + 10, xs, yp2 + 26, head=True)
    f.text(xs, yp2 + 38, "x'", 12, bold=True)
    # adaLN MLP on the right
    f.box(560, 214, 150, 34, "c", sub="D-vector, same for all tokens", fs=12, subfs=9, fill="goldsoft", stroke="gold")
    f.arrow(635, 248, 635, 272, "SiLU → Linear (D → 6D)", lx=700, ly=260, fs=9.5, anchor="middle")
    f.box(560, 272, 150, 50, ["γ_{1}, β_{1}, α_{1}", "γ_{2}, β_{2}, α_{2}"], fs=11, fill="goldsoft", stroke="gold")
    f.poly([(560, 290), (520, 290), (520, y1 + bh / 2 + 4), (456, y1 + bh / 2 + 4)], color="gold", head=False, dash=True)
    f.poly([(520, 290), (520, y2 + bh / 2 + 4), (456, y2 + bh / 2 + 4)], color="gold", head=False, dash=True)
    f.lines(635, 372, ["zero-initialized ⇒ γ = β = α = 0", "⇒ x' = x: block starts as identity;", "gates α learn first, then the branches"], 10.5, fill="ink", lh=14)
    f.lines(635, 430, ["vs. cross-attention to c: +15% FLOPs,", "worse FID in the DiT paper"], 10, fill="muted", lh=13, italic=True)
    return f


# ---------------------------------------------------------------- 15 unet
@reg("unet", "Stable Diffusion's U-Net backbone and where the two conditions enter. The timestep goes into every residual block as an added vector (FiLM-style); the text goes into every attention block as keys and values for cross-attention, with the image positions as queries. Skip connections concatenate encoder features into the decoder at the same resolution.")
def unet():
    f = Fig("unet", 780, 400, "Latent-diffusion U-Net: encoder and decoder levels with skip connections; zoom on a level with a ResBlock conditioned on t and a transformer block with self-attention, cross-attention to text and an MLP")
    levels = [("64² × 320", 0), ("32² × 640", 1), ("16² × 1280", 2)]
    lw, lh, gap = 130, 38, 54
    xl, xr = 30, 420
    for name, i in levels:
        y = 30 + i * gap
        f.box(xl + i * 40, y, lw, lh, name, sub="ResBlock + Transformer", fs=10.5, subfs=8.5)
        f.box(xr - i * 40, y, lw, lh, name, sub="ResBlock + Transformer", fs=10.5, subfs=8.5)
        f.arrow(xl + i * 40 + lw + 2, y + lh / 2, xr - i * 40 - 2, y + lh / 2, "skip: concat" if i == 0 else None, ly=y + lh / 2 - 8, fs=9.5, dash=True, color="muted")
        if i < 2:
            f.arrow(xl + i * 40 + lw / 2, y + lh, xl + (i + 1) * 40 + lw / 2, y + gap)
            f.arrow(xr - (i + 1) * 40 + lw / 2, y + gap, xr - i * 40 + lw / 2, y + lh)
    f.box(xl + 2 * 40 + lw / 2 + 60, 30 + 3 * gap + 2, 120, 32, "mid 8² × 1280", fs=10.5)
    f.arrow(xl + 2 * 40 + lw / 2, 30 + 2 * gap + lh, xl + 2 * 40 + lw / 2 + 62, 30 + 3 * gap + 18)
    f.arrow(xl + 2 * 40 + lw / 2 + 180, 30 + 3 * gap + 18, xr - 2 * 40 + lw / 2, 30 + 2 * gap + lh)
    f.text(xl + lw / 2, 18, "z_{t} in (64 × 64 × 4)", 10.5, fill="muted")
    f.text(xr + lw / 2, 18, "ε̂ out (64 × 64 × 4)", 10.5, fill="muted")
    # conditions
    f.box(600, 40, 160, 34, "timestep t → emb(t)", fs=10.5, fill="goldsoft", stroke="gold")
    f.box(600, 92, 160, 34, "text → CLIP encoder", sub="77 tokens × 768", fs=10.5, subfs=9, fill="bluesoft")
    f.text(680, 142, "into every level ↓", 9.5, fill="muted", italic=True)
    # zoom
    f.line(20, 236, 760, 236, "line", 1)
    f.text(20, 252, "one level, zoomed", 12.5, "start", bold=True)
    y = 272
    f.box(20, y, 170, 60, "ResBlock", sub="conv → + Linear(emb(t)) → conv", fs=11.5, subfs=9, fill="panel")
    f.poly([(680, 74), (770, 74), (770, 228), (105, 228), (105, y - 2)], color="gold", head=True, dash=True)
    f.arrow(190, y + 30, 230, y + 30, "flatten", ly=y + 16, fs=9.5)
    f.text(210, y + 44, "(HW) × C", 9, fill="muted")
    f.rect(230, y - 12, 420, 84, fill="none", stroke="muted", rx=8, dash=True)
    f.text(240, y - 2, "transformer block (\"spatial transformer\")", 9.5, "start", "muted", italic=True)
    f.box(240, y + 8, 110, 44, "self-attention", sub="image ↔ image", fs=11, subfs=9)
    f.arrow(350, y + 30, 372, y + 30)
    f.box(372, y + 8, 130, 44, "cross-attention", sub="Q: image, K,V: text", fs=11, subfs=9, fill="bluesoft")
    f.arrow(502, y + 30, 524, y + 30)
    f.box(524, y + 8, 110, 44, "MLP (GEGLU)", sub="each with residual", fs=11, subfs=9)
    f.poly([(680, 126), (700, 126), (700, 232), (437, 232), (437, y + 6)], color="blue", head=True, dash=True)
    f.arrow(650, y + 30, 690, y + 30, "reshape", ly=y + 16, fs=9.5)
    f.box(690, y + 8, 70, 44, "C × H × W", fs=10)
    f.lines(390, 372, ["down ×2 between encoder levels (strided conv), up ×2 in the decoder; skips concatenate same-resolution features",
                       "at the 64² level of SD v1: 4096 queries × 77 keys per head (not 4096²); 8 heads of width 40"], 10, fill="muted", lh=14)
    return f


# ---------------------------------------------------------------- 16 mmdit
@reg("mmdit", "The MMDiT block of Stable Diffusion 3 and FLUX. Text and image tokens keep separate weights (their own modulation, Q/K/V, output projection and MLP), but their queries, keys and values are concatenated into one joint attention over all M + N tokens, so each stream can read the other at every layer. The weight matrix has four blocks: text→text, text→image, image→text and image→image.")
def mmdit():
    f = Fig("mmdit", 780, 400, "MMDiT block: two token streams with separate weights joined by one attention over the concatenated sequence")
    def stream(x, name, color, stroke, M):
        f.text(x + 90, 22, name, 12.5, bold=True)
        f.text(x + 90, 38, M, 10, fill="muted")
        f.box(x + 20, 52, 140, 30, "LN + adaLN mod", fs=10.5, fill=color, stroke=stroke)
        f.arrow(x + 90, 82, x + 90, 100)
        f.box(x + 20, 100, 140, 30, "own W_{Q}, W_{K}, W_{V}", fs=10.5, fill=color, stroke=stroke)
        f.arrow(x + 90, 130, x + 90, 160)
        # after attention
        f.arrow(x + 90, 236, x + 90, 256)
        f.box(x + 20, 256, 140, 30, "own W_{O}, + residual", fs=10.5, fill=color, stroke=stroke)
        f.arrow(x + 90, 286, x + 90, 304)
        f.box(x + 20, 304, 140, 30, "LN + mod → own MLP → +", fs=10, fill=color, stroke=stroke)
    stream(40, "text stream  c", "bluesoft", "blue", "M tokens (CLIP + T5 in SD3)")
    stream(300, "image stream  x", "goldsoft", "gold", "N latent patches")
    f.box(40, 160, 400, 76, "", fill="white", stroke="ink")
    f.text(240, 176, "joint attention over [c ; x]  (M + N tokens)", 12, bold=True)
    f.text(240, 196, "softmax( [Q_{c}; Q_{x}] [K_{c}; K_{x}]ᵀ / √d ) [V_{c}; V_{x}]", 11)
    f.text(240, 218, "split the output back into M text rows and N image rows", 10, fill="muted")
    # weight matrix blocks
    bx, by, bs = 540, 100, 170
    f.rect(bx, by, bs, bs, fill="white", stroke="ink", rx=2)
    m = int(bs * 0.3)
    f.rect(bx, by, m, m, fill="bluesoft", stroke="blue", rx=0)
    f.rect(bx + m, by, bs - m, m, fill="greysoft", stroke="muted", rx=0)
    f.rect(bx, by + m, m, bs - m, fill="greysoft", stroke="muted", rx=0)
    f.rect(bx + m, by + m, bs - m, bs - m, fill="goldsoft", stroke="gold", rx=0)
    f.text(bx + m / 2, by + m / 2, "c→c", 10)
    f.text(bx + m + (bs - m) / 2, by + m / 2, "text reads image", 10)
    f.lines(bx + m / 2, by + m + (bs - m) / 2, ["image", "reads", "text"], 10)
    f.text(bx + m + (bs - m) / 2, by + m + (bs - m) / 2, "x→x (self-attention)", 10)
    f.text(bx + bs / 2, by - 12, "(M + N) × (M + N) attention weights", 10.5, fill="muted")
    f.text(bx + bs / 2, by + bs + 14, "cross-attention keeps only the", 10, fill="muted")
    f.text(bx + bs / 2, by + bs + 28, "\"image reads text\" block", 10, fill="muted")
    f.lines(240, 364, ["modulation (adaLN) from timestep + pooled text; QK-RMSNorm for stability",
                       "FLUX: double-stream MMDiT blocks, then single-stream blocks with shared weights"], 10, fill="muted", lh=14)
    return f


# ---------------------------------------------------------------- 17 conditioning
@reg("conditioning", "Four ways to inject a condition into a transformer, drawn as what changes in one block. In-context: the condition becomes extra tokens in the same sequence. adaLN / FiLM: a vector modulates every token identically through shift, scale and gate. Cross-attention: tokens query a separate condition sequence. Joint attention: both sequences are attended together with separate weights per stream.")
def conditioning():
    f = Fig("conditioning", 780, 330, "Four conditioning mechanisms: in-context tokens, adaLN modulation, cross-attention, joint attention")
    def panel(x, title, note):
        f.text(x + 90, 20, title, 12.5, bold=True)
        f.lines(x + 90, 292, note, 9.5, fill="muted", lh=12.5, italic=True)
    # 1 in-context
    panel(10, "In-context tokens", ["DiT ablation, PixArt-Σ style,", "Flamingo's text; cost (N+M)²,", "one shared set of weights"])
    for k in range(5):
        f.rect(20 + k * 28, 60, 24, 26, fill="goldsoft", stroke="gold", rx=3)
    for k in range(2):
        f.rect(20 + (5 + k) * 28, 60, 24, 26, fill="bluesoft", stroke="blue", rx=3)
    f.text(76, 100, "x tokens", 9.5, fill="muted")
    f.text(188, 100, "c tokens", 9.5, fill="muted")
    f.arrow(110, 112, 110, 140)
    f.box(30, 140, 160, 36, "self-attention over all", sub="then MLP, as usual", fs=10.5, subfs=9)
    # 2 adaLN
    panel(205, "adaLN / FiLM", ["DiT, SD3 (timestep, class,", "pooled text); cheapest;", "one vector for all tokens"])
    f.box(215, 56, 80, 28, "c vector", fs=10, fill="bluesoft")
    f.arrow(255, 84, 255, 110, "MLP", lx=272, fs=9.5)
    f.box(215, 110, 80, 28, "γ, β, α", fs=10, fill="bluesoft")
    f.box(310, 56, 70, 28, "x tokens", fs=10, fill="goldsoft", stroke="gold")
    f.arrow(345, 84, 345, 140)
    f.box(305, 140, 80, 36, "LN·(1+γ)+β", sub="→ sublayer → ·α", fs=9.5, subfs=8.5)
    f.arrow(295, 124, 304, 150, color="blue", dash=True)
    # 3 cross
    panel(400, "Cross-attention", ["SD U-Net, PixArt-α, Wan,", "encoder–decoder; cost N·M;", "c is never updated"])
    f.box(410, 56, 80, 28, "x tokens", fs=10, fill="goldsoft", stroke="gold")
    f.box(500, 56, 80, 28, "c tokens", fs=10, fill="bluesoft")
    f.arrow(450, 84, 470, 140, "Q", lx=448, ly=116, fs=9.5)
    f.arrow(540, 84, 510, 140, "K, V", lx=548, ly=116, fs=9.5)
    f.box(410, 140, 160, 36, "cross-attention", sub="after self-attention, before MLP", fs=10.5, subfs=8.5)
    # 4 joint
    panel(595, "Joint attention", ["MMDiT (SD3, FLUX); both", "streams updated; cost (N+M)²;", "separate weights per stream"])
    f.box(605, 56, 80, 28, "x tokens", fs=10, fill="goldsoft", stroke="gold")
    f.box(695, 56, 80, 28, "c tokens", fs=10, fill="bluesoft")
    f.arrow(645, 84, 655, 140, "own QKV", lx=628, ly=116, fs=9)
    f.arrow(735, 84, 725, 140, "own QKV", lx=752, ly=116, fs=9)
    f.box(605, 140, 170, 36, "one attention over [x; c]", sub="split back; own W_{O}, own MLP", fs=10.5, subfs=8.5)
    # arrows out
    for x in (110, 345, 490, 690):
        f.arrow(x, 176, x, 200)
        f.text(x, 212, "updated x", 9.5, fill="muted")
    f.text(690, 226, "(and updated c)", 9.5, fill="muted")
    return f


# ---------------------------------------------------------------- 18 gated / perceiver
@reg("gatedxattn", "Two tools for attaching a frozen model to new inputs. Left: a gated cross-attention layer (Flamingo) inserted into a frozen stack, multiplied by tanh of a gate initialized at zero, so the pretrained model is unchanged at the start of training. Right: a Perceiver resampler compresses a variable-length context into a fixed number of tokens by letting learned queries cross-attend to it.")
def gatedxattn():
    f = Fig("gatedxattn", 780, 300, "Gated cross-attention inserted into a frozen stack, and a Perceiver resampler that maps a variable-length context to a fixed number of tokens")
    # left
    f.text(190, 20, "Gated cross-attention (Flamingo-style)", 12.5, bold=True)
    xs = 60
    f.box(xs - 40, 40, 80, 28, "x (frozen)", fs=10, fill="greysoft", stroke="muted")
    f.line(xs, 68, xs, 100)
    f.box(110, 86, 44, 30, "LN", fs=10.5)
    f.box(164, 86, 120, 30, "cross-attention", fs=10.5, fill="bluesoft")
    f.box(294, 86, 70, 30, "× tanh(α)", sub="α init 0", fs=10, subfs=8.5, fill="goldsoft", stroke="gold")
    f.line(xs, 101, 110, 101, head=True)
    f.arrow(154, 101, 164, 101)
    f.arrow(284, 101, 294, 101)
    f.poly([(364, 101), (380, 101), (380, 140), (xs + 10, 140)], head=True)
    f.line(xs, 100, xs, 130)
    f.circle(xs, 140, 10, "+")
    f.box(170, 150, 110, 26, "context K, V", fs=10, fill="bluesoft", dash=True)
    f.arrow(224, 150, 224, 118, color="blue", dash=True)
    f.arrow(xs, 150, xs, 184)
    f.box(xs - 40, 184, 80, 28, "frozen block", fs=10, fill="greysoft", stroke="muted")
    f.arrow(xs, 212, xs, 236)
    f.text(xs, 248, "x'", 11, bold=True)
    f.lines(250, 220, ["only the new layers train; at α = 0 the output", "equals the frozen model's, so training starts", "from the pretrained behaviour (same idea as", "adaLN-Zero, ControlNet's zero convs, LoRA's B = 0)"], 10, fill="muted", lh=13)
    # right
    f.text(590, 20, "Perceiver resampler", 12.5, bold=True)
    for k in range(7):
        f.rect(430 + k * 24, 50, 20, 24, fill="bluesoft", stroke="blue", rx=3)
    f.text(430 + 7 * 24 + 10, 62, "…", 12)
    f.text(520, 86, "context, any length M (encoder states, frames)", 9.5, fill="muted")
    for k in range(4):
        f.rect(640 + k * 24, 50, 20, 24, fill="goldsoft", stroke="gold", rx=3)
    f.text(686, 86, "R learned queries", 9.5, fill="muted")
    f.box(470, 120, 240, 40, "cross-attention + MLP (× few)", sub="queries attend to [context ; queries]", fs=10.5, subfs=9)
    f.arrow(514, 92, 540, 118, "K, V", lx=510, ly=110, fs=9.5)
    f.arrow(686, 92, 650, 118, "Q", lx=680, ly=110, fs=9.5)
    f.arrow(590, 160, 590, 186)
    for k in range(4):
        f.rect(542 + k * 24, 186, 20, 24, fill="goldsoft", stroke="gold", rx=3)
    f.text(590, 224, "R output tokens, fixed shape whatever M is", 10, fill="muted")
    f.lines(590, 258, ["cost R·M, not M²; used in Flamingo, BLIP-2 (Q-Former),", "TET's bridge; the resampler then feeds a backbone"], 10, fill="muted", lh=13)
    return f


# ---------------------------------------------------------------- 19 swin
@reg("swin", "Swin's two moves. Left: attention is computed inside non-overlapping M×M windows, so cost is linear in the number of tokens; the next block shifts the window grid by M/2 so tokens near a border can see across it. Right: patch merging halves the grid and doubles the channels between stages, giving a feature pyramid like a CNN's.")
def swin():
    f = Fig("swin", 780, 270, "Swin Transformer: windowed attention, shifted windows, and patch merging into a four-stage pyramid")
    def grid(x, y, n, cell, win, shift, title):
        f.text(x + n * cell / 2, y - 14, title, 11.5, bold=True)
        for i in range(n + 1):
            f.line(x, y + i * cell, x + n * cell, y + i * cell, "line", 0.8)
            f.line(x + i * cell, y, x + i * cell, y + n * cell, "line", 0.8)
        off = shift * cell
        k = 0
        for wy in range(-1, n // win + 1):
            for wx in range(-1, n // win + 1):
                x0, y0 = x + wx * win * cell + off, y + wy * win * cell + off
                x1, y1 = x0 + win * cell, y0 + win * cell
                cx0, cy0 = max(x0, x), max(y0, y)
                cx1, cy1 = min(x1, x + n * cell), min(y1, y + n * cell)
                if cx1 - cx0 > 0 and cy1 - cy0 > 0:
                    f.rect(cx0, cy0, cx1 - cx0, cy1 - cy0, fill="none", stroke="blue", rx=0, sw=1.8)
        f.rect(x + 3 * cell, y + 3 * cell, cell, cell, fill="goldsoft", stroke="gold", rx=0)
    grid(30, 40, 8, 18, 4, 0, "block ℓ: windows M = 4")
    grid(220, 40, 8, 18, 4, 2, "block ℓ+1: shifted by M/2")
    f.text(165, 210, "the marked token's neighbourhood changes:", 10, fill="muted")
    f.text(165, 224, "attention within a window, then across the old borders", 10, fill="muted")
    f.text(165, 246, "cost 4hwC² + 2M²hwC instead of 4hwC² + 2(hw)²C", 10.5, fill="ink")
    # pyramid
    f.text(590, 26, "patch merging → pyramid (Swin-T)", 11.5, bold=True)
    stages = [("56² × 96", 112), ("28² × 192", 92), ("14² × 384", 76), ("7² × 768", 64)]
    y = 44
    for i, (lab, w) in enumerate(stages):
        f.box(590 - w / 2, y, w, 30, lab, fs=10)
        f.text(590 + 80, y + 15, f"stage {i + 1}, stride {4 * 2 ** i}", 9.5, "start", "muted")
        if i < 3:
            f.arrow(590, y + 30, 590, y + 46, "merge 2×2 → 4C → 2C", lx=520, ly=y + 40, fs=9)
        y += 46
    f.text(590, 236, "like ResNet's C2–C5: FPN heads for detection plug in", 10, fill="muted")
    return f


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--sheet" else None
    parts = []
    for i, (name, (fn, cap)) in enumerate(FIGS.items(), 1):
        parts.append(fn().svg(cap, i))
    if out:
        css = pathlib.Path(__file__).resolve().parents[3] / "style.css"
        ncss = pathlib.Path(__file__).resolve().parents[3] / "notes" / "notes.css"
        out.write_text(f'<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="{css}"><link rel="stylesheet" href="{ncss}">'
                       f'<body class="note-page"><main class="container"><article class="note" style="max-width:780px;margin:0 auto">' + "\n".join(parts) + "</article></main>")
        print("wrote", out, len(parts), "figures")
    else:
        print("\n".join(parts))
