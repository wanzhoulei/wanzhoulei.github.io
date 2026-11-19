---
title: "Transformers, Step by Step"
subtitle: "The architecture drawn before it is derived: the whole model, the block, attention and its three uses, positions, the families, counting, inference, FlashAttention, and the vision and generation backbones (ViT, DiT, U-Net with cross-attention, MMDiT) with the design space of conditioning."
order: 0.5
updated: 2026-11-18
---

<!-- GENERATED from tools/notes/figs/transformers.template.md by tools/notes/figs/assemble_transformers.py; edit the template and re-run it, then tools/build_notes.py. -->

A transformer maps $N$ token vectors to $N$ token vectors of the same width by alternating two operations: attention, which mixes information across positions according to content, and an MLP applied to each position on its own, with residual connections and normalization holding the stack together. That one block, stacked $L$ times, is GPT, BERT, ViT, DiT, the text encoders of text-to-image models and the MMDiT backbones of SD3 and FLUX. This page is organized around pictures of the structure: the whole model first, then each component, then the vision and generation variants, with the derivations an interviewer actually asks for (why $\sqrt{d_k}$, parameter and FLOP counts, KV-cache size, adaLN-Zero) and nothing else. Optimizers and schedules are on [deep learning training](deep-learning-training.html); self-supervised vision encoders on [vision backbones](vision-backbones.html); pretraining data, decoding, LoRA and alignment on [large language models](llms.html).

## Notation

Rows are tokens: $X \in \mathbb{R}^{N \times D}$, and a linear layer is $XW$ with $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ (PyTorch stores the transpose).

| Symbol | Meaning |
|---|---|
| $B, N, D$ | batch size, number of tokens, model width ($d_{\text{model}}$) |
| $M$ | number of keys when it differs from $N$ (cross-attention, KV cache) |
| $h$, $d_h = D/h$ | number of heads, width of one head |
| $L$ | number of blocks |
| $V$ | vocabulary size |
| $d_{\text{ff}}$ | MLP hidden width, usually $4D$ |
| $W_Q, W_K, W_V, W_O$ | query, key, value and output projections, each $D \times D$ |
| $S = QK^\top/\sqrt{d_k}$, $A = \operatorname{softmax}(S)$ | scores and attention weights, softmax over the key axis |
| $E \in \mathbb{R}^{V \times D}$, $W_U$ | token embedding table, output (unembedding) matrix |
| $P$, $p$ | patch size on pixels (ViT), on latents (DiT) |
| $\Psi$, $T$ | parameter count, training tokens |
| MAC | one multiply-add, 2 FLOPs |

## The whole model

```{=html}
<figure class="fig" id="fig-overview"><svg viewBox="0 0 780 620" role="img" aria-label="Whole decoder-only transformer: embedding, L pre-LN blocks of attention and MLP on a residual stream, final norm, output head" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:780px;margin:0 auto"><defs><marker id="ah-overview-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<rect x="80" y="14" width="140" height="34" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="150.0" y="31.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">Token IDs</text>
<text x="660" y="31" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11.5px">(B, N) integers</text>
<line x1="150" y1="48" x2="150" y2="70" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-overview-ink)"/>
<rect x="20" y="70" width="260" height="44" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="150.0" y="85.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">Embedding lookup E[t] + position</text>
<text x="150.0" y="100.4" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">(learned or sinusoidal; RoPE acts inside attention)</text>
<text x="660" y="92" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11.5px">(B, N, D)</text>
<line x1="150" y1="114" x2="150" y2="150" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-overview-ink)"/>
<rect x="110" y="140" width="420" height="236" rx="10" style="fill:none;stroke:var(--muted);stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="120" y="152" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">one block (pre-LN)</text>
<rect x="210" y="168" width="60" height="34" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="240.0" y="185.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">LN</text>
<rect x="290" y="168" width="170" height="34" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="375.0" y="178.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">Masked multi-head</text>
<text x="375.0" y="193.4" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">self-attention</text>
<line x1="270" y1="185.0" x2="290" y2="185.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-overview-ink)"/>
<line x1="150" y1="154" x2="150" y2="224" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<line x1="150" y1="185.0" x2="210" y2="185.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-overview-ink)"/>
<polyline points="460,185.0 474,185.0 474,234 160,234" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-overview-ink)"/>
<circle cx="150" cy="234" r="10" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="150" y="234" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px">+</text>
<rect x="210" y="260" width="60" height="34" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="240.0" y="277.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">LN</text>
<rect x="290" y="260" width="170" height="34" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="375.0" y="270.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">MLP</text>
<text x="375.0" y="285.4" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">D → 4D → D</text>
<line x1="270" y1="277.0" x2="290" y2="277.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-overview-ink)"/>
<line x1="150" y1="244" x2="150" y2="316" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<line x1="150" y1="277.0" x2="210" y2="277.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-overview-ink)"/>
<polyline points="460,277.0 474,277.0 474,326 160,326" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-overview-ink)"/>
<circle cx="150" cy="326" r="10" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="150" y="326" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px">+</text>
<path d="M540,140 H546 V376 H540" style="fill:none;stroke:var(--muted);stroke-width:1"/>
<text x="554" y="258.0" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:13px">× L</text>
<text x="660" y="160" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11.5px">(B, N, D) in</text>
<text x="660" y="372" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11.5px">(B, N, D) out</text>
<line x1="150" y1="336" x2="150" y2="404" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-overview-ink)"/>
<rect x="100" y="404" width="100" height="32" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="150.0" y="420.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">Final LN</text>
<line x1="150" y1="436" x2="150" y2="462" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-overview-ink)"/>
<rect x="40" y="462" width="220" height="44" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="150.0" y="477.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">Output head W<tspan dy="3.6" font-size="8.6">U</tspan><tspan dy="-3.6">​</tspan></text>
<text x="150.0" y="492.4" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">D → V logits (often tied to E)</text>
<text x="660" y="484" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11.5px">(B, N, V)</text>
<line x1="150" y1="506" x2="150" y2="536" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-overview-ink)"/>
<rect x="20" y="536" width="260" height="44" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="150.0" y="551.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">softmax → cross-entropy</text>
<text x="150.0" y="566.4" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">target at position t is token t+1</text>
<text x="660" y="558" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11.5px">N predictions per pass</text>
<text x="660" y="248.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px;font-style:italic">attention is the only place</text>
<text x="660" y="262.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px;font-style:italic">tokens exchange information;</text>
<text x="660" y="276.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px;font-style:italic">LN, MLP and + act per token</text>
</svg><figcaption><strong>Figure 1.</strong> A decoder-only transformer from token IDs to next-token probabilities. The residual stream (vertical line) carries one D-vector per token through L identical blocks; each block adds the output of an attention sub-layer and of an MLP sub-layer to it. Shapes on the right are for a batch of B sequences of N tokens.</figcaption></figure>
```
Read Figure 1 top to bottom. Token IDs of shape $(B, N)$ index an embedding table and become $(B, N, D)$; position is added here (learned or sinusoidal vectors) or applied inside every attention layer (RoPE). Then $L$ identical blocks. Each block owns nothing but two residual branches: a normalized copy of the stream goes through multi-head attention and is added back, then a normalized copy goes through an MLP and is added back. Elhage et al. call the vertical line the residual stream: the final vector at a position is the embedding plus the sum of every sub-layer's output, $x_L = x_0 + \sum_\ell (\text{attn}_\ell + \text{mlp}_\ell)$. Attention is the only operation in the whole network whose output at one position depends on other positions; normalization, the MLP and the additions act per token. After the last block, a final LayerNorm and a $D \times V$ head give logits at every position, and the loss is cross-entropy against the input shifted by one, so one forward pass yields $N$ next-token predictions trained in parallel.

Why this replaced recurrence and convolution: any two positions are connected inside one layer (path length $O(1)$ instead of $O(n)$ for an RNN), all positions of a training sequence are processed at once as large matrix products, and the mixing pattern is decided per input by content rather than fixed by offset. The price is cost quadratic in $N$ and no built-in order or locality, so positions have to be encoded and locality learned.

```{=html}
<figure class="fig" id="fig-families"><svg viewBox="0 0 760 330" role="img" aria-label="Encoder-only, decoder-only and encoder-decoder transformers side by side" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs><marker id="ah-families-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker><marker id="ah-families-link" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--link)"/></marker></defs>
<text x="110" y="18" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Encoder-only</text>
<text x="110" y="36" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">BERT, ViT, DiT, text encoders</text>
<rect x="40" y="262" width="140" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="110.0" y="277.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">input tokens</text>
<rect x="20" y="188" width="180" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="110.0" y="207.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">MLP</text>
<rect x="20" y="144" width="180" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="110.0" y="156.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">Self-attention</text>
<text x="110.0" y="171.05" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">bidirectional: every token sees all</text>
<line x1="110" y1="262" x2="110" y2="232" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-families-ink)"/>
<path d="M206,144 H212 V232 H206" style="fill:none;stroke:var(--muted);stroke-width:1"/>
<text x="220" y="188.0" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">× L</text>
<line x1="110" y1="144" x2="110" y2="60" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-families-ink)"/>
<rect x="30" y="44" width="160" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="110.0" y="59.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">representation per token</text>
<text x="380" y="18" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Decoder-only</text>
<text x="380" y="36" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">GPT, Llama</text>
<rect x="310" y="262" width="140" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="380.0" y="277.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">prefix tokens</text>
<rect x="290" y="188" width="180" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="380.0" y="207.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">MLP</text>
<rect x="290" y="144" width="180" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="380.0" y="156.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">Self-attention</text>
<text x="380.0" y="171.05" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">causal: token t sees ≤ t</text>
<line x1="380" y1="262" x2="380" y2="232" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-families-ink)"/>
<path d="M476,144 H482 V232 H476" style="fill:none;stroke:var(--muted);stroke-width:1"/>
<text x="490" y="188.0" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">× L</text>
<line x1="380" y1="144" x2="380" y2="60" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-families-ink)"/>
<rect x="300" y="44" width="160" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="380.0" y="59.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">next-token logits</text>
<text x="650" y="18" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Encoder–decoder</text>
<text x="650" y="36" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">original transformer, T5</text>
<rect x="580" y="262" width="140" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="650.0" y="277.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">target prefix</text>
<rect x="560" y="188" width="180" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="650.0" y="207.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">MLP</text>
<rect x="560" y="144" width="180" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="650.0" y="156.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">Cross-attention</text>
<text x="650.0" y="171.05" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">Q from decoder, K,V from encoder</text>
<rect x="560" y="100" width="180" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="650.0" y="112.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">Self-attention</text>
<text x="650.0" y="127.05" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">causal</text>
<line x1="650" y1="262" x2="650" y2="232" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-families-ink)"/>
<path d="M746,100 H752 V232 H746" style="fill:none;stroke:var(--muted);stroke-width:1"/>
<text x="760" y="166.0" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">× L</text>
<line x1="650" y1="100" x2="650" y2="60" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-families-ink)"/>
<rect x="570" y="44" width="160" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="650.0" y="59.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">target tokens</text>
<rect x="560" y="296" width="180" height="28" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="650.0" y="310.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">encoder output (M × D)</text>
<polyline points="740,310 752,310 752,163 740,163" style="stroke:var(--link);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-families-link)"/>
</svg><figcaption><strong>Figure 2.</strong> The same block in three arrangements. Encoder-only stacks use bidirectional attention and read out a representation; decoder-only stacks use causal attention and predict the next token; encoder–decoder models add a cross-attention sub-layer in every decoder block whose keys and values come from the encoder's output.</figcaption></figure>
```
The same block is used in three arrangements (Figure 2). Encoder-only stacks (BERT, ViT, DiT, the text encoders of diffusion models) use bidirectional attention and read out a representation per token. Decoder-only stacks (GPT, Llama) use a causal mask and predict the next token. Encoder–decoder models (the original transformer, T5) add a third sub-layer to each decoder block, cross-attention, whose queries are decoder states and whose keys and values are the encoder's final outputs. Decoder-only became the default for LLMs because one objective on raw text gives a training signal at every position, generation is built in, and the KV cache makes serving efficient; encoder-only remains the default wherever the whole input is available at once, which includes every vision and diffusion backbone on this page.

::: {.qa asked="Amazon AS, NVIDIA, Qualcomm, ByteDance, Taotian (5)"}
**Walk through the transformer architecture end to end. Why does it capture long-range dependencies better than an RNN?**

Decoder-only language model: token IDs $(B, N)$ index an embedding table to give $(B, N, D)$; position is added (learned or sinusoidal) or applied inside attention (RoPE). Then $L$ pre-LN blocks: $h = x + \operatorname{MHA}(\mathrm{LN}(x))$ with a causal mask, then $x' = h + \operatorname{MLP}(\mathrm{LN}(h))$ with a $4D$ hidden width. Final LN, a linear head to $V$ logits (often tied to the embedding), cross-entropy against the sequence shifted by one, all positions in parallel. Encoder-only models drop the mask; encoder–decoder models add cross-attention from a causal decoder to a bidirectional encoder. Long range: in an RNN, information from token $i$ reaches $j$ through $j - i$ state updates, so the gradient is a product of $j - i$ Jacobians that vanishes or explodes, and the whole prefix is squeezed into one state vector. In self-attention any two positions are one layer apart, lookup is by content, and past tokens are kept individually. Costs: $O(N^2 D)$ attention, a KV cache that grows with context, and no built-in order.
:::

::: {.qa asked="Meta research intern/RS, Glassdoor (topic only)"}
**What are the main architectural choices in Llama, and why?**

A decoder-only transformer assembled from changes that had each worked elsewhere. Pre-normalization with RMSNorm: a clean identity path for stable deep training, cheaper than LayerNorm. SwiGLU MLPs with width $\tfrac23 \cdot 4D$ (11,008 for $D = 4096$), which beat GELU MLPs at equal parameter count. RoPE in every layer instead of absolute positions, so scores depend on offsets; Llama 3 raises the base to 500,000 for long context. No bias terms. Untied input and output embeddings. Grouped-query attention in the 34B and 70B models of Llama 2 and in all Llama 3 models, to shrink the KV cache. And the data decision: relatively small models trained on far more tokens than compute-optimal (2T for Llama 2), because a small model is cheap to serve.
:::

## From input to tokens

```{=html}
<figure class="fig" id="fig-tokenize"><svg viewBox="0 0 780 350" role="img" aria-label="Text tokens, image patches and latent patches all become an N by D matrix of token vectors" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:780px;margin:0 auto"><defs><marker id="ah-tokenize-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="14" y="48" text-anchor="start" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Text</text>
<rect x="70" y="32" width="120" height="32" rx="6" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="130.0" y="48.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">&quot;the cat sat&quot;</text>
<line x1="190" y1="48" x2="250" y2="48" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-tokenize-ink)"/>
<text x="220.0" y="18" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">tokenizer (BPE)</text>
<rect x="250" y="32" width="120" height="32" rx="6" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="310.0" y="48.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">[464, 3797, 3332]</text>
<line x1="370" y1="48" x2="430" y2="48" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-tokenize-ink)"/>
<text x="400.0" y="18" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">row lookup E[t]</text>
<rect x="430" y="26" width="60" height="44" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="430" y1="40.666666666666664" x2="490" y2="40.666666666666664" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="430" y1="55.33333333333333" x2="490" y2="55.33333333333333" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="440.0" y1="26" x2="440.0" y2="70" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="450.0" y1="26" x2="450.0" y2="70" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="460.0" y1="26" x2="460.0" y2="70" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="470.0" y1="26" x2="470.0" y2="70" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="480.0" y1="26" x2="480.0" y2="70" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="430" y="26" width="60" height="44" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="460.0" y="81" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × D</text>
<text x="560" y="48" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N tokens of D numbers each</text>
<text x="14" y="160" text-anchor="start" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Image</text>
<rect x="70" y="110" width="100" height="100" rx="2" style="fill:#eceff4;stroke:var(--muted);stroke-width:1.2"/>
<line x1="95" y1="110" x2="95" y2="210" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="70" y1="135" x2="170" y2="135" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="120" y1="110" x2="120" y2="210" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="70" y1="160" x2="170" y2="160" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="145" y1="110" x2="145" y2="210" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="70" y1="185" x2="170" y2="185" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<rect x="95" y="135" width="25" height="25" rx="0" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.6"/>
<text x="120" y="222" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">H × W × C, patch size P</text>
<line x1="172" y1="160" x2="250" y2="160" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-tokenize-ink)"/>
<text x="211.0" y="130" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">flatten one patch</text>
<rect x="250" y="144" width="70" height="32" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="285.0" y="160.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">P²C</text>
<line x1="320" y1="160" x2="430" y2="160" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-tokenize-ink)"/>
<text x="375.0" y="130" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">linear map P²C → D</text>
<text x="375" y="192" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">(= conv, kernel P, stride P)</text>
<rect x="430" y="144" width="46" height="32" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="453.0" y="160.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">D</text>
<text x="453" y="206" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">one token</text>
<line x1="476" y1="160" x2="530" y2="160" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-tokenize-ink)"/>
<text x="503.0" y="130" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">all patches</text>
<rect x="530" y="130" width="60" height="60" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="530" y1="145.0" x2="590" y2="145.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="530" y1="160.0" x2="590" y2="160.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="530" y1="175.0" x2="590" y2="175.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="540.0" y1="130" x2="540.0" y2="190" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="550.0" y1="130" x2="550.0" y2="190" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="560.0" y1="130" x2="560.0" y2="190" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="570.0" y1="130" x2="570.0" y2="190" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="580.0" y1="130" x2="580.0" y2="190" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="530" y="130" width="60" height="60" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="560.0" y="201" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × D, N = HW/P²</text>
<text x="690" y="129.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">ViT-B/16 at 224²:</text>
<text x="690" y="143.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N = 14 × 14 = 196</text>
<text x="690" y="157.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">(+1 class token)</text>
<text x="690" y="171.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">P²C = 768 → D = 768</text>
<text x="14" y="290" text-anchor="start" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Latent</text>
<rect x="70" y="260" width="60" height="60" rx="2" style="fill:#eceff4;stroke:var(--muted);stroke-width:1.2"/>
<line x1="85" y1="260" x2="85" y2="320" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="70" y1="275" x2="130" y2="275" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="100" y1="260" x2="100" y2="320" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="70" y1="290" x2="130" y2="290" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="115" y1="260" x2="115" y2="320" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="70" y1="305" x2="130" y2="305" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<text x="100" y="333" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">h × w × c (VAE)</text>
<line x1="132" y1="290" x2="250" y2="290" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-tokenize-ink)"/>
<text x="191.0" y="258" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">patch size p (DiT: p = 2)</text>
<rect x="250" y="274" width="70" height="32" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="285.0" y="290.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">p²c</text>
<line x1="320" y1="290" x2="430" y2="290" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-tokenize-ink)"/>
<text x="375.0" y="258" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">linear p²c → D</text>
<rect x="430" y="274" width="46" height="32" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="453.0" y="290.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">D</text>
<line x1="476" y1="290" x2="530" y2="290" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-tokenize-ink)"/>
<rect x="530" y="262" width="60" height="56" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="530" y1="276.0" x2="590" y2="276.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="530" y1="290.0" x2="590" y2="290.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="530" y1="304.0" x2="590" y2="304.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="540.0" y1="262" x2="540.0" y2="318" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="550.0" y1="262" x2="550.0" y2="318" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="560.0" y1="262" x2="560.0" y2="318" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="570.0" y1="262" x2="570.0" y2="318" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="580.0" y1="262" x2="580.0" y2="318" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="530" y="262" width="60" height="56" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="560.0" y="329" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × D, N = hw/p²</text>
<text x="690" y="263.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">DiT-XL/2 at 256²:</text>
<text x="690" y="277.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">latent 32 × 32 × 4,</text>
<text x="690" y="291.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N = 16 × 16 = 256,</text>
<text x="690" y="305.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">p²c = 16 → D = 1152</text>
</svg><figcaption><strong>Figure 3.</strong> Three inputs, one interface. Text becomes subword IDs and an embedding lookup; an image is cut into P×P patches and each flattened patch is projected by one linear map; a VAE latent is patchified the same way with a smaller patch. Everything downstream sees an (N, D) matrix.</figcaption></figure>
```
Everything before the first block exists to produce an $(N, D)$ matrix (Figure 3). Text: a subword tokenizer (byte-pair encoding in GPT-2 and most LLMs, $V = 50{,}257$ for GPT-2) maps text to integer IDs, and the embedding layer is a row lookup in $E \in \mathbb{R}^{V \times D}$, which is a linear layer on one-hot inputs; only rows of tokens present in the batch receive gradient. Images: ViT cuts the image into non-overlapping $P \times P$ patches, $N = HW/P^2$ of them, flattens each to $P^2 C$ numbers and applies one shared linear map to $D$, which is exactly a convolution with kernel and stride $P$; ViT-B/16 at $224^2$ has $N = 196$ patches of $768$ numbers each, projected to $D = 768$. Latents: DiT does the same on a VAE latent, $32 \times 32 \times 4$ for a $256^2$ image, with latent patch $p = 2$, giving $256$ tokens of $16$ numbers each mapped to $D = 1152$; halving $p$ quadruples $N$, which is why DiT-XL/8, XL/4 and XL/2 cost 7.4, 29.1 and 118.6 Gflops. At the output, DiT maps each token back to $p \times p \times 2C$ numbers and reshapes ("unpatchify").

::: qa
**How many tokens does DiT with $p = 2$ see for a $512 \times 512$ image with an f8 VAE, and how do the costs change compared with $256 \times 256$?**

The latent is $64 \times 64 \times 4$, so $N = (64/2)^2 = 1024$ tokens, four times as many. Per-token work (projections and MLPs) grows $4\times$; the attention score matrices have $N^2$ entries, so that part grows $16\times$.
:::

::: {.qa asked="Amazon AS intern, Glassdoor 2026"}
**What does a tokenizer do, and how does byte-pair encoding work?**

A tokenizer maps text to integer IDs from a fixed vocabulary and back. BPE training: write the corpus as base symbols (bytes or characters), count adjacent pairs weighted by frequency, merge the most frequent pair into a new symbol, record the merge, repeat until the vocabulary reaches its target size; encoding applies the recorded merges in order, so unseen words still get a segmentation. Byte-level BPE can encode any string because all 256 bytes are in the vocabulary. A larger vocabulary gives shorter sequences but a bigger embedding and head. Side effects: inconsistent number chunking, longer sequences for languages underrepresented in the tokenizer's data, and weakness at spelling, since the model sees tokens rather than letters.
:::

## The block

```{=html}
<figure class="fig" id="fig-block"><svg viewBox="0 0 760 300" role="img" aria-label="A pre-LN transformer block with tensor shapes" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs><marker id="ah-block-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="70" y="24" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">X  (N × D)</text>
<line x1="70" y1="36" x2="70" y2="60" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<rect x="130" y="60" width="64" height="40" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="162.0" y="80.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">LN</text>
<rect x="214" y="60" width="230" height="40" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="329.0" y="74.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">multi-head self-attention</text>
<text x="329.0" y="88.4" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">Q, K, V = XW<tspan dy="2.7" font-size="6.5">Q</tspan><tspan dy="-2.7">​</tspan>, XW<tspan dy="2.7" font-size="6.5">K</tspan><tspan dy="-2.7">​</tspan>, XW<tspan dy="2.7" font-size="6.5">V</tspan><tspan dy="-2.7">​</tspan>; out · W<tspan dy="2.7" font-size="6.5">O</tspan><tspan dy="-2.7">​</tspan></text>
<line x1="70" y1="80.0" x2="130" y2="80.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-block-ink)"/>
<line x1="194" y1="80.0" x2="214" y2="80.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-block-ink)"/>
<text x="470" y="69.0" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × D</text>
<polyline points="444,80.0 530,80.0 530,126 80,126" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-block-ink)"/>
<line x1="70" y1="60" x2="70" y2="116" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<circle cx="70" cy="126" r="10" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="70" y="126" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px">+</text>
<rect x="130" y="152" width="64" height="40" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="162.0" y="172.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">LN</text>
<rect x="214" y="152" width="70" height="40" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="249.0" y="166.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">Linear</text>
<text x="249.0" y="179.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">D → 4D</text>
<rect x="296" y="152" width="60" height="40" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="326.0" y="172.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">GELU</text>
<rect x="368" y="152" width="76" height="40" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="406.0" y="166.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">Linear</text>
<text x="406.0" y="179.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">4D → D</text>
<line x1="70" y1="172.0" x2="130" y2="172.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-block-ink)"/>
<line x1="194" y1="172.0" x2="214" y2="172.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-block-ink)"/>
<line x1="284" y1="172.0" x2="296" y2="172.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-block-ink)"/>
<line x1="356" y1="172.0" x2="368" y2="172.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-block-ink)"/>
<text x="320" y="144" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × 4D</text>
<text x="470" y="161.0" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × D</text>
<polyline points="444,172.0 530,172.0 530,218 80,218" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-block-ink)"/>
<line x1="70" y1="136" x2="70" y2="208" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<circle cx="70" cy="218" r="10" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="70" y="218" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px">+</text>
<line x1="70" y1="228" x2="70" y2="248" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-block-ink)"/>
<text x="70" y="262" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">X&#x27;  (N × D)</text>
<text x="640" y="129.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px;font-style:italic">per token:</text>
<text x="640" y="143.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px;font-style:italic">LN, MLP, +</text>
<text x="640" y="157.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px;font-style:italic">across tokens:</text>
<text x="640" y="171.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px;font-style:italic">attention only</text>
<text x="640" y="230" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">params per block ≈ 12D²</text>
<text x="640" y="248" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">(4D² attention + 8D² MLP)</text>
</svg><figcaption><strong>Figure 4.</strong> One pre-LN block with shapes. Each sub-layer reads a normalized copy of the stream, computes something, and adds the result back. The MLP widens each token to 4D and back; attention is the only step whose output at one position depends on other positions.</figcaption></figure>
```
One pre-LN block (Figure 4) is

$$
h = x + \operatorname{MHA}\big(\mathrm{LN}_1(x)\big), \qquad x' = h + \operatorname{MLP}\big(\mathrm{LN}_2(h)\big).
$$

**The MLP.** $\operatorname{MLP}(u) = \phi(uW_1 + b_1)W_2 + b_2$ with $W_1 \in \mathbb{R}^{D \times 4D}$, $W_2 \in \mathbb{R}^{4D \times D}$: $8D^2$ weights, two thirds of the block. The factor 4 is the original paper's convention ($512 \to 2048$); Kaplan et al. found loss nearly flat in the ratio at fixed parameter count. $\phi$ is GELU in BERT, GPT-2 and ViT. Llama, PaLM and most recent LLMs use the gated SwiGLU, $\big(\operatorname{SiLU}(uW_1) \odot uW_3\big)W_2$, three matrices instead of two; to keep $8D^2$ weights its hidden width is $\tfrac23 \cdot 4D$, which for Llama 7B is $10{,}923$, rounded to a multiple of 256 as $11{,}008$. Shazeer found GLU variants lower perplexity at matched parameters and gave no theory for it.

**Normalization.** LayerNorm centers and scales the $D$ features of one token, $\gamma \odot (x - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$, $2D$ parameters; RMSNorm drops the mean and the shift, $\gamma \odot x / \sqrt{\tfrac1D \sum_j x_j^2 + \epsilon}$, $D$ parameters. Both are per token, so they do not depend on the batch, on padding or on other positions; batch statistics would let a causal model see future tokens and would differ between training and inference, which is why BatchNorm is absent from transformers.

```{=html}
<figure class="fig" id="fig-prepost"><svg viewBox="0 0 760 250" role="img" aria-label="Post-LN versus pre-LN placement of layer normalization in a block" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs><marker id="ah-prepost-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="170" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Post-LN (2017)</text>
<text x="50" y="42" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">x</text>
<line x1="50" y1="50" x2="50" y2="74" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<rect x="90" y="70" width="150" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="165.0" y="85.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">F  (attention / MLP)</text>
<line x1="50" y1="85" x2="90" y2="85" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-prepost-ink)"/>
<polyline points="240,85 258,85 258,140 60,140" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-prepost-ink)"/>
<line x1="50" y1="74" x2="50" y2="130" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<circle cx="50" cy="140" r="10" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="50" y="140" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px">+</text>
<line x1="50" y1="150" x2="50" y2="166" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-prepost-ink)"/>
<rect x="25" y="166" width="50" height="28" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="50.0" y="180.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">LN</text>
<line x1="50" y1="194" x2="50" y2="200" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-prepost-ink)"/>
<text x="50" y="212" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px;font-weight:700">x&#x27; = LN(x + F(x))</text>
<text x="200" y="180" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">every path crosses a LN Jacobian</text>
<text x="550" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Pre-LN (GPT-2, Llama, ViT, DiT)</text>
<text x="430" y="42" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">x</text>
<line x1="430" y1="50" x2="430" y2="74" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<rect x="470" y="70" width="50" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="495.0" y="85.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">LN</text>
<rect x="530" y="70" width="150" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="605.0" y="85.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">F  (attention / MLP)</text>
<line x1="430" y1="85" x2="470" y2="85" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-prepost-ink)"/>
<line x1="520" y1="85" x2="530" y2="85" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-prepost-ink)"/>
<polyline points="680,85 698,85 698,140 440,140" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-prepost-ink)"/>
<line x1="430" y1="74" x2="430" y2="130" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<circle cx="430" cy="140" r="10" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="430" y="140" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px">+</text>
<line x1="430" y1="150" x2="430" y2="200" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-prepost-ink)"/>
<text x="430" y="212" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px;font-weight:700">x&#x27; = x + F(LN(x))</text>
<text x="580" y="175" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">identity path: ∂x&#x27;/∂x = I + ∂F/∂x</text>
</svg><figcaption><strong>Figure 5.</strong> Where the normalization sits. Post-LN (original transformer) normalizes after each residual addition, so every gradient path passes through a LayerNorm at every layer. Pre-LN (GPT-2 onward) normalizes the sub-layer input and leaves an identity path from the loss straight down the stream; it needs a final LN before the head.</figcaption></figure>
```
**Pre-LN versus post-LN.** The 2017 transformer normalized after each residual addition, $x_{\ell+1} = \mathrm{LN}(x_\ell + F_\ell(x_\ell))$; GPT-2 and everything since normalizes the sub-layer input, $x_{\ell+1} = x_\ell + F_\ell(\mathrm{LN}(x_\ell))$, with one final LN before the head (Figure 5). The difference is the gradient path. In pre-LN, $x_L = x_\ell + \sum_{k \ge \ell} F_k(\mathrm{LN}(x_k))$, so $\partial x_L/\partial x_\ell = I + \sum_k \partial F_k/\partial x_\ell$ contains the identity: the gradient reaches every layer without passing through any weight matrix. In post-LN every path crosses a LayerNorm Jacobian at every layer, which rescales the gradient by $\sqrt{D}/\|\text{input}\|$; Xiong et al. show the top layers' gradients are then large at initialization, which is why post-LN needs learning-rate warmup and pre-LN trains without it. The cost of pre-LN is that the stream's norm grows with depth (linearly in expectation), so late layers add relatively less; current LLMs and DiTs accept that and use pre-norm, usually RMSNorm.

**Dropout, final norm, head.** The 2017 model applied dropout 0.1 to each sub-layer output and to the embeddings; large-scale pretraining, which sees each token about once, mostly uses none. The head maps the final $D$-vector to $V$ logits; tying $W_U = E$ saves $VD$ parameters, 38.6M of GPT-2 small's 124.4M (31%). Llama keeps them separate.

::: {.qa asked="Google intern, Glassdoor (role unclear)"}
**Compare BatchNorm, LayerNorm, InstanceNorm and GroupNorm.**

All compute $(x - \mu)/\sqrt{\sigma^2 + \epsilon}$ followed by a learned per-channel scale and shift; they differ in which axes are pooled. For $(B, C, H, W)$: BatchNorm pools over $(B, H, W)$ per channel, so it depends on the batch, keeps running averages for inference, and degrades with small batches. LayerNorm pools over $(C, H, W)$ per sample; in transformers over the $D$ features of each token, independent of batch and of other positions, identical in training and inference. InstanceNorm pools over $(H, W)$ per sample and channel, removing per-image contrast and style. GroupNorm pools over $(H, W)$ and a group of channels, between the two, and is the usual choice in diffusion U-Nets with small per-GPU batches. Transformers use LN or RMSNorm because sequences vary in length, batches can be small, and a causal model must not share statistics across positions.
:::

::: qa
**Why does a post-LN transformer need warmup, and why does pre-LN need it less?**

In pre-LN, $\partial x_L/\partial x_\ell$ contains the identity, so gradients reach every layer directly and the final LN scales them down by about $1/\sqrt{L}$ because the stream grows. In post-LN every path passes through a LayerNorm Jacobian per layer, and at initialization the top layers' gradients are large; a full learning rate on them destabilizes training, so warmup starts small. Pre-LN trains without warmup in Xiong et al.'s experiments, though warmup stays common for Adam's noisy early moment estimates.
:::

## Attention

```{=html}
<figure class="fig" id="fig-attention"><svg viewBox="0 0 760 330" role="img" aria-label="Attention computed as matrix products: projections, scores, softmax, weighted sum of values" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs><marker id="ah-attention-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<rect x="20" y="110" width="60" height="90" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="20" y1="128.0" x2="80" y2="128.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="20" y1="146.0" x2="80" y2="146.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="20" y1="164.0" x2="80" y2="164.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="20" y1="182.0" x2="80" y2="182.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="35.0" y1="110" x2="35.0" y2="200" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="50.0" y1="110" x2="50.0" y2="200" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="65.0" y1="110" x2="65.0" y2="200" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="20" y="110" width="60" height="90" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="50.0" y="155.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">X</text>
<text x="50.0" y="211" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × D</text>
<line x1="82" y1="155" x2="120" y2="60" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-attention-ink)"/>
<text x="92" y="96" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">W<tspan dy="3.3" font-size="7.9">Q</tspan><tspan dy="-3.3">​</tspan></text>
<line x1="82" y1="155" x2="120" y2="150" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-attention-ink)"/>
<text x="100" y="170" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">W<tspan dy="3.3" font-size="7.9">K</tspan><tspan dy="-3.3">​</tspan></text>
<line x1="82" y1="155" x2="120" y2="240" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-attention-ink)"/>
<text x="92" y="214" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">W<tspan dy="3.3" font-size="7.9">V</tspan><tspan dy="-3.3">​</tspan></text>
<rect x="120" y="20" width="50" height="60" rx="2" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<line x1="120" y1="32.0" x2="170" y2="32.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="44.0" x2="170" y2="44.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="56.0" x2="170" y2="56.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="68.0" x2="170" y2="68.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="136.66666666666666" y1="20" x2="136.66666666666666" y2="80" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="153.33333333333334" y1="20" x2="153.33333333333334" y2="80" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="120" y="20" width="50" height="60" rx="2" style="fill:none;stroke:var(--gold);stroke-width:1.2"/>
<text x="145.0" y="50.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">Q</text>
<text x="145.0" y="91" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × d</text>
<rect x="120" y="110" width="50" height="60" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="120" y1="122.0" x2="170" y2="122.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="134.0" x2="170" y2="134.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="146.0" x2="170" y2="146.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="158.0" x2="170" y2="158.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="136.66666666666666" y1="110" x2="136.66666666666666" y2="170" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="153.33333333333334" y1="110" x2="153.33333333333334" y2="170" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="120" y="110" width="50" height="60" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="145.0" y="140.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">K</text>
<text x="145.0" y="181" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">M × d</text>
<rect x="120" y="200" width="50" height="60" rx="2" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<line x1="120" y1="212.0" x2="170" y2="212.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="224.0" x2="170" y2="224.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="236.0" x2="170" y2="236.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="248.0" x2="170" y2="248.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="136.66666666666666" y1="200" x2="136.66666666666666" y2="260" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="153.33333333333334" y1="200" x2="153.33333333333334" y2="260" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="120" y="200" width="50" height="60" rx="2" style="fill:none;stroke:#2e7d32;stroke-width:1.2"/>
<text x="145.0" y="230.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">V</text>
<text x="145.0" y="271" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">M × d<tspan dy="3.1" font-size="7.6">v</tspan><tspan dy="-3.1">​</tspan></text>
<line x1="172" y1="50" x2="236" y2="95" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-attention-ink)"/>
<line x1="172" y1="140" x2="236" y2="110" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-attention-ink)"/>
<rect x="236" y="70" width="70" height="70" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="236" y1="84.0" x2="306" y2="84.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="236" y1="98.0" x2="306" y2="98.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="236" y1="112.0" x2="306" y2="112.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="236" y1="126.0" x2="306" y2="126.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="250.0" y1="70" x2="250.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="264.0" y1="70" x2="264.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="278.0" y1="70" x2="278.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="292.0" y1="70" x2="292.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="236" y="70" width="70" height="70" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="271.0" y="105.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">S</text>
<text x="271.0" y="151" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × M</text>
<text x="271" y="56" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">QKᵀ / √d</text>
<line x1="308" y1="105" x2="352" y2="105" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-attention-ink)"/>
<text x="330" y="60" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">softmax</text>
<text x="330" y="74" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">per row</text>
<rect x="352" y="70" width="70" height="70" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="352.0" y="84.0" width="14.0" height="14.0" style="fill:#fff3d6;stroke:none"/>
<rect x="366.0" y="84.0" width="14.0" height="14.0" style="fill:#fff3d6;stroke:none"/>
<rect x="380.0" y="84.0" width="14.0" height="14.0" style="fill:#fff3d6;stroke:none"/>
<rect x="394.0" y="84.0" width="14.0" height="14.0" style="fill:#fff3d6;stroke:none"/>
<rect x="408.0" y="84.0" width="14.0" height="14.0" style="fill:#fff3d6;stroke:none"/>
<line x1="352" y1="84.0" x2="422" y2="84.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="352" y1="98.0" x2="422" y2="98.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="352" y1="112.0" x2="422" y2="112.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="352" y1="126.0" x2="422" y2="126.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="366.0" y1="70" x2="366.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="380.0" y1="70" x2="380.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="394.0" y1="70" x2="394.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="408.0" y1="70" x2="408.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="352" y="70" width="70" height="70" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="387.0" y="105.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">A</text>
<text x="387.0" y="151" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × M, rows sum to 1</text>
<line x1="424" y1="105" x2="470" y2="105" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-attention-ink)"/>
<text x="447" y="60" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">A · V</text>
<line x1="172" y1="230" x2="470" y2="230" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-attention-ink)"/>
<polyline points="470,230 490,230 490,140" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-attention-ink)"/>
<rect x="470" y="70" width="50" height="70" rx="2" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<line x1="470" y1="84.0" x2="520" y2="84.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="470" y1="98.0" x2="520" y2="98.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="470" y1="112.0" x2="520" y2="112.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="470" y1="126.0" x2="520" y2="126.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="486.6666666666667" y1="70" x2="486.6666666666667" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="503.3333333333333" y1="70" x2="503.3333333333333" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="470" y="70" width="50" height="70" rx="2" style="fill:none;stroke:#2e7d32;stroke-width:1.2"/>
<text x="495.0" y="105.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">out</text>
<text x="495.0" y="151" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × d<tspan dy="3.1" font-size="7.6">v</tspan><tspan dy="-3.1">​</tspan></text>
<line x1="522" y1="105" x2="560" y2="105" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-attention-ink)"/>
<text x="541" y="60" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">W<tspan dy="3.3" font-size="7.9">O</tspan><tspan dy="-3.3">​</tspan></text>
<rect x="560" y="70" width="50" height="70" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="560" y1="84.0" x2="610" y2="84.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="560" y1="98.0" x2="610" y2="98.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="560" y1="112.0" x2="610" y2="112.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="560" y1="126.0" x2="610" y2="126.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="572.5" y1="70" x2="572.5" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="585.0" y1="70" x2="585.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="597.5" y1="70" x2="597.5" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="560" y="70" width="50" height="70" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="585.0" y="151" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × D</text>
<text x="655" y="166.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">self-attention: M = N, K, V from X</text>
<text x="655" y="180.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">cross-attention: K, V from a context C</text>
<text x="655" y="194.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px"></text>
<text x="655" y="208.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">row i of A: weights of query i over keys</text>
<text x="655" y="222.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">row i of out = Σ<tspan dy="3.0" font-size="7.2">j</tspan><tspan dy="-3.0">​</tspan> a<tspan dy="3.0" font-size="7.2">ij</tspan><tspan dy="-3.0">​</tspan> v<tspan dy="3.0" font-size="7.2">j</tspan><tspan dy="-3.0">​</tspan></text>
<text x="655" y="236.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px"></text>
<text x="655" y="250.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">cost N·M·d (S) + N·M·d<tspan dy="3.0" font-size="7.2">v</tspan><tspan dy="-3.0">​</tspan> (A·V);</text>
<text x="655" y="264.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">N·M scores per head in memory</text>
</svg><figcaption><strong>Figure 6.</strong> Scaled dot-product attention as matrices. Three projections of the same input give queries, keys and values; the score matrix compares every query with every key; a row-wise softmax turns each row into weights that sum to one; the output of each query is the weighted average of the value rows.</figcaption></figure>
```
Figure 6 is the whole operation:

$$
\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V, \qquad Q = XW_Q \in \mathbb{R}^{N \times d_k},\ K = CW_K \in \mathbb{R}^{M \times d_k},\ V = CW_V \in \mathbb{R}^{M \times d_v},
$$

with $C = X$ for self-attention and the softmax over each row, that is over the key axis. The score between positions $i$ and $j$ is the bilinear form $x_i^\top W_Q W_K^\top x_j$, of rank at most $d_k$ and not symmetric; the output row $i$ is a convex combination of the value rows, so the output scale does not grow with $M$. Cost: $NMd_k$ MACs for $QK^\top$, $NMd_v$ for $AV$, and $NM$ numbers of memory per head if the scores are materialized; in self-attention both are quadratic in $N$.

**Why $\sqrt{d_k}$.** If the components of $q$ and $k$ are independent with mean 0 and variance 1, which is roughly the case at initialization after normalization, then $\mathbb{E}[q^\top k] = 0$ and $\operatorname{Var}[q^\top k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$, because the $d_k$ products are uncorrelated with variance 1 each. Unscaled logits would have standard deviation 8 for $d_k = 64$, which puts the softmax near one-hot; its Jacobian $\operatorname{diag}(p) - pp^\top$ is then near zero and the gradients to $Q$, $K$ and everything behind them vanish, while the value path still trains. Dividing by $\sqrt{d_k}$ gives unit-variance logits for any head width. It is an initialization argument: logits can still grow during training, which QK-normalization addresses (see "Training stability"). Softmax is shift-invariant, $\operatorname{softmax}(z + c\mathbf 1) = \operatorname{softmax}(z)$, which is what lets implementations subtract the row maximum and what FlashAttention's online softmax exploits. Without positions or masks the operation is permutation equivariant (permuting the rows of $X$ permutes the outputs the same way), which is why order has to be injected.

```{=html}
<figure class="fig" id="fig-uses"><svg viewBox="0 0 760 290" role="img" aria-label="Self-attention, causal self-attention and cross-attention: sources of Q, K, V and the allowed score pattern" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs><marker id="ah-uses-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="125" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Self (encoder)</text>
<rect x="10" y="40" width="100" height="30" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="60.0" y="55.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">X  (N × D)</text>
<rect x="140" y="40" width="100" height="30" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="190.0" y="55.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">X  (N × D)</text>
<text x="60" y="84" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">Q</text>
<text x="190" y="84" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">K, V</text>
<line x1="60" y1="70" x2="60" y2="158.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-uses-ink)"/>
<line x1="190" y1="70" x2="189" y2="98" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-uses-ink)"/>
<rect x="75" y="100" width="120" height="120" rx="2" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<line x1="75" y1="120.0" x2="195" y2="120.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="75" y1="140.0" x2="195" y2="140.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="75" y1="160.0" x2="195" y2="160.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="75" y1="180.0" x2="195" y2="180.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="75" y1="200.0" x2="195" y2="200.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="95.0" y1="100" x2="95.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="115.0" y1="100" x2="115.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="135.0" y1="100" x2="135.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="155.0" y1="100" x2="155.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="175.0" y1="100" x2="175.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="75" y="100" width="120" height="120" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="135.0" y="232" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">scores 6 × 6</text>
<text x="67" y="160.0" text-anchor="end" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px" transform="rotate(-90 67 160.0)">queries</text>
<text x="77" y="94" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">keys →</text>
<text x="125" y="245.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px;font-style:italic">BERT, ViT, DiT: every token</text>
<text x="125" y="258.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px;font-style:italic">reads every token</text>
<text x="380" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Causal (decoder)</text>
<rect x="265" y="40" width="100" height="30" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="315.0" y="55.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">X  (N × D)</text>
<rect x="395" y="40" width="100" height="30" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="445.0" y="55.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">X  (N × D)</text>
<text x="315" y="84" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">Q</text>
<text x="445" y="84" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">K, V</text>
<line x1="315" y1="70" x2="315" y2="158.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-uses-ink)"/>
<line x1="445" y1="70" x2="444" y2="98" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-uses-ink)"/>
<rect x="330" y="100" width="120" height="120" rx="2" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<rect x="350.0" y="100.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="370.0" y="100.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="390.0" y="100.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="410.0" y="100.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="430.0" y="100.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="370.0" y="120.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="390.0" y="120.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="410.0" y="120.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="430.0" y="120.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="390.0" y="140.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="410.0" y="140.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="430.0" y="140.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="410.0" y="160.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="430.0" y="160.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<rect x="430.0" y="180.0" width="20.0" height="20.0" style="fill:#eceff4;stroke:none"/>
<line x1="330" y1="120.0" x2="450" y2="120.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="330" y1="140.0" x2="450" y2="140.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="330" y1="160.0" x2="450" y2="160.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="330" y1="180.0" x2="450" y2="180.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="330" y1="200.0" x2="450" y2="200.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="350.0" y1="100" x2="350.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="370.0" y1="100" x2="370.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="390.0" y1="100" x2="390.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="410.0" y1="100" x2="410.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="430.0" y1="100" x2="430.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="330" y="100" width="120" height="120" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="390.0" y="232" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">scores 6 × 6</text>
<text x="322" y="160.0" text-anchor="end" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px" transform="rotate(-90 322 160.0)">queries</text>
<text x="332" y="94" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">keys →</text>
<text x="380" y="245.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px;font-style:italic">GPT: token i reads only j ≤ i;</text>
<text x="380" y="258.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px;font-style:italic">one pass trains N predictions</text>
<text x="635" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Cross</text>
<rect x="520" y="40" width="100" height="30" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="570.0" y="55.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">X  (N × D)</text>
<rect x="650" y="40" width="100" height="30" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="700.0" y="55.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">C  (M × D<tspan dy="3.1" font-size="7.6">c</tspan><tspan dy="-3.1">​</tspan>)</text>
<text x="570" y="84" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">Q</text>
<text x="700" y="84" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:11px">K, V</text>
<line x1="570" y1="70" x2="570" y2="158.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-uses-ink)"/>
<line x1="700" y1="70" x2="659" y2="98" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-uses-ink)"/>
<rect x="585" y="100" width="80" height="120" rx="2" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<line x1="585" y1="120.0" x2="665" y2="120.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="585" y1="140.0" x2="665" y2="140.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="585" y1="160.0" x2="665" y2="160.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="585" y1="180.0" x2="665" y2="180.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="585" y1="200.0" x2="665" y2="200.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="605.0" y1="100" x2="605.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="625.0" y1="100" x2="625.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="645.0" y1="100" x2="645.0" y2="220" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="585" y="100" width="80" height="120" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="625.0" y="232" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">scores 6 × 4</text>
<text x="577" y="160.0" text-anchor="end" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px" transform="rotate(-90 577 160.0)">queries</text>
<text x="587" y="94" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">keys →</text>
<text x="635" y="245.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px;font-style:italic">SD U-Net: image queries,</text>
<text x="635" y="258.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px;font-style:italic">77 text keys and values</text>
</svg><figcaption><strong>Figure 7.</strong> The three uses of the same operation differ only in where Q, K, V come from and which scores are allowed (white = allowed, grey = set to −∞ before the softmax). Self-attention: all from X, all allowed. Causal: all from X, query i sees keys j ≤ i. Cross-attention: Q from the sequence being updated, K and V from another sequence (text, encoder output), no mask needed.</figcaption></figure>
```
**The three uses** (Figure 7) differ only in where $Q$, $K$, $V$ come from and which scores are allowed.

- *Self-attention, bidirectional.* All from $X$; every token reads every token. BERT, ViT, DiT.
- *Causal self-attention.* A language model factorizes $p(x_1, \dots, x_N) = \prod_t p(x_t \mid x_{<t})$; to train all $N$ conditionals in one pass, position $t$ must not see positions after $t$. Add $M_{ij} = -\infty$ for $j > i$ to the scores before the softmax, so those keys get exactly zero weight and the rest renormalize. By induction over layers, the vector at position $t$ in every layer depends only on $x_{\le t}$, so each position's output is a legitimate prediction of $x_{t+1}$, and the same fact is what makes the KV cache exact. Packed documents use a block-diagonal mask so tokens do not attend across boundaries (Llama 3).
- *Cross-attention.* Queries from the sequence being updated, keys and values from another sequence $C \in \mathbb{R}^{M \times D_c}$, with $W_K, W_V \in \mathbb{R}^{D_c \times D}$; the output has the query sequence's length and the cost is $O(NMD)$, cheap when the context is short. The decoder of a translation model reads the encoder this way; Stable Diffusion's U-Net reads 77 CLIP text tokens this way at every resolution (at the $64^2$ level of SD v1: 4096 queries of width 320, 77 keys projected from width 768, so $4096 \times 77$ weights per head rather than $4096^2$); PixArt-$\alpha$ and Wan add a cross-attention sub-layer to each DiT block for text.

**Padding.** Shorter sequences in a batch are padded; a padding mask of shape $(B, 1, 1, M)$ sets the score of every pad key to $-\infty$. A row whose keys are all masked is the softmax of all $-\infty$, which is NaN; left padding with a causal mask produces exactly that at the first pad query, so pad queries must be allowed one key and dropped from the loss. Masked logits must be $-\infty$ or the dtype minimum, not $-10^9$, which overflows float16.

```{=html}
<figure class="fig" id="fig-heads"><svg viewBox="0 0 760 300" role="img" aria-label="Multi-head attention: split channels into heads, attend separately, concatenate, project" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs><marker id="ah-heads-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<rect x="20" y="110" width="60" height="90" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="20" y1="128.0" x2="80" y2="128.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="20" y1="146.0" x2="80" y2="146.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="20" y1="164.0" x2="80" y2="164.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="20" y1="182.0" x2="80" y2="182.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="35.0" y1="110" x2="35.0" y2="200" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="50.0" y1="110" x2="50.0" y2="200" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="65.0" y1="110" x2="65.0" y2="200" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="20" y="110" width="60" height="90" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="50.0" y="155.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">X</text>
<text x="50.0" y="211" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × D</text>
<line x1="82" y1="155" x2="120" y2="155" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<text x="100" y="172" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">W<tspan dy="3.1" font-size="7.6">Q,K,V</tspan><tspan dy="-3.1">​</tspan></text>
<rect x="120" y="95" width="96" height="120" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="120.0" y="95.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="132.0" y="95.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="144.0" y="95.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="156.0" y="95.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="168.0" y="95.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="180.0" y="95.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="192.0" y="95.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="204.0" y="95.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="120.0" y="119.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="132.0" y="119.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="144.0" y="119.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="156.0" y="119.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="168.0" y="119.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="180.0" y="119.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="192.0" y="119.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="204.0" y="119.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="120.0" y="143.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="132.0" y="143.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="144.0" y="143.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="156.0" y="143.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="168.0" y="143.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="180.0" y="143.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="192.0" y="143.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="204.0" y="143.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="120.0" y="167.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="132.0" y="167.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="144.0" y="167.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="156.0" y="167.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="168.0" y="167.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="180.0" y="167.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="192.0" y="167.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="204.0" y="167.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="120.0" y="191.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="132.0" y="191.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="144.0" y="191.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="156.0" y="191.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="168.0" y="191.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="180.0" y="191.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="192.0" y="191.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="204.0" y="191.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<line x1="120" y1="119.0" x2="216" y2="119.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="143.0" x2="216" y2="143.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="167.0" x2="216" y2="167.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="191.0" x2="216" y2="191.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="132.0" y1="95" x2="132.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="144.0" y1="95" x2="144.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="156.0" y1="95" x2="156.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="168.0" y1="95" x2="168.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="180.0" y1="95" x2="180.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="192.0" y1="95" x2="192.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="204.0" y1="95" x2="204.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="120" y="95" width="96" height="120" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="168.0" y="226" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">Q, K, V each N × D</text>
<text x="168" y="84" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">column blocks = heads</text>
<text x="168" y="246" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px;font-family:ui-monospace,Menlo,Consolas,monospace">view(B,N,h,d<tspan dy="2.7" font-size="6.5">h</tspan><tspan dy="-2.7">​</tspan>).transpose(1,2)</text>
<line x1="218" y1="110" x2="270" y2="58" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<rect x="270" y="40" width="150" height="36" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="345.0" y="52.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">head 1: Attn(Q<tspan dy="3.1" font-size="7.6">1</tspan><tspan dy="-3.1">​</tspan>, K<tspan dy="3.1" font-size="7.6">1</tspan><tspan dy="-3.1">​</tspan>, V<tspan dy="3.1" font-size="7.6">1</tspan><tspan dy="-3.1">​</tspan>)</text>
<text x="345.0" y="65.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">N × N weights, out N × d<tspan dy="2.7" font-size="6.5">h</tspan><tspan dy="-2.7">​</tspan></text>
<line x1="420" y1="58" x2="470" y2="110" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<line x1="218" y1="135" x2="270" y2="118" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<rect x="270" y="100" width="150" height="36" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="345.0" y="112.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">head 2: Attn(Q<tspan dy="3.1" font-size="7.6">2</tspan><tspan dy="-3.1">​</tspan>, K<tspan dy="3.1" font-size="7.6">2</tspan><tspan dy="-3.1">​</tspan>, V<tspan dy="3.1" font-size="7.6">2</tspan><tspan dy="-3.1">​</tspan>)</text>
<text x="345.0" y="125.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">N × N weights, out N × d<tspan dy="2.7" font-size="6.5">h</tspan><tspan dy="-2.7">​</tspan></text>
<line x1="420" y1="118" x2="470" y2="135" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<line x1="218" y1="160" x2="270" y2="178" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<rect x="270" y="160" width="150" height="36" rx="6" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<text x="345.0" y="172.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">head 3: Attn(Q<tspan dy="3.1" font-size="7.6">3</tspan><tspan dy="-3.1">​</tspan>, K<tspan dy="3.1" font-size="7.6">3</tspan><tspan dy="-3.1">​</tspan>, V<tspan dy="3.1" font-size="7.6">3</tspan><tspan dy="-3.1">​</tspan>)</text>
<text x="345.0" y="185.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">N × N weights, out N × d<tspan dy="2.7" font-size="6.5">h</tspan><tspan dy="-2.7">​</tspan></text>
<line x1="420" y1="178" x2="470" y2="160" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<line x1="218" y1="185" x2="270" y2="238" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<rect x="270" y="220" width="150" height="36" rx="6" style="fill:#fbe9e7;stroke:#b3261e;stroke-width:1.2"/>
<text x="345.0" y="232.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">head 4: Attn(Q<tspan dy="3.1" font-size="7.6">4</tspan><tspan dy="-3.1">​</tspan>, K<tspan dy="3.1" font-size="7.6">4</tspan><tspan dy="-3.1">​</tspan>, V<tspan dy="3.1" font-size="7.6">4</tspan><tspan dy="-3.1">​</tspan>)</text>
<text x="345.0" y="245.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">N × N weights, out N × d<tspan dy="2.7" font-size="6.5">h</tspan><tspan dy="-2.7">​</tspan></text>
<line x1="420" y1="238" x2="470" y2="185" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<rect x="470" y="95" width="96" height="120" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="470.0" y="95.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="482.0" y="95.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="494.0" y="95.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="506.0" y="95.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="518.0" y="95.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="530.0" y="95.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="542.0" y="95.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="554.0" y="95.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="470.0" y="119.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="482.0" y="119.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="494.0" y="119.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="506.0" y="119.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="518.0" y="119.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="530.0" y="119.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="542.0" y="119.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="554.0" y="119.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="470.0" y="143.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="482.0" y="143.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="494.0" y="143.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="506.0" y="143.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="518.0" y="143.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="530.0" y="143.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="542.0" y="143.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="554.0" y="143.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="470.0" y="167.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="482.0" y="167.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="494.0" y="167.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="506.0" y="167.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="518.0" y="167.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="530.0" y="167.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="542.0" y="167.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="554.0" y="167.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="470.0" y="191.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="482.0" y="191.0" width="12.0" height="24.0" style="fill:#fff3d6;stroke:none"/>
<rect x="494.0" y="191.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="506.0" y="191.0" width="12.0" height="24.0" style="fill:#e7eef7;stroke:none"/>
<rect x="518.0" y="191.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="530.0" y="191.0" width="12.0" height="24.0" style="fill:#e6f2e8;stroke:none"/>
<rect x="542.0" y="191.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<rect x="554.0" y="191.0" width="12.0" height="24.0" style="fill:#fbe9e7;stroke:none"/>
<line x1="470" y1="119.0" x2="566" y2="119.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="470" y1="143.0" x2="566" y2="143.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="470" y1="167.0" x2="566" y2="167.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="470" y1="191.0" x2="566" y2="191.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="482.0" y1="95" x2="482.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="494.0" y1="95" x2="494.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="506.0" y1="95" x2="506.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="518.0" y1="95" x2="518.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="530.0" y1="95" x2="530.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="542.0" y1="95" x2="542.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="554.0" y1="95" x2="554.0" y2="215" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="470" y="95" width="96" height="120" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="518.0" y="226" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">concat: N × D</text>
<line x1="568" y1="155" x2="610" y2="155" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-heads-ink)"/>
<text x="589.0" y="141" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">W<tspan dy="3.3" font-size="7.9">O</tspan><tspan dy="-3.3">​</tspan></text>
<rect x="610" y="110" width="60" height="90" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="610" y1="128.0" x2="670" y2="128.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="610" y1="146.0" x2="670" y2="146.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="610" y1="164.0" x2="670" y2="164.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="610" y1="182.0" x2="670" y2="182.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="625.0" y1="110" x2="625.0" y2="200" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="640.0" y1="110" x2="640.0" y2="200" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="655.0" y1="110" x2="655.0" y2="200" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="610" y="110" width="60" height="90" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="640.0" y="155.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">out</text>
<text x="640.0" y="211" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × D</text>
<text x="380" y="269.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">h = 12, d<tspan dy="3.1" font-size="7.6">h</tspan><tspan dy="-3.1">​</tspan> = 64 in GPT-2 / BERT / ViT-B;  h = 32, d<tspan dy="3.1" font-size="7.6">h</tspan><tspan dy="-3.1">​</tspan> = 128 in Llama 2 7B</text>
<text x="380" y="282.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">W<tspan dy="3.1" font-size="7.6">O</tspan><tspan dy="-3.1">​</tspan> lets each head write into all D channels; the heads&#x27; contributions add</text>
</svg><figcaption><strong>Figure 8.</strong> Multi-head attention. The D channels of Q, K and V are split into h blocks of width d_h = D/h; each block runs its own attention with its own N×N weights; the h outputs are concatenated along channels and mixed by W_O. Parameters and FLOPs do not depend on h.</figcaption></figure>
```
**Multi-head attention** (Figure 8) runs $h$ attention operations on $h$ column blocks of $Q$, $K$ and $V$, each of width $d_h = D/h$, and concatenates the results before $W_O$:

$$
\operatorname{MHA}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O, \qquad \text{head}_i = \operatorname{Attn}\big(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}\big).
$$

In code $W_Q$ is one $D \times D$ matrix and the heads are `view(B, N, h, d_h).transpose(1, 2)`; the split must happen on the last axis before the transpose, or tokens and heads get mixed. The parameters are $4D^2 + 4D$ and the cost is $4ND^2 + 2N^2D$ MACs whatever $h$ is, since $h$ score products of size $N^2 d_h$ sum to $N^2 D$; what changes with $h$ is that there are $h$ independent $N \times N$ weight patterns per layer, each of rank at most $d_h$. One head would have to average everything a query needs into one distribution and one set of channels; several heads let a token attend to different things for different purposes and land them in different channel blocks. $W_O$ then maps each head's $d_h$ channels back into the full stream and the heads' contributions add, $\sum_i \text{head}_i W_O^{(i)}$; per head, $W_V^{(i)}W_O^{(i)}$ is a rank-$d_h$ read-and-write on the residual stream (Elhage et al.'s OV circuit, with $W_Q^{(i)}W_K^{(i)\top}$ the QK circuit). Typical widths: $d_h = 64$ in GPT-2, BERT and ViT-B; 128 in Llama 2.

::: {.qa asked="ByteDance, Kling, TME, Meta, NVIDIA (10+ reports)"}
**How does scaled dot-product attention work, and why divide by $\sqrt{d_k}$?**

Project the inputs to queries, keys and values; score each query against every key with a dot product, divide by $\sqrt{d_k}$, softmax over the keys, and return the weighted average of the values. Each output is a convex combination of values weighted by query–key similarity. With unit-variance independent components, $q^\top k$ has variance $d_k$, so unscaled logits have standard deviation 8 at $d_k = 64$, which saturates the softmax; near one-hot its Jacobian $\operatorname{diag}(p) - pp^\top$ vanishes and $Q$ and $K$ stop learning. The scaling restores unit variance for any head width. It is an initialization argument; QK-norm handles logit growth during training.
:::

::: {.qa asked="ByteDance, Kling, TME, Amap (Nowcoder)"}
**Why multiple heads, and where do Q, K, V come from in causal self-attention and in cross-attention?**

Each head has its own projections, so each query gets $h$ attention distributions based on different learned similarities, with results in separate channel blocks instead of averaged together; splitting $D$ into $h$ heads of width $D/h$ keeps parameters ($4D^2$) and FLOPs ($4ND^2 + 2N^2D$) unchanged, and $W_O$ mixes the heads back into the stream. In causal self-attention, $Q$, $K$, $V$ are linear maps of the same sequence and scores with future keys are set to $-\infty$. In cross-attention, $Q$ comes from the sequence being updated (decoder states, image tokens) and $K$, $V$ from the conditioning sequence (encoder outputs, text embeddings), so the output has the query sequence's length.
:::

::: qa
**Count the parameters and FLOPs of an MHA layer with width $D$, $h$ heads and $N$ tokens. Which depend on $h$?**

Parameters $4D^2 + 4D$. MACs $4ND^2$ for the projections and $2N^2D$ for $QK^\top$ and $AV$; double for FLOPs. Neither depends on $h$; only the number of $N \times N$ score matrices ($h$ per sample) and the per-head rank $d_h$ do.
:::

::: qa
**Your padded batch produces NaNs after the first attention layer. Likely cause?**

A query row in which every key is masked: softmax of all $-\infty$ is $0/0$. Typical causes: left padding with a causal mask, or an inverted mask convention. Let pad queries attend to at least one key and exclude them from the loss, or zero fully masked rows, and check that masked logits are not filled with a constant that overflows in half precision.
:::

## Position

Attention is permutation equivariant, so position is injected either by adding a vector to each token's input (absolute) or by making the scores depend on positions (RoPE, ALiBi, relative biases).

**Learned absolute.** A table $P_{\text{pos}} \in \mathbb{R}^{N_{\max} \times D}$ added to the embedding: 1024 rows in GPT-2, 512 in BERT, one per patch plus one for the class token in ViT. Nothing is defined beyond $N_{\max}$; for images the table is reshaped to its 2-D grid and interpolated when the resolution changes.

**Sinusoidal.** The 2017 fixed vector, $\mathrm{PE}(\text{pos}, 2i) = \sin(\text{pos}\,\omega_i)$, $\mathrm{PE}(\text{pos}, 2i+1) = \cos(\text{pos}\,\omega_i)$, $\omega_i = 10000^{-2i/D}$: each channel pair rotates at its own frequency, wavelengths from $2\pi$ to $10000 \cdot 2\pi$, so a shift by $k$ is a fixed rotation of every pair and relative position is linearly accessible. It is added to the content once at the bottom, so scores mix content and position terms, and it extrapolates poorly beyond the training length. DiT uses the 2-D version, half the channels for rows and half for columns.

```{=html}
<figure class="fig" id="fig-rope"><svg viewBox="0 0 760 250" role="img" aria-label="Rotary position embedding: query and key pairs rotated by their position angles; the score depends on the angle difference" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs><marker id="ah-rope-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--blue)"/></marker><marker id="ah-rope-gold" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--gold)"/></marker></defs>
<line x1="40" y1="130" x2="180" y2="130" style="stroke:var(--line);stroke-width:1;fill:none"/>
<line x1="110" y1="200" x2="110" y2="60" style="stroke:var(--line);stroke-width:1;fill:none"/>
<circle cx="110" cy="130" r="60" style="fill:none;stroke:var(--line);stroke-width:1;stroke-dasharray:3 3"/>
<text x="110" y="40" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">pair (q<tspan dy="3.6" font-size="8.6">1</tspan><tspan dy="-3.6">​</tspan>, q<tspan dy="3.6" font-size="8.6">2</tspan><tspan dy="-3.6">​</tspan>), (k<tspan dy="3.6" font-size="8.6">1</tspan><tspan dy="-3.6">​</tspan>, k<tspan dy="3.6" font-size="8.6">2</tspan><tspan dy="-3.6">​</tspan>)</text>
<line x1="110" y1="130" x2="166.36236277084274" y2="109.42613155267293" style="stroke:var(--gold);stroke-width:2;fill:none" marker-end="url(#ah-rope-gold)"/>
<text x="179.51358075070604" y="104.6255622482966" text-anchor="middle" dominant-baseline="middle" style="fill:var(--gold);font-size:11px">q</text>
<line x1="110" y1="130" x2="137.21576728553464" y2="76.52755839631388" style="stroke:var(--blue);stroke-width:2;fill:none" marker-end="url(#ah-rope-blue)"/>
<text x="143.56611298549274" y="64.05065535545378" text-anchor="middle" dominant-baseline="middle" style="fill:var(--blue);font-size:11px">k</text>
<line x1="310" y1="130" x2="450" y2="130" style="stroke:var(--line);stroke-width:1;fill:none"/>
<line x1="380" y1="200" x2="380" y2="60" style="stroke:var(--line);stroke-width:1;fill:none"/>
<circle cx="380" cy="130" r="60" style="fill:none;stroke:var(--line);stroke-width:1;stroke-dasharray:3 3"/>
<text x="380" y="40" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">rotate by position: q by mθ, k by nθ</text>
<line x1="380" y1="130" x2="398.91934174371613" y2="73.06092283866482" style="stroke:var(--gold);stroke-width:2;fill:none" marker-end="url(#ah-rope-gold)"/>
<text x="405.22578899162147" y="54.0812304515531" text-anchor="middle" dominant-baseline="middle" style="fill:var(--gold);font-size:11px">R<tspan dy="3.3" font-size="7.9">mθ</tspan><tspan dy="-3.3">​</tspan> q</text>
<line x1="380" y1="130" x2="328.58667479786317" y2="99.06991769071215" style="stroke:var(--blue);stroke-width:2;fill:none" marker-end="url(#ah-rope-blue)"/>
<text x="311.4488997304842" y="88.75989025428287" text-anchor="middle" dominant-baseline="middle" style="fill:var(--blue);font-size:11px">R<tspan dy="3.3" font-size="7.9">nθ</tspan><tspan dy="-3.3">​</tspan> k</text>
<path d="M417.57490851389514,116.28408770178194 A40,40 0 0,0 392.61289449581074,92.04061522577655" style="fill:none;stroke:var(--gold);stroke-width:1;stroke-dasharray:3 2"/>
<path d="M393.6078836427673,103.26377919815694 A30,30 0 0,0 354.29333739893156,114.53495884535607" style="fill:none;stroke:var(--blue);stroke-width:1;stroke-dasharray:3 2"/>
<text x="630" y="67.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">⟨R<tspan dy="3.4" font-size="8.3">mθ</tspan><tspan dy="-3.4">​</tspan> q, R<tspan dy="3.4" font-size="8.3">nθ</tspan><tspan dy="-3.4">​</tspan> k⟩ = qᵀ R<tspan dy="3.4" font-size="8.3">(n−m)θ</tspan><tspan dy="-3.4">​</tspan> k</text>
<text x="630" y="84.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px"></text>
<text x="630" y="101.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">angle between them:</text>
<text x="630" y="118.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">(n − m)θ + (original angle)</text>
<text x="630" y="135.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px"></text>
<text x="630" y="152.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">same offset ⇒ same score</text>
<text x="630" y="198.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">d/2 pairs per head, θ<tspan dy="3.0" font-size="7.2">i</tspan><tspan dy="-3.0">​</tspan> = 10000<tspan dy="-3.0" font-size="7.2">−2i/d</tspan><tspan dy="3.0">​</tspan>:</text>
<text x="630" y="211.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">fast pairs resolve near offsets, slow pairs far ones</text>
</svg><figcaption><strong>Figure 9.</strong> RoPE rotates each two-channel pair of a query by its position angle mθ and of a key by nθ. Because rotations compose by adding angles, the dot product of the rotated pair depends only on the offset n − m: the score is relative while the inputs stay absolute.</figcaption></figure>
```
**RoPE** (Figure 9) rotates rather than adds. Split a head's $d$ channels into $d/2$ pairs; rotate pair $i$ of the query at position $m$ by angle $m\theta_i$ and of the key at position $n$ by $n\theta_i$, with $\theta_i = 10000^{-2i/d}$. Since $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$,

$$
\big(R_m q\big)^\top \big(R_n k\big) = q^\top R_{n-m}\, k ,
$$

so the score depends on positions only through the offset $n - m$, with no parameters, applied to $q$ and $k$ after the projections in every layer and never to $v$. With $d = 128$ the fastest pair turns one radian per token and the slowest has a wavelength of about 54,400 tokens; Llama 3's base of 500,000 stretches that to about 2.6M. Beyond the training length $L$, the slow pairs see angles they never saw. Position interpolation (Chen et al.) replaces $m$ by $m/s$ for an extension factor $s$, which keeps every angle in range but slows the fast pairs too, so it needs a short fine-tune. NTK-aware scaling changes the base instead, $b' = b\, s^{d/(d-2)}$, which leaves the fastest pair untouched, divides the slowest frequency by exactly $s$, and scales the rest geometrically; YaRN refines this per frequency band. For images, axial 2-D RoPE rotates half of a head's pairs by the row coordinate and half by the column (FLUX, DINOv3); video adds a third group for time (M-RoPE).

**ALiBi** uses no position vectors: it subtracts $m_{\text{head}} \cdot (i - j)$ from each causal score, with slopes $2^{-8k/h}$ across heads, so some heads look locally and some far; it extrapolates to longer sequences better than sinusoidal encodings but builds in a recency bias. T5 instead learns a scalar bias per head per bucket of relative distance.

::: {.qa asked="Alibaba Amap, TME, ByteDance (Nowcoder)"}
**Compare sinusoidal positional encodings with RoPE.**

Sinusoidal adds a fixed vector to the input, pair $i$ being $(\sin \text{pos}\,\omega_i, \cos \text{pos}\,\omega_i)$; a shift by $k$ rotates each pair by a fixed angle, so relative position is linearly accessible, but position enters once at the bottom and scores contain content–position cross terms. RoPE rotates each pair of $q$ and $k$ by $m\theta_i$ in every layer, and $R_m^\top R_n = R_{n-m}$ makes the score depend only on the offset. Neither has parameters; RoPE leaves $v$ untouched and extends to longer contexts by rescaling positions or the base, usually with brief fine-tuning. For images and video the rotary channels are split across axes.
:::

::: {.qa asked="Company unnamed (via AIGC interview repo)"}
**Implement 1D rotary position embedding.**

Pair adjacent channels as a complex number and multiply by $e^{im\theta_i}$:

```python
import torch

def rope(x, pos, base=10000.0):
    # x: (..., T, d) with d even; pos: (T,) absolute positions (offset by the cache length when decoding)
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)        # (d/2,)
    rot = torch.polar(torch.ones_like(theta), pos.double()[:, None] * theta)  # e^{i m theta_i}, (T, d/2)
    xc = torch.view_as_complex(x.double().reshape(*x.shape[:-1], d // 2, 2))
    return torch.view_as_real(xc * rot).flatten(-2).to(x.dtype)
```

Apply per head to $q$ and $k$ after the projections, never to $v$; with a KV cache, rotate keys with their absolute positions before caching; compute angles in float32 or float64. Open-source LLM code often pairs channel $i$ with $i + d/2$ ("rotate half"); the conventions differ by a fixed channel permutation, so a checkpoint works only with its own.
:::

## The families and their objectives

| Family | Attention | Objective | Examples |
|---|---|---|---|
| Encoder-only | bidirectional | masked tokens (BERT: 15% of positions, 80/10/10 mask/random/keep); supervised or self-supervised targets (ViT); diffusion loss (DiT) | embeddings, retrieval, classification, dense prediction, diffusion backbones |
| Decoder-only | causal | next-token cross-entropy $-\frac1N \sum_t \log p_\theta(x_{t+1} \mid x_{\le t})$ | LLMs; autoregressive image and audio token models |
| Encoder–decoder | bidirectional encoder, causal decoder with cross-attention | cross-entropy on target given source; T5's span corruption | translation, summarization; T5's encoder is reused frozen by Imagen, SD3 and FLUX as a text encoder |

Perplexity is $\exp$ of the mean next-token cross-entropy; a freshly initialized model predicts nearly uniformly, so its loss should start near $\ln V$ (10.8 for GPT-2's vocabulary), and a much larger initial loss points to an initialization bug. Teacher forcing, feeding the true prefix during training, means the model never trains on its own samples; the mismatch at inference is exposure bias.

::: {.qa asked="Apple ML research intern (NLP team), Jointaro 2023"}
**Explain BERT: architecture, pretraining objectives, and how it differs from GPT.**

A transformer encoder: bidirectional attention, learned positions up to 512, WordPiece tokens, a `[CLS]` token and segment embeddings. Base: 12 layers, width 768, 12 heads, 110M parameters; Large: 24, 1024, 16, 340M. Pretraining: masked language modeling (15% of positions; 80% `[MASK]`, 10% random, 10% unchanged, so the model keeps a good representation of every token since it cannot tell which were altered) plus next-sentence prediction, which RoBERTa showed could be dropped. Fine-tuning adds a head on `[CLS]` or on every token. GPT is a causal decoder trained on next-token prediction: every position is a target and generation is native; BERT's bidirectional context gives stronger per-parameter representations for understanding but no left-to-right generative model.
:::

## Counting

Every number here was checked in Python against the released models.

**Parameters.** Per block: attention $4D^2 + 4D$, MLP $8D^2 + 5D$, two LayerNorms $4D$, so $12D^2 + 13D$; plus $VD$ for the embedding, $N_{\max}D$ for learned positions, $2D$ for the final norm, and nothing for a tied head.

| GPT-2 small ($L = 12$, $D = 768$, $V = 50{,}257$, context 1024, tied) | Count |
|---|---|
| Token embedding $VD$ | 38,597,376 |
| Position embedding $1024D$ | 786,432 |
| 12 blocks at $12D^2 + 13D = 7{,}087{,}872$ each | 85,054,464 |
| Final LayerNorm | 1,536 |
| **Total** (matches Hugging Face's `GPT2LMHeadModel`) | **124,439,808** |

Llama 2 7B ($L = 32$, $D = 4096$, SwiGLU $d_{\text{ff}} = 11{,}008$, $V = 32{,}000$, untied, no biases, RMSNorm): per block $4D^2 + 3Dd_{\text{ff}} + 2D = 202{,}383{,}360$; times 32, plus $2VD$ for the separate embedding and head and the final norm, gives 6,738,415,616.

**FLOPs per token, forward.** Applying a weight matrix to one token uses each weight once: 2 FLOPs per parameter per token. Attention adds, per layer, $4MD$ FLOPs for one query against $M$ keys, about $2ND$ averaged over a causal sequence of length $N$ if masked work is skipped; that is the $2 n_{\text{layer}} n_{\text{ctx}} d_{\text{model}}$ term in Kaplan et al.'s $C_{\text{forward}} \approx 2\Psi + 2n_{\text{layer}}n_{\text{ctx}}d_{\text{model}}$. For GPT-2 small at $N = 1024$: 170 MFLOPs for the block matrices, 77 for the (tied, but still multiplied) head, 19 to 38 for attention; the head is a large share in a small model and negligible in a 7B one.

**Training compute, $C \approx 6\Psi T$.** For one linear layer, the forward product costs $2TD_{\text{in}}D_{\text{out}}$ FLOPs and the backward pass is two products of the same size (input gradient and weight gradient), so 6 FLOPs per parameter per token; full activation checkpointing makes it about $8\Psi T$. Llama 2 7B on 2.0T tokens: $6 \times 6.74 \times 10^9 \times 2 \times 10^{12} = 8.1 \times 10^{22}$ FLOPs; Meta's 184,320 A100-hours imply $1.22 \times 10^{14}$ FLOP/s per GPU, 39% of the 312 TFLOP/s bf16 peak, which is the model FLOPs utilization. Chinchilla's compute-optimal rule is about 20 tokens per parameter; Llama-style models go far past it because a small model is cheap to serve.

**Activation memory.** Per layer without recomputation, about $sbD(34 + 5as/D)$ bytes for sequence length $s$, batch $b$, $a$ heads and 16-bit activations (Korthikanti et al.); the $5as^2b$ part is the score-sized tensors. GPT-2 small at $s = 1024$, $b = 1$: 85.5 MiB per layer, 1 GiB for 12 layers, against about 2 GB of model state for mixed-precision Adam (16 bytes per parameter). FlashAttention removes the $s^2$ term; checkpointing trades compute for the rest.

**KV cache.** $2 \times L \times h_{\text{kv}} \times d_h \times N \times B$ numbers. Llama 2 7B in fp16: $2 \cdot 32 \cdot 32 \cdot 128 \cdot 2$ bytes $= 512$ KiB per token, 2 GiB for 4096 tokens, 32 GiB for a batch of 16 such sequences, more than the 13.5 GB of weights.

**When attention dominates.** Per layer $4ND^2$ (projections) $+ 8ND^2$ (MLP) $+ 2N^2D$ (scores and weighted sum) MACs, so attention exceeds the MLP at $N > 4D$ and all weight matmuls at $N > 6D$:

| Width | Attention > MLP | Attention > all matmuls |
|---|---|---|
| $D = 768$ (GPT-2 small) | 3,072 tokens | 4,608 |
| $D = 1152$ (DiT-XL) | 4,608 | 6,912 |
| $D = 4096$ (Llama 2 7B) | 16,384 | 24,576 |

At GPT-2's 1024-token context attention is 22% of the block matmuls; DiT-XL/2 at 256 tokens spends under 4% on scores, and summing $12ND^2 + 2N^2D$ over its 28 layers reproduces the paper's 118.6 Gflops (which therefore counts MACs). At $1024^2$ images with 4,096 or 16,384 tokens the balance flips, which is the motivation for windowed and factorized attention on the vision side.

::: {.qa asked="ByteDance 2026, Alibaba Taotian, OpenAI applied RS"}
**Complexity of a transformer layer, which term dominates for long sequences, and how would you change attention for longer inputs?**

$12ND^2 + 2N^2D$ MACs per layer: $4ND^2$ for Q, K, V, O, $8ND^2$ for a $4\times$ MLP, $2N^2D$ for scores and the weighted sum. The quadratic term passes the MLP at $N > 4D$ and everything at $N > 6D$ (4,608 tokens at $D = 768$), and materialized scores need $hN^2$ numbers. For longer inputs: FlashAttention (exact, removes the $N^2$ memory but not the FLOPs); sliding-window or block-sparse attention with a few global tokens (linear in $N$); linear or low-rank attention at some quality cost; factorized attention for video (spatial then temporal); RoPE scaling to stretch the context; GQA or MQA when the KV cache is the bottleneck. Choose by which resource binds: compute, activation memory, or cache memory.
:::

::: {.qa asked="Runway (Glassdoor 2025–26)"}
**Estimate the compute and memory needed to train a given network.**

Compute about $6\Psi T$ FLOPs plus attention terms for long contexts, divided by 30 to 50% of peak throughput; Llama 2 7B on 2T tokens is $8.1 \times 10^{22}$ FLOPs and its reported A100-hours imply 39% of peak. Memory: about 16 bytes per parameter of model state with mixed-precision Adam, sharded with ZeRO or FSDP, plus activations of roughly $sbD(34 + 5as/D)$ bytes per layer, reduced by FlashAttention and checkpointing.
:::

## Inference

```{=html}
<figure class="fig" id="fig-kvcache"><svg viewBox="0 0 760 250" role="img" aria-label="KV cache during decoding: cached keys and values grow by one row per step; the new query attends over all of them" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs><marker id="ah-kvcache-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--blue)"/></marker><marker id="ah-kvcache-gold" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--gold)"/></marker><marker id="ah-kvcache-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:#2e7d32"/></marker><marker id="ah-kvcache-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="130" y="22" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">step t: new token only</text>
<rect x="40" y="40" width="180" height="32" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="130.0" y="56.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">x<tspan dy="3.4" font-size="8.3">t</tspan><tspan dy="-3.4">​</tspan>  (1 × D)</text>
<line x1="130" y1="72" x2="130" y2="96" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-kvcache-ink)"/>
<text x="200" y="84" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">W<tspan dy="3.1" font-size="7.6">Q</tspan><tspan dy="-3.1">​</tspan>, W<tspan dy="3.1" font-size="7.6">K</tspan><tspan dy="-3.1">​</tspan>, W<tspan dy="3.1" font-size="7.6">V</tspan><tspan dy="-3.1">​</tspan></text>
<rect x="40" y="96" width="50" height="30" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="65.0" y="111.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">q<tspan dy="3.4" font-size="8.3">t</tspan><tspan dy="-3.4">​</tspan></text>
<rect x="105" y="96" width="50" height="30" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="130.0" y="111.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">k<tspan dy="3.4" font-size="8.3">t</tspan><tspan dy="-3.4">​</tspan></text>
<rect x="170" y="96" width="50" height="30" rx="6" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<text x="195.0" y="111.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">v<tspan dy="3.4" font-size="8.3">t</tspan><tspan dy="-3.4">​</tspan></text>
<text x="430" y="22" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">cache per layer (B, h<tspan dy="3.8" font-size="9.0">kv</tspan><tspan dy="-3.8">​</tspan>, t, d<tspan dy="3.8" font-size="9.0">h</tspan><tspan dy="-3.8">​</tspan>)</text>
<rect x="330" y="40" width="90" height="120" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="330" y1="60.0" x2="420" y2="60.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="330" y1="80.0" x2="420" y2="80.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="330" y1="100.0" x2="420" y2="100.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="330" y1="120.0" x2="420" y2="120.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="330" y1="140.0" x2="420" y2="140.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="360.0" y1="40" x2="360.0" y2="160" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="390.0" y1="40" x2="390.0" y2="160" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="330" y="40" width="90" height="120" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="375.0" y="100.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">K</text>
<text x="375.0" y="171" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">keys 1..t−1</text>
<rect x="450" y="40" width="90" height="120" rx="2" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<line x1="450" y1="60.0" x2="540" y2="60.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="450" y1="80.0" x2="540" y2="80.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="450" y1="100.0" x2="540" y2="100.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="450" y1="120.0" x2="540" y2="120.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="450" y1="140.0" x2="540" y2="140.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="480.0" y1="40" x2="480.0" y2="160" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="510.0" y1="40" x2="510.0" y2="160" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="450" y="40" width="90" height="120" rx="2" style="fill:none;stroke:#2e7d32;stroke-width:1.2"/>
<text x="495.0" y="100.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">V</text>
<text x="495.0" y="171" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">values 1..t−1</text>
<rect x="330" y="160" width="90" height="20" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2;stroke-dasharray:5 4"/>
<rect x="450" y="160" width="90" height="20" rx="2" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="375" y="170" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:9.5px">k<tspan dy="2.9" font-size="6.8">t</tspan><tspan dy="-2.9">​</tspan> appended</text>
<text x="495" y="170" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:9.5px">v<tspan dy="2.9" font-size="6.8">t</tspan><tspan dy="-2.9">​</tspan> appended</text>
<polyline points="155,111 260,111 260,170 328,170" style="stroke:var(--blue);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-kvcache-blue)"/>
<polyline points="220,111 240,111 240,190 448,190 448,170" style="stroke:#2e7d32;stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-kvcache-green)"/>
<polyline points="65,126 65,215 600,215" style="stroke:var(--gold);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-kvcache-gold)"/>
<text x="330" y="228" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">q<tspan dy="3.1" font-size="7.6">t</tspan><tspan dy="-3.1">​</tspan> · [K; k<tspan dy="3.1" font-size="7.6">t</tspan><tspan dy="-3.1">​</tspan>]ᵀ → softmax → · [V; v<tspan dy="3.1" font-size="7.6">t</tspan><tspan dy="-3.1">​</tspan>]   (1 × t scores, no mask needed)</text>
<rect x="600" y="196" width="140" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="670.0" y="209.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">output for token t</text>
<text x="670.0" y="222.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">then next layer</text>
<text x="650" y="70.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">why exact: with a causal mask,</text>
<text x="650" y="85.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">k<tspan dy="3.1" font-size="7.6">j</tspan><tspan dy="-3.1">​</tspan>, v<tspan dy="3.1" font-size="7.6">j</tspan><tspan dy="-3.1">​</tspan> depend only on</text>
<text x="650" y="100.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">tokens ≤ j, so they never change</text>
<text x="650" y="115.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px"></text>
<text x="650" y="130.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">size: 2 · L · h<tspan dy="3.1" font-size="7.6">kv</tspan><tspan dy="-3.1">​</tspan> · d<tspan dy="3.1" font-size="7.6">h</tspan><tspan dy="-3.1">​</tspan> · t numbers</text>
</svg><figcaption><strong>Figure 10.</strong> Decoding with a KV cache. Under a causal mask the keys and values of earlier positions never change, so each layer stores them; a new token computes only its own q, k, v, appends k and v, and attends its single query over the whole cache. Prefill fills the cache from the prompt in one pass.</figcaption></figure>
```
**The KV cache** (Figure 10). Generation runs the prompt once (prefill), then appends one token per step. Without caching, step $t$ reruns the whole prefix. Under a causal mask the hidden state at position $j$ depends only on tokens $\le j$, so $k_j$ and $v_j$ computed when $j$ was the newest token are exactly what a recomputation would give; each layer stores them, of shape $(B, h_{\text{kv}}, t, d_h)$ with RoPE already applied, and a step computes $q$, $k$, $v$ for the new token only, appends $k$ and $v$, and attends one query over the cache with no mask. This is exact, not an approximation. It does not apply to bidirectional attention: a DiT recomputes everything at every denoising step because every token changes.

**Prefill is compute-bound, decode is memory-bound.** Prefill multiplies each weight by all $N_p$ prompt tokens. Decode reads every weight from HBM once per step and uses it for 2 FLOPs per sequence, about $B$ FLOPs per byte in fp16, while an A100 needs about 150 FLOPs per byte (312 TFLOP/s over 2 TB/s) to be compute-bound. At batch 1 a 7B fp16 model needs at least $13.5\ \text{GB} / 2\ \text{TB/s} \approx 6.6$ ms per token, about 150 tokens per second, however fast the arithmetic is; each sequence also reads its own cache every step. Remedies: larger batches, quantized weights and caches, smaller caches, paged cache allocation (PagedAttention), and speculative decoding, where a small draft model proposes several tokens that the large model verifies in one pass without changing the output distribution.

```{=html}
<figure class="fig" id="fig-gqa"><svg viewBox="0 0 760 225" role="img" aria-label="Multi-head, grouped-query and multi-query attention: how query heads map to key-value heads" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs></defs>
<text x="125" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">MHA</text>
<text x="125" y="38" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">Llama 2 7B</text>
<rect x="12.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="39.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="67.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="94.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="122.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="149.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="177.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="204.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="125" y="52" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">query heads (h = 8)</text>
<rect x="12.0" y="140" width="23.5" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="39.5" y="140" width="23.5" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="67.0" y="140" width="23.5" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="94.5" y="140" width="23.5" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="122.0" y="140" width="23.5" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="149.5" y="140" width="23.5" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="177.0" y="140" width="23.5" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="204.5" y="140" width="23.5" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="125" y="182" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">K, V heads (h<tspan dy="3.0" font-size="7.2">kv</tspan><tspan dy="-3.0">​</tspan> = 8)</text>
<line x1="23.75" y1="90" x2="23.75" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="51.25" y1="90" x2="51.25" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="78.75" y1="90" x2="78.75" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="106.25" y1="90" x2="106.25" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="133.75" y1="90" x2="133.75" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="161.25" y1="90" x2="161.25" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="188.75" y1="90" x2="188.75" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="216.25" y1="90" x2="216.25" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<text x="125" y="206" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">KV cache = 8/8 of MHA</text>
<text x="385" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">GQA</text>
<text x="385" y="38" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">Llama 2 70B, Llama 3, Mistral (groups of 4 or 8)</text>
<rect x="272.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="299.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="327.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="354.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="382.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="409.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="437.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="464.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="385" y="52" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">query heads (h = 8)</text>
<rect x="272.0" y="140" width="106.0" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="382.0" y="140" width="106.0" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="385" y="182" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">K, V heads (h<tspan dy="3.0" font-size="7.2">kv</tspan><tspan dy="-3.0">​</tspan> = 2)</text>
<line x1="283.75" y1="90" x2="325.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="311.25" y1="90" x2="325.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="338.75" y1="90" x2="325.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="366.25" y1="90" x2="325.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="393.75" y1="90" x2="435.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="421.25" y1="90" x2="435.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="448.75" y1="90" x2="435.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="476.25" y1="90" x2="435.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<text x="385" y="206" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">KV cache = 2/8 of MHA</text>
<text x="645" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">MQA</text>
<text x="645" y="38" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">PaLM, Falcon</text>
<rect x="532.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="559.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="587.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="614.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="642.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="669.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="697.0" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="724.5" y="60" width="23.5" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="645" y="52" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">query heads (h = 8)</text>
<rect x="532.0" y="140" width="216.0" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="645" y="182" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">K, V heads (h<tspan dy="3.0" font-size="7.2">kv</tspan><tspan dy="-3.0">​</tspan> = 1)</text>
<line x1="543.75" y1="90" x2="640.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="571.25" y1="90" x2="640.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="598.75" y1="90" x2="640.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="626.25" y1="90" x2="640.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="653.75" y1="90" x2="640.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="681.25" y1="90" x2="640.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="708.75" y1="90" x2="640.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<line x1="736.25" y1="90" x2="640.0" y2="140" style="stroke:var(--muted);stroke-width:1;fill:none"/>
<text x="645" y="206" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">KV cache = 1/8 of MHA</text>
</svg><figcaption><strong>Figure 11.</strong> Standard multi-head attention gives every query head its own key and value head. Grouped-query attention shares one K/V head among a group of query heads; multi-query attention shares a single K/V head among all of them. Only the K and V side shrinks, so the cache and the K/V projections shrink by h_kv / h while the query side is unchanged.</figcaption></figure>
```
**MQA and GQA** (Figure 11) shrink the cache by sharing key and value heads among query heads: one K/V head for all (multi-query) or $h_{\text{kv}}$ groups (grouped-query); the K and V projections shrink from $D \times D$ to $D \times h_{\text{kv}}d_h$ and the cache by $h_{\text{kv}}/h$.

| Configuration | $L$ | $h$ | $h_{\text{kv}}$ | Cache per token (fp16) | 4096 tokens |
|---|---|---|---|---|---|
| Llama 2 7B (MHA) | 32 | 32 | 32 | 512 KiB | 2 GiB |
| Mistral 7B (GQA) | 32 | 32 | 8 | 128 KiB | 0.5 GiB |
| Llama 2 70B (GQA) | 80 | 64 | 8 | 320 KiB | 1.25 GiB |
| Llama 2 70B with MHA | 80 | 64 | 64 | 2.5 MiB | 10 GiB |

GQA's quality is close to MHA's and MQA's somewhat lower; Ainslie et al. convert an MHA checkpoint by mean-pooling K/V heads within each group and uptraining with about 5% of the pretraining compute.

**Choosing the next token.** $p = \operatorname{softmax}(z/\tau)$; $\tau \to 0$ is greedy. Top-$k$ keeps the $k$ most likely tokens; top-$p$ keeps the smallest set whose probability reaches $p$, which adapts to the model's confidence. Beam search keeps the $B$ best partial sequences by total log-probability and suits tasks with one right answer; for open-ended text it produces bland, repetitive output.

::: {.qa asked="ByteDance LLM algorithm, Nowcoder 2026"}
**What does a KV cache store, how large is it, and how do MQA and GQA change it?**

Keys and values of past tokens per layer, which never change under a causal mask, so each step computes $q$, $k$, $v$ for the new token only and attends one query over the cache; exact, not approximate. Size $2 \cdot L \cdot h_{\text{kv}} \cdot d_h \cdot N \cdot B$ numbers: 512 KiB per token for Llama 2 7B in fp16, 2 GiB at 4096 tokens. MQA shares one K/V head across all query heads (cache divided by $h$); GQA uses $h_{\text{kv}}$ shared heads (Mistral 7B and Llama 2 70B use 8). Since decoding is bandwidth-bound, a smaller cache directly speeds it up and allows larger batches, at a small quality cost.
:::

::: {.qa asked="Tencent multimodal, Nowcoder 2026"}
**How does an LLM turn hidden states into token probabilities, and how does decoding choose the next token?**

The last position's final vector goes through the final norm and $W_U \in \mathbb{R}^{V \times D}$ (often tied to the input embedding) to give $V$ logits, and $\operatorname{softmax}(z/\tau)$ gives the next-token distribution; the chain rule $\log p(x_{1:n}) = \sum_t \log p(x_t \mid x_{<t})$ is why training minimizes mean negative log-probability. Decoding: greedy (argmax, deterministic, repetitive), beam search (best $B$ partial sequences, good for translation, needs length normalization), top-$k$, top-$p$ (adapts to confidence), with temperature sharpening or flattening first.
:::

::: pitfall
Three KV-cache bugs. Positions: the new token's position is the cache length, for learned embeddings and for RoPE. Masks: PyTorch's `scaled_dot_product_attention(..., is_causal=True)` aligns the triangle to the top-left when query and key lengths differ, so a single new query sees only the first key; pass an explicit mask, or `is_causal=False` for one query. Scope: the cache is valid only for causal attention, so it does not carry across the denoising steps of a diffusion transformer.
:::

## Efficient attention

```{=html}
<figure class="fig" id="fig-flash"><svg viewBox="0 0 760 330" role="img" aria-label="Standard attention versus FlashAttention: what is written to HBM and what stays in SRAM" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:760px;margin:0 auto"><defs><marker id="ah-flash-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="180" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">Standard attention</text>
<rect x="20" y="36" width="320" height="262" rx="8" style="fill:#eceff4;stroke:var(--muted);stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="40" y="50" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">HBM (GPU memory, ~2 TB/s)</text>
<rect x="40" y="70" width="40" height="60" rx="2" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<line x1="40" y1="85.0" x2="80" y2="85.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="40" y1="100.0" x2="80" y2="100.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="40" y1="115.0" x2="80" y2="115.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="60.0" y1="70" x2="60.0" y2="130" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="40" y="70" width="40" height="60" rx="2" style="fill:none;stroke:var(--gold);stroke-width:1.2"/>
<text x="60.0" y="100.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">Q</text>
<rect x="90" y="70" width="40" height="60" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="90" y1="85.0" x2="130" y2="85.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="90" y1="100.0" x2="130" y2="100.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="90" y1="115.0" x2="130" y2="115.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="110.0" y1="70" x2="110.0" y2="130" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="90" y="70" width="40" height="60" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="110.0" y="100.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">K</text>
<rect x="140" y="70" width="40" height="60" rx="2" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<line x1="140" y1="85.0" x2="180" y2="85.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="140" y1="100.0" x2="180" y2="100.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="140" y1="115.0" x2="180" y2="115.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="160.0" y1="70" x2="160.0" y2="130" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="140" y="70" width="40" height="60" rx="2" style="fill:none;stroke:#2e7d32;stroke-width:1.2"/>
<text x="160.0" y="100.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">V</text>
<rect x="210" y="60" width="80" height="80" rx="2" style="fill:#fbe9e7;stroke:#b3261e;stroke-width:1.2"/>
<line x1="210" y1="76.0" x2="290" y2="76.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="210" y1="92.0" x2="290" y2="92.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="210" y1="108.0" x2="290" y2="108.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="210" y1="124.0" x2="290" y2="124.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="226.0" y1="60" x2="226.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="242.0" y1="60" x2="242.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="258.0" y1="60" x2="258.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="274.0" y1="60" x2="274.0" y2="140" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="210" y="60" width="80" height="80" rx="2" style="fill:none;stroke:#b3261e;stroke-width:1.2"/>
<text x="250.0" y="100.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px;font-weight:700">S = QKᵀ</text>
<text x="250.0" y="151" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × N, written</text>
<rect x="210" y="180" width="80" height="80" rx="2" style="fill:#fbe9e7;stroke:#b3261e;stroke-width:1.2"/>
<line x1="210" y1="196.0" x2="290" y2="196.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="210" y1="212.0" x2="290" y2="212.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="210" y1="228.0" x2="290" y2="228.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="210" y1="244.0" x2="290" y2="244.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="226.0" y1="180" x2="226.0" y2="260" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="242.0" y1="180" x2="242.0" y2="260" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="258.0" y1="180" x2="258.0" y2="260" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="274.0" y1="180" x2="274.0" y2="260" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="210" y="180" width="80" height="80" rx="2" style="fill:none;stroke:#b3261e;stroke-width:1.2"/>
<text x="250.0" y="220.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px;font-weight:700">P = softmax</text>
<text x="250.0" y="271" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">N × N, written and read</text>
<rect x="40" y="190" width="60" height="60" rx="2" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<line x1="40" y1="205.0" x2="100" y2="205.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="40" y1="220.0" x2="100" y2="220.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="40" y1="235.0" x2="100" y2="235.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="70.0" y1="190" x2="70.0" y2="250" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="40" y="190" width="60" height="60" rx="2" style="fill:none;stroke:#2e7d32;stroke-width:1.2"/>
<text x="70.0" y="220.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px;font-weight:700">O = PV</text>
<line x1="182" y1="100" x2="208" y2="100" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-flash-ink)"/>
<line x1="250" y1="152" x2="250" y2="178" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-flash-ink)"/>
<line x1="208" y1="220" x2="102" y2="220" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-flash-ink)"/>
<text x="180" y="286" text-anchor="middle" dominant-baseline="middle" style="fill:#b3261e;font-size:10.5px">4 passes over N² numbers: memory-bound</text>
<text x="560" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px;font-weight:700">FlashAttention</text>
<rect x="380" y="36" width="360" height="100" rx="8" style="fill:#eceff4;stroke:var(--muted);stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="400" y="50" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px;font-style:italic">HBM</text>
<rect x="400" y="66" width="40" height="60" rx="2" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<line x1="400" y1="81.0" x2="440" y2="81.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="400" y1="96.0" x2="440" y2="96.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="400" y1="111.0" x2="440" y2="111.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="420.0" y1="66" x2="420.0" y2="126" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="400" y="66" width="40" height="60" rx="2" style="fill:none;stroke:var(--gold);stroke-width:1.2"/>
<text x="420.0" y="96.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">Q</text>
<rect x="450" y="66" width="40" height="60" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="450" y1="81.0" x2="490" y2="81.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="450" y1="96.0" x2="490" y2="96.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="450" y1="111.0" x2="490" y2="111.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="470.0" y1="66" x2="470.0" y2="126" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="450" y="66" width="40" height="60" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="470.0" y="96.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">K</text>
<rect x="500" y="66" width="40" height="60" rx="2" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<line x1="500" y1="81.0" x2="540" y2="81.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="500" y1="96.0" x2="540" y2="96.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="500" y1="111.0" x2="540" y2="111.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="520.0" y1="66" x2="520.0" y2="126" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="500" y="66" width="40" height="60" rx="2" style="fill:none;stroke:#2e7d32;stroke-width:1.2"/>
<text x="520.0" y="96.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">V</text>
<rect x="660" y="66" width="60" height="60" rx="2" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<line x1="660" y1="81.0" x2="720" y2="81.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="660" y1="96.0" x2="720" y2="96.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="660" y1="111.0" x2="720" y2="111.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="690.0" y1="66" x2="690.0" y2="126" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="660" y="66" width="60" height="60" rx="2" style="fill:none;stroke:#2e7d32;stroke-width:1.2"/>
<text x="690.0" y="96.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">O</text>
<text x="690" y="56" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">only O is written</text>
<rect x="380" y="172" width="360" height="126" rx="8" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="400" y="186" text-anchor="start" dominant-baseline="middle" style="fill:var(--blue);font-size:10.5px;font-style:italic">SRAM (on-chip, ~19 TB/s, ~192 KB per SM)</text>
<rect x="400" y="202" width="40" height="30" rx="2" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<line x1="400" y1="217.0" x2="440" y2="217.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="420.0" y1="202" x2="420.0" y2="232" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="400" y="202" width="40" height="30" rx="2" style="fill:none;stroke:var(--gold);stroke-width:1.2"/>
<text x="420.0" y="217.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px;font-weight:700">Q<tspan dy="3.0" font-size="7.2">i</tspan><tspan dy="-3.0">​</tspan></text>
<rect x="450" y="202" width="40" height="30" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="450" y1="217.0" x2="490" y2="217.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="470.0" y1="202" x2="470.0" y2="232" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="450" y="202" width="40" height="30" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="470.0" y="217.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px;font-weight:700">K<tspan dy="3.0" font-size="7.2">j</tspan><tspan dy="-3.0">​</tspan></text>
<rect x="500" y="202" width="40" height="30" rx="2" style="fill:#e6f2e8;stroke:#2e7d32;stroke-width:1.2"/>
<line x1="500" y1="217.0" x2="540" y2="217.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="520.0" y1="202" x2="520.0" y2="232" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="500" y="202" width="40" height="30" rx="2" style="fill:none;stroke:#2e7d32;stroke-width:1.2"/>
<text x="520.0" y="217.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px;font-weight:700">V<tspan dy="3.0" font-size="7.2">j</tspan><tspan dy="-3.0">​</tspan></text>
<rect x="560" y="202" width="40" height="30" rx="2" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<line x1="560" y1="217.0" x2="600" y2="217.0" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="580.0" y1="202" x2="580.0" y2="232" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="560" y="202" width="40" height="30" rx="2" style="fill:none;stroke:var(--blue);stroke-width:1.2"/>
<text x="580.0" y="217.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px;font-weight:700">S<tspan dy="3.0" font-size="7.2">ij</tspan><tspan dy="-3.0">​</tspan></text>
<text x="670" y="210.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">per query row: m, ℓ, o</text>
<text x="670" y="223.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">rescaled by e<tspan dy="-3.0" font-size="7.2">m−m&#x27;</tspan><tspan dy="3.0">​</tspan></text>
<text x="560" y="256" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">loop j over K/V blocks; Q<tspan dy="3.0" font-size="7.2">i</tspan><tspan dy="-3.0">​</tspan> rows never leave SRAM</text>
<text x="560" y="274" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">backward: recompute S<tspan dy="3.0" font-size="7.2">ij</tspan><tspan dy="-3.0">​</tspan> from Q, K and the saved log-sum-exp</text>
<line x1="420" y1="128" x2="420" y2="200" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-dasharray:5 4" marker-end="url(#ah-flash-ink)"/>
<line x1="470" y1="128" x2="470" y2="200" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-dasharray:5 4" marker-end="url(#ah-flash-ink)"/>
<text x="495" y="152" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">block j of K, V</text>
<line x1="520" y1="128" x2="520" y2="200" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-dasharray:5 4" marker-end="url(#ah-flash-ink)"/>
<line x1="690" y1="200" x2="690" y2="128" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-dasharray:5 4" marker-end="url(#ah-flash-ink)"/>
<text x="560" y="316" text-anchor="middle" dominant-baseline="middle" style="fill:var(--blue);font-size:10.5px">no N × N tensor anywhere: memory O(N), FLOPs still O(N²d)</text>
</svg><figcaption><strong>Figure 12.</strong> FlashAttention never materializes the N×N score matrix. Standard attention writes S and P to GPU memory (HBM) and reads them back, which is memory-bound. FlashAttention loads a block of queries and loops over blocks of keys and values inside fast on-chip SRAM, keeping a running max, running normalizer and running output per query row (online softmax), and writes only the final output. Same result, O(N) extra memory instead of O(N²), far less traffic; FLOPs unchanged.</figcaption></figure>
```
**FlashAttention** (Figure 12). A plain implementation writes the $N \times N$ scores to HBM, reads them back for the softmax, writes the probabilities, and reads them again to multiply by $V$; those steps do few FLOPs per byte, so attention is memory-bound, and the $N \times N$ tensors are the activation-memory peak. FlashAttention computes attention in tiles that fit in SRAM inside one kernel and never writes $S$ or $P$. The obstacle, that a row's softmax normalizer needs the whole row, is removed by the online softmax: for each query row keep the running maximum $m$, the running normalizer $\ell = \sum_j e^{s_j - m}$ and the running output $o = \sum_j e^{s_j - m} v_j$; for each new block of keys with maximum $m'$,

$$
\ell' = e^{m - m'}\ell + \sum_{j \in \mathcal B} e^{s_j - m'}, \qquad o' = e^{m - m'} o + \sum_{j \in \mathcal B} e^{s_j - m'} v_j ,
$$

and the final output is $o/\ell$, exactly the softmax-weighted average because the common factor $e^{-m}$ cancels. The kernel writes only the output and one number per row (the log-sum-exp), and the backward pass recomputes score tiles from $Q$, $K$ and that statistic rather than reading a stored $P$. HBM traffic drops from $\Omega(Nd + N^2)$ to $O(N^2d^2/M_{\text{SRAM}})$, extra memory from $O(N^2)$ to $O(N)$, and wall-clock drops because the operation was memory-bound; FLOPs stay $O(N^2 d)$ and slightly increase in the backward pass. It is exact. PyTorch's `scaled_dot_product_attention` dispatches to kernels of this kind.

**Sliding-window attention.** Each query attends to the previous $W$ keys: cost $O(NWD)$, cache capped at $W$ entries per layer, and information still travels about $L \cdot W$ tokens through depth (Mistral 7B: $W = 4096$, 32 layers). Longformer and BigBird add a few global tokens; Swin's windows are the image version.

**Linear attention.** Replace $e^{q^\top k}$ by $\phi(q)^\top \phi(k)$ for a positive feature map $\phi: \mathbb{R}^d \to \mathbb{R}^r$; then $\text{out}_i = \phi(q_i)^\top \big(\sum_j \phi(k_j)v_j^\top\big) / \phi(q_i)^\top \sum_j \phi(k_j)$, the $r \times d_v$ summary is shared by all queries, cost is $O(Nrd)$, and the causal version is a recurrence with a fixed-size matrix state. It loses sharpness: the implicit weight matrix has rank at most $r$, weight ratios grow polynomially rather than exponentially in the score gap, and the state must compress the whole context. Retrieving one exact token from a long context is where it suffers; gated and decaying recurrent variants narrow the gap.

::: qa
**Is FlashAttention an approximation? What does it reduce?**

No: tiling plus the online softmax give the same softmax attention up to rounding. It reduces HBM traffic and extra memory (from $O(N^2)$ to $O(N)$) because the scores never leave SRAM and the backward pass recomputes them; wall-clock drops because attention was memory-bound. FLOPs stay $O(N^2 d)$.
:::

::: qa
**What does linear attention gain and lose?**

The key–value summary $\sum_j \phi(k_j)v_j^\top$ is computed once, so cost is linear in $N$ and the causal form is an RNN with constant memory per step. It loses softmax's exponential sharpness and full-rank weight patterns, so exact retrieval from long contexts degrades.
:::

## Vision and generation backbones

The blocks above are modality-agnostic; what differs between ViT, DiT, the diffusion U-Net and MMDiT is how the input becomes tokens, how the condition enters, and whether the attention is global, windowed or split across streams.

### ViT

```{=html}
<figure class="fig" id="fig-vit"><svg viewBox="0 0 780 300" role="img" aria-label="Vision Transformer: image to patches to tokens, class token and positions, encoder blocks, classification head" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:780px;margin:0 auto"><defs><marker id="ah-vit-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<rect x="20" y="60" width="90" height="90" rx="2" style="fill:#eceff4;stroke:var(--muted);stroke-width:1.2"/>
<line x1="50" y1="60" x2="50" y2="150" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="20" y1="90" x2="110" y2="90" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="80" y1="60" x2="80" y2="150" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<line x1="20" y1="120" x2="110" y2="120" style="stroke:var(--muted);stroke-width:0.8;fill:none"/>
<text x="65" y="164" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">224 × 224 × 3</text>
<line x1="112" y1="105" x2="150" y2="105" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-vit-ink)"/>
<text x="131.0" y="92" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">P = 16</text>
<text x="150" y="46" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">flatten + linear (768 → D)</text>
<rect x="150" y="90" width="22" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="176" y="90" width="22" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="202" y="90" width="22" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="228" y="90" width="22" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="254" y="90" width="22" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="280" y="90" width="22" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="306" y="90" width="22" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="332" y="90" width="22" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="358" y="90" width="22" height="30" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="390" y="90" width="22" height="30" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="401" y="105" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:9px">cls</text>
<text x="285" y="150" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">196 patch tokens + 1 class token = 197 × D</text>
<text x="285" y="180" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">+ learned position embedding (197 × D)</text>
<circle cx="265" cy="180" r="9" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="265" y="180" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">+</text>
<line x1="265" y1="120" x2="265" y2="170" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-vit-ink)"/>
<line x1="265" y1="189" x2="265" y2="210" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-vit-ink)"/>
<rect x="150" y="210" width="230" height="40" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="265.0" y="224.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">encoder block × L</text>
<text x="265.0" y="238.4" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">LN → MHA (bidirectional) → + ; LN → MLP → +</text>
<line x1="380" y1="230" x2="430" y2="230" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-vit-ink)"/>
<text x="405.0" y="216" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">197 × D</text>
<rect x="430" y="212" width="90" height="36" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="475.0" y="224.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">take cls</text>
<text x="475.0" y="237.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">or mean-pool</text>
<line x1="520" y1="230" x2="560" y2="230" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-vit-ink)"/>
<text x="540.0" y="216" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">D</text>
<rect x="560" y="212" width="90" height="36" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="605.0" y="230.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">LN + MLP head</text>
<line x1="650" y1="230" x2="690" y2="230" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-vit-ink)"/>
<rect x="690" y="212" width="80" height="36" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="730.0" y="230.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">1000 classes</text>
<text x="640" y="55.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">ViT-B/16: L = 12, D = 768, h = 12</text>
<text x="640" y="69.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">≈ 86M parameters, 197 tokens</text>
<text x="640" y="83.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">ViT-L/16: L = 24, D = 1024, 304M</text>
<text x="640" y="97.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px"></text>
<text x="640" y="111.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">no locality built in: needs</text>
<text x="640" y="125.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">large pretraining or strong augmentation</text>
</svg><figcaption><strong>Figure 13.</strong> The Vision Transformer. Patchify, embed, prepend a learned class token, add learned position embeddings, run L standard pre-LN encoder blocks with bidirectional attention, and classify from the class token (or from the mean of the patch tokens). Nothing in the blocks is image-specific.</figcaption></figure>
```
ViT is the encoder stack applied to patches (Figure 13): patchify, prepend a learned class token, add learned position embeddings, run pre-LN blocks with bidirectional attention, classify from the class token or from the mean of the patch tokens (the paper's appendix shows both work once the learning rate is retuned). ViT-B/16: $L = 12$, $D = 768$, $h = 12$, 197 tokens, 86M parameters; ViT-L/16: 24, 1024, 16, 304M. Nothing in the blocks knows it is looking at an image: 2-D structure enters only through patchifying and through position embeddings, which are reshaped to the grid and interpolated when the resolution changes (DINOv3 randomizes the coordinate box during training to make this robust, and RoPE-based ViTs use coordinates directly). With so little built in, ViT trained on ImageNet-1k alone trailed comparable ResNets and overtook them with ImageNet-21k or JFT-300M pretraining; DeiT closed the ImageNet-1k gap with heavy augmentation and distillation from a CNN teacher; Raghu et al. found that ViTs pretrained on little data fail to develop locally attending heads in their lowest layers.

```{=html}
<figure class="fig" id="fig-swin"><svg viewBox="0 0 780 270" role="img" aria-label="Swin Transformer: windowed attention, shifted windows, and patch merging into a four-stage pyramid" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:780px;margin:0 auto"><defs><marker id="ah-swin-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="102.0" y="26" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px;font-weight:700">block ℓ: windows M = 4</text>
<line x1="30" y1="40" x2="174" y2="40" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="30" y1="40" x2="30" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="30" y1="58" x2="174" y2="58" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="48" y1="40" x2="48" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="30" y1="76" x2="174" y2="76" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="66" y1="40" x2="66" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="30" y1="94" x2="174" y2="94" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="84" y1="40" x2="84" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="30" y1="112" x2="174" y2="112" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="102" y1="40" x2="102" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="30" y1="130" x2="174" y2="130" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="120" y1="40" x2="120" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="30" y1="148" x2="174" y2="148" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="138" y1="40" x2="138" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="30" y1="166" x2="174" y2="166" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="156" y1="40" x2="156" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="30" y1="184" x2="174" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="174" y1="40" x2="174" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="30" y="40" width="72" height="72" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="102" y="40" width="72" height="72" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="30" y="112" width="72" height="72" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="102" y="112" width="72" height="72" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="84" y="94" width="18" height="18" rx="0" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="292.0" y="26" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px;font-weight:700">block ℓ+1: shifted by M/2</text>
<line x1="220" y1="40" x2="364" y2="40" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="220" y1="40" x2="220" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="220" y1="58" x2="364" y2="58" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="238" y1="40" x2="238" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="220" y1="76" x2="364" y2="76" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="256" y1="40" x2="256" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="220" y1="94" x2="364" y2="94" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="274" y1="40" x2="274" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="220" y1="112" x2="364" y2="112" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="292" y1="40" x2="292" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="220" y1="130" x2="364" y2="130" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="310" y1="40" x2="310" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="220" y1="148" x2="364" y2="148" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="328" y1="40" x2="328" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="220" y1="166" x2="364" y2="166" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="346" y1="40" x2="346" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="220" y1="184" x2="364" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<line x1="364" y1="40" x2="364" y2="184" style="stroke:var(--line);stroke-width:0.8;fill:none"/>
<rect x="220" y="40" width="36" height="36" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="256" y="40" width="72" height="36" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="328" y="40" width="36" height="36" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="220" y="76" width="36" height="72" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="256" y="76" width="72" height="72" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="328" y="76" width="36" height="72" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="220" y="148" width="36" height="36" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="256" y="148" width="72" height="36" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="328" y="148" width="36" height="36" rx="0" style="fill:none;stroke:var(--blue);stroke-width:1.8"/>
<rect x="274" y="94" width="18" height="18" rx="0" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="165" y="210" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">the marked token&#x27;s neighbourhood changes:</text>
<text x="165" y="224" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">attention within a window, then across the old borders</text>
<text x="165" y="246" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">cost 4hwC² + 2M²hwC instead of 4hwC² + 2(hw)²C</text>
<text x="590" y="26" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px;font-weight:700">patch merging → pyramid (Swin-T)</text>
<rect x="534.0" y="44" width="112" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="590.0" y="59.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">56² × 96</text>
<text x="670" y="59" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">stage 1, stride 4</text>
<line x1="590" y1="74" x2="590" y2="90" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-swin-ink)"/>
<text x="520" y="84" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">merge 2×2 → 4C → 2C</text>
<rect x="544.0" y="90" width="92" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="590.0" y="105.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">28² × 192</text>
<text x="670" y="105" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">stage 2, stride 8</text>
<line x1="590" y1="120" x2="590" y2="136" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-swin-ink)"/>
<text x="520" y="130" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">merge 2×2 → 4C → 2C</text>
<rect x="552.0" y="136" width="76" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="590.0" y="151.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">14² × 384</text>
<text x="670" y="151" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">stage 3, stride 16</text>
<line x1="590" y1="166" x2="590" y2="182" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-swin-ink)"/>
<text x="520" y="176" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">merge 2×2 → 4C → 2C</text>
<rect x="558.0" y="182" width="64" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="590.0" y="197.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">7² × 768</text>
<text x="670" y="197" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">stage 4, stride 32</text>
<text x="590" y="236" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">like ResNet&#x27;s C2–C5: FPN heads for detection plug in</text>
</svg><figcaption><strong>Figure 14.</strong> Swin's two moves. Left: attention is computed inside non-overlapping M×M windows, so cost is linear in the number of tokens; the next block shifts the window grid by M/2 so tokens near a border can see across it. Right: patch merging halves the grid and doubles the channels between stages, giving a feature pyramid like a CNN's.</figcaption></figure>
```
**Swin and the hierarchical alternative** (Figure 14). Global attention over $hw$ tokens costs $4hwC^2 + 2(hw)^2C$; Swin computes attention inside $M \times M$ windows ($M = 7$), $4hwC^2 + 2M^2hwC$, linear in $hw$, and shifts the window grid by $M/2$ in alternate blocks so tokens near a border see across it. Patch merging halves the grid and doubles the channels between stages, giving a four-stage pyramid at strides 4 to 32 that plugs into FPN-style detection and segmentation heads, which a plain ViT's single coarse map does not. ConvNeXt showed that most of ViT's early advantage over ResNet was recipe and macro design rather than attention: a modernized ResNet (patchify stem, depthwise $7 \times 7$ convolutions, inverted bottlenecks, GELU, LayerNorm) matches Swin at equal size. Self-supervised ViT encoders (DINO, DINOv2, DINOv3, MAE) and what their features carry are on [vision backbones](vision-backbones.html).

### How a condition enters: the design space

```{=html}
<figure class="fig" id="fig-conditioning"><svg viewBox="0 0 780 330" role="img" aria-label="Four conditioning mechanisms: in-context tokens, adaLN modulation, cross-attention, joint attention" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:780px;margin:0 auto"><defs><marker id="ah-conditioning-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--blue)"/></marker><marker id="ah-conditioning-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="100" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">In-context tokens</text>
<text x="100" y="279.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">DiT ablation, PixArt-Σ style,</text>
<text x="100" y="292.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">Flamingo&#x27;s text; cost (N+M)²,</text>
<text x="100" y="304.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">one shared set of weights</text>
<rect x="20" y="60" width="24" height="26" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="48" y="60" width="24" height="26" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="76" y="60" width="24" height="26" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="104" y="60" width="24" height="26" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="132" y="60" width="24" height="26" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="160" y="60" width="24" height="26" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="188" y="60" width="24" height="26" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="76" y="100" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">x tokens</text>
<text x="188" y="100" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">c tokens</text>
<line x1="110" y1="112" x2="110" y2="140" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<rect x="30" y="140" width="160" height="36" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="110.0" y="152.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">self-attention over all</text>
<text x="110.0" y="165.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">then MLP, as usual</text>
<text x="295" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">adaLN / FiLM</text>
<text x="295" y="279.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">DiT, SD3 (timestep, class,</text>
<text x="295" y="292.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">pooled text); cheapest;</text>
<text x="295" y="304.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">one vector for all tokens</text>
<rect x="215" y="56" width="80" height="28" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="255.0" y="70.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">c vector</text>
<line x1="255" y1="84" x2="255" y2="110" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<text x="272" y="89.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">MLP</text>
<rect x="215" y="110" width="80" height="28" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="255.0" y="124.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">γ, β, α</text>
<rect x="310" y="56" width="70" height="28" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="345.0" y="70.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">x tokens</text>
<line x1="345" y1="84" x2="345" y2="140" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<rect x="305" y="140" width="80" height="36" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="345.0" y="152.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:9.5px">LN·(1+γ)+β</text>
<text x="345.0" y="164.65" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">→ sublayer → ·α</text>
<line x1="295" y1="124" x2="304" y2="150" style="stroke:var(--blue);stroke-width:1.3;fill:none;stroke-dasharray:5 4" marker-end="url(#ah-conditioning-blue)"/>
<text x="490" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">Cross-attention</text>
<text x="490" y="279.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">SD U-Net, PixArt-α, Wan,</text>
<text x="490" y="292.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">encoder–decoder; cost N·M;</text>
<text x="490" y="304.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">c is never updated</text>
<rect x="410" y="56" width="80" height="28" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="450.0" y="70.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">x tokens</text>
<rect x="500" y="56" width="80" height="28" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="540.0" y="70.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">c tokens</text>
<line x1="450" y1="84" x2="470" y2="140" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<text x="448" y="116" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">Q</text>
<line x1="540" y1="84" x2="510" y2="140" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<text x="548" y="116" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">K, V</text>
<rect x="410" y="140" width="160" height="36" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="490.0" y="152.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">cross-attention</text>
<text x="490.0" y="165.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">after self-attention, before MLP</text>
<text x="685" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">Joint attention</text>
<text x="685" y="279.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">MMDiT (SD3, FLUX); both</text>
<text x="685" y="292.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">streams updated; cost (N+M)²;</text>
<text x="685" y="304.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">separate weights per stream</text>
<rect x="605" y="56" width="80" height="28" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="645.0" y="70.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">x tokens</text>
<rect x="695" y="56" width="80" height="28" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="735.0" y="70.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">c tokens</text>
<line x1="645" y1="84" x2="655" y2="140" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<text x="628" y="116" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">own QKV</text>
<line x1="735" y1="84" x2="725" y2="140" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<text x="752" y="116" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">own QKV</text>
<rect x="605" y="140" width="170" height="36" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="690.0" y="152.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">one attention over [x; c]</text>
<text x="690.0" y="165.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">split back; own W<tspan dy="2.5" font-size="6.1">O</tspan><tspan dy="-2.5">​</tspan>, own MLP</text>
<line x1="110" y1="176" x2="110" y2="200" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<text x="110" y="212" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">updated x</text>
<line x1="345" y1="176" x2="345" y2="200" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<text x="345" y="212" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">updated x</text>
<line x1="490" y1="176" x2="490" y2="200" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<text x="490" y="212" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">updated x</text>
<line x1="690" y1="176" x2="690" y2="200" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-conditioning-ink)"/>
<text x="690" y="212" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">updated x</text>
<text x="690" y="226" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">(and updated c)</text>
</svg><figcaption><strong>Figure 15.</strong> Four ways to inject a condition into a transformer, drawn as what changes in one block. In-context: the condition becomes extra tokens in the same sequence. adaLN / FiLM: a vector modulates every token identically through shift, scale and gate. Cross-attention: tokens query a separate condition sequence. Joint attention: both sequences are attended together with separate weights per stream.</figcaption></figure>
```
Every conditional transformer has to get the condition in somewhere, and there are four places (Figure 15). The choice decides cost, parameter sharing and which direction information flows.

| Mechanism | How | Cost added | Where used |
|---|---|---|---|
| In-context tokens | append the condition's tokens to the sequence; ordinary self-attention sees them | $(N + M)^2$ scores, shared weights | DiT's ablation, Flamingo's text tokens, PixArt-Σ-style text in sequence, unified any-to-any models |
| adaLN / FiLM | a conditioning vector $c$ is mapped by an MLP to a shift, scale and gate that modulate every token identically | negligible | DiT (timestep + class), SD3 and FLUX (timestep + pooled text), the ResBlocks of diffusion U-Nets (timestep) |
| Cross-attention | tokens query the condition sequence; $K$, $V$ from the condition | $NMD$ per layer | SD U-Net, PixArt-α, Wan, encoder–decoder translation |
| Joint attention | both sequences attended together, each with its own projections and MLP | $(N + M)^2$ scores, separate weights | MMDiT in SD3 and FLUX |

Two rules of thumb fall out. A global signal with no spatial structure (timestep, class, a pooled embedding) belongs in adaLN: it is cheapest and the DiT paper found it best for class conditioning, with cross-attention costing 15% more FLOPs for a worse FID. A sequence with internal structure (text tokens) needs attention; cross-attention leaves the text untouched and costs $NM$, joint attention lets the text be updated by the image at every layer and costs $(N+M)^2$, and SD3's comparison found joint attention better than both cross-attention and a shared-weight in-context DiT.

### DiT

```{=html}
<figure class="fig" id="fig-dit"><svg viewBox="0 0 780 470" role="img" aria-label="DiT: patchified noisy latent through adaLN-Zero blocks to an unpatchified noise prediction; block detail with shift, scale and gate from the conditioning vector" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:780px;margin:0 auto"><defs><marker id="ah-dit-gold" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--gold)"/></marker><marker id="ah-dit-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<rect x="20" y="40" width="110" height="44" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="75.0" y="55.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">noisy latent z<tspan dy="3.3" font-size="7.9">t</tspan><tspan dy="-3.3">​</tspan></text>
<text x="75.0" y="69.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">32 × 32 × 4</text>
<line x1="130" y1="62" x2="170" y2="62" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<text x="150.0" y="26" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">patchify p = 2</text>
<rect x="170" y="40" width="110" height="44" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="225.0" y="55.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">256 tokens × D</text>
<text x="225.0" y="69.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">+ 2D sin-cos pos</text>
<line x1="280" y1="62" x2="320" y2="62" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<rect x="320" y="34" width="150" height="56" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="395.0" y="55.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">DiT block × L</text>
<text x="395.0" y="70.4" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">(28 for XL, D = 1152)</text>
<line x1="470" y1="62" x2="510" y2="62" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<rect x="510" y="40" width="120" height="44" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="570.0" y="55.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">LN + linear</text>
<text x="570.0" y="69.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">D → p·p·2c per token</text>
<line x1="630" y1="62" x2="668" y2="62" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<text x="649.0" y="26" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">unpatchify</text>
<rect x="668" y="40" width="100" height="44" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="718.0" y="55.825" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">ε̂ (and Σ)</text>
<text x="718.0" y="69.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">32 × 32 × 4 (× 2)</text>
<rect x="300" y="120" width="190" height="36" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="395.0" y="132.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">c = emb(t) + emb(y)</text>
<text x="395.0" y="145.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">timestep + class (or pooled text)</text>
<line x1="395" y1="120" x2="395" y2="86" style="stroke:var(--gold);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-gold)"/>
<text x="410" y="104" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">to every block</text>
<text x="250" y="138" text-anchor="end" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">bidirectional attention; no KV cache across steps</text>
<line x1="20" y1="176" x2="760" y2="176" style="stroke:var(--line);stroke-width:1;fill:none"/>
<text x="20" y="192" text-anchor="start" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">one DiT block (adaLN-Zero)</text>
<text x="70" y="214" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">x</text>
<line x1="70" y1="222" x2="70" y2="236" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<rect x="120" y="236" width="48" height="34" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="144.0" y="253.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">LN</text>
<rect x="178" y="236" width="96" height="34" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="226.0" y="253.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">(1+γ<tspan dy="3.1" font-size="7.6">1</tspan><tspan dy="-3.1">​</tspan>) ⊙ · + β<tspan dy="3.1" font-size="7.6">1</tspan><tspan dy="-3.1">​</tspan></text>
<rect x="284" y="236" width="110" height="34" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="339.0" y="253.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">multi-head attn</text>
<rect x="404" y="236" width="50" height="34" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="429.0" y="253.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">⊙ α<tspan dy="3.3" font-size="7.9">1</tspan><tspan dy="-3.3">​</tspan></text>
<line x1="168" y1="253.0" x2="178" y2="253.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<line x1="274" y1="253.0" x2="284" y2="253.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<line x1="394" y1="253.0" x2="404" y2="253.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<line x1="70" y1="253.0" x2="120" y2="253.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<polyline points="454,253.0 480,253.0 480,292 80,292" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-dit-ink)"/>
<line x1="70" y1="236" x2="70" y2="282" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<circle cx="70" cy="292" r="10" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="70" y="292" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px">+</text>
<rect x="120" y="314" width="48" height="34" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="144.0" y="331.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">LN</text>
<rect x="178" y="314" width="96" height="34" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="226.0" y="331.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">(1+γ<tspan dy="3.1" font-size="7.6">2</tspan><tspan dy="-3.1">​</tspan>) ⊙ · + β<tspan dy="3.1" font-size="7.6">2</tspan><tspan dy="-3.1">​</tspan></text>
<rect x="284" y="314" width="110" height="34" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="339.0" y="331.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">MLP (4D)</text>
<rect x="404" y="314" width="50" height="34" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="429.0" y="331.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">⊙ α<tspan dy="3.3" font-size="7.9">2</tspan><tspan dy="-3.3">​</tspan></text>
<line x1="168" y1="331.0" x2="178" y2="331.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<line x1="274" y1="331.0" x2="284" y2="331.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<line x1="394" y1="331.0" x2="404" y2="331.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<line x1="70" y1="331.0" x2="120" y2="331.0" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<polyline points="454,331.0 480,331.0 480,370 80,370" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-dit-ink)"/>
<line x1="70" y1="302" x2="70" y2="360" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<circle cx="70" cy="370" r="10" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="70" y="370" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px">+</text>
<line x1="70" y1="380" x2="70" y2="396" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<text x="70" y="408" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">x&#x27;</text>
<rect x="560" y="214" width="150" height="34" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="635.0" y="225.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">c</text>
<text x="635.0" y="239.4" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">D-vector, same for all tokens</text>
<line x1="635" y1="248" x2="635" y2="272" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-dit-ink)"/>
<text x="700" y="260" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">SiLU → Linear (D → 6D)</text>
<rect x="560" y="272" width="150" height="50" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="635.0" y="290.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">γ<tspan dy="3.3" font-size="7.9">1</tspan><tspan dy="-3.3">​</tspan>, β<tspan dy="3.3" font-size="7.9">1</tspan><tspan dy="-3.3">​</tspan>, α<tspan dy="3.3" font-size="7.9">1</tspan><tspan dy="-3.3">​</tspan></text>
<text x="635.0" y="304.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">γ<tspan dy="3.3" font-size="7.9">2</tspan><tspan dy="-3.3">​</tspan>, β<tspan dy="3.3" font-size="7.9">2</tspan><tspan dy="-3.3">​</tspan>, α<tspan dy="3.3" font-size="7.9">2</tspan><tspan dy="-3.3">​</tspan></text>
<polyline points="560,290 520,290 520,257.0 456,257.0" style="stroke:var(--gold);stroke-width:1.3;fill:none;stroke-linejoin:round;stroke-dasharray:5 4"/>
<polyline points="520,290 520,335.0 456,335.0" style="stroke:var(--gold);stroke-width:1.3;fill:none;stroke-linejoin:round;stroke-dasharray:5 4"/>
<text x="635" y="358.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">zero-initialized ⇒ γ = β = α = 0</text>
<text x="635" y="372.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">⇒ x&#x27; = x: block starts as identity;</text>
<text x="635" y="386.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">gates α learn first, then the branches</text>
<text x="635" y="423.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px;font-style:italic">vs. cross-attention to c: +15% FLOPs,</text>
<text x="635" y="436.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px;font-style:italic">worse FID in the DiT paper</text>
</svg><figcaption><strong>Figure 16.</strong> The Diffusion Transformer. Top: the whole model, a ViT-style encoder on patchified VAE latents whose output is unpatchified back into a latent-shaped prediction. Bottom: one DiT block with adaLN-Zero. The conditioning vector c (timestep + class, or pooled text) is mapped by a zero-initialized linear layer to a shift, scale and gate for each sub-layer; at initialization all six are zero, so every block is the identity and the network starts as a no-op.</figcaption></figure>
```
DiT (Figure 16) is a ViT on VAE latents: the noisy latent is patchified ($256$ tokens at $p = 2$ for $256^2$ images) with fixed 2-D sine-cosine positions, run through $L$ blocks with bidirectional attention, and decoded by a final LN and linear layer to $p \times p \times 2C$ numbers per token (noise and a diagonal covariance), then unpatchified. The timestep embedding and the class embedding are summed into one vector $c$, and each block regresses six $D$-vectors from it, $(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) = \operatorname{Linear}(\operatorname{SiLU}(c))$, used as

$$
\begin{aligned}
h &= x + \alpha_1 \odot \operatorname{MHA}\big((1 + \gamma_1) \odot \mathrm{LN}(x) + \beta_1\big), \\
x' &= h + \alpha_2 \odot \operatorname{MLP}\big((1 + \gamma_2) \odot \mathrm{LN}(h) + \beta_2\big),
\end{aligned}
$$

with LayerNorms that carry no affine parameters of their own. The same $\gamma, \beta, \alpha$ apply to every token: this is how a global signal changes the behavior of the whole network, and the scale is written $1 + \gamma$ so that zero means unchanged.

**adaLN-Zero.** The final linear layer of each block's modulation MLP is initialized to zero, so at initialization $\gamma = \beta = \alpha = 0$, $h = x$, $x' = x$: every block is the identity and the whole network is a no-op (its output layer is zero-initialized too). Starting a deep residual network at the identity is the same trick as zero-initializing the last BatchNorm scale in ResNet blocks and the last convolution of diffusion U-Net blocks, ControlNet's zero convolutions and LoRA's $B = 0$. What trains first: with $\alpha = 0$ the gradient into the attention and MLP weights is multiplied by $\alpha$ and is zero, but $\partial\mathcal L/\partial \alpha_1 = \partial\mathcal L/\partial h \odot \operatorname{MHA}(\cdot)$ is nonzero because the randomly initialized branch has a nonzero output, so the modulation layer moves first, the gates open, and then the branches learn. DiT-XL/2: 28 layers, $D = 1152$, 16 heads, 675M parameters, 118.6 Gflops, FID 2.27 on class-conditional ImageNet $256^2$ with guidance 1.5; FID improved monotonically with Gflops whether spent on width, depth or smaller patches. No causal mask and no KV cache: all tokens are denoised together and every token changes at every sampling step.

### The U-Net with cross-attention

```{=html}
<figure class="fig" id="fig-unet"><svg viewBox="0 0 780 400" role="img" aria-label="Latent-diffusion U-Net: encoder and decoder levels with skip connections; zoom on a level with a ResBlock conditioned on t and a transformer block with self-attention, cross-attention to text and an MLP" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:780px;margin:0 auto"><defs><marker id="ah-unet-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--blue)"/></marker><marker id="ah-unet-gold" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--gold)"/></marker><marker id="ah-unet-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker><marker id="ah-unet-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>
<rect x="30" y="30" width="130" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="95.0" y="43.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">64² × 320</text>
<text x="95.0" y="56.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">ResBlock + Transformer</text>
<rect x="420" y="30" width="130" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="485.0" y="43.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">64² × 320</text>
<text x="485.0" y="56.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">ResBlock + Transformer</text>
<line x1="162" y1="49.0" x2="418" y2="49.0" style="stroke:var(--muted);stroke-width:1.3;fill:none;stroke-dasharray:5 4" marker-end="url(#ah-unet-muted)"/>
<text x="290.0" y="41.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">skip: concat</text>
<line x1="95.0" y1="68" x2="135.0" y2="84" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<line x1="445.0" y1="84" x2="485.0" y2="68" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<rect x="70" y="84" width="130" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="135.0" y="97.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">32² × 640</text>
<text x="135.0" y="110.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">ResBlock + Transformer</text>
<rect x="380" y="84" width="130" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="445.0" y="97.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">32² × 640</text>
<text x="445.0" y="110.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">ResBlock + Transformer</text>
<line x1="202" y1="103.0" x2="378" y2="103.0" style="stroke:var(--muted);stroke-width:1.3;fill:none;stroke-dasharray:5 4" marker-end="url(#ah-unet-muted)"/>
<line x1="135.0" y1="122" x2="175.0" y2="138" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<line x1="405.0" y1="138" x2="445.0" y2="122" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<rect x="110" y="138" width="130" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="175.0" y="151.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">16² × 1280</text>
<text x="175.0" y="164.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">ResBlock + Transformer</text>
<rect x="340" y="138" width="130" height="38" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="405.0" y="151.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">16² × 1280</text>
<text x="405.0" y="164.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">ResBlock + Transformer</text>
<line x1="242" y1="157.0" x2="338" y2="157.0" style="stroke:var(--muted);stroke-width:1.3;fill:none;stroke-dasharray:5 4" marker-end="url(#ah-unet-muted)"/>
<rect x="235.0" y="194" width="120" height="32" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="295.0" y="210.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">mid 8² × 1280</text>
<line x1="175.0" y1="176" x2="237.0" y2="210" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<line x1="355.0" y1="210" x2="405.0" y2="176" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<text x="95.0" y="18" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">z<tspan dy="3.1" font-size="7.6">t</tspan><tspan dy="-3.1">​</tspan> in (64 × 64 × 4)</text>
<text x="485.0" y="18" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">ε̂ out (64 × 64 × 4)</text>
<rect x="600" y="40" width="160" height="34" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="680.0" y="57.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">timestep t → emb(t)</text>
<rect x="600" y="92" width="160" height="34" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="680.0" y="103.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">text → CLIP encoder</text>
<text x="680.0" y="116.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">77 tokens × 768</text>
<text x="680" y="142" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">into every level ↓</text>
<line x1="20" y1="236" x2="760" y2="236" style="stroke:var(--line);stroke-width:1;fill:none"/>
<text x="20" y="252" text-anchor="start" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">one level, zoomed</text>
<rect x="20" y="272" width="170" height="60" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="105.0" y="296.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11.5px">ResBlock</text>
<text x="105.0" y="310.05" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">conv → + Linear(emb(t)) → conv</text>
<polyline points="680,74 770,74 770,228 105,228 105,270" style="stroke:var(--gold);stroke-width:1.3;fill:none;stroke-linejoin:round;stroke-dasharray:5 4" marker-end="url(#ah-unet-gold)"/>
<line x1="190" y1="302" x2="230" y2="302" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<text x="210.0" y="288" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">flatten</text>
<text x="210" y="316" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">(HW) × C</text>
<rect x="230" y="260" width="420" height="84" rx="8" style="fill:none;stroke:var(--muted);stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="240" y="270" text-anchor="start" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px;font-style:italic">transformer block (&quot;spatial transformer&quot;)</text>
<rect x="240" y="280" width="110" height="44" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="295.0" y="296.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">self-attention</text>
<text x="295.0" y="309.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">image ↔ image</text>
<line x1="350" y1="302" x2="372" y2="302" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<rect x="372" y="280" width="130" height="44" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="437.0" y="296.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">cross-attention</text>
<text x="437.0" y="309.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">Q: image, K,V: text</text>
<line x1="502" y1="302" x2="524" y2="302" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<rect x="524" y="280" width="110" height="44" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="579.0" y="296.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">MLP (GEGLU)</text>
<text x="579.0" y="309.7" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">each with residual</text>
<polyline points="680,126 700,126 700,232 437,232 437,278" style="stroke:var(--blue);stroke-width:1.3;fill:none;stroke-linejoin:round;stroke-dasharray:5 4" marker-end="url(#ah-unet-blue)"/>
<line x1="650" y1="302" x2="690" y2="302" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-unet-ink)"/>
<text x="670.0" y="288" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">reshape</text>
<rect x="690" y="280" width="70" height="44" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="725.0" y="302.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">C × H × W</text>
<text x="390" y="365.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">down ×2 between encoder levels (strided conv), up ×2 in the decoder; skips concatenate same-resolution features</text>
<text x="390" y="379.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">at the 64² level of SD v1: 4096 queries × 77 keys per head (not 4096²); 8 heads of width 40</text>
</svg><figcaption><strong>Figure 17.</strong> Stable Diffusion's U-Net backbone and where the two conditions enter. The timestep goes into every residual block as an added vector (FiLM-style); the text goes into every attention block as keys and values for cross-attention, with the image positions as queries. Skip connections concatenate encoder features into the decoder at the same resolution.</figcaption></figure>
```
Latent diffusion's backbone (Figure 17) is a convolutional U-Net with transformer blocks inside it. The encoder halves the resolution and raises the channels level by level ($64^2 \times 320 \to 32^2 \times 640 \to 16^2 \times 1280$ in SD v1, plus a mid block at $8^2$), the decoder reverses it, and skip connections concatenate each encoder level's features onto the decoder input at the same resolution, which is what keeps fine detail. Each level has ResBlocks and a transformer block. The timestep enters every ResBlock as an embedding passed through a linear layer and added to the feature map, a FiLM-style modulation, and the text enters every transformer block: the feature map is flattened to $(HW) \times C$ tokens, which pass through self-attention over image positions, cross-attention whose keys and values are the 77 CLIP text tokens, and a GEGLU MLP, each with a residual, and are reshaped back. At the $64^2$ level that is 4096 queries of width 320 against 77 keys projected from width 768, with 8 heads of width 40: $4096 \times 77$ weights per head, so cross-attention is cheap next to the $4096^2$ self-attention at the same level. Each cross-attention row says which words an image location reads, which is what Prompt-to-Prompt edits. Classifier-free guidance reruns the network with the empty prompt's embeddings. SDXL enlarges the transformer blocks at the lower-resolution levels and adds a second text encoder; SD3 and FLUX drop the U-Net for MMDiT.

### MMDiT

```{=html}
<figure class="fig" id="fig-mmdit"><svg viewBox="0 0 780 400" role="img" aria-label="MMDiT block: two token streams with separate weights joined by one attention over the concatenated sequence" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:780px;margin:0 auto"><defs><marker id="ah-mmdit-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="130" y="22" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">text stream  c</text>
<text x="130" y="38" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">M tokens (CLIP + T5 in SD3)</text>
<rect x="60" y="52" width="140" height="30" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="130.0" y="67.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">LN + adaLN mod</text>
<line x1="130" y1="82" x2="130" y2="100" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-mmdit-ink)"/>
<rect x="60" y="100" width="140" height="30" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="130.0" y="115.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">own W<tspan dy="3.1" font-size="7.6">Q</tspan><tspan dy="-3.1">​</tspan>, W<tspan dy="3.1" font-size="7.6">K</tspan><tspan dy="-3.1">​</tspan>, W<tspan dy="3.1" font-size="7.6">V</tspan><tspan dy="-3.1">​</tspan></text>
<line x1="130" y1="130" x2="130" y2="160" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-mmdit-ink)"/>
<line x1="130" y1="236" x2="130" y2="256" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-mmdit-ink)"/>
<rect x="60" y="256" width="140" height="30" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="130.0" y="271.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">own W<tspan dy="3.1" font-size="7.6">O</tspan><tspan dy="-3.1">​</tspan>, + residual</text>
<line x1="130" y1="286" x2="130" y2="304" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-mmdit-ink)"/>
<rect x="60" y="304" width="140" height="30" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="130.0" y="319.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">LN + mod → own MLP → +</text>
<text x="390" y="22" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">image stream  x</text>
<text x="390" y="38" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">N latent patches</text>
<rect x="320" y="52" width="140" height="30" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="390.0" y="67.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">LN + adaLN mod</text>
<line x1="390" y1="82" x2="390" y2="100" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-mmdit-ink)"/>
<rect x="320" y="100" width="140" height="30" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="390.0" y="115.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">own W<tspan dy="3.1" font-size="7.6">Q</tspan><tspan dy="-3.1">​</tspan>, W<tspan dy="3.1" font-size="7.6">K</tspan><tspan dy="-3.1">​</tspan>, W<tspan dy="3.1" font-size="7.6">V</tspan><tspan dy="-3.1">​</tspan></text>
<line x1="390" y1="130" x2="390" y2="160" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-mmdit-ink)"/>
<line x1="390" y1="236" x2="390" y2="256" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-mmdit-ink)"/>
<rect x="320" y="256" width="140" height="30" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="390.0" y="271.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">own W<tspan dy="3.1" font-size="7.6">O</tspan><tspan dy="-3.1">​</tspan>, + residual</text>
<line x1="390" y1="286" x2="390" y2="304" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-mmdit-ink)"/>
<rect x="320" y="304" width="140" height="30" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="390.0" y="319.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">LN + mod → own MLP → +</text>
<rect x="40" y="160" width="400" height="76" rx="6" style="fill:#ffffff;stroke:var(--ink);stroke-width:1.2"/>
<text x="240" y="176" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px;font-weight:700">joint attention over [c ; x]  (M + N tokens)</text>
<text x="240" y="196" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px">softmax( [Q<tspan dy="3.3" font-size="7.9">c</tspan><tspan dy="-3.3">​</tspan>; Q<tspan dy="3.3" font-size="7.9">x</tspan><tspan dy="-3.3">​</tspan>] [K<tspan dy="3.3" font-size="7.9">c</tspan><tspan dy="-3.3">​</tspan>; K<tspan dy="3.3" font-size="7.9">x</tspan><tspan dy="-3.3">​</tspan>]ᵀ / √d ) [V<tspan dy="3.3" font-size="7.9">c</tspan><tspan dy="-3.3">​</tspan>; V<tspan dy="3.3" font-size="7.9">x</tspan><tspan dy="-3.3">​</tspan>]</text>
<text x="240" y="218" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">split the output back into M text rows and N image rows</text>
<rect x="540" y="100" width="170" height="170" rx="2" style="fill:#ffffff;stroke:var(--ink);stroke-width:1.2"/>
<rect x="540" y="100" width="51" height="51" rx="0" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="591" y="100" width="119" height="51" rx="0" style="fill:#eceff4;stroke:var(--muted);stroke-width:1.2"/>
<rect x="540" y="151" width="51" height="119" rx="0" style="fill:#eceff4;stroke:var(--muted);stroke-width:1.2"/>
<rect x="591" y="151" width="119" height="119" rx="0" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="565.5" y="125.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">c→c</text>
<text x="650.5" y="125.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">text reads image</text>
<text x="565.5" y="198.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">image</text>
<text x="565.5" y="210.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">reads</text>
<text x="565.5" y="222.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">text</text>
<text x="650.5" y="210.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">x→x (self-attention)</text>
<text x="625.0" y="88" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10.5px">(M + N) × (M + N) attention weights</text>
<text x="625.0" y="284" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">cross-attention keeps only the</text>
<text x="625.0" y="298" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">&quot;image reads text&quot; block</text>
<text x="240" y="357.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">modulation (adaLN) from timestep + pooled text; QK-RMSNorm for stability</text>
<text x="240" y="371.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">FLUX: double-stream MMDiT blocks, then single-stream blocks with shared weights</text>
</svg><figcaption><strong>Figure 18.</strong> The MMDiT block of Stable Diffusion 3 and FLUX. Text and image tokens keep separate weights (their own modulation, Q/K/V, output projection and MLP), but their queries, keys and values are concatenated into one joint attention over all M + N tokens, so each stream can read the other at every layer. The weight matrix has four blocks: text→text, text→image, image→text and image→image.</figcaption></figure>
```
SD3's MMDiT (Figure 18) keeps text and image as two token streams with separate weights, their own adaLN modulation, $Q$/$K$/$V$ and output projections, and MLPs, but concatenates their queries, keys and values into one attention over all $M + N$ tokens:

$$
\begin{pmatrix} O_c \\ O_x \end{pmatrix} = \operatorname{Attn}\!\left(\begin{pmatrix} Q_c \\ Q_x \end{pmatrix}, \begin{pmatrix} K_c \\ K_x \end{pmatrix}, \begin{pmatrix} V_c \\ V_x \end{pmatrix}\right),
$$

then splits the output back into each stream's own $W_O$ and MLP. The $(M + N) \times (M + N)$ weight matrix has four blocks: text reading text, text reading image, image reading text (the one block cross-attention has) and image reading image (self-attention). Unlike cross-attention, the text representation is updated by the image at every layer, at $(N + M)^2$ scores instead of $N^2 + NM$. SD3's text sequence is 77 CLIP tokens zero-padded to width 4096 followed by 77 T5-XXL tokens; pooled CLIP vectors and the timestep enter through adaLN; RMSNorm on queries and keys keeps bf16 training stable at high resolution. FLUX.1 uses double-stream MMDiT blocks followed by single-stream blocks with shared weights, and 2-D RoPE on image-token coordinates.

### Attaching new inputs to a frozen model

```{=html}
<figure class="fig" id="fig-gatedxattn"><svg viewBox="0 0 780 300" role="img" aria-label="Gated cross-attention inserted into a frozen stack, and a Perceiver resampler that maps a variable-length context to a fixed number of tokens" style="font-family:var(--sans);width:100%;height:auto;display:block;max-width:780px;margin:0 auto"><defs><marker id="ah-gatedxattn-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--blue)"/></marker><marker id="ah-gatedxattn-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs>
<text x="190" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">Gated cross-attention (Flamingo-style)</text>
<rect x="20" y="40" width="80" height="28" rx="6" style="fill:#eceff4;stroke:var(--muted);stroke-width:1.2"/>
<text x="60.0" y="54.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">x (frozen)</text>
<line x1="60" y1="68" x2="60" y2="100" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<rect x="110" y="86" width="44" height="30" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="132.0" y="101.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">LN</text>
<rect x="164" y="86" width="120" height="30" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="224.0" y="101.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">cross-attention</text>
<rect x="294" y="86" width="70" height="30" rx="6" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="329.0" y="95.475" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">× tanh(α)</text>
<text x="329.0" y="108.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:8.5px">α init 0</text>
<line x1="60" y1="101" x2="110" y2="101" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-gatedxattn-ink)"/>
<line x1="154" y1="101" x2="164" y2="101" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-gatedxattn-ink)"/>
<line x1="284" y1="101" x2="294" y2="101" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-gatedxattn-ink)"/>
<polyline points="364,101 380,101 380,140 70,140" style="stroke:var(--ink);stroke-width:1.3;fill:none;stroke-linejoin:round" marker-end="url(#ah-gatedxattn-ink)"/>
<line x1="60" y1="100" x2="60" y2="130" style="stroke:var(--ink);stroke-width:1.3;fill:none"/>
<circle cx="60" cy="140" r="10" style="fill:#ffffff;stroke:var(--blue);stroke-width:1.2"/>
<text x="60" y="140" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:13px">+</text>
<rect x="170" y="150" width="110" height="26" rx="6" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="225.0" y="163.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">context K, V</text>
<line x1="224" y1="150" x2="224" y2="118" style="stroke:var(--blue);stroke-width:1.3;fill:none;stroke-dasharray:5 4" marker-end="url(#ah-gatedxattn-blue)"/>
<line x1="60" y1="150" x2="60" y2="184" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-gatedxattn-ink)"/>
<rect x="20" y="184" width="80" height="28" rx="6" style="fill:#eceff4;stroke:var(--muted);stroke-width:1.2"/>
<text x="60.0" y="198.0" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">frozen block</text>
<line x1="60" y1="212" x2="60" y2="236" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-gatedxattn-ink)"/>
<text x="60" y="248" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:11px;font-weight:700">x&#x27;</text>
<text x="250" y="200.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">only the new layers train; at α = 0 the output</text>
<text x="250" y="213.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">equals the frozen model&#x27;s, so training starts</text>
<text x="250" y="226.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">from the pretrained behaviour (same idea as</text>
<text x="250" y="239.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">adaLN-Zero, ControlNet&#x27;s zero convs, LoRA&#x27;s B = 0)</text>
<text x="590" y="20" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12.5px;font-weight:700">Perceiver resampler</text>
<rect x="430" y="50" width="20" height="24" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="454" y="50" width="20" height="24" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="478" y="50" width="20" height="24" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="502" y="50" width="20" height="24" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="526" y="50" width="20" height="24" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="550" y="50" width="20" height="24" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<rect x="574" y="50" width="20" height="24" rx="3" style="fill:#e7eef7;stroke:var(--blue);stroke-width:1.2"/>
<text x="608" y="62" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:12px">…</text>
<text x="520" y="86" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">context, any length M (encoder states, frames)</text>
<rect x="640" y="50" width="20" height="24" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="664" y="50" width="20" height="24" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="688" y="50" width="20" height="24" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="712" y="50" width="20" height="24" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="686" y="86" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">R learned queries</text>
<rect x="470" y="120" width="240" height="40" rx="6" style="fill:var(--panel);stroke:var(--blue);stroke-width:1.2"/>
<text x="590.0" y="134.15" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10.5px">cross-attention + MLP (× few)</text>
<text x="590.0" y="147.35" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9px">queries attend to [context ; queries]</text>
<line x1="514" y1="92" x2="540" y2="118" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-gatedxattn-ink)"/>
<text x="510" y="110" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">K, V</text>
<line x1="686" y1="92" x2="650" y2="118" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-gatedxattn-ink)"/>
<text x="680" y="110" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:9.5px">Q</text>
<line x1="590" y1="160" x2="590" y2="186" style="stroke:var(--ink);stroke-width:1.3;fill:none" marker-end="url(#ah-gatedxattn-ink)"/>
<rect x="542" y="186" width="20" height="24" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="566" y="186" width="20" height="24" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="590" y="186" width="20" height="24" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<rect x="614" y="186" width="20" height="24" rx="3" style="fill:#fff3d6;stroke:var(--gold);stroke-width:1.2"/>
<text x="590" y="224" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">R output tokens, fixed shape whatever M is</text>
<text x="590" y="251.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">cost R·M, not M²; used in Flamingo, BLIP-2 (Q-Former),</text>
<text x="590" y="264.5" text-anchor="middle" dominant-baseline="middle" style="fill:var(--muted);font-size:10px">TET&#x27;s bridge; the resampler then feeds a backbone</text>
</svg><figcaption><strong>Figure 19.</strong> Two tools for attaching a frozen model to new inputs. Left: a gated cross-attention layer (Flamingo) inserted into a frozen stack, multiplied by tanh of a gate initialized at zero, so the pretrained model is unchanged at the start of training. Right: a Perceiver resampler compresses a variable-length context into a fixed number of tokens by letting learned queries cross-attend to it.</figcaption></figure>
```
Two devices recur whenever a pretrained transformer must accept something it was not trained on (Figure 19). A gated cross-attention layer (Flamingo) is inserted between frozen blocks: $x \leftarrow x + \tanh(\alpha)\cdot\operatorname{CrossAttn}(\mathrm{LN}(x), C)$ with $\alpha$ initialized to zero, so the frozen model's function is unchanged at step zero and only the new layers train; this is the same zero-initialization logic as adaLN-Zero. A Perceiver resampler turns a variable-length context into a fixed number $R$ of tokens: $R$ learned queries cross-attend to the context (and to themselves) through a few layers and the $R$ outputs are what the backbone consumes; the cost is $RM$ rather than $M^2$, and the backbone always sees the same shape, which is what makes BLIP-2's Q-Former and Flamingo's vision input work. The same pattern appears in encoder-swapping and adapter work: keep the backbone frozen, train a small bridge, and decide carefully what the bridge is asked to match.

### Side by side

| | ViT-B/16 | DiT-XL/2 | SD v1 U-Net | SD3 MMDiT |
|---|---|---|---|---|
| Input tokens | 196 patches + cls, $P = 16$ | 256 latent patches, $p = 2$ | $64^2$ latent map, flattened per level | $N$ latent patches + 154 text tokens |
| Attention | global, bidirectional | global, bidirectional | self + cross inside a conv U-Net | joint over both streams |
| Condition | none (class via head) | timestep + class via adaLN-Zero | timestep via FiLM in ResBlocks, text via cross-attention | timestep + pooled text via adaLN, text tokens via joint attention |
| Positions | learned, interpolated | fixed 2-D sin-cos | convolution carries position | 2-D positions (RoPE in FLUX) |
| Size | 86M | 675M, 118.6 Gflops | 860M | 2B to 8B |

::: {.qa asked="Xiaomi multimodal, image-gen intern (Nowcoder 2024)"}
**Describe the ViT architecture and how DiT adapts it for diffusion.**

ViT: $P \times P$ patches embedded linearly (a convolution with kernel and stride $P$), a class token, learned position embeddings, pre-LN blocks with bidirectional attention, classification from the class token or the mean of patch tokens. DiT runs the same stack on a VAE latent, $32 \times 32 \times 4$ with patch 2, so 256 tokens with fixed 2-D sine-cosine positions; timestep and class embeddings are summed into $c$, and each block regresses a scale, shift and gate per sub-layer from $c$ (adaLN-Zero), zero-initialized so every block starts as the identity; a linear decoder maps each token to $p \times p \times 2C$ and unpatchify restores the latent. FID improved steadily with Gflops. Text-conditioned variants add cross-attention (PixArt-α) or joint attention over text and image tokens (MMDiT).
:::

::: {.qa asked="Google intern, Ant, Alibaba Amap (3 reports)"}
**Drawbacks of ViTs compared with CNNs, and vice versa?**

CNNs build in locality and translation equivariance, so they learn from less data, scale linearly with pixel count and give multi-scale feature pyramids for free. Plain ViTs need more data, augmentation or distillation; attention is quadratic in token count, which limits resolution; they produce one coarse-scale map, a handicap for detection and segmentation; learned positions must be interpolated for new resolutions. In return ViTs have a global receptive field from the first layer, keep improving with data and compute, and share the token interface with text, which is what makes DiT and MMDiT possible. Swin and ConvNeXt sit between.
:::

::: qa
**Cross-attention or joint attention for text conditioning?**

Cross-attention (SD U-Net, PixArt-α) takes queries from image tokens and keys and values from a fixed text sequence: text informs the image, the text never changes, cost $O(NM)$. Joint attention (MMDiT) concatenates the streams, each with its own projections, and runs one attention over both, so the text representation is updated layer by layer, at $(N + M)^2$ cost. SD3 found the joint design better. A global vector such as the timestep goes through adaLN in either case.
:::

::: qa
**At initialization, which parameters of a DiT block receive gradient?**

With adaLN-Zero, $\alpha = \gamma = \beta = 0$, so the block is the identity and the gradient to the attention and MLP weights, scaled by $\alpha$, is zero. The modulation layer does get gradient, since $\partial\mathcal L/\partial\alpha$ is the upstream gradient times the nonzero branch output; after one step $\alpha \ne 0$ and the branches begin to learn.
:::

::: key
**How to say it in an interview.** "DiT is a ViT on VAE latents: patchify a $32 \times 32 \times 4$ latent with patch size 2 into 256 tokens, run bidirectional pre-LN blocks, decode each token back to a patch of noise. Timestep and class enter through adaLN-Zero, a zero-initialized layer mapping the conditioning vector to a shift, scale and gate per sub-layer, so every block starts as the identity. For text, the SD U-Net uses cross-attention with image queries and text keys and values; SD3's MMDiT concatenates text and image tokens, with separate weights per stream, into one joint attention so they update each other."
:::

## Mixture of experts

An MoE layer replaces a block's MLP with $E$ expert MLPs and a router $p(x) = \operatorname{softmax}(xW_r)$; each token goes to its top-$k$ experts, $y = \sum_{i \in \text{TopK}} g_i(x)\,\mathrm{MLP}_i(x)$, with $g_i$ the router probability (Switch, $k = 1$) or the renormalized top-$k$ probabilities (Mixtral, $k = 2$). Compute per token is $k$ MLPs regardless of $E$. Without a constraint the router collapses onto the experts chosen early, so Switch adds $\mathcal L_{\text{aux}} = \alpha E \sum_i f_i P_i$ with $f_i$ the fraction of tokens dispatched to expert $i$ (not differentiable) and $P_i$ the mean router probability for it (differentiable), which equals $\alpha$ under uniform routing and pushes probability away hardest from the busiest experts; $\alpha = 10^{-2}$. Each expert also gets a fixed capacity, tokens per batch over $E$ times a capacity factor of 1 to 1.25, and overflow tokens skip the layer through the residual path. Mixtral 8x7B: 32 layers, $D = 4096$, eight SwiGLU experts with $d_{\text{ff}} = 14{,}336$ (176M weights each), top-2, GQA with 8 KV heads: 46.7B total parameters, 12.9B active per token; FLOPs follow the active count, memory the total, and at small decoding batches the bytes read per step approach the total because different tokens hit different experts. Experts are spread across GPUs with all-to-all token exchange; the router runs in float32 and ST-MoE adds a z-loss on router logits.

::: {.qa asked="Adobe research intern, London (Glassdoor)"}
**Implement a Mixture-of-Experts layer for a transformer.**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d, n_exp, k=2, alpha=1e-2):
        super().__init__()
        self.k, self.alpha, self.E = k, alpha, n_exp
        self.router = nn.Linear(d, n_exp, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(n_exp))

    def forward(self, x):                                   # x: (T, d), tokens flattened
        p = self.router(x).softmax(-1)                      # (T, E)
        w, idx = p.topk(self.k, dim=-1)                     # (T, k)
        w = w / w.sum(-1, keepdim=True)                     # renormalize over the chosen k
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)   # tokens that chose expert e
            if tok.numel():
                y.index_add_(0, tok, w[tok, slot, None] * expert(x[tok]))
        f = F.one_hot(idx[:, 0], self.E).float().mean(0)    # fraction whose top choice is e
        aux = self.alpha * self.E * (f * p.mean(0)).sum()   # Switch load-balancing loss
        return y, aux
```

The router receives gradient through the weights $w$. Say that production code sorts tokens by expert and uses grouped matmuls, that a capacity limit drops overflow tokens to the residual path, and that under expert parallelism tokens move between GPUs with all-to-all communication.
:::

::: qa
**An MoE has 47B parameters but 13B active. What does each number determine?**

Active parameters set FLOPs per token (about $2 \times 13$B forward) and so training and prefill compute, like a 13B dense model. Total parameters set the memory to hold the model and, at small decoding batches, roughly the bytes read per step. An MoE buys a larger model's quality at a smaller model's compute, paid in memory and communication.
:::

## Training stability

Transformer-specific failure modes; optimizers and schedules are on [deep learning training](deep-learning-training.html).

**Warmup.** Linear warmup over the first few thousand steps (4000 in the 2017 model, 2000 in Llama 2 with cosine decay to 10%, AdamW $\beta_2 = 0.95$, clipping at 1.0), for post-LN's large initial gradients and for Adam's noisy early second-moment estimates.

**Attention-logit growth and QK-norm.** Nothing bounds $\|W_Q\|$ and $\|W_K\|$, so logits can grow until attention is one-hot with zero entropy and the softmax Jacobian vanishes; ViT-22B diverged this way at around 8B parameters and fixed it by normalizing queries and keys before the dot product, $\operatorname{softmax}\big(\mathrm{LN}(XW_Q)\mathrm{LN}(XW_K)^\top/\sqrt d\big)$; SD3 uses RMSNorm on $q$ and $k$ for the same reason in bf16 at high resolution; Wortsman et al. reproduce the instability in small models at high learning rates.

**Output-logit drift and z-loss.** Softmax is shift-invariant, so the loss does not pin down $\log Z = \log \sum_v e^{z_v}$, which can drift to magnitudes where bf16 loses accuracy; PaLM adds $10^{-4}\log^2 Z$ to keep it near 0, and ST-MoE applies the same to router logits. Other standard measures: gradient clipping, no bias terms, bf16 compute with fp32 master weights, and GPT-2's scaling of residual output weights by $1/\sqrt{2L}$ at initialization.

::: qa
**Your large transformer's loss spikes and diverges after a few thousand steps and attention entropy is collapsing. What do you do?**

Logits are growing until the softmax is one-hot. Add QK-normalization per head before the dot product; check warmup length and peak learning rate; clip gradients; consider $\beta_2 = 0.95$ and a z-loss on the output logits; run the softmax in float32. Monitor the maximum attention logit and attention entropy per layer to confirm.
:::

## PyTorch, from scratch

Multi-head attention with causal and padding masks, cross-attention and a KV cache; a pre-LN block; a tiny decoder-only model with a greedy loop that runs with or without the cache. The checks compare against PyTorch's fused kernel and compare cached with uncached generation.

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d, h, d_ctx=None):
        super().__init__()
        assert d % h == 0
        self.h, self.dh = h, d // h
        self.wq = nn.Linear(d, d)
        self.wkv = nn.Linear(d_ctx or d, 2 * d)    # K and V projections fused; context width may differ
        self.wo = nn.Linear(d, d)

    def forward(self, x, ctx=None, causal=False, key_pad=None, cache=None):
        # x: (B, N, d) queries. ctx: (B, M, d_ctx) for cross-attention, None for self.
        # key_pad: (B, M) bool, True = padding key. cache: dict of past k, v (self-attn only).
        B, N, _ = x.shape
        src = x if ctx is None else ctx
        q = self.wq(x).view(B, N, self.h, self.dh).transpose(1, 2)          # (B, h, N, dh)
        k, v = self.wkv(src).view(B, -1, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if cache is not None:                                               # append new k, v
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        M = k.shape[2]                                                      # k, v: (B, h, M, dh)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)                    # (B, h, N, M)
        if causal:  # query i sits at absolute position M - N + i (bottom-right aligned mask)
            ok = torch.ones(N, M, dtype=torch.bool, device=x.device).tril(M - N)
            s = s.masked_fill(~ok, float("-inf"))
        if key_pad is not None:
            s = s.masked_fill(key_pad[:, None, None, :], float("-inf"))
        out = s.softmax(dim=-1) @ v                                         # (B, h, N, dh)
        return self.wo(out.transpose(1, 2).reshape(B, N, -1))              # (B, N, d)

class Block(nn.Module):  # pre-LN: x + f(LN(x))
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MHA(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), causal=True, cache=cache)
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, V, d, h, L, n_max):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(n_max, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight                                  # weight tying

    def forward(self, idx, caches=None, start=0):
        pos = torch.arange(start, start + idx.shape[1], device=idx.device)  # offset by cache length
        x = self.tok(idx) + self.pos(pos)
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.head(self.ln_f(x))                                      # (B, N, V)

@torch.no_grad()
def generate(model, idx, steps, use_cache=True):
    caches = [{} for _ in model.blocks] if use_cache else None
    logits = model(idx, caches)                                             # prefill
    for _ in range(steps):
        nxt = logits[:, -1].argmax(-1, keepdim=True)                        # greedy
        idx = torch.cat([idx, nxt], dim=1)
        logits = model(nxt, caches, start=idx.shape[1] - 1) if use_cache else model(idx)
    return idx

# Checked with torch 2.11, float32: self-attention with causal + padding masks matches
# F.scaled_dot_product_attention to 9e-8, cross-attention to 6e-8; cached and uncached
# greedy generation give identical tokens with logits within 6e-6.
```

The lines that matter in an interview: `tril(M - N)` aligns the causal mask to the bottom-right so that $N$ new queries at the end of $M$ keys see the right prefix (in training $M = N$; in cached decoding $N = 1$ and the row is all True), which is exactly what `is_causal=True` does not do when $N \ne M$; `key_pad[:, None, None, :]` broadcasts a $(B, M)$ mask over heads and queries; the cache concatenates along the time axis and is used only by self-attention, since cross-attention keys would be computed once and reused; and `start=idx.shape[1] - 1` gives the new token its true position.

::: {.qa asked="ByteDance, NVIDIA, OpenAI, Meta, Baidu +4 (9+ reports)"}
**Implement multi-head attention in PyTorch, supporting masks and cross-attention. Then build a transformer block.**

Write `MHA` and `Block` above, saying the shapes aloud: project, `view(B, N, h, d_h)`, transpose to $(B, h, N, d_h)$, scores $QK^\top/\sqrt{d_h}$ of shape $(B, h, N, M)$, masked scores to $-\infty$, softmax over the last axis, multiply by $V$, transpose back, reshape to $(B, N, D)$, output projection. Cross-attention is the same module with keys and values from the context. Mention the fully-masked-row NaN, dropout on attention weights, $O(NMD)$ time and $O(hNM)$ memory if scores are materialized, and that production code calls `scaled_dot_product_attention`, which dispatches to FlashAttention-style kernels; verify against it on a small input. The block is pre-LN, `x + attn(ln1(x))`, `x + mlp(ln2(x))`; a DiT block wraps each sub-layer in a condition-dependent shift, scale and gate.
:::

::: {.qa asked="OpenAI RE (Aced, ~2025)"}
**Add a KV cache to a transformer decoder. What changes?**

Each self-attention layer keeps $K, V$ of shape $(B, h_{\text{kv}}, t, d_h)$; prefill fills them, and each step computes $q, k, v$ for the new token only, appends $k, v$, and attends one query over the cache. Positions must be offset by the cache length (learned embeddings and RoPE alike, rotating keys before caching); the causal mask must be bottom-right aligned when queries are fewer than keys and is unnecessary for one query; batched decoding must track per-sequence lengths. Memory grows linearly, $2Lh_{\text{kv}}d_h$ numbers per token, so serving systems pre-allocate or page it. It is exact because causal hidden states never depend on later tokens; test it by comparing cached and uncached logits.
:::

## Rapid-fire review

::: qa
**Where do the $12D^2$ parameters per block come from?**

$4D^2$ for $W_Q, W_K, W_V, W_O$ and $8D^2$ for a $4\times$ MLP.
:::

::: qa
**Which scale goes in the attention softmax with $h$ heads?**

$1/\sqrt{d_h}$, the per-head width, not $1/\sqrt D$.
:::

::: qa
**Does the number of heads change MHA's parameters or FLOPs?**

No: $4D^2$ and $4ND^2 + 2N^2D$ for any $h$; it changes the number and rank of the score matrices.
:::

::: qa
**At what sequence length does attention cost more than the MLP?**

$N > 4D$; more than all weight matmuls at $N > 6D$.
:::

::: qa
**Where does $6\Psi T$ come from?**

2 FLOPs per parameter per token forward, 4 backward.
:::

::: qa
**KV-cache size per token for Llama 2 7B in fp16?**

512 KiB; 2 GiB for 4096 tokens; 128 KiB with 8 KV heads.
:::

::: qa
**Why is RoPE relative?**

$\langle R_m q, R_n k\rangle = q^\top R_{n-m}k$: rotations compose by adding angles.
:::

::: qa
**Pre-LN or post-LN, in one line?**

Pre-LN keeps an identity gradient path and trains without much warmup; post-LN puts a LayerNorm Jacobian on every path.
:::

::: qa
**Why is the KV cache exact, and why does it not apply to DiT?**

Under a causal mask the state at position $j$ depends only on tokens $\le j$; DiT's attention is bidirectional and every token changes each step.
:::

::: qa
**Batch-1 decode latency floor for a 7B fp16 model on an A100?**

About 13.5 GB / 2 TB/s $\approx$ 6.6 ms per token.
:::

::: qa
**What does FlashAttention not reduce?**

FLOPs; it reduces HBM traffic and memory.
:::

::: qa
**What makes a DiT block the identity at initialization?**

The zero-initialized modulation layer: gates $\alpha$ (and $\gamma$, $\beta$) are 0.
:::

::: qa
**Where does the timestep enter a DiT, a diffusion U-Net and MMDiT?**

adaLN-Zero in every DiT block; FiLM-style addition in every U-Net ResBlock; adaLN in MMDiT together with the pooled text.
:::

::: qa
**Cross-attention cost at SD v1's $64^2$ level?**

$4096 \times 77$ scores per head, against $4096^2$ for self-attention.
:::

::: qa
**Swin's attention cost per block?**

$4hwC^2 + 2M^2hwC$, linear in $hw$, against $4hwC^2 + 2(hw)^2C$ globally.
:::

::: qa
**What do QK-norm and z-loss each prevent?**

QK-norm bounds attention logits; z-loss keeps the output log-normalizer near 0.
:::

## References

1. Vaswani et al., "Attention Is All You Need," 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Devlin et al., "BERT," 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
3. Radford et al., "Language Models are Unsupervised Multitask Learners," 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
4. Dosovitskiy et al., "An Image is Worth 16x16 Words," 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
5. Touvron et al., "Training data-efficient image transformers (DeiT)," 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
6. Liu et al., "Swin Transformer," 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
7. Liu et al., "A ConvNet for the 2020s," 2022. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
8. Raghu et al., "Do Vision Transformers See Like Convolutional Neural Networks?," 2021. [arXiv:2108.08810](https://arxiv.org/abs/2108.08810)
9. Peebles and Xie, "Scalable Diffusion Models with Transformers," 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
10. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
11. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
12. Chen et al., "PixArt-α," 2023. [arXiv:2310.00426](https://arxiv.org/abs/2310.00426)
13. Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning," 2022. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
14. Jaegle et al., "Perceiver: General Perception with Iterative Attention," 2021. [arXiv:2103.03206](https://arxiv.org/abs/2103.03206)
15. Li et al., "BLIP-2," 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
16. Hertz et al., "Prompt-to-Prompt Image Editing with Cross Attention Control," 2022. [arXiv:2208.01626](https://arxiv.org/abs/2208.01626)
17. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
18. Press, Smith and Lewis, "Train Short, Test Long (ALiBi)," 2022. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
19. Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation," 2023. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)
20. Peng et al., "YaRN," 2023. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
21. Heo et al., "Rotary Position Embedding for Vision Transformer," 2024. [arXiv:2403.13298](https://arxiv.org/abs/2403.13298)
22. Xiong et al., "On Layer Normalization in the Transformer Architecture," 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
23. Shazeer, "GLU Variants Improve Transformer," 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
24. Ba, Kiros and Hinton, "Layer Normalization," 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
25. Zhang and Sennrich, "Root Mean Square Layer Normalization," 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
26. Press and Wolf, "Using the Output Embedding to Improve Language Models," 2017. [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)
27. Elhage et al., "A Mathematical Framework for Transformer Circuits," 2021. [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)
28. Kaplan et al., "Scaling Laws for Neural Language Models," 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
29. Hoffmann et al., "Training Compute-Optimal Large Language Models," 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
30. Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," 2022. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
31. Dao et al., "FlashAttention," 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
32. Dao, "FlashAttention-2," 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
33. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
34. Ainslie et al., "GQA," 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
35. Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention," 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
36. Leviathan, Kalman and Matias, "Fast Inference from Transformers via Speculative Decoding," 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
37. Katharopoulos et al., "Transformers are RNNs: Linear Attention," 2020. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
38. Beltagy, Peters and Cohan, "Longformer," 2020. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
39. Fedus, Zoph and Shazeer, "Switch Transformers," 2022. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
40. Zoph et al., "ST-MoE," 2022. [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)
41. Jiang et al., "Mixtral of Experts," 2024. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
42. Touvron et al., "Llama 2," 2023. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
43. Grattafiori et al., "The Llama 3 Herd of Models," 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
44. Jiang et al., "Mistral 7B," 2023. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
45. Raffel et al., "T5," 2020. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
46. Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters," 2023. [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)
47. Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities," 2023. [arXiv:2309.14322](https://arxiv.org/abs/2309.14322)
48. Chowdhery et al., "PaLM," 2022. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
49. Wan Team, "Wan: Open and Advanced Large-Scale Video Generative Models," 2025. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
50. Podell et al., "SDXL," 2023. [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)
