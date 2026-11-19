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

{{fig:overview}}

Read Figure 1 top to bottom. Token IDs of shape $(B, N)$ index an embedding table and become $(B, N, D)$; position is added here (learned or sinusoidal vectors) or applied inside every attention layer (RoPE). Then $L$ identical blocks. Each block owns nothing but two residual branches: a normalized copy of the stream goes through multi-head attention and is added back, then a normalized copy goes through an MLP and is added back. Elhage et al. call the vertical line the residual stream: the final vector at a position is the embedding plus the sum of every sub-layer's output, $x_L = x_0 + \sum_\ell (\text{attn}_\ell + \text{mlp}_\ell)$. Attention is the only operation in the whole network whose output at one position depends on other positions; normalization, the MLP and the additions act per token. After the last block, a final LayerNorm and a $D \times V$ head give logits at every position, and the loss is cross-entropy against the input shifted by one, so one forward pass yields $N$ next-token predictions trained in parallel.

Why this replaced recurrence and convolution: any two positions are connected inside one layer (path length $O(1)$ instead of $O(n)$ for an RNN), all positions of a training sequence are processed at once as large matrix products, and the mixing pattern is decided per input by content rather than fixed by offset. The price is cost quadratic in $N$ and no built-in order or locality, so positions have to be encoded and locality learned.

{{fig:families}}

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

{{fig:tokenize}}

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

{{fig:block}}

One pre-LN block (Figure 4) is

$$
h = x + \operatorname{MHA}\big(\mathrm{LN}_1(x)\big), \qquad x' = h + \operatorname{MLP}\big(\mathrm{LN}_2(h)\big).
$$

**The MLP.** $\operatorname{MLP}(u) = \phi(uW_1 + b_1)W_2 + b_2$ with $W_1 \in \mathbb{R}^{D \times 4D}$, $W_2 \in \mathbb{R}^{4D \times D}$: $8D^2$ weights, two thirds of the block. The factor 4 is the original paper's convention ($512 \to 2048$); Kaplan et al. found loss nearly flat in the ratio at fixed parameter count. $\phi$ is GELU in BERT, GPT-2 and ViT. Llama, PaLM and most recent LLMs use the gated SwiGLU, $\big(\operatorname{SiLU}(uW_1) \odot uW_3\big)W_2$, three matrices instead of two; to keep $8D^2$ weights its hidden width is $\tfrac23 \cdot 4D$, which for Llama 7B is $10{,}923$, rounded to a multiple of 256 as $11{,}008$. Shazeer found GLU variants lower perplexity at matched parameters and gave no theory for it.

**Normalization.** LayerNorm centers and scales the $D$ features of one token, $\gamma \odot (x - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$, $2D$ parameters; RMSNorm drops the mean and the shift, $\gamma \odot x / \sqrt{\tfrac1D \sum_j x_j^2 + \epsilon}$, $D$ parameters. Both are per token, so they do not depend on the batch, on padding or on other positions; batch statistics would let a causal model see future tokens and would differ between training and inference, which is why BatchNorm is absent from transformers.

{{fig:prepost}}

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

{{fig:attention}}

Figure 6 is the whole operation:

$$
\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V, \qquad Q = XW_Q \in \mathbb{R}^{N \times d_k},\ K = CW_K \in \mathbb{R}^{M \times d_k},\ V = CW_V \in \mathbb{R}^{M \times d_v},
$$

with $C = X$ for self-attention and the softmax over each row, that is over the key axis. The score between positions $i$ and $j$ is the bilinear form $x_i^\top W_Q W_K^\top x_j$, of rank at most $d_k$ and not symmetric; the output row $i$ is a convex combination of the value rows, so the output scale does not grow with $M$. Cost: $NMd_k$ MACs for $QK^\top$, $NMd_v$ for $AV$, and $NM$ numbers of memory per head if the scores are materialized; in self-attention both are quadratic in $N$.

**Why $\sqrt{d_k}$.** If the components of $q$ and $k$ are independent with mean 0 and variance 1, which is roughly the case at initialization after normalization, then $\mathbb{E}[q^\top k] = 0$ and $\operatorname{Var}[q^\top k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$, because the $d_k$ products are uncorrelated with variance 1 each. Unscaled logits would have standard deviation 8 for $d_k = 64$, which puts the softmax near one-hot; its Jacobian $\operatorname{diag}(p) - pp^\top$ is then near zero and the gradients to $Q$, $K$ and everything behind them vanish, while the value path still trains. Dividing by $\sqrt{d_k}$ gives unit-variance logits for any head width. It is an initialization argument: logits can still grow during training, which QK-normalization addresses (see "Training stability"). Softmax is shift-invariant, $\operatorname{softmax}(z + c\mathbf 1) = \operatorname{softmax}(z)$, which is what lets implementations subtract the row maximum and what FlashAttention's online softmax exploits. Without positions or masks the operation is permutation equivariant (permuting the rows of $X$ permutes the outputs the same way), which is why order has to be injected.

{{fig:uses}}

**The three uses** (Figure 7) differ only in where $Q$, $K$, $V$ come from and which scores are allowed.

- *Self-attention, bidirectional.* All from $X$; every token reads every token. BERT, ViT, DiT.
- *Causal self-attention.* A language model factorizes $p(x_1, \dots, x_N) = \prod_t p(x_t \mid x_{<t})$; to train all $N$ conditionals in one pass, position $t$ must not see positions after $t$. Add $M_{ij} = -\infty$ for $j > i$ to the scores before the softmax, so those keys get exactly zero weight and the rest renormalize. By induction over layers, the vector at position $t$ in every layer depends only on $x_{\le t}$, so each position's output is a legitimate prediction of $x_{t+1}$, and the same fact is what makes the KV cache exact. Packed documents use a block-diagonal mask so tokens do not attend across boundaries (Llama 3).
- *Cross-attention.* Queries from the sequence being updated, keys and values from another sequence $C \in \mathbb{R}^{M \times D_c}$, with $W_K, W_V \in \mathbb{R}^{D_c \times D}$; the output has the query sequence's length and the cost is $O(NMD)$, cheap when the context is short. The decoder of a translation model reads the encoder this way; Stable Diffusion's U-Net reads 77 CLIP text tokens this way at every resolution (at the $64^2$ level of SD v1: 4096 queries of width 320, 77 keys projected from width 768, so $4096 \times 77$ weights per head rather than $4096^2$); PixArt-$\alpha$ and Wan add a cross-attention sub-layer to each DiT block for text.

**Padding.** Shorter sequences in a batch are padded; a padding mask of shape $(B, 1, 1, M)$ sets the score of every pad key to $-\infty$. A row whose keys are all masked is the softmax of all $-\infty$, which is NaN; left padding with a causal mask produces exactly that at the first pad query, so pad queries must be allowed one key and dropped from the loss. Masked logits must be $-\infty$ or the dtype minimum, not $-10^9$, which overflows float16.

{{fig:heads}}

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

{{fig:rope}}

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

{{fig:kvcache}}

**The KV cache** (Figure 10). Generation runs the prompt once (prefill), then appends one token per step. Without caching, step $t$ reruns the whole prefix. Under a causal mask the hidden state at position $j$ depends only on tokens $\le j$, so $k_j$ and $v_j$ computed when $j$ was the newest token are exactly what a recomputation would give; each layer stores them, of shape $(B, h_{\text{kv}}, t, d_h)$ with RoPE already applied, and a step computes $q$, $k$, $v$ for the new token only, appends $k$ and $v$, and attends one query over the cache with no mask. This is exact, not an approximation. It does not apply to bidirectional attention: a DiT recomputes everything at every denoising step because every token changes.

**Prefill is compute-bound, decode is memory-bound.** Prefill multiplies each weight by all $N_p$ prompt tokens. Decode reads every weight from HBM once per step and uses it for 2 FLOPs per sequence, about $B$ FLOPs per byte in fp16, while an A100 needs about 150 FLOPs per byte (312 TFLOP/s over 2 TB/s) to be compute-bound. At batch 1 a 7B fp16 model needs at least $13.5\ \text{GB} / 2\ \text{TB/s} \approx 6.6$ ms per token, about 150 tokens per second, however fast the arithmetic is; each sequence also reads its own cache every step. Remedies: larger batches, quantized weights and caches, smaller caches, paged cache allocation (PagedAttention), and speculative decoding, where a small draft model proposes several tokens that the large model verifies in one pass without changing the output distribution.

{{fig:gqa}}

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

{{fig:flash}}

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

{{fig:vit}}

ViT is the encoder stack applied to patches (Figure 13): patchify, prepend a learned class token, add learned position embeddings, run pre-LN blocks with bidirectional attention, classify from the class token or from the mean of the patch tokens (the paper's appendix shows both work once the learning rate is retuned). ViT-B/16: $L = 12$, $D = 768$, $h = 12$, 197 tokens, 86M parameters; ViT-L/16: 24, 1024, 16, 304M. Nothing in the blocks knows it is looking at an image: 2-D structure enters only through patchifying and through position embeddings, which are reshaped to the grid and interpolated when the resolution changes (DINOv3 randomizes the coordinate box during training to make this robust, and RoPE-based ViTs use coordinates directly). With so little built in, ViT trained on ImageNet-1k alone trailed comparable ResNets and overtook them with ImageNet-21k or JFT-300M pretraining; DeiT closed the ImageNet-1k gap with heavy augmentation and distillation from a CNN teacher; Raghu et al. found that ViTs pretrained on little data fail to develop locally attending heads in their lowest layers.

{{fig:swin}}

**Swin and the hierarchical alternative** (Figure 14). Global attention over $hw$ tokens costs $4hwC^2 + 2(hw)^2C$; Swin computes attention inside $M \times M$ windows ($M = 7$), $4hwC^2 + 2M^2hwC$, linear in $hw$, and shifts the window grid by $M/2$ in alternate blocks so tokens near a border see across it. Patch merging halves the grid and doubles the channels between stages, giving a four-stage pyramid at strides 4 to 32 that plugs into FPN-style detection and segmentation heads, which a plain ViT's single coarse map does not. ConvNeXt showed that most of ViT's early advantage over ResNet was recipe and macro design rather than attention: a modernized ResNet (patchify stem, depthwise $7 \times 7$ convolutions, inverted bottlenecks, GELU, LayerNorm) matches Swin at equal size. Self-supervised ViT encoders (DINO, DINOv2, DINOv3, MAE) and what their features carry are on [vision backbones](vision-backbones.html).

### How a condition enters: the design space

{{fig:conditioning}}

Every conditional transformer has to get the condition in somewhere, and there are four places (Figure 15). The choice decides cost, parameter sharing and which direction information flows.

| Mechanism | How | Cost added | Where used |
|---|---|---|---|
| In-context tokens | append the condition's tokens to the sequence; ordinary self-attention sees them | $(N + M)^2$ scores, shared weights | DiT's ablation, Flamingo's text tokens, PixArt-Σ-style text in sequence, unified any-to-any models |
| adaLN / FiLM | a conditioning vector $c$ is mapped by an MLP to a shift, scale and gate that modulate every token identically | negligible | DiT (timestep + class), SD3 and FLUX (timestep + pooled text), the ResBlocks of diffusion U-Nets (timestep) |
| Cross-attention | tokens query the condition sequence; $K$, $V$ from the condition | $NMD$ per layer | SD U-Net, PixArt-α, Wan, encoder–decoder translation |
| Joint attention | both sequences attended together, each with its own projections and MLP | $(N + M)^2$ scores, separate weights | MMDiT in SD3 and FLUX |

Two rules of thumb fall out. A global signal with no spatial structure (timestep, class, a pooled embedding) belongs in adaLN: it is cheapest and the DiT paper found it best for class conditioning, with cross-attention costing 15% more FLOPs for a worse FID. A sequence with internal structure (text tokens) needs attention; cross-attention leaves the text untouched and costs $NM$, joint attention lets the text be updated by the image at every layer and costs $(N+M)^2$, and SD3's comparison found joint attention better than both cross-attention and a shared-weight in-context DiT.

### DiT

{{fig:dit}}

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

{{fig:unet}}

Latent diffusion's backbone (Figure 17) is a convolutional U-Net with transformer blocks inside it. The encoder halves the resolution and raises the channels level by level ($64^2 \times 320 \to 32^2 \times 640 \to 16^2 \times 1280$ in SD v1, plus a mid block at $8^2$), the decoder reverses it, and skip connections concatenate each encoder level's features onto the decoder input at the same resolution, which is what keeps fine detail. Each level has ResBlocks and a transformer block. The timestep enters every ResBlock as an embedding passed through a linear layer and added to the feature map, a FiLM-style modulation, and the text enters every transformer block: the feature map is flattened to $(HW) \times C$ tokens, which pass through self-attention over image positions, cross-attention whose keys and values are the 77 CLIP text tokens, and a GEGLU MLP, each with a residual, and are reshaped back. At the $64^2$ level that is 4096 queries of width 320 against 77 keys projected from width 768, with 8 heads of width 40: $4096 \times 77$ weights per head, so cross-attention is cheap next to the $4096^2$ self-attention at the same level. Each cross-attention row says which words an image location reads, which is what Prompt-to-Prompt edits. Classifier-free guidance reruns the network with the empty prompt's embeddings. SDXL enlarges the transformer blocks at the lower-resolution levels and adds a second text encoder; SD3 and FLUX drop the U-Net for MMDiT.

### MMDiT

{{fig:mmdit}}

SD3's MMDiT (Figure 18) keeps text and image as two token streams with separate weights, their own adaLN modulation, $Q$/$K$/$V$ and output projections, and MLPs, but concatenates their queries, keys and values into one attention over all $M + N$ tokens:

$$
\begin{pmatrix} O_c \\ O_x \end{pmatrix} = \operatorname{Attn}\!\left(\begin{pmatrix} Q_c \\ Q_x \end{pmatrix}, \begin{pmatrix} K_c \\ K_x \end{pmatrix}, \begin{pmatrix} V_c \\ V_x \end{pmatrix}\right),
$$

then splits the output back into each stream's own $W_O$ and MLP. The $(M + N) \times (M + N)$ weight matrix has four blocks: text reading text, text reading image, image reading text (the one block cross-attention has) and image reading image (self-attention). Unlike cross-attention, the text representation is updated by the image at every layer, at $(N + M)^2$ scores instead of $N^2 + NM$. SD3's text sequence is 77 CLIP tokens zero-padded to width 4096 followed by 77 T5-XXL tokens; pooled CLIP vectors and the timestep enter through adaLN; RMSNorm on queries and keys keeps bf16 training stable at high resolution. FLUX.1 uses double-stream MMDiT blocks followed by single-stream blocks with shared weights, and 2-D RoPE on image-token coordinates.

### Attaching new inputs to a frozen model

{{fig:gatedxattn}}

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
