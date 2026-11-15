---
title: "Transformers, Step by Step"
subtitle: "A guided reading from tokens to FlashAttention: attention computed by hand, masks, heads, position encodings, the block, parameter and FLOP counting, KV caching, DiT conditioning, mixture of experts, and a small PyTorch model checked against the library."
order: 0.5
updated: 2026-11-15
---

A transformer maps a sequence of $N$ vectors to $N$ new vectors of the same width. It alternates two operations: attention, in which every position reads from other positions according to their content, and a small MLP applied to each position on its own. Residual connections and normalization hold the stack together. That is the whole architecture behind GPT-style language models, BERT, the Vision Transformer (ViT), the Diffusion Transformer (DiT), the text encoders of text-to-image models, and the MMDiT backbones of Stable Diffusion 3 and FLUX. The [diffusion notes](diffusion-models.html) treat the denoiser as a black box $\epsilon_\theta(x_t, t)$; this page opens that box for the case where it is a transformer.

**Why transformers replaced recurrence and convolution.** Vaswani et al. (2017) compared layer types on three counts in their Table 1. With $n$ the sequence length, $d$ the width and $k$ the kernel size:

| Layer type | Cost per layer | Sequential steps | Longest path between two positions |
|---|---|---|---|
| Self-attention | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k n d^2)$ | $O(1)$ | $O(n/k)$, or $O(\log_k n)$ with dilation |

Three advantages follow from that table.

1. **Short paths.** In a recurrent network, information from token $i$ reaches token $j$ through $j - i$ state updates. The gradient $\partial h_j / \partial h_i$ is a product of $j - i$ Jacobians, which tends to shrink or grow geometrically, and the whole prefix has to fit in one fixed-size state vector. In self-attention, any two positions are connected inside one layer.
2. **Parallel training.** A recurrent network must compute $h_1$ before $h_2$. A transformer processes every position of a training sequence at once, as large matrix products that GPUs run efficiently. With a causal mask (below), one forward pass yields a next-token prediction at every position.
3. **Lookup by content.** A convolution mixes a fixed set of relative offsets with fixed weights. Attention decides, separately for each input, which positions to read, by comparing their content.

The price is a cost quadratic in $n$, and no built-in notion of order or locality: order has to be injected with position encodings, and locality has to be learned from data.

**Roadmap.** The page first turns text, images and latents into an $N \times D$ matrix, then builds attention from scratch: a three-token example by hand, the $\sqrt{d_k}$ factor, the softmax Jacobian and permutation equivariance. Next come masks and cross-attention, multi-head attention, position encodings (sinusoidal, RoPE with long-context scaling, ALiBi), the full block with pre-LN versus post-LN, and the three model families with their losses. "Counting" verifies GPT-2's 124M parameters and derives the $6\Psi T$ compute rule, activation memory and KV-cache size. Then inference (KV cache, MQA and GQA, sampling), efficient attention (FlashAttention's online softmax, local and linear attention), the vision and generation models (ViT, DiT, Stable Diffusion, MMDiT), mixture of experts, training stability, a from-scratch PyTorch model, a rapid-fire review and references. Practice questions follow the section they test. Swin, ConvNeXt and self-supervised encoders are on [vision backbones](vision-backbones.html); optimizers and schedules are on [deep learning training](deep-learning-training.html). Pretraining data and scaling, decoding, LoRA, RLHF and DPO for language models are on [large language models](llms.html).

## Notation

Rows are tokens throughout: $X \in \mathbb{R}^{N \times D}$ has one row per token, and a linear layer is $XW$ with $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$. (PyTorch's `nn.Linear` stores the transpose, of shape $(D_{\text{out}}, D_{\text{in}})$, and computes $xW^\top$.)

| Symbol | Meaning |
|---|---|
| $B$ | batch size |
| $N$ | number of tokens (queries); $M$ is the number of keys when it differs (cross-attention, KV cache) |
| $D$ | model width, the size of each token vector in the residual stream ($d_{\text{model}}$ in papers) |
| $h$ | number of attention heads |
| $d_h = D/h$ | width of one head |
| $d_k$, $d_v$ | query/key width and value width of one attention operation; $d_k = d_v = d_h$ in multi-head attention |
| $L$ | number of layers (blocks) |
| $V$ | vocabulary size |
| $d_{\text{ff}}$ | hidden width of the MLP, usually $4D$ |
| $X$; $x_i$ | the $N \times D$ input to a layer; $x_i$ is token $i$'s vector (a row, or a column vector inside formulas) |
| $W_Q, W_K, W_V, W_O$ | query, key, value and output projections, each $D \times D$ in a standard layer |
| $Q = XW_Q$, $K$, $V$ | query, key and value matrices; rows $q_i$, $k_j$, $v_j$ |
| $S = QK^\top/\sqrt{d_k}$ | the score (logit) matrix, entries $s_{ij}$ |
| $A = \operatorname{softmax}(S)$ | attention weights, softmax over each row; entries $a_{ij}$ |
| $E \in \mathbb{R}^{V \times D}$ | token embedding table; $W_U$ is the output (unembedding) matrix |
| $P$; $p$ | patch size for images (ViT); patch size on latents (DiT) |
| $\theta_i$ | rotation frequency of the $i$-th channel pair in RoPE |
| $\mathrm{LN}$ | layer normalization over the $D$ features of one token |
| $\Psi$ | parameter count; $T$ is the number of training tokens |
| $\tau$ | sampling temperature |
| MAC | one multiply-add, which is 2 FLOPs |

## Tokens and embeddings

Every transformer starts by turning its input into a matrix of shape $(B, N, D)$: a batch of $B$ sequences, each of $N$ tokens, each token a $D$-dimensional vector. The sources differ by modality; everything after this step is shared.

### Text tokens

A tokenizer maps text to integer IDs from a fixed vocabulary of size $V$. Most current language models use subword vocabularies learned by byte-pair encoding (BPE): start from single bytes or characters, count adjacent pairs in a corpus, merge the most frequent pair into a new symbol, and repeat until the vocabulary reaches its target size. To encode new text, apply the learned merges in the order they were learned. Frequent words end up as one token and rare words split into pieces; byte-level BPE (GPT-2, $V = 50{,}257$) can encode any string because the 256 byte values are in the vocabulary. The output is an integer tensor of shape $(B, N)$. The [question bank](interview-question-bank.html) has a worked BPE example and the variants (WordPiece, Unigram, SentencePiece).

### The embedding lookup is a matrix product

An embedding table $E \in \mathbb{R}^{V \times D}$ stores one row per vocabulary entry, and token ID $t$ becomes the row $E[t]$. Write $e_t \in \mathbb{R}^V$ for the one-hot vector with a 1 in position $t$. Then

$$
E[t] = e_t^\top E ,
$$

because multiplying a one-hot row vector by a matrix selects one row. So the embedding layer is a linear layer applied to one-hot inputs, implemented as indexing to avoid a $V \times D$ product. Two consequences:

- **Sparse gradients.** For a batch with tokens $t_1, \dots, t_n$ and upstream gradients $g_1, \dots, g_n \in \mathbb{R}^D$, $\partial \mathcal{L}/\partial E = \sum_i e_{t_i} g_i^\top$. Only rows of tokens that occur in the batch receive gradient, so rare tokens are updated rarely.
- **Shapes.** IDs of shape $(B, N)$ become vectors of shape $(B, N, D)$.

The original transformer multiplies the embeddings by $\sqrt{D}$ before adding its position encodings.

### Image patches (ViT)

An image of shape $(B, C, H, W)$ is cut into non-overlapping $P \times P$ patches. That gives $N = HW/P^2$ patches, each with $P^2 C$ numbers, and a learned linear map takes each to $D$ dimensions. In code this is a convolution with kernel size and stride both equal to $P$: each output position sees exactly one patch, and the kernel weights are shared across positions, which is the same as one linear map applied to every flattened patch. Then `flatten(2).transpose(1, 2)` turns $(B, D, H/P, W/P)$ into $(B, N, D)$.

For ViT-B/16 on $224 \times 224$ RGB images: $N = (224/16)^2 = 196$ patches of $16 \cdot 16 \cdot 3 = 768$ numbers each, projected to $D = 768$ with $768 \cdot 768 + 768 = 590{,}592$ parameters. ViT prepends a learned class token, giving 197 tokens.

### Latent patches (DiT)

DiT applies the same patchify step to the latent of a pretrained VAE rather than to pixels. A $256 \times 256 \times 3$ image becomes a $32 \times 32 \times 4$ latent (downsampling factor 8). With latent patch size $p$, the number of tokens is $(32/p)^2$: $p = 2$ gives 256 tokens of $2 \cdot 2 \cdot 4 = 16$ numbers each, which a linear layer maps to $D = 1152$ in DiT-XL. Halving $p$ multiplies $N$ by 4. Because DiT's 256 tokens are few compared with its width (see "Counting"), per-token work dominates and compute grows almost linearly in $N$: the DiT paper lists 7.39, 29.05 and 118.64 Gflops for DiT-XL/8, XL/4 and XL/2. At the output, a linear layer maps each token to $p \times p \times 2C$ numbers (noise and a diagonal covariance), and an "unpatchify" reshape restores the latent layout.

::: qa
**How many tokens does DiT with $p = 2$ see for a $512 \times 512$ image with an f8 VAE, and how do the costs change compared with $256 \times 256$?**

The latent is $64 \times 64 \times 4$, so $N = (64/2)^2 = 1024$ tokens, four times as many. Per-token work (projections and MLPs) grows $4\times$; the attention score matrices have $N^2$ entries, so that part grows $16\times$.
:::

::: {.qa asked="Amazon AS intern, Glassdoor 2026"}
**What does a tokenizer do, and how does byte-pair encoding work?**

A tokenizer maps text to integer IDs from a fixed vocabulary and back; subword tokenizers sit between characters (long sequences) and whole words (huge vocabulary, unknown words). BPE training: write each corpus word as base symbols (bytes or characters), count adjacent pairs weighted by word frequency, merge the most frequent pair into a new symbol, record the merge, and repeat until the vocabulary reaches its target size. Encoding applies the recorded merges in order, so unseen words still get a segmentation. With word counts `cat` 6, `cats` 2, `hat` 4, `that` 3, the pair (a, t) occurs $6 + 2 + 4 + 3 = 15$ times and is merged first, then (c, at) with 8 and (h, at) with 7. A larger vocabulary gives shorter sequences but bigger embedding and output matrices. Side effects: inconsistent number chunking, higher cost for languages underrepresented in the tokenizer's data, and trouble with spelling, since the model sees tokens, not letters.
:::

## Attention from scratch

### Attention as a soft dictionary lookup

A Python dictionary stores pairs $(k_j, v_j)$; a query $q$ returns the $v_j$ whose key equals $q$. Attention makes this lookup soft and differentiable. Score every key against the query, turn the scores into positive weights that sum to one, and return the weighted average of the values:

$$
s_j = \frac{q^\top k_j}{\sqrt{d_k}}, \qquad
a_j = \frac{e^{s_j}}{\sum_{l=1}^{M} e^{s_l}}, \qquad
\text{out} = \sum_{j=1}^{M} a_j v_j .
$$

If one score is much larger than the rest, the weights are nearly one-hot and this is a hard lookup. Otherwise the output blends several values. Every step is differentiable in $q$, the keys and the values.

In self-attention every token plays all three roles. Token $i$ emits a query $q_i$ (what it is looking for), a key $k_i$ (what it offers to be found by) and a value $v_i$ (what it hands over when found), each through its own learned matrix: $Q = XW_Q$, $K = XW_K$, $V = XW_V$. The score between tokens $i$ and $j$ is

$$
q_i^\top k_j = x_i^\top W_Q W_K^\top x_j ,
$$

a learned bilinear form of rank at most $d_k$. It is not symmetric: token $i$ can attend strongly to $j$ while $j$ ignores $i$. Separate key and value maps let the features used for matching differ from the features that get copied.

### A worked example by hand

Take three tokens with $d_k = d_v = 2$, and suppose the projections have already produced

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}, \qquad
K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \qquad
V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 2 & 2 \end{pmatrix}.
$$

**Step 1: dot products.** Entry $(i, j)$ of $QK^\top$ is $q_i^\top k_j$. For example $q_2^\top k_3 = 0 \cdot 1 + 2 \cdot 1 = 2$ and $q_3^\top k_3 = 1 + 1 = 2$:

$$
QK^\top = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 2 & 2 \\ 1 & 1 & 2 \end{pmatrix}.
$$

**Step 2: scale.** Divide by $\sqrt{d_k} = \sqrt 2 \approx 1.4142$:

$$
S = \begin{pmatrix} 0.7071 & 0 & 0.7071 \\ 0 & 1.4142 & 1.4142 \\ 0.7071 & 0.7071 & 1.4142 \end{pmatrix}.
$$

**Step 3: exponentiate.** $e^{0} = 1$, $e^{0.7071} = 2.0281$, $e^{1.4142} = 4.1133$. The rows become $(2.0281, 1, 2.0281)$ with sum $5.0562$; $(1, 4.1133, 4.1133)$ with sum $9.2265$; and $(2.0281, 2.0281, 4.1133)$ with sum $8.1695$.

**Step 4: normalize each row.** Divide each row by its sum:

$$
A = \begin{pmatrix} 0.4011 & 0.1978 & 0.4011 \\ 0.1084 & 0.4458 & 0.4458 \\ 0.2483 & 0.2483 & 0.5035 \end{pmatrix}.
$$

**Step 5: average the values.** Row $i$ of the output is $\sum_j a_{ij} v_j$:

$$
\begin{aligned}
\text{out}_1 &= 0.4011\,(1,0) + 0.1978\,(0,1) + 0.4011\,(2,2) = (1.2033,\ 1.0000), \\
\text{out}_2 &= 0.1084\,(1,0) + 0.4458\,(0,1) + 0.4458\,(2,2) = (1.0000,\ 1.3374), \\
\text{out}_3 &= 0.2483\,(1,0) + 0.2483\,(0,1) + 0.5035\,(2,2) = (1.2552,\ 1.2552),
\end{aligned}
$$

up to rounding in the last digit. Read it as a lookup. Query 2, $(0, 2)$, looks for the second feature; keys 2 and 3 both have it, so they tie at 0.4458 and key 1 gets little. Query 3, $(1, 1)$, matches key 3 best (score 2 against 1). Each output is a convex combination of the value rows, so it lies in the triangle with corners $(1,0)$, $(0,1)$ and $(2,2)$. If every score were multiplied by 10, row 3 would become $(0.0008, 0.0008, 0.9983)$, almost a hard lookup of $v_3$.

### The matrix form

Stacking all queries gives scaled dot-product attention:

$$
\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V, \qquad Q \in \mathbb{R}^{N \times d_k},\ K \in \mathbb{R}^{M \times d_k},\ V \in \mathbb{R}^{M \times d_v},
$$

with the softmax taken over each row, that is, over the key axis. $S$ and $A$ are $N \times M$ and the output is $N \times d_v$: one output per query. The cost is $NMd_k$ multiply-adds for $QK^\top$ and $NMd_v$ for $AV$, and storing $S$ or $A$ takes $NM$ numbers. In self-attention $M = N$, so both time and memory are quadratic in the sequence length.

### Why softmax

- **Convex weights.** The weights are positive and sum to one, so the output lies in the convex hull of the values and its scale does not grow with the number of keys.
- **A smooth argmax.** $\operatorname{softmax}(z/\tau)$ tends to the one-hot vector of the largest entry as $\tau \to 0$; softmax is the gradient of $\log \sum_j e^{z_j}$, a smooth $\max$.
- **Shift invariance.** $\operatorname{softmax}(z + c\mathbf{1}) = \operatorname{softmax}(z)$ because $e^c$ cancels. Implementations subtract the row maximum so no exponential overflows; FlashAttention relies on the same identity.
- **Ratios depend on score gaps.** $a_i / a_j = e^{s_i - s_j}$: a gap of $\ln 10 \approx 2.3$ means ten times the weight.

### Why divide by $\sqrt{d_k}$

Assume the components of $q$ and $k$ are independent with mean 0 and variance 1. That is roughly true at initialization when the layer input is normalized and the projections use variance-preserving initialization.

**Mean.** $\mathbb{E}[q^\top k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0$, because $q_i$ and $k_i$ are independent and each has mean zero.

**Variance.** Expand the variance of the sum:

$$
\operatorname{Var}[q^\top k] = \sum_{i=1}^{d_k} \operatorname{Var}[q_i k_i] + \sum_{i \ne j} \operatorname{Cov}[q_i k_i,\ q_j k_j].
$$

Each cross term is $\mathbb{E}[q_i q_j]\,\mathbb{E}[k_i k_j] - 0 = 0$, because different coordinates are independent with mean zero. Each diagonal term is $\mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] - 0 = 1$. So

$$
\operatorname{Var}[q^\top k] = d_k, \qquad \operatorname{std}[q^\top k] = \sqrt{d_k}.
$$

A simulation with $d_k = 64$ gives a standard deviation of 7.99. Dividing by $\sqrt{d_k}$ brings the logits back to variance 1 for any head width, so how peaked attention is at initialization does not depend on $d_h$. The next subsection shows why unit-scale logits matter. Dividing by $d_k$ instead would give logit variance $1/d_k$ and nearly uniform attention in wide heads at the start. That is not always wrong: the μP parameterization (Yang et al., 2022) uses $1/d_k$ on purpose, because after training $q$ and $k$ become correlated and their dot product grows like $d_k$ rather than $\sqrt{d_k}$. The $\sqrt{d_k}$ argument is about initialization.

### The softmax Jacobian, and why saturation kills gradients

Let $p = \operatorname{softmax}(z)$, so $p_i = e^{z_i}/Z$ with $Z = \sum_l e^{z_l}$. Differentiate with the quotient rule, using $\partial Z/\partial z_j = e^{z_j}$:

$$
\frac{\partial p_i}{\partial z_j} = \frac{\delta_{ij} e^{z_i}}{Z} - \frac{e^{z_i} e^{z_j}}{Z^2} = p_i(\delta_{ij} - p_j),
\qquad\text{so}\qquad
J = \operatorname{diag}(p) - pp^\top .
$$

$J$ is symmetric, so in backpropagation, with upstream gradient $g = \partial \mathcal{L}/\partial p$,

$$
\frac{\partial \mathcal{L}}{\partial z} = Jg = p \odot \big(g - (p^\top g)\,\mathbf{1}\big).
$$

If $p$ is exactly one-hot, $p = e_k$, then $J = \operatorname{diag}(e_k) - e_k e_k^\top = 0$, and near one-hot every entry is small. For logits $z = (s, 0, 0)$:

| $s$ | $p$ | $J_{11} = p_1(1-p_1)$ |
|---|---|---|
| 1 | $(0.576, 0.212, 0.212)$ | 0.244 |
| 4 | $(0.965, 0.018, 0.018)$ | 0.034 |
| 8 | $(0.9993, 0.0003, 0.0003)$ | 0.00067 |

At $s = 8$ the gradient through the softmax is about 360 times smaller than at $s = 1$, and unscaled logits with $d_k = 64$ have standard deviation 8, so gaps of that size are common at initialization. What vanishes is the gradient with respect to the scores, and through them to $Q$, $K$, $W_Q$, $W_K$ and the inputs along that path. What survives is the gradient to the values, since $\partial\, \text{out}_i / \partial v_j = a_{ij}$. So a saturated head keeps updating what it copies but stops learning where to look.

### Permutation equivariance, proved

**Claim.** Let $P$ be an $N \times N$ permutation matrix (it reorders the rows of whatever it multiplies) and $F(X) = \operatorname{softmax}(XW_Q (XW_K)^\top/\sqrt{d_k})\, XW_V$ be self-attention without a mask. Then $F(PX) = P\,F(X)$.

**Proof.**

1. The projections act row by row, so the permuted input gives $Q' = PXW_Q = PQ$, and likewise $K' = PK$ and $V' = PV$.
2. The scores become $S' = PQ(PK)^\top/\sqrt{d_k} = P S P^\top$: rows and columns are reordered by the same permutation.
3. Row-wise softmax commutes with this: $\operatorname{softmax}(PSP^\top) = P\operatorname{softmax}(S)P^\top$. Reordering rows only changes which row is which, and reordering the entries inside a row does not change the row's sum of exponentials.
4. The output is $A'V' = PAP^\top PV = PAV$, because $P^\top P = I$ for a permutation matrix. $\blacksquare$

The MLP and LayerNorm act on each token separately, so they are equivariant too, and so is a whole block and a whole stack. A transformer without position information therefore treats its input as a set: shuffle the tokens and the outputs shuffle the same way. For cross-attention, permuting the context tokens (keys and values together) leaves the output unchanged: $A' = AP^\top$, $V' = PV$, so $A'V' = AV$. A causal mask breaks equivariance because it depends on the index, and position encodings break it on purpose.

::: {.qa asked="ByteDance, Kling, TME, Meta, NVIDIA (10+ reports)"}
**How does scaled dot-product attention work, and why divide by $\sqrt{d_k}$?**

Project the inputs to queries, keys and values. Score each query against every key with a dot product, divide by $\sqrt{d_k}$, apply a softmax over the keys, and return the weighted average of the values: $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. Each output is a convex combination of values, weighted by how well its query matches each key. If the components of $q$ and $k$ are independent with mean 0 and variance 1, then $q^\top k$ has mean 0 and variance $d_k$, because the $d_k$ products each have variance 1 and are uncorrelated. Unscaled logits would have standard deviation 8 for $d_k = 64$, pushing the softmax toward one-hot, where its Jacobian $\operatorname{diag}(p) - pp^\top$ is nearly zero and the gradients to $Q$ and $K$ vanish. Dividing by $\sqrt{d_k}$ restores unit variance for any head width. This is an initialization argument; logits can still grow during training, which QK-normalization addresses.
:::

::: qa
**An attention head's weights have become nearly one-hot. Which parameters still get gradient?**

The value path does: $\partial\,\text{out}_i/\partial v_j = a_{ij}$, so $W_V$ (and $W_O$) still learn what is copied. The score path does not: the softmax Jacobian $\operatorname{diag}(p) - pp^\top$ is close to zero near a one-hot $p$, so $W_Q$, $W_K$ and the inputs through them get almost no gradient, and the head stops changing where it looks.
:::

::: qa
**Is self-attention permutation equivariant? What breaks the symmetry in practice?**

Yes. For a permutation matrix $P$, $Q, K, V$ become $PQ, PK, PV$, the scores become $PSP^\top$, row-wise softmax commutes with that, and $PAP^\top PV = PAV$. MLPs and LayerNorm are per-token, so the whole unmasked stack is equivariant. Position encodings break the symmetry deliberately, and a causal mask breaks it because it depends on the index. Without either, "dog bites man" and "man bites dog" would produce the same set of outputs.
:::

::: key
**How to say it in an interview.** "Attention is a differentiable dictionary lookup. Each token makes a query, a key and a value with three linear maps. A query is compared with every key by a dot product, the scores are divided by $\sqrt{d_k}$ and turned into weights with a softmax over the keys, and the output is the weighted average of the values. The scaling keeps the logits at unit variance at initialization, since a dot product of $d_k$ independent unit-variance terms has variance $d_k$; without it the softmax saturates and its Jacobian, $\operatorname{diag}(p) - pp^\top$, kills the gradients to the queries and keys. Without positions or masks the operation is permutation equivariant, so order has to be injected."
:::

::: pitfall
Common mistakes when writing attention by hand: taking the softmax over the query axis (`dim=-2`) instead of the key axis (`dim=-1`); scaling by $\sqrt{D}$ instead of the per-head $\sqrt{d_h}$; forgetting that a row whose keys are all masked gives $\operatorname{softmax}$ of all $-\infty$, which is NaN; and filling masked logits with a large constant such as $-10^9$, which is out of range in float16 (PyTorch raises an overflow error); use $-\infty$ or the dtype's minimum.
:::

## Masks and the three uses of attention

The same operation is used in three ways. They differ only in where $Q$, $K$ and $V$ come from and in which scores are allowed.

### Self-attention

$Q$, $K$ and $V$ all come from the same $X$. In an encoder (BERT, ViT, DiT) every token may attend to every other token; this is called bidirectional attention.

### Causal masking

A language model factorizes the probability of a sequence with the chain rule, $p(x_1, \dots, x_N) = \prod_{t} p(x_t \mid x_{<t})$. To train all $N$ conditionals in one pass, position $t$ must not see positions after $t$. Add a mask matrix to the scores before the softmax:

$$
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}, \qquad A = \operatorname{softmax}(S + M).
$$

Since $e^{-\infty} = 0$, a masked key gets exactly zero weight, and the remaining weights renormalize over the visible keys.

**Worked mask.** For the four tokens "the cat sat down":

$$
M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix},
\qquad
A = \begin{pmatrix} 1 & 0 & 0 & 0 \\ \tfrac12 & \tfrac12 & 0 & 0 \\ \tfrac13 & \tfrac13 & \tfrac13 & 0 \\ \tfrac14 & \tfrac14 & \tfrac14 & \tfrac14 \end{pmatrix},
$$

where $A$ is what you get if every raw score is equal: row $t$ averages the first $t$ tokens. Training feeds the sequence as input and the same sequence shifted left by one as the targets:

| Position | Visible input | Target |
|---|---|---|
| 1 | the | cat |
| 2 | the cat | sat |
| 3 | the cat sat | down |
| 4 | the cat sat down | the next token in the data |

One forward pass over $N$ tokens produces $N$ next-token predictions, and the loss averages all of them.

**Why there is no leakage.** Claim: with a causal mask in every layer, the vector at position $t$ in every layer depends only on $x_1, \dots, x_t$. By induction over layers. The input vector at $t$ depends on token $t$ and the position $t$ only. If every layer-$\ell$ vector at position $j$ depends only on tokens $\le j$, then layer $\ell + 1$'s attention at position $t$ mixes values from positions $j \le t$, each depending only on tokens $\le j \le t$, and the MLP and LayerNorm act on position $t$ alone. $\blacksquare$ So the output at $t$ is a legitimate function of the prefix. The same fact makes KV caching exact (see "Inference"). A recurrent network also yields $N$ predictions per sequence, but it computes them one after another; the transformer computes them in parallel.

When several documents are packed into one training sequence, the causal mask can be made block-diagonal so tokens do not attend across document boundaries; Llama 3 does this.

### Padding masks

Sequences in a batch have different lengths, so shorter ones are padded to a common length with a pad token. A padding mask sets the score to $-\infty$ for every key that is a pad token, so real tokens never read from padding. It has shape $(B, 1, 1, M)$ and broadcasts over heads and queries; combine it with the causal mask by a logical AND on "allowed". Queries at pad positions still produce outputs, which the loss must ignore.

A row whose keys are all masked gives the softmax of a vector of $-\infty$, which is NaN, and the NaN spreads through later layers. Left padding plus a causal mask does this: the first pad query can see only itself, and itself is masked. Let pad queries attend to something harmless and drop them in the loss, or zero those rows. (Batched generation pads on the left so every sequence's last real token sits at the same index.)

### Cross-attention

Queries come from one sequence and keys and values from another. With $X \in \mathbb{R}^{N \times D}$ the sequence being updated and $C \in \mathbb{R}^{M \times D_c}$ the context:

$$
Q = XW_Q,\quad K = CW_K,\quad V = CW_V, \qquad W_Q \in \mathbb{R}^{D \times D},\ W_K, W_V \in \mathbb{R}^{D_c \times D}.
$$

The scores are $N \times M$ and the output is $N \times D$, so the output has the length of the query sequence. The cost is $O(NMD)$, cheap when the context is short. No causal mask is needed because the whole context is available, though a padding mask on the context keys may be. Three examples:

1. **Encoder–decoder translation** (the original transformer). Each decoder block has causal self-attention, then cross-attention whose queries are decoder states and whose keys and values are the encoder's outputs, then an MLP. The encoder's keys and values are computed once per source sentence and reused at every decoding step.
2. **Stable Diffusion's U-Net.** Queries are image features, keys and values are CLIP text-token embeddings. At the $64 \times 64$ latent level of Stable Diffusion v1, $Q$ is $(B, 4096, 320)$ and the text is $(B, 77, 768)$, so $W_K$ and $W_V$ are $768 \times 320$ and the weights $A$ are $4096 \times 77$ per head. Details in "Transformers in vision and generation".
3. **Text-conditioned DiTs.** PixArt-$\alpha$ adds a cross-attention layer to text tokens in each block, and Wan 2.1 does the same with umT5 text tokens. The original DiT paper tried a cross-attention block for class conditioning; it added about 15% to the Gflops and gave worse FID than adaLN-Zero.

::: qa
**Why does a causal mask let one forward pass train on $N$ next-token predictions?**

The mask sets $s_{ij} = -\infty$ for $j > i$, so in every layer position $i$ only mixes values from positions $\le i$. By induction over layers, the final vector at position $i$ depends only on $x_1, \dots, x_i$. Each position's output can therefore be trained to predict $x_{i+1}$ without seeing it, and all $N$ predictions come from the same pass, computed in parallel. Feeding the true prefix rather than the model's own samples is called teacher forcing.
:::

::: qa
**Your padded batch produces NaNs after the first attention layer. What is a likely cause?**

A query row in which every key is masked: the softmax of all $-\infty$ is $0/0$. Typical causes are left padding combined with a causal mask, or an inverted mask (True meaning "block" in one place and "allow" in another). Fix the mask convention, let pad queries attend to at least one key and exclude them from the loss, or zero fully masked rows. Also check that masked logits are not filled with a constant that overflows in half precision.
:::

## Multi-head attention

Multi-head attention (MHA) runs $h$ attention operations in parallel on different learned projections and concatenates the results:

$$
\text{head}_i = \operatorname{Attn}\big(XW_Q^{(i)},\ XW_K^{(i)},\ XW_V^{(i)}\big), \qquad
\operatorname{MHA}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O,
$$

with $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{D \times d_h}$, $d_h = D/h$, and $W_O \in \mathbb{R}^{D \times D}$. In code, $W_Q = [W_Q^{(1)} \cdots W_Q^{(h)}]$ is a single $D \times D$ matrix and the heads are blocks of its columns.

### The shapes, step by step

Take $B = 2$, $N = 5$, $D = 12$, $h = 3$, so $d_h = 4$.

1. `q = x @ W_Q` has shape $(2, 5, 12)$.
2. `q.view(B, N, h, d_h)` gives $(2, 5, 3, 4)$: the 12 channels are split into 3 blocks of 4, and channel $c$ of head $i$ is column $4i + c$.
3. `.transpose(1, 2)` gives $(2, 3, 5, 4)$: heads become a batch dimension.
4. `q @ k.transpose(-2, -1)` multiplies $(2, 3, 5, 4)$ by $(2, 3, 4, 5)$ to give scores of shape $(2, 3, 5, 5)$: one $N \times N$ matrix per head per sample.
5. Softmax over the last axis, then `@ v` gives $(2, 3, 5, 4)$.
6. `.transpose(1, 2).reshape(B, N, D)` gives $(2, 5, 12)$. This is the concatenation of the heads.
7. Multiply by $W_O$: $(2, 5, 12)$.

### Parameters and cost

$W_Q$, $W_K$, $W_V$ and $W_O$ are each $D \times D$, so MHA has $4D^2$ weights plus $4D$ biases, whatever $h$ is. For GPT-2 small, $4 \cdot 768^2 + 4 \cdot 768 = 2{,}362{,}368$.

The projections cost $4ND^2$ multiply-adds whatever $h$ is. Each head's $QK^\top$ costs $N^2 d_h$, so all heads together cost $h N^2 d_h = N^2 D$, the same as a single head of width $D$; likewise for $AV$. So the layer costs

$$
4ND^2 + 2N^2D \ \text{multiply-adds}
$$

for any number of heads. What does change with $h$: there are $h$ score matrices, so materializing them takes $hN^2$ numbers, and very small $d_h$ uses matrix hardware less efficiently.

### Why several heads help

One head gives each query one weight distribution and one averaged output. Suppose a token needs two things: the identity of the previous token, and the subject several words back. A single head must split its weight between the two and average their values into the same channels. With several heads, each query has $h$ independent distributions, each with its own similarity $W_Q^{(i)} W_K^{(i)\top}$, and the results land in separate channel blocks of the concatenation, so they are not averaged together. The cost is that each head's score matrix has rank at most $d_h$; typical head widths are 64 (GPT-2, BERT, ViT-B) or 128 (Llama 2). Heads are also partly redundant: Michel et al. (2019) removed a large fraction of heads from trained models at test time with little loss in accuracy.

### What the output projection does

Split $W_O$ into $h$ blocks of rows, $W_O^{(i)} \in \mathbb{R}^{d_h \times D}$. Block matrix multiplication gives

$$
\operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O = \sum_{i=1}^{h} \text{head}_i\, W_O^{(i)} .
$$

So $W_O$ maps each head's $d_h$-dimensional result back into the full $D$-dimensional residual stream, and the heads' contributions add. Without it, head $i$ could write only into its own $d_h$ channels. Per head, the value-to-output map $W_V^{(i)} W_O^{(i)}$ is a $D \times D$ matrix of rank at most $d_h$ (Elhage et al., 2021, call it the head's OV circuit, and $W_Q^{(i)} W_K^{(i)\top}$ its QK circuit). MHA is thus $h$ low-rank read-and-write operations on the residual stream.

::: {.qa asked="ByteDance, Kling, TME, Amap (Nowcoder)"}
**Why use multiple heads, and where do Q, K and V come from in causal self-attention and in cross-attention?**

Each head has its own projections, so each query gets $h$ separate attention distributions, each based on a different learned similarity, and their results go to separate channel blocks instead of being averaged into one. Splitting $D$ into $h$ heads of width $D/h$ keeps parameters ($4D^2$) and FLOPs ($4ND^2 + 2N^2D$) the same as one full-width head; the output projection then maps the concatenated heads back into the residual stream and mixes them. In causal self-attention, $Q$, $K$ and $V$ are all linear maps of the same sequence, and scores with future keys are set to $-\infty$ before the softmax. In cross-attention, $Q$ comes from the sequence being updated (decoder states, or image tokens in Stable Diffusion) and $K$, $V$ come from the conditioning sequence (encoder outputs, or text embeddings), so the output has the length of the query sequence.
:::

::: qa
**Count the parameters and FLOPs of an MHA layer with width $D$, $h$ heads and $N$ tokens. Which depend on $h$?**

Parameters: $4D^2 + 4D$. Multiply-adds: $4ND^2$ for the projections and $2N^2D$ for $QK^\top$ and $AV$; double them for FLOPs. None depends on $h$. Only the number of $N \times N$ score matrices ($h$ per sample) and the per-head rank limit $d_h$ do.
:::

::: pitfall
Reshaping $(B, N, D)$ straight to $(B, h, N, d_h)$ without the transpose gives the right shape and the wrong result: it mixes tokens and heads. Split the last axis first, `view(B, N, h, d_h)`, then transpose. After a transpose the tensor is not contiguous, so merge the heads with `.reshape` (or `.contiguous().view`).
:::

## Position information

Attention is permutation equivariant, so a transformer needs to be told where each token is. There are two broad ways: add a position vector to each token's input (absolute encodings), or change the scores so they depend on positions (RoPE, ALiBi, relative biases).

### Learned absolute embeddings

Keep a table $P_{\text{pos}} \in \mathbb{R}^{N_{\max} \times D}$ and add row $t$ to the embedding of the token at position $t$: $x_t = E[\text{token}_t] + P_{\text{pos}}[t]$. GPT-2 has 1024 positions ($1024 \cdot 768 = 786{,}432$ parameters), BERT has 512, and ViT has one per patch plus one for the class token. It is simple and works well, but it says nothing about positions beyond $N_{\max}$; for images, the table is reshaped to its 2-D grid and interpolated when the resolution changes (see [vision backbones](vision-backbones.html)).

### Sinusoidal encodings

The original transformer adds a fixed vector:

$$
\mathrm{PE}(\text{pos}, 2i) = \sin(\text{pos}\cdot\omega_i), \qquad \mathrm{PE}(\text{pos}, 2i+1) = \cos(\text{pos}\cdot\omega_i), \qquad \omega_i = 10000^{-2i/D},
$$

for $i = 0, \dots, D/2 - 1$. The wavelengths $2\pi/\omega_i$ form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. Each pair of channels is a clock hand turning at its own speed: fast pairs separate neighbouring positions and slow pairs separate distant ones.

**Claim.** For each offset $k$ there is a matrix $M_k$, the same for every position, with $\mathrm{PE}(\text{pos}+k) = M_k\, \mathrm{PE}(\text{pos})$.

**Proof.** Fix a pair $i$ and write $\omega = \omega_i$. The angle-addition formulas give

$$
\begin{aligned}
\sin\big((\text{pos}+k)\omega\big) &= \sin(\text{pos}\,\omega)\cos(k\omega) + \cos(\text{pos}\,\omega)\sin(k\omega), \\
\cos\big((\text{pos}+k)\omega\big) &= \cos(\text{pos}\,\omega)\cos(k\omega) - \sin(\text{pos}\,\omega)\sin(k\omega),
\end{aligned}
$$

which in matrix form is

$$
\begin{pmatrix} \sin((\text{pos}+k)\omega) \\ \cos((\text{pos}+k)\omega) \end{pmatrix}
= \begin{pmatrix} \cos k\omega & \sin k\omega \\ -\sin k\omega & \cos k\omega \end{pmatrix}
\begin{pmatrix} \sin(\text{pos}\,\omega) \\ \cos(\text{pos}\,\omega) \end{pmatrix}.
$$

The $2 \times 2$ matrix is a rotation by $-k\omega$ and depends on $k$ only. Stacking the $D/2$ pairs block-diagonally gives $M_k$. $\blacksquare$

**Corollary.** $\mathrm{PE}(\text{pos})^\top \mathrm{PE}(\text{pos}+k) = \sum_i \cos(k\omega_i)$, because each pair contributes $\sin a \sin b + \cos a \cos b = \cos(b - a)$. It depends only on $k$: with $D = 8$ and $k = 4$ it is 2.2666 whether pos is 0, 3 or 10.

**Limitation.** The encoding is added to the content, so a score $(x_i + p_i)^\top W_Q W_K^\top (x_j + p_j)$ mixes content and position terms; nothing forces it to depend only on $i - j$, and sinusoidal models extrapolate poorly beyond their training length (Press et al., 2022).

### Rotary position embedding (RoPE)

RoPE (Su et al., 2021) rotates queries and keys instead of adding anything, chosen so that the score depends only on the offset.

**Two dimensions, with complex numbers.** Identify a vector $a = (a_1, a_2)$ with the complex number $\tilde a = a_1 + i a_2$. Two facts:

- The real inner product is $a^\top b = \operatorname{Re}(\tilde a\, \overline{\tilde b})$, because $(a_1 + ia_2)(b_1 - ib_2) = a_1 b_1 + a_2 b_2 + i(a_2 b_1 - a_1 b_2)$.
- Multiplying by $e^{i\varphi}$ rotates by the angle $\varphi$: $e^{i\varphi}\tilde a$ corresponds to $R(\varphi)a$ with $R(\varphi) = \begin{pmatrix} \cos\varphi & -\sin\varphi \\ \sin\varphi & \cos\varphi \end{pmatrix}$.

Encode the query at position $m$ as $\tilde q\, e^{im\theta}$ and the key at position $n$ as $\tilde k\, e^{in\theta}$. Their score is

$$
\begin{aligned}
\big\langle R(m\theta)q,\ R(n\theta)k \big\rangle
&= \operatorname{Re}\Big(\tilde q\, e^{im\theta}\ \overline{\tilde k\, e^{in\theta}}\Big) && \text{(first fact)} \\
&= \operatorname{Re}\Big(\tilde q\, \overline{\tilde k}\ e^{im\theta} e^{-in\theta}\Big) && \text{(conjugate of a product)} \\
&= \operatorname{Re}\Big(\tilde q\, \overline{\tilde k}\ e^{i(m-n)\theta}\Big),
\end{aligned}
$$

which depends on $m$ and $n$ only through $m - n$. $\blacksquare$ In real form this is $q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R((n-m)\theta)\, k$, since $R(\alpha)^\top = R(-\alpha)$ and rotations compose by adding angles.

**Check by hand.** Take $q = (1, 2)$, $k = (3, 1)$, $\theta = 0.5$. Then $\tilde q\, \overline{\tilde k} = (1 + 2i)(3 - i) = 3 - i + 6i - 2i^2 = 5 + 5i$. For $(m, n) = (0, 2)$ the offset is $-2$ and the angle is $-1$ rad: $\operatorname{Re}\big((5 + 5i)(\cos 1 - i \sin 1)\big) = 5\cos 1 + 5 \sin 1 = 6.909$. For $(m, n) = (5, 7)$ the offset is the same and so is the score, 6.909. For $(5, 6)$ the angle is $-0.5$ and the score is $5(\cos 0.5 + \sin 0.5) = 6.785$. At offset 0 it is $\operatorname{Re}(5 + 5i) = 5 = q^\top k$.

**Any even dimension.** Split a head's $d$ channels into $d/2$ pairs and rotate pair $i$ by $m\theta_i$, with

$$
\theta_i = b^{-2i/d}, \qquad i = 0, \dots, d/2 - 1, \qquad b = 10000
$$

(RoFormer indexes from 1 and writes $10000^{-2(i-1)/d}$). The rotation $R_m$ is block-diagonal, and the score is a sum of the two-dimensional ones:

$$
(R_m q)^\top (R_n k) = q^\top R_{n-m}\, k = \sum_{i=0}^{d/2-1} \operatorname{Re}\Big(\tilde q_i\, \overline{\tilde k_i}\ e^{i(m-n)\theta_i}\Big).
$$

RoPE has no parameters. It is applied to $q$ and $k$ in every layer and head, after the projections, and not to $v$. Rotations preserve norms, and position 0 is the identity. With head width 128 and $b = 10000$, the fastest pair turns 1 radian per token (wavelength $2\pi \approx 6.3$ tokens) and the slowest has $\theta_{63} = 10000^{-126/128} \approx 1.15 \times 10^{-4}$, a wavelength of about 54,400 tokens. Llama 3 raises the base to 500,000, which stretches the slowest wavelength to about 2.56 million tokens. Su et al. also show that an upper bound on the score's size decays as $|m - n|$ grows.

**Longer contexts.** A model trained on $L$ tokens has seen only the angles $m\theta_i$ with $m < L$. For pairs whose wavelength exceeds $L$, positions beyond $L$ produce angles never seen in training, and quality drops. Write $s = L'/L$ for the extension factor.

- **Position interpolation** (Chen et al., 2023) replaces $m$ by $m/s$, so every angle stays inside the trained range. Chen et al. extended LLaMA models to 32,768 tokens with at most 1000 fine-tuning steps. The downside is that it slows the fast pairs too: neighbouring tokens now differ by $\theta_0/s$ instead of $\theta_0$, which makes local order harder to read.
- **NTK-aware scaling** (proposed online by "bloc97" in 2023 and written up in the YaRN paper, Peng et al., 2023) changes the base instead: $b' = b \cdot s^{d/(d-2)}$. Then $\theta_i' = b'^{-2i/d} = \theta_i \cdot s^{-2i/(d-2)}$. The fastest pair ($i = 0$) is unchanged, the slowest ($i = d/2 - 1$, exponent $-1$) is divided by exactly $s$ as in position interpolation, and the pairs in between are scaled geometrically. With $d = 128$ and $s = 4$, $b' \approx 40{,}890$; $\theta_1$ moves by only 2.2% while the slowest frequency moves by $4\times$.
- **Refinements.** YaRN treats frequency bands differently and adds an attention temperature; Code Llama raised the base to $10^6$; Llama 3 trains with base 500,000.

### ALiBi

ALiBi (Press et al., 2022) uses no position vectors at all. It adds a fixed penalty proportional to distance to each causal score:

$$
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} - m_{\text{head}} \cdot (i - j), \qquad j \le i .
$$

The slopes are a geometric sequence across heads: for 8 heads, $1/2, 1/4, \dots, 1/256$; in general $2^{-8k/h}$ for $k = 1, \dots, h$. In the head with slope $1/2$, a key 10 tokens back loses 5 in score, a factor $e^{-5} \approx 0.0067$ in weight compared with an identical key at distance 0. Steep heads look locally and shallow heads look far. The design goal was to train on short sequences and run on longer ones, and Press et al. found that ALiBi extrapolates beyond its training length better than sinusoidal encodings. It builds in a recency bias, which suits language but limits retrieval from far back.

T5 uses a related idea: a learned scalar per head for each bucket of relative distance, added to the logits.

### Two-dimensional RoPE for images

A token at grid position $(r, c)$ needs two coordinates. Axial 2-D RoPE splits each head's channels in half: pairs in the first half rotate by $r\theta_i$ and pairs in the second half by $c\theta_i$. Applying the proof above to each half, the score depends only on $(r - r', c - c')$. Video models add a third group for time, as in Qwen2-VL's M-RoPE (see the [question bank](interview-question-bank.html)), and FLUX applies RoPE to its image-token coordinates. Axial pairs cannot represent diagonal directions; RoPE-Mixed (Heo et al., 2024) learns frequencies that mix the axes (see [vision backbones](vision-backbones.html)). DiT itself adds fixed 2-D sine-cosine embeddings at the input, half the channels for rows and half for columns.

::: {.qa asked="Alibaba Amap, TME, ByteDance (Nowcoder)"}
**Compare sinusoidal positional encodings with RoPE.**

Sinusoidal encodings add a fixed vector to the input embedding, pair $i$ being $(\sin \text{pos}\,\omega_i, \cos \text{pos}\,\omega_i)$ with $\omega_i = 10000^{-2i/D}$. A shift by $k$ is a fixed rotation of each pair, so relative position is linearly accessible, but position is mixed into the content once at the bottom and the scores contain content–position cross terms. RoPE instead rotates each two-dimensional pair of $q$ and $k$ by $m\theta_i$ in every layer, and since $R_m^\top R_n = R_{n-m}$, the score $q^\top R_{n-m} k$ depends on positions only through the offset. Neither has parameters. RoPE does not touch the values, and it extends to longer contexts by rescaling positions (position interpolation) or the base (NTK-aware scaling, YaRN), usually with brief fine-tuning. For images and video, the rotary channels are split across axes (2-D RoPE, M-RoPE).
:::

::: {.qa asked="Company unnamed (via AIGC interview repo)"}
**Implement 1D rotary position embedding (RoPE).**

Pair adjacent channels $(2i, 2i+1)$ as a complex number and multiply by $e^{im\theta_i}$, which is exactly the derivation above:

```python
import torch

def rope(x, pos, base=10000.0):
    # x: (..., T, d) with d even; pos: (T,) absolute positions (offset them when using a KV cache)
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)        # (d/2,)
    rot = torch.polar(torch.ones_like(theta), pos.double()[:, None] * theta)  # e^{i m theta_i}, (T, d/2)
    xc = torch.view_as_complex(x.double().reshape(*x.shape[:-1], d // 2, 2))  # pairs -> complex
    return torch.view_as_real(xc * rot).flatten(-2).to(x.dtype)
```

Checked: for random $q, k \in \mathbb{R}^{64}$, the score at positions $(3, 10)$ equals the score at $(103, 110)$ and differs from $(3, 11)$, and the norm is preserved. Apply it per head to $q$ and $k$ after the projections, never to $v$; with a KV cache, rotate keys with their absolute positions before caching them; compute the angles in float32 or float64, since large positions lose precision in half precision. Open-source LLM code often pairs channel $i$ with $i + d/2$ ("rotate half") instead; the two conventions differ by a fixed channel permutation, so a pretrained checkpoint only works with its own convention.
:::

::: qa
**Why does position interpolation need fine-tuning, and what does NTK-aware scaling change?**

Interpolation divides every angle by $s$, so all angles stay in the trained range, but the fast pairs now turn $s$ times more slowly, so neighbouring tokens look closer together than the model has ever seen; a short fine-tune lets it adapt. NTK-aware scaling raises the base to $b\,s^{d/(d-2)}$, which leaves the fastest frequency unchanged, divides the slowest by exactly $s$, and scales the rest geometrically in between. Local order is preserved and only the slow, long-range pairs are interpolated, so it degrades less without fine-tuning.
:::

## The transformer block

### The residual stream

Each token carries one $D$-dimensional vector through the whole network. Every sub-layer reads that vector (through a normalization and a projection) and writes back by addition. In the now-standard pre-LN form, one block is

$$
\begin{aligned}
h &= x + \operatorname{MHA}\big(\mathrm{LN}_1(x)\big), \\
x' &= h + \operatorname{MLP}\big(\mathrm{LN}_2(h)\big).
\end{aligned}
$$

Unrolling $L$ blocks, the final vector is the input embedding plus the sum of every sub-layer's output, $x_L = x_0 + \sum_{\ell} \big(\text{attn}_\ell + \text{mlp}_\ell\big)$. Elhage et al. (2021) call this shared channel the residual stream. Attention is the only operation that moves information between positions; the MLP, the normalizations and the additions act on each token separately. (PaLM and ViT-22B run the two sub-layers in parallel, $x' = x + \operatorname{MLP}(\mathrm{LN}(x)) + \operatorname{MHA}(\mathrm{LN}(x))$, so their input projections can be fused.)

### The MLP

The MLP (also called the feed-forward network) is

$$
\operatorname{MLP}(u) = \phi(uW_1 + b_1)\, W_2 + b_2, \qquad W_1 \in \mathbb{R}^{D \times d_{\text{ff}}},\ W_2 \in \mathbb{R}^{d_{\text{ff}} \times D},
$$

with $d_{\text{ff}} = 4D$ in most models, giving $8D^2 + 5D$ parameters, two thirds of the block's weights. The factor 4 comes from the original transformer ($D = 512$, $d_{\text{ff}} = 2048$) and is a convention more than an optimum: Kaplan et al. (2020) found that, at a fixed number of non-embedding parameters, loss changes by only a few percent over a wide range of shapes, including the ratio $d_{\text{ff}}/D$.

**GELU.** BERT, GPT-2 and ViT use $\phi(u) = \mathrm{GELU}(u) = u\,\Phi(u)$, where $\Phi$ is the standard normal CDF. It is smooth, close to ReLU for large $|u|$, and slightly negative for negative inputs (its minimum is about $-0.17$ at $u \approx -0.75$). A tanh approximation, $0.5u\big(1 + \tanh(\sqrt{2/\pi}\,(u + 0.044715u^3))\big)$, is common in code.

**SwiGLU.** Llama, PaLM and many recent models use a gated MLP:

$$
\operatorname{MLP}(u) = \big(\operatorname{SiLU}(uW_1) \odot uW_3\big)\, W_2, \qquad \operatorname{SiLU}(z) = z\,\sigma(z),
$$

where $\odot$ is the elementwise product and $W_3 \in \mathbb{R}^{D \times d_{\text{ff}}}$ is a second input projection that acts as a learned gate. With three matrices the MLP has $3Dd_{\text{ff}}$ weights. Matching the $8D^2$ of a standard MLP needs

$$
3Dd_{\text{ff}} = 8D^2 \quad\Longrightarrow\quad d_{\text{ff}} = \frac{8D}{3} = \frac{2}{3}\cdot 4D .
$$

For Llama 7B, $\tfrac23 \cdot 4 \cdot 4096 = 10{,}922.7$, rounded up to a multiple of 256 gives the actual $d_{\text{ff}} = 11{,}008$, and the MLP has $3 \cdot 4096 \cdot 11{,}008 = 135{,}266{,}304$ weights against $8 \cdot 4096^2 = 134{,}217{,}728$. Shazeer (2020) found that GLU variants lowered perplexity at matched parameter count (shrinking T5-Base's $d_{\text{ff}}$ from 3072 to 2048 for the three-matrix versions) and offered no theory for why.

### LayerNorm and RMSNorm

Both normalize the $D$ features of one token. LayerNorm (Ba et al., 2016) centers and scales:

$$
\mu = \frac1D \sum_{j=1}^D x_j, \qquad \sigma^2 = \frac1D \sum_{j=1}^D (x_j - \mu)^2, \qquad
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta ,
$$

with learned $\gamma, \beta \in \mathbb{R}^D$: $2D$ parameters. RMSNorm (Zhang and Sennrich, 2019) skips the centering and the shift:

$$
\mathrm{RMS}(x) = \sqrt{\frac1D \sum_{j=1}^D x_j^2 + \epsilon}, \qquad \mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)},
$$

with $D$ parameters and slightly less work.

**Worked example.** For $x = (1, 2, 3, 6)$ with $\gamma = 1$, $\beta = 0$, $\epsilon = 0$: $\mu = 3$, $\sigma^2 = (4 + 1 + 0 + 9)/4 = 3.5$, so $\mathrm{LN}(x) = (-2, -1, 0, 3)/1.8708 = (-1.069, -0.535, 0, 1.604)$, which has mean 0 and variance 1. $\mathrm{RMS}(x) = \sqrt{(1 + 4 + 9 + 36)/4} = \sqrt{12.5} = 3.536$, so $\mathrm{RMSNorm}(x) = (0.283, 0.566, 0.849, 1.697)$, which has mean square 1 but keeps its nonzero mean.

**Why not BatchNorm.** Per-token statistics do not depend on the batch, on padding, or on other positions. Batch statistics computed across positions would let a causal model see future tokens, would be noisy for small or variable-length batches, and would differ between training and inference. Normalization layers in general are compared on [vision backbones](vision-backbones.html).

### Pre-LN versus post-LN

The original transformer normalized after the residual addition (post-LN); GPT-2 and nearly all recent models normalize the input of each sub-layer (pre-LN):

$$
\text{post-LN:}\quad x_{\ell+1} = \mathrm{LN}\big(x_\ell + F_\ell(x_\ell)\big), \qquad\qquad
\text{pre-LN:}\quad x_{\ell+1} = x_\ell + F_\ell\big(\mathrm{LN}(x_\ell)\big),
$$

where $F_\ell$ is an attention or MLP sub-layer. Pre-LN adds one final LN before the output head.

**The gradient path in pre-LN.** Write $G_\ell(x) = F_\ell(\mathrm{LN}(x))$ and unroll from layer $\ell$ to the top:

$$
x_L = x_\ell + \sum_{k=\ell}^{L-1} G_k(x_k)
\qquad\Longrightarrow\qquad
\frac{\partial x_L}{\partial x_\ell} = I + \sum_{k=\ell}^{L-1} \frac{\partial G_k(x_k)}{\partial x_\ell}.
$$

The identity term carries the gradient that arrives at $x_L$ straight down to every layer, without passing through any weight matrix or normalization.

**The gradient path in post-LN.** By the chain rule,

$$
\frac{\partial x_{\ell+1}}{\partial x_\ell} = J_{\mathrm{LN}}\big(x_\ell + F_\ell(x_\ell)\big)\Big(I + \frac{\partial F_\ell}{\partial x_\ell}\Big),
\qquad
\frac{\partial x_L}{\partial x_\ell} = \prod_{k=\ell}^{L-1} J_{\mathrm{LN}}(\cdot)\Big(I + \frac{\partial F_k}{\partial x_k}\Big).
$$

Every path from the top to layer $\ell$ goes through $L - \ell$ LayerNorm Jacobians. With $\gamma = 1$ and $y = x - \mu\mathbf{1}$, Xiong et al. (2020) write this Jacobian as

$$
J_{\mathrm{LN}}(x) = \frac{\sqrt D}{\|y\|}\left(I - \frac{yy^\top}{\|y\|^2}\right)\left(I - \frac{\mathbf{1}\mathbf{1}^\top}{D}\right),
$$

which removes the mean and the radial direction and scales by $\sqrt{D}/\|y\|$: LayerNorm divides the gradient by the size of its input. In post-LN the gradient is rescaled this way at every layer and nothing bypasses it. Xiong et al. prove that at initialization the gradient of the last layer's MLP weights has norm $O(d\sqrt{\ln d})$ in post-LN, independent of depth, and their further analysis and measurements indicate that post-LN gradients are large near the output and shrink toward the input. In pre-LN the residual stream's squared norm grows linearly with depth (between $(1 + \ell/2)d$ and $(1 + 3\ell/2)d$ in expectation), so the final LN divides by roughly $\sqrt{L}$ and the same gradient is $O(d\sqrt{\ln d / L})$.

**Why post-LN needs warmup.** Large gradients in the top layers combined with a large learning rate give large, destabilizing updates. A warmup that starts the learning rate near zero and raises it over a few thousand steps keeps updates small until training leaves that regime; the original transformer used 4000 warmup steps. Xiong et al. trained pre-LN transformers without warmup and matched the baselines.

**The trade-off.** In pre-LN the stream keeps growing, so a late layer adds a small vector to a large one and deep layers can contribute relatively little. Liu et al. (2020) argue that post-LN's heavy dependence on its residual branches makes training unstable, while a light dependence (as in pre-LN) can limit final quality. Current LLMs and DiTs use pre-norm, often RMSNorm, sometimes with QK-norm (see "Training stability"). GPT-2 also scales the residual layers' output weights by $1/\sqrt{N_{\text{res}}}$ at initialization ($N_{\text{res}}$ residual layers), because each addition to the stream adds variance.

### Dropout

The original transformer applies dropout ($P = 0.1$ in the base model) to each sub-layer's output before the residual addition and to the sum of embeddings and position encodings: in pre-LN form, $x + \operatorname{Dropout}(F(\mathrm{LN}(x)))$. BERT and GPT-2 also drop attention weights after the softmax. Large-scale LLM pretraining, which sees each token about once, often uses none: PaLM was pretrained without dropout and used 0.1 for fine-tuning.

### Final norm, output head and weight tying

In pre-LN the stream is never normalized inside the residual path, so a final LN comes before the head. The output head maps each final vector $\hat x_t \in \mathbb{R}^D$ to $V$ logits, $z_t = W_U \hat x_t$ with $W_U \in \mathbb{R}^{V \times D}$, and a softmax gives the next-token distribution. Weight tying sets $W_U = E$, the input embedding table (Press and Wolf, 2017; the original transformer shares it too). The argument is that both map between tokens and the same $D$-dimensional space. It saves $VD$ parameters: $38{,}597{,}376$ of GPT-2 small's $124{,}439{,}808$, about 31%. Llama keeps the two matrices separate.

::: {.qa asked="Amazon AS, NVIDIA, Qualcomm, ByteDance, Taotian (5)"}
**Walk through the transformer architecture end to end. Why does it capture long-range dependencies better than an RNN?**

For a decoder-only language model: token IDs $(B, N)$ index an embedding table to give $(B, N, D)$; position is added (learned or sinusoidal vectors) or applied inside attention (RoPE). Then $L$ identical pre-LN blocks: $h = x + \operatorname{MHA}(\mathrm{LN}(x))$ with a causal mask, then $x' = h + \operatorname{MLP}(\mathrm{LN}(h))$ with a $4D$-wide hidden layer. A final LN and a linear head (often tied to the embedding) give $V$ logits per position, trained with cross-entropy against the input shifted by one, all positions in parallel. Encoder-only models (BERT, ViT) drop the causal mask; encoder–decoder models add cross-attention from a causal decoder to a bidirectional encoder. Long range: in an RNN, information from token $i$ reaches $j$ through $j - i$ state updates, so the gradient is a product of $j - i$ Jacobians that tends to vanish or explode, and everything must pass through one fixed-size state. In self-attention any two positions are connected in one layer (path length $O(1)$), lookup is by content rather than distance, and past tokens are kept individually rather than compressed. Costs: $O(N^2 D)$ attention compute, a KV cache that grows with context, and no built-in order, so positions must be encoded.
:::

::: {.qa asked="Meta research intern/RS, Glassdoor (topic only)"}
**What are the main architectural choices in Llama, and why were they made?**

Llama is a decoder-only transformer assembled from changes that had worked elsewhere. Pre-normalization with RMSNorm: normalizing each sub-layer's input keeps a clean identity path, which makes deep stacks train more stably, and RMSNorm is cheaper than LayerNorm. SwiGLU MLPs with width $\tfrac23 \cdot 4D$ (11,008 for $D = 4096$): gated MLPs gave lower perplexity than GELU MLPs at equal parameter count. RoPE in every layer instead of absolute position embeddings, so scores depend on relative offsets; Llama 3 raises the base to 500,000 for long contexts. No bias terms. Untied input and output embeddings. Grouped-query attention in Llama 2's 34B and 70B models and in all Llama 3 models, which shrinks the KV cache and speeds up decoding. The other distinctive decision was data: training relatively small models on far more tokens than compute-optimal (Llama 2 used 2T tokens), so they are cheap to serve.
:::

::: {.qa asked="Google intern, Glassdoor (role unclear)"}
**Compare BatchNorm, LayerNorm, InstanceNorm and GroupNorm.**

All compute $(x - \mu)/\sqrt{\sigma^2 + \epsilon}$ followed by a learned per-channel scale and shift; they differ in which axes the statistics are pooled over. For an image tensor $(B, C, H, W)$: BatchNorm pools over $(B, H, W)$ for each channel, so it depends on the batch, keeps running averages for inference, and suffers with small batches. LayerNorm pools over $(C, H, W)$ for each sample; in transformers it pools over the $D$ features of each token, independent of batch and of other positions, and behaves the same in training and inference. InstanceNorm pools over $(H, W)$ for each sample and channel, which removes per-image contrast and style (useful in style transfer). GroupNorm pools over $(H, W)$ and a group of channels for each sample, between LayerNorm (one group) and InstanceNorm (one channel per group), and is the usual choice in diffusion U-Nets with small per-GPU batches. Each usually learns a scale and shift per channel, $2C$ parameters ($2D$ for a transformer's LayerNorm; RMSNorm has $D$); PyTorch's `nn.LayerNorm` over $(C, H, W)$ would instead learn one pair per element. Transformers use LayerNorm or RMSNorm because sequences vary in length, batches can be small, and causal models must not share statistics across positions.
:::

::: qa
**Why does a post-LN transformer need learning-rate warmup, and why does pre-LN need it less?**

In pre-LN, $x_L = x_\ell + \sum_k G_k(x_k)$, so $\partial x_L/\partial x_\ell$ contains the identity and the gradient reaches every layer directly; the growing stream also makes the final LN shrink gradients by about $1/\sqrt{L}$. In post-LN every path passes through a LayerNorm Jacobian at every layer, which rescales the gradient by $\sqrt D/\|\text{input}\|$, and Xiong et al. show that at initialization the top layers' gradients are large ($O(d\sqrt{\ln d})$, independent of depth). A full learning rate on those gradients destabilizes training, so warmup starts with small steps. Pre-LN trains without warmup in their experiments, though warmup is still common for other reasons, such as Adam's noisy early moment estimates.
:::

::: key
**How to say it in an interview.** "A block reads from and writes to the residual stream: $x + \operatorname{MHA}(\mathrm{LN}(x))$, then the same with an MLP that is four times wider, or a SwiGLU with width $\tfrac23 \cdot 4D$. Attention is the only place tokens exchange information; everything else is per token. Normalization is per token, LayerNorm or RMSNorm, because batch statistics would mix positions and samples. Pre-LN keeps an identity path from the loss to every layer, so it trains stably; post-LN sends the gradient through a LayerNorm Jacobian at every layer, which makes the top layers' gradients large at initialization and is why it needs warmup. At the end: a final norm and a linear head to the vocabulary, often tied to the input embedding."
:::

::: pitfall
Two easy bugs: forgetting the final LayerNorm in a pre-LN model, so the head sees the unnormalized, depth-growing stream; and normalizing over the wrong axis. `nn.LayerNorm(D)` normalizes the last axis, which is correct for $(B, N, D)$ tokens but not for $(B, C, H, W)$ feature maps, which need a permute first or GroupNorm.
:::

## Three families and their training objectives

The block above is used in three arrangements.

### Encoder-only: BERT

All attention is bidirectional, so the model sees the whole input at every layer. BERT (Devlin et al., 2019) is pretrained with masked language modeling (MLM): choose 15% of positions; replace 80% of those with a `[MASK]` token, 10% with a random token and leave 10% unchanged; predict the original tokens with cross-entropy at the chosen positions only. The 80/10/10 split exists because `[MASK]` never appears at fine-tuning time, and the model cannot tell which tokens were altered, so it must keep a good representation of every token. BERT-Base has 12 layers, $D = 768$, 12 heads and 110M parameters; BERT-Large has 24 layers, $D = 1024$, 16 heads and 340M. Only 15% of positions give a loss per pass, and MLM does not define a left-to-right factorization, so BERT is not a text generator. Encoders are used for classification, retrieval and embeddings; ViT and DiT are encoder-style stacks too, trained with other objectives.

### Decoder-only: GPT

All attention is causal. The loss is next-token cross-entropy averaged over positions:

$$
\mathcal{L}(\theta) = -\frac1N \sum_{t=1}^{N} \log p_\theta(x_{t+1} \mid x_{\le t}),
$$

computed in one pass thanks to the causal mask. Feeding the true prefix during training is called teacher forcing; at inference the model conditions on its own samples, a distribution it never trained on (exposure bias). **Perplexity** is $\exp(\mathcal{L})$, the exponential of the mean negative log-probability.

**Worked example.** Suppose the model assigns probabilities 0.5, 0.25, 0.125 and 0.5 to the correct next tokens at four positions. The negative log-probabilities are $\ln 2, \ln 4, \ln 8, \ln 2 = 0.693, 1.386, 2.079, 0.693$; the mean is 1.213 nats, and the perplexity is $e^{1.213} = 3.36$, which equals $2^{7/4}$. The model is on average as uncertain as a uniform choice among 3.36 tokens. At initialization the logits are near zero, the prediction is near uniform, and the loss should be close to $\ln V$, which is $\ln 50{,}257 = 10.82$ for GPT-2. An initial loss far above that points to an initialization bug.

### Encoder–decoder: the original transformer and T5

A bidirectional encoder reads the source; a causal decoder generates the target, cross-attending to the encoder's outputs in every block. The loss is cross-entropy on the target tokens given the source. T5 (Raffel et al., 2020) casts every task as text in, text out, and pretrains with span corruption: drop about 15% of the tokens in spans (mean length 3), replace each span with a sentinel token, and train the decoder to output the dropped spans. Its encoder is reused as a frozen text encoder by text-to-image models: Imagen and Stable Diffusion 3 use T5-XXL encoders, and FLUX uses T5-XXL token embeddings.

### When each is used

| Family | Attention | Training objective | Typical uses |
|---|---|---|---|
| Encoder-only | bidirectional | masked tokens (BERT); supervised or self-supervised targets (ViT); diffusion loss (DiT) | embeddings, retrieval, classification, dense prediction, diffusion backbones |
| Decoder-only | causal | next-token cross-entropy | LLMs; autoregressive models of image or audio tokens |
| Encoder–decoder | bidirectional encoder, causal decoder with cross-attention | cross-entropy on target given source | translation, summarization; source of text encoders for diffusion |

Decoder-only models became the default for LLMs: one objective on raw text gives a training signal at every position, generation is built in, and the KV cache makes serving efficient.

::: {.qa asked="Apple ML research intern (NLP team), Jointaro 2023"}
**Explain BERT: its architecture, pretraining objectives, and how it differs from GPT.**

BERT is a transformer encoder: bidirectional attention, learned position embeddings up to 512, WordPiece tokens (30,000 vocabulary), a `[CLS]` token in front and `[SEP]` between segments, plus segment embeddings. BERT-Base: 12 layers, width 768, 12 heads, 110M parameters; Large: 24, 1024, 16, 340M. Pretraining: masked language modeling (15% of positions; 80% `[MASK]`, 10% random, 10% unchanged; predict the originals) and next-sentence prediction, which RoBERTa later found could be removed without loss. Fine-tuning adds one output layer on `[CLS]` or on every token. GPT is a decoder with a causal mask trained on next-token prediction: every position is a target and generation is native, while BERT's bidirectional context gives stronger representations per parameter for understanding tasks but no left-to-right generative model.
:::

::: qa
**What is perplexity, and what loss should a freshly initialized language model have?**

Perplexity is $\exp$ of the mean next-token cross-entropy, the geometric mean of $1/p(\text{correct token})$: the effective number of equally likely choices. A fresh model with near-zero logits predicts nearly uniformly, so its loss should be about $\ln V$ (10.82 nats for $V = 50{,}257$) and its perplexity about $V$. Much higher means the logits start too large.
:::

## Counting, with worked examples

Every number in this section was checked in Python.

### Parameters

Per block, ignoring nothing:

- attention: $4D^2 + 4D$ (four $D \times D$ matrices and their biases);
- MLP: $D \cdot 4D + 4D + 4D \cdot D + D = 8D^2 + 5D$;
- two LayerNorms: $4D$.

That is $12D^2 + 13D$ per block. Add $VD$ for the token embedding, $N_{\max} D$ for learned positions, $2D$ for the final LayerNorm, and nothing for a tied head.

**GPT-2 small** ($L = 12$, $D = 768$, $V = 50{,}257$, context 1024, tied embeddings):

| Part | Formula | Count |
|---|---|---|
| Token embedding | $VD$ | 38,597,376 |
| Position embedding | $1024\,D$ | 786,432 |
| One block | $12D^2 + 13D = 7{,}077{,}888 + 9{,}984$ | 7,087,872 |
| 12 blocks | | 85,054,464 |
| Final LayerNorm | $2D$ | 1,536 |
| Output head | tied | 0 |
| **Total** | | **124,439,808** |

Instantiating Hugging Face's default `GPT2LMHeadModel` gives the same 124,439,808. (The GPT-2 paper's table lists 117M for this model.) The same formula gives 354.8M, 774.0M and 1,557.6M for GPT-2 medium ($L = 24$, $D = 1024$), large ($36$, $1280$) and XL ($48$, $1600$).

**Llama 2 7B** ($L = 32$, $D = 4096$, SwiGLU with $d_{\text{ff}} = 11{,}008$, $V = 32{,}000$, untied, no biases, RMSNorm): per block $4D^2 + 3Dd_{\text{ff}} + 2D = 67{,}108{,}864 + 135{,}266{,}304 + 8{,}192 = 202{,}383{,}360$; times 32 is 6,476,267,520; adding $2VD = 262{,}144{,}000$ for the separate embedding and head and 4,096 for the final norm gives 6,738,415,616, the "7B".

### FLOPs per token in the forward pass

A product of an $m \times k$ and a $k \times n$ matrix costs $mkn$ multiply-adds, which is $2mkn$ FLOPs. Applying a weight matrix to one token uses each weight once, so the weight matrices cost **2 FLOPs per parameter per token**. The embedding lookup is indexing and costs nothing, but a tied output head is a matrix product of $2VD$ FLOPs. Attention adds a term that depends on context: per layer, one token's query is dotted with $M$ keys and its weights average $M$ values, $2MD$ multiply-adds or $4MD$ FLOPs. In a causal model token $t$ sees $t$ keys, so averaged over a sequence of length $N$ this is about $2ND$ FLOPs per layer per token if masked work is skipped, which is the $2 n_{\text{layer}} n_{\text{ctx}} d_{\text{model}}$ term in Kaplan et al.'s $C_{\text{forward}} \approx 2\Psi + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{model}}$.

For GPT-2 small at $N = 1024$:

| Term | Formula | MFLOPs per token |
|---|---|---|
| Block weight matrices | $2 \times 85{,}054{,}464$ | 170.1 |
| Output head | $2 \cdot 768 \cdot 50{,}257$ | 77.2 |
| Attention, full $1024 \times 1024$ | $4 \cdot 12 \cdot 1024 \cdot 768$ | 37.7 |
| Attention, causal work only | $2 \cdot 12 \cdot 1024 \cdot 768$ | 18.9 |

So about 266 to 285 MFLOPs per token. In a small model the output head is a large share; in a 7B model it is small.

### Training compute: $C \approx 6\Psi T$

Take one linear layer $Y = XW$ with $X \in \mathbb{R}^{T \times D_{\text{in}}}$ ($T$ tokens) and $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$.

- **Forward:** $Y = XW$ costs $2TD_{\text{in}}D_{\text{out}}$ FLOPs.
- **Backward, input gradient:** $\partial \mathcal{L}/\partial X = (\partial \mathcal{L}/\partial Y)\, W^\top$ costs $2TD_{\text{out}}D_{\text{in}}$, needed to continue backpropagation.
- **Backward, weight gradient:** $\partial \mathcal{L}/\partial W = X^\top (\partial \mathcal{L}/\partial Y)$ costs $2D_{\text{in}}TD_{\text{out}}$.

The backward pass is two products the size of the forward one, so per parameter per token the cost is 2 (forward) + 4 (backward) = 6 FLOPs. Summed over all weight matrices, training on $T$ tokens costs

$$
C \approx 6\,\Psi\, T .
$$

The rule leaves out attention-score FLOPs (small while $N$ is well below $6D$, see below) and assumes no recomputation; full activation checkpointing reruns the forward pass, giving about $8\Psi T$.

**Worked example.** Llama 2 7B was trained on 2.0T tokens: $C \approx 6 \times 6.74 \times 10^9 \times 2.0 \times 10^{12} = 8.1 \times 10^{22}$ FLOPs. Meta reports 184,320 A100-80GB GPU-hours for it, which is $6.64 \times 10^8$ GPU-seconds, so each GPU sustained about $1.22 \times 10^{14}$ FLOP/s, or 39% of the A100's dense bf16 peak of 312 TFLOP/s. That ratio is the model FLOPs utilization (MFU). Turned around, the rule answers "how many tokens can I afford": $T = C/(6\Psi)$. Hoffmann et al. (2022) found compute-optimal training at roughly 20 tokens per parameter (their Chinchilla model: 70B parameters, 1.4T tokens); Llama-style models are trained far beyond that because a smaller model is cheaper to serve.

### Activation memory

Training stores intermediate activations for the backward pass. Korthikanti et al. (2022) count, per layer, with 16-bit activations, 1-byte dropout masks and no recomputation,

$$
\text{activation bytes per layer} = sbD\left(34 + \frac{5as}{D}\right),
$$

with $s$ the sequence length, $b$ the batch size and $a$ the number of heads. The attention sub-layer contributes $11sbD + 5as^2b$ (its input, $Q$, $K$, $V$, the softmax output and its dropout mask, and so on); the MLP contributes $19sbD$, mostly its $4D$-wide pre-activation and activation; the two LayerNorms contribute $4sbD$. The $5as^2b$ part is the score-sized tensors.

For GPT-2 small with $s = 1024$, $b = 1$: $34sbD$ is 25.5 MiB and $5as^2b$ is 60 MiB, so 85.5 MiB per layer and 1.0 GiB for 12 layers, per sequence. Compare the model state: mixed-precision Adam stores about 16 bytes per parameter, $124\text{M} \times 16 \approx 2.0$ GB. Activations grow with batch size and dominate quickly (about 8 GiB at $b = 8$). FlashAttention removes the $s^2$ term by never storing the scores, and activation checkpointing trades extra compute for memory; see [deep learning training](deep-learning-training.html).

### KV-cache size

During generation each layer keeps the keys and values of all previous tokens:

$$
\text{KV-cache bytes} = 2 \times L \times h_{\text{kv}} \times d_h \times N \times B \times (\text{bytes per number}),
$$

where the 2 counts K and V, and $h_{\text{kv}}$ is the number of key-value heads ($h_{\text{kv}} = h$ in standard MHA).

**Worked example.** Llama 2 7B ($L = 32$, $h_{\text{kv}} = 32$, $d_h = 128$, fp16): $2 \cdot 32 \cdot 32 \cdot 128 \cdot 2 = 524{,}288$ bytes, or 512 KiB per token. A 4096-token sequence needs 2 GiB, and a batch of 16 such sequences needs 32 GiB, more than the 13.5 GB (12.6 GiB) of fp16 weights. GPT-2 small needs $2 \cdot 12 \cdot 768 \cdot 2$ bytes = 36 KiB per token, 36 MiB for 1024 tokens.

### When does attention cost more than the MLP?

Per layer, the forward multiply-adds are $4ND^2$ for the four projections, $8ND^2$ for the MLP and $2N^2D$ for $QK^\top$ and $AV$. So

$$
2N^2D > 8ND^2 \iff N > 4D, \qquad\qquad 2N^2D > 12ND^2 \iff N > 6D .
$$

| Model width | Attention exceeds MLP at | Attention exceeds all weight matmuls at |
|---|---|---|
| $D = 768$ (GPT-2 small) | 3,072 tokens | 4,608 tokens |
| $D = 1152$ (DiT-XL) | 4,608 tokens | 6,912 tokens |
| $D = 4096$ (Llama 2 7B) | 16,384 tokens | 24,576 tokens |

A causal kernel that skips masked tiles halves the attention work and doubles these thresholds. At GPT-2's 1024-token context, attention is 22% of the block matmul FLOPs. DiT-XL/2 at 256 tokens spends $N/(6D) = 256/6912 = 3.7\%$ as much on the score products as on the weight matrices. Summing $12ND^2 + 2N^2D$ over its 28 layers gives 118.4G multiply-adds, which with the adaLN projections (0.2G) matches the paper's 118.6 "Gflops", so that figure counts multiply-adds. [Vision backbones](vision-backbones.html) works the same arithmetic for $1024 \times 1024$ images, where attention dominates.

::: {.qa asked="ByteDance 2026, Alibaba Taotian, OpenAI applied RS"}
**What is the complexity of a transformer layer, which term dominates for long sequences, and how would you change attention for longer inputs?**

Per layer with $N$ tokens and width $D$: $4ND^2$ multiply-adds for the Q, K, V and output projections, $8ND^2$ for a $4\times$ MLP, and $2N^2D$ for the scores and the weighted sum, so about $12ND^2 + 2N^2D$. The quadratic term exceeds the MLP once $N > 4D$ and all weight matmuls once $N > 6D$ (4,608 tokens at $D = 768$), and a materialized attention matrix needs $hN^2$ numbers. For longer inputs: FlashAttention keeps attention exact and removes the $N^2$ memory, but not the $N^2$ FLOPs; sliding-window or block-sparse attention, often with a few global tokens, makes cost linear in $N$; linear attention or low-rank approximations, usually at some quality cost; factorized attention for video (spatial, then temporal); RoPE scaling to stretch the trained context; and GQA or MQA when the bottleneck is the KV cache at inference. Pick by which resource is the bottleneck: compute, activation memory or cache memory.
:::

::: {.qa asked="Runway (Glassdoor 2025–26)"}
**Estimate the compute and memory needed to train a given network.**

Compute: about $6\Psi T$ FLOPs for a transformer with $\Psi$ parameters trained on $T$ tokens (2 forward, 4 backward per parameter per token), plus attention terms when the context is long; divide by achieved throughput, typically 30 to 50% of the GPU's peak. Example: Llama 2 7B on 2T tokens is $8.1 \times 10^{22}$ FLOPs, and the reported 184,320 A100-hours imply 39% of the 312 TFLOP/s bf16 peak. Memory: model state of about 16 bytes per parameter with mixed-precision Adam (bf16 weights and gradients, fp32 master weights and two moments), sharded across GPUs with ZeRO or FSDP if needed; plus activations, about $sbD(34 + 5as/D)$ bytes per layer without recomputation (1 GiB per 1024-token sequence for GPT-2 small), reduced by FlashAttention and checkpointing.
:::

## Inference

### The decoding loop

Generation from a decoder-only model:

1. **Prefill.** Run the prompt through the model in one pass and take the logits at the last position.
2. **Choose** the next token from those logits (greedy or sampling) and append it.
3. **Repeat** with the extended sequence until an end token or a length limit.

Without any caching, step $t$ reruns the whole prefix. Over $T$ generated tokens the weight matmuls then cost on the order of $\sum_t t = O(T^2)$ token-passes and attention $\sum_t t^2 D = O(T^3 D)$. With a KV cache each step processes one new token: the weight matmuls cost $2\Psi$ FLOPs per step and attention $O(tD)$ per layer.

### The KV cache: what it stores and why it is exact

Each layer keeps the keys and values of all previous positions, of shape $(B, h_{\text{kv}}, t, d_h)$, with RoPE already applied to the keys. A decoding step computes $q$, $k$ and $v$ for the new token only, appends $k$ and $v$ to the cache, and attends the single new query to all cached keys. Queries are not cached, because only the newest one is ever used, and a single newest query needs no causal mask since it may see every earlier position.

**Why it is exact.** By the induction in "Causal masking", the layer-$\ell$ vector at position $j$ depends only on tokens $\le j$. Appending new tokens does not change it, so $k_j$ and $v_j$ are the same numbers whether they were computed when $j$ was the newest token or recomputed now. Reusing them is not an approximation; results differ from recomputation only by floating-point rounding, because the kernels see different shapes. The model in "PyTorch, from scratch" checks this: cached and uncached generation pick identical tokens and their logits differ by at most $6 \times 10^{-6}$ in float32 and $10^{-14}$ in float64. The argument fails for bidirectional attention, where earlier tokens attend to later ones. A DiT, for example, cannot cache keys and values across denoising steps, because every token changes at every step.

### Prefill versus decode

**Prefill** processes the $N_p$ prompt tokens together. Its matrix products have shape $(N_p \times D)(D \times D)$, so each weight read from memory is used $N_p$ times, and the GPU is limited by arithmetic: prefill is compute-bound.

**Decode** processes one token per sequence. Each step reads every weight from high-bandwidth memory (HBM) and uses it for 2 FLOPs per sequence in the batch. In fp16 a weight is 2 bytes, so the arithmetic intensity is about $B$ FLOPs per byte for batch size $B$. An A100 delivers about 312 TFLOP/s of dense bf16 compute and about 2.0 TB/s of HBM bandwidth, a ratio of about 153 FLOPs per byte. Decoding is therefore limited by memory bandwidth until the batch reaches the low hundreds. At batch 1, a 7B model in fp16 (13.5 GB of weights) needs at least $13.5\ \text{GB} / 2.04\ \text{TB/s} \approx 6.6$ ms per token, about 150 tokens per second, however fast the arithmetic units are. Each sequence also reads its own KV cache at every step, and batching does not share that cost.

The remedies: larger batches, quantized weights and caches, smaller caches (MQA and GQA, next), paged cache allocation (PagedAttention; Kwon et al., 2023), and speculative decoding, where a small draft model proposes several tokens that the large model verifies in one pass, with an acceptance rule that leaves the output distribution unchanged (Leviathan et al., 2023).

### Multi-query and grouped-query attention

In MHA every query head has its own key and value head. Multi-query attention (MQA; Shazeer, 2019) shares a single key head and a single value head across all $h$ query heads. Grouped-query attention (GQA; Ainslie et al., 2023) sits in between: $h_{\text{kv}}$ key-value heads, each shared by a group of $h/h_{\text{kv}}$ query heads. The cache shrinks by the factor $h_{\text{kv}}/h$, and the key and value projections shrink from $D \times D$ to $D \times h_{\text{kv}}d_h$. In code, compute $k$ and $v$ with $h_{\text{kv}}$ heads and repeat (or broadcast) each across its group.

| Configuration | $L$ | $h$ | $h_{\text{kv}}$ | Cache per token (fp16) | 4096 tokens |
|---|---|---|---|---|---|
| Llama 2 7B (MHA) | 32 | 32 | 32 | 512 KiB | 2 GiB |
| Mistral 7B (GQA) | 32 | 32 | 8 | 128 KiB | 0.5 GiB |
| Same dimensions with MQA | 32 | 32 | 1 | 16 KiB | 64 MiB |
| Llama 2 70B (GQA) | 80 | 64 | 8 | 320 KiB | 1.25 GiB |
| Llama 2 70B if it used MHA | 80 | 64 | 64 | 2.5 MiB | 10 GiB |

All rows have $d_h = 128$. GQA's quality is close to MHA's, and MQA's somewhat lower. Ainslie et al. convert an existing MHA checkpoint by mean-pooling the key and value heads within each group and continuing training ("uptraining") with about 5% of the original pretraining compute.

### Temperature, top-k and top-p

With logits $z$ and temperature $\tau$, the sampling distribution is $p = \operatorname{softmax}(z/\tau)$. For $z = (2, 1, 0.5, -1)$:

| $\tau$ | $p$ |
|---|---|
| 1 | $(0.610, 0.224, 0.136, 0.030)$ |
| 0.5 | $(0.842, 0.114, 0.042, 0.002)$ |
| 2 | $(0.434, 0.264, 0.205, 0.097)$ |

Lower temperature sharpens the distribution and $\tau \to 0$ is greedy decoding; higher temperature flattens it. **Top-k** keeps the $k$ most likely tokens and renormalizes: with $k = 2$ at $\tau = 1$, $(0.731, 0.269)$. **Top-p** (nucleus sampling; Holtzman et al., 2020) keeps the smallest set of most likely tokens whose total probability reaches $p$. The cumulative sums here are 0.610, 0.834, 0.970, so $p = 0.8$ keeps two tokens and $p = 0.95$ keeps three. Top-p adapts to the model's confidence, keeping few tokens when the distribution is peaked and many when it is flat. Beam search, which keeps the $B$ best partial sequences by total log-probability, suits tasks with one right answer such as translation, and tends to produce bland, repetitive text when used for open-ended generation.

::: {.qa asked="ByteDance LLM algorithm, Nowcoder 2026"}
**What does a KV cache store, how large is it, and how do MQA and GQA change it?**

In causal decoding the keys and values of past tokens never change (the hidden state at position $j$ depends only on tokens $\le j$), so each layer caches them and every new step computes $q$, $k$, $v$ for the new token only, appends $k$, $v$, and attends one query to the cache. It is exact, not an approximation. Size: $2 \times L \times h_{\text{kv}} \times d_h \times N \times B$ numbers; Llama 2 7B in fp16 needs 512 KiB per token, 2 GiB for 4096 tokens, more than the weights for a batch of 16 long sequences. MQA shares one K/V head across all query heads (cache divided by $h$); GQA uses $h_{\text{kv}}$ shared heads (cache scaled by $h_{\text{kv}}/h$; Mistral 7B and Llama 2 70B use 8). Because decoding is bound by memory bandwidth, a smaller cache directly speeds it up and allows bigger batches, at a small quality cost.
:::

::: {.qa asked="Tencent multimodal, Nowcoder 2026"}
**How does an LLM turn hidden states into token probabilities, and how does decoding choose the next token?**

The last position's final vector goes through the final norm and the output matrix $W_U \in \mathbb{R}^{V \times D}$ (often tied to the input embedding) to give $V$ logits; $\operatorname{softmax}(z/\tau)$ turns them into a next-token distribution, with $\tau = 1$ in training. The chain rule $\log p(x_{1:n}) = \sum_t \log p(x_t \mid x_{<t})$ is why training minimizes the mean negative log-probability and why everything is done in log space. Decoding is a rule applied to that distribution: greedy takes the argmax (deterministic, can repeat itself); beam search keeps the $B$ highest-scoring partial sequences (good for translation, bland for open-ended text, needs length normalization); top-k samples from the $k$ most likely tokens; top-p samples from the smallest set reaching probability $p$, which adapts to how confident the model is. Temperature below 1 sharpens and above 1 flattens before any of these.
:::

::: qa
**Why is decoding memory-bound while prefill is compute-bound?**

In prefill each weight read from HBM multiplies all $N_p$ prompt tokens. In decoding each weight is read once per step and used for 2 FLOPs per sequence, about $B$ FLOPs per byte in fp16, while an A100 needs about 150 FLOPs per byte to be compute-bound. So a step takes roughly (weight bytes + cache bytes) / bandwidth: at least 6.6 ms for a 7B fp16 model at batch 1. Batching, quantization, GQA and speculative decoding raise the useful work per byte read.
:::

::: pitfall
Three KV-cache bugs. First, positions: the new token's position is the cache length, not 0, both for learned position embeddings and for RoPE. Second, masks: PyTorch's `F.scaled_dot_product_attention(..., is_causal=True)` aligns the triangular mask to the top-left corner when the query and key lengths differ, so a single new query with `is_causal=True` sees only the first key; pass an explicit mask, or `is_causal=False` for one query. Third, scope: the cache is valid only for causal attention, so it does not carry across the denoising steps of a bidirectional diffusion transformer.
:::

## Efficient attention

### FlashAttention

**The problem.** A plain implementation writes $S = QK^\top$ ($N \times N$) to GPU memory, reads it back for the softmax, writes $P = \operatorname{softmax}(S)$, and reads it back again to multiply by $V$. The FlashAttention paper describes the A100's memory hierarchy as 40–80 GB of HBM at 1.5–2.0 TB/s and 192 KB of on-chip SRAM per streaming multiprocessor (108 of them) at about 19 TB/s. The softmax, masking and dropout steps do few FLOPs per byte, so they are limited by HBM traffic, and the $N \times N$ matrices need $O(N^2)$ memory. FlashAttention (Dao et al., 2022) computes attention in tiles that fit in SRAM, inside one kernel, and never writes $S$ or $P$ to HBM. The only obstacle is that each row's softmax normalizer needs all of that row's scores. The online softmax removes it.

**The online softmax.** Fix one query with scores $s_1, \dots, s_M$ and values $v_1, \dots, v_M$. The target is

$$
o^\star = \frac{\sum_j e^{s_j} v_j}{\sum_j e^{s_j}} .
$$

Process the keys in blocks and keep three running quantities: the largest score so far $m$, the sum $\ell$ of $e^{s_j - m}$ so far, and the vector $o$ of $e^{s_j - m}v_j$ summed so far. Start with $m = -\infty$, $\ell = 0$, $o = 0$. For each new block $\mathcal{B}$ of keys:

$$
\begin{aligned}
m' &= \max\Big(m,\ \max_{j \in \mathcal{B}} s_j\Big), \\
\ell' &= e^{m - m'}\,\ell + \sum_{j \in \mathcal{B}} e^{s_j - m'}, \\
o' &= e^{m - m'}\,o + \sum_{j \in \mathcal{B}} e^{s_j - m'}\, v_j .
\end{aligned}
$$

After the last block, output $o/\ell$.

**Proof that it is exact.** Claim: after processing a set $\mathcal{S}$ of keys, $m = \max_{j \in \mathcal{S}} s_j$, $\ell = \sum_{j \in \mathcal{S}} e^{s_j - m}$ and $o = \sum_{j \in \mathcal{S}} e^{s_j - m}v_j$. It holds before the first block (empty sums, with $e^{-\infty} = 0$). Suppose it holds for $\mathcal{S}$ and we add $\mathcal{B}$. The new $m'$ is the maximum over $\mathcal{S} \cup \mathcal{B}$ by definition. Multiplying the old sums by $e^{m - m'}$ turns each term $e^{s_j - m}$ into $e^{s_j - m'}$, so after adding the new block's terms, $\ell'$ and $o'$ are the sums over $\mathcal{S} \cup \mathcal{B}$ relative to $m'$. At the end,

$$
\frac{o}{\ell} = \frac{\sum_j e^{s_j - m}\, v_j}{\sum_j e^{s_j - m}} = \frac{e^{-m} \sum_j e^{s_j} v_j}{e^{-m} \sum_j e^{s_j}} = o^\star,
$$

since the common factor $e^{-m}$ cancels. $\blacksquare$ Every exponent is at most 0, so nothing overflows.

**A two-block example.** One query with scores $(1, 2 \mid 4, 0)$ and scalar values $(1, 2 \mid 3, 4)$, split into two blocks of two.

- Block 1: $m = 2$; $\ell = e^{-1} + e^{0} = 0.3679 + 1 = 1.3679$; $o = 0.3679 \cdot 1 + 1 \cdot 2 = 2.3679$.
- Block 2: $m' = \max(2, 4) = 4$, so the old sums are rescaled by $e^{2-4} = 0.1353$. Then $\ell' = 0.1353 \times 1.3679 + e^{0} + e^{-4} = 0.1851 + 1 + 0.0183 = 1.2034$ and $o' = 0.1353 \times 2.3679 + 1 \cdot 3 + 0.0183 \cdot 4 = 0.3205 + 3 + 0.0733 = 3.3937$.
- Output: $3.3937 / 1.2034 = 2.8200$.

Directly: $e^{s} = (2.718, 7.389, 54.598, 1)$ with sum $65.705$, weights $(0.0414, 0.1125, 0.8310, 0.0152)$, and output $\sum_j a_j v_j = 2.8200$, the same. As a further check, $\ell' e^{m'} = 1.2034 \times 54.598 = 65.70$, the full sum of exponentials.

**Tiling and the backward pass.** The kernel loads a block of queries into SRAM and loops over key and value blocks, keeping $m$, $\ell$ and $o$ for each query row on chip; it writes only the output and one number per row, the log-sum-exp $m + \ln \ell$. The backward pass recomputes the score tiles from $Q$, $K$ and that saved statistic instead of reading a stored $P$: more FLOPs, much less memory traffic.

**What it reduces and what it does not.** HBM accesses drop from $\Omega(Nd + N^2)$ to $O(N^2 d^2 / M)$ for SRAM size $M$, and extra memory from $O(N^2)$ to $O(N)$; the paper reports attention up to 7.6 times faster on GPT-2. FLOPs stay $O(N^2 d)$, slightly more in the backward pass because of recomputation. The result is exact softmax attention up to rounding, not an approximation, so the quadratic compute remains. FlashAttention-2 and -3 improve how the work is split across the GPU and use newer hardware features; PyTorch's `F.scaled_dot_product_attention` dispatches to fused kernels of this kind when it can.

### Sliding-window and local attention

Each query attends only to the previous $W$ keys. Cost becomes $O(NWD)$ instead of $O(N^2 D)$, and the KV cache per layer is capped at $W$ entries, so it can be a rolling buffer. Information still travels further through depth: after $L$ layers a token can be influenced by roughly $L \cdot W$ earlier tokens. Mistral 7B uses $W = 4096$ with 32 layers, a theoretical span of about 131K tokens. Longformer and BigBird add a few global tokens that attend everywhere; Swin's windows are the image version (see [vision backbones](vision-backbones.html)).

### Linear attention, and why it loses sharpness

Replace $e^{q^\top k}$ by a kernel $\phi(q)^\top\phi(k)$ with a positive feature map $\phi: \mathbb{R}^{d} \to \mathbb{R}^r$ (Katharopoulos et al., 2020, use $\phi(x) = \operatorname{elu}(x) + 1$). Associativity then lets the sum over keys be computed once:

$$
\text{out}_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j)\, v_j^\top}{\sum_j \phi(q_i)^\top \phi(k_j)}
= \frac{\phi(q_i)^\top \big(\sum_j \phi(k_j) v_j^\top\big)}{\phi(q_i)^\top \big(\sum_j \phi(k_j)\big)} .
$$

The $r \times d_v$ matrix $\sum_j \phi(k_j)v_j^\top$ is shared by all queries, so the cost is $O(Nrd)$ instead of $O(N^2 d)$. In the causal case the sums become running sums, $S_t = S_{t-1} + \phi(k_t)v_t^\top$, which is a recurrent network with a fixed-size matrix state and constant memory per generated token.

It loses sharpness for three reasons. The implicit weight matrix $\phi(Q)\phi(K)^\top$ has rank at most $r$, whereas $\exp(QK^\top/\sqrt d)$ is generally full rank. Softmax weight ratios are $e^{s_i - s_j}$, exponential in the score gap, while kernel weight ratios grow only as fast as the features do, so singling out one key among thousands, such as copying an exact earlier token, is hard. And the fixed-size state must compress the whole context, like an RNN's. Quality usually trails softmax attention; recurrent designs that add gating or decay to the state narrow the gap.

::: qa
**Is FlashAttention an approximation? What exactly does it reduce?**

No. It computes the same softmax attention up to floating-point rounding: tiling plus the online softmax, which rescales the running sums by $e^{m - m'}$ whenever the running maximum changes, gives exactly $\sum_j e^{s_j}v_j / \sum_j e^{s_j}$. It reduces HBM traffic (from $\Omega(Nd + N^2)$ to $O(N^2d^2/M)$) and extra memory (from $O(N^2)$ to $O(N)$), because the $N \times N$ scores never leave on-chip SRAM and the backward pass recomputes them. Wall-clock time drops because attention was memory-bound. FLOPs stay $O(N^2 d)$.
:::

::: qa
**Derive the online softmax update and show it is exact.**

Keep the running max $m$, $\ell = \sum e^{s_j - m}$ and $o = \sum e^{s_j - m}v_j$ over the keys seen so far. For a new block, $m' = \max(m, \max_{\mathcal{B}} s_j)$, then multiply $\ell$ and $o$ by $e^{m - m'}$ and add the block's terms $e^{s_j - m'}$ and $e^{s_j - m'}v_j$. By induction the invariants hold with the new max, because the rescaling converts every old term to the new reference. At the end $o/\ell$ equals the softmax-weighted average since the common $e^{-m}$ cancels. Example: scores $(1, 2 \mid 4, 0)$ and values $(1, 2 \mid 3, 4)$ give 2.8200 both ways.
:::

::: qa
**What does linear attention gain and lose?**

The key-value summary $\sum_j \phi(k_j)v_j^\top$ is computed once, so cost is linear in $N$, and the causal form is an RNN with constant memory per step. It loses softmax's exponential sharpness and full-rank patterns (rank at most the feature dimension), so retrieving one exact token from a long context suffers.
:::

::: key
**How to say it in an interview.** "FlashAttention is exact attention with less memory traffic. Standard attention writes the $N \times N$ scores and probabilities to HBM and reads them back, which is memory-bound. FlashAttention tiles $Q$, $K$, $V$ into SRAM and uses an online softmax: a running max, a running sum of exponentials and a running weighted sum of values, both sums rescaled by $e^{m - m'}$ when the max grows. The backward pass recomputes tiles from the saved log-sum-exp. Memory becomes linear and wall-clock drops, but FLOPs stay quadratic."
:::

## Transformers in vision and generation

### ViT

ViT (Dosovitskiy et al., 2021) is the encoder above applied to image patches: patchify (196 tokens for a $224$ image with $P = 16$), prepend a learned class token, add learned position embeddings, run pre-LN blocks with bidirectional attention, and classify from the final class-token vector. ViT-B/16 ($L = 12$, $D = 768$, $h = 12$) has $12 \times 7{,}087{,}872$ block parameters plus the patch embedding, 197 position embeddings and the class token: 85.8M, about 86.6M with a 1000-class head. Its appendix shows that average-pooling the patch outputs works as well as the class token once the learning rate is retuned. With few built-in assumptions about images, ViT needs large-scale pretraining or strong augmentation to beat CNNs; Swin, ConvNeXt, DeiT and self-supervised ViTs are on [vision backbones](vision-backbones.html).

### DiT and adaLN-Zero

DiT (Peebles and Xie, 2023) is a ViT-style stack that denoises VAE latents. The input is the noisy latent, patchified to $N$ tokens (256 for a $256 \times 256$ image at $p = 2$) with fixed 2-D sine-cosine position embeddings. The timestep embedding and the class embedding are summed into a conditioning vector $c$. Every block contains an MLP that regresses six $D$-dimensional vectors from $c$, $(\beta_1, \gamma_1, \alpha_1, \beta_2, \gamma_2, \alpha_2) = \operatorname{Linear}(\operatorname{SiLU}(c))$, and uses them as follows:

$$
\begin{aligned}
h &= x + \alpha_1 \odot \operatorname{MHA}\big((1 + \gamma_1) \odot \mathrm{LN}(x) + \beta_1\big), \\
x' &= h + \alpha_2 \odot \operatorname{MLP}\big((1 + \gamma_2) \odot \mathrm{LN}(h) + \beta_2\big),
\end{aligned}
$$

with LayerNorms that have no affine parameters of their own. The same $\gamma, \beta, \alpha$ apply to every token, so this is how a global signal (timestep, class, pooled text) changes the behavior of the whole network. The reference code writes the scale as $1 + \gamma$ so that a zero output means "unchanged".

**The zero initialization.** The final linear layer of each block's adaLN MLP starts with zero weights and biases, so at initialization $\gamma = \beta = \alpha = 0$ for every input. Then $h = x + 0 \odot \operatorname{MHA}(\cdot) = x$ and $x' = h + 0 = x$: every block is exactly the identity function. DiT also zero-initializes the final output layer, so the untrained network predicts zero. Starting a deep residual network at the identity is a known help to optimization: the DiT paper cites zero-initializing the last BatchNorm scale in each ResNet block (Goyal et al.) and the last convolution of each block in diffusion U-Nets. Signals and gradients pass straight through at first, and each block learns a correction gated in by $\alpha$. Among the conditioning designs DiT compared (in-context tokens, cross-attention, adaLN and adaLN-Zero), adaLN-Zero gave the best FID throughout training at negligible extra cost.

**What learns at the first step.** With $\alpha = 0$, the gradient reaching the attention and MLP weights inside a block is multiplied by $\alpha$, so it is zero: those weights do not move at step one. The gate itself does get a gradient, $\partial \mathcal{L}/\partial \alpha_1 = (\partial \mathcal{L}/\partial h) \odot \operatorname{MHA}(\cdot)$, which is nonzero because the randomly initialized branch produces a nonzero output even while gated off. So the adaLN projections move first, $\alpha$ becomes nonzero, and from then on the branches train. LoRA's $B = 0$ and ControlNet's zero convolutions use the same trick.

**Diffusion-specific points.** Attention is bidirectional because all tokens are denoised together, so there is no causal mask and no KV cache across sampling steps. The timestep enters every block through modulation rather than as a token. DiT-XL/2 has 28 layers, $D = 1152$, 16 heads and 675M parameters, costs 118.6 Gflops per forward pass, and reached FID 2.27 on class-conditional ImageNet $256 \times 256$ with classifier-free guidance scale 1.5. See [diffusion models](diffusion-models.html) for the training objective and [text-to-image and multimodal](text-to-image-multimodal.html) for text-conditioned variants.

### Cross-attention in Stable Diffusion's U-Net

Latent diffusion (Rombach et al., 2022) conditions a U-Net on text by cross-attention: queries from the flattened U-Net features, keys and values from a text encoder's output. In Stable Diffusion v1 the text encoder is CLIP ViT-L/14's text transformer, which outputs 77 tokens of width 768. At each attention level, a "spatial transformer" block flattens the feature map $(B, C, H, W)$ to tokens $(B, HW, C)$, applies self-attention over image tokens, cross-attention to the text, and an MLP, each with a residual connection, and reshapes back. At the $64 \times 64$ level with $C = 320$:

- queries: $(B, 4096, 320)$;
- keys and values: $(B, 77, 768)$ projected by $768 \times 320$ matrices to $(B, 77, 320)$;
- with 8 heads ($d_h = 40$), the weights are $(B, 8, 4096, 77)$: each image token gets a mixture of word vectors.

That is $4096 \times 77$ scores per head instead of $4096^2$ for self-attention. Each weight row says which words an image location reads, so cross-attention maps localize words, which Prompt-to-Prompt (Hertz et al., 2022) edits. Classifier-free guidance reruns the network with an empty prompt's embeddings.

### MMDiT: joint attention in SD3 and FLUX

Stable Diffusion 3's MMDiT (Esser et al., 2024) keeps two token streams, $M$ text tokens and $N$ image tokens, each with its own weights: its own adaLN modulation, Q, K, V and output projections, and MLP. In each block both streams compute their queries, keys and values with their own weights; the two sequences are concatenated, and one attention runs over all $N + M$ tokens:

$$
\begin{pmatrix} O_c \\ O_x \end{pmatrix} = \operatorname{Attn}\left( \begin{pmatrix} Q_c \\ Q_x \end{pmatrix}, \begin{pmatrix} K_c \\ K_x \end{pmatrix}, \begin{pmatrix} V_c \\ V_x \end{pmatrix} \right).
$$

The outputs are split back and pass through each stream's own output projection and MLP. The $(M + N) \times (M + N)$ weight matrix has four blocks: image to image (self-attention), image to text (what cross-attention does), text to image and text to text. Unlike cross-attention, the text tokens are updated by the image at every layer, at a cost of $(N + M)^2$ scores per head instead of $N^2 + NM$. In SD3 the text sequence is 77 CLIP tokens (zero-padded to 4096 channels) followed by 77 T5 tokens, $154 \times 4096$ in all; pooled CLIP vectors and the timestep enter through adaLN; and RMSNorm on queries and keys keeps mixed-precision training stable at high resolution. In the paper's comparison MMDiT beat a cross-attention DiT and a DiT that concatenates text and image tokens with one shared set of weights. FLUX.1 uses MMDiT-style double-stream blocks followed by single-stream blocks with shared weights, and applies RoPE to the image-token coordinates.

::: {.qa asked="Xiaomi multimodal, image-gen intern (Nowcoder 2024)"}
**Describe the ViT architecture and how DiT adapts it for diffusion.**

ViT cuts the image into $P \times P$ patches, embeds each linearly (a convolution with kernel and stride $P$), prepends a class token, adds learned position embeddings, runs pre-LN transformer blocks with bidirectional attention, and classifies from the class token or the mean of patch tokens. DiT runs the same kind of stack on a VAE latent: a $32 \times 32 \times 4$ latent for $256$-pixel images, patch size 2, so 256 tokens, with fixed 2-D sine-cosine positions. Timestep and class embeddings are summed into $c$, and each block regresses a scale, shift and gate per sub-layer from $c$ (adaLN-Zero), with the regression zero-initialized so every block starts as the identity. A linear decoder maps each token to $p \times p \times 2C$ (noise and variance), followed by unpatchify. FID improved steadily with Gflops, by making the model larger or the patches smaller. Text-to-image variants add cross-attention (PixArt-$\alpha$) or joint attention over text and image tokens (MMDiT).
:::

::: {.qa asked="Google intern, Ant, Alibaba Amap (3 reports)"}
**What are the drawbacks of ViTs compared with CNNs, and vice versa?**

CNNs build in locality and translation equivariance, so they learn from less data, scale linearly with pixel count and form multi-scale feature pyramids naturally. ViTs build in almost none of that: they need more data, strong augmentation or distillation; attention is quadratic in the token count, which limits resolution; plain ViTs give a single coarse-scale feature map, a handicap for detection and segmentation; and learned position embeddings must be interpolated for new resolutions. In return ViTs have a global receptive field from the first layer, keep improving with more data and compute, and share the token interface with text, which simplifies multimodal and generative models such as DiT and MMDiT. Hybrids sit between (Swin's windows, ConvNeXt's modernized convolutions).
:::

::: qa
**Cross-attention or joint attention for text conditioning: what is the difference?**

Cross-attention (Stable Diffusion's U-Net, PixArt-$\alpha$) takes queries from image tokens and keys and values from a fixed text sequence: text informs the image, the text tokens never change, and the cost is $O(NM)$. Joint attention (MMDiT) concatenates text and image tokens, each projected with its own weights, and runs one attention over both, so information flows both ways and the text representation is updated layer by layer, at $(N + M)^2$ cost. SD3 found the joint design better in its comparison.
:::

::: qa
**At initialization, which parameters of a DiT block receive gradient?**

With adaLN-Zero, $\alpha = \gamma = \beta = 0$, so the block is the identity and the gradient to the attention and MLP weights, scaled by $\alpha$, is zero. The adaLN regression layer does receive gradient, because $\partial\mathcal{L}/\partial\alpha$ is the upstream gradient times the (nonzero) branch output. After one step $\alpha \ne 0$ and the branches begin to learn. The zero-initialized final layer also gets gradient, since its input is nonzero.
:::

::: key
**How to say it in an interview.** "DiT is a ViT on VAE latents: patchify a $32 \times 32 \times 4$ latent with patch size 2 into 256 tokens, run bidirectional pre-LN blocks, and decode each token back to a patch of noise. The timestep and class enter through adaLN-Zero: a zero-initialized layer maps the conditioning vector to a shift, scale and gate per sub-layer, so every block starts as the identity; the gates learn first, then the branches. For text, Stable Diffusion's U-Net uses cross-attention (image queries, text keys and values), while SD3's MMDiT concatenates text and image tokens, with separate weights per stream, into one joint attention, so they update each other."
:::

## Mixture of experts

A mixture-of-experts (MoE) layer replaces a block's MLP with $E$ expert MLPs and a router. The router is a linear map $W_r \in \mathbb{R}^{D \times E}$ followed by a softmax, $p(x) = \operatorname{softmax}(xW_r)$. Each token is sent to its top-$k$ experts and the outputs are combined:

$$
y = \sum_{i \in \operatorname{TopK}(p(x))} g_i(x)\, \mathrm{MLP}_i(x).
$$

Switch Transformer (Fedus et al., 2022) uses $k = 1$ with $g_i = p_i$, the router probability, which is what lets the router receive gradient through the output. Mixtral uses $k = 2$ with a softmax over the two selected logits, which equals renormalizing their probabilities. Each token's compute is $k$ expert MLPs no matter how large $E$ is.

**Load balancing.** Without a constraint the router tends to collapse: the experts chosen early improve, so they are chosen more. Switch adds an auxiliary loss for each MoE layer. For a batch of $T$ tokens,

$$
\mathcal{L}_{\text{aux}} = \alpha\, E \sum_{i=1}^{E} f_i P_i, \qquad
f_i = \frac{1}{T}\sum_{x} \mathbf{1}\{\arg\max p(x) = i\}, \qquad
P_i = \frac1T \sum_x p_i(x),
$$

where $f_i$ is the fraction of tokens dispatched to expert $i$ and $P_i$ the average router probability for it. The counts $f$ are not differentiable but $P$ is, and $\partial \mathcal{L}_{\text{aux}}/\partial P_i = \alpha E f_i$, so probability is pushed away hardest from the busiest experts. Under uniform routing $f_i = P_i = 1/E$, so $\sum_i f_i P_i = 1/E$ and the loss is exactly $\alpha$; the factor $E$ keeps that baseline independent of the number of experts. Switch uses $\alpha = 10^{-2}$. With $E = 4$:

| Routing | $f$ | $P$ | $\mathcal{L}_{\text{aux}}/\alpha$ |
|---|---|---|---|
| uniform | $(0.25, 0.25, 0.25, 0.25)$ | $(0.25, 0.25, 0.25, 0.25)$ | 1.00 |
| mildly skewed | $(0.4, 0.2, 0.2, 0.2)$ | $(0.4, 0.2, 0.2, 0.2)$ | 1.12 |
| collapsed | $(1, 0, 0, 0)$ | $(0.7, 0.1, 0.1, 0.1)$ | 2.80 |

**Capacity.** For fixed tensor shapes, each expert gets a fixed number of slots per batch:

$$
\text{expert capacity} = \frac{\text{tokens per batch}}{E} \times \text{capacity factor}.
$$

With 4096 tokens, 64 experts and capacity factor 1.25, each expert takes 80 tokens. Tokens beyond an expert's capacity skip it and pass to the next layer through the residual connection. A larger factor drops fewer tokens but wastes compute and communication on empty slots; Switch found factors of 1.0 to 1.25 worked well.

**Total versus active parameters.** Mixtral 8x7B has 32 layers, $D = 4096$, eight SwiGLU experts per layer with $d_{\text{ff}} = 14{,}336$, top-2 routing, GQA with 8 KV heads and $V = 32{,}000$. One expert has $3 \cdot 4096 \cdot 14{,}336 = 176.2$M weights; all experts in all layers have $8 \times 32 \times 176.2\text{M} = 45.1$B. Attention adds 41.9M per layer (1.34B total) and the embedding and head add 0.26B, so the total is about 46.7B. Each token uses two experts per layer, $2 \times 32 \times 176.2\text{M} = 11.3$B, plus the shared 1.6B: about 12.9B active parameters. The paper rounds these to 47B and 13B (not $8 \times 7 = 56$B, since attention and embeddings are shared). FLOPs per token follow the active count, about $2 \times 12.9$B per forward pass. Memory follows the total, since every expert must be stored, and at small decoding batches different tokens hit different experts, so the bytes read per step approach the total.

In practice experts are spread across GPUs (expert parallelism), which requires all-to-all exchanges of tokens; Switch computes the router in float32 for stability; and ST-MoE (Zoph et al., 2022) adds a router z-loss (coefficient 0.001) that penalizes large router logits.

::: {.qa asked="Adobe research intern, London (Glassdoor)"}
**Implement a Mixture-of-Experts layer for a transformer.**

A router gives per-token probabilities over experts; take the top $k$, renormalize their weights, run each expert only on the tokens routed to it, and scatter-add the weighted outputs. Add the Switch load-balancing loss.

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
        p = self.router(x).softmax(-1)                      # (T, E) router probabilities
        w, idx = p.topk(self.k, dim=-1)                     # (T, k) weights and expert ids
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

Checked against a per-token loop over the chosen experts (equal to $10^{-6}$), and the router receives gradient through the weights $w$. Points to mention: the loop over experts is fine in an interview, while production code sorts tokens by expert and uses grouped matrix multiplies; a capacity limit drops overflow tokens to the residual path; with $k > 1$ some implementations count all $k$ choices in $f$; under expert parallelism tokens move between GPUs with all-to-all communication.
:::

::: qa
**An MoE model has 47B parameters but 13B active. What does each number determine?**

Active parameters set the FLOPs per token (about $2 \times 13$B per forward pass) and so the training and prefill compute, similar to a 13B dense model. Total parameters set the memory needed to hold the model and, at small decoding batches, roughly the bytes read per step, since different tokens use different experts. So an MoE buys the quality of a larger model at the compute of a smaller one, paid for in memory and communication.
:::

## Training stability

Optimizers and schedules are covered in [deep learning training](deep-learning-training.html); these are the transformer-specific failure modes.

**Warmup.** Start the learning rate near zero and raise it linearly over the first few thousand steps. The original transformer used $\text{lr} = D^{-0.5}\min(\text{step}^{-0.5},\ \text{step}\cdot\text{warmup}^{-1.5})$ with 4000 warmup steps; Llama 2 used 2000 warmup steps, cosine decay to 10% of the peak, AdamW with $\beta_2 = 0.95$ and gradient clipping at 1.0. Two reasons: post-LN's large initial gradients (above), and Adam's second-moment estimate, which early on averages over few steps and can give erratic step sizes (Liu et al., 2020).

**Attention-logit growth and QK-norm.** Nothing bounds the norms of $W_Q$ and $W_K$, so attention logits can keep growing during training. Large logits make attention nearly one-hot with near-zero entropy, the softmax Jacobian vanishes, and training can diverge. ViT-22B (Dehghani et al., 2023) observed divergence after a few thousand steps in models of around 8B parameters, caused by extremely large attention logits, and fixed it by applying LayerNorm to queries and keys before the dot product:

$$
A = \operatorname{softmax}\!\left(\frac{1}{\sqrt{d}}\,\mathrm{LN}(XW_Q)\,\mathrm{LN}(XW_K)^\top\right).
$$

After normalization each query and key has a controlled norm set by learnable scales, so the logits cannot run away. SD3 used RMSNorm on queries and keys to keep bf16 mixed-precision training stable at high resolution, where it otherwise diverged unless trained in full precision at about twice the cost. Wortsman et al. (2023) reproduced this instability in small models trained at high learning rates and showed that QK-layernorm mitigates it.

**Output-logit drift and z-loss.** Softmax is shift-invariant, so the log normalizer $\log Z = \log \sum_v e^{z_v}$ of the output logits is not pinned down by the loss and can drift to large magnitudes, where bf16 rounding errors grow. PaLM adds $10^{-4} \log^2 Z$ to the loss to keep $\log Z$ near 0 and found it increased training stability. ST-MoE applies the same idea to router logits, and Wortsman et al. reproduced output-logit divergence at small scale and its fix by z-loss.

Other common measures are gradient clipping, AdamW with $\beta_2 = 0.95$, removing bias terms, and bf16 compute with fp32 master weights.

::: qa
**Your large transformer's loss spikes and diverges after a few thousand steps, and attention entropy is collapsing. What do you do?**

The attention logits are probably growing until the softmax is one-hot. Add QK-normalization (LayerNorm or RMSNorm on $q$ and $k$ per head before the dot product), which bounds the logits; check that warmup is long enough and the peak learning rate is not too high; clip gradients; consider $\beta_2 = 0.95$ and a z-loss on the output logits; and confirm the softmax runs in float32. Monitor the maximum attention logit and attention entropy per layer to confirm the diagnosis.
:::

::: qa
**What does z-loss do, given that softmax is shift-invariant?**

Because adding a constant to every logit leaves the softmax unchanged, cross-entropy cannot control the overall level of the logits, and $\log Z$ can drift far from 0, where low-precision arithmetic loses accuracy. The z-loss $10^{-4}\log^2 Z$ picks the shift that makes $\log Z \approx 0$ without changing the predicted distribution, which PaLM found stabilizes training; the router z-loss in MoE does the same for router logits.
:::

## PyTorch, from scratch

The model below has multi-head attention with causal masks, padding masks, cross-attention and a KV cache; a pre-LN block; and a tiny decoder-only model with a greedy generation loop that can run with or without the cache. The checks at the end compare the attention with PyTorch's fused kernel and the cached generation with the uncached one.

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d, h, d_ctx=None):
        super().__init__()
        assert d % h == 0
        self.h, self.dh = h, d // h
        self.wq = nn.Linear(d, d)
        self.wkv = nn.Linear(d_ctx or d, 2 * d)    # K and V projections fused
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
        if causal:  # query i sits at absolute position M - N + i
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
        pos = torch.arange(start, start + idx.shape[1], device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.head(self.ln_f(x))                                      # (B, N, V)

@torch.no_grad()
def generate(model, idx, steps, use_cache=True):
    caches = [{} for _ in model.blocks] if use_cache else None
    logits = model(idx, caches)                                             # prefill
    seen = []
    for _ in range(steps):
        seen.append(logits[:, -1])
        nxt = logits[:, -1].argmax(-1, keepdim=True)                        # greedy
        idx = torch.cat([idx, nxt], dim=1)
        if use_cache:
            logits = model(nxt, caches, start=idx.shape[1] - 1)             # one token
        else:
            logits = model(idx)                                             # whole prefix
    return idx, torch.stack(seen, dim=1)

# ---- checks ----
# Checked with torch 2.11, float32: self-attention with causal + padding masks matches
# F.scaled_dot_product_attention to 9e-8 and cross-attention to 6e-8; cached and uncached
# greedy generation give identical tokens, logits within 6e-6 (1e-14 in float64).
torch.manual_seed(0)
B, N, M, d, h = 2, 5, 7, 32, 4
att, x = MHA(d, h), torch.randn(B, N, d)
pad = torch.zeros(B, N, dtype=torch.bool); pad[1, -2:] = True               # sample 1: 2 pad keys
q = att.wq(x).view(B, N, h, -1).transpose(1, 2)
k, v = att.wkv(x).view(B, N, 2, h, -1).permute(2, 0, 3, 1, 4)
allow = torch.ones(N, N, dtype=torch.bool).tril() & ~pad[:, None, None, :]   # True = may attend
ref = att.wo(F.scaled_dot_product_attention(q, k, v, attn_mask=allow).transpose(1, 2).reshape(B, N, d))
print("self-attn causal+pad max diff", (att(x, causal=True, key_pad=pad) - ref).abs().max().item())

catt, ctx = MHA(d, h, d_ctx=24), torch.randn(B, M, 24)
q = catt.wq(x).view(B, N, h, -1).transpose(1, 2)
k, v = catt.wkv(ctx).view(B, M, 2, h, -1).permute(2, 0, 3, 1, 4)
ref = catt.wo(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, d))
print("cross-attn max diff", (catt(x, ctx=ctx) - ref).abs().max().item())

model = TinyGPT(V=50, d=32, h=4, L=2, n_max=64).eval()
prompt = torch.randint(0, 50, (2, 6))
ids_c, lg_c = generate(model, prompt, 10, use_cache=True)
ids_u, lg_u = generate(model, prompt, 10, use_cache=False)
print("same tokens", torch.equal(ids_c, ids_u), "max logit diff", (lg_c - lg_u).abs().max().item())
```

**The key lines.**

- `self.wkv = nn.Linear(d_ctx or d, 2 * d)`: keys and values come from the context, whose width may differ from the query width; fusing them into one matrix is one matmul instead of two. Self-attention is the case `ctx=None`.
- `.view(B, -1, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)`: splits the $2D$ outputs into (K or V, head, channel), then moves to $(2, B, h, M, d_h)$, so unpacking gives `k` and `v` each of shape $(B, h, M, d_h)$.
- The cache branch concatenates new keys and values along the time axis (`dim=2`) and stores the result. Only self-attention uses it; cross-attention keys would be computed once and reused instead.
- `tril(M - N)`: the causal mask for $N$ queries at the end of $M$ keys. Query $i$ sits at absolute position $M - N + i$ and may see keys $j \le M - N + i$. In training $M = N$ and this is the usual lower triangle; during cached decoding $N = 1$ and the single row is all True. This bottom-right alignment is exactly what `is_causal=True` does not do when $N \ne M$.
- `key_pad[:, None, None, :]` broadcasts a $(B, M)$ padding mask over heads and queries. In the check, sample 1 has two padded keys at the end, so with the causal mask no row is fully masked.
- `out.transpose(1, 2).reshape(B, N, -1)` concatenates the heads before `wo`. `Block` is pre-LN; `TinyGPT` adds learned positions, ends with `ln_f`, and ties `head.weight` to the token embedding.
- `start=idx.shape[1] - 1`: the new token's position equals the number of tokens before it. Forgetting this offset is a common KV-cache bug.
- `generate` keeps the logits used at each step, so the cached and uncached runs are compared number by number, not only by the tokens they pick.

::: {.qa asked="ByteDance, NVIDIA, OpenAI, Meta, Baidu +4 (9+ reports)"}
**Implement multi-head attention in PyTorch, supporting masks and cross-attention. Then build a transformer block.**

Write the `MHA` and `Block` classes above, saying the shapes aloud: project, `view(B, N, h, d_h)`, transpose to $(B, h, N, d_h)$, scores $QK^\top/\sqrt{d_h}$ of shape $(B, h, N, M)$, fill masked scores with $-\infty$, softmax over the last axis, multiply by $V$, transpose back, reshape to $(B, N, D)$, output projection. Cross-attention is the same module with keys and values projected from the context, whose length $M$ and width may differ. Mention that a fully masked row gives NaN, that dropout on the attention weights is common in training, that the cost is $O(NMD)$ time and $O(hNM)$ memory if the scores are materialized, and that production code calls `F.scaled_dot_product_attention`, which dispatches to fused FlashAttention-style kernels. Then verify against that function on a small input, as in the checks above. The block is pre-LN, `x = x + attn(ln1(x)); x = x + mlp(ln2(x))` with a $4D$ GELU MLP; a DiT block adds condition-dependent shift, scale and gate around each sub-layer.
:::

::: {.qa asked="OpenAI RE (Aced, ~2025)"}
**Add a KV cache to a transformer decoder. What changes?**

Each self-attention layer keeps tensors $K, V$ of shape $(B, h_{\text{kv}}, t, d_h)$. Prefill runs the prompt once and fills them; each decoding step computes $q$, $k$, $v$ for the new token only, appends $k$ and $v$, and attends the one query over the whole cache. Positions must be offset by the cache length (for learned positions and for RoPE, rotating keys before caching them). The causal mask must be aligned to the bottom-right when queries are fewer than keys, and is unnecessary for a single query. Batched decoding must track per-sequence lengths and padding. Memory grows linearly with length ($2Lh_{\text{kv}}d_h$ numbers per token), so serving systems pre-allocate or page it. It is exact because causal hidden states never depend on later tokens; test it by checking that cached and uncached logits agree to numerical tolerance, as the code above does.
:::

## Rapid-fire review

::: qa
**Where do the $12D^2$ parameters per block come from?**

$4D^2$ for $W_Q, W_K, W_V, W_O$ and $8D^2$ for a $4\times$ MLP; biases and LayerNorms add $13D$.
:::

::: qa
**Which scale goes in the attention softmax with $h$ heads?**

$1/\sqrt{d_h}$, the per-head width, not $1/\sqrt{D}$.
:::

::: qa
**Write the softmax Jacobian.**

$\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$, that is $\operatorname{diag}(p) - pp^\top$; it vanishes as $p$ becomes one-hot.
:::

::: qa
**Does the number of heads change MHA's parameters or FLOPs?**

No: $4D^2$ parameters and $4ND^2 + 2N^2D$ multiply-adds for any $h$. It changes the number of score matrices and the rank of each.
:::

::: qa
**At what sequence length does attention cost more than the MLP?**

$N > 4D$; more than all weight matmuls at $N > 6D$ (3,072 and 4,608 tokens for $D = 768$).
:::

::: qa
**Where does $6\Psi T$ come from?**

2 FLOPs per parameter per token forward, 4 backward (one matmul for the input gradient, one for the weight gradient).
:::

::: qa
**KV-cache size per token for Llama 2 7B in fp16?**

$2 \cdot 32 \cdot 32 \cdot 128 \cdot 2$ bytes = 512 KiB; 2 GiB for 4096 tokens. GQA with 8 KV heads makes it 128 KiB.
:::

::: qa
**Why is RoPE relative?**

$\langle R_m q, R_n k\rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k$, because rotations compose by adding angles.
:::

::: qa
**ALiBi slopes for 8 heads?**

$1/2, 1/4, \dots, 1/256$, subtracted as $m \cdot (i - j)$ from causal scores.
:::

::: qa
**LayerNorm versus RMSNorm parameters?**

LayerNorm has $\gamma$ and $\beta$ ($2D$); RMSNorm has only $\gamma$ ($D$) and does not subtract the mean.
:::

::: qa
**Pre-LN or post-LN, in one line?**

Pre-LN keeps an identity gradient path and trains stably without much warmup; post-LN puts a LayerNorm Jacobian on every path and needs warmup.
:::

::: qa
**How much does weight tying save in GPT-2 small?**

$VD = 38.6$M parameters, about 31% of 124.4M.
:::

::: qa
**Why is the KV cache exact?**

Under a causal mask, the hidden state at position $j$ depends only on tokens $\le j$, so its keys and values never change.
:::

::: qa
**Lower bound on batch-1 decode latency for a 7B fp16 model on an A100?**

About 13.5 GB / 2.0 TB/s $\approx$ 6.6 ms per token, since every weight is read once per step.
:::

::: qa
**What does FlashAttention not reduce?**

FLOPs; they stay $O(N^2 d)$. It reduces HBM traffic and memory.
:::

::: qa
**Receptive field of sliding-window attention after $L$ layers?**

About $L \cdot W$ tokens; for Mistral 7B, $32 \times 4096 \approx 131$K.
:::

::: qa
**What makes a DiT block the identity at initialization?**

The adaLN regression is zero-initialized, so the residual gates $\alpha$ are 0.
:::

::: qa
**What is the Switch auxiliary loss under uniform routing?**

Exactly $\alpha$, because $E\sum_i (1/E)(1/E) = 1$.
:::

::: qa
**Mixtral 8x7B total and active parameters?**

About 46.7B total and 12.9B active per token (top-2 of 8 experts, shared attention and embeddings).
:::

::: qa
**What do QK-norm and z-loss each prevent?**

QK-norm bounds attention logits, preventing one-hot, zero-entropy attention; z-loss keeps the output log normalizer $\log Z$ near 0.
:::

::: qa
**How many tokens does ViT-B/16 process at $224 \times 224$?**

196 patches plus one class token, 197.
:::

## References

1. Vaswani et al., "Attention Is All You Need," 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
3. Radford et al., "Language Models are Unsupervised Multitask Learners," OpenAI, 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
4. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
5. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
6. Press, Smith and Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation," 2022. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
7. Xiong et al., "On Layer Normalization in the Transformer Architecture," 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
8. Shazeer, "GLU Variants Improve Transformer," 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
9. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
10. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
11. Peebles and Xie, "Scalable Diffusion Models with Transformers," 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
12. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
13. Fedus, Zoph and Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," 2022. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
14. Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models," 2022. [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)
15. Jiang et al., "Mixtral of Experts," 2024. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
16. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
17. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
18. Kaplan et al., "Scaling Laws for Neural Language Models," 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
19. Hoffmann et al., "Training Compute-Optimal Large Language Models," 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
20. Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters," 2023. [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)
21. Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities," 2023. [arXiv:2309.14322](https://arxiv.org/abs/2309.14322)
22. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," 2022. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
23. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," 2020. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
24. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," 2019. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
25. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," 2023. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
26. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," 2023. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
27. Grattafiori et al., "The Llama 3 Herd of Models," 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
28. Jiang et al., "Mistral 7B," 2023. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
29. Rozière et al., "Code Llama: Open Foundation Models for Code," 2023. [arXiv:2308.12950](https://arxiv.org/abs/2308.12950)
30. Ba, Kiros and Hinton, "Layer Normalization," 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
31. Zhang and Sennrich, "Root Mean Square Layer Normalization," 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
32. Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)," 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
33. Press and Wolf, "Using the Output Embedding to Improve Language Models," 2017. [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)
34. Liu et al., "Understanding the Difficulty of Training Transformers," 2020. [arXiv:2004.08249](https://arxiv.org/abs/2004.08249)
35. Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond," 2020. [arXiv:1908.03265](https://arxiv.org/abs/1908.03265)
36. Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer," 2022. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
37. Elhage et al., "A Mathematical Framework for Transformer Circuits," Transformer Circuits Thread, 2021. [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)
38. Michel, Levy and Neubig, "Are Sixteen Heads Really Better than One?," 2019. [arXiv:1905.10650](https://arxiv.org/abs/1905.10650)
39. Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation," 2023. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)
40. Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," 2023. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
41. Heo et al., "Rotary Position Embedding for Vision Transformer," 2024. [arXiv:2403.13298](https://arxiv.org/abs/2403.13298)
42. Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," 2022. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
43. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
44. Leviathan, Kalman and Matias, "Fast Inference from Transformers via Speculative Decoding," 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
45. Holtzman et al., "The Curious Case of Neural Text Degeneration," 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
46. Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention," 2020. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
47. Beltagy, Peters and Cohan, "Longformer: The Long-Document Transformer," 2020. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
48. Zaheer et al., "Big Bird: Transformers for Longer Sequences," 2020. [arXiv:2007.14062](https://arxiv.org/abs/2007.14062)
49. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
50. Hertz et al., "Prompt-to-Prompt Image Editing with Cross Attention Control," 2022. [arXiv:2208.01626](https://arxiv.org/abs/2208.01626)
51. Chen et al., "PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis," 2023. [arXiv:2310.00426](https://arxiv.org/abs/2310.00426)
52. Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding," 2022. [arXiv:2205.11487](https://arxiv.org/abs/2205.11487)
53. Wan Team, "Wan: Open and Advanced Large-Scale Video Generative Models," 2025. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
54. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
