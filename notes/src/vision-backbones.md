---
title: "Vision Backbones and Representations"
subtitle: "Convolution and attention arithmetic, residual and normalization layers, ViTs, self-supervised features, and how pretrained encoders now feed into generative models."
order: 6
updated: 2026-11-08
---

Every generative vision system is built on a backbone: a U-Net or transformer that denoises, a tokenizer that compresses, and often a pretrained encoder that conditions or supervises. Interviews test it in two ways. The first is counting: shapes, parameters, FLOPs, receptive fields, token counts and attention memory, done on a whiteboard without hesitation. The second is judgment: why residual connections and normalization help, what a ViT gives up compared with a CNN, and what different pretrained encoders actually encode. That last question matters more now that REPA-style alignment and representation autoencoders make pretrained features part of the generator. Training recipes live in [deep learning training](deep-learning-training.html) and generator architectures in [text-to-image and multimodal](text-to-image-multimodal.html).

## Convolution arithmetic

### Output size

Notation: input side $n$, kernel $k$, padding $p$ per side, stride $s$, dilation $d$. Dilation spaces the taps $d$ apart, so the kernel spans $d(k-1)+1$ pixels. The padded length is $n+2p$, the last valid window start is $n+2p-d(k-1)-1$, and starts come every $s$ pixels from 0:

$$
n_{\text{out}} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1 .
$$

This is PyTorch's `Conv2d` formula; with $s>1$ the floor silently drops border pixels. Example, the ResNet stem at 224: a $7\times7$ conv with $s=2, p=3$ gives $\lfloor 223/2\rfloor + 1 = 112$; the $3\times3$ max-pool with $s=2, p=1$ gives $\lfloor 111/2 \rfloor + 1 = 56$.

### Parameters and compute

A conv with $C_\text{in}$ inputs, $C_\text{out}$ outputs, a $k_h\times k_w$ kernel and $g$ groups has

$$
\#\text{params} = \frac{k_h k_w C_\text{in}}{g}\, C_\text{out} + C_\text{out}, \qquad
\text{MACs} = H_\text{out} W_\text{out} C_\text{out}\cdot \frac{k_h k_w C_\text{in}}{g},
$$

since each output element is one dot product of length $k_h k_w C_\text{in}/g$; FLOPs $\approx 2\times$ MACs. Parameters do not depend on resolution; compute does. Example: a $3\times3$ conv from 64 to 128 channels on $56\times56$ ($s=1$, $p=1$) has $9\cdot64\cdot128 = 73{,}728$ weights ($73{,}856$ with bias) and $56\cdot56\cdot128\cdot576 \approx 231$M MACs.

::: pitfall
Many papers report multiply-accumulates and call them FLOPs; say which you are counting. And a conv followed by BatchNorm needs no bias: BN's mean subtraction cancels it and $\beta$ replaces it.
:::

### Receptive field

Track the receptive field (RF) size $r$ and the jump $j$, the input-pixel distance between neighboring units of the current layer:

$$
r_l = r_{l-1} + d_l\,(k_l-1)\, j_{l-1}, \qquad j_l = j_{l-1}\, s_l, \qquad r_0 = j_0 = 1 .
$$

Each layer adds $d_l(k_l-1)$ of its own steps, each worth $j_{l-1}$ input pixels. Three stacked $3\times3$ convs give $r = 7$ with $27C^2$ weights, against $49C^2$ for one $7\times7$ (VGG's argument), plus two extra nonlinearities. ResNet stem: the $7\times7/2$ conv gives $r=7, j=2$; the $3\times3/2$ pool gives $r = 7 + 2\cdot2 = 11, j=4$; the next $3\times3$ conv gives $r = 11 + 2\cdot4 = 19$. Early strides multiply the growth of every later layer.

The theoretical RF is an upper bound. Luo et al. showed the effective receptive field (the gradient of a central output unit with respect to each input pixel) is roughly Gaussian and grows like $\sqrt{n}$ with depth $n$, while the theoretical RF grows like $n$.

### Dilated, depthwise-separable and 1x1 convolutions

**Dilated convolution** keeps the parameters but widens the extent. Stacking $3\times3$ convs with dilations $1, 2, \dots, 2^{L-1}$ gives $r = 2^{L+1}-1$: exponential RF growth at full resolution (Yu and Koltun), at the cost of gridding, where a unit sees only a sparse lattice of inputs.

**Depthwise-separable convolution** factors a $k\times k$ conv into a per-channel $k\times k$ conv ($g = C_\text{in}$) and a $1\times1$ pointwise conv. Ignoring bias,

$$
\frac{k^2 C_\text{in} + C_\text{in} C_\text{out}}{k^2 C_\text{in} C_\text{out}} = \frac{1}{C_\text{out}} + \frac{1}{k^2}.
$$

For the example above: $8{,}768$ vs $73{,}728$ parameters, a ratio of about 0.12, and the same ratio for MACs. MobileNets report 8 to 9 times less computation for $3\times3$ kernels.

**A $1\times1$ convolution** is a per-pixel linear layer across channels (Network in Network): it changes width cheaply, mixes channels after depthwise convs, and implements projection shortcuts.

### Transposed convolution and checkerboard artifacts

A transposed conv is the adjoint of a strided conv: each input pixel scatters a scaled copy of the kernel onto the output, with copies $s$ apart. Output size: $(n-1)s - 2p + d(k-1) + a + 1$ with output padding $a$, so $k=4, s=2, p=1$ maps 32 to 64. In 1D each output receives $\lfloor k/s \rfloor$ or $\lceil k/s \rceil$ contributions. When $s$ does not divide $k$ (say $k=3, s=2$) neighbors alternate between 1 and 2, so even a constant input gives a periodic output, which in 2D becomes a checkerboard (Odena, Dumoulin and Olah). Even overlap does not fully prevent it. The standard fix, used in common diffusion U-Nets and the Stable Diffusion VAE decoder, is nearest-neighbor upsampling followed by a regular conv.

::: qa
**How many parameters does a 2D conv layer have, and how much compute does it take?**

Weights $k_h k_w (C_\text{in}/g) C_\text{out}$ plus $C_\text{out}$ biases; the kernel spans every input channel of its group, which people forget. A $3\times3$ conv from 3 to 64 channels has $1{,}728 + 64 = 1{,}792$ parameters, at any image size. Compute is $H_\text{out}W_\text{out}C_\text{out}\cdot k_hk_wC_\text{in}/g$ MACs. Drop the bias when BN follows.
:::

::: qa
**What is the receptive field of a CNN unit, and how do you compute it?**

It is the input region that can influence the unit. Use $r_l = r_{l-1} + d_l(k_l-1)j_{l-1}$ with $j_l = j_{l-1}s_l$: each layer adds $d(k-1)$ steps, each the size of the product of all earlier strides. Three $3\times3$ convs give 7; an early stride-2 layer doubles what every later layer adds. The nuance: the effective RF is Gaussian-shaped and much smaller, growing like the square root of depth.
:::

::: qa
**What is a depthwise-separable convolution, and how much does it save?**

It splits a conv into per-channel spatial filtering (groups $= C_\text{in}$) and $1\times1$ channel mixing. Parameters and MACs drop to a fraction $1/C_\text{out} + 1/k^2$ of the original, close to $1/9$ for $3\times3$ kernels and wide layers. It assumes spatial and channel correlations factor (Xception, MobileNets, ConvNeXt). On GPUs the wall-clock gain is smaller than the FLOP gain, because depthwise convs do little arithmetic per byte moved.
:::

## Residual connections

A residual block computes $y = x + F(x)$. He et al. were responding to *degradation*: deeper plain networks had higher *training* error than shallower ones, so it was not overfitting, and with BatchNorm their plain nets had healthy forward signals and gradient norms. They diagnosed an optimization difficulty. A residual block makes the identity the default ($F=0$), so extra depth cannot make a good solution harder to reach.

The gradient is $\partial \mathcal{L}/\partial x = \partial \mathcal{L}/\partial y\,(I + \partial F/\partial x)$. Pre-activation ResNets move BN and ReLU inside the branch so the shortcut is a pure identity across blocks:

$$
x_L = x_l + \sum_{i=l}^{L-1} F(x_i), \qquad
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i)\right).
$$

The "1" carries the gradient from any deep layer straight to any shallow one. With it, a 1001-layer ResNet reached 4.92% error on CIFAR-10, against 7.61% with the original ordering. Pre-LN transformer blocks, $x + \text{Attn}(\text{LN}(x))$, use the same idea, and DiT's adaLN-Zero initializes each residual branch's gate to zero so the network starts as the identity. Skips also visibly smooth the loss surface (Li et al.).

**U-Net skips serve a different purpose.** Encoder features at each resolution are *concatenated* into the decoder at the same resolution, restoring high-frequency detail the bottleneck cannot carry. ResNet skips are short, additive and help optimization; U-Net skips are long and help information flow. Diffusion U-Nets use both.

::: {.qa asked="Anduril CV engineer, Glassdoor"}
**What helped ResNets avoid the vanishing-gradient problem? Why the shortcut?**

The identity shortcut. With $y = x + F(x)$ the Jacobian is $I + \partial F/\partial x$, and in pre-activation form the gradient at layer $l$ contains an unattenuated copy of the gradient at any deeper layer, so it cannot shrink multiplicatively with depth. The nuance worth adding: the original paper argued its BN-equipped plain nets did not have vanishing gradients but suffered degradation, a harder optimization problem. Shortcuts fix that by making the identity the default, so each block learns only a correction.
:::

## Normalization

Every normalizer computes $\hat x = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$ and then a learned per-channel affine $\gamma \hat x + \beta$. They differ only in which axes the statistics use. For a conv activation of shape $(N, C, H, W)$:

| Method | Statistics over | Uses the batch? | Typical home |
|---|---|---|---|
| BatchNorm | $N, H, W$, per channel | yes | CNN classifiers, large batches |
| LayerNorm | all features of one sample (per token in transformers) | no | transformers |
| InstanceNorm | $H, W$, per sample and channel | no | style transfer |
| GroupNorm | $H, W$ and $C/G$ channels, per sample | no | small-batch detection, diffusion U-Nets |
| RMSNorm | like LayerNorm, but no mean and no $\beta$ | no | LLMs, recent DiTs |

GroupNorm with $G=1$ is LayerNorm, and with $G=C$ it is InstanceNorm.

**BatchNorm** uses batch statistics in training, with gradients flowing through them, and running averages at evaluation (an EMA; PyTorch's momentum is 0.1). It was motivated as reducing "internal covariate shift", but Santurkar et al. found its benefit is better explained by a smoother loss and gradients, which permit larger learning rates. It regularizes because each sample's output depends on the random other samples in its batch. It breaks with small or non-i.i.d. batches, variable-length sequences, and contrastive learning, where batch statistics can leak which samples are positives (MoCo shuffles BN across GPUs for this reason). At batch size 2, GroupNorm has 10.6% lower ImageNet error than BN on ResNet-50.

**InstanceNorm** removes each image's per-channel mean and variance, which carry contrast and style, so it helped fast stylization. **AdaIN** substitutes another image's statistics, $\text{AdaIN}(x,y) = \sigma(y)\,\frac{x-\mu(x)}{\sigma(x)} + \mu(y)$, and StyleGAN uses it to inject style. **RMSNorm** drops re-centering, $\bar x_i = g_i\, x_i / \sqrt{\tfrac1d\sum_j x_j^2}$, keeping rescaling invariance at lower cost (7–64% faster in its paper). **adaLN** makes $\gamma, \beta$ functions of the conditioning (FiLM-style); DiT regresses them from the timestep and class embeddings, and adaLN-Zero adds a per-branch gate initialized to zero.

::: {.qa asked="Waymo, Glassdoor 2022"}
**What does batch normalization do, and what are its parameters?**

It standardizes each channel with the mini-batch mean and variance over batch and space, then applies a learned scale $\gamma$ and shift $\beta$ so the layer can still represent the identity. Per channel: two learned parameters and two buffers (running mean and variance) used at evaluation; hyperparameters are $\epsilon$ and the momentum. It allows higher learning rates and its batch noise regularizes. Classic bugs: forgetting eval mode, and tiny batches.
:::

::: {.qa asked="Google intern, Glassdoor 2024"}
**What is the difference among batch, layer, instance and group normalization?**

Only the reduction axes. BN averages over batch and space per channel, coupling samples and needing large i.i.d. batches. LN averages over all features of one sample (per token in a transformer), so it ignores the batch and matches between training and evaluation. IN averages over space per sample and channel, removing per-image contrast and style. GN averages over space and a group of channels per sample; it sits between LN and IN and is the usual choice for small batches, as in detection or high-resolution diffusion training.
:::

::: {.qa asked="NVIDIA ML researcher, Glassdoor 2022"}
**Implement a BatchNorm2d forward pass. What details matter?**

```python
def bn2d(x, gamma, beta, rm, rv, training, m=0.1, eps=1e-5):
    # x: (N, C, H, W); rm, rv: running buffers of shape (C,)
    if training:
        mu = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)       # normalize with biased var
        n = x.numel() / x.shape[1]
        with torch.no_grad():                             # buffers are not trained
            rm.mul_(1 - m).add_(m * mu)
            rv.mul_(1 - m).add_(m * var * n / (n - 1))    # PyTorch stores unbiased var
    else:
        mu, var = rm, rv
    s = (1, -1, 1, 1)
    xh = (x - mu.view(s)) / torch.sqrt(var.view(s) + eps)
    return gamma.view(s) * xh + beta.view(s)
```

Reduce over $(N, H, W)$, not $C$; let gradients flow through `mu` and `var` in training; update the running buffers without gradient; use them at evaluation so the output is deterministic.
:::

::: qa
**Why do diffusion U-Nets use GroupNorm and DiTs use adaLN instead of BatchNorm?**

Diffusion batches mix very different noise levels, often with small per-GPU batches at high resolution, so batch statistics would be noisy and would differ between training and sampling; GroupNorm and LayerNorm are per-sample. The network must also change behavior with timestep and condition, and modulating $\gamma$, $\beta$ and a residual gate from the conditioning embedding does that cheaply and globally. DiT found adaLN-Zero gave the best FID among the conditioning schemes it tried.
:::

## Vision Transformers

**Patchify.** An $H\times W\times C$ image cut into $P\times P$ patches gives $N = HW/P^2$ tokens, each projected linearly from $P^2C$ values to width $D$ (a conv with kernel and stride $P$). ViT-B/16 at 224: 196 patches plus one CLS token; the patch embedding has $16\cdot16\cdot3\cdot768 + 768 = 590{,}592$ parameters. Each block has about $4D^2$ attention parameters ($Q, K, V, O$) and $8D^2$ MLP parameters, so $12D^2 \approx 7.1$M; 12 blocks give about 86M with embeddings.

**CLS token vs pooling.** ViT classified from a CLS token, as in BERT, but its appendix shows global average pooling works equally well once the learning rate is tuned.

**Position embeddings.** Attention is permutation-equivariant, so position must be injected. *Learned absolute* embeddings (ViT) store one vector per grid cell; ViT's ablation found a large gap between none and any, but little difference among 1D, 2D and relative variants. *Sinusoidal* embeddings use $\sin(m/10000^{2i/D})$ and $\cos(m/10000^{2i/D})$, with half the channels for rows and half for columns in 2D (as in DiT). *RoPE* rotates each pair of query and key dimensions by $m\theta_i$, $\theta_i = 10000^{-2(i-1)/d}$; since rotations compose, $\langle R_m q, R_n k\rangle = q^\top R_{n-m} k$ depends only on the offset. 2D axial RoPE rotates half the head dimensions by $x$ and half by $y$; RoPE-Mixed learns frequencies that mix the axes and extrapolated better to larger images (Heo et al.).

**Changing resolution.** Keep the patch size so the grid grows. With learned embeddings, reshape the table to its 2D grid, interpolate it in 2D (14×14 to 24×24 for 224 to 384 at $P=16$; bicubic is common), keep the CLS embedding, and fine-tune briefly, as ViT does. With RoPE, positions are coordinates: extrapolate or rescale them into the training range. Both work for moderate changes, but quality falls far outside the trained range. DINOv3 randomly rescales the coordinate box during training to gain robustness.

**Inductive biases.** A CNN builds locality, 2D neighborhoods and translation equivariance into every layer. A ViT uses 2D structure only when patchifying and interpolating position embeddings; everything else is learned. So ViT trained on ImageNet-1k alone trailed comparable ResNets by a few points, but overtook them with ImageNet-21k or JFT-300M pretraining. DeiT closed the ImageNet-1k gap with heavy augmentation and distillation. Raghu et al. found that ViT-L/16 and ViT-H/14 trained only on ImageNet-1k did not learn locally attending heads in their lowest layers, unlike the same models pretrained on JFT-300M, and performed much worse; ViT-B/32 learned local heads either way.

**Hierarchical designs.** Swin computes attention inside $M\times M$ windows ($M=7$), shifts them by half a window in alternate blocks so information crosses borders, and merges patches into a four-stage pyramid (strides 4 to 32) that fits FPN-style dense heads:

$$
\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C, \qquad \Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC ,
$$

linear in $hw$; at stage 1 at 224 ($hw = 3136$) the attention term shrinks $64\times$. ConvNeXt went the other way, modernizing ResNet-50 step by step: the training recipe alone took it from 76.1% to 78.8%, then a (3,3,9,3) stage ratio, a $4\times4$ patchify stem, depthwise $7\times7$ convs, inverted bottlenecks, GELU, fewer norms and activations, LN and separate downsampling layers reached 82.0% for ConvNeXt-T, above Swin-T. Much of the early gap was recipe and macro design, not attention itself.

::: {.qa asked="Apple senior CV engineer, Glassdoor"}
**Explain Vision Transformers.**

Cut the image into $P\times P$ patches, project each to a $D$-dimensional token, add position embeddings, optionally prepend a CLS token, and run a pre-LN transformer encoder; read out from CLS or mean-pooled tokens. A 224 image with $P=16$ gives 196 tokens; ViT-B ($D=768$, 12 layers) has about 86M parameters. Every layer mixes information globally but locality is not built in, so ViTs need large-scale pretraining or strong augmentation to beat CNNs, and scale well with data.
:::

## Attention and its cost

Scaled dot-product attention with $Q\in\mathbb{R}^{N\times d_k}$, $K\in\mathbb{R}^{M\times d_k}$, $V\in\mathbb{R}^{M\times d_v}$:

$$
\text{Attn}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V .
$$

**Why $\sqrt{d_k}$.** If the components of $q$ and $k$ are independent with mean 0 and variance 1, then $\mathbb{E}[q^\top k] = 0$ and

$$
\operatorname{Var}[q^\top k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = d_k .
$$

Unscaled logits have standard deviation $\sqrt{d_k}$ (8 for $d_k=64$), which saturates softmax toward one-hot. Its Jacobian $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$ then vanishes, and so do the gradients to $Q$ and $K$. The argument holds near initialization; logits can still grow in training, which is why some large ViTs add QK-normalization (ViT-22B reported divergence from attention-logit growth without it).

**Multi-head attention** splits $D$ into $h$ heads of width $D/h$; parameters ($4D^2$) and FLOPs match one head of width $D$, but the layer gets $h$ independent attention patterns.

**Counting.** Per layer with $N$ tokens and width $D$: projections cost $4ND^2$ MACs, $QK^\top$ and the product with $V$ cost $N^2D$ each, and a 4× MLP costs $8ND^2$:

$$
\text{MACs per layer} \approx 12ND^2 + 2N^2D,
$$

so the quadratic term dominates once $N > 6D$. A materialized attention matrix holds $hN^2$ entries per layer per sample.

**Worked example: a 1024×1024 image.**

| Tokenization | Tokens $N$ | Score entries per head |
|---|---|---|
| Pixels, $P=16$ | $64^2 = 4{,}096$ | 16.8M (32 MiB in fp16) |
| f8 VAE latent ($128^2$), DiT patch 2 | $4{,}096$ | 16.8M |
| f8 latent with patch 1, or pixels with $P=8$ | $128^2 = 16{,}384$ | 268M (512 MiB in fp16) |

At DiT-XL width $D=1152$ and $N=4096$, $2N^2D \approx 38.7$G MACs against $12ND^2 \approx 65.2$G, so the two $N^2$ products are about 37% of each layer's MACs; at $N = 16{,}384$ it is 618G against 261G, about 70%. Going from 256² to 1024² at fixed patch size multiplies tokens by 16 and attention cost by 256. Materializing all 16 heads at $N=4096$ takes 512 MiB per layer per image, so exact attention at this scale is usually run with fused kernels.

**Cheaper attention.** Windowed attention (Swin, neighborhood attention) is linear in $N$ but needs shifts or global tokens for long range. Linear attention replaces $\exp(q^\top k)$ with $\phi(q)^\top\phi(k)$ and reorders the product,

$$
\text{out}_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)},
$$

costing $O(Nd^2)$ because the $d\times d$ sum is computed once (its causal form is an RNN). The implicit attention matrix has rank at most the feature dimension, so it cannot be as sharply peaked as softmax, and quality usually lags.

**FlashAttention** keeps attention exact and attacks memory traffic. Standard attention writes $S = QK^\top$ and $P = \operatorname{softmax}(S)$ to HBM and reads them back, so it is memory-bound. FlashAttention tiles $Q, K, V$ into SRAM-sized blocks inside one fused kernel and computes softmax incrementally. For one query row, after each key block with scores $s_k$:

$$
m' = \max(m, \max_k s_k), \quad \ell' = e^{m-m'}\ell + \sum_k e^{s_k - m'}, \quad o' = e^{m-m'}o + \sum_k e^{s_k-m'}v_k ,
$$

and the output is $o/\ell$. The backward pass recomputes score blocks from $Q$, $K$ and the saved row statistics. Memory becomes linear in $N$, HBM accesses fall to $O(N^2d^2/M)$ for SRAM size $M$ (against $\Omega(Nd + N^2)$), and attention ran up to 7.6× faster on GPT-2. FLOPs stay quadratic, slightly higher due to recomputation.

**KV caching** stores past keys and values in causal decoding (autoregressive image models, causal video generators). It does not apply across the denoising steps of a bidirectional diffusion transformer, where every token changes.

::: qa
**Why do we divide attention logits by $\sqrt{d_k}$ and not by $d_k$?**

With unit-variance independent components, $q^\top k$ has variance $d_k$. Large logits push softmax to near one-hot, where its Jacobian $p_i(\delta_{ij}-p_j)$ and the gradients to $Q$ and $K$ vanish. Dividing by $\sqrt{d_k}$ makes the logit variance 1 for any head width. Dividing by $d_k$ over-shrinks it to $1/d_k$, giving nearly uniform attention that learns slowly.
:::

::: qa
**Is FlashAttention an approximation? What exactly does it reduce?**

No; it computes the same softmax attention up to floating-point rounding. It reduces HBM traffic and memory, not arithmetic. Tiling plus an online softmax that rescales partial sums whenever the running max changes means the $N\times N$ matrix never reaches slow memory, and the backward pass recomputes tiles instead of storing them. Memory becomes linear in $N$ and wall-clock time drops because attention was memory-bound, but FLOPs stay $O(N^2d)$.
:::

::: {.qa asked="Apple MLE, Glassdoor"}
**What is a KV cache, and how does it help inference?**

In causal decoding each new token attends to all earlier tokens, whose keys and values never change. Caching them per layer means each step projects only the new token and computes one attention row, $O(ND)$ attention work per layer per step instead of re-encoding the prefix. The cost is memory, $2 \times \text{layers} \times N \times D$ values per sequence: 2 GiB for 32 layers, $D=4096$ and 4096 tokens in fp16. That memory often limits batch size, which grouped-query attention, cache quantization and paged allocation reduce.
:::

## Self-supervised representation learning

### Contrastive learning and InfoNCE

Take an anchor $x$, one positive $y^+$ (another view, or the paired caption) and $N-1$ negatives, with $\ell_2$-normalized embeddings, cosine similarity $s$ and temperature $\tau$:

$$
\mathcal{L}_N = -\,\mathbb{E}\left[\log \frac{\exp(s(x,y^+)/\tau)}{\sum_{j=1}^{N}\exp(s(x,y_j)/\tau)}\right],
$$

the cross-entropy of an $N$-way "which one is the positive" classification.

**Mutual-information bound.** Draw the positive from $p(y\mid x)$ and negatives from $p(y)$. The Bayes-optimal critic is the density ratio, $e^{f^*(x,y)} \propto p(y\mid x)/p(y)$. Substituting it gives CPC's approximate argument:

$$
\begin{aligned}
\mathcal{L}_N^{\text{opt}} &= \mathbb{E}\log\left[1 + \frac{p(y^+)}{p(y^+\mid x)}\sum_{j\in\text{neg}} \frac{p(y_j\mid x)}{p(y_j)}\right]
\approx \mathbb{E}\log\left[1 + \frac{p(y^+)}{p(y^+\mid x)}(N-1)\right] \\
&\gtrsim \mathbb{E}\log\left[N\,\frac{p(y^+)}{p(y^+\mid x)}\right] = \log N - I(x;y).
\end{aligned}
$$

Both steps are heuristic: the first replaces the sum over negatives by $N-1$ times its expectation (which is 1), and the second, $1+(N-1)a \ge Na$ with $a = p(y^+)/p(y^+\mid x)$, holds only when $a\le 1$. The exact statement is Poole et al.'s: $I(x;y) \ge \log N - \mathcal{L}_N$ for any critic. Consequences: the bound cannot exceed $\log N$ ($N = 4096$ candidates certify at most $\ln 4096 \approx 8.3$ nats), one reason contrastive methods want many negatives; and a tighter bound does not by itself mean better features, since the augmentations decide what the views share.

**Temperature.** With $p_j$ the softmax weight of candidate $j$,

$$
\frac{\partial \mathcal{L}}{\partial s^+} = -\frac{1-p^+}{\tau}, \qquad \frac{\partial \mathcal{L}}{\partial s_j} = \frac{p_j}{\tau}\ \ (j \ne +),
$$

so negatives are pushed in proportion to their softmax weight. Small $\tau$ concentrates the push on the hardest negatives, giving uniform embeddings but also separating semantically similar images; large $\tau$ treats negatives evenly (Wang and Liu's uniformity–tolerance dilemma). As $N\to\infty$, Wang and Isola show that

$$
\mathcal{L}_N - \log N \to -\tfrac{1}{\tau}\,\mathbb{E}_{\text{pos}}\big[f(x)^\top f(y)\big] + \mathbb{E}_x \log \mathbb{E}_{x^-}\big[e^{f(x)^\top f(x^-)/\tau}\big],
$$

an *alignment* term for positives plus a *uniformity* term on the hypersphere.

### SimCLR and MoCo

SimCLR takes two strong augmentations per image (crop and color distortion matter most), encodes them into $h$, maps them through an MLP head to $z$, and applies the loss over $2N$ views at batch 4096. Linear probes on $h$ beat probes on $z$ by more than 10%, because the head absorbs the invariances the loss imposes, such as discarding color. MoCo decouples negatives from batch size with a queue of 65,536 keys encoded by a momentum copy of the query encoder, $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$ with $m = 0.999$, which keeps queued keys consistent; with $m=0$ training failed.

### CLIP and SigLIP

CLIP trains image and text encoders on 400M pairs at batch 32,768 with a symmetric cross-entropy over the $N\times N$ cosine matrix: image-to-text over rows and text-to-image over columns, each an InfoNCE with $N-1$ in-batch negatives. The temperature is learned, initialized at the equivalent of 0.07 and clipped so logits scale by at most 100. Zero-shot classification follows from embedding class-name prompts.

SigLIP scores every pair as an independent binary classification:

$$
\mathcal{L} = -\frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{|B|} \log \sigma\big(z_{ij}\,(t\, x_i^\top y_j + b)\big), \qquad z_{ij} = +1 \text{ if } i=j, \ -1 \text{ otherwise},
$$

with $t = e^{t'}$, $t'$ initialized to $\log 10$, and $b$ initialized to $-10$. This follows the paper's pseudocode (logits $= t\,x_i^\top y_j + b$); its displayed equation, $\log\big(1/(1+e^{z_{ij}(-t\,x_i^\top y_j + b)})\big)$, has the opposite sign on $b$, which does not fit the $-10$ initialization. Minimizing the loss raises $x_i^\top y_i$ for matching pairs and lowers $x_i^\top y_j$ for the rest. A batch has $|B|$ positives and $|B|^2-|B|$ negatives, so the negative bias starts the model near this prior instead of spending early steps correcting it. With no batch-wide normalization, devices can exchange text embeddings in turn and accumulate the loss in chunks, never holding the full matrix. Sigmoid beat softmax below about 16k batch, and both saturated around 32k. SigLIP 2 adds objectives aimed at localization and dense features.

### Self-distillation: the DINO family

**DINO** trains a student and an EMA teacher (momentum 0.996 rising to 1) with no labels and no negatives. Both output $K$-way softmaxes, with student temperature $\tau_s = 0.1$ and teacher $\tau_t$ warmed from 0.04 to 0.07. The teacher sees only two global 224² crops; the student also sees several local 96² crops:

$$
\min_{\theta_s} \sum_{x\in\{x^g_1,x^g_2\}}\ \sum_{x'\ne x} H\big(P_t(x), P_s(x')\big), \qquad H(a,b) = -\sum_k a_k\log b_k ,
$$

forcing local-to-global correspondence. *Centering* (subtracting an EMA of the teacher's mean output) stops one dimension from dominating but pushes toward uniform; *sharpening* (the low $\tau_t$) does the opposite; balanced, with the EMA teacher, they prevent collapse. The CLS attention maps end up segmenting foreground objects.

**DINOv2** curated LVD-142M and combined the DINO loss with iBOT's masked patch loss (the student predicts the teacher's patch tokens at masked positions), Sinkhorn-Knopp centering and a KoLeo term that spreads features, then distilled a 1.1B ViT-g/14 into smaller models.

**Registers.** Darcet et al. found that DINOv2 models of size ViT-L and up, after about a third of training, turn roughly 2% of patch tokens in redundant background regions into high-norm tokens (2.37% above norm 150 in ViT-g) that hold global rather than local information and corrupt attention maps. DINO v1 does not show them; supervised DeiT-III and OpenCLIP do, already at ViT-B. Appending a few learnable tokens (commonly 4), discarded at the output, removes the artifacts and improves dense prediction and object discovery.

**DINOv3** (2025) scales to a 7B ViT (patch 16, 4 registers, RoPE) on LVD-1689M, curated from about 17B images. In long runs its global metrics kept improving while dense features degraded: segmentation peaked around 200k iterations as patch features drifted toward the CLS token. *Gram anchoring*, switched on after 1M iterations, adds

$$
\mathcal{L}_{\text{Gram}} = \big\lVert X_S X_S^\top - X_G X_G^\top \big\rVert_F^2 ,
$$

where $X_S, X_G \in \mathbb{R}^{P\times d}$ hold $\ell_2$-normalized patch features of the student and of an earlier "Gram teacher" checkpoint. It constrains the patch-to-patch similarity structure, not the features themselves. Distillation into smaller ViT and ConvNeXt students follows.

### Masked image modeling and JEPA

**MAE** masks a random 75% of patches. The encoder sees only the visible 25%, with no mask tokens, which speeds pretraining 3× or more. A lightweight decoder (8 blocks, width 512 by default) adds mask tokens and reconstructs, with MSE on masked patches only; per-patch normalized pixel targets work better. **I-JEPA** predicts representations instead of pixels. A context encoder sees a large block (scale 0.85–1.0) with targets removed; an EMA target encoder sees the full image, and 4 target blocks (scale 0.15–0.2, aspect 0.75–1.5) are cut from its output; a narrow predictor regresses them with an L2 loss. It needs no hand-crafted augmentations and can ignore unpredictable pixel detail.

### What different features capture

Early layers are local and low-level (edges, color, texture); later layers are more semantic and invariant. The objective matters as much as depth. Invariance objectives (contrastive, DINO, CLIP) discard what the augmentations vary (color, exact position, fine texture) and give linearly separable features; reconstruction objectives (MAE, autoencoders) keep layout and detail but probe worse. CLIP features align with text but localize less well; DINOv2 and DINOv3 give the strongest frozen dense features (correspondence, segmentation, depth probes). Intermediate diffusion-model features also carry semantic correspondence.

::: qa
**Derive why minimizing InfoNCE maximizes a lower bound on mutual information. What limits the bound?**

The loss is $N$-way classification of the positive among samples from the marginal, and its optimal critic is the density ratio $p(y\mid x)/p(y)$. Substituting it, replacing the negatives' sum by $N-1$ times its mean, and using $1+(N-1)a \approx Na$ gives CPC's approximate result $\mathcal{L}^{\text{opt}}_N \gtrsim \log N - I(x;y)$. The exact statement, proved by Poole et al., is $I(x;y) \ge \log N - \mathcal{L}_N$ for any critic. The bound saturates at $\log N$, so large MI needs many negatives, and maximizing it is not the same as learning good features: the choice of views matters more.
:::

::: {.qa asked="TikTok algorithm intern, Glassdoor 2025"}
**Explain contrastive learning. How does it handle the imbalance between positive and negative samples?**

It trains an encoder so that views sharing content (augmentations of one image, or an image and its caption) land close together and other samples land far apart, usually with InfoNCE. The softmax form handles the imbalance by construction: each anchor contributes one $N$-way cross-entropy, and the total gradient on all negatives, $\sum_j p_j/\tau = (1-p^+)/\tau$, exactly matches the gradient on the positive however many negatives there are. The temperature decides how that push is shared among negatives. In pairwise sigmoid losses like SigLIP the imbalance is real, and a bias initialized at $-10$ absorbs it. Also raise false negatives.
:::

::: {.qa asked="Coactive AI applied ML RS, Glassdoor 2025"}
**How does CLIP build a shared image-text representation, and what changes in SigLIP?**

Two encoders project into one normalized space. In a batch of $N$ pairs, a symmetric cross-entropy over the $N\times N$ cosine matrix makes each image pick its caption among $N$ and vice versa, with a learned temperature. The embeddings capture what distinguishes web captions (categories, attributes, scene type, rendered text) and capture localization and counting less well. SigLIP replaces the batch softmax with an independent sigmoid per pair plus a learned bias. That removes batch-wide normalization, makes chunked multi-device training cheap in memory, and works better at small batches; the advantage fades above about 16k.
:::

::: qa
**How does DINO avoid collapse without negatives?**

There are two collapse modes: every image mapping to the same one-hot output, or to a uniform one. Centering stops any single dimension from dominating but drifts toward uniform; sharpening, through a teacher temperature of 0.04–0.07 against 0.1 for the student, pushes toward peaked outputs. The two balance. The EMA teacher adds stability (without momentum the method fails), and multi-crop gives the student a real local-to-global prediction task.
:::

::: qa
**Why does MAE mask 75% of patches, and how do its features differ from DINO's?**

Images are redundant, so at low masking ratios a patch can be inpainted from its neighbors with little understanding; 75% forces holistic reasoning, and skipping masked patches in the encoder makes pretraining 3× or more faster. Pixel reconstruction keeps layout and low-level information, so MAE fine-tunes well but linear-probes worse. DINO's invariance objective gives semantically separable features and clean attention maps while discarding some low-level detail. DINOv2 combines both ideas with an image-level DINO loss and a masked patch-level iBOT loss.
:::

## Representations for generation

The premise: a diffusion transformer must learn both a good internal representation and a denoiser, and the first part is slow. There are three ways to bring in a pretrained encoder.

**REPA** adds a regularizer. Let $y^*$ be frozen encoder patch features of the clean image, $h_t$ an early hidden state for the noisy input, $h_\phi$ a trainable MLP and sim the cosine similarity:

$$
\mathcal{L}_{\text{REPA}} = -\,\mathbb{E}_{x^*,\epsilon,t}\left[\frac1N\sum_{n=1}^{N} \text{sim}\big(y^{*[n]}, h_\phi(h_t^{[n]})\big)\right], \qquad \mathcal{L} = \mathcal{L}_{\text{velocity}} + \lambda\,\mathcal{L}_{\text{REPA}}, \ \ \lambda = 0.5 .
$$

DINOv2 worked best among the encoders tested, and aligning around layer 8 of SiT-XL/2 was best. Without guidance, SiT-XL/2 with REPA reaches FID 7.9 at 400K iterations against 8.3 for the vanilla model at 7M (over 17.5× faster), and 1.42 with CFG and a guidance interval. The motivation: vanilla DiT/SiT hidden states align only weakly with DINOv2.

**VA-VAE** aligns the tokenizer. More latent channels improve reconstruction but hurt generation at a fixed budget; for LightningDiT-XL without CFG (FID-10K), f16d16 gets rFID 0.49 and gFID 8.28, f16d32 0.26 and 10.92, f16d64 0.17 and 17.24. The fix is a vision-foundation loss that projects the latent to DINOv2's feature dimension and adds two margin terms:

$$
\mathcal{L}_{\text{mcos}} = \frac{1}{hw}\sum_{i,j}\mathrm{ReLU}\big(1 - m_1 - \cos(z'_{ij}, f_{ij})\big), \quad
\mathcal{L}_{\text{mdms}} = \frac{1}{N^2}\sum_{i,j}\mathrm{ReLU}\big(\lvert\cos(z_i,z_j) - \cos(f_i,f_j)\rvert - m_2\big),
$$

with $m_1=0.5$, $m_2=0.25$ and an adaptive weight. f16d32 with this loss gets rFID 0.28 and gFID 8.22. With the improved LightningDiT, the system reaches FID 1.35 at 800 epochs and 2.11 at 64 epochs, which the paper calls over 21× faster convergence than the original DiT.

**RAE** (representation autoencoder) removes the VAE. A frozen encoder (default DINOv2-B with registers; inputs resized to 224, giving a $16\times16\times768$ latent) is paired with a trained ViT-XL decoder (L1, LPIPS and GAN losses) that outputs 256². Reconstruction rFID is 0.49, against 0.62 for the SD-VAE. Diffusing in this latent needs three changes: transformer width at least the token dimension (narrower models fail even to overfit one image); a dimension-dependent schedule shift $t_m = \alpha t_n / (1 + (\alpha-1)t_n)$ with $\alpha = \sqrt{m/n}$, where $m$ is tokens × channels and $n = 4096$; and a decoder trained on noise-perturbed latents. A shallow, wide head (DiT$^{\text{DH}}$, 2 layers of width 2048) adds width cheaply. Results: FID 1.51 without guidance (800 epochs) and 1.13 with AutoGuidance at both 256 and 512, with no auxiliary alignment loss.

::: key
REPA aligns the generator's hidden states to a pretrained encoder, VA-VAE aligns the tokenizer's latent to one, and RAE uses the encoder's feature space as the latent itself. All three bet that semantic structure, not reconstruction fidelity alone, makes a latent space easy to model.
:::

::: qa
**Why does REPA speed up diffusion training, and why align an early layer?**

A denoiser must infer what is in the noisy image before predicting the clean signal, and learning that representation from the denoising loss alone is slow. REPA supplies it by pulling early hidden states, through a small MLP, toward DINOv2 features of the *clean* image at every noise level. Aligning an early layer (about 8 of 28 in SiT-XL/2) leaves later layers free to model detail; later layers worked worse in the ablation, plausibly because they must specialize in predicting the output.
:::

::: qa
**What goes wrong when you run diffusion in a high-dimensional encoder latent, as RAE does, and how is it fixed?**

Three things. A transformer narrower than the token dimension cannot fit the data, so width must be at least the token dimension. With more total dimensions, signal survives more noise, so the usual schedule under-noises; a shift scaling with $\sqrt{\text{dim}/4096}$ moves training toward higher noise. Generated latents land slightly off the encoder's output manifold, so the decoder is trained on noise-augmented latents. A wide, shallow head adds width cheaply.
:::

## Optical flow

Optical flow assigns each pixel a displacement $(u, v)$ between frames; video pipelines use it to filter static clips, measure motion and condition on it (see [video generation](video-generation.html)).

**Brightness constancy**, $I(x+u, y+v, t+1) = I(x, y, t)$, linearizes to

$$
I_x u + I_y v + I_t = 0 ,
$$

one equation in two unknowns per pixel. This is the *aperture problem*: only the component along the gradient, $-I_t/\lVert\nabla I\rVert$, is determined. Lucas–Kanade assumes constant flow in a window and solves $\big(\sum \nabla I\,\nabla I^\top\big)[u, v]^\top = -\sum \nabla I\, I_t$, well-posed only where the structure tensor is well-conditioned (corners). Horn–Schunck adds global smoothness, minimizing $\int (I_xu + I_yv + I_t)^2 + \alpha^2(\lVert\nabla u\rVert^2 + \lVert\nabla v\rVert^2)$. Coarse-to-fine pyramids handle larger motions.

**RAFT** (1) encodes both frames at 1/8 resolution ($D=256$) plus a context encoding of frame 1; (2) builds an all-pairs correlation volume $C_{ijkl} = \sum_h g^1_{ijh} g^2_{klh}$ of shape $H\times W\times H\times W$ and average-pools its last two dimensions into a 4-level pyramid; (3) starting from $f_0 = 0$, a convolutional GRU looks up correlations within radius 4 around $x + f_k(x)$ at every level and outputs $\Delta f$, so $f_{k+1} = f_k + \Delta f$. It unrolls 12 iterations in training (more at test time), upsamples with learned convex combinations, and trains with $\sum_{i}\gamma^{N-i}\lVert f_{\text{gt}} - f_i\rVert_1$, $\gamma = 0.8$. The full volume keeps every candidate match, so small fast objects are not lost at coarse levels.

::: {.qa asked="NVIDIA phone screen, Glassdoor 2020"}
**What is optical flow, where is it used, and what are its limitations?**

It is the apparent per-pixel motion between frames, estimated classically from linearized brightness constancy, $I_xu + I_yv + I_t = 0$, or by learned models like RAFT. Uses: stabilization, frame interpolation, tracking, and in generative video, data filtering, motion metrics and conditioning. Limitations: the aperture problem, large displacements that break linearization, occlusions with no correspondence, illumination changes, textureless regions, and blurred motion boundaries. It measures 2D image motion, not 3D scene motion.
:::

## Rapid-fire review

::: qa
**How many parameters and buffers does BatchNorm2d with $C$ channels have?**

$2C$ learned ($\gamma, \beta$) and $2C$ buffers (running mean and variance), plus a batch counter in PyTorch.
:::

::: qa
**How do you fold BN into the preceding conv for inference?**

$W' = \gamma W/\sqrt{\sigma^2+\epsilon}$ per output channel and $b' = \gamma(b-\mu)/\sqrt{\sigma^2+\epsilon} + \beta$, with the running statistics; exact at evaluation.
:::

::: qa
**What extent does a $3\times3$ conv with dilation 2 cover?**

$d(k-1)+1 = 5$ pixels per side with 9 weights; padding 2 keeps the size.
:::

::: qa
**How do you avoid checkerboard artifacts when upsampling?**

Upsample with nearest-neighbor or bilinear interpolation, then apply a regular conv, so every output pixel receives the same number of contributions.
:::

::: qa
**Post-LN vs pre-LN transformers?**

Post-LN normalizes after the residual sum, giving large gradients near the output at initialization and needing warmup; pre-LN keeps a clean identity path and trains more stably (Xiong et al.).
:::

::: qa
**What does linear attention give up?**

Sharpness: kernelized scores make the attention matrix low-rank, so it cannot focus as precisely as softmax, in exchange for $O(Nd^2)$ cost.
:::

::: qa
**Why does Swin shift its windows?**

Without shifts, attention never crosses window borders within a stage; alternating shifted windows connect neighboring windows at almost no extra cost.
:::

::: qa
**Why does SimCLR evaluate the layer before the projection head?**

The head becomes invariant to the augmentations and discards information such as color; the layer below keeps it and probes more than 10% better.
:::

::: qa
**What are registers in ViTs?**

Extra learnable tokens, discarded at the output, that give large ViTs a place for global computation, so background patch tokens are not repurposed into high-norm artifacts.
:::

::: qa
**What does DINOv3's Gram anchoring constrain?**

The Gram matrix of normalized patch features, matched to an earlier checkpoint, which keeps the patch similarity structure and stops dense features degrading in long training.
:::

::: qa
**ResNet skips vs U-Net skips?**

ResNet skips add a block's input to its output to ease optimization; U-Net skips concatenate encoder features into the decoder to restore spatial detail.
:::

## References

1. He et al., "Deep Residual Learning for Image Recognition," 2015. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
2. He et al., "Identity Mappings in Deep Residual Networks," 2016. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
3. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
4. Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," 2014. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
5. Dumoulin and Visin, "A guide to convolution arithmetic for deep learning," 2016. [arXiv:1603.07285](https://arxiv.org/abs/1603.07285)
6. Luo et al., "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks," 2016. [arXiv:1701.04128](https://arxiv.org/abs/1701.04128)
7. Yu and Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions," 2015. [arXiv:1511.07122](https://arxiv.org/abs/1511.07122)
8. Lin et al., "Network In Network," 2013. [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
9. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," 2016. [arXiv:1610.02357](https://arxiv.org/abs/1610.02357)
10. Howard et al., "MobileNets," 2017. [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
11. Odena, Dumoulin and Olah, "Deconvolution and Checkerboard Artifacts," Distill, 2016. [distill.pub](https://distill.pub/2016/deconv-checkerboard/)
12. Li et al., "Visualizing the Loss Landscape of Neural Nets," 2017. [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)
13. Ioffe and Szegedy, "Batch Normalization," 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
14. Santurkar et al., "How Does Batch Normalization Help Optimization?," 2018. [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
15. Ba, Kiros and Hinton, "Layer Normalization," 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
16. Ulyanov, Vedaldi and Lempitsky, "Instance Normalization: The Missing Ingredient for Fast Stylization," 2016. [arXiv:1607.08022](https://arxiv.org/abs/1607.08022)
17. Huang and Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization," 2017. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
18. Karras et al., "A Style-Based Generator Architecture for Generative Adversarial Networks," 2018. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
19. Wu and He, "Group Normalization," 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
20. Zhang and Sennrich, "Root Mean Square Layer Normalization," 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
21. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," 2017. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
22. Peebles and Xie, "Scalable Diffusion Models with Transformers," 2022. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
23. Xiong et al., "On Layer Normalization in the Transformer Architecture," 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
24. Vaswani et al., "Attention Is All You Need," 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
25. Dosovitskiy et al., "An Image is Worth 16x16 Words," 2020. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
26. Touvron et al., "Training data-efficient image transformers & distillation through attention," 2020. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
27. Raghu et al., "Do Vision Transformers See Like Convolutional Neural Networks?," 2021. [arXiv:2108.08810](https://arxiv.org/abs/2108.08810)
28. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
29. Heo et al., "Rotary Position Embedding for Vision Transformer," 2024. [arXiv:2403.13298](https://arxiv.org/abs/2403.13298)
30. Liu et al., "Swin Transformer," 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
31. Liu et al., "A ConvNet for the 2020s," 2022. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
32. Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters," 2023. [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)
33. Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention," 2020. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
34. Milakov and Gimelshein, "Online normalizer calculation for softmax," 2018. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
35. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
36. Dao, "FlashAttention-2," 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
37. van den Oord, Li and Vinyals, "Representation Learning with Contrastive Predictive Coding," 2018. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
38. Poole et al., "On Variational Bounds of Mutual Information," 2019. [arXiv:1905.06922](https://arxiv.org/abs/1905.06922)
39. Wang and Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere," 2020. [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)
40. Wang and Liu, "Understanding the Behaviour of Contrastive Loss," 2020. [arXiv:2012.09740](https://arxiv.org/abs/2012.09740)
41. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)," 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
42. He et al., "Momentum Contrast for Unsupervised Visual Representation Learning," 2019. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
43. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision (CLIP)," 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
44. Zhai et al., "Sigmoid Loss for Language Image Pre-Training," 2023. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
45. Tschannen et al., "SigLIP 2," 2025. [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)
46. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers (DINO)," 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
47. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
48. Darcet et al., "Vision Transformers Need Registers," 2023. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
49. Siméoni et al., "DINOv3," 2025. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
50. He et al., "Masked Autoencoders Are Scalable Vision Learners," 2021. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
51. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)," 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
52. Tang et al., "Emergent Correspondence from Image Diffusion," 2023. [arXiv:2306.03881](https://arxiv.org/abs/2306.03881)
53. Yu et al., "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think (REPA)," 2024. [arXiv:2410.06940](https://arxiv.org/abs/2410.06940)
54. Yao, Yang and Wang, "Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models (VA-VAE)," 2025. [arXiv:2501.01423](https://arxiv.org/abs/2501.01423)
55. Zheng et al., "Diffusion Transformers with Representation Autoencoders," 2025. [arXiv:2510.11690](https://arxiv.org/abs/2510.11690)
56. Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow," 2020. [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)
