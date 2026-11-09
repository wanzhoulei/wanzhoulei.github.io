---
title: "Text-to-Image and Multimodal Generation"
subtitle: "How text gets into an image generator: conditioning blocks, text encoders, captions, guidance, adapters, editing, unified models, and post-training."
order: 9
updated: 2026-11-08
---

A text-to-image model is a conditional generator $p_\theta(x \mid c)$ over images (usually VAE latents) given a prompt $c$. The sampler is covered in [diffusion models](diffusion-models.html) and [flow matching](flow-matching.html). This page covers what makes the condition work: how text enters the network, which encoder produces it, what captions the model saw, how guidance strengthens it, and how pretrained models are extended with adapters, editing methods, unified architectures and reward-based post-training. Interviews here mix architecture questions (where do Q, K, V come from), short derivations (LoRA gradients, two-scale guidance, Diffusion-DPO) and data judgment. Several of the largest reported quality jumps came from the text encoder and the captions rather than the sampler (Imagen, DALL-E 3).

## Reference systems

| Model | Backbone | Text encoder(s) | How text enters |
|---|---|---|---|
| SD 1.x (LDM) | latent U-Net | CLIP ViT-L/14 | cross-attention |
| Imagen | pixel U-Net cascade, 64 to 1024 px | frozen T5-XXL | cross-attention + pooled vector added to timestep embedding |
| SDXL | 2.6B latent U-Net | CLIP ViT-L + OpenCLIP ViT-bigG | cross-attention + OpenCLIP pooled vector |
| PixArt-$\alpha$ | DiT | Flan-T5-XXL, 120 tokens | cross-attention; timestep via adaLN-single |
| SD3 | MMDiT, rectified flow | CLIP-L, CLIP-G, T5-XXL | joint attention; pooled CLIP in modulation |
| FLUX.1 | 19 double-stream + 38 single-stream blocks | T5-v1.1-XXL + pooled CLIP-L | joint attention; pooled CLIP + timestep (+ guidance) in modulation |

## Conditioning mechanisms

Four ways to inject a condition into a denoiser $f_\theta(x_t, t, c)$. They differ in what kind of condition they suit (pixel-aligned map, global vector, token sequence) and in cost.

### Concatenation

**Channel concatenation** suits pixel-aligned conditions: the low-resolution input in super-resolution, mask and masked image in inpainting, the source image in instruction editing. The input becomes $[x_t; E(c)]$, and the first layer's new input weights are usually zero-initialized so the pretrained model is unchanged at step 0 (InstructPix2Pix does this). **Sequence concatenation** appends condition tokens to the image tokens and lets self-attention mix them, as in DiT's "in-context" variant and FLUX.1 Kontext's context-image tokens. It needs no new layers but pays $(N+M)^2$ attention, and shared weights treat both modalities identically.

### Cross-attention

Latent Diffusion takes queries from spatial features $\varphi(z_t) \in \mathbb{R}^{N\times d_z}$ and keys and values from text-encoder outputs $\tau(y) \in \mathbb{R}^{M\times d_\tau}$:

$$
Q = \varphi(z_t) W_Q,\quad K = \tau(y) W_K,\quad V = \tau(y) W_V,\qquad
\mathrm{CA} = \underbrace{\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big)}_{A \in \mathbb{R}^{N \times M}} V .
$$

Row $i$ of $A$ is a distribution over the $M$ words for location $i$, so each location outputs its own convex combination of word values. That per-location routing is what binds "red" to the cube and "blue" to the sphere, and it makes the columns of $A$ interpretable word masks. Cost is $O(NMd)$ with small $M$ (77 for CLIP). The limitation: text features are the same frozen encoder output at every layer and never adapt to the image.

### adaLN and adaLN-Zero

FiLM-style conditioning from a global vector $e = \mathrm{emb}(t) + \mathrm{emb}(y)$. A DiT block with adaLN-Zero (LayerNorm without its own affine):

$$
\begin{aligned}
(\beta_1,\gamma_1,\alpha_1,\beta_2,\gamma_2,\alpha_2) &= \mathrm{MLP}(e),\\
x &\leftarrow x + \alpha_1 \odot \mathrm{Attn}\big((1+\gamma_1)\odot \mathrm{LN}(x) + \beta_1\big),\\
x &\leftarrow x + \alpha_2 \odot \mathrm{FFN}\big((1+\gamma_2)\odot \mathrm{LN}(x) + \beta_2\big).
\end{aligned}
$$

The MLP's last layer is zero-initialized, so $\alpha = 0$ and every block starts as the identity. In DiT's class-conditional ablation (XL/2, 400K steps) FID was 19.47 for adaLN-Zero, 25.21 for adaLN, 26.14 for cross-attention (about 15% more compute) and 35.24 for in-context. But adaLN applies one affine map to all tokens, so it is global: a pooled text vector cannot make location $i$ red and location $j$ blue, and SD3's authors note pooled text keeps only coarse information. Text-to-image DiTs therefore keep adaLN for the timestep (plus pooled text) and add a sequence path. PixArt-$\alpha$ adds cross-attention and replaces per-block modulation MLPs with one shared MLP plus learned per-block offsets ("adaLN-single"), cutting parameters 26%.

### MMDiT joint attention

SD3's MMDiT keeps an image stream $X \in \mathbb{R}^{N\times d}$ and a text stream $C \in \mathbb{R}^{M\times d}$, each with its own modulated LayerNorm, QKV and output projections and MLP. Only attention is shared:

$$
\begin{bmatrix} O_X \\ O_C \end{bmatrix}
= \mathrm{softmax}\!\left(\frac{1}{\sqrt d}\begin{bmatrix} Q_X \\ Q_C \end{bmatrix}\begin{bmatrix} K_X \\ K_C \end{bmatrix}^{\!\top}\right)\begin{bmatrix} V_X \\ V_C \end{bmatrix},
\qquad Q_X = \tilde X W_Q^{X},\; Q_C = \tilde C W_Q^{C}.
$$

One softmax covers all four directions of attention, and text tokens are rewritten at every layer, so the text representation can adapt to the image. Modulation comes from $\mathrm{emb}(t)$ plus an MLP of pooled CLIP embeddings. SD3 adds RMSNorm on queries and keys to stop logit growth, enabling bf16 training. In its controlled comparison MMDiT beat a cross-attention DiT, a UViT and a vanilla DiT that concatenates text and image tokens with shared weights (MMDiT with one weight set). A third weight set (separate CLIP and T5 streams) gave only a small gain. FLUX.1 runs 19 such double-stream blocks, then concatenates the streams into 38 single-stream blocks with shared weights and parallel attention and MLP.

| Mechanism | Condition | Per-location routing | Text sees image | Extra cost | Examples |
|---|---|---|---|---|---|
| Channel concat | pixel-aligned map | by construction | n/a | ~0 | SR, inpainting, InstructPix2Pix |
| Token concat | any tokens | yes | yes | $(N+M)^2$ attention | DiT in-context, Kontext |
| Cross-attention | token sequence | yes | no | $NM$ attention | SD 1.x/XL, Imagen, PixArt-$\alpha$ |
| adaLN(-Zero) | global vector | no | n/a | small MLP | timestep, class, pooled text |
| Joint attention | token sequence | yes | yes | $(N+M)^2$, ~2x params | SD3, FLUX.1 |

::: key
Match the mechanism to the condition's shape: pixel-aligned maps go in by concatenation or residual addition, global vectors by modulation, token sequences by attention. Modern models use two at once: modulation for timestep and pooled text, attention for the word sequence.
:::

::: qa
**In Stable Diffusion's cross-attention, where do Q, K and V come from, and what does the attention map mean?**

Queries come from the U-Net's spatial features at that layer (flattened to $N$ tokens); keys and values come from the text encoder's $M$ token embeddings, each through a learned projection. $A = \mathrm{softmax}(QK^\top/\sqrt d)$ gives, for each location, how much it reads each word, and the output there is the weighted average of word values, added back to the residual stream. Consequences: regions can read different words (attribute binding), each word's column is a soft segmentation map (which Prompt-to-Prompt edits), and cost is linear in prompt length. The text never sees the image, which is what joint attention changes.
:::

::: qa
**Why zero-initialize the adaLN gate?**

With $\alpha = 0$ every residual branch outputs zero, so the network starts as the identity and gradients reach all blocks through the skip path; each block then learns how much to contribute, like zero-initializing the last BatchNorm scale in a ResNet. In DiT's ablation this beat plain adaLN by about 6 FID.
:::

::: {.qa asked="Baidu multimodal algorithm intern, Nowcoder 2026"}
**How is the text condition injected in FLUX?**

Two paths. Per-token: T5-v1.1-XXL hidden states (4096-d) are projected to model width and form the text stream of 19 double-stream blocks, which keep separate text and image weights but share one joint attention; the streams are then concatenated for 38 single-stream blocks with shared weights. Global: the pooled CLIP ViT-L/14 embedding (768-d) passes through an MLP and is added to the timestep embedding (and, in the guidance-distilled dev model, a guidance-scale embedding); this vector sets shift, scale and gate in every block. Positions use 3D RoPE over (index, height, width); in the reference code image tokens get $(0,h,w)$ and text tokens all zeros, so word order comes from T5 itself.
:::

## Text encoders

**CLIP text tower.** Contrastively trained against images, 77-token context, causal mask, pooled embedding at the end-of-text token. Its space is already aligned with visual concepts and styles, but a contrastive loss on short alt-text rewards knowing *which* things appear more than *how* they relate: such models act like bags of words on order and binding probes (Yuksekgonul et al.).

**T5 encoder.** Text-only, bidirectional, trained by span corruption on a large corpus. Imagen's key finding: scaling the frozen text encoder improved fidelity and alignment much more than scaling the U-Net, and on DrawBench raters preferred T5-XXL over CLIP on both (the two were similar on MS-COCO). The image model learns language only from captions, a small noisy corpus, so a strong frozen LM supplies syntax, counting and rare words it could not learn alone.

**Decoder-only LLMs.** Playground v3 conditions only on a decoder-only LLM, Sana uses a small decoder LLM with an instruction prefix, and Qwen-Image conditions on Qwen2.5-VL. They bring world knowledge, multilingual input and long prompts. The catch: under causal attention early tokens never see later words, and last-layer states are tuned for next-token prediction, so systems read hidden states from many layers (Playground v3 feeds each LLM layer to the matching image-model layer), add trainable layers on top, or prepend a template.

**Pooled versus per-token.** A pooled vector is a cheap global summary for the modulation path; per-token embeddings are needed for binding and layout. Imagen, SDXL, SD3 and FLUX use both.

**Spelling.** Subword tokenizers hide letters. Liu et al.'s character-aware models (adding ByT5) beat character-blind competitors on visual spelling by 30+ points on rare words, and the DALL-E 3 report suspects T5 tokenization contributes to its unreliable text. SD3 measured T5's role: dropping it at inference left aesthetics unchanged (50% win rate) but lowered prompt adherence (46%) and typography (38%).

::: {.qa asked="Taotian AIGC image algorithm, Nowcoder 2025"}
**How would you train a model to render text in images accurately?**

Data: captions must transcribe visible text, usually in quotes, via OCR or a captioner trained to include words, plus synthetic rendered-text images for rare words and scripts. Encoder: a character-aware or large encoder (ByT5, T5-XXL, an LLM) with a per-token path; joint attention helps glyph layout. Resolution: small glyphs must survive the VAE and patchification, so check reconstruction of text and train at high resolution. Post-training: OCR-based rewards work well; Flow-GRPO raised SD3.5-M's text accuracy from 0.59 to 0.92. Evaluate OCR accuracy per script, since logographic scripts behave differently from Latin.
:::

## Captions, recaptioning and prompt rewriting

Web alt-text omits most of an image, so the model learns to ignore prompt details it could never ground. The DALL-E 3 report fixed this with data: train a captioner, fine-tune it to produce short (main subject) and descriptive (subject, surroundings, colors, visible text) captions, and recaption everything. Among blends of 65% to 95% synthetic captions, more synthetic scored better, and DALL-E 3 used 95% synthetic plus 5% ground truth. Because the model then expects long captions, GPT-4 "upsampled" short user prompts at inference. Prompt rewriting maps prompts into the training-caption distribution but can add unrequested content, override explicit constraints, homogenize style and add latency; mitigations are instructing the rewriter to keep every stated constraint, showing the rewritten prompt, and training on mixed caption lengths.

PixArt-$\alpha$ captioned SAM images with LLaVA: 29.3 nouns per image on average, versus 6.4 for LAION alt-text. SD3 used 50% CogVLM captions, lifting GenEval from 43.27 to 49.78. The failure mode is inheritance: captioner hallucinations become false labels (DALL-E 3 reports hallucinated plant and bird species) and captioner blind spots become generator blind spots (its spatial words were unreliable).

::: qa
**DALL-E 3 trained on 95% synthetic captions. Why not 100%?**

A captioner has stylistic habits (casing, punctuation, length, starting with "a") that the generator absorbs as requirements, so a model trained only on captioner text can break on human-written prompts. The 5% of real captions regularizes the conditioning toward human phrasing. The general lesson is to match inference prompts to the training caption distribution, through the data mix, through rewriting, or both.
:::

## Training practicalities

**Aspect ratio and micro-conditioning.** Square crops cut off heads and teach the model to crop; dropping images under 256 px would have removed 39% of SDXL's data. SDXL Fourier-embeds the original size $(h,w)$ and crop offsets $(c_{\text{top}}, c_{\text{left}})$ and adds them to the timestep embedding; at inference, crop $(0,0)$ and a large size give centered, sharp images. It also buckets images by aspect ratio at about $1024^2$ pixels with sides in multiples of 64, draws each batch from one bucket, and conditions on the target size.

**Positional encodings.** Absolute embeddings on a fixed grid must be interpolated for new sizes, which changes what each position means. SD3 builds 2D sinusoidal embeddings on an extended and interpolated grid and center-crops per aspect ratio. RoPE encodes relative position inside attention and works for any $H\times W$; FLUX uses 3D axial RoPE, FiT and Lumina-Next study resolution extrapolation, and NaViT packs native-resolution sequences. Far beyond trained sizes, models still tend to repeat objects.

**Resolution-dependent shift.** At high resolution neighboring pixels are redundant, so a given noise level destroys less information. SD3 remaps rectified-flow time ($x_t = (1-t)x_0 + t\epsilon$) from $n$ to $m$ pixels:

$$
t_m = \frac{\alpha\, t_n}{1 + (\alpha - 1)\, t_n},\quad \alpha = \sqrt{m/n}
\quad\Longrightarrow\quad \frac{1 - t_m}{t_m} = \frac{1}{\alpha}\cdot\frac{1 - t_n}{t_n},
$$

dividing the amplitude signal-to-noise ratio by $\alpha$; SD3 used a shift of 3.0 at $1024^2$ for training and sampling.

**Stages.** Tokens scale with $HW$ and attention with $(HW)^2$, so models learn composition cheaply at low resolution, then detail, then style. SDXL: 600K steps at 256 px, 200K at 512 px, then multi-aspect near 1024 px. SD3: 256 px pretraining, higher-resolution mixed aspect ratios, then DPO. PixArt-$\alpha$: pixel dependency (from an ImageNet DiT), text-image alignment (dense captions), aesthetics. Emu's quality-tuning on a few thousand curated images after 1.1B pairs won 82.9% of comparisons against its pretrained model, the image analogue of SFT.

**Filtering.** NSFW classifiers, aesthetic and CLIP-similarity thresholds, watermark detection, a resolution floor (or size conditioning) and deduplication; SD3's SSCD-based dedup cut potentially memorized examples 5x. Heavy aesthetic filtering costs concept coverage, so it is usually stronger in later stages.

::: {.qa asked="Taotian AIGC image algorithm, Nowcoder 2025"}
**What did SDXL change relative to SD 1.x/2.x, and why?**

A 3x larger U-Net (2.6B) with transformer blocks shifted to lower resolutions. Two text encoders (CLIP ViT-L and OpenCLIP ViT-bigG) with penultimate features concatenated to 2048 channels, plus the OpenCLIP pooled embedding as a global condition. Size and crop micro-conditioning, so small and cropped images become usable without teaching blur or cropping. Multi-aspect bucketed training near $1024^2$ after 256 and 512 px stages. An optional refiner, specialized on low-noise steps, that re-noises and denoises the base output SDEdit-style. A retrained autoencoder with larger batch and EMA weights.
:::

## Guidance with text

The CFG derivation is in [diffusion models](diffusion-models.html). Convention here: $\tilde\epsilon = \epsilon_\theta(x_t, \varnothing) + w\,[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)]$, so $w = 1$ is the plain conditional model (Ho and Salimans' $w'$ equals $w - 1$); flow models combine $v$ the same way. Training uses **conditioning dropout**: replace $c$ with a null (empty-string embedding or zeros) about 10% of the time (Imagen zeroed text embeddings 10%).

**Negative prompts** replace $\varnothing$ with $c_-$:

$$
\tilde\epsilon = \epsilon(x_t, c_-) + w\,[\epsilon(x_t, c) - \epsilon(x_t, c_-)]
\;\;\leftrightarrow\;\;
\nabla_x \log\!\big[p(x\mid c)^{w}\, p(x\mid c_-)^{1-w}\big].
$$

For $w > 1$ the exponent on $p(x\mid c_-)$ is negative, so regions likely under the negative prompt are suppressed. Ban et al. found negative prompts act with a delay, after the positive content appears, and remove concepts by cancellation in latent space.

::: pitfall
"A negative prompt subtracts a concept." It replaces the unconditional branch, so guidance pushes toward $c$ relative to $c_-$. At $w = 1$ it has no effect at all, and it can only remove what the model associates with the negative text.
:::

**High-guidance artifacts.** Large $w$ inflates the prediction and oversaturates. Imagen's dynamic thresholding clips $\hat x_0$ at a percentile $s$ and divides by $s$ (pixel space). Lin et al.'s CFG rescale works in latent space: $x = \phi\, x_{\text{cfg}}\,\sigma_{\text{pos}}/\sigma_{\text{cfg}} + (1-\phi)\,x_{\text{cfg}}$ with $\phi = 0.7$, where $\sigma$ are standard deviations of the conditional and guided predictions.

**Guidance distillation.** Meng et al. train a student $\epsilon_\eta(x_t, c, w)$ to match the teacher's guided output over a range of $w$, halving per-step cost. Black Forest Labs' model card says FLUX.1 [dev] was "trained using guidance distillation".

::: {.qa asked="Image-generation vision intern, Nowcoder 2024"}
**How is classifier-free guidance trained and used at inference in a text-to-image model?**

Train one network with the prompt replaced by a null embedding for a random ~10% of examples, so it learns $p(x\mid c)$ and $p(x)$ with shared weights. At each sampling step, run it on $(x_t, c)$ and $(x_t, \varnothing)$ as a batch of two and form $\epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$. The guided score is that of $p_t(x)\,p_t(c\mid x)^w$ at each noise level, so samples approximately follow this tilted density: sharper alignment, less diversity, saturation at large $w$. The null used at inference must match training (empty-string embedding versus zeros), and the second pass doubles compute.
:::

::: qa
**What do you lose with a guidance-distilled model?**

The unconditional branch no longer exists as a separate output, so negative prompts, guidance on extra conditions and tricks such as CFG rescale are not directly available; running two passes on a distilled model is outside its training distribution. Usable $w$ is limited to the range seen in distillation. Fine-tuning needs care: by reasoning (not a reported result), the plain denoising loss targets the unguided conditional prediction, not the guided output the student was distilled to produce, so it can pull the model away from its guidance behavior. In exchange, sampling is about twice as fast.
:::

## Personalization and control

### LoRA

For a frozen $W_0 \in \mathbb{R}^{d\times k}$, LoRA learns a rank-$r$ update:

$$
h = W_0 x + \frac{\alpha}{r} B A x,\qquad B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k},\; r \ll \min(d,k),
$$

with $r(d+k)$ parameters instead of $dk$ (2.5% for $d = k = 1280$, $r = 16$). $A$ starts Gaussian and $B$ at zero, so $\Delta W = 0$. The $\alpha/r$ factor keeps the update scale roughly independent of $r$; with Adam, tuning $\alpha$ is close to tuning the learning rate. With $G = \partial L/\partial h$:

$$
\frac{\partial L}{\partial B} = \frac{\alpha}{r}\, G\,(Ax)^\top,\qquad
\frac{\partial L}{\partial A} = \frac{\alpha}{r}\, B^\top G\, x^\top ,
$$

so at step 0 only $B$ moves. Merging $W = W_0 + \frac{\alpha}{r}BA$ removes inference overhead. Several LoRAs add as $W_0 + \sum_i s_i \frac{\alpha_i}{r_i} B_i A_i$ with user strengths $s_i$, and interfere when they touch the same directions. In image models LoRA usually targets attention projections.

### DreamBooth and Textual Inversion

**DreamBooth** fine-tunes the generator on 3 to 5 images captioned "a [V] [class noun]", where [V] is a rare token (real words carry meaning; random strings split into letter tokens). To fight *language drift* (forgetting the generic class) and lost diversity, it adds a prior-preservation term on about 1000 images the frozen model generated from "a [class noun]":

$$
\mathbb{E}\Big[w_t \big\|\hat x_\theta(\alpha_t x + \sigma_t \epsilon, c) - x\big\|^2
+ \lambda\, w_{t'} \big\|\hat x_\theta(\alpha_{t'} x_{\text{pr}} + \sigma_{t'} \epsilon', c_{\text{pr}}) - x_{\text{pr}}\big\|^2\Big],\quad \lambda = 1 .
$$

**Textual Inversion** freezes everything and learns one embedding $v_*$ for a pseudo-word: $v_* = \arg\min_v \mathbb{E}\,\|\epsilon - \epsilon_\theta(z_t, t, c_\theta(y; v))\|^2$. It is tiny but can only reach what the frozen model already expresses, so identity fidelity is lower.

### ControlNet, T2I-Adapter, IP-Adapter

**ControlNet** copies the U-Net encoder (12 blocks plus the middle block) as a trainable branch fed with a spatial condition (edges, depth, pose) and adds its outputs to the frozen U-Net's skip connections through zero convolutions $\mathcal{Z}$, 1x1 convs with zero weights and biases:

$$
y_c = \mathcal{F}(x;\Theta) + \mathcal{Z}\big(\mathcal{F}(x + \mathcal{Z}(c;\Theta_{z1});\Theta_c);\Theta_{z2}\big).
$$

At step 0, $y_c = \mathcal{F}(x;\Theta)$ exactly. Training replaces 50% of prompts with empty strings, and the paper reports "sudden convergence", typically within 10K steps. Cost is roughly an extra encoder pass per sampling step. **T2I-Adapter** is a ~77M-parameter conv network adding condition features to the encoder at four scales; adapters compose as weighted sums, and since it has no time input, training samples $t$ with a cubic schedule biased toward noisy steps. **IP-Adapter** adds an image prompt via *decoupled cross-attention*, with a shared query and new key/value projections (initialized from the text ones) for image tokens $c_i$:

$$
Z = \mathrm{Attn}(Q, c_t W_K, c_t W_V) + \lambda\, \mathrm{Attn}(Q, c_i W'_K, c_i W'_V),
$$

where $c_i$ is an OpenCLIP ViT-H/14 image embedding projected to 4 tokens. About 22M parameters train; the base stays frozen, so it transfers to community fine-tunes, and $\lambda = 0$ recovers the original model.

::: {.qa asked="Baidu multimodal algorithm intern, Nowcoder 2026"}
**Write the LoRA formula. How are A and B initialized, and could you instead set A = 0 and B random?**

$h = W_0x + \frac{\alpha}{r}BAx$ with $A$ Gaussian and $B = 0$, so training starts exactly at the pretrained model. The swap also starts at $\Delta W = 0$ and still trains: with $A = 0$, $\partial L/\partial B = 0$ but $\partial L/\partial A = \frac{\alpha}{r}B^\top G x^\top \neq 0$. Setting both to zero fails, since both gradients vanish. The schemes are not equivalent in practice: Hayou et al. show the standard one tolerates larger learning rates without instability and performs better on average. PiSSA instead initializes $A, B$ from the top singular vectors of $W_0$ and freezes the residual.
:::

::: {.qa asked="Image-generation vision intern, Nowcoder 2024"}
**Can a LoRA achieve the same effect as prompt engineering?**

Partly. A prompt only moves the conditioning input, so it reaches outputs the frozen model already supports, such as known styles and compositions. A LoRA changes the weights, so it can add concepts the model never learned (a specific face, product or art style) and make a behavior reliable without long prompts. A LoRA can imitate a prompt, but that costs training and may reduce diversity or leak style into unrelated prompts. Try prompting first (cheap and reversible), then Textual Inversion, then LoRA or DreamBooth when the concept is outside the model's range.
:::

::: {.qa asked="Image-generation vision intern, Nowcoder 2024"}
**Compare LoRA, DreamBooth and Textual Inversion for personalization.**

Textual Inversion learns one embedding with everything frozen: tiny and harmless to the model, but bounded by its existing range, so fidelity is weakest. DreamBooth updates all weights with a rare-token identifier and prior preservation: highest fidelity, but a full checkpoint per subject and real overfitting and drift risk. LoRA restricts the update to low-rank factors: a few MB, mergeable and composable, with fidelity close to full fine-tuning. In practice the DreamBooth recipe is now usually trained through LoRA. Training captions are often produced by a captioner such as BLIP, with the identifier inserted.
:::

::: qa
**Why does ControlNet use zero convolutions? If the weights start at zero, how does anything learn?**

A random connection would inject large random features into a frozen, finely balanced network at step 0 and send noisy gradients into the copied encoder. Zero init makes the model exactly the pretrained one at the start. It still learns because a linear layer's weight gradient is the upstream gradient times its input, $(\partial L/\partial y)\,u^\top$, nonzero even when $W = 0$. After one step $W \neq 0$ and gradients reach the copy, which starts from pretrained rather than random features.
:::

::: qa
**IP-Adapter versus ControlNet: what does each control, and why does IP-Adapter decouple its cross-attention?**

ControlNet controls spatial structure: a pixel-aligned map whose features are added location by location to the skip connections. IP-Adapter controls content and style: a global CLIP image embedding entering through cross-attention, with no pixel alignment, like an image-valued prompt. They compose (pose from ControlNet, identity or style from IP-Adapter). If image tokens were concatenated to text tokens in the original cross-attention, they would pass through projections trained for text and compete with words in one softmax. Separate projections and a separate softmax let the image branch specialize, $\lambda$ trades image against text, and $\lambda = 0$ restores the base model.
:::

## Editing

**SDEdit** noises a guide image (photo or rough painting) to time $t_0$ and denoises; larger $t_0$ is more realistic but less faithful, and $t_0 \in [0.3, 0.6]$ suited stroke guides. It needs no training but cannot preserve exact detail.

**Inversion.** Real-image editing needs a latent that reconstructs the image. DDIM inversion runs the sampler backwards assuming $\epsilon_\theta(x_{t+1}) \approx \epsilon_\theta(x_t)$:
$x_{t+1} = \sqrt{\bar\alpha_{t+1}}\,\hat x_0(x_t) + \sqrt{1-\bar\alpha_{t+1}}\,\epsilon_\theta(x_t, t, c)$.
The per-step error is small without guidance, but CFG near $w = 7.5$ amplifies it.

**Prompt-to-Prompt** generates source (prompt $P$, cross-attention maps $M_t$) and target ($P^*$, maps $M^*_t$) with the same seed. Word swap uses $M_t$ while $t \ge \tau$ and $M^*_t$ after, injecting source maps during the early steps that fix layout. Adding a phrase injects maps only for shared tokens, and re-weighting scales one word's column. **Null-text inversion** makes this work on real images: invert at $w = 1$ to get a pivot trajectory $z^*_t$, then at $w = 7.5$ optimize a per-step null embedding, $\min_{\varnothing_t} \|z^*_{t-1} - z_{t-1}(\bar z_t, \varnothing_t, C)\|^2$. Weights and prompt embedding stay fixed, keeping the attention maps meaningful.

**InstructPix2Pix** trains an editor on synthetic data: GPT-3, fine-tuned on 700 human-written examples, writes caption, instruction and edited caption; Stable Diffusion with Prompt-to-Prompt renders before/after pairs; CLIP filters leave 454,445 examples. The source image is channel-concatenated, and two guidance scales are used (derivation below).

**Instruction and in-context editors.** Emu Edit multi-tasks across editing and vision tasks with learned task embeddings, and MagicBrush supplied manually annotated edit data. In 2025 examples, FLUX.1 Kontext appends context-image tokens to the sequence with a RoPE offset (beating channel concatenation in its tests); Qwen-Image's editor feeds the input to both Qwen2.5-VL (semantics) and the VAE (appearance); Step1X-Edit pairs a multimodal LLM with a diffusion decoder; and Gemini 2.5 Flash Image advertises character consistency and multi-image fusion. The shared pattern: the source image becomes context, and data quality decides results.

::: qa
**Derive InstructPix2Pix's two-scale guidance. What do $s_I$ and $s_T$ control?**

By Bayes, $p(z\mid c_I, c_T) \propto p(c_T\mid c_I, z)\,p(c_I\mid z)\,p(z)$, so the score is $\nabla\log p(z) + \nabla\log p(c_I\mid z) + \nabla\log p(c_T\mid c_I,z)$. Each implicit-classifier term is a difference of model scores: $\nabla\log p(c_I\mid z) = \nabla\log p(z\mid c_I) - \nabla\log p(z)$ and $\nabla\log p(c_T\mid c_I,z) = \nabla\log p(z\mid c_I,c_T) - \nabla\log p(z\mid c_I)$. Scaling them separately gives
$\tilde e = e(z,\varnothing,\varnothing) + s_I[e(z,c_I,\varnothing) - e(z,\varnothing,\varnothing)] + s_T[e(z,c_I,c_T) - e(z,c_I,\varnothing)]$.
$s_I$ sets faithfulness to the input and $s_T$ edit strength (typical: $s_I \in [1, 1.5]$, $s_T \in [5, 10]$). One network provides all terms because training drops $c_I$, $c_T$ and both for 5% of examples each.
:::

::: {.qa asked="Meituan CV intern, Nowcoder 2024"}
**How would you remove an unwanted object from a street photo so the result looks real?**

Get a mask (click plus a segmentation model, dilated to cover shadows and reflections), then use a dedicated inpainting model whose input concatenates noisy latent, mask and masked image, or an instruction editor. Masking a plain text-to-image model blends poorly and tends to hallucinate a new object, so prompt for the background ("empty sidewalk") or add the object class as a negative prompt. Paste unmasked pixels back after decoding so the VAE does not alter them, and check leftover shadows and continuity of lines and perspective.
:::

## Unified understanding and generation

| Model | Generation representation | Image objective | Image-token attention | Notes |
|---|---|---|---|---|
| Chameleon (2024) | VQ tokens, 1024 per 512 px image | next-token CE | causal | early fusion; QK-norm, z-loss for stability |
| Transfusion (2024) | VAE-latent patches | diffusion; $\mathcal{L}_{\text{LM}} + 5\,\mathcal{L}_{\text{DDPM}}$ | bidirectional within image | 7B, ~2T tokens |
| Show-o (2024) | MAGVIT-v2 tokens | masked-token prediction | full within image | Phi-1.5 base |
| Janus (2024) | VQ tokens; SigLIP for understanding | next-token CE | causal | decoupled encoders; Janus-Pro scales it |
| BAGEL (2025) | FLUX VAE latents; SigLIP2 for understanding | rectified flow; CE:MSE = 0.25:1 | shared attention, two experts | 7B active / 14B total, interleaved data |

Chameleon's tokens give one loss and interleaved output, but it reports weak reconstruction of text-heavy images and needed stability fixes because modalities competed through growing norms. Transfusion keeps images continuous; against its own data- and compute-controlled Chameleon-style baseline it reports exceeding the baseline on text-to-image "at less than a third of the compute", and matching its FID with about 34x less compute. BAGEL's Mixture-of-Transformers (separate weights for understanding and generation tokens, shared self-attention) beat dense and MoE variants in a controlled 1.5B study, and intelligent editing needed about 3.6T training tokens to reach 85% of its peak, versus about 0.18T for understanding.

::: qa
**Discrete tokens versus continuous diffusion inside the LM: what are the trade-offs, and why does Janus use two visual encoders?**

Discrete tokens give one loss, one sampler and LLM infrastructure unchanged, but pay with quantization error (small text, faces), slow sequential decoding of ~1000 tokens, and a quality ceiling set by the tokenizer. Continuous latents with a diffusion loss keep fidelity and denoise all patches in parallel, but require balancing two losses ($\lambda = 5$ in Transfusion), a mixed attention mask, and handling of noisy versus clean image copies in context. Separately, understanding wants semantic features and generation wants reconstructive detail; Janus found a shared encoder hurt understanding, so it reads with SigLIP and writes with a VQ tokenizer, a split BAGEL keeps.
:::

## Post-training and alignment

**Rewards.** Learned preference models: ImageReward (BLIP-based, ~137K expert comparisons), PickScore (Pick-a-Pic), HPSv2, the LAION aesthetic predictor. Rule-based: object detectors (GenEval), OCR. Metrics are covered in [evaluation metrics](evaluation-metrics.html).

**Backpropagating a differentiable reward.** Maximize $\mathbb{E}_{c,x_T}[r(x_0^\theta(x_T, c), c)]$ through the sampler. DRaFT backpropagates through the full chain with checkpointing and LoRA; DRaFT-K truncates to the last $K$ steps, and $K = 1$ was most compute-efficient because long-chain gradients explode; AlignProp is similar. ReFL samples to a random late step, scores the one-shot $\hat x_0$, and mixes in the pretraining loss. These are low-variance but need differentiable rewards and exploit them fast.

**Diffusion-DPO.** DPO solves KL-regularized reward maximization in closed form, $r = \beta\log(\pi^*/\pi_{\text{ref}}) + \text{const}$, and fits it with a Bradley-Terry loss. A diffusion marginal $p_\theta(x_0)$ is intractable, so Wallace et al. define the ratio on whole chains $x_{0:T}$, move the chain expectation outside the convex $-\log\sigma$ with Jensen (an upper bound), replace the reverse posterior with the forward process $q$, and reduce per-step Gaussian KLs to noise-prediction errors. For $x^w_t, x^l_t \sim q(x_t\mid x_0)$:

$$
\mathcal{L} = -\mathbb{E}\log\sigma\Big(-\beta T\omega(\lambda_t)\big[
\underbrace{\|\epsilon^w - \epsilon_\theta(x_t^w,t)\|^2 - \|\epsilon^w - \epsilon_{\text{ref}}(x_t^w,t)\|^2}_{\Delta_w}
- \underbrace{\big(\|\epsilon^l - \epsilon_\theta(x_t^l,t)\|^2 - \|\epsilon^l - \epsilon_{\text{ref}}(x_t^l,t)\|^2\big)}_{\Delta_l}\big]\Big).
$$

$\Delta_w < 0$ means the model denoises the winner better than the reference; the loss pushes $\Delta_w - \Delta_l$ down. It is offline and reward-model-free; SDXL was tuned on 851K Pick-a-Pic pairs with $\beta = 5000$.

**RL.** DDPO treats denoising as an MDP (state $(c,t,x_t)$, action $x_{t-1}$, terminal reward $r(x_0,c)$). Each step is Gaussian, so

$$
\nabla_\theta J = \mathbb{E}\Big[r(x_0,c)\sum_{t=1}^{T}\nabla_\theta\log p_\theta(x_{t-1}\mid x_t,c)\Big],
$$

with an importance-sampled, PPO-clipped variant for several updates per batch. DPOK adds a KL penalty bounded by per-step KLs, each $\|\mu_\theta - \mu_{\text{pre}}\|^2/2\sigma_t^2$, and found online RL beat reward-weighted fine-tuning. Flow-GRPO adapts GRPO to rectified flow, whose ODE sampler has no per-step randomness to explore with. Using the score $\nabla\log p_t(x_t) = -(x_t + (1-t)v)/t$ for $x_t = (1-t)x_0 + t\epsilon$, it samples with a marginal-preserving reverse-time SDE, run from $t = 1$ to $0$ (so $dt < 0$), namely $dx = [v - \tfrac{\sigma_t^2}{2}\nabla\log p_t]\,dt + \sigma_t\, d\bar w_t$ with the score substituted:

$$
dx_t = \Big[v_\theta + \frac{\sigma_t^2}{2t}\big(x_t + (1-t)\,v_\theta\big)\Big]dt + \sigma_t\, d\bar w_t,\qquad \sigma_t = a\sqrt{\tfrac{t}{1-t}},
$$

so each step is Gaussian with closed-form ratios and KL. It samples $G$ images per prompt, uses advantages $\hat A_i = (R_i - \mathrm{mean}(R))/\mathrm{std}(R)$, clipping and KL, and trains with 10-step rollouts while sampling with 40. On SD3.5-M, GenEval rose from 0.63 to 0.95.

::: {.qa asked="Kuaishou multimodal LLM algorithm engineer, Nowcoder 2026"}
**Is DPO closer to SFT or to GRPO?**

In objective it is closer to RL: it solves the same KL-regularized reward maximization as RLHF, with the reward implicit in $\log\pi_\theta/\pi_{\text{ref}}$ under a Bradley-Terry model. In training dynamics it is closer to SFT: offline, fixed pairs, no sampling from the current policy, no exploration. Diffusion-DPO trains on fixed Pick-a-Pic pairs with a denoising-style loss, while Flow-GRPO samples and scores fresh images every iteration. So DPO is cheaper and more stable but limited by data coverage; online methods can find new high-reward outputs, and new reward hacks.
:::

::: {.qa asked="Kuaishou multimodal LLM algorithm engineer, Nowcoder 2026"}
**In GRPO, what happens when every sample in a group gets the same reward?**

All advantages $(R_i - \bar R)/(\mathrm{std}(R) + \varepsilon)$ are zero, so the policy-gradient term vanishes and the prompt contributes only the KL term; its sampling compute is wasted. Binary rewards (correct object count, exact OCR match) make this common on too-easy or too-hard prompts, and more common as training improves. Remedies: filter and resample such groups (DAPO's dynamic sampling), use graded rewards, increase $G$, or schedule prompt difficulty.
:::

::: qa
**What reward hacking appears in diffusion RL, and how do you mitigate it?**

DDPO's incompressibility reward ended in high-frequency noise, and its LLaVA counting reward was gamed by a typographic attack: writing text resembling the requested number instead of drawing that many animals. DRaFT's aesthetic runs collapsed to a few similar images, and Flow-GRPO without KL converged to one style. Mitigations: early stopping on held-out evaluation, KL to the reference (Flow-GRPO notes it is not equivalent to early stopping but trains slower), LoRA with scaled-down merges (DRaFT's best), pretraining-loss mixing (ReFL), multiple rewards, and checking diversity and human ratings not used in training.
:::

## Rapid-fire review

::: qa
**What is CLIP's text length limit?**

77 tokens including start and end tokens; standard pipelines truncate beyond it. Workarounds are chunked embeddings or a longer-context encoder such as T5.
:::

::: qa
**Why normalize queries and keys in SD3 and Chameleon?**

Growing attention logits collapse attention entropy and cause divergence, especially in bf16. SD3 uses RMSNorm on Q and K; Chameleon adds z-loss and norm reordering.
:::

::: qa
**What does SDXL's crop conditioning fix?**

Random crops teach the model to generate cut-off subjects. Conditioning on crop offsets and setting them to $(0,0)$ at inference gives centered compositions while keeping crop augmentation.
:::

::: qa
**What is language drift in DreamBooth?**

Fine-tuning on "a [V] dog" makes the model forget generic dogs and lose diversity. Prior preservation on frozen-model samples of "a dog" counters it.
:::

::: qa
**When would you not merge a LoRA into the base weights?**

When switching or reweighting adapters per request, serving many adapters from one base, or when the base is quantized. Merging several is plain addition, but they can interfere.
:::

::: qa
**Why does ControlNet drop half the prompts in training?**

With an empty prompt the only source of semantics is the condition image, forcing the branch to recognize content from edges or depth.
:::

::: qa
**Why does T2I-Adapter sample timesteps with a cubic schedule?**

It has no time input, and control matters most in early, noisy steps where layout forms; uniform sampling gave weak guidance.
:::

::: qa
**Why does high CFG oversaturate, and two fixes?**

$w(\epsilon_c - \epsilon_\varnothing)$ inflates the prediction and pushes $\hat x_0$ out of range. Dynamic thresholding clips and rescales in pixel space; CFG rescale matches the guided prediction's standard deviation to the conditional one.
:::

::: qa
**Why does ReFL score a late denoising step?**

Early one-shot $\hat x_0$ predictions are blurry and ImageReward scores barely separate samples there; they became distinguishable only near the end of a 40-step chain.
:::

::: qa
**Why does Flow-GRPO train with 10 steps but sample with 40?**

Rollouts dominate cost and the reward only needs a recognizable image; this sped data collection over 4x without hurting final reward.
:::

::: qa
**Why does null-text inversion tune the null embedding, not the prompt?**

With CFG the unconditional branch strongly shapes every step, so it can steer reconstruction; tuning the prompt embedding would distort the cross-attention maps that Prompt-to-Prompt edits.
:::

## References

1. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," 2021. <https://arxiv.org/abs/2112.10752>
2. Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Imagen), 2022. <https://arxiv.org/abs/2205.11487>
3. Betker et al., "Improving Image Generation with Better Captions" (DALL-E 3), OpenAI, 2023. <https://cdn.openai.com/papers/dall-e-3.pdf>
4. Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis," 2023. <https://arxiv.org/abs/2307.01952>
5. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3), 2024. <https://arxiv.org/abs/2403.03206>
6. Peebles and Xie, "Scalable Diffusion Models with Transformers" (DiT), 2022. <https://arxiv.org/abs/2212.09748>
7. Chen et al., "PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis," 2023. <https://arxiv.org/abs/2310.00426>
8. Black Forest Labs, "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space," 2025. <https://arxiv.org/abs/2506.15742>
9. Black Forest Labs, FLUX.1 reference implementation. <https://github.com/black-forest-labs/flux>
10. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021. <https://arxiv.org/abs/2103.00020>
11. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5), 2019. <https://arxiv.org/abs/1910.10683>
12. Yuksekgonul et al., "When and Why Vision-Language Models Behave like Bags-of-Words, and What to Do About It?," 2022. <https://arxiv.org/abs/2210.01936>
13. Liu et al., "Character-Aware Models Improve Visual Text Rendering," 2022. <https://arxiv.org/abs/2212.10562>
14. Liu et al., "Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models," 2024. <https://arxiv.org/abs/2409.10695>
15. Xie et al., "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers," 2024. <https://arxiv.org/abs/2410.10629>
16. Wu et al., "Qwen-Image Technical Report," 2025. <https://arxiv.org/abs/2508.02324>
17. Dai et al., "Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack," 2023. <https://arxiv.org/abs/2309.15807>
18. Dehghani et al., "Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution," 2023. <https://arxiv.org/abs/2307.06304>
19. Lu et al., "FiT: Flexible Vision Transformer for Diffusion Model," 2024. <https://arxiv.org/abs/2402.12376>
20. Zhuo et al., "Lumina-Next: Making Lumina-T2X Stronger and Faster with Next-DiT," 2024. <https://arxiv.org/abs/2406.18583>
21. Ho and Salimans, "Classifier-Free Diffusion Guidance," 2022. <https://arxiv.org/abs/2207.12598>
22. Meng et al., "On Distillation of Guided Diffusion Models," 2022. <https://arxiv.org/abs/2210.03142>
23. Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed," 2023. <https://arxiv.org/abs/2305.08891>
24. Ban et al., "Understanding the Impact of Negative Prompts: When and How Do They Take Effect?," 2024. <https://arxiv.org/abs/2406.02965>
25. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021. <https://arxiv.org/abs/2106.09685>
26. Hayou et al., "The Impact of Initialization on LoRA Finetuning Dynamics," 2024. <https://arxiv.org/abs/2406.08447>
27. Meng et al., "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models," 2024. <https://arxiv.org/abs/2404.02948>
28. Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation," 2022. <https://arxiv.org/abs/2208.12242>
29. Gal et al., "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion," 2022. <https://arxiv.org/abs/2208.01618>
30. Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet), 2023. <https://arxiv.org/abs/2302.05543>
31. Mou et al., "T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models," 2023. <https://arxiv.org/abs/2302.08453>
32. Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models," 2023. <https://arxiv.org/abs/2308.06721>
33. Meng et al., "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations," 2021. <https://arxiv.org/abs/2108.01073>
34. Hertz et al., "Prompt-to-Prompt Image Editing with Cross Attention Control," 2022. <https://arxiv.org/abs/2208.01626>
35. Mokady et al., "Null-text Inversion for Editing Real Images using Guided Diffusion Models," 2022. <https://arxiv.org/abs/2211.09794>
36. Brooks et al., "InstructPix2Pix: Learning to Follow Image Editing Instructions," 2022. <https://arxiv.org/abs/2211.09800>
37. Sheynin et al., "Emu Edit: Precise Image Editing via Recognition and Generation Tasks," 2023. <https://arxiv.org/abs/2311.10089>
38. Zhang et al., "MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing," 2023. <https://arxiv.org/abs/2306.10012>
39. Liu et al., "Step1X-Edit: A Practical Framework for General Image Editing," 2025. <https://arxiv.org/abs/2504.17761>
40. Google, "Introducing Gemini 2.5 Flash Image," 2025. <https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/>
41. Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models," 2024. <https://arxiv.org/abs/2405.09818>
42. Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model," 2024. <https://arxiv.org/abs/2408.11039>
43. Xie et al., "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation," 2024. <https://arxiv.org/abs/2408.12528>
44. Wu et al., "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation," 2024. <https://arxiv.org/abs/2410.13848>
45. Chen et al., "Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling," 2025. <https://arxiv.org/abs/2501.17811>
46. Deng et al., "Emerging Properties in Unified Multimodal Pretraining" (BAGEL), 2025. <https://arxiv.org/abs/2505.14683>
47. Xu et al., "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation," 2023. <https://arxiv.org/abs/2304.05977>
48. Kirstain et al., "Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation," 2023. <https://arxiv.org/abs/2305.01569>
49. Wu et al., "Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis," 2023. <https://arxiv.org/abs/2306.09341>
50. Ghosh et al., "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment," 2023. <https://arxiv.org/abs/2310.11513>
51. Clark et al., "Directly Fine-Tuning Diffusion Models on Differentiable Rewards" (DRaFT), 2023. <https://arxiv.org/abs/2309.17400>
52. Prabhudesai et al., "Aligning Text-to-Image Diffusion Models with Reward Backpropagation" (AlignProp), 2023. <https://arxiv.org/abs/2310.03739>
53. Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization," 2023. <https://arxiv.org/abs/2311.12908>
54. Black et al., "Training Diffusion Models with Reinforcement Learning" (DDPO), 2023. <https://arxiv.org/abs/2305.13301>
55. Fan et al., "DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models," 2023. <https://arxiv.org/abs/2305.16381>
56. Liu et al., "Flow-GRPO: Training Flow Matching Models via Online RL," 2025. <https://arxiv.org/abs/2505.05470>
57. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO), 2024. <https://arxiv.org/abs/2402.03300>
58. Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale," 2025. <https://arxiv.org/abs/2503.14476>
59. Interview reports cited in `asked=` tags (Nowcoder): image-generation intern 2024 <https://www.nowcoder.com/feed/main/detail/2b94140c0cc54718b4cd1c8ba02d35a2>; Baidu multimodal intern 2026 <https://www.nowcoder.com/feed/main/detail/32dd1dff10bd4cce90d314be29793000>; Taotian AIGC 2025 <https://www.nowcoder.com/discuss/730099920601907200>; Meituan CV 2024 <https://www.nowcoder.com/feed/main/detail/0d1475aec6ec4f4d92a16bc71d7bd7f9>; Kuaishou multimodal 2026 <https://www.nowcoder.com/discuss/942532364704534528>
