---
title: "Video Generation"
subtitle: "How text- and image-to-video models compress, model, condition, train, speed up and evaluate spacetime data, from 3D U-Nets to video DiTs and autoregressive world models."
order: 8
updated: 2026-11-08
---

Video generation extends image diffusion along time. That one extra axis multiplies the token count by one to two orders of magnitude, adds new errors (flicker, morphing, implausible motion), and makes good training data scarcer. Most design choices in modern video models respond to these pressures: spatiotemporal autoencoders, factorized or sparse attention, cascades, staged curricula and step distillation. Interviewers like the topic because it tests reasoning about compute, about train-test gaps, and about evaluation when the standard metric is weak. This page assumes [diffusion models](diffusion-models.html) and [flow matching](flow-matching.html); text conditioning is covered in [text-to-image and multimodal](text-to-image-multimodal.html).

## Why video is hard

### Tokens and compute

A latent video DiT sees a sequence whose length is set by the autoencoder and the patch size. With temporal factor $c_t$, spatial factor $c_s$, patch $p_t\times p\times p$, and a causal autoencoder that keeps the first frame separately (explained below), an $F$-frame $H\times W$ clip becomes

$$
L \;=\; \frac{1}{p_t}\Big(1+\frac{F-1}{c_t}\Big)\cdot\frac{H}{c_s\,p}\cdot\frac{W}{c_s\,p}.
$$

Worked examples at settings used in each report (factors written time x height x width throughout):

| Setting | Latent grid $T\times h\times w$ | Patch | Tokens $L$ |
|---|---|---|---|
| 1024x1024 image, 8x VAE | $1\times128\times128$ | $2\times2$ | 4,096 |
| Wan 2.1: 81 frames, 1280x720, VAE $4\times8\times8$ | $21\times90\times160$ | $1\times2\times2$ | 75,600 |
| HunyuanVideo: 129 frames, 1280x720, VAE $4\times8\times8$ | $33\times90\times160$ | $1\times2\times2$ | 118,800 |
| Movie Gen: 256 frames, 768x768, TAE $8\times8\times8$ | $32\times96\times96$ | $1\times2\times2$ | 73,728 |
| LTX-Video: 121 frames, 768x512, VAE $8\times32\times32$ | $16\times16\times24$ | none | 6,144 |

The Movie Gen row matches the "73K video tokens" context stated in its report [11].

Why attention dominates: per layer of width $d$, $QK^\top$ and $AV$ cost about $2L^2d$ FLOPs each, so attention costs $\approx4L^2d$, while the QKV/output projections ($8Ld^2$) and a 4x MLP ($16Ld^2$) cost $\approx24Ld^2$. Hence

$$
\frac{\text{attention}}{\text{linear}} \approx \frac{4L^2d}{24Ld^2} = \frac{L}{6d}.
$$

For HunyuanVideo ($d=3072$) the crossover is $L=6d\approx18{,}000$. A rough estimate, assuming dense attention over all video tokens, one attention layer plus one 4x MLP per block, and ignoring text tokens, softmax, norms and modulation, puts attention at $\frac{L/6d}{1+L/6d}\approx87\%$ of FLOPs for $L\approx1.2\times10^5$. That is in line with the Sliding Tile Attention paper's measurement that attention took about 800 of 945 seconds of wall-clock time for a 5 s 720p HunyuanVideo clip [21].

### Beyond compute

- **Temporal consistency.** Identity, texture and lighting must persist; frames sampled independently flicker because each is a separate draw from a wide conditional distribution.
- **Motion realism.** Dynamics (gravity, contact, articulation) are weakly constrained by appearance. Nearly static video is a common shortcut that scores well on consistency metrics.
- **Data.** Text-video data is smaller and noisier (cuts, watermarks, overlaid text, slideshows) and harder to caption, so most models pre-train on images and train jointly on images and videos.

::: key
Video cost is quadratic in a token count roughly 18-30x larger than a high-resolution image's. Most architecture choices either shrink $L$ (stronger autoencoders, cascades) or avoid the $L^2$ term (factorized, local or sparse attention).
:::

::: qa
**Estimate the sequence length for 5 seconds of 720p video, and say what dominates the cost.**

Take 81 frames at 16 fps, a causal $4\times8\times8$ VAE and a $1\times2\times2$ patch: $1+80/4=21$ latent frames, a $90\times160$ latent grid, $45\times80=3600$ tokens per latent frame, so $L=75{,}600$. That is about 18x a 1024px image's 4,096 tokens, so full attention costs about $18^2\approx340$ times more per layer. Since attention/linear FLOPs scale as $L/(6d)$, attention dominates once $L$ exceeds about $6d$ (roughly 18,000 tokens for $d=3072$). Reducing $L$ or making attention sub-quadratic therefore matters more than shrinking the MLPs.
:::

## Video autoencoders

### From per-frame to causal 3D VAEs

Early latent video models applied an image VAE per frame: no temporal compression, and the decoder flickers because frames are decoded independently (Video LDM fine-tuned its decoder on video for this reason [19]). Current models train 3D VAEs that also downsample time.

A **temporally causal 3D convolution** pads $k_t-1$ frames before the input and none after, so each output depends only on current and past frames [14]. Consequences:

1. The first frame is encoded on its own, so an image is a 1-frame video and one autoencoder serves both, enabling joint image-video training of the generator.
2. With temporal stride $s$, $1+sk$ frames map to $1+k$ latents. Hence the default lengths 49 (CogVideoX), 81 (Wan), 129 (HunyuanVideo), and 121 (LTX-Video, $s=8$).
3. Encoding can run chunk by chunk: Wan's VAE processes 4-frame chunks and caches the previous chunk's convolution inputs, so memory does not grow with length [9].

Movie Gen's TAE is not causal; it compresses $8\times$ along every axis [11].

### Compression ratios and the trade-off

Pixel values per latent value is $c_t c_s^2\cdot3/C$:

- CogVideoX, HunyuanVideo, Wan 2.1: $4\times8\times8$, $C=16$, so $\tfrac{256\cdot3}{16}=48$, i.e. 1:48.
- Movie Gen: $8\times8\times8$, $C=16$: 1:96.
- LTX-Video: $8\times32\times32$, $C=128$: 1:192, with no transformer patchify [10].

More compression means fewer tokens but harder reconstruction and often harder generation. In CogVideoX's ablation, going from 16 to 32 channels at $4\times8\times8$ raised reconstruction PSNR from 28.7 to 30.5, while an $8\times16\times16$ VAE with 128 channels made diffusion training converge with great difficulty [6]. Higher-capacity latents decode better and are harder to model, the same tension as for image latents ([GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html)). LTX-Video compensates for extreme compression by conditioning its decoder on a noise level in $[0,0.2]$ so the decoder performs the last denoising step in pixel space [10]. Movie Gen found high-norm latent "dots" that decode to spot artifacts and removed them with a hinge outlier penalty $\max\big(0,\lVert X_{ij}-\bar X\rVert-r\lVert\mathrm{Std}(X)\rVert\big)$ averaged over positions, $r=3$ [11].

::: pitfall
"Higher compression is better because it saves compute." Reconstruction PSNR and generation quality can move in opposite directions; judge an autoencoder by the downstream generator at fixed compute.
:::

::: qa
**Why do video VAEs use causal temporal convolutions and encode the first frame separately?**

Causal padding means the latent for frame $t$ never depends on later frames, so the first latent depends only on the first frame. An image is then a valid 1-frame video, which lets one autoencoder and one generator train on images and videos together, and it explains the $1+4k$ frame counts. Causality also permits chunked encoding and decoding with a feature cache, bounding memory for long clips, and it suits generators that produce latents left to right. The cost is that the first latent has different statistics (no temporal pooling), which can show up later, for example in rolling-cache long-video generation (see Self Forcing).
:::

::: qa
**Your new VAE reconstructs better than the old one, but the diffusion model trained on it is worse. Why might that be?**

Better reconstruction often comes from more channels or weaker regularization, which makes the latent distribution higher-entropy and harder to model at a fixed budget; CogVideoX saw PSNR improve with channels while very aggressive, wide latents hurt diffusion convergence. Other suspects are latent outliers or badly scaled channels (Movie Gen's spot artifacts) and a noise schedule tuned for the old latent's statistics. I would compare generator loss and samples at equal FLOPs, inspect per-channel latent statistics, and check whether a shifted noise schedule closes the gap.
:::

## Architectures

### U-Net era: inflate an image model

**Video Diffusion Models** (Ho et al., 2022) factorized a 3D U-Net over space and time: each $3\times3$ convolution became $1\times3\times3$, and a temporal attention block with relative position embeddings followed each spatial attention block [1]. For joint image-video training, independent images were appended to each video with temporal attention masked.

To extend a sample, VDM generates a block $x^b$ conditioned on a known block $x^a$. The "replacement" method (noise $x^a$ to the current level and let the model fill in $x^b$) produced blocks incoherent with $x^a$. With $z_t=\alpha_tx+\sigma_t\epsilon$ and Tweedie's formula $\mathbb{E}[x\mid z_t]=(z_t+\sigma_t^2\nabla\log p(z_t))/\alpha_t$, conditioning on $x^a$ adds $\log q(x^a\mid z_t)$ to the log-density:

$$
\mathbb{E}[x^b\mid z_t,x^a] = \mathbb{E}[x^b\mid z_t] + \frac{\sigma_t^2}{\alpha_t}\nabla_{z^b_t}\log q(x^a\mid z_t).
$$

Replacement drops the second term. Approximating $q(x^a\mid z_t)\approx\mathcal{N}\big(\hat x^a_\theta(z_t),\tfrac{\sigma_t^2}{\alpha_t^2}I\big)$ and adding a weight $w_r$ (values above 1 worked better) gives **reconstruction guidance**:

$$
\tilde x^b_\theta(z_t) = \hat x^b_\theta(z_t) - \frac{w_r\,\alpha_t}{2}\nabla_{z^b_t}\big\lVert x^a-\hat x^a_\theta(z_t)\big\rVert_2^2 .
$$

**Imagen Video** (2022) used a cascade of 7 models (a 5.6B base model at 16 frames, 40x24, 3 fps, then 3 spatial and 3 temporal super-resolution models; 11.6B diffusion parameters) with a frozen T5-XXL, reaching 128 frames at 1280x768, 24 fps [2]. It used v-prediction, temporal convolutions rather than attention in the SR models, and progressive distillation to 8 steps per model. **Make-A-Video** (2022) used no text-video pairs: text alignment came from a text-to-image prior and motion from unlabeled video, through pseudo-3D layers (a 1D temporal convolution after each 2D convolution, temporal attention after each spatial attention) initialized as identities [3]. **AnimateDiff** (2023) froze the base Stable Diffusion and trained only a plug-in motion module (temporal transformer layers, output projections zero-initialized) on WebVid-10M; the module is then inserted into personalized fine-tunes of the same base, and MotionLoRA adapts it to specific motions [4]. **Stable Video Diffusion** (2023) made the three-stage recipe explicit (image pre-training, curated video pre-training, high-quality fine-tuning) [5].

::: qa
**When you add temporal layers to a pretrained text-to-image model, why initialize them to the identity or to zero?**

At initialization the inflated network then computes exactly the image model's output on each frame, so training starts from a strong per-frame prior rather than a corrupted one. Make-A-Video initializes temporal layers as identities; AnimateDiff zero-initializes the motion module's output projection so its residual branch contributes nothing at first. Gradients grow the temporal pathway gradually, which is more stable and preserves image quality. Freezing the spatial weights as well (AnimateDiff) keeps compatibility with community fine-tunes and LoRAs, at the cost of limiting how far appearance can adapt to video statistics.
:::

### DiT era: spacetime patches and full attention

OpenAI's Sora report (2024) describes, at a high level, the recipe most open models now follow: a video compression network maps video to a spatiotemporal latent, the latent is cut into **spacetime patches** used as tokens, and a diffusion transformer (DiT [7]) is trained jointly on images and videos of variable duration, resolution and aspect ratio, at native size rather than square crops [12]. The report says it applied DALL-E 3 style re-captioning to videos and used GPT to expand short prompts. It shows samples with 3D consistency and object permanence (which it says hold often, not always) and lists failures such as glass shattering. It states that model and implementation details are not included.

**Full 3D vs factorized attention.** With $T$ latent frames of $S$ tokens, full attention costs $\propto(TS)^2$; alternating spatial attention (within a frame) and temporal attention (across frames at one location) costs $\propto TS^2+ST^2=TS(S+T)$, about 21x cheaper for Wan's $T=21$, $S=3600$. The price: temporal attention links only tokens at the same position, but a moving object changes position, so its information must hop through intermediate tokens. CogVideoX switched to 3D full attention for this reason [6], HunyuanVideo reports full attention beating divided attention [8], and Latte compares factorized DiT variants [20].

| Model | Size | Autoencoder | Text pathway | Notes |
|---|---|---|---|---|
| CogVideoX [6] | 2B / 5B | causal $4\times8\times8$, $C=16$ | T5 tokens concatenated with video tokens; expert AdaLN | 3D full attention, 3D RoPE, v-prediction, zero terminal SNR |
| HunyuanVideo [8] | 13B | causal $4\times8\times8$, $C=16$ | decoder-only MLLM plus a global CLIP text feature; dual-stream then single-stream blocks | flow matching |
| Wan 2.1 [9] | 1.3B / 14B | causal $4\times8\times8$, $C=16$ | umT5 via cross-attention | flow matching; time MLP shared across blocks |
| LTX-Video [10] | 1.9B | $8\times32\times32$, $C=128$ | cross-attention | decoder does the last denoising step |
| Movie Gen [11] | 30B | TAE $8\times8\times8$, $C=16$ | UL2, ByT5, long-prompt MetaCLIP via cross-attention | LLaMa3-style blocks; 7B upsampler to 1080p |

According to the Wan 2.2 GitHub README (2025; no paper), its A14B models split denoising between a high-noise expert (early steps, layout) and a low-noise expert (later steps, detail), switching at a step set by an SNR threshold: about 27B parameters in total, about 14B active per step [9]. Google's Veo 3 report gives only an outline: latent diffusion run jointly over audio latents and spatiotemporal video latents, a transformer denoiser, and Gemini-generated captions at several levels of detail [13].

::: {.qa asked="Ant Group video algorithm engineer, Nowcoder 2024"}
**How are diffusion transformers applied to video generation, for example in Sora?**

A video autoencoder compresses the clip in space and time; the latent is split into spacetime patches (for example $1\times2\times2$ latent voxels), each becoming a token with a 3D position encoding (3D RoPE in CogVideoX and HunyuanVideo, factorized learned embeddings in Movie Gen). A transformer with AdaLN timestep modulation denoises all tokens, and text enters by cross-attention or joint attention. Because tokens form a plain sequence, images and videos of different sizes and lengths train together, which is what Sora calls training at native size. Relative to a U-Net, the transformer scales more predictably and reuses LLM tooling, but the sequence is very long, so attention is the bottleneck.
:::

::: qa
**Full 3D attention or factorized space-time attention? Give the costs and the argument each way.**

Full attention costs $O((TS)^2)$ per layer and lets any token see any other, so moving objects are tracked directly; CogVideoX argues this matters for large motion and found 2D+1D attention less stable, and HunyuanVideo reports full attention outperforming divided attention. Factorized attention costs $O(TS(S+T))$, about a factor $T$ cheaper when $S\gg T$, and reuses image-model spatial weights. Its weakness is that temporal attention connects only co-located tokens, so motion must propagate through several layers. Large current models mostly pay for full attention and recover efficiency with sparse or local patterns that match what full attention actually attends to.
:::

::: {.qa asked="Taobao/Tmall AIGC image algorithm interview, Nowcoder 2025"}
**What is your technical assessment of Sora?**

The stated recipe is simple: compress video into a latent, cut it into spacetime patches, and train a diffusion transformer at scale on native-size data with detailed synthetic captions. The useful lessons are that variable-size training avoids framing errors caused by cropping, that re-captioning plus prompt expansion improves prompt following, and that sample quality improved with training compute; the report attributes 3D consistency and object permanence to scale rather than built-in inductive biases. The admitted limits are physical interactions such as breaking glass, and later benchmarks such as Physics-IQ found physical accuracy largely unrelated to visual realism. The report gives no architecture or data details, so claims about its internals are speculation.
:::

## Conditioning

**Text-to-video** follows the image recipe ([text-to-image](text-to-image-multimodal.html)) with two twists: training captions are long (Movie Gen's average about 100 words), and because users write short prompts, an LLM rewrites prompts at inference to match.

**Image-to-video** mechanisms:

- *Mask plus channel concatenation.* Wan concatenates the image latent (zeros for other frames) and a binary mask to the noisy latent and injects CLIP image features by cross-attention; moving the mask gives first-last-frame interpolation and continuation from the same model [9].
- *Noise-augmented concatenation.* SVD concatenates a slightly noised, encoded conditioning frame (copied along time) and replaces the text embedding with the CLIP image embedding [5].
- *Per-token timesteps.* LTX-Video assigns conditioning tokens a small diffusion time and the rest $t=1$ [10].

SVD shifts training noise higher for I2V because the image is a strong signal, increases guidance linearly across frames, and conditions on frame rate and a motion score so motion is controllable [5].

**Camera and motion control.** CameraCtrl encodes each pixel's camera ray as a Plücker embedding $(o\times d,\,d)\in\mathbb{R}^6$ (camera center $o$, ray direction $d$) and trains a plug-in encoder beside a frozen video model [24]. Movie Gen instead labels 16 camera-motion classes with a classifier and prefixes them to captions [11]. Trajectory conditioning (for example MotionCtrl [40]) handles object motion.

::: qa
**How would you fine-tune a text-to-video model into a text-and-image-to-video model?**

Concatenate the encoded image (frame 0, zeros elsewhere) and a mask channel to the noisy latent, widening the input projection with zero-initialized weights so the model starts as the original T2V model; optionally add image-encoder features through cross-attention for identity. Drop text and image independently during training so guidance can be applied to each, and randomize which frames are given (first, last, both). Filter clips whose first frame differs strongly from the rest, as Wan did with SigLIP similarity. Expect to shift training noise higher, because the image makes low-noise steps too easy.
:::

::: qa
**Your image-to-video model returns nearly static videos. What are the likely causes and fixes?**

The image explains most of every frame, so copying it is the cheapest way to lower the loss, and low-motion training clips reinforce that. On the data side, filter static clips by optical flow and balance motion levels. On the model side, condition on a motion score or frame rate (as SVD does) and raise it at inference, shift training noise higher, and noise-augment the conditioning image so it cannot be copied exactly. At sampling, separate text and image guidance scales, or guidance that increases across frames, trade first-frame fidelity for motion.
:::

## Training recipe

### Data curation and captioning

Published pipelines look alike [5, 6, 8, 11]: split videos into single-shot clips (SVD runs cut detection at three frame rates to catch fades), drop clips with little or jittery motion (optical flow, motion vectors), drop heavy overlaid text (OCR), score aesthetic and technical quality, deduplicate by embedding similarity, and rebalance concepts (Movie Gen samples clips from concept clusters according to the inverse square root of cluster size). In SVD's ablation, a model trained on a curated 2.3M-clip subset was preferred to one trained on the full 9.8M uncurated set [5]. Captions come from video VLMs fine-tuned for dense description (CogVLM2-Caption, LLaMa3-Video; Gemini for Veo).

::: {.qa asked="SenseTime R&D algorithm intern, Nowcoder 2026"}
**How would you build an automated filtering pipeline for video training data, and how do you check that it is not throwing away good clips?**

Order filters from cheap to expensive: metadata (resolution, duration, aspect ratio), shot detection, optical-flow motion scores, OCR, quality and aesthetic predictors, embedding deduplication, then VLM judgements and captioning on the survivors. Store scores, not pass/fail flags, so thresholds can change without reprocessing. To estimate false rejections, have humans label a uniform sample of each filter's rejected pool; the share of good clips there is the miss rate, and a sample of the accepted pool gives precision. The decisive test is downstream: train small models on data filtered at different thresholds under equal compute, since a filter can look clean yet remove the rare or motion-rich clips that matter most.
:::

### Curricula

Large reports train in stages of rising cost: text-to-image pre-training at 256px, joint image-video training at low resolution and short duration, then higher resolution and longer clips, then fine-tuning on a small hand-picked set. Wan went from 256px images to 192px 5 s clips at 16 fps, then 480px, then 720px [9]; HunyuanVideo from $256\times256\times65$ to $720\times1280\times129$ [8]; Movie Gen from 256px images to 256px and then 768px image-video training [11]. Cheap low-resolution stages teach layout and motion; expensive stages mainly add detail. Batches are bucketed by duration and aspect ratio (or packed, as in CogVideoX) to avoid padding.

### Objective and noise schedule

Most recent models use flow matching on the linear path. In the rectified-flow convention $x_t=(1-t)x_0+t\epsilon$ (noise at $t=1$) the network regresses $v=\epsilon-x_0$ with $t$ drawn from a logit-normal; Wan, HunyuanVideo and Movie Gen write the same path with time reversed [8, 9, 11] (derivation in [flow matching](flow-matching.html)). The path reaches pure noise exactly, giving zero terminal SNR, which Movie Gen stresses for video; CogVideoX instead gets zero terminal SNR with v-prediction and a rescaled schedule [6, 11, 26].

A given noise level destroys less information at higher resolution or frame count, because noise is independent per pixel while signal is correlated. SD3's argument [27]: for a constant image seen through $n$ pixels, averaging leaves uncertainty $\sigma(t,n)\propto\frac{t}{(1-t)\sqrt n}$. Setting $\sigma(t_m,m)=\sigma(t_n,n)$ gives

$$
\frac{t_m}{1-t_m}=\alpha\,\frac{t_n}{1-t_n},\qquad t_m=\frac{\alpha\,t_n}{1+(\alpha-1)\,t_n},\qquad \alpha=\sqrt{m/n},
$$

a log-SNR shift of $-2\log\alpha$. HunyuanVideo applies this form at inference with $s=7$ for 50 steps and $s=17$ below 20 steps [8]. Extending the argument to video by counting frames in $n$ is a heuristic, not something SD3 derives: frames are strongly but not perfectly correlated, so it suggests larger shifts for video, and in practice the shift is tuned empirically.

::: qa
**Why do video models shift the timestep schedule toward high noise, and how do you derive the shift?**

Averaging $n$ correlated pixels keeps the signal and cuts independent noise by $\sqrt n$, so at fixed $t$ a larger or longer input is effectively less noisy. Without enough training and sampling effort at high noise, the model does not learn global layout and motion well, and samples look locally plausible but globally incoherent. Equating the uncertainty $t/((1-t)\sqrt n)$ across sizes gives $t_m=\alpha t_n/(1+(\alpha-1)t_n)$ with $\alpha=\sqrt{m/n}$. Because real images are not constant, $\alpha$ is tuned in practice, and fewer sampling steps call for larger shifts.
:::

## Cascades and refiners

A cascade generates a low-resolution (sometimes low-frame-rate) video, then upsamples with conditional diffusion models. The base model spends its token budget on layout and motion; super-resolution needs only local context and can run on tiles; each stage trains independently. Movie Gen's 7B spatial upsampler takes 768px to 1080p: the input is bilinearly upsampled, encoded with a frame-wise VAE, concatenated channel-wise to the noisy latent, and trained on second-order degradations mixed with real TAE artifacts so train and test inputs match [11]; Imagen Video's noise conditioning augmentation serves the same purpose [2]. See [restoration and super-resolution](restoration-super-resolution.html).

Per-frame super-resolution flickers because each frame hallucinates different detail. Make-A-Video ran its first SR stage across space and time and gave its final per-frame stage the same initial noise for every frame [3]. Lumiere argued against temporal cascades: sparse keyframes alias fast motion, TSR models with short windows cannot resolve that ambiguity consistently, and TSR trained on real video meets generated frames at test time; it generates all 80 frames at once and keeps only spatial SR [25]. Recent latent DiTs mostly generate at the target frame rate; Movie Gen notes its $8\times$ temporal compression lets it skip frame interpolation [11].

::: qa
**Why do cascades help video generation, and what goes wrong with them?**

The base model handles semantics and motion at a size where full attention is affordable, and refiners handle texture, which is local; each stage gets its own resolution and data mix. Failure modes: a train-test gap between stages (SR trained on degraded real video but applied to generated video, reduced by noise conditioning augmentation and realistic degradations), flicker when SR runs per frame, and temporal aliasing when keyframes are too sparse to determine fast motion. Base-model errors also persist, because refiners are trained to stay faithful to their input.
:::

## Long video and autoregressive diffusion

Bidirectional models generate a fixed-length clip. Longer video needs extension (reconstruction guidance, mask-based continuation) or a causal model that generates chunk by chunk with a KV cache, which also enables streaming and interaction.

**Exposure bias.** Teacher forcing trains $p_\theta(x_i\mid x^{\text{data}}_{<i})$, but sampling uses $p_\theta(x_i\mid\hat x_{<i})$ with generated context. Small errors push the context off the training distribution, the next prediction gets worse, and errors compound into drift: oversaturation, blur, identity loss.

**Diffusion Forcing** (Chen et al., 2024) trains a causal denoiser with an independent noise level per frame [15]. In causal-transformer form,

$$
\mathcal{L}=\mathbb{E}\sum_{i=1}^{N}\big\lVert \epsilon_i-\epsilon_\theta\big(x^{k_1}_1,\dots,x^{k_i}_i;\,k_{1:i}\big)\big\rVert^2,\qquad k_i\sim\mathcal{U}\{0,\dots,K\}\ \text{independently}.
$$

Teacher forcing ($k_{<i}=0$) and full-sequence diffusion (all $k_i$ equal) are special cases. At sampling, a schedule matrix sets each frame's noise level per step, for example keeping future frames noisier. Feeding context at a small noise level $0<k\ll K$ makes the model tolerant of imperfect context and stabilizes rollouts past the training horizon; GameNGen noised context frames during training for the same reason [28].

**CausVid** (Yin et al., 2024) distills a 50-step bidirectional teacher into a 4-step student with block-causal attention (bidirectional within a chunk, causal across chunks) [16]. The student is initialized on teacher ODE pairs, then trained with distribution matching distillation (DMD), whose gradient is the difference between data and generator scores evaluated on noised student samples. With KV caching it reports 1.3 s initial latency and about 9.4 FPS on one GPU.

**Self Forcing** (Huang et al., 2025) trains on the model's own rollouts: each chunk is generated conditioned on earlier generated chunks through the KV cache, and a video-level distribution-matching loss (DMD, SiD or GAN) is applied to the result [17]. For affordability it backpropagates only through the final denoising step of each chunk, chooses that step at random, and stops gradients into the cache. For long videos it uses a rolling KV cache; since a causal VAE's first latent is statistically special, training blocks attention to the first chunk when denoising the last one. Built on Wan2.1-1.3B, it reports 17 FPS with sub-second latency on one H100. FramePack names the core tension, *forgetting* (too little history) versus *drifting* (history that propagates errors), and compresses less important frames more to fit long histories [29].

::: qa
**What is exposure bias in autoregressive video diffusion, and what are three ways to reduce it?**

Training conditions on real past frames and inference on generated ones, so the model meets unseen context and errors compound. (1) Robustness: noise context frames during training and condition on a small noise level at inference (Diffusion Forcing, GameNGen). (2) Train on the inference distribution: roll out the model in training and apply a sequence-level loss (Self Forcing), as GANs do implicitly. (3) Limit propagation: shorter or compressed contexts, a clean anchor frame, or sampling orders that fix endpoints first (FramePack). Each trades some memory or training cost for stability.
:::

::: qa
**How does Diffusion Forcing relate to teacher forcing and to full-sequence diffusion?**

They differ only in the joint distribution of per-frame noise levels. Teacher forcing uses a clean past and a noisy current frame; full-sequence diffusion shares one level across all frames with a non-causal network. Diffusion Forcing samples levels independently with a causal network, so it learns all mixtures, including both special cases. The sampling schedule becomes a free choice: one frame at a time, a pyramid of rising noise into the future, or guidance with lookahead. Slightly noised context also stabilizes long rollouts.
:::

::: qa
**CausVid and Self Forcing both distill a causal few-step generator with DMD. What is the key difference?**

CausVid builds the student's inputs by noising chunks of real videos, so during training the student never conditions on its own outputs, and DMD matches a distribution the model does not produce at inference. Self Forcing generates the whole video autoregressively during training, each chunk conditioned on the model's earlier chunks, and applies the distribution-matching loss to that rollout. This removes the train-test mismatch at the cost of sequential training compute, controlled with few-step models and gradient truncation.
:::

## Efficiency

The token arithmetic sets the priorities: shrink $L$, avoid dense $L^2$ attention, and cut network evaluations.

- **Sparse and local attention.** Pretrained video DiTs attend mostly within local 3D windows. Sliding Tile Attention computes tile-aligned windowed attention; on HunyuanVideo it cut end-to-end time from 945 s to 685 s without training and to 268 s with fine-tuning [21]. Sparse VideoGen profiles each head online as spatial (within-frame) or temporal (same location across frames) and applies the matching sparse mask, reporting up to 2.33x on HunyuanVideo without training [22].
- **Step distillation.** Progressive distillation took Imagen Video to 8 steps per model [2]; DMD gives 4-step causal generators [16, 17]; adversarial post-training against real data (APT) generates 2 s, 1280x720, 24 fps clips in one step in real time [23]. Guidance distillation removes the extra CFG pass; HunyuanVideo reports about 1.9x [8].
- **Feature caching.** Features change slowly between steps. Pyramid Attention Broadcast reuses attention outputs across steps, longest for the attention types that vary least [30]; TeaCache reuses cached outputs when the change in timestep-modulated inputs is small, reporting up to 4.41x on Open-Sora-Plan with a 0.07% VBench drop [31]; Wan caches attention outputs and recomputes the unconditional branch only every few steps, about 1.62x on its 14B model [9].
- **Fewer tokens.** High-compression autoencoders and cascades reduce $L$ before any of the above.

::: qa
**You need a 5x speed-up on a 13B video DiT without retraining from scratch. What do you try, in what order?**

Profile first: by the rough $L/(6d)$ estimate, attention is over 80% of FLOPs at $10^5$ tokens and $d\approx3000$. Then the cheap options: guidance distillation or fewer unconditional passes (up to half the work), step caching (TeaCache reports up to 4.41x), and a profiled sparse or tiled attention mask (Sparse VideoGen reports up to 2.33x). Gains combine only partially because they remove overlapping computation. If that is not enough, fine-tune with local attention, then distill to a few steps with DMD or adversarial post-training, the largest win but the riskiest for motion and diversity. Check every change on a fixed prompt set with human A/B tests and motion-sensitive metrics.
:::

::: qa
**Why does feature caching work for diffusion transformers, and when does it fail?**

Adjacent steps see nearly the same input, so many features, especially attention outputs in middle steps, barely change and can be reused. PAB found the step-to-step attention difference U-shaped over sampling, so reuse is safest mid-trajectory; TeaCache recomputes only when an accumulated input-change estimate passes a threshold. Caching fails when each step changes the output a lot (few-step or distilled samplers), in early steps that set layout and motion, and on fast motion or fine detail, where stale features show up as blur or ghosting.
:::

## Evaluation

**FVD** [39] fits Gaussians to I3D features (an action classifier trained on Kinetics) of real and generated clips:

$$
\mathrm{FVD}=\lVert\mu_r-\mu_g\rVert_2^2+\operatorname{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big).
$$

Ge et al. showed that FVD barely changes under severe temporal corruption and that picking motionless samples from a large pool can lower it substantially, because supervised I3D features are biased toward per-frame content; self-supervised VideoMAE-v2 features reduce the bias [32]. FVD also inherits FID's issues: Gaussian assumption, sample-size bias, and sensitivity to clip length, resizing and sampling protocol (VDM notes that sampling test clips with or without replacement changes reported numbers [1]). See [evaluation metrics](evaluation-metrics.html).

**VBench** splits quality into 16 dimensions (subject and background consistency, flicker, motion smoothness, dynamic degree, aesthetics, object class, spatial relations and more), each with an automatic evaluator validated against human preference [33]. Its authors note that nearly static videos score well on consistency, so read consistency together with dynamic degree. **Human evaluation** is the reference: Movie Gen used pairwise A/B tests on a released 1003-prompt benchmark along text alignment, visual quality, and realness and aesthetics, and explained why it avoided automated metrics [11]. Physics-IQ and VideoPhy probe physical plausibility [34, 35].

::: qa
**What are the main problems with FVD, and what would you report instead or in addition?**

Its features come from a supervised Kinetics classifier that weighs appearance over motion, so sharp frames with wrong or missing motion can score well, temporal corruption barely moves it, and selecting static samples lowers it. It assumes Gaussian features, is biased at small sample sizes, needs a matched reference set, and depends on clip length, frame rate, resizing and sampling details, so cross-paper numbers are rarely comparable. I would add FVD with self-supervised features, VBench dimensions read in pairs (consistency with dynamic degree), prompt-alignment scores, and blinded pairwise human preference with confidence intervals on a fixed prompt set.
:::

## World models and failure modes

A world model predicts future observations given actions; a video generator becomes one when conditioned on actions and run causally in real time.

- **Genie** (Bruce et al., 2024): an 11B model trained on about 30,000 hours of 2D platformer gameplay without action labels [18]. A latent action model learns a discrete action (8 codes) between consecutive frames; a spatiotemporal tokenizer and a MaskGIT dynamics model generate frames given a chosen action. It ran at about 1 FPS with 16 frames of memory.
- **Genie 2** (Dec 2024) is described as an autoregressive latent diffusion model with a causally masked transformer, consistent for up to a minute [36]. **Genie 3** (Aug 2025) runs at 720p, 24 fps in real time, consistent for a few minutes [37].
- **GameNGen** simulated DOOM at 20 FPS on one TPU with a modified Stable Diffusion conditioned on past frames and actions [28]. Veo 3 solves some perception and reasoning tasks zero-shot by prompting, such as edge detection and maze solving [38].

**Common failure modes**: wrong physics (collisions, fluids, breakage), broken object permanence (objects vanish or change after occlusion), morphing (merging limbs, identity drift), wrong counts and extra limbs, near-static "living photo" output, camera motion substituting for object motion, and drift in long rollouts. Physics-IQ found physical understanding "severely limited, and unrelated to visual realism" [34].

::: qa
**Do video diffusion models understand physics?**

Partially and inconsistently. They capture frequent, visible regularities (rigid motion, gravity on common objects, plausible lighting), and large models show some 3D consistency and object persistence. But the objective rewards matching pixel statistics, not conservation laws, so rare or causally subtle events (breakage, fluids, multi-object contact) often fail, and Physics-IQ found physical accuracy unrelated to realism. Evidence of understanding should come from interventions, such as action-conditioned or counterfactual prediction, not from how real samples look.
:::

::: qa
**How does Genie learn controllable actions from videos without action labels?**

A latent action model acts as an information bottleneck: an encoder sees frames up to $x_{t+1}$ and outputs a code $a_t$ from an 8-entry VQ codebook, and a decoder must predict $x_{t+1}$ from $x_{1:t}$ and $a_t$ alone. Since the decoder already has the past, the code must carry the most predictive change between frames, which in platformers aligns with movement. At inference the encoder is dropped, a user picks codes, and the dynamics model predicts the next frame's tokens. The tiny codebook keeps actions playable and consistent across environments.
:::

## Rapid-fire review

::: qa
**Why do many video models generate 49, 81, 121 or 129 frames?**

Causal VAEs map $1+sk$ frames to $1+k$ latents. With $s=4$: 49, 81, 129 frames give 13, 21, 33 latents; with $s=8$, 121 gives 16.
:::

::: qa
**What is the compression ratio of a $4\times8\times8$ VAE with 16 channels?**

$4\cdot8\cdot8\cdot3/16=48$ pixel values per latent value, 1:48; a $1\times2\times2$ patch then makes one token per $4\times16\times16$ pixel block.
:::

::: qa
**Why did Imagen Video use v-prediction?**

It kept progressive distillation numerically stable and avoided the color shifts seen with $\epsilon$-prediction at high resolution.
:::

::: qa
**What is a motion module?**

Temporal attention layers inserted into a frozen text-to-image U-Net and trained on video (AnimateDiff), so one module animates many fine-tunes of the same base model.
:::

::: qa
**What is 3D RoPE?**

Rotary embeddings applied separately to time, height and width on disjoint channel groups and concatenated, giving relative positions on all three axes for variable sizes.
:::

::: qa
**What does noise conditioning augmentation do in a cascade?**

It noises the low-resolution input during SR training and tells the model the level, so the SR model tolerates artifacts of generated inputs.
:::

::: qa
**Why condition on frame rate or a motion score?**

Training data mixes frame rates and motion levels; explicit inputs remove that ambiguity and give a control knob at inference (SVD, Make-A-Video).
:::

::: qa
**What does a KV cache buy in causal video generation?**

Keys and values of past chunks are computed once and reused, so each new chunk attends to the cache instead of recomputing the sequence; a rolling cache bounds memory.
:::

::: qa
**What is a Plücker embedding for camera control?**

The 6D vector $(o\times d,d)$ of the camera ray through each pixel; it encodes pose and intrinsics densely in pixel space, easier for a network than raw matrices.
:::

::: qa
**Why resample concept clusters by inverse square root of cluster size?**

Movie Gen says only that clips are sampled according to the inverse square root of cluster size. Read as a per-clip weight $\propto n_c^{-1/2}$, each cluster's total mass becomes $\propto\sqrt{n_c}$: big clusters lose dominance without tiny, noisy clusters taking over, a middle ground between natural frequency ($\propto n_c$) and uniform over clusters.
:::

::: qa
**What is the Wan 2.2 mixture of experts?**

Per its GitHub README, two experts of about 14B each (about 27B total) specialized by noise level, one for early high-noise steps (overall layout) and one for later low-noise steps (detail), switched at a step set by an SNR threshold; per-step compute stays roughly that of one 14B model.
:::

::: qa
**Why do teacher-forced autoregressive video models drift?**

They never train on their own outputs, so early errors create out-of-distribution context and errors compound (exposure bias).
:::

## References

1. Ho et al., "Video Diffusion Models," 2022. [arXiv:2204.03458](https://arxiv.org/abs/2204.03458)
2. Ho et al., "Imagen Video: High Definition Video Generation with Diffusion Models," 2022. [arXiv:2210.02303](https://arxiv.org/abs/2210.02303)
3. Singer et al., "Make-A-Video: Text-to-Video Generation without Text-Video Data," 2022. [arXiv:2209.14792](https://arxiv.org/abs/2209.14792)
4. Guo et al., "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning," 2023. [arXiv:2307.04725](https://arxiv.org/abs/2307.04725)
5. Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets," 2023. [arXiv:2311.15127](https://arxiv.org/abs/2311.15127)
6. Yang et al., "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer," 2024. [arXiv:2408.06072](https://arxiv.org/abs/2408.06072)
7. Peebles and Xie, "Scalable Diffusion Models with Transformers," 2022. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
8. Kong et al., "HunyuanVideo: A Systematic Framework for Large Video Generative Models," 2024. [arXiv:2412.03603](https://arxiv.org/abs/2412.03603)
9. Wan Team, "Wan: Open and Advanced Large-Scale Video Generative Models," 2025. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314); Wan 2.2 release notes: [github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
10. HaCohen et al., "LTX-Video: Realtime Video Latent Diffusion," 2024. [arXiv:2501.00103](https://arxiv.org/abs/2501.00103)
11. Polyak et al., "Movie Gen: A Cast of Media Foundation Models," 2024. [arXiv:2410.13720](https://arxiv.org/abs/2410.13720)
12. Brooks et al. (OpenAI), "Video generation models as world simulators," technical report, 2024. [openai.com](https://openai.com/research/video-generation-models-as-world-simulators)
13. Google DeepMind, "Veo 3 Tech Report," 2025. [PDF](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf)
14. Yu et al., "Language Model Beats Diffusion: Tokenizer is Key to Visual Generation" (MAGVIT-v2), 2023. [arXiv:2310.05737](https://arxiv.org/abs/2310.05737)
15. Chen et al., "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion," 2024. [arXiv:2407.01392](https://arxiv.org/abs/2407.01392)
16. Yin et al., "From Slow Bidirectional to Fast Autoregressive Video Diffusion Models" (CausVid), 2024. [arXiv:2412.07772](https://arxiv.org/abs/2412.07772)
17. Huang et al., "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion," 2025. [arXiv:2506.08009](https://arxiv.org/abs/2506.08009)
18. Bruce et al., "Genie: Generative Interactive Environments," 2024. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
19. Blattmann et al., "Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models," 2023. [arXiv:2304.08818](https://arxiv.org/abs/2304.08818)
20. Ma et al., "Latte: Latent Diffusion Transformer for Video Generation," 2024. [arXiv:2401.03048](https://arxiv.org/abs/2401.03048)
21. Zhang et al., "Fast Video Generation with Sliding Tile Attention," 2025. [arXiv:2502.04507](https://arxiv.org/abs/2502.04507)
22. Xi et al., "Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity," 2025. [arXiv:2502.01776](https://arxiv.org/abs/2502.01776)
23. Lin et al., "Diffusion Adversarial Post-Training for One-Step Video Generation," 2025. [arXiv:2501.08316](https://arxiv.org/abs/2501.08316)
24. He et al., "CameraCtrl: Enabling Camera Control for Video Diffusion Models," 2024. [arXiv:2404.02101](https://arxiv.org/abs/2404.02101)
25. Bar-Tal et al., "Lumiere: A Space-Time Diffusion Model for Video Generation," 2024. [arXiv:2401.12945](https://arxiv.org/abs/2401.12945)
26. Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed," 2023. [arXiv:2305.08891](https://arxiv.org/abs/2305.08891)
27. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
28. Valevski et al., "Diffusion Models Are Real-Time Game Engines" (GameNGen), 2024. [arXiv:2408.14837](https://arxiv.org/abs/2408.14837)
29. Zhang et al., "Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models" (FramePack), 2025. [arXiv:2504.12626](https://arxiv.org/abs/2504.12626)
30. Zhao et al., "Real-Time Video Generation with Pyramid Attention Broadcast," 2024. [arXiv:2408.12588](https://arxiv.org/abs/2408.12588)
31. Liu et al., "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model" (TeaCache), 2024. [arXiv:2411.19108](https://arxiv.org/abs/2411.19108)
32. Ge et al., "On the Content Bias in Fréchet Video Distance," 2024. [arXiv:2404.12391](https://arxiv.org/abs/2404.12391)
33. Huang et al., "VBench: Comprehensive Benchmark Suite for Video Generative Models," 2023. [arXiv:2311.17982](https://arxiv.org/abs/2311.17982)
34. Motamed et al., "Do generative video models understand physical principles?" (Physics-IQ), 2025. [arXiv:2501.09038](https://arxiv.org/abs/2501.09038)
35. Bansal et al., "VideoPhy: Evaluating Physical Commonsense for Video Generation," 2024. [arXiv:2406.03520](https://arxiv.org/abs/2406.03520)
36. Google DeepMind, "Genie 2: A large-scale foundation world model," blog, Dec 2024. [deepmind.google](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
37. Google DeepMind, "Genie 3: A new frontier for world models," blog, Aug 2025. [deepmind.google](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)
38. Wiedemer et al., "Video models are zero-shot learners and reasoners," 2025. [arXiv:2509.20328](https://arxiv.org/abs/2509.20328)
39. Unterthiner et al., "Towards Accurate Generative Models of Video: A New Metric & Challenges" (FVD), 2018. [arXiv:1812.01717](https://arxiv.org/abs/1812.01717)
40. Wang et al., "MotionCtrl: A Unified and Flexible Motion Controller for Video Generation," 2023. [arXiv:2312.03641](https://arxiv.org/abs/2312.03641)
