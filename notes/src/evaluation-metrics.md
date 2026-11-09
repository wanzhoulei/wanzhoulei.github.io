---
title: "Evaluating Generative and Restoration Models"
subtitle: "What FID and its alternatives, precision/recall, alignment and preference scores, PSNR/SSIM/LPIPS, no-reference IQA and human studies actually measure, and the perception–distortion trade-off that makes them disagree."
order: 4
updated: 2026-11-08
---

A learned evaluation metric has three parts: a feature extractor, a statistic computed in that feature space, and a protocol (sample count, reference set, resizing, seeds). Changing any one of them can reorder models. Interviewers ask about metrics because your answers show whether you know what your model is being rewarded for, and whether you can tell a real gain from a measurement artifact. This page covers distribution metrics, text–image and video metrics, full-reference and no-reference image quality, the perception–distortion trade-off, and human studies. Model details are on the other pages ([diffusion models](diffusion-models.html), [restoration and super-resolution](restoration-super-resolution.html), [text-to-image](text-to-image-multimodal.html), [video generation](video-generation.html)).

## A map of what metrics measure

Every metric answers some mix of three questions. **Fidelity:** does each sample look real? **Coverage:** does the set of samples span the variety of the data? **Consistency with the input:** does the output match the prompt (alignment) or the measured signal (distortion)?

| Family | Compares | Mainly answers |
|---|---|---|
| IS | samples vs. an ImageNet classifier | fidelity plus class diversity |
| FID, KID, FD$_\text{DINOv2}$, CMMD | sample set vs. reference set | fidelity and coverage, mixed |
| Precision/recall, density/coverage | sample set vs. reference set | fidelity and coverage, separately |
| CLIPScore, VQAScore, GenEval, DPG-Bench | image vs. prompt | alignment |
| ImageReward, PickScore, HPSv2, arenas | images and prompt, learned from votes | overall preference |
| FVD, VBench | videos vs. references or prompts | video quality and consistency |
| PSNR, SSIM, LPIPS, DISTS | output vs. ground truth | distortion |
| NIQE, MUSIQ, MANIQA, CLIP-IQA, Q-Align | single image | perceived quality |

Theis et al. (2016) showed that log-likelihood, sample quality and Parzen estimates can be largely independent in high dimensions, so a model should be evaluated for the use it was built for.

::: key
Most surprises come from the feature extractor (what it can and cannot see) and the protocol (sample size, resizing, reference statistics), not from the distance formula. When two metrics disagree, first ask what each one's features are sensitive to.
:::

## Inception Score

With $p(y\mid x)$ the Inception-v3 class posterior and $p(y)=\mathbb{E}_{x\sim p_g}[p(y\mid x)]$, Salimans et al. (2016) define $\mathrm{IS}=\exp\big(\mathbb{E}_{x}\,\mathrm{KL}(p(y\mid x)\,\|\,p(y))\big)$. Expanding the KL:

$$\mathbb{E}_x\,\mathrm{KL}\big(p(y\mid x)\,\|\,p(y)\big)=\mathbb{E}_x\sum_y p(y\mid x)\log p(y\mid x)-\sum_y p(y)\log p(y)=H(y)-H(y\mid x)=I(x;y).$$

IS is the exponentiated mutual information between an image and its predicted label. It is high when each image is classified confidently and all classes are used, and since $I\le H(y)\le\log 1000$, IS is at most 1000. It never looks at real data, it cannot see diversity within a class, it only makes sense for ImageNet-like content, and Barratt and Sharma (2018) showed it depends on implementation details and can be raised by adversarial examples when optimized directly.

::: qa
**Why can a mode-collapsed generator still get a high Inception Score?**

IS equals $\exp(H(y)-H(y\mid x))$, so it depends only on the distribution of predicted labels. A generator that outputs one sharp, confidently classified image per class makes $H(y\mid x)\approx 0$ and $H(y)=\log 1000$, which is the maximum score, even though it has collapsed to 1000 points. Diversity within a class, and any mismatch with real images, are invisible because IS never uses real data. Reference-based metrics (FID, recall) catch this.
:::

## Fréchet Inception Distance

### From Wasserstein-2 to the formula

Heusel et al. (2017) embed images with Inception-v3 (2048-d pool3), fit Gaussians to real and generated features, and compute their Fréchet distance, which for Gaussians equals the squared Wasserstein-2 distance:

$$\mathrm{FID}=\lVert\mu_r-\mu_g\rVert_2^2+\mathrm{Tr}\Big(\Sigma_r+\Sigma_g-2\big(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}\big)^{1/2}\Big).$$

**Derivation.** $W_2^2$ minimizes $\mathbb{E}\lVert X-Y\rVert^2$ over couplings, and for any coupling

$$\mathbb{E}\lVert X-Y\rVert^2=\lVert\mu_r-\mu_g\rVert^2+\mathrm{Tr}\,\Sigma_r+\mathrm{Tr}\,\Sigma_g-2\,\mathrm{Tr}\,\mathrm{Cov}(X,Y).$$

So the task is to maximize the trace of the cross-covariance. In 1-D, Cauchy–Schwarz gives $\mathrm{Cov}\le\sigma_r\sigma_g$, attained by the monotone coupling $Y=\mu_g+\frac{\sigma_g}{\sigma_r}(X-\mu_r)$, so $W_2^2=(\mu_r-\mu_g)^2+(\sigma_r-\sigma_g)^2$. If the covariances commute (eigenvalues $\lambda_i,\gamma_i$), each eigen-direction is a 1-D problem, and the sum $\sum_i(\sqrt{\lambda_i}-\sqrt{\gamma_i})^2$ equals the trace term above. In general the maximal trace is $\mathrm{Tr}\big((\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}\big)$. Code computes `sqrtm(Sigma_r @ Sigma_g)`, which has the same trace because the two matrices are similar, and discards small imaginary parts from numerical error.

Two consequences follow. FID sees only first and second moments: very different feature distributions with equal mean and covariance get FID 0, and Inception features fail normality tests (Jayasumana et al., 2024). The covariance term also penalizes both too little spread (mode dropping) and too much (junk samples), so fidelity and coverage are mixed into one number.

### Estimation: bias and sample size

The finite-sample estimate is biased, and the bias depends on the model. The mean term alone has

$$\mathbb{E}\,\lVert\hat\mu_r-\hat\mu_g\rVert^2=\lVert\mu_r-\mu_g\rVert^2+\frac{\mathrm{Tr}\,\Sigma_r}{N_r}+\frac{\mathrm{Tr}\,\Sigma_g}{N_g},$$

so a generator with more spread-out features pays more at the same $N$. The square-root term is a nonlinear function of noisy covariances and is biased too. The sample covariance is singular unless $N>2048$, and the TTUR reference code recommends at least 10k. Bińkowski et al. (2018) note that no unbiased FID estimator exists, and they construct cases where the bias reverses the order of two models even at 50k samples. Chong and Forsyth (2020) found $\mathrm{FID}_N$ close to linear in $1/N$ with a model-dependent slope, and extrapolate a line fitted over $N\in[5\text{k},50\text{k}]$ to $1/N\to 0$. In practice, use the same $N$ and reference statistics for all models and state them: 50k against ImageNet training statistics, or 30k COCO captions for zero-shot text-to-image.

### Preprocessing sensitivity

Inception sees 299×299 inputs, so finer detail is invisible, and the resize itself matters. Parmar et al. (clean-fid) showed that common resize routines (OpenCV, PyTorch and TensorFlow defaults) use a fixed-width filter whatever the downsampling factor, which aliases. Resizing the same FFHQ images from 1024 to 299 with the bilinear resizers of the PyTorch and TensorFlow FID code, instead of antialiased PIL bicubic, gives an FID of about 4.3 between two copies of one image set. When the reference and generated sets were resized differently, a StyleGAN2 FID rose from 3.0 to 7.5. Compression matters too: on LSUN Church, whose training images were stored as JPEG (quality 75), the best FID came from JPEG-compressing the samples at quality 87. FIDs from different pipelines are not comparable.

### What FID rewards and misses

- **ImageNet class information.** Kynkäänniemi et al. (2023) resampled generated sets to match the real set's top-N ImageNet class histogram and cut FID substantially without changing any image. An ImageNet-pretrained FastGAN matched StyleGAN2's FID but lost in human evaluation.
- **Diffusion models.** In a large human study, Stein et al. (2023) found people rated diffusion models most realistic while FID did not, and no tested metric correlated strongly with human realism. They recommend DINOv2 ViT-L/14 features.
- **Guidance.** In the Pick-a-Pic study, nine generators (three Stable Diffusion variants at guidance 3, 6 and 9) were compared on 100 MS-COCO captions, and FID-based win rates correlated at −0.90 with human ones. The authors link this to guidance: stronger guidance looks better but is less diverse. Report FID against CLIPScore over a guidance sweep.
- **Blind spots.** FID does not see local failures (hands, text), per-sample prompt adherence, or memorization: copying the training set scores very well.

::: pitfall
"Lower FID means better images" is not a safe reading. FID can improve by matching ImageNet class frequencies, compressing like the reference set, or memorizing, and it can get worse when guidance makes each image look better but the set less diverse.
:::

::: qa
**Derive the FID formula. Why does a matrix square root appear?**

FID is the squared $W_2$ distance between Gaussians fitted to features: the minimum of $\mathbb{E}\lVert X-Y\rVert^2$ over couplings. Expanding gives $\lVert\mu_r-\mu_g\rVert^2+\mathrm{Tr}\Sigma_r+\mathrm{Tr}\Sigma_g-2\,\mathrm{Tr}\,\mathrm{Cov}(X,Y)$, so we maximize the cross-covariance trace. In 1-D the maximum is $\sigma_r\sigma_g$, from perfect correlation, giving $(\sigma_r-\sigma_g)^2$. For commuting covariances each eigen-direction contributes $(\sqrt{\lambda_i}-\sqrt{\gamma_i})^2$. In general $\sigma_r\sigma_g$ becomes $\mathrm{Tr}\big((\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}\big)$, the cross term of the Bures–Wasserstein distance between covariances, which reduces to $\mathrm{Tr}(\Sigma_r^{1/2}\Sigma_g^{1/2})$ when they commute.
:::

::: qa
**Why is FID biased, and why does that matter for comparing models?**

Even the mean term has expectation $\lVert\mu_r-\mu_g\rVert^2+\mathrm{Tr}\Sigma_r/N_r+\mathrm{Tr}\Sigma_g/N_g$, and the square-root term is nonlinear in noisy covariance estimates. The bias depends on the generator's own covariance, so two models at the same $N$ carry different biases, and rankings can change with $N$. Fixes: compare at the same $N$ against the same statistics, use enough samples, extrapolate in $1/N$, or use an unbiased statistic such as KID.
:::

::: qa
**Two papers report FID 2.1 and 2.3 on ImageNet 256, or your change moves FID from 3.10 to 2.85. What do you check before believing the gap?**

First check that the protocol is identical: sample count, reference statistics, resizing and file format (clean vs. legacy), class sampling, Inception weights. Then check the sampler budget (NFE, guidance and whether it was tuned for FID) and run several sampling seeds. Near FID 2–3, seed variance and pipeline differences can be a meaningful share of a 0.2–0.3 gap. Next look at precision/recall and a second feature space (FD$_\text{DINOv2}$ or CMMD). If the ranking flips between feature spaces, the gain is fragile, and a 2AFC human study should decide.
:::

::: {.qa asked="Tencent Youtu CV intern, 2nd round (Nowcoder)"}
**Walk through the metrics you would use for a generation or editing model: CLIP-I, CLIP-T, LPIPS, FID. What does each measure?**

CLIP-T is the image–prompt cosine similarity in CLIP space, a coarse alignment score that is weak on counting and attribute binding. CLIP-I is the cosine similarity between CLIP image embeddings of the output and a reference, so it measures semantic identity preservation, not pixels. LPIPS is a per-pair perceptual distance, useful for checking that an edit leaves the rest of the image intact. FID compares feature distributions over thousands of images, so it measures set-level realism and diversity and says nothing about any single sample. Use them together, because each can be gamed alone: copying the input maximizes CLIP-I and minimizes LPIPS while ignoring the edit.
:::

## Alternatives: KID, FD$_\text{DINOv2}$, CMMD

For a kernel $k$ with feature map $\phi$, $\mathrm{MMD}^2(P,Q)=\lVert\mathbb{E}_P\phi-\mathbb{E}_Q\phi\rVert_{\mathcal{H}}^2=\mathbb{E}\,k(x,x')+\mathbb{E}\,k(y,y')-2\,\mathbb{E}\,k(x,y)$. The unbiased U-statistic drops the diagonal terms $k(x_i,x_i)$, which are the source of the plug-in estimator's upward bias:

$$\widehat{\mathrm{MMD}}^2_u=\frac{1}{m(m-1)}\sum_{i\ne j}k(x_i,x_j)+\frac{1}{n(n-1)}\sum_{i\ne j}k(y_i,y_j)-\frac{2}{mn}\sum_{i,j}k(x_i,y_j).$$

- **KID** (Bińkowski et al., 2018) applies this to Inception features with $k(x,y)=(x^\top y/d+1)^3$, which compares feature moments up to third order without bandwidth tuning. It has no Gaussian assumption and is comparable across sample sizes. Estimates can be slightly negative, and people report the mean ± std over subsets.
- **FD$_\text{DINOv2}$** keeps the Fréchet formula with DINOv2 ViT-L/14 features (Stein et al., 2023). EDM2 (Karras et al., 2024) reports it next to FID and ships separate configurations whose EMA length and guidance were tuned for each metric. The best settings differ, so "best model" depends on the metric.
- **CMMD** (Jayasumana et al., 2024) is the unbiased MMD over CLIP ViT-L/14@336px embeddings with a Gaussian kernel ($\sigma=10$), scaled by 1000. In their Stable Diffusion test, FID needed more than 20k images for a reliable estimate, while CMMD was stable on small sets. In a human study, raters preferred the full Muse sampler (24 base and 8 super-resolution refinement iterations) to one stopped early (20 and 3) in 92.5% vs. 6.9% of comparisons; COCO-30K FID favoured the early-stopped version, and CMMD agreed with the raters. CMMD inherits CLIP's biases, and CLIP features are likely less sensitive to low-level texture.

::: qa
**KID vs. FID: what does "unbiased" buy you, what does it not fix, and when do Inception- and DINOv2-based rankings disagree?**

KID's estimate has expectation equal to the population $\mathrm{MMD}^2$ at any $N$, so small-sample and cross-paper comparisons are meaningful, and there is no model-dependent bias term to reorder rankings. It does not fix the feature space: KID still uses Inception, so ImageNet-class leakage and blindness to fine detail remain. Inception- and DINOv2-based rankings tend to disagree for models with ImageNet-pretrained components, for diffusion vs. GAN comparisons, and at high guidance. Stein et al.'s human data favour DINOv2 features, and a disagreement is a signal to run a human study.
:::

## Precision, recall, density and coverage

FID cannot say whether a bad score comes from bad samples or from missing modes. Sajjadi et al. (2018) separated the two, and Kynkäänniemi et al. (2019) gave the standard k-NN version. Let $\{x_i\}_{i=1}^N$ be real features and $\{y_j\}_{j=1}^M$ generated features (VGG-16 fc2), and let $\mathrm{NND}_k(x_i)$ be the distance from $x_i$ to its $k$-th nearest neighbour in its own set:

$$\begin{aligned}
\mathrm{precision}&=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}\Big[y_j\in\textstyle\bigcup_i B\big(x_i,\mathrm{NND}_k(x_i)\big)\Big],\\
\mathrm{recall}&=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\Big[x_i\in\textstyle\bigcup_j B\big(y_j,\mathrm{NND}_k(y_j)\big)\Big],
\end{aligned}$$

with $k=3$ and 50k samples per set. Stronger truncation raises precision and lowers recall. Naeem et al. (2020) noted that a union of balls is inflated by outliers, and that neither metric equals 1 when the distributions are identical. They proposed

$$\begin{aligned}
\mathrm{density}&=\frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N}\mathbf{1}\big[y_j\in B(x_i,\mathrm{NND}_k(x_i))\big],\\
\mathrm{coverage}&=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\big[\exists\,j:\ y_j\in B(x_i,\mathrm{NND}_k(x_i))\big].
\end{aligned}$$

When the distributions are identical, $\mathbb{E}[\text{density}]=1$ and $\mathbb{E}[\text{coverage}]=1-\prod_{l=1}^{k}\frac{N-l}{M+N-l}$, which tends to $1-2^{-k}$ when $M=N\to\infty$. They recommend $k=5$ and $M=N=10$k, which gives about 0.969, and random embeddings for domains far from ImageNet.

| Symptom | Likely cause |
|---|---|
| High precision/density, low recall/coverage | mode dropping, high guidance or truncation |
| Low precision, high recall | artifacts spread over the data range, too little guidance |
| Both low | domain mismatch, preprocessing bug |
| FID improves while recall drops | diversity traded for fidelity |

::: {.qa asked="NIO algorithm intern (Nowcoder, 2024)"}
**Explain the difference between precision and recall in your own words, without reciting formulas.**

Precision asks how much of what you produced is right, and recall asks how much of what exists you found. A cautious system that only answers when sure has high precision and low recall, and one that answers everything has the reverse. For generative models, precision is the fraction of samples that land on the real-data manifold (quality), and recall is the fraction of real data the model can produce (coverage). Raising guidance or truncation is like raising a classifier's threshold: precision rises and recall falls.
:::

::: qa
**Why did Naeem et al. replace precision/recall with density/coverage?**

Improved precision/recall estimate each manifold as a union of k-NN balls, and outliers have large radii, so one real outlier can make many poor samples count as precise. The metrics also miss 1 under a perfect model, and $k$ has no principled choice. Density counts overlaps with real neighbourhoods instead of testing membership in a union, so no single ball dominates, and its expectation is 1 for a perfect model. Coverage builds balls only around trusted real samples and has a closed-form expectation that guides the choice of $k$ and sample sizes.
:::

## Text–image alignment and human preference

**CLIPScore** (Hessel et al., 2021) is $w\cdot\max(\cos(E_T(c),E_I(v)),0)$ with $w=2.5$ and ViT-B/32, averaged over a prompt set. Many text-to-image papers instead report $100\cdot\max(\cos,0)$ with other CLIP backbones, so values are not comparable across papers. It is cheap and reference-free. But CLIP mostly matches the gist of a caption: it is weak at counting, spatial relations and attribute binding, and it saturates for strong models.

The fix is to break the prompt into checks that a specialist model can verify:

- **GenEval** (Ghosh et al., 2023): six tasks (single object, two objects, counting, colours, position, attribute binding). Mask2Former detects objects, CLIP ViT-L/14 classifies colours on crops, box centroids decide position, and each image gets a binary score. GenEval agreed with annotators on 83% of images (inter-annotator 88%, thresholded CLIPScore 80%).
- **T2I-CompBench** (Huang et al., 2023): 6,000 prompts covering colour, shape and texture binding, spatial and non-spatial relations, and complex compositions. It is scored with disentangled BLIP-VQA, a UniDet-based spatial check, and a "3-in-1" metric for complex prompts.
- **DPG-Bench** (from ELLA): 1,065 dense prompts averaging about 67 words. Davidsonian-scene-graph questions are answered by mPLUG-large.
- **VQAScore** (Lin et al., 2024): the probability that a VQA model answers "Yes" to "Does this figure show '{text}'?".

The judge's error rate sets a limit on all of these benchmarks.

**Preference models.** ImageReward (BLIP-based, 137k expert comparisons), PickScore (CLIP-H on Pick-a-Pic, over 500k user choices, 70.5% held-out accuracy vs. 68.0% for experts) and HPSv2 (CLIP ViT-H/14, 798k choices) learn a utility $s(c,x)$. PickScore and HPSv2 train it with a Bradley–Terry objective: $p(x_1\succ x_2\mid c)=e^{s(c,x_1)}/(e^{s(c,x_1)}+e^{s(c,x_2)})$, with cross-entropy against the human choice. Once a preference model is used as a reward or for checkpoint selection, the generator exploits it (saturated colours, favoured styles), so evaluate with one you did not optimize and with humans. Prompt sources are biased too: HPSv2's authors found that 28.5% of DiffusionDB prompts contain "artstation", and rewrote the prompts with ChatGPT to remove such style words. **Arenas** fit pairwise user votes with Bradley–Terry; Chatbot Arena moved from online Elo to a BT maximum-likelihood fit with robust confidence intervals.

::: qa
**A model gets a high CLIPScore on "a red cube on top of a blue sphere" but draws a blue cube next to a red sphere. Why, and what would you use instead?**

CLIP's pooled embedding behaves partly like a bag of concepts. "Red", "blue", "cube" and "sphere" are all present, so similarity is high whatever the binding or relation. Decompose the prompt instead: detection plus per-object colour classification and geometric rules (GenEval-style), or VQA questions such as "Is the cube red?" and "Is the cube on the sphere?" (VQAScore, DPG-Bench, BLIP-VQA). Validate the judge against a small human-labelled set, since its errors limit the benchmark.
:::

::: qa
**How are PickScore and HPSv2 trained, and why shouldn't one be your only metric?**

They fine-tune CLIP-H so that a scaled image–text similarity acts as a utility. A softmax over two images' scores gives a Bradley–Terry preference probability, trained with cross-entropy against which image people chose. They predict held-out choices well, but they encode one population's taste on one prompt distribution. If you fine-tune or select checkpoints against them, the generator learns their blind spots and the score rises faster than real quality. Report a held-out preference model, distribution and alignment metrics, and a human study.
:::

::: {.qa asked="Tencent Youtu CV intern, 2nd round (Nowcoder)"}
**How would you build a metric that flags anomalous or ambiguous generations?**

Set-level metrics cannot flag single images, so you need per-sample scores. For anomalies: no-reference quality and artifact detectors, keypoint or pose checks for hands and bodies, and a vision-language judge with targeted questions ("Does each hand have five fingers?"). For ambiguity: split the prompt into atomic facts scored by VQA, and measure disagreement across judges or across seeds of the same prompt, since high variance signals an underspecified prompt. Calibrate each detector on a human-labelled set and report the flagger's own precision and recall.
:::

## Video: FVD and VBench

**FVD** (Unterthiner et al., 2018) applies the Fréchet formula to features from I3D trained on Kinetics-400; KVD is the kernel analogue. Clip length, fps, resolution and the number of videos all change the value. Ge et al. (2024) showed a **content bias**. In one UCF-101 test (per-frame elastic distortion), temporal corruption with frame quality held fixed raised I3D-based FVD by 16.8% but an FVD on VideoMAE-v2 features (fine-tuned on SSv2) by 213%. Resampling a weighted subset of motionless videos (the first frame repeated) cut I3D FVD by roughly 30–55%, but the VideoMAE-v2 version by only about 5–13%. They trace this to a supervised classifier trained on actions recognizable from single frames, and recommend self-supervised video features.

**VBench** (Huang et al., 2023) scores 16 dimensions, each with a specialist model and checked against human preference labels. *Temporal quality:* subject consistency, background consistency, temporal flickering, motion smoothness, dynamic degree. *Frame-wise quality:* aesthetic quality and imaging quality (MUSIQ). *Semantics:* object class, multiple objects, human action, colour, spatial relationship, scene. *Style:* appearance style, temporal style. Plus *overall consistency*. Read them together: a static video scores near the top on consistency and flicker and poorly on dynamic degree.

::: qa
**How can a video model that generates almost no motion still get a good FVD, and how would you evaluate a text-to-video model instead?**

I3D features come from an action classifier trained on Kinetics, where most actions can be recognized from appearance, so FVD mainly tracks per-frame quality. Ge et al. showed temporal corruption barely moves it and motionless videos can halve it. A better evaluation separates axes: frame quality (FD$_\text{DINOv2}$, an NR score), temporal quality (FVD with self-supervised features, VBench's consistency, smoothness and dynamic-degree dimensions read together), alignment (VQA or video-language checks, including actions and camera motion), and a side-by-side human study. Fix clip length, fps and resolution, and match inference cost.
:::

## Full-reference metrics for restoration

**PSNR.** $\mathrm{PSNR}=10\log_{10}(\mathrm{MAX}^2/\mathrm{MSE})$, with MSE averaged over pixels and channels. Super-resolution papers usually compute it on the Y channel after cropping a border equal to the scale factor, and these conventions can shift results by tenths of a dB or more.

**Why MSE prefers blur.** With $m(y)=\mathbb{E}[X\mid Y=y]$, $\mathbb{E}[\lVert X-\hat x\rVert^2\mid y]=\mathbb{E}[\lVert X-m\rVert^2\mid y]+\lVert m-\hat x\rVert^2$, so the posterior mean is optimal. When $y$ is ambiguous, the posterior has many sharp modes (edge positions, texture phases), and their average is smooth. If an edge sits at one of two positions with equal probability, the MMSE output is a half-intensity two-step ramp. A posterior sample $\hat X\sim p(x\mid y)$, drawn independently of $X$ given $y$, has a zero cross term:

$$\mathbb{E}\lVert X-\hat X\rVert^2=\mathbb{E}\lVert X-m\rVert^2+\mathbb{E}\lVert\hat X-m\rVert^2=2\,\mathrm{MMSE},$$

so an exact posterior sampler loses 3 dB ($10\log_{10}2$, in average MSE) relative to the MMSE estimator. Other estimators with perfectly realistic outputs can lose less (see the perception–distortion section). L1 gives the per-pixel posterior median, still a single point summary of the posterior.

**SSIM** (Wang et al., 2004) compares local windows on luminance, contrast and structure:

$$l=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1},\quad c=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2},\quad s=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}.$$

With $C_3=C_2/2$, the identity $2\sigma_x\sigma_y+C_2=2(\sigma_x\sigma_y+C_3)$ collapses $c\cdot s$ and gives

$$\mathrm{SSIM}=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)},$$

with $C_i=(K_iL)^2$, $K_1=0.01$, $K_2=0.03$, an 11×11 Gaussian window ($\sigma=1.5$), and an average over the map (MSSIM). In the paper's "Boat" example, images with identical MSE (210) range from MSSIM 0.99 (mean shift) to 0.69 (JPEG). SSIM is still pixel-aligned, so it does not reward realistic texture that differs from the ground-truth texture.

**LPIPS** (Zhang et al., 2018) unit-normalizes deep features $\hat y^l$ (AlexNet, VGG or SqueezeNet) across channels and computes

$$d(x,x_0)=\sum_l\frac{1}{H_lW_l}\sum_{h,w}\big\lVert w_l\odot(\hat y^l_{hw}-\hat y^l_{0,hw})\big\rVert_2^2,$$

with non-negative channel weights $w_l$ fitted to the 2AFC training judgments of BAPPS, a dataset of 484k human judgments (2AFC and JND) on 64×64 patches. On 2AFC agreement, humans score 73.9%, L2 63.2%, SSIM 63.1%, and LPIPS about 70%. It is a common training loss, so evaluating with it can flatter models trained on it. **DISTS** (Ding et al., 2020) uses VGG16 with $\ell_2$ pooling and compares SSIM-like terms on global feature means (texture) and on global variances and covariance (structure), with learned weights summing to 1. Because it uses global statistics, it tolerates resampled texture, which suits generative restoration.

::: {.qa asked="Alibaba DME, Nowcoder question list 2024"}
**What is the difference between PSNR, SSIM and LPIPS?**

PSNR is a log-scaled pixel MSE. It weighs every pixel error equally and is maximized by the blurry posterior mean. SSIM compares local means, contrasts and correlations, so it is less sensitive than MSE to uniform brightness and contrast shifts and emphasizes structure, but it is still a handcrafted, pixel-aligned statistic. LPIPS is a distance between normalized deep features with weights fitted to human 2AFC judgments, so it tracks perceived similarity better (about 70% vs. 63% agreement on BAPPS) and penalizes blur. All three need ground truth. Report PSNR/SSIM for fidelity and LPIPS or DISTS for perceptual similarity.
:::

::: {.qa asked="GenAI RS loop, candidate prep blog 2026"}
**Write down SSIM and explain each term.**

$\mathrm{SSIM}=l\cdot c\cdot s$, with luminance $l=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}$, contrast $c=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}$, and structure $s=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}$, the correlation of the normalized patches. With $C_3=C_2/2$ it becomes $\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$. The constants $(0.01L)^2$ and $(0.03L)^2$ stabilize near-zero denominators. It is computed in 11×11 Gaussian windows and averaged, is at most 1, and can be negative.
:::

::: {.qa asked="Baidu AIGC intern (Nowcoder, 2024)"}
**What is the difference between L1 and L2 losses, and what are their pros and cons?**

L2's minimizer under uncertainty is the conditional mean. It is smooth with strong gradients on large errors, but it is outlier-sensitive and gives blurry averages for multimodal targets. L1's minimizer is the per-pixel conditional median. It is more robust to outliers, and in practice it often trains to better restorations: Zhao et al. (2017) found L1-trained networks reached lower L2 error than L2-trained ones and avoided L2's splotchy artifacts in flat regions, which they attribute to L2 getting stuck in poorer local minima rather than to the minimizer itself. Its constant-magnitude gradient makes convergence near the optimum less smooth. Both are point estimates, so neither produces realistic texture when the posterior is spread out. PSNR directly rewards the L2 solution.
:::

## No-reference image quality assessment

Real-world restoration rarely has ground truth.

- **NIQE** (Mittal et al., 2013) fits a multivariate Gaussian $(\nu_1,\Sigma_1)$ to 36 natural-scene-statistics features from sharp 96×96 patches of pristine images, fits $(\nu_2,\Sigma_2)$ to all 96×96 patches of the test image (no sharpness selection, since blur is itself a distortion), and reports $\sqrt{(\nu_1-\nu_2)^\top\big(\tfrac{\Sigma_1+\Sigma_2}{2}\big)^{-1}(\nu_1-\nu_2)}$. It needs no human scores. PIRM 2018's **Perceptual Index** is $\tfrac12\big((10-\mathrm{Ma})+\mathrm{NIQE}\big)$; lower is better.
- **MUSIQ** (Ke et al., 2021) is a transformer over multi-scale patches at native aspect ratio, with hash-based spatial and scale embeddings, trained on MOS data.
- **MANIQA** (Yang et al., 2022) combines ViT features, channel attention and Swin blocks with patch-weighted scoring. It won the NTIRE 2022 no-reference track.
- **CLIP-IQA** (Wang et al., 2023) takes the softmax of the cosine similarities between the image and "Good photo." / "Bad photo.", using ResNet-50 CLIP with the positional embedding removed so any resolution works.
- **Q-Align** (Wu et al., 2023) fine-tunes mPLUG-Owl2 to output one of five levels (bad … excellent) obtained by binning MOS. Its score is $\sum_{i=1}^5 i\,p_i$, with a softmax over only the five level tokens.

**Gaming.** Each is trained on particular distortions (camera, compression, early super-resolution). Generative restorers can raise these scores with added high-frequency texture, noise-like detail or over-sharpening, even when the detail is wrong. Using an NR score as a loss, reward or checkpoint selector invites the generator to exploit it. Pair it with a fidelity check: a full-reference metric on synthetic pairs, or input consistency (degrade the output and compare it with the input).

::: qa
**Your diffusion super-resolution model improves MUSIQ, MANIQA and CLIP-IQA but loses 2 dB PSNR versus a regression baseline. Is it better?**

The two models aim at different points on the perception–distortion curve, so neither metric alone decides. A 2 dB loss fits a move from near-MMSE towards realistic outputs: exact posterior sampling costs about 3 dB relative to the MMSE estimator, and the best perfectly realistic estimator costs at most that. But NR gains can come from hallucinated texture, so check LPIPS/DISTS on synthetic pairs, input consistency, and a 2AFC human study. Compare against perceptual methods at similar distortion, ideally by tracing each method's curve with a knob such as loss weight or sampling temperature. For medical or forensic uses, hallucination is a failure whatever the NR score says.
:::

::: qa
**How does Q-Align turn a multimodal LLM into a quality regressor, and why levels instead of numbers?**

Human studies collect categorical ratings and average them into a MOS. Q-Align mirrors this: MOS values are binned into five words, and the LMM learns to output the word. At inference it applies a softmax over the five level tokens and takes the expectation $\sum_i i\,p_i$, which gives a continuous score that reflects the model's uncertainty. Level words suit a language model's pretraining better than generating calibrated digits, and the expectation gives finer resolution than an argmax.
:::

## The perception–distortion trade-off

Blau and Michaeli (2018) let $X\sim p_X$ be a clean image, $Y$ its degraded observation, and $\hat X$ any, possibly stochastic, estimator. **Distortion** is $\mathbb{E}[\Delta(X,\hat X)]$ for any full-reference $\Delta$. **Perceptual quality** is a divergence $d(p_X,p_{\hat X})$ between distributions. This is motivated by the real-vs-fake test, where the best achievable observer accuracy is $\tfrac12+\tfrac12 d_{\mathrm{TV}}(p_X,p_{\hat X})$. Define

$$P(D)=\min_{p_{\hat X\mid Y}}\ d\big(p_X,p_{\hat X}\big)\quad\text{s.t.}\quad\mathbb{E}\big[\Delta(X,\hat X)\big]\le D.$$

**Theorem.** $P(D)$ is non-increasing, and convex if $d(p,q)$ is convex in $q$ (as for f-divergences). Nothing is assumed about $\Delta$, so the trade-off is not an MSE artifact. When the degradation loses information, no algorithm reaches both minimum distortion and perfect realism. The MMSE estimate averages sharp alternatives and leaves the image manifold. Perfect realism forces committing to one mode at a time, which costs distortion. Convexity means the curve is steepest at the low-distortion end: the first bit of extra distortion buys the most realism.

**Closed form for MSE and $W_2$.** With $X^*=\mathbb{E}[X\mid Y]$, $D^*=\mathbb{E}\lVert X-X^*\rVert^2$ and $P^*=W_2(p_X,p_{X^*})$, Freirich et al. (2021) proved

$$D(P)=D^*+\big[(P^*-P)_+\big]^2,$$

attained by $\hat X_P=(1-P/P^*)\hat X_0+(P/P^*)X^*$, a straight interpolation between the best perfect-realism estimator and the MMSE estimate. The endpoint follows from orthogonality. Any $\hat X$ built from $Y$ and independent noise has $\mathbb{E}\lVert X-\hat X\rVert^2=D^*+\mathbb{E}\lVert X^*-\hat X\rVert^2$. With $p_{\hat X}=p_X$ the second term is at least $W_2^2(p_{X^*},p_X)\le D^*$. So perfect realism costs at most $2D^*$, and posterior sampling costs exactly $2D^*$.

**Consequences.** Report one full-reference and one no-reference metric, and compare methods in the perception–distortion plane. PIRM 2018 ranked entries by PI within RMSE regions (≤11.5, ≤12.5, ≤16). In its human study of the 10 top submissions, RMSE and SSIM were anti-correlated with mean opinion scores and PI was the most correlated measure (Spearman 0.83), though PI tracked human scores across regions better than within one region. Blau and Michaeli (2019) add rate: $R(D,P)=\min I(X;\hat X)$ subject to both constraints. Requiring realism raises the rate–distortion curve, and for MSE $R(D,0)\le R(D/2,\infty)$: realism at distortion $D$ costs no more bits than plain compression at $D/2$. Stochastic decoders matter at low rates. So generative codecs and tokenizers trade PSNR for realism, and should be compared at matched rate on both axes.

::: qa
**State the perception–distortion trade-off. Is it specific to MSE?**

Let perceptual quality be a divergence between the distributions of outputs and natural images, and distortion an expected pairwise error. Blau and Michaeli showed that the best divergence achievable at distortion $D$ is non-increasing in $D$ and, for divergences convex in their second argument, convex. When the degradation loses information, minimum distortion and zero divergence cannot both be reached. No assumption is placed on the distortion measure, so it holds for MSE, SSIM-type and deep-feature distortions alike. Minimizing any expected pairwise loss pulls towards a summary of the posterior, while realism needs a committed sample.
:::

::: qa
**Show that posterior sampling achieves perfect perceptual quality at exactly twice the MMSE.**

If $\hat X\sim p(x\mid y)$, then $p_{\hat X}=p_X$ marginally, so the divergence is zero. Write $X-\hat X=(X-m)-(\hat X-m)$ with $m=\mathbb{E}[X\mid y]$. Given $y$, the two are independent and identically distributed, so the cross term vanishes and each square has expectation $\mathrm{tr}\,\mathrm{Cov}(X\mid y)$. Averaging over $y$ gives $2\,\mathrm{MMSE}$, a 3 dB PSNR loss. Freirich et al. show the best perfect-realism estimator achieves $D^*+W_2^2(p_X,p_{X^*})\le 2D^*$.
:::

## Human evaluation

**Protocols.** In *2AFC / side-by-side* tests, raters pick the better of two outputs for the same input on a named axis (realism, alignment, overall). *Real-vs-fake* tests ask raters to spot the generated image, and chance-level accuracy means success. *MOS* collects 1–5 ratings, which need per-rater normalization and screening. *Arenas* fit a global ranking from pairwise votes.

**Bradley–Terry and Elo.** BT models $P(i\succ j)=\sigma(\theta_i-\theta_j)$. For one vote where $A$ beats $B$, $\partial_{\theta_A}\log\sigma(\theta_A-\theta_B)=1-\sigma(\theta_A-\theta_B)$, and Elo's update $R_A\leftarrow R_A+K(S_A-E_A)$, with $E_A=1/(1+10^{(R_B-R_A)/400})$, is exactly one stochastic-gradient step on the BT likelihood (scale $400/\ln 10$). Online Elo therefore depends on vote order and $K$. A batch maximum-likelihood fit with bootstrap or sandwich intervals is preferred.

**Sample size.** A win rate from $n$ independent votes has standard error at most $0.5/\sqrt n$. A 95% half-width $h$ needs $n\approx 0.96/h^2$: 385 for ±5 points, 1,068 for ±3. Detecting 55% vs. 50% at $\alpha=0.05$ with 80% power needs

$$n\approx\left(\frac{1.96\cdot 0.5+0.84\sqrt{0.55\cdot 0.45}}{0.05}\right)^2\approx 782.$$

Votes cluster by prompt and rater, which shrinks the effective $n$ by $1+(m-1)\rho$. Bootstrap over prompts and raters, or fit a mixed-effects logistic model. Correct for multiple comparisons, randomize left/right placement, hide model identity, fix prompts in advance, and include gold items.

::: qa
**How many 2AFC judgments do you need to claim your model is preferred, and what makes naive counts misleading?**

Treat the win rate as binomial: ±3 points at 95% needs about 1,070 judgments, and detecting a true 55% preference with 80% power needs about 780. But votes are correlated: raters on the same prompt share its difficulty, and a prolific rater contributes many dependent votes, so the effective $n$ is smaller by the design effect $1+(m-1)\rho$. Use a bootstrap that resamples prompts and raters, or a mixed-effects model, and report the interval, not just the percentage.
:::

## Reporting checklist

- State the protocol: sample count, reference statistics, resizing and format, feature extractor and weights, prompt set.
- Use several seeds and report mean ± std or bootstrap intervals.
- Report complementary metrics: FID with FD$_\text{DINOv2}$ or CMMD, precision/recall or density/coverage, a task metric (alignment or fidelity), and a human study for headline claims.
- Show curves, not points: guidance sweeps, NFE sweeps, the perception–distortion plane.
- Match compute: same NFE and resolution, baselines tuned equally (for example their best guidance), and training compute reported.
- Never select checkpoints on the reported test metric, or present a reward you optimized as evidence. Check memorization with nearest-neighbour retrieval against the training set.

::: {.qa asked="GenAI RS loop, candidate prep blog 2026"}
**Which metric would you actually trust for a product, and when does a metric stop meaning anything?**

I would trust the measure closest to the user decision: side-by-side human preference on prompts sampled from real traffic, with confidence intervals, plus targeted checks for known failures (text, anatomy, safety). Automatic metrics are for fast iteration once they have been checked against those human results for this model family. A metric stops meaning anything when it saturates (gaps within seed noise, as FID near the top), when it is optimized directly (reward hacking, NR-IQA gaming), when outputs leave its feature extractor's training distribution, or when protocols differ across the models compared.
:::

::: qa
**Design an evaluation framework for an AIGC model that generates e-commerce product images.**

Start from what can go wrong for the business. Product fidelity: compare logo, colour and shape with the source photo using CLIP-I/DINO similarity, masked LPIPS and OCR. Image quality: NR-IQA or aesthetic scores plus artifact detectors. Brief compliance: VQA checks on scene and style. Policy compliance: safety detectors. Calibrate these offline against side-by-side ratings from designers or merchants, then run online A/B tests on click-through, conversion and returns with guardrail metrics. Break results down by product category, since averages hide where the model fails.
:::

## Rapid-fire review

::: qa
**Which Inception layer and input size does FID use?**

The 2048-d pool3 activations, with images resized to 299×299. The resize method is part of the metric.
:::

::: qa
**What is the maximum Inception Score?**

1000, the number of classes, because $\mathrm{IS}=\exp(I(x;y))\le\exp(H(y))$.
:::

::: qa
**Is FID between a dataset and a held-out split of itself zero?**

Not at finite $N$. The estimator is biased upward, roughly by an amount that shrinks like $1/N$.
:::

::: qa
**PSNR of an 8-bit image whose MSE is 1?**

$10\log_{10}255^2\approx 48.13$ dB. Halving the MSE adds about 3.01 dB.
:::

::: qa
**What does CLIP-IQA compute?**

A softmax over the image's cosine similarities to "Good photo." and "Bad photo.". The probability of the positive prompt is the score.
:::

::: qa
**What gives FVD its features, and what is its main bias?**

I3D trained on Kinetics-400. FVD is driven mostly by per-frame appearance and only weakly by temporal artifacts.
:::

::: qa
**Raising guidance makes precision go up and recall go down. What happened?**

Samples moved towards dense, prototypical regions: each image looks better, but less of the data distribution is covered.
:::

::: qa
**What is the expected coverage of a perfect model with $k=5$ and $M=N$ large?**

About $1-2^{-5}\approx 0.97$. Expected density is 1.
:::

::: qa
**How do you check fidelity in real-world super-resolution without ground truth?**

Degrade the output with the assumed forward model and compare it with the input (input-consistency PSNR), and report this next to no-reference scores.
:::

::: {.qa asked="GenAI RS loop, candidate prep blog 2026"}
**What is VIF measuring?**

Visual Information Fidelity (Sheikh and Bovik, 2006) models the reference's wavelet coefficients as a natural-image source (a Gaussian scale mixture), the distortion as a channel with gain and additive noise, and human vision as further additive noise. VIF is a ratio of two mutual informations with the source: what can be extracted from the perceived distorted image over what can be extracted from the perceived reference. It equals 1 for an undistorted image, falls as information is lost, and can exceed 1 for contrast enhancement.
:::

::: qa
**Why do arena leaderboards fit Bradley–Terry instead of reporting raw Elo?**

Elo is online SGD on the BT likelihood, so it depends on vote order and $K$. A batch BT fit uses all votes symmetrically and gives confidence intervals.
:::

## References

1. Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (FID), 2017. <https://arxiv.org/abs/1706.08500>
2. TTUR reference FID implementation. <https://github.com/bioinf-jku/TTUR>
3. Salimans et al., "Improved Techniques for Training GANs" (Inception Score), 2016. <https://arxiv.org/abs/1606.03498>
4. Barratt and Sharma, "A Note on the Inception Score," 2018. <https://arxiv.org/abs/1801.01973>
5. Theis, van den Oord, Bethge, "A note on the evaluation of generative models," 2016. <https://arxiv.org/abs/1511.01844>
6. Bińkowski et al., "Demystifying MMD GANs" (KID), 2018. <https://arxiv.org/abs/1801.01401>
7. Chong and Forsyth, "Effectively Unbiased FID and Inception Score and where to find them," 2020. <https://arxiv.org/abs/1911.07023>
8. Parmar, Zhang, Zhu, "On Aliased Resizing and Surprising Subtleties in GAN Evaluation" (clean-fid), 2022. <https://arxiv.org/abs/2104.11222>; code: <https://github.com/GaParmar/clean-fid>
9. Kynkäänniemi et al., "The Role of ImageNet Classes in Fréchet Inception Distance," 2023. <https://arxiv.org/abs/2203.06026>
10. Stein et al., "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models," 2023. <https://arxiv.org/abs/2306.04675>
11. Jayasumana et al., "Rethinking FID: Towards a Better Evaluation Metric for Image Generation" (CMMD), 2024. <https://arxiv.org/abs/2401.09603>
12. Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models" (EDM2), 2024. <https://arxiv.org/abs/2312.02696>; code: <https://github.com/NVlabs/edm2>
13. Sajjadi et al., "Assessing Generative Models via Precision and Recall," 2018. <https://arxiv.org/abs/1806.00035>
14. Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models," 2019. <https://arxiv.org/abs/1904.06991>
15. Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models," 2020. <https://arxiv.org/abs/2002.09797>
16. Hessel et al., "CLIPScore: A Reference-free Evaluation Metric for Image Captioning," 2021. <https://arxiv.org/abs/2104.08718>
17. Ghosh, Hajishirzi, Schmidt, "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment," 2023. <https://arxiv.org/abs/2310.11513>
18. Huang et al., "T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation," 2023. <https://arxiv.org/abs/2307.06350>
19. Hu et al., "ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment" (DPG-Bench), 2024. <https://arxiv.org/abs/2403.05135>
20. Lin et al., "Evaluating Text-to-Visual Generation with Image-to-Text Generation" (VQAScore), 2024. <https://arxiv.org/abs/2404.01291>
21. Xu et al., "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation," 2023. <https://arxiv.org/abs/2304.05977>
22. Kirstain et al., "Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation," 2023. <https://arxiv.org/abs/2305.01569>
23. Wu et al., "Human Preference Score v2," 2023. <https://arxiv.org/abs/2306.09341>
24. Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference," 2024. <https://arxiv.org/abs/2403.04132>
25. Unterthiner et al., "Towards Accurate Generative Models of Video: A New Metric & Challenges" (FVD), 2018. <https://arxiv.org/abs/1812.01717>
26. Ge et al., "On the Content Bias in Fréchet Video Distance," 2024. <https://arxiv.org/abs/2404.12391>
27. Huang et al., "VBench: Comprehensive Benchmark Suite for Video Generative Models," 2023. <https://arxiv.org/abs/2311.17982>
28. Wang, Bovik, Sheikh, Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," IEEE TIP 2004. <https://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf>
29. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (LPIPS), 2018. <https://arxiv.org/abs/1801.03924>
30. Ding, Ma, Wang, Simoncelli, "Image Quality Assessment: Unifying Structure and Texture Similarity" (DISTS), 2020. <https://arxiv.org/abs/2004.07728>
31. Zhao, Gallo, Frosio, Kautz, "Loss Functions for Image Restoration with Neural Networks," IEEE TCI 2017. <https://arxiv.org/abs/1511.08861>
32. Sheikh and Bovik, "Image Information and Visual Quality" (VIF), IEEE TIP 2006. <https://doi.org/10.1109/TIP.2005.859378>
33. Mittal, Soundararajan, Bovik, "Making a 'Completely Blind' Image Quality Analyzer" (NIQE), IEEE SPL 2013. <https://doi.org/10.1109/LSP.2012.2227726>
34. Ke et al., "MUSIQ: Multi-scale Image Quality Transformer," 2021. <https://arxiv.org/abs/2108.05997>
35. Yang et al., "MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment," 2022. <https://arxiv.org/abs/2204.08958>
36. Wang, Chan, Loy, "Exploring CLIP for Assessing the Look and Feel of Images" (CLIP-IQA), 2023. <https://arxiv.org/abs/2207.12396>
37. Wu et al., "Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels," 2023. <https://arxiv.org/abs/2312.17090>
38. Blau and Michaeli, "The Perception-Distortion Tradeoff," 2018. <https://arxiv.org/abs/1711.06077>
39. Blau et al., "The 2018 PIRM Challenge on Perceptual Image Super-resolution," 2018. <https://arxiv.org/abs/1809.07517>
40. Freirich, Michaeli, Meir, "A Theory of the Distortion-Perception Tradeoff in Wasserstein Space," 2021. <https://arxiv.org/abs/2107.02555>
41. Blau and Michaeli, "Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff," 2019. <https://arxiv.org/abs/1901.07821>
