---
title: "Image and Video Restoration"
subtitle: "Super-resolution and restoration as inverse problems: signal-processing basics, degradation models, regression, GAN and diffusion restorers, posterior samplers, and video SR."
order: 7
updated: 2026-11-08
---

Image restoration recovers a clean image from a low-resolution, blurry, noisy or compressed observation. Super-resolution (SR) is the flagship task and contains most of the field's ideas: the inverse map is not unique, the true degradation is unknown, and the best answer depends on whether you are scored on pixel error or on realism. Interviewers use it to check whether you can connect estimation theory (MMSE, MAP, posterior sampling), signal processing (sampling, aliasing, blur) and generative priors (GANs, diffusion). Metric definitions are in [evaluation metrics](evaluation-metrics.html), diffusion basics in [diffusion models](diffusion-models.html), GAN theory in [GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html).

## Restoration as an inverse problem

### Forward model and ill-posedness

Write the observation as

$$y = A(x) + n,$$

with $x$ the clean image, $A$ the degradation and $n$ noise. Denoising has $A = I$, deblurring $Ax = k * x$, inpainting a mask, SR $Ax = (k * x)\downarrow_s$ (blur, then keep every $s$-th pixel), and JPEG artifact removal a nonlinear $A$. In the diffusion sections $x_0$ is the clean image and $x_t$ the noisy state.

A problem is well-posed if a solution exists, is unique and depends continuously on the data. SR fails uniqueness: for $\times s$ SR of $N$ pixels, $y$ has $N/s^2$ entries, so the null space of $A$ has dimension at least $N(1-1/s^2)$, which is 15/16 of all directions at $\times 4$. Deblurring fails stability. For linear $A = U\Sigma V^\top$ with singular values $s_i$,

$$A^{+}y=\sum_{i:\,s_i>0}(v_i^\top x)\,v_i+\sum_{i:\,s_i>0}\frac{u_i^\top n}{s_i}\,v_i .$$

The first sum is the row-space projection of $x$, so null-space content is lost; the second divides noise by small singular values. Any restorer needs a prior to fill the null space and to regularize ill-conditioned directions.

### MAP, MMSE and posterior sampling

From $p(x\mid y) \propto p(y\mid x)\,p(x)$ there are three standard outputs. **MAP** maximizes $\log p(y\mid x) + \log p(x)$; with Gaussian noise this is the classical $\frac{1}{2\sigma^2}\|y - A(x)\|^2 + R(x)$. **MMSE** returns $\mathbb{E}[x\mid y]$ and minimizes expected squared error, the quantity PSNR scores. **Posterior sampling** draws $\hat x \sim p(x\mid y)$, a plausible image whose null-space content varies between draws. If $x \sim \mathcal{N}(\mu, \Sigma)$, $A$ is linear and $n \sim \mathcal{N}(0, \sigma^2 I)$, the posterior is Gaussian and MAP equals MMSE:

$$\hat x=\mu+\Sigma A^\top(A\Sigma A^\top+\sigma^2 I)^{-1}(y-A\mu).$$

For convolutions with a stationary prior this becomes the Wiener filter below. Natural images are far from Gaussian, which is where the three estimators separate.

### Regression to the mean

For any $f$, insert $\mu(y) = \mathbb{E}[x\mid y]$:

$$\mathbb{E}\|x-f(y)\|^2=\mathbb{E}\|x-\mu(y)\|^2+\mathbb{E}\|\mu(y)-f(y)\|^2+2\,\mathbb{E}\big[(x-\mu(y))^\top(\mu(y)-f(y))\big].$$

Given $y$, $\mu(y)-f(y)$ is fixed and $\mathbb{E}[x-\mu(y)\mid y]=0$, so the cross term vanishes and the L2 optimum is $f^*=\mu(y)$. Now take a frequency $\omega$ whose phase the input does not determine (say, the sub-pixel position of a fine texture), $X(\omega) = a\,e^{i\varphi}$ with $\varphi$ uniform given $y$:

$$\mathbb{E}[X(\omega)\mid y]=a\,\mathbb{E}[e^{i\varphi}]=0,$$

although every plausible image has energy $a^2$ there. The mean cancels undetermined detail. L1 training gives the per-pixel conditional median, which also suppresses undetermined detail.

### The perception–distortion trade-off in practice

Blau and Michaeli showed that low distortion and natural-looking output statistics conflict for ill-posed problems (theorem in [evaluation metrics](evaluation-metrics.html)). One consequence takes two lines: draw $\hat x \sim p(\cdot\mid y)$ independently of $x$ given $y$, with $\mu = \mathbb{E}[x\mid y]$:

$$\mathbb{E}\|x-\hat x\|^2=\mathbb{E}\|x-\mu\|^2+\mathbb{E}\|\hat x-\mu\|^2=2\,\mathbb{E}\|x-\mu\|^2 .$$

The cross term vanishes by conditional independence, and $\hat x$ has the same conditional law as $x$. Posterior sampling costs exactly twice the MMSE ($10\log_{10}2 \approx 3$ dB of PSNR), and Blau and Michaeli prove that perfect perceptual quality never requires more than this factor. In practice: PSNR-oriented and perceptual models are different products; good systems expose a knob along the curve (ESRGAN network interpolation, CodeFormer's $w$, the starting noise level in diffusion SR); and the lowest MSE at perfect realism comes from estimating the posterior mean and transporting it to the image distribution (Freirich et al.; approximated by PMRF).

::: key
An L2-trained restorer outputs the average of all plausible answers, and averaging cancels whatever the input does not determine. Generative restorers pay a bounded MSE cost (exactly 2x MMSE for posterior sampling) for outputs that look real.
:::

::: pitfall
"Diffusion SR hallucinates because it is broken." A posterior sampler must invent null-space content; that is what separates it from the posterior mean. The real questions are whether the invention is consistent with $y$ and whether the user can control how much is invented.
:::

::: qa
**Why does an SR network trained with MSE produce blurry images, and why doesn't a bigger network fix it?**

The MSE minimizer is the posterior mean $\mathbb{E}[x\mid y]$, so even a perfect, unlimited network outputs the average of all high-resolution images consistent with the input. Where the input does not determine a detail (texture phase, exact edge position), candidates disagree and averaging suppresses that band; with uniformly uncertain phase the conditional mean at that frequency is zero. Capacity only improves the estimate of the same mean. The fix is a different objective: adversarial or perceptual losses, or sampling from $p(x\mid y)$.
:::

::: qa
**Compare MAP, MMSE and posterior sampling. Which maximizes PSNR, and what does sampling cost?**

MMSE minimizes expected MSE, so it is the PSNR-oriented choice, but it blurs when the posterior is spread over many images. MAP is the posterior mode; with a Gaussian prior, linear operator and Gaussian noise it equals MMSE, and with other priors it is often an atypical image (Blau and Michaeli show its output distribution is also wrong). Posterior sampling returns typical images with natural statistics, at exactly twice the MMSE, about 3 dB lower PSNR.
:::

## Signal-processing basics

### Sampling and aliasing

Sampling with period $T$ multiplies by an impulse train, whose Fourier transform is an impulse train with spacing $1/T$, so

$$X_s(f)=\frac{1}{T}\sum_{k\in\mathbb{Z}}X\!\left(f-\frac{k}{T}\right).$$

The spectrum is copied every $f_s = 1/T$. If $X$ is band-limited to $|f| < B$ and $f_s > 2B$, the copies do not overlap and sinc interpolation recovers $x$; $f_s/2$ is the Nyquist frequency. Content above it lands on $|f - kf_s|$: aliasing, seen as moiré and jagged edges.

### Downsampling, blur and the Fourier domain

For decimation $y[n] = x[sn]$, multiply by the comb $c[m] = \frac{1}{s}\sum_{k=0}^{s-1}e^{i2\pi km/s}$ (1 when $s$ divides $m$, else 0), which adds spectral copies shifted by $2\pi k/s$, then drop zeros, which stretches frequency by $s$:

$$Y(e^{i\omega})=\frac{1}{s}\sum_{k=0}^{s-1}X\!\left(e^{i(\omega-2\pi k)/s}\right).$$

The band $\pi/s \le |\omega| \le \pi$ folds onto the retained band. An anti-aliasing filter low-passes near $\pi/s$ first, trading aliasing for blur.

Blur multiplies the spectrum: $Y = KX + N$. A Gaussian with standard deviation $\sigma$ pixels has $K(f) = e^{-2\pi^2\sigma^2 f^2}$ ($f$ in cycles per pixel; at $\sigma=1.5$, $f=0.25$ the gain is about 0.06), and disk or motion kernels have exact zeros. The inverse $Y/K = X + N/K$ explodes where $|K|$ is small. Minimizing $\mathbb{E}|X - GY|^2$ per frequency gives the Wiener filter

$$G(\omega)=\frac{K^*(\omega)\,S_x(\omega)}{|K(\omega)|^2S_x(\omega)+S_n(\omega)},$$

which inverts only where signal power beats noise power. It is the Fourier form of the Gaussian posterior mean, hence soft. The SR model $y = (k*x)\downarrow_s + n$ does both: the kernel attenuates, decimation folds what remains above $\pi/s$.

### Interpolation kernels and resizers

Nearest neighbor is a box kernel (sinc response, blocky), bilinear a triangle ($\mathrm{sinc}^2$, blurrier), bicubic Keys' cubic convolution, whose negative lobes approximate sinc better and overshoot slightly. MATLAB's `imresize` uses $a=-0.5$; PyTorch's `F.interpolate` and OpenCV's `INTER_CUBIC` use $a=-0.75$. When shrinking, `imresize` also stretches the kernel by the scale factor by default, an antialiasing prefilter; `F.interpolate` does not unless `antialias=True`, and OpenCV's cubic mode does not either. DIV2K's bicubic LR images were made with `imresize` at default settings, and the usual Set5/Set14/B100/Urban100 LR inputs are, by common practice, generated the same way, so LR made by another resizer is a different degradation and shifts the reported PSNR.

### Real cameras are not bicubic

A real image passes through the optics' point-spread function (often spatially varying), motion blur, pixel-area integration, a Bayer filter and demosaicing, shot noise (Poisson) plus read noise (roughly Gaussian), an ISP (denoising, sharpening, tone curve) that makes noise correlated and signal-dependent, JPEG, and often resizing and recompression when shared. A bicubic-trained network inverts one known, noise-free kernel. If the true kernel is wider the output is over-smoothed, if narrower over-sharpened (the IKC observation), and noise or JPEG blocks get upscaled into structured artifacts.

Aliasing is also why multi-frame SR works: perfectly anti-aliased frames would carry nothing above Nyquist, but real frames alias differently under sub-pixel shifts, so aligned frames jointly constrain the high-resolution content.

::: {.qa asked="Megvii CV algorithm campus interview, Nowcoder 2022"}
**Hand-write nearest-neighbor and bilinear image interpolation.**

Map each output pixel center back to the input: $u = (j + 0.5)\,W_{in}/W_{out} - 0.5$ (PyTorch `align_corners=False`). Nearest neighbor rounds $u$ (PyTorch's `nearest-exact`; the legacy `nearest` mode takes $\lfloor j\,W_{in}/W_{out}\rfloor$); bilinear blends neighbors separably. This matches `F.interpolate(mode="bilinear")`. When shrinking a lot, pre-filter or area-average to avoid aliasing.

```python
import numpy as np
def bilinear_resize(img, out_h, out_w):
    H, W = img.shape[:2]
    ys = np.clip((np.arange(out_h) + 0.5) * H / out_h - 0.5, 0, H - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * W / out_w - 0.5, 0, W - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    if img.ndim == 3: wy, wx = wy[..., None], wx[..., None]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy
```
:::

::: qa
**What is aliasing, why low-pass before downsampling, and what does that cost?**

Sampling copies the spectrum at multiples of the rate, so content above half the new rate shows up at a false lower frequency. After decimation by $s$, the band above $\pi/s$ folds into the kept band and cannot be separated from true content without a prior. A low-pass near $\pi/s$ removes it first. The cost is blur: real filters roll off early and sharp cutoffs ring, so resizers choose different compromises.
:::

::: {.qa asked="vivo CV algorithm written test, Nowcoder 2022"}
**What noise is in RAW images, how do you model or estimate it, and what loss would you use for denoising?**

Mainly shot noise (Poisson, variance proportional to signal) and read noise (Gaussian), plus fixed-pattern noise, hot pixels and quantization. A standard model is $y \sim \mathcal{N}(x,\ \lambda_{\text{read}} + \lambda_{\text{shot}}x)$, with both parameters set by gain (ISO); estimate them from dark frames and flat fields or from flat image patches. Denoise in RAW, where noise is still pixel-independent. Use L1 or Charbonnier, ideally computed after a simulated ISP so errors are weighted as in the final image. The Anscombe transform $2\sqrt{y+3/8}$ makes Poisson noise roughly unit-variance Gaussian.
:::

## Degradation modeling: from bicubic to real-world SR

The classical blind-SR model is

$$y=\big[(x\otimes k)\downarrow_s+n\big]_{\mathrm{JPEG}}.$$

Early approaches fed a known kernel and noise level to the network (SRMD), estimated the kernel first (KernelGAN with an internal GAN; IKC by iterative correction from artifacts), or trained on the test image itself (ZSSR). What worked best in the wild was synthesizing training pairs from a broad degradation distribution.

**BSRGAN** randomly shuffles the order of its operations: two blurs (isotropic and anisotropic Gaussian), downsampling chosen from nearest, bilinear and bicubic, Gaussian noise, JPEG at random quality, and camera-sensor noise made by running an image backward through an ISP model, adding RAW noise and running it forward.

**Real-ESRGAN** uses a second-order model, $x_{LR} = D_2(D_1(x))$, each stage with its own parameters, to imitate capture followed by editing and re-sharing. Each stage applies blur (Gaussian, generalized Gaussian, plateau or sinc kernels), a random up or down resize, Gaussian or Poisson noise, and JPEG. A final 2D sinc filter, $k(i,j)=\frac{\omega_c}{2\pi\sqrt{i^2+j^2}}J_1\big(\omega_c\sqrt{i^2+j^2}\big)$, imitates ringing and overshoot; its order relative to the last JPEG is random. Training is L1 first (Real-ESRNet), then L1 plus perceptual plus adversarial losses with a U-Net discriminator under spectral normalization.

**The domain gap** is the mismatch between synthetic and real degradations. Too mild, and real noise gets amplified; too strong, and real texture (grass, skin, film grain) is removed as noise. Real paired sets such as RealSR and DRealSR, captured by changing focal length, are physically real but small and camera-specific.

::: key
For real-world SR the training-pair generator is a model component: it decides what the network treats as signal and what as degradation.
:::

::: qa
**Compare the BSRGAN and Real-ESRGAN pipelines. Why does Real-ESRGAN add a sinc filter?**

Both use blur, resize, noise and JPEG. BSRGAN randomly shuffles their order and adds ISP-processed sensor noise; Real-ESRGAN applies the whole chain twice with independent parameters, imitating repeated processing. The sinc filter, an ideal low-pass, creates the ringing and overshoot bands near edges that over-sharpening and band-limiting produce and that blur or noise do not imitate. Without it, models tend to amplify such halos.
:::

::: qa
**A model trained on bicubic pairs looks bad on real phone photos. Why, and how do you close the gap?**

It learned to invert one clean, known kernel, while real photos have a different and often varying point-spread function, correlated ISP noise, JPEG blocks and halos; a wider true kernel gives over-smoothing, and noise becomes fake texture. Train on a broad realistic pipeline (BSRGAN- or Real-ESRGAN-style with the target camera's noise model), tune its strength on held-out real captures rather than synthetic PSNR, add real pairs if possible, and inspect failures by type.
:::

::: {.qa asked="Hikvision CV algorithm campus interview, Nowcoder 2022"}
**How would you do super-resolution at arbitrary, non-integer scales?**

The baseline is a fixed-scale model plus a bicubic resize. Meta-SR predicts upscaling filter weights from the scale factor and each output pixel's sub-pixel offset. LIIF treats the image as a continuous function: an MLP maps a nearby latent code, the relative coordinate and the output cell size to RGB, so it can be queried at any resolution, extrapolating beyond training scales with falling quality. Sample training scales to match deployment and blend neighboring codes to avoid seams.
:::

## Architectures and losses

| Model | Year | Key idea |
|---|---|---|
| SRCNN | 2014 | Three conv layers on a bicubic-upsampled input |
| VDSR | 2016 | 20 layers, global residual learning, gradient clipping |
| ESPCN | 2016 | Compute at low resolution; sub-pixel convolution (pixel shuffle) |
| EDSR | 2017 | Residual blocks without BatchNorm, residual scaling 0.1, L1 loss |
| RCAN | 2018 | Residual-in-residual with long and short skips; channel attention |
| SwinIR | 2021 | Residual Swin Transformer blocks (shifted-window attention) |
| HAT | 2022 | Channel plus window attention, overlapping cross-attention, same-task pretraining |

The common skeleton is a shallow convolution, a deep body at low resolution, a pixel-shuffle upsampler and a global skip, so the body predicts only the missing high frequencies. RCAN's channel attention is squeeze-and-excitation, $s = \sigma(W_U\,\delta(W_D\,\mathrm{GAP}(F)))$, $F' = s\odot F$. SwinIR beat CNN baselines with up to 67% fewer parameters; HAT's attribution analysis showed SwinIR still used a limited range of input pixels, and HAT added channel attention and overlapping cross-attention to use more. For general restoration, Restormer attends across channels (linear in pixel count) and NAFNet shows that a simple gated network without nonlinear activations is competitive.

**Losses.** L2 targets the posterior mean, L1 the median; Charbonnier $\sqrt{(\hat x-x)^2+\epsilon^2}$ is a smooth L1. The perceptual loss compares fixed-network features, $\frac{1}{C_jH_jW_j}\|\phi_j(\hat x)-\phi_j(x)\|_2^2$ with VGG features $\phi_j$, matching structure without pinning pixels; alone it can leave grid artifacts and color shifts. Adversarial losses push toward the image distribution. LPIPS is now a common training loss, especially for fine-tuning diffusion SR; a model trained on LPIPS should not be judged by LPIPS alone.

::: qa
**Why do SR networks such as EDSR and ESRGAN remove BatchNorm?**

EDSR argues BN removes the range flexibility of features, which a regression task that must preserve absolute intensity needs, and dropping it saved about 40% of training memory. ESRGAN adds that BN causes artifacts when train and test statistics differ and in deep GAN training. Small patch batches also give noisy statistics that couple unrelated images. Residual scaling and careful initialization keep BN-free networks stable.
:::

::: qa
**Should you train with L1 or L2 if you are evaluated on PSNR?**

Often L1, although PSNR is a function of MSE. Zhao et al. found L1-trained networks reached higher PSNR and SSIM than L2-trained ones and attributed it to optimization: the L2 networks settled in worse minima. A common intuition is that L2 gradients shrink with the error while L1 gradients keep constant size. EDSR also chose L1 after finding it converged better in its own experiments. L1 is also robust to outliers such as misaligned regions in real pairs; both losses still blur texture.
:::

::: qa
**Pixel shuffle or transposed convolution? Where do checkerboard artifacts come from?**

A transposed convolution whose kernel size is not divisible by its stride covers output pixels unevenly and prints a checkerboard. Pixel shuffle predicts $s^2C$ channels at low resolution and rearranges them, so convolutions stay cheap; its $s^2$ sub-pixel filters can still disagree at initialization, which ICNR fixes by initializing them identically (equivalent to nearest-neighbor upsampling then convolution). Resize-then-convolve avoids the artifact at higher compute.
:::

## GAN-based SR

**SRGAN** combines a residual generator with a VGG feature content loss and an adversarial term weighted $10^{-3}$; its PSNR was lower than the MSE baseline but its mean opinion scores were much higher. **ESRGAN** made four changes: RRDB blocks without BatchNorm; a relativistic average discriminator, $D_{Ra}(x_r,x_f)=\sigma\big(C(x_r)-\mathbb{E}_{x_f}[C(x_f)]\big)$, so generator gradients come from real and fake samples; a perceptual loss on VGG features before activation (post-activation features are sparse and shifted brightness); and network interpolation, $\theta=(1-\alpha)\theta_{\text{PSNR}}+\alpha\theta_{\text{GAN}}$, as a realism knob. Its generator loss is $\mathcal{L}_{\text{percep}}+\lambda\mathcal{L}^{Ra}_G+\eta\mathcal{L}_1$ with $\lambda=5\times10^{-3}$, $\eta=10^{-2}$. **Real-ESRGAN** keeps that generator and adds the synthetic pipeline and a U-Net discriminator.

GAN-SR artifacts are plausible but wrong high frequencies: texture on flat areas, wavy strokes on text, color speckle, halos. The discriminator rewards local texture statistics rather than correctness. LDL penalizes regions flagged by local statistics of the residual during training; DeSRA detects artifacts at test time by comparing local variance of GAN and MSE outputs, then fine-tunes on a few examples.

::: qa
**What did ESRGAN change relative to SRGAN, and why does each change help?**

RRDB blocks without BN give a deeper, denser generator without BN artifacts. The relativistic discriminator judges whether real images look more realistic than the average fake, giving the generator gradients from both real and generated data and sharper edges. Pre-activation VGG features give denser supervision and consistent brightness. Network interpolation between PSNR- and GAN-trained weights trades realism for fidelity at inference, and removed artifacts better than blending output images.
:::

## Diffusion-based SR

### SR3 and LDM-SR

SR3 trains $\epsilon_\theta(x_t,\tilde y,t)$ with the bicubic-upsampled input $\tilde y$ concatenated to $x_t$, using the standard DDPM loss. On 8x face SR it reached a human fool rate close to 50%, against at most 34% for GAN baselines, and it can be cascaded for generation. LDM-SR does the same in a pretrained autoencoder's latent space. Both main models were trained on bicubic degradation; LDM also reports LDM-BSR, trained on more diverse degradations.

### Adapting text-to-image priors

The Real-ISR line keeps a pretrained text-to-image latent diffusion model and trains an adapter that injects the input.

- **StableSR** freezes Stable Diffusion and trains a time-aware encoder that modulates its features. A controllable feature wrapping module with a scalar weight trades quality against fidelity, and progressive aggregation sampling fuses overlapping latent tiles at every step for arbitrary sizes.
- **DiffBIR** first removes degradation with a restoration module, then generates lost detail with IRControlNet; region-adaptive restoration guidance balances realness and fidelity.
- **PASD** adds pixel-aware cross-attention, a degradation removal module, high-level semantic conditioning and an adjustable noise schedule.
- **SeeSR** trains a degradation-aware prompt extractor that outputs tags (hard prompts) and embeddings (soft prompts), and mixes the low-resolution latent into the initial noise.
- **SUPIR** uses SDXL (2.6B parameters), a large adaptor with a ZeroSFT connector, 20 million captioned high-quality images, negative-quality prompts, and restoration-guided sampling.

### Residual shifting (ResShift)

Starting from pure noise wastes steps when the answer is close to the input. ResShift builds a chain from $x_0$ to the upsampled input $y_0$ (in VQGAN latent space). With $e_0 = y_0 - x_0$, an increasing schedule $\eta_1 \approx 0,\dots,\eta_T \approx 1$ and $\alpha_t = \eta_t - \eta_{t-1}$:

$$q(x_t\mid x_{t-1},y_0)=\mathcal{N}\big(x_t;\ x_{t-1}+\alpha_t e_0,\ \kappa^2\alpha_t I\big).$$

Independent increments sum to the marginal

$$q(x_t\mid x_0,y_0)=\mathcal{N}\big(x_t;\ x_0+\eta_t e_0,\ \kappa^2\eta_t I\big),$$

whose mean $(1-\eta_t)x_0+\eta_t y_0$ slides from HR to LR, so $x_T \approx \mathcal{N}(y_0,\kappa^2 I)$. Gaussian conditioning gives

$$q(x_{t-1}\mid x_t,x_0,y_0)=\mathcal{N}\Big(x_{t-1};\ \tfrac{\eta_{t-1}}{\eta_t}x_t+\tfrac{\alpha_t}{\eta_t}x_0,\ \kappa^2\tfrac{\eta_{t-1}}{\eta_t}\alpha_t I\Big),$$

and a network predicts $x_0$. The reported model uses $T = 15$ and $\kappa = 2$.

### One-step methods

**SinSR** distills a deterministic ResShift sampler into one step, with a consistency-preserving loss on ground truth. **OSEDiff** starts from the low-quality latent rather than noise, fine-tunes SD's encoder and UNet with LoRA, and trains with MSE and LPIPS plus variational score distillation (VSD) in latent space, reporting more than 100x speedup over StableSR. **AddSR** adapts adversarial diffusion distillation; **S3Diff** uses a degradation-guided LoRA; **PiSA-SR** trains a pixel-fidelity LoRA ($\ell_2$) and a semantic LoRA (LPIPS and classifier score distillation) with separate inference scales; **TSD-SR** uses target score distillation; **InvSR** inverts the input into an intermediate diffusion state and runs one to five steps. A common pattern since OSEDiff: start from the input, adapt a large prior with LoRA, and replace the trajectory with a distribution-matching loss.

### Hallucination, fidelity and cost

Hallucination comes from filling the null space, wrong semantics on ambiguous inputs, prompts and classifier-free guidance, and the SD autoencoder ($f=8$) failing on fine structures such as small text. Knobs: start from a noised input and lower that noise; raise conditioning strength (feature-wrapping weight, ControlNet scale, fidelity LoRA); add measurement-consistency guidance; extract correct tags; lower CFG; restore low frequencies or color from the input afterward. Cost is roughly steps times one denoiser pass, plus VAE encoding and decoding, which take a large share at 4K, especially for one-step models; large images need overlapping tiles. One-step models remove the step factor (multi-step samplers use tens to hundreds), but backbone and VAE size set the floor.

::: qa
**Why do diffusion SR models hallucinate, and what knobs reduce it?**

They sample a learned posterior, so they fill the null space with plausible content, and severe degradation makes even semantics ambiguous. Prompts and CFG push toward what the prior finds likely, and the latent autoencoder cannot represent some fine structures. Reduce it by starting from a less-noised input, strengthening conditioning, adding measurement-consistency guidance, using degradation-robust tags, lowering CFG, and correcting low frequencies from the input. Each knob moves along the perception–distortion curve; none removes it.
:::

::: qa
**Derive ResShift's forward marginal. Why does it need far fewer steps than LDM-based SR?**

Each transition adds $\alpha_t e_0$ and independent noise of variance $\kappa^2\alpha_t$. Since $\sum_{s\le t}\alpha_s=\eta_t$, the marginal is $\mathcal{N}(x_0+\eta_t e_0,\ \kappa^2\eta_t I)$, so $x_T\approx\mathcal{N}(y_0,\kappa^2I)$. The chain carries only the HR-LR residual and sampling starts at the input, not white noise, so 15 steps suffice. Too small or too large a $\kappa$ gives smooth results.
:::

::: qa
**How do one-step methods like OSEDiff work, and what do you give up?**

OSEDiff encodes the input, runs the LoRA-tuned UNet once at a fixed timestep, and decodes. Training combines MSE and LPIPS with VSD, which compares the frozen SD score with a score model fitted to the generator's outputs and pulls the output distribution toward SD's. You lose stochasticity, the option to spend more compute for quality, and some generative strength on very degraded inputs.
:::

## Training-free posterior samplers

These methods combine an unconditional diffusion prior with a known operator and need no paired training. The conditional score splits as

$$\nabla_{x_t}\log p_t(x_t\mid y)=\nabla_{x_t}\log p_t(x_t)+\nabla_{x_t}\log p_t(y\mid x_t),\qquad p_t(y\mid x_t)=\int p(y\mid x_0)\,p(x_0\mid x_t)\,dx_0 ,$$

and the second term is intractable. **DPS** replaces $p(x_0\mid x_t)$ by a point mass at its Tweedie mean (variance-preserving notation):

$$\hat x_0(x_t)=\frac{1}{\sqrt{\bar\alpha_t}}\big(x_t+(1-\bar\alpha_t)s_\theta(x_t,t)\big),\qquad \nabla_{x_t}\log p(y\mid x_t)\approx-\frac{1}{2\sigma_y^2}\nabla_{x_t}\big\|y-A(\hat x_0(x_t))\big\|^2 .$$

Each step takes an unconditional update to $x'_{t-1}$, then $x_{t-1}=x'_{t-1}-\zeta_t\nabla_{x_t}\|y-A(\hat x_0)\|^2$ with $\zeta_t=\zeta'/\|y-A(\hat x_0)\|$. It backpropagates through the denoiser, uses 1000 steps, and handles any differentiable $A$ (phase retrieval, non-uniform blur). The approximation error is a Jensen gap that shrinks as measurement noise grows.

**ΠGDM** uses $p(x_0\mid x_t)\approx\mathcal{N}(\hat x_t,r_t^2I)$, so for $y=Hx_0+z$, $p(y\mid x_t)\approx\mathcal{N}(H\hat x_t,\ r_t^2HH^\top+\sigma_y^2I)$ and

$$\nabla_{x_t}\log p_t(y\mid x_t)\approx\Big[(y-H\hat x_t)^\top\big(r_t^2HH^\top+\sigma_y^2I\big)^{-1}H\,\frac{\partial\hat x_t}{\partial x_t}\Big]^\top,$$

a vector-Jacobian product. With $\sigma_y=0$ it needs only a pseudoinverse $H^\dagger$, so it extends to non-differentiable operators such as JPEG. **DDRM** runs diffusion in the SVD basis of $H$: zero-singular-value components come from the prior, the rest are pulled toward the measurement depending on whether $\sigma_y/s_i$ is below the diffusion noise; about 20 network evaluations suffice. **DDNM** replaces the denoised estimate with $A^{+}y+(I-A^{+}A)\hat x_0(x_t)$ in noise-free problems.

| Method | Operator | Per-step cost | Consistency with $y$ |
|---|---|---|---|
| DDRM | Linear, needs SVD | Forward pass | Exact in observed components (noise-free) |
| DDNM | Linear, needs $A^{+}$ | Forward pass | Exact projection (noise-free) |
| DPS | Any differentiable | Forward + backward | Soft, step-size dependent |
| ΠGDM | Linear or pseudo-invertible | Forward + VJP | Soft, uses noise model |

All need $A$ exactly, so they suit known-operator benchmarks and scientific imaging more than blind real-world SR.

::: qa
**Derive the DPS update. What is its key approximation, and when does it break?**

Bayes splits the conditional score into the prior score plus $\nabla_{x_t}\log p_t(y\mid x_t)$, with $p_t(y\mid x_t)=\mathbb{E}_{p(x_0\mid x_t)}[p(y\mid x_0)]$. DPS moves the expectation inside, $p(y\mid x_t)\approx p(y\mid\mathbb{E}[x_0\mid x_t])$, using Tweedie's mean; Gaussian noise gives the squared-residual gradient, computed by backpropagating through the denoiser. It struggles when $p(x_0\mid x_t)$ is broad (early steps), is sensitive to step size, costs a backward pass per step, and enforces consistency only softly; for linear problems DDRM or DDNM are faster and, without noise, exactly consistent.
:::

## Video restoration and video SR

Video adds information, since neighboring frames see the scene with sub-pixel shifts, and a new failure mode: flicker, which viewers notice even when each frame looks fine.

**Alignment.** Flow-based alignment estimates $F_{t\to t-1}$ (SpyNet, RAFT) and backward-warps with bilinear sampling, $\hat h(p)=h_{t-1}\big(p+F_{t\to t-1}(p)\big)$; warping features rather than images tolerates flow errors (BasicVSR measured 0.17 dB lost with image warping). Deformable convolution predicts per-tap offsets from both frames, $y(p)=\sum_{k}w_k\,x(p+p_k+\Delta p_k)\,m_k$ ($m_k$ in DCNv2), as in TDAN and EDVR's coarse-to-fine pyramid alignment. Chan et al. showed it equals $K$ warps with separate offset fields followed by a convolution; flow alignment is the one-field special case. Offset diversity helps, but offsets can diverge in training, so BasicVSR++ predicts them as residuals on optical flow.

**Propagation.** Sliding windows (EDVR) fuse a few frames; forward recurrence (FRVSR) sees only the past. BasicVSR runs two recurrences,

$$h^b_i=F_b(x_i,x_{i+1},h^b_{i+1}),\qquad h^f_i=F_f(x_i,x_{i-1},h^f_{i-1}),\qquad y_i=U(h^f_i,h^b_i),$$

with $F=R\circ W\circ S$ (flow, warp, residual blocks). BasicVSR++ adds second-order grid propagation and flow-guided deformable alignment for 0.82 dB at similar size. Bidirectionality needs the whole clip, which costs latency. On real video, long-term propagation amplifies noise and artifacts, so RealBasicVSR cleans frames before propagation.

**Metrics.** The warping error of Lai et al. is

$$E_{\text{warp}}(V_t,V_{t+1})=\frac{1}{\sum_i M^{(i)}_t}\sum_i M^{(i)}_t\big\|V^{(i)}_t-\hat V^{(i)}_{t+1}\big\|_2^2,$$

with $\hat V_{t+1}$ the flow-warped next frame and $M_t$ a non-occlusion mask, averaged over frame pairs. TecoGAN's metrics compare the output's temporal change with the ground truth's:

$$\mathrm{tOF}=\big\|\mathrm{OF}(b_{t-1},b_t)-\mathrm{OF}(g_{t-1},g_t)\big\|_1,\qquad \mathrm{tLP}=\big\|\mathrm{LP}(b_{t-1},b_t)-\mathrm{LP}(g_{t-1},g_t)\big\|_1,$$

with $b$ ground truth, $g$ output, OF optical flow and LP the LPIPS distance.

**Diffusion VSR.**

- **Upscale-A-Video** builds on the SD x4 upscaler, with temporal layers in the U-Net and VAE decoder, a training-free flow-guided recurrent latent propagation, and text and noise-level controls.
- **MGLD-VSR** steers the latent sampling path with a motion-guided loss from input optical flow and fine-tunes a temporal decoder.
- **STAR** uses a text-to-video prior (I2VGen-XL by default; CogVideoX variants) via ControlNet, a local information enhancement module before global attention, and a dynamic frequency loss.
- **SeedVR** is a restoration diffusion transformer (MM-DiT style, about 2.5B parameters) with shifted-window attention on $64\times64$ latent windows, variable-size boundary windows, a causal video VAE, and mixed image-video training; it handles arbitrary length and resolution but samples in many steps.
- **SeedVR2** is one-step via diffusion adversarial post-training: initialized from SeedVR, progressively distilled, then trained adversarially against real data, with adaptive window attention, RpGAN with approximated R1 and R2 penalties, and a discriminator feature-matching loss in place of LPIPS; a 7B model plus a 3B model distilled from it.
- **DOVE** fine-tunes CogVideoX into a one-step VSR model with two-stage latent-then-pixel training.

::: qa
**Compare flow-based and deformable alignment for video SR.**

Flow-based alignment warps with one motion vector per pixel: interpretable and able to use a strong pretrained flow network, but brittle at occlusions and large non-rigid motion. Deformable alignment learns $K$ offsets per location, equivalent to $K$ warps followed by a convolution, so it can hedge across several candidate correspondences. Its offsets can diverge and destabilize training. BasicVSR++ combines them by predicting deformable offsets as residuals on optical flow.
:::

::: qa
**How do you measure temporal consistency in video SR, and why is warping error alone misleading?**

Warping error warps frame $t+1$ onto frame $t$ with optical flow and averages the masked squared difference. tOF and tLP compare the output's frame-to-frame motion and perceptual change with the ground truth's. Warping error rewards outputs that barely change, so blurry or static results score well, and it depends on the flow estimator. Pair a temporal metric with a spatial quality metric and watch the clips.
:::

::: qa
**Why does running an image diffusion SR model frame by frame flicker, and how do video diffusion SR models fix it?**

Each frame gets its own noise sample and its own null-space content, so textures change even on static scenes, and tile seams move. Fixes: start from the noised input and correlate noise across frames; add temporal layers or 3D attention to the denoiser (Upscale-A-Video, SeedVR); propagate or fuse latents along optical flow; add motion-guided losses during sampling (MGLD-VSR); add temporal layers to the VAE decoder; or start from a pretrained video prior (STAR, DOVE).
:::

::: qa
**How does SeedVR2 achieve one-step video restoration, and how does that differ from distillation?**

It initializes from multi-step SeedVR, runs progressive distillation, then fine-tunes the whole network adversarially against real high-quality video. Pure distillation matches a teacher's samples, so it needs expensive teacher outputs and is capped by the teacher; adversarial training against real data needs no teacher samples and is not bounded by the teacher by construction. For high resolution, SeedVR2 adapts its attention window to the output size, stabilizes training with RpGAN and approximated R1 and R2 penalties, and replaces LPIPS with discriminator feature matching.
:::

## Face and blind restoration

Faces have a strong class prior, and viewers notice identity errors. **GFPGAN** drives a pretrained StyleGAN2 from a degradation-removal network; channel-split spatial feature transforms keep some prior channels unchanged (realness) and modulate the rest with input features (fidelity), with facial-component and identity losses. **CodeFormer** learns a VQGAN codebook of high-quality faces, predicts the code sequence with a transformer, and mixes encoder features back with weight $w\in[0,1]$ (smaller $w$ for quality, larger for fidelity). Pipelines usually restore aligned face crops, paste them back, and upscale the background with a general model. Watch for identity drift and "beautification".

::: qa
**Compare GFPGAN and CodeFormer. Why does a discrete codebook help under severe degradation?**

GFPGAN uses a continuous StyleGAN2 prior with feature-transform injection and identity losses. CodeFormer turns restoration into predicting discrete codes from a learned codebook. Choosing among finitely many high-quality codes is far more constrained than regressing a continuous latent, so heavily degraded inputs still map to coherent faces, and the transformer uses global context where local evidence is gone.
:::

## Rapid-fire review

::: qa
**What is global residual learning?**

The network predicts only the difference from an upsampled input, so low frequencies pass through and capacity goes to missing detail.
:::

::: qa
**What does "blind" mean in blind SR?**

The degradation is unknown at test time; the model estimates it or is trained to be robust across many degradations.
:::

::: qa
**Why does Real-ESRGAN use pixel-unshuffle for its x2 and x1 models?**

It moves resolution into channels so the heavy RRDB body runs at lower resolution, saving memory and compute.
:::

::: qa
**Why a U-Net discriminator with spectral normalization in Real-ESRGAN?**

It gives per-pixel realness feedback across a wide degradation space; spectral normalization stabilizes it and reduced over-sharpening.
:::

::: qa
**Why can latent diffusion SR shift colors, and what fixes it?**

Low frequencies drift through the VAE and sampler. StableSR and others match channel statistics to the input or swap in the input's low-frequency wavelet bands.
:::

::: qa
**Why does SeeSR put the LR latent into the initial noise?**

SD's training schedule never fully turns the image into noise, but sampling starts from pure noise; starting from a noised LR latent closes that gap and reduces spurious random detail.
:::

::: qa
**Why do tiled diffusion pipelines blend overlapping tiles at every step?**

Independent tiles drift apart and leave seams; fusing overlaps at each step (StableSR) keeps neighbors consistent.
:::

::: qa
**What is DDNM's range–null-space decomposition?**

$x=A^{+}Ax+(I-A^{+}A)x$: the first part comes from noise-free measurements ($A^{+}y$), the second from the diffusion prior.
:::

::: qa
**What is HAT's same-task pretraining?**

Pretraining the SR model on ImageNet for the same SR task before fine-tuning on the target dataset, rather than multi-task pretraining.
:::

::: {.qa asked="Horizon Robotics CV campus interview, Nowcoder 2022"}
**How would you combine super-resolution and semantic segmentation?**

Segmentation can tell SR which texture to generate where (SFT-GAN modulates features with segmentation probability maps; SeeSR uses tags). Conversely, SR can help segment small or low-resolution objects.
:::

::: qa
**Why is a diffusion SR model's PSNR usually below SwinIR's?**

SwinIR regresses toward the posterior mean, the MSE minimizer; an exact posterior sampler has twice that MSE (about 3 dB lower PSNR), and a learned sampler usually loses more, in exchange for realism.
:::

## References

1. Blau and Michaeli, "The Perception-Distortion Tradeoff," 2018. <https://arxiv.org/abs/1711.06077>
2. Freirich, Michaeli, Meir, "A Theory of the Distortion-Perception Tradeoff in Wasserstein Space," 2021. <https://arxiv.org/abs/2107.02555>
3. Ohayon, Michaeli, Elad, "Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration," 2024. <https://arxiv.org/abs/2410.00418>
4. Zhao et al., "Loss Functions for Neural Networks for Image Processing," 2015. <https://arxiv.org/abs/1511.08861>
5. Dong et al., "Image Super-Resolution Using Deep Convolutional Networks" (SRCNN), 2014. <https://arxiv.org/abs/1501.00092>
6. Kim, Lee, Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks" (VDSR), 2015. <https://arxiv.org/abs/1511.04587>
7. Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" (ESPCN), 2016. <https://arxiv.org/abs/1609.05158>
8. Odena, Dumoulin, Olah, "Deconvolution and Checkerboard Artifacts," Distill, 2016. <https://distill.pub/2016/deconv-checkerboard/>
9. Aitken et al., "Checkerboard artifact free sub-pixel convolution," 2017. <https://arxiv.org/abs/1707.02937>
10. Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution" (EDSR), 2017. <https://arxiv.org/abs/1707.02921>
11. Zhang et al., "Image Super-Resolution Using Very Deep Residual Channel Attention Networks" (RCAN), 2018. <https://arxiv.org/abs/1807.02758>
12. Liang et al., "SwinIR: Image Restoration Using Swin Transformer," 2021. <https://arxiv.org/abs/2108.10257>
13. Chen et al., "Activating More Pixels in Image Super-Resolution Transformer" (HAT), 2022. <https://arxiv.org/abs/2205.04437>
14. Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration," 2021. <https://arxiv.org/abs/2111.09881>
15. Chen et al., "Simple Baselines for Image Restoration" (NAFNet), 2022. <https://arxiv.org/abs/2204.04676>
16. Johnson, Alahi, Fei-Fei, "Perceptual Losses for Real-Time Style Transfer and Super-Resolution," 2016. <https://arxiv.org/abs/1603.08155>
17. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (LPIPS), 2018. <https://arxiv.org/abs/1801.03924>
18. Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" (SRGAN), 2016. <https://arxiv.org/abs/1609.04802>
19. Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks," 2018. <https://arxiv.org/abs/1809.00219>
20. Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data," 2021. <https://arxiv.org/abs/2107.10833>
21. Zhang et al., "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution" (BSRGAN), 2021. <https://arxiv.org/abs/2103.14006>
22. Zhang, Zuo, Zhang, "Learning a Single Convolutional Super-Resolution Network for Multiple Degradations" (SRMD), 2017. <https://arxiv.org/abs/1712.06116>
23. Gu et al., "Blind Super-Resolution With Iterative Kernel Correction" (IKC), 2019. <https://arxiv.org/abs/1904.03377>
24. Bell-Kligler, Shocher, Irani, "Blind Super-Resolution Kernel Estimation using an Internal-GAN" (KernelGAN), 2019. <https://arxiv.org/abs/1909.06581>
25. Shocher, Cohen, Irani, "'Zero-Shot' Super-Resolution using Deep Internal Learning," 2017. <https://arxiv.org/abs/1712.06087>
26. Cai et al., "Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model" (RealSR), 2019. <https://arxiv.org/abs/1904.00523>
27. Wei et al., "Component Divide-and-Conquer for Real-World Image Super-Resolution" (DRealSR), 2020. <https://arxiv.org/abs/2008.01928>
28. Brooks et al., "Unprocessing Images for Learned Raw Denoising," 2018. <https://arxiv.org/abs/1811.11127>
29. Hu et al., "Meta-SR: A Magnification-Arbitrary Network for Super-Resolution," 2019. <https://arxiv.org/abs/1903.00875>
30. Chen, Liu, Wang, "Learning Continuous Image Representation with Local Implicit Image Function" (LIIF), 2020. <https://arxiv.org/abs/2012.09161>
31. Wang et al., "Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform" (SFT-GAN), 2018. <https://arxiv.org/abs/1804.02815>
32. Liang, Zeng, Zhang, "Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution" (LDL), 2022. <https://arxiv.org/abs/2203.09195>
33. Xie et al., "DeSRA: Detect and Delete the Artifacts of GAN-based Real-World Super-Resolution Models," 2023. <https://arxiv.org/abs/2307.02457>
34. Saharia et al., "Image Super-Resolution via Iterative Refinement" (SR3), 2021. <https://arxiv.org/abs/2104.07636>
35. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," 2021. <https://arxiv.org/abs/2112.10752>
36. Wang et al., "Exploiting Diffusion Prior for Real-World Image Super-Resolution" (StableSR), 2023. <https://arxiv.org/abs/2305.07015>
37. Lin et al., "DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior," 2023. <https://arxiv.org/abs/2308.15070>
38. Yang et al., "Pixel-Aware Stable Diffusion for Realistic Image Super-resolution and Personalized Stylization" (PASD), 2023. <https://arxiv.org/abs/2308.14469>
39. Wu et al., "SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution," 2023. <https://arxiv.org/abs/2311.16518>
40. Yu et al., "Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild" (SUPIR), 2024. <https://arxiv.org/abs/2401.13627>
41. Yue, Wang, Loy, "ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting," 2023. <https://arxiv.org/abs/2307.12348>
42. Wang et al., "SinSR: Diffusion-Based Image Super-Resolution in a Single Step," 2023. <https://arxiv.org/abs/2311.14760>
43. Wu et al., "One-Step Effective Diffusion Network for Real-World Image Super-Resolution" (OSEDiff), 2024. <https://arxiv.org/abs/2406.08177>
44. Xie et al., "AddSR: Accelerating Diffusion-based Blind Super-Resolution with Adversarial Diffusion Distillation," 2024. <https://arxiv.org/abs/2404.01717>
45. Zhang et al., "Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors" (S3Diff), 2024. <https://arxiv.org/abs/2409.17058>
46. Sun et al., "Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach" (PiSA-SR), 2024. <https://arxiv.org/abs/2412.03017>
47. Dong et al., "TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution," 2024. <https://arxiv.org/abs/2411.18263>
48. Yue, Liao, Loy, "Arbitrary-steps Image Super-resolution via Diffusion Inversion" (InvSR), 2024. <https://arxiv.org/abs/2412.09013>
49. Kawar et al., "Denoising Diffusion Restoration Models" (DDRM), 2022. <https://arxiv.org/abs/2201.11793>
50. Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems" (DPS), 2022. <https://arxiv.org/abs/2209.14687>
51. Song, Vahdat, Mardani, Kautz, "Pseudoinverse-Guided Diffusion Models for Inverse Problems" (ΠGDM), ICLR 2023. <https://iclr.cc/virtual/2023/poster/11030>
52. Wang, Yu, Zhang, "Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model" (DDNM), 2022. <https://arxiv.org/abs/2212.00490>
53. Dai et al., "Deformable Convolutional Networks," 2017. <https://arxiv.org/abs/1703.06211>
54. Zhu et al., "Deformable ConvNets v2: More Deformable, Better Results," 2018. <https://arxiv.org/abs/1811.11168>
55. Ranjan and Black, "Optical Flow Estimation using a Spatial Pyramid Network" (SpyNet), 2016. <https://arxiv.org/abs/1611.00850>
56. Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow," 2020. <https://arxiv.org/abs/2003.12039>
57. Tian et al., "TDAN: Temporally Deformable Alignment Network for Video Super-Resolution," 2018. <https://arxiv.org/abs/1812.02898>
58. Wang et al., "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks," 2019. <https://arxiv.org/abs/1905.02716>
59. Chan et al., "Understanding Deformable Alignment in Video Super-Resolution," 2020. <https://arxiv.org/abs/2009.07265>
60. Sajjadi, Vemulapalli, Brown, "Frame-Recurrent Video Super-Resolution" (FRVSR), 2018. <https://arxiv.org/abs/1801.04590>
61. Chan et al., "BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond," 2020. <https://arxiv.org/abs/2012.02181>
62. Chan et al., "BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment," 2021. <https://arxiv.org/abs/2104.13371>
63. Chan et al., "Investigating Tradeoffs in Real-World Video Super-Resolution" (RealBasicVSR), 2021. <https://arxiv.org/abs/2111.12704>
64. Lai et al., "Learning Blind Video Temporal Consistency," 2018. <https://arxiv.org/abs/1808.00449>
65. Chu et al., "Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation" (TecoGAN), 2018. <https://arxiv.org/abs/1811.09393>
66. Zhou et al., "Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution," 2023. <https://arxiv.org/abs/2312.06640>
67. Yang et al., "Motion-Guided Latent Diffusion for Temporally Consistent Real-world Video Super-resolution" (MGLD-VSR), 2023. <https://arxiv.org/abs/2312.00853>
68. Xie et al., "STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution," 2025. <https://arxiv.org/abs/2501.02976>
69. Wang et al., "SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration," 2025. <https://arxiv.org/abs/2501.01320>
70. Wang et al., "SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training," 2025. <https://arxiv.org/abs/2506.05301>
71. Lin et al., "Diffusion Adversarial Post-Training for One-Step Video Generation" (APT), 2025. <https://arxiv.org/abs/2501.08316>
72. Chen et al., "DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution," 2025. <https://arxiv.org/abs/2505.16239>
73. Wang et al., "Towards Real-World Blind Face Restoration with Generative Facial Prior" (GFPGAN), 2021. <https://arxiv.org/abs/2101.04061>
74. Zhou et al., "Towards Robust Blind Face Restoration with Codebook Lookup Transformer" (CodeFormer), 2022. <https://arxiv.org/abs/2206.11253>
75. Wang, Chen, Hoi, "Deep Learning for Image Super-resolution: A Survey," 2019. <https://arxiv.org/abs/1902.06068>
76. Liu et al., "Blind Image Super-Resolution: A Survey and Beyond," 2021. <https://arxiv.org/abs/2107.03055>
77. Li et al., "Diffusion Models for Image Restoration and Enhancement: A Comprehensive Survey," 2023. <https://arxiv.org/abs/2308.09388>
78. Interview reports behind the `asked=` tags (Megvii, vivo, Hikvision, Horizon Robotics), Nowcoder, January 2022. <https://www.nowcoder.com/discuss/353159089435910144>
