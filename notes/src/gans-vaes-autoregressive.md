---
title: "GANs, VAEs and Autoregressive Models"
subtitle: "Adversarial, variational, quantized, autoregressive and flow-based generators: their objectives, failure modes, and where each one still appears in modern image generation."
order: 5
updated: 2026-11-08
---

Before diffusion, image generation was split among GANs (adversarial training), VAEs (a variational bound), autoregressive models (an exact chain-rule likelihood) and normalizing flows (exact likelihood through invertible maps). None of them went away. Adversarial losses sharpen super-resolution, latent autoencoders and few-step samplers. VAEs and VQ tokenizers define the latent spaces that diffusion and transformers work in, and next-scale prediction competes with diffusion on ImageNet. Interviewers use this material to test whether you can derive an objective, explain why it fails in a specific way, and connect a 2014 idea to a 2025 system. Diffusion math is in [diffusion models](diffusion-models.html), distillation in [flow matching](flow-matching.html), and metrics in [evaluation metrics](evaluation-metrics.html).

## The landscape: which divergence are you minimizing?

Autoregressive models and flows give an exact $\log p_\theta(x)$, VAEs and diffusion optimize a bound, and GANs define only a sampler. Maximum likelihood is forward-KL minimization, because $\mathrm{KL}(p_{\text{data}}\|p_\theta) = \mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}] - \mathbb{E}_{p_{\text{data}}}[\log p_\theta]$ and the first term is constant. Forward KL blows up wherever $p_\theta\approx 0$ but data exists, so it is *mode-covering*. Reverse KL $\mathrm{KL}(p_\theta\|p_{\text{data}})$ penalizes mass where there is no data, so it is *mode-seeking*.

::: key
Likelihood training (VAE, autoregressive, flow) tends to cover all modes but can put mass on implausible in-between images, which shows up as blur. Adversarial training behaves like reverse KL: samples look sharp, but modes get dropped.
:::

## GANs

### The minimax game and the optimal discriminator

Goodfellow et al. (2014) train $G$ and $D: x\mapsto(0,1)$ on

$$
\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))].
$$

For fixed $G$ with density $p_g$, maximize $\int [p_{\text{data}}\log D + p_g\log(1-D)]\,dx$ pointwise. Since $a\log y + b\log(1-y)$ peaks at $y = a/(a+b)$, $D^* = p_{\text{data}}/(p_{\text{data}}+p_g)$. Substituting, with $m = \tfrac12(p_{\text{data}}+p_g)$:

$$
C(G) = \mathbb{E}_{p_{\text{data}}}\Big[\log\tfrac{p_{\text{data}}}{2m}\Big] + \mathbb{E}_{p_g}\Big[\log\tfrac{p_g}{2m}\Big] = -\log 4 + 2\,\mathrm{JSD}(p_{\text{data}}\|p_g),
$$

minimized at $p_g = p_{\text{data}}$, where $D^*\equiv 1/2$. This assumes an infinitely flexible discriminator trained to optimality before each generator step, which real training does not provide.

::: {.qa asked="360 computer vision interview, Nowcoder 2019"}
**Write down the GAN loss function and explain what it optimizes.**

$\min_G\max_D \mathbb{E}_{p_{\text{data}}}[\log D(x)] + \mathbb{E}_{p_z}[\log(1-D(G(z)))]$. For fixed $G$, $D$ is a binary classifier trained with cross-entropy, and its optimum is $D^* = p_{\text{data}}/(p_{\text{data}}+p_g)$. Substituting gives $-\log 4 + 2\,\mathrm{JSD}(p_{\text{data}}\|p_g)$, so an ideal generator step lowers the JS divergence, with optimum $p_g = p_{\text{data}}$ and $D = 1/2$. In practice $G$ uses the non-saturating loss and $D$ is not optimal, so the JSD reading is only a guide.
:::

### The non-saturating generator loss

Write $D = \sigma(\ell)$. The minimax generator term $\log(1-\sigma(\ell))$ has $\partial/\partial\ell = -\sigma(\ell)$, which vanishes when $D$ confidently rejects fakes, as it does early in training. Minimizing $-\log D(G(z))$ instead gives $\partial/\partial\ell = -(1-\sigma(\ell))\approx -1$ in that regime, with the same fixed point. The divergence changes, though. Arjovsky & Bottou (2017, Thm. 2.5) show that with $D^*$ held fixed,

$$
\mathbb{E}_z\big[-\nabla_\theta \log D^*(G_\theta(z))\big] = \nabla_\theta\big[\mathrm{KL}(p_g\|p_{\text{data}}) - 2\,\mathrm{JSD}(p_g\|p_{\text{data}})\big].
$$

Sketch: $-\log D^* = \log(p_g/p_{\text{data}}) - \log(1-D^*)$, and with $D^*$ fixed, $\mathbb{E}_{p_g}[\log(1-D^*)]$ has the same $\theta$-gradient as $2\,\mathrm{JSD}$. The KL term is the reverse KL, which penalizes unrealistic samples heavily and dropped modes very little; the JSD term is symmetric and does not undo that bias.

::: qa
**Why use $-\log D(G(z))$ instead of $\log(1-D(G(z)))$ for the generator, and what does it cost?**

The minimax loss saturates: its gradient with respect to the logit is $-\sigma(\ell)$, which goes to zero when fakes are confidently rejected, as they are early in training. The non-saturating loss has gradient $-(1-\sigma(\ell))$, large in that regime, with the same fixed point. The cost: with an optimal $D$ the generator follows $\nabla[\mathrm{KL}(p_g\|p_{\text{data}}) - 2\mathrm{JSD}]$, which is mode-seeking and helps explain both sharp samples and mode dropping.
:::

### Instability and mode collapse

There are three distinct causes. **Disjoint supports:** images lie near low-dimensional manifolds, so early on $p_g$ and $p_{\text{data}}$ barely overlap. A near-perfect $D$ then exists, JSD sits at $\log 2$, and gradients vanish. **Game dynamics:** in Mescheder et al.'s Dirac-GAN (data $\delta_0$, generator $\delta_\theta$, $D(x) = \psi x$, loss $f(\psi\theta)+f(0)$ with $f(t) = -\log(1+e^{-t})$), the gradient field at the equilibrium has Jacobian $\begin{pmatrix}0 & -f'(0)\\ f'(0) & 0\end{pmatrix}$ with eigenvalues $\pm i f'(0)$. The parameters rotate around the equilibrium: simultaneous updates spiral outward and alternating updates keep circling. **Mode collapse:** against a frozen $D$, the generator's best response maps every $z$ to $\arg\max_x D(x)$. Because $D$ scores samples one at a time, it cannot see that a batch is all the same image, so $G$ and $D$ chase each other from mode to mode. Collapse shows up as low recall at high precision. Remedies: batch statistics (minibatch discrimination, minibatch-std), better-conditioned dynamics (TTUR, unrolled GANs, zero-centered penalties), and augmentation (DiffAugment, ADA) for small data.

::: {.qa asked="Alibaba Cloud vision, Nowcoder AIGC roundup 2025"}
**Why are GANs hard to train?**

First, the objective is a game, not a minimization: near equilibrium the gradient field rotates, so gradient steps oscillate or diverge. Second, real and fake distributions barely overlap, so $D$ separates them almost perfectly and the generator gets tiny or noisy gradients. Third, the generator's best response is to collapse onto whatever $D$ currently likes, which causes mode collapse and cycling. Remedies: gradient control on $D$ (WGAN-GP, spectral norm, R1/R2), non-saturating or relativistic losses, and batch-level features or augmentation.
:::

### Wasserstein GAN and the gradient penalty

Arjovsky et al. (2017) use $W_1(p,q) = \inf_{\gamma\in\Pi(p,q)}\mathbb{E}_\gamma\|x-y\|$. For a unit segment at $x=0$ and the same segment shifted to $x=\theta$, $W_1 = |\theta|$, while $\mathrm{JSD} = \log 2$ and $\mathrm{KL}=\infty$ for every $\theta\ne0$. Only $W_1$ tells the generator which way to move. By Kantorovich-Rubinstein duality,

$$
W_1(p_{\text{data}}, p_g) = \sup_{\|f\|_L\le 1}\mathbb{E}_{p_{\text{data}}}[f(x)] - \mathbb{E}_{p_g}[f(x)],
$$

so a real-valued critic maximizes the gap and $G$ minimizes $-\mathbb{E}_z f(G(z))$. Weight clipping to $[-0.01, 0.01]$ enforced the constraint crudely. WGAN-GP (Gulrajani et al., 2017) uses the fact that the optimal critic has unit gradient norm between coupled points:

$$
L_{\text{critic}} = \mathbb{E}_{p_g}[f(\tilde x)] - \mathbb{E}_{p_{\text{data}}}[f(x)] + \lambda\,\mathbb{E}_{\hat x}\big[(\|\nabla_{\hat x}f(\hat x)\|_2 - 1)^2\big], \quad \hat x = \epsilon x + (1-\epsilon)\tilde x,
$$

with $\epsilon\sim U[0,1]$, $\lambda = 10$, 5 critic steps per generator step, and Adam with $\beta_1 = 0$, $\beta_2 = 0.9$. Mescheder et al. (2018) show that WGAN and WGAN-GP with finitely many critic steps are not always locally convergent.

::: {.qa asked="Douyin and 360 vision interviews, Nowcoder 2019"}
**What problem does WGAN solve, and how do you implement WGAN-GP?**

With disjoint supports, the original loss sees a constant JS divergence ($\log 2$), so the generator gets no direction. $W_1$ changes continuously as $p_g$ moves, even without overlap. In practice: remove the sigmoid, train the critic on `f(fake).mean() - f(real).mean()` and the generator on `-f(fake).mean()`. For WGAN-GP, build per-example interpolates $\hat x$, compute $\nabla_{\hat x}f$ with `create_graph=True`, and add $10(\|\nabla\|-1)^2$. Interpolates are used because enforcing Lipschitz everywhere is intractable. Batch norm is avoided in the critic because it couples samples and breaks the per-sample penalty.
:::

### Zero-centered penalties: R1 and R2

Mescheder et al. (2018) prove local convergence for zero-centered gradient penalties:

$$
R_1 = \frac{\gamma}{2}\,\mathbb{E}_{p_{\text{data}}}\|\nabla_x D(x)\|^2, \qquad R_2 = \frac{\gamma}{2}\,\mathbb{E}_{p_g}\|\nabla_x D(x)\|^2.
$$

At equilibrium the ideal $D$ is flat, so zero gradient agrees with the equilibrium. In the Dirac-GAN, R1 adds a $-\gamma\psi$ damping term that turns the rotation into a spiral toward the equilibrium. StyleGAN2 applies R1 lazily, every 16 minibatches. R3GAN (Huang et al., 2024) combines a relativistic pairing loss with both penalties. On StackedMNIST, R1 alone diverged, while R1 + R2 was stable and covered all 1000 modes.

::: qa
**Why is R1 centered at zero while the WGAN-GP penalty is centered at one?**

They come from different theories. WGAN-GP approximates the 1-Lipschitz constraint of the Kantorovich dual, whose optimal critic has unit-norm gradients between coupled samples. R1 comes from a local-convergence analysis of the standard GAN game. When $p_g = p_{\text{data}}$ the discriminator should be flat, so pulling $\|\nabla_x D\|$ to zero keeps the equilibrium fixed and damps the rotation. A one-centered penalty keeps a slope even when the distributions match, so the generator keeps being pushed where it should stop.
:::

### Spectral normalization, hinge and relativistic losses

**Spectral normalization** (Miyato et al., 2018) sets $\bar W = W/\sigma(W)$, with $\sigma$ from one power-iteration step per training step. With 1-Lipschitz activations, $\mathrm{Lip}(D)\le\prod_l\sigma(\bar W_l) = 1$. It needs no extra backward pass, unlike a gradient penalty, and is used in SAGAN, BigGAN and Real-ESRGAN. **Hinge loss** (Lim & Ye, 2017):

$$
L_D = \mathbb{E}[\max(0, 1 - D(x))] + \mathbb{E}[\max(0, 1 + D(G(z)))], \qquad L_G = -\mathbb{E}[D(G(z))].
$$

As in an SVM, $D$ stops pushing on samples it already classifies correctly by a margin. It is used in BigGAN, StyleGAN-T and ADD. **Relativistic losses** (Jolicoeur-Martineau, 2018) have $D$ estimate whether real data is more realistic than fake: $L_D = -\mathbb{E}\log\sigma(C(x_r) - C(x_f))$ and $L_G = -\mathbb{E}\log\sigma(C(x_f) - C(x_r))$. The average variant (RaGAN) compares each sample with the batch mean of the opposite class. Because only differences matter, generator gradients also flow through real samples, and the loss does not saturate on an absolute scale.

::: {.qa asked="Alibaba Taotian AI-application intern, Nowcoder 2026"}
**What is ESRGAN, and how did it change SRGAN's losses?**

SRGAN trains a super-resolution network with a VGG-feature content loss plus a $10^{-3}$-weighted adversarial loss. ESRGAN makes three changes. (1) A relativistic average discriminator predicts whether a real image is more realistic than the average fake, and the generator loss uses both real and fake terms, giving sharper textures. (2) The perceptual loss uses VGG features *before* activation, which are less sparse and keep brightness information. (3) The total loss is $L_{\text{percep}} + 5\times10^{-3}L_G^{Ra} + 10^{-2}L_1$. The generator also uses RRDB blocks without batch norm.
:::

### StyleGAN in brief

An 8-layer MLP maps $z$ to $w$ (512-d), so the learned $\mathcal{W}$ need not be Gaussian-shaped and can be less entangled. A learned constant input is styled per layer by an affine projection of $w$. StyleGAN1's AdaIN caused "droplet" artifacts, because the generator hid signal strength in local spikes. StyleGAN2 uses modulation and demodulation instead, $w'_{ijk} = s_iw_{ijk}$ and $w''_{ijk} = w'_{ijk}/\sqrt{\sum_{i,k}w'^2_{ijk}+\epsilon}$, which normalizes expected rather than actual statistics. Noise inputs add stochastic detail. The path length regularizer $\mathbb{E}_{w,y\sim\mathcal{N}(0,I)}(\|J_w^\top y\|_2 - a)^2$, with $a$ a running average, makes a fixed step in $\mathcal{W}$ change the image by a fixed amount.

::: {.qa asked="TP-LINK image algorithm interview, Nowcoder 2022"}
**Walk through how GAN generators evolved from the 2014 GAN to StyleGAN2.**

The 2014 GAN used MLPs. DCGAN made convolutional GANs train reliably. WGAN-GP and spectral norm stabilized the discriminator. ProGAN grew resolution progressively and added minibatch-std. SAGAN and BigGAN added self-attention, hinge loss and huge class-conditional batches. StyleGAN introduced the mapping network, a constant input, per-layer AdaIN styles and noise, giving scale-specific control. StyleGAN2 replaced AdaIN with weight demodulation (no droplets), dropped progressive growing for skip and residual designs, and added lazy R1 and path-length regularization.
:::

### Discriminators on pretrained features

Projected GAN (Sauer et al., 2021) passes images through a frozen EfficientNet-Lite1, mixes channels and scales with fixed random convolutions, and trains four small discriminators on the multi-scale features. It matched prior best FIDs up to 40 times faster. Vision-aided GAN (Kumari et al., 2022) ensembles off-the-shelf backbones. StyleGAN-T uses a frozen DINO ViT-S/16 with five heads and a per-token hinge loss, and ADD uses frozen DINOv2 ViT-S features with hinge loss and R1. The heads only have to separate real from fake in a semantic space, which is easier. The catch: FID is also computed in an ImageNet feature space, and matching ImageNet statistics can lower FID without improving images (Kynkäänniemi et al., 2023).

### Where adversarial losses live today

The perception-distortion trade-off (Blau & Michaeli, 2018) shows that past some point, low MSE and realistic outputs conflict, so [super-resolution](restoration-super-resolution.html) adds a discriminator. Autoencoders and tokenizers (VQGAN, SD VAEs, MAGVIT-v2, DC-AE) add a GAN loss, often with a patch discriminator, to L1 + LPIPS. Few-step distillation (ADD, LADD, DMD2) adds adversarial terms (see [flow matching](flow-matching.html)). Standalone GANs remain strong on narrow domains and as fast upsamplers (GigaGAN).

::: {.qa asked="SenseTime multimodal intern interview, Nowcoder 2024"}
**Why would you still use StyleGAN, or any GAN, when diffusion models exist?**

One forward pass means millisecond latency, which matters for real-time avatars, on-device filters and interactive editing. StyleGAN's $\mathcal{W}$ space is smooth enough for easy inversion and attribute editing on narrow domains such as faces, where its quality is still competitive. Adversarial losses also recover sharp detail in autoencoder decoders, super-resolution and distilled samplers. The trade-off is weaker mode coverage and harder scaling to diverse text-conditioned data.
:::

## VAEs

### The ELBO, derived two ways

The model is $p_\theta(x,z) = p(z)p_\theta(x|z)$ with $p(z) = \mathcal{N}(0,I)$, and the encoder is $q_\phi(z|x)$ (Kingma & Welling, 2013). **Jensen:**

$$
\log p_\theta(x) = \log\mathbb{E}_{q_\phi}\Big[\frac{p_\theta(x,z)}{q_\phi(z|x)}\Big] \ge \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x)\|p(z)) =: \mathcal{L}(x).
$$

**KL decomposition:** expanding $p(z|x) = p(x,z)/p(x)$,

$$
\mathrm{KL}(q_\phi(z|x)\|p_\theta(z|x)) = \mathbb{E}_{q_\phi}[\log q_\phi(z|x) - \log p_\theta(x,z)] + \log p_\theta(x) \;\Longrightarrow\; \log p_\theta(x) = \mathcal{L}(x) + \mathrm{KL}(q_\phi\|p_\theta(z|x)).
$$

The second route gives the exact gap: maximizing $\mathcal{L}$ over $\phi$ minimizes the *reverse* KL to the true posterior, so VAE posteriors tend to be too narrow. With a fixed-variance Gaussian decoder, $-\log p(x|z) = \|x-\mu_\theta(z)\|^2/2\sigma^2 + \text{const}$, so $\sigma^2$ balances reconstruction against KL.

::: {.qa asked="Google MLE screen, PracHub question bank 2026"}
**Explain VAEs: the ELBO, the KL term and the reparameterization trick.**

A VAE is a latent-variable model trained with an amortized posterior $q_\phi(z|x)$. Since $\log p(x)$ is intractable, we maximize $\mathcal{L} = \mathbb{E}_q[\log p_\theta(x|z)] - \mathrm{KL}(q\|p(z))$, which follows from Jensen or from $\log p(x) = \mathcal{L} + \mathrm{KL}(q\|p(z|x))$. The first term is reconstruction. The KL keeps posteriors near the prior so that prior samples decode well. Gradients flow through the sampling step via $z = \mu_\phi + \sigma_\phi\odot\epsilon$. For Gaussians the KL is closed-form, so only reconstruction needs a Monte Carlo sample.
:::

### Reparameterization and the Gaussian KL

The score-function estimator $\mathbb{E}_q[f(z)\nabla_\phi\log q_\phi(z|x)]$ is unbiased but high-variance. Writing $z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon$ with $\epsilon\sim\mathcal{N}(0,I)$ gives the pathwise gradient $\mathbb{E}_\epsilon[\nabla_z f(z)\,\partial z/\partial\phi]$, which uses $\nabla f$ and usually has much lower variance. For $q = \mathcal{N}(\mu,\sigma^2)$ and $p = \mathcal{N}(0,1)$:

$$
\begin{aligned}
\mathrm{KL}(q\|p) &= \Big(-\tfrac12\log(2\pi\sigma^2) - \tfrac12\Big) - \Big(-\tfrac12\log 2\pi - \tfrac12\mathbb{E}_q[z^2]\Big) \\
&= \tfrac12\big(\mu^2 + \sigma^2 - 1 - \log\sigma^2\big),
\end{aligned}
$$

using $\mathbb{E}_q[z^2] = \mu^2+\sigma^2$. For a diagonal posterior, sum over dimensions. Each term is zero only at $\mu=0$, $\sigma=1$.

::: {.qa asked="iFlytek 2021 and Baidu multimodal 2025, Nowcoder"}
**Sampling is not differentiable. How does a VAE backpropagate through it?**

It moves the randomness out of the parameter path: sample $\epsilon\sim\mathcal{N}(0,I)$ and set $z = \mu_\phi + \sigma_\phi\odot\epsilon$. Given $\epsilon$, $z$ is differentiable in $\phi$ ($\partial z/\partial\mu = 1$, $\partial z/\partial\sigma = \epsilon$), so ordinary backprop works. This needs a distribution that is a differentiable transform of fixed noise. Categorical latents need Gumbel-softmax or straight-through, and REINFORCE is unbiased but much noisier.
:::

::: {.qa asked="Meitu AIGC vision algorithm interview, Nowcoder 2024"}
**What is the difference between an autoencoder and a VAE?**

An autoencoder learns a deterministic $x\to z\to\hat x$ for reconstruction only. Its latent space has no density, so decoding a random $z$ usually gives garbage. A VAE's encoder outputs a distribution, and the ELBO adds a KL term tying posteriors to a prior, so $z\sim p(z)$ decodes to plausible images and training bounds $\log p(x)$. The costs are blur and possible posterior collapse. Latent-diffusion autoencoders sit in between: their KL weight is about $10^{-6}$, so they are mostly regularized autoencoders whose real prior is the diffusion model trained afterwards.
:::

### Why VAE samples are blurry

With a fixed-variance Gaussian decoder the reconstruction term is MSE, so the optimal decoder outputs $\mathbb{E}[x|z]$. The KL term forces posteriors of different images to overlap, so each $z$ fits many slightly different images, and their average has smoothed edges and texture. Three more effects add to this: maximum likelihood is mode-covering, the aggregate posterior $\mathbb{E}_x q(z|x)$ rarely matches $\mathcal{N}(0,I)$ (prior samples land in "holes"), and pixel likelihood undervalues high-frequency detail. Fixes: perceptual and adversarial losses, hierarchical VAEs (NVAE), and a strong learned prior over latents, which is what latent diffusion is.

::: {.qa asked="Samsung Xi'an multimodal interview, Nowcoder 2024"}
**Why are images generated by a VAE blurry?**

Mainly because of the Gaussian (MSE) reconstruction likelihood. Under MSE the best prediction is the conditional mean $\mathbb{E}[x|z]$, and because the KL term makes posteriors overlap, each $z$ is ambiguous, so the decoder averages several plausible images. Maximum likelihood is also mode-covering, the prior and aggregate posterior mismatch sends samples to poorly trained regions, and pixel losses undervalue high-frequency detail. GANs avoid this because the discriminator directly penalizes averaged-looking outputs.
:::

### Posterior collapse and $\beta$-VAE

Posterior collapse means $q(z|x)\approx p(z)$ for every $x$, so the decoder ignores $z$. If the decoder can model $p(x)$ alone (autoregressive decoders), storing nothing in $z$ is cheapest, since the KL is paid on every example. Lucas et al. (2019) show collapse even in linear VAEs, tied to observation noise. $\beta$-VAE (Higgins et al., 2017) optimizes $\mathbb{E}_q[\log p(x|z)] - \beta\,\mathrm{KL}$. With $\beta>1$ the bottleneck tightens, which encourages disentanglement on simple data at a reconstruction cost and still gives a looser lower bound. With $\beta<1$ it is no longer a bound, which is the regime LDM's tiny KL weight is in.

::: qa
**What is posterior collapse, why does it happen, and how do you prevent it?**

The approximate posterior equals the prior for every input, so $z$ carries no information. It happens when the decoder is expressive enough that information in $z$ costs more KL than it saves in reconstruction, or when the KL term wins early in optimization. Diagnose it with per-dimension KL near zero and few active units. Fixes: KL annealing from 0 (Bowman et al., 2015), free bits, which stop penalizing KL below a floor per group (Kingma et al., 2016), weaker or limited-context decoders, $\beta<1$, or feeding $z$ into every decoder layer.
:::

## Discrete latents

### VQ-VAE

VQ-VAE (van den Oord et al., 2017) quantizes each encoder vector to its nearest code, $z_q = e_k$ with $k = \arg\min_j\|z_e(x) - e_j\|_2$, and trains

$$
L = -\log p(x|z_q) + \|\operatorname{sg}[z_e] - e\|_2^2 + \beta\|z_e - \operatorname{sg}[e]\|_2^2, \qquad \beta = 0.25.
$$

Reconstruction trains the decoder and, through the straight-through estimator $z_q^{\text{ST}} = z_e + \operatorname{sg}[z_q - z_e]$, the encoder. The codebook term moves selected codes toward encoder outputs, like online k-means (EMA updates are an alternative). Commitment keeps the encoder near its codes. Generation needs a second stage: an autoregressive prior over code indices (PixelCNN originally, transformers later).

::: {.qa asked="iFlytek vision 2024, Taotian multimodal 2026, Nowcoder"}
**How is VQ-VAE/VQGAN trained, and how do gradients pass through the quantizer?**

Stage 1 trains encoder, codebook and decoder with reconstruction + codebook loss $\|\operatorname{sg}[z_e]-e\|^2$ + commitment $\beta\|z_e-\operatorname{sg}[e]\|^2$. Argmin has zero gradient almost everywhere, so the straight-through estimator sends $z_q$ forward and copies $\partial L/\partial z_q$ onto $z_e$ backward: `z_q = z_e + (z_q - z_e).detach()`. Codes are trained only by the codebook loss (or EMA), and commitment keeps $z_e$ close to $z_q$, which limits the estimator's bias. Stage 2 freezes the autoencoder and trains a transformer on code indices with cross-entropy.
:::

### Codebook collapse

Codes that start far from the encoder's outputs, or that the encoder drifts away from, stop being selected, so the effective vocabulary shrinks. Measure code usage or the perplexity of the average assignment. Beyond restarts (as in Jukebox) and EMA, ViT-VQGAN projects to a low-dimensional lookup space (for example 768-d to 32-d) and $\ell_2$-normalizes codes, which kept an 8192-entry codebook well used.

::: {.qa asked="Alibaba Amap recsys interview (RQ-VAE), Nowcoder 2026"}
**What is codebook collapse in VQ/RQ-VAE, and how do you prevent it?**

Only a small subset of codes is ever selected. Unselected codes get no gradient, so once outside the encoder's output region they stay dead. In residual quantization it also appears as many inputs sharing one code tuple. Prevention: k-means initialization, EMA updates, re-seeding rarely used codes from current encoder outputs, low-dimensional $\ell_2$-normalized lookup, an entropy term rewarding uniform average usage, or codebook-free quantizers (FSQ, LFQ). Track usage at every level.
:::

### VQGAN

VQGAN (Esser et al., 2021) replaces the L2 reconstruction with a perceptual (LPIPS) loss and adds a patch discriminator:

$$
\mathcal{Q}^* = \arg\min_{E,G,\mathcal{Z}}\max_D\,\mathbb{E}_x\big[\mathcal{L}_{VQ} + \lambda\mathcal{L}_{GAN}\big], \qquad \lambda = \frac{\|\nabla_{G_L}\mathcal{L}_{rec}\|}{\|\nabla_{G_L}\mathcal{L}_{GAN}\| + \delta},
$$

where $G_L$ is the last decoder layer and $\delta = 10^{-6}$. The adaptive weight matches the adversarial gradient's size to the reconstruction gradient's, so neither dominates. The perceptual and adversarial terms allow a coarse grid: with $f=16$ and $|\mathcal{Z}| = 1024$, a $256^2$ image becomes 256 tokens, short enough for a GPT-style transformer, which samples larger images with a sliding attention window.

::: {.qa asked="Megvii vision algorithm interview, Nowcoder 2024"}
**How many losses does VQGAN have, and what does each do?**

The autoencoder stage has four loss terms and one adaptive weight. Reconstruction (L1 plus LPIPS) preserves structure. The codebook loss moves codes toward encoder outputs. The commitment loss keeps the encoder near its codes. A patch-discriminator adversarial loss makes local texture look real and removes blur. The adaptive weight $\lambda$, a ratio of gradient norms at the last decoder layer, balances adversarial against reconstruction. The GAN term usually starts after a warm-up. Stage 2 adds the transformer's cross-entropy over indices.
:::

### FSQ and lookup-free quantization

FSQ (Mentzer et al., 2023) projects to very few dimensions ($d<10$), bounds them, and rounds: $\hat z = \operatorname{round}(\lfloor L/2\rfloor\tanh(z))$ for odd $L$ (even $L$ needs a half-step offset). The implied codebook is the product grid, so levels $[8,5,5,5]$ give 1000 codes and $[8,8,8,5,5,5]$ give 64,000. Rounding uses straight-through, $x + \operatorname{sg}(\operatorname{round}(x)-x)$. There is no codebook or commitment loss, and the paper reports near-complete code usage without auxiliary losses. LFQ (MAGVIT-v2, Yu et al., 2023) binarizes each dimension, $q(z_i) = \operatorname{sign}(z_i)$, with index $\sum_i 2^{i-1}\mathbb{1}[z_i>0]$ and vocabulary $2^{18}\approx 262\text{k}$. An entropy penalty $\mathbb{E}[H(q(z))] - H[\mathbb{E}\,q(z)]$ rewards confident per-sample assignments and uniform average usage, and reconstruction and generation kept improving with vocabulary size. RQ-VAE instead quantizes, subtracts, and quantizes the residual several times per position.

## Autoencoders for latent diffusion

### Downsampling factor and channel count

An autoencoder maps $H\times W\times3$ to $\tfrac Hf\times\tfrac Wf\times c$, a compression of $3f^2/c$. A DiT with patch size $p$ sees $(H/fp)^2$ tokens, so $f$ is the main compute lever.

| Autoencoder | $f$ | $c$ | $3f^2/c$ | Latent scaling |
|---|---|---|---|---|
| SD 1.x/2.x | 8 | 4 | 48 | 0.18215 |
| SDXL | 8 | 4 | 48 | 0.13025 |
| SD3 / FLUX | 8 | 16 | 12 | FLUX: scale 0.3611, shift 0.1159 |
| DC-AE f32c32 / f64c128 | 32 / 64 | 32 / 128 | 96 | n/a |

LDM (Rombach et al., 2022) compared $f\in\{1,\dots,32\}$. At $f\le2$ training was slow, spending compute on imperceptible detail. At $f=32$ fidelity stopped improving early. The paper found $f$ between 4 and 16 balanced efficiency and quality, with $f = 4$ and 8 best in its sampling-speed comparison. The autoencoders combine L1/L2, LPIPS, a patch GAN, and either a KL penalty weighted about $10^{-6}$ or a VQ layer. Latents are rescaled to roughly unit variance to match the noise schedule.

::: {.qa asked="Alibaba Taotian algorithm interview, Nowcoder 2025"}
**How does the SD 1.5 VAE differ from the FLUX VAE? How many channels does FLUX's latent have?**

Both are KL-regularized convolutional autoencoders with $8\times$ downsampling, trained with reconstruction, perceptual and adversarial losses. SD 1.5's latent has 4 channels (compression 48), which loses small text, small faces and fine texture. FLUX's has 16 (compression 12), so reconstruction is much better. FLUX uses a scale and a shift factor, and its transformer packs $2\times2$ latent patches into 64-dim tokens. SD3 made the same change and found that models on the 16-channel latent need more capacity to match the FID of 4- or 8-channel ones, but scale better.
:::

### The reconstruction versus generation tension

More channels improve reconstruction. In SD3's comparison, going from $c=4$ to $c=16$ took rFID from 2.41 to 1.06. But the diffusion model then has more information per token to model. Yao et al. (2025) call this an optimization dilemma: $f16$ latents with 32 or 64 channels reconstruct better but generate worse, unless the latent is aligned to DINOv2 features. Their VA-VAE does this and reaches FID 1.35 on ImageNet 256. DC-AE (Chen et al., 2025) instead pushes $f$ to 32-128. *Residual autoencoding* adds parameter-free space-to-channel shortcuts. *Decoupled high-resolution adaptation* trains at low resolution, tunes only middle layers at high resolution, and applies the GAN loss only to the decoder head at the end. On ImageNet 512 it reports 19.1x faster inference and 17.9x faster training than SD-VAE-f8, with better FID.

::: qa
**Why not keep adding channels until reconstruction is perfect?**

The generator has to model the latent distribution too. A high-capacity latent with weak regularization carries more bits per token and is less smooth, so the diffusion model needs more parameters and training to reach the same FID, and its errors in fine latent detail decode into artifacts. SD3 found that small models do worse on a 16-channel latent and the gap closes only as depth grows. VA-VAE found that 32 or 64 channels generate worse unless the latent is regularized toward semantic features. The target is a latent that is both reconstructable and easy to model.
:::

## Autoregressive and masked image generation

### PixelCNN and token transformers

The chain rule gives an exact likelihood, $\log p(x) = \sum_i\log p(x_i|x_{<i})$, here over pixels in raster order and RGB channels. PixelCNN (van den Oord et al., 2016) uses masked convolutions: mask A in the first layer lets a channel see only earlier pixels and the already-predicted channels of the current pixel (G sees R, not G), and mask B in later layers also connects each channel to itself, which is safe because the features it reads are already masked. Gated PixelCNN fixes the receptive-field blind spot with vertical and horizontal stacks. PixelCNN++ replaces the 256-way softmax with a discretized logistic mixture. Training is parallel, but sampling takes $H\cdot W\cdot3$ sequential steps, and samples were locally plausible but globally incoherent. Two-stage models compress first: VQGAN + GPT, DALL-E (a discrete VAE and a 12B transformer), Parti (ViT-VQGAN tokens, up to 20B parameters) and LlamaGen. Their weak points are arbitrary raster order, error accumulation, sequential decoding, and a tokenizer quality ceiling.

### MaskGIT and Muse

MaskGIT (Chang et al., 2022) trains a bidirectional transformer BERT-style. It masks $\lceil\gamma(r)N\rceil$ tokens with $r\sim U[0,1)$ and $\gamma(r) = \cos(\pi r/2)$, and applies cross-entropy on masked positions only. Decoding starts fully masked. At step $t$ it predicts all masked tokens in parallel, samples them, and re-masks the $\lceil\gamma(t/T)N\rceil$ least confident. It uses $T = 8$ steps at $256^2$, and the paper reports decoding up to 64x faster than autoregression. Muse applies the same procedure to text-to-image, conditioned on T5-XXL embeddings, with a 256-pixel base model and a 512-pixel super-resolution transformer.

::: qa
**How does MaskGIT decode, and what does it give up relative to autoregressive decoding?**

It starts with every token masked. At each of $T\approx8$ to 12 steps it predicts every masked position in parallel, keeps the most confident samples, and re-masks the rest on a cosine schedule that commits few tokens early. That replaces $N = 256$ to 1024 sequential steps, and bidirectional context suits 2D data. The cost: tokens committed in the same step are sampled independently given the context, so too few steps produce inconsistencies. There is no exact likelihood, and quality depends on schedule and temperature heuristics.
:::

### VAR: next-scale prediction

VAR (Tian et al., 2024) predicts whole scales instead of single tokens. A multi-scale residual VQ-VAE with one shared codebook ($V = 4096$) quantizes the feature map at the coarsest scale, upsamples and subtracts that reconstruction, then quantizes the residual at the next scale. For $256^2$ images the token maps have sides 1, 2, 3, 4, 5, 6, 8, 10, 13 and 16: ten scales, 680 tokens in total. Generation factorizes as

$$
p(r_1,\ldots,r_K) = \prod_{k=1}^K p(r_k\mid r_1,\ldots,r_{k-1}),
$$

with every token of $r_k$ predicted in parallel under a block-causal mask. On ImageNet 256 it reports FID 1.73 and IS 350.2 (from 18.65 and 80.4 for its AR baseline), about 20x faster inference than that baseline, and power-law scaling.

::: qa
**What is next-scale prediction, and why does it beat raster-order token prediction?**

VAR predicts the next-resolution token map conditioned on all coarser maps, and the residual tokenizer makes each scale encode what the coarser ones missed. All tokens in a scale come out in parallel, so ten steps replace 256 at $256^2$. The order matches image structure (layout first, then detail), keeps 2D neighbourhoods intact, and removes raster AR's left-to-right bias. Tokens within a scale are conditionally independent, but the coarse prefix constrains them strongly.
:::

### MAR: continuous tokens with a diffusion loss

MAR (Li et al., 2024) drops vector quantization. A transformer over continuous tokens (from an LDM KL-16 tokenizer) outputs a vector $z$ for each position it predicts. A small MLP (3 residual blocks, width 1024, about 21M parameters) models that token's distribution with

$$
\mathcal{L}(z,x) = \mathbb{E}_{\epsilon,t}\big[\|\epsilon - \epsilon_\theta(x_t\mid t,z)\|^2\big], \qquad x_t = \sqrt{\bar\alpha_t}\,x + \sqrt{1-\bar\alpha_t}\,\epsilon,
$$

and sampling runs a short reverse diffusion per token, with a temperature on the noise. The backbone uses masked autoregression: random order, bidirectional attention, several tokens per step. MAR-H with CFG reaches FID 1.55 on ImageNet 256, and MAR-L generates in under 0.3 s per image.

::: qa
**For continuous tokens, why use a diffusion loss instead of regressing the next token with MSE?**

MSE predicts the conditional mean. When several next tokens are plausible, the mean matches none of them, and the averages compound as they are fed back, which is VAE blur one token at a time. Autoregression needs to *sample* from $p(x_i\mid\text{context})$. Discrete tokens get that from a softmax, at the cost of quantization error. A small diffusion head gives a multimodal, sampleable distribution over continuous values, with a temperature knob.
:::

## Normalizing flows

A flow is a bijection $z = f(x)$ with $z\sim p_Z$. Change of variables gives

$$
\log p_X(x) = \log p_Z(f(x)) + \log\Big|\det\frac{\partial f(x)}{\partial x}\Big|,
$$

and log-determinants add across composed layers. Affine coupling (RealNVP) keeps half the variables, $y_a = x_a$, and transforms the rest, $y_b = x_b\odot\exp(s(x_a)) + t(x_a)$. The Jacobian is triangular, so $\log|\det| = \sum s(x_a)$, and the inverse is just as cheap. Glow adds invertible $1\times1$ convolutions with log-determinant $h\cdot w\cdot\log|\det W|$, which is cheap with an LU parameterization. Flows give exact likelihood, exact encoding and one-pass sampling. Their limits: every layer preserves dimension, so there is no bottleneck and memory is large; mapping a Gaussian onto thin or disconnected data needs badly conditioned Jacobians; and invertibility restricts the architecture. Samples lagged for years. TarFlow (Zhai et al., 2024) narrowed the gap with transformer autoregressive flows, noise augmentation and post-hoc denoising. [Flow matching](flow-matching.html) is a different method: it regresses a velocity field.

## Comparing the families

| Family | Density | Sampling | Mode coverage | Sample quality |
|---|---|---|---|---|
| GAN | none | 1 pass | weak | sharp |
| VAE | lower bound | 1 pass | good | blurry without perceptual/GAN losses |
| Flow | exact | 1 pass (coupling) | good | historically weaker |
| AR over tokens | exact (over tokens) | $N$ sequential steps | good | strong with a good tokenizer |
| Masked / next-scale | none (masked); exact over token maps (VAR) | about 8-64 steps | good | strong |
| Diffusion | bound | tens of steps unless distilled | good | strong |

Xiao et al. (2022) name the **generative learning trilemma**: high sample quality, mode coverage and fast sampling are hard to get together. GANs give quality and speed, VAEs and flows give speed and coverage, and diffusion gives quality and coverage. Their denoising diffusion GANs model each large denoising step with a conditional GAN and report about 2000x faster sampling than the original diffusion model on CIFAR-10. Later distillation and consistency models attack the same trilemma from the diffusion side.

**Likelihood and sample quality can disagree** (Theis et al., 2016). If $p = 0.01\,q + 0.99\,r$ for a good model $q$ and junk $r$, then $\log p(x)\ge\log q(x) - \log 100$. The log-likelihood drops by at most 4.61 nats per image, which is negligible for a $256^2$ RGB image whose total is hundreds of thousands of nats, yet 99% of samples are junk. In the other direction, a mode-dropping model can produce excellent samples while assigning near-zero likelihood to the modes it dropped. Likelihood is dominated by imperceptible detail, which is why bits/dim and FID often rank models differently.

::: {.qa asked="ByteDance vision algorithm interview, Nowcoder 2023"}
**Compare GANs and VAEs.**

A VAE maximizes a lower bound on $\log p(x)$ (reconstruction minus KL to a prior). A GAN plays a min-max game that corresponds roughly to a JS or reverse-KL objective and never computes a density. VAEs train stably, cover modes, and provide an encoder and a likelihood bound, but give blurry samples because of the MSE likelihood and mode covering. GANs give sharp one-pass samples but are unstable, prone to mode collapse, and have no encoder or likelihood. VQGAN and SD autoencoders combine both.
:::

::: {.qa asked="Meitu AIGC vision algorithm interview, Nowcoder 2024"}
**How do GANs differ from diffusion models, and why do diffusion models usually give better results?**

A GAN learns a one-step map from noise through an adversarial game. A diffusion model learns to reverse a fixed noising process with a regression (denoising) loss and samples iteratively. The diffusion objective is stable, non-adversarial and tied to a likelihood bound, so it covers modes and scales predictably with data and model size. Iterative sampling splits generation into many easy steps, and guidance gives a quality-diversity knob. GANs remain far faster at inference and are still used as losses. See [diffusion models](diffusion-models.html).
:::

## Rapid-fire review

::: qa
**What does the optimal discriminator output at the GAN equilibrium?**

$1/2$ everywhere, because $D^* = p_{\text{data}}/(p_{\text{data}}+p_g)$ and $p_g = p_{\text{data}}$. The value of the game is $-\log 4$.
:::

::: qa
**What is TTUR?**

Separate learning rates for discriminator and generator. Heusel et al. (2017) prove convergence to a local Nash equilibrium under stochastic-approximation assumptions.
:::

::: qa
**What does the truncation trick trade?**

$w' = \bar w + \psi(w-\bar w)$ with $\psi<1$ moves samples toward the dense centre of $\mathcal{W}$, raising fidelity and lowering diversity. Guidance in diffusion is a similar knob.
:::

::: qa
**How do DiffAugment and ADA help small-data GANs?**

They apply the same differentiable augmentations to real and fake images before $D$, so $D$ cannot memorize the data and the augmentations do not leak into samples. ADA adapts the augmentation probability from an overfitting signal.
:::

::: {.qa asked="ByteDance vision algorithm interview, Nowcoder 2023"}
**Why push the VAE's latent distribution toward $\mathcal{N}(0,I)$?**

At generation time we sample $z\sim p(z)$ with no encoder, so training codes must fill the region the prior samples from. The KL keeps posteriors overlapping and near the prior, so random $z$ decode to valid images.
:::

::: qa
**Why does the VQ-VAE loss have no KL term?**

With a deterministic one-hot posterior and a uniform prior over $K$ codes, the KL is the constant $\log K$, so it drops out of the gradient.
:::

::: {.qa asked="iFlytek interview, Nowcoder 2023"}
**What advantages does VQ-VAE have over a Gaussian VAE?**

Discrete codes plug directly into transformers trained with cross-entropy, there is little posterior collapse, and a learned prior over codes replaces the weak $\mathcal{N}(0,I)$, so samples are sharper. The costs are codebook collapse and straight-through bias.
:::

::: {.qa asked="ByteDance AIGC algorithm intern interview, Nowcoder 2026"}
**What role does the VAE play in Stable Diffusion?**

It compresses $512\times512\times3$ images to $64\times64\times4$ latents, so diffusion runs in a space 48 times smaller. Its GAN-trained decoder maps denoised latents back to pixels and supplies much of the fine texture.
:::

::: {.qa asked="JD machine learning algorithm interview, Nowcoder 2024"}
**What is the Stable Diffusion VAE scaling factor, and why does it exist?**

0.18215 for SD 1.x/2.x (0.13025 for SDXL). Latents are multiplied by it before diffusion and divided by it before decoding. It was chosen to give the latents roughly unit variance, which means the raw latents' standard deviation was estimated at about $1/0.18215\approx5.5$. The noise schedule assumes unit-variance data, so the wrong factor changes the effective SNR at every timestep.
:::

::: qa
**What is the perception-distortion trade-off?**

Blau & Michaeli (2018) prove that beyond some point, lower distortion (e.g., MSE) and outputs indistinguishable from real images conflict. This is why super-resolution systems add adversarial losses and accept lower PSNR.
:::

## References

1. Goodfellow et al., "Generative Adversarial Networks", 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
2. Arjovsky & Bottou, "Towards Principled Methods for Training Generative Adversarial Networks", 2017. [arXiv:1701.04862](https://arxiv.org/abs/1701.04862)
3. Arjovsky, Chintala & Bottou, "Wasserstein GAN", 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
4. Gulrajani et al., "Improved Training of Wasserstein GANs", 2017. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
5. Mescheder, Geiger & Nowozin, "Which Training Methods for GANs do actually Converge?", 2018. [arXiv:1801.04406](https://arxiv.org/abs/1801.04406)
6. Miyato et al., "Spectral Normalization for Generative Adversarial Networks", 2018. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
7. Lim & Ye, "Geometric GAN", 2017. [arXiv:1705.02894](https://arxiv.org/abs/1705.02894)
8. Jolicoeur-Martineau, "The relativistic discriminator: a key element missing from standard GAN", 2018. [arXiv:1807.00734](https://arxiv.org/abs/1807.00734)
9. Huang et al., "The GAN is dead; long live the GAN! A Modern GAN Baseline", NeurIPS 2024. [arXiv:2501.05441](https://arxiv.org/abs/2501.05441)
10. Salimans et al., "Improved Techniques for Training GANs", 2016. [arXiv:1606.03498](https://arxiv.org/abs/1606.03498)
11. Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", 2017. [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)
12. Karras et al., "Progressive Growing of GANs for Improved Quality, Stability, and Variation", 2017. [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
13. Brock, Donahue & Simonyan, "Large Scale GAN Training for High Fidelity Natural Image Synthesis", 2018. [arXiv:1809.11096](https://arxiv.org/abs/1809.11096)
14. Karras, Laine & Aila, "A Style-Based Generator Architecture for Generative Adversarial Networks", 2018. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
15. Karras et al., "Analyzing and Improving the Image Quality of StyleGAN", 2019. [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)
16. Karras et al., "Training Generative Adversarial Networks with Limited Data", 2020. [arXiv:2006.06676](https://arxiv.org/abs/2006.06676)
17. Zhao et al., "Differentiable Augmentation for Data-Efficient GAN Training", 2020. [arXiv:2006.10738](https://arxiv.org/abs/2006.10738)
18. Sauer et al., "Projected GANs Converge Faster", 2021. [arXiv:2111.01007](https://arxiv.org/abs/2111.01007)
19. Kumari et al., "Ensembling Off-the-shelf Models for GAN Training", 2021. [arXiv:2112.09130](https://arxiv.org/abs/2112.09130)
20. Sauer et al., "StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis", 2023. [arXiv:2301.09515](https://arxiv.org/abs/2301.09515)
21. Sauer et al., "Adversarial Diffusion Distillation", 2023. [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)
22. Kynkäänniemi et al., "The Role of ImageNet Classes in Fréchet Inception Distance", 2022. [arXiv:2203.06026](https://arxiv.org/abs/2203.06026)
23. Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network", 2016. [arXiv:1609.04802](https://arxiv.org/abs/1609.04802)
24. Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks", 2018. [arXiv:1809.00219](https://arxiv.org/abs/1809.00219)
25. Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data", 2021. [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)
26. Blau & Michaeli, "The Perception-Distortion Tradeoff", 2017. [arXiv:1711.06077](https://arxiv.org/abs/1711.06077)
27. Kang et al., "Scaling up GANs for Text-to-Image Synthesis" (GigaGAN), 2023. [arXiv:2303.05511](https://arxiv.org/abs/2303.05511)
28. Kingma & Welling, "Auto-Encoding Variational Bayes", 2013. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
29. Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", ICLR 2017. [OpenReview](https://openreview.net/forum?id=Sy2fzU9gl)
30. Bowman et al., "Generating Sentences from a Continuous Space", 2015. [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)
31. Kingma et al., "Improving Variational Inference with Inverse Autoregressive Flow", 2016. [arXiv:1606.04934](https://arxiv.org/abs/1606.04934)
32. Lucas et al., "Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse", 2019. [arXiv:1911.02469](https://arxiv.org/abs/1911.02469)
33. Jang, Gu & Poole, "Categorical Reparameterization with Gumbel-Softmax", 2016. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
34. Vahdat & Kautz, "NVAE: A Deep Hierarchical Variational Autoencoder", 2020. [arXiv:2007.03898](https://arxiv.org/abs/2007.03898)
35. van den Oord, Vinyals & Kavukcuoglu, "Neural Discrete Representation Learning" (VQ-VAE), 2017. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
36. Esser, Rombach & Ommer, "Taming Transformers for High-Resolution Image Synthesis" (VQGAN), 2020. [arXiv:2012.09841](https://arxiv.org/abs/2012.09841)
37. Yu et al., "Vector-quantized Image Modeling with Improved VQGAN", 2021. [arXiv:2110.04627](https://arxiv.org/abs/2110.04627)
38. Lee et al., "Autoregressive Image Generation using Residual Quantization" (RQ-VAE), 2022. [arXiv:2203.01941](https://arxiv.org/abs/2203.01941)
39. Dhariwal et al., "Jukebox: A Generative Model for Music", 2020. [arXiv:2005.00341](https://arxiv.org/abs/2005.00341)
40. Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", 2023. [arXiv:2309.15505](https://arxiv.org/abs/2309.15505)
41. Yu et al., "Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation" (MAGVIT-v2), 2023. [arXiv:2310.05737](https://arxiv.org/abs/2310.05737)
42. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", 2021. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
43. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3), 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
44. Yao, Yang & Wang, "Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models", 2025. [arXiv:2501.01423](https://arxiv.org/abs/2501.01423)
45. Chen et al., "Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models", 2024. [arXiv:2410.10733](https://arxiv.org/abs/2410.10733)
46. van den Oord, Kalchbrenner & Kavukcuoglu, "Pixel Recurrent Neural Networks", 2016. [arXiv:1601.06759](https://arxiv.org/abs/1601.06759)
47. van den Oord et al., "Conditional Image Generation with PixelCNN Decoders", 2016. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328)
48. Salimans et al., "PixelCNN++", 2017. [arXiv:1701.05517](https://arxiv.org/abs/1701.05517)
49. Ramesh et al., "Zero-Shot Text-to-Image Generation" (DALL-E), 2021. [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)
50. Yu et al., "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation" (Parti), 2022. [arXiv:2206.10789](https://arxiv.org/abs/2206.10789)
51. Sun et al., "Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation", 2024. [arXiv:2406.06525](https://arxiv.org/abs/2406.06525)
52. Chang et al., "MaskGIT: Masked Generative Image Transformer", 2022. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200)
53. Chang et al., "Muse: Text-To-Image Generation via Masked Generative Transformers", 2023. [arXiv:2301.00704](https://arxiv.org/abs/2301.00704)
54. Tian et al., "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction", 2024. [arXiv:2404.02905](https://arxiv.org/abs/2404.02905)
55. Li et al., "Autoregressive Image Generation without Vector Quantization", 2024. [arXiv:2406.11838](https://arxiv.org/abs/2406.11838)
56. Dinh, Sohl-Dickstein & Bengio, "Density estimation using Real NVP", 2016. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
57. Kingma & Dhariwal, "Glow: Generative Flow with Invertible 1x1 Convolutions", 2018. [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)
58. Zhai et al., "Normalizing Flows are Capable Generative Models" (TarFlow), 2024. [arXiv:2412.06329](https://arxiv.org/abs/2412.06329)
59. Xiao, Kreis & Vahdat, "Tackling the Generative Learning Trilemma with Denoising Diffusion GANs", 2021. [arXiv:2112.07804](https://arxiv.org/abs/2112.07804)
60. Theis, van den Oord & Bethge, "A note on the evaluation of generative models", 2015. [arXiv:1511.01844](https://arxiv.org/abs/1511.01844)
