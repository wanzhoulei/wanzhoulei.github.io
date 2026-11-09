---
title: "Diffusion Models"
subtitle: "DDPM, score matching, the SDE view, parameterizations, noise schedules, loss weighting, samplers, guidance and latent diffusion, derived in one notation."
order: 1
updated: 2026-11-08
---

A diffusion model learns to reverse a fixed process that slowly corrupts data with Gaussian noise. Training is a plain regression problem at random noise levels, and sampling calls the same network many times, removing a little noise each time. Most current image and video generators are diffusion models or close relatives, and flow matching reuses most of the same ideas. Interviewers like the topic because short derivations separate memorized slogans ("predict the noise") from real understanding: why the loss is a mean-squared error, what the network actually estimates, why samplers can skip steps, and why guidance trades diversity for fidelity.

**Notation.** Data $x_0 \sim p_{\text{data}}$, noise $\epsilon \sim \mathcal{N}(0, I)$. DDPM uses a variance schedule $\beta_1, \dots, \beta_T$ and $\bar{\alpha}_t = \prod_{s \le t} (1 - \beta_s)$. For the general view I write the noisy sample as $x_t = \alpha_t x_0 + \sigma_t \epsilon$, so $\alpha_t$ always means the signal scale (for DDPM, $\alpha_t = \sqrt{\bar{\alpha}_t}$ and $\sigma_t = \sqrt{1 - \bar{\alpha}_t}$; I avoid DDPM's own symbol $\alpha_t = 1 - \beta_t$). $\mathrm{SNR}(t) = \alpha_t^2 / \sigma_t^2$ and $\lambda_t = \log \mathrm{SNR}(t)$. The EDM sections use the variance-exploding form $x = x_0 + \sigma \epsilon$, which is the VP sample divided by $\alpha_t$, with $\sigma = \sigma_t / \alpha_t$. The standard deviation of a reverse sampling step is written $\varsigma_t$ (DDPM and DDIM call it $\sigma_t$, which would clash with the noise scale).

## The forward process

The forward process is a fixed Markov chain that adds a small amount of Gaussian noise at each step and shrinks the signal so the variance stays bounded:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\big).
$$

Nothing is learned here. DDPM uses $T = 1000$ and $\beta_t$ rising linearly from $10^{-4}$ to $0.02$.

**Closed form.** Each forward step draws its own fresh noise. Written as a sampling rule,

$$
x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t, \qquad \epsilon_1, \dots, \epsilon_T \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, I), \ \text{independent of } x_0 .
$$

Claim: for every $t$ there is a single standard Gaussian $\bar{\epsilon}_t \sim \mathcal{N}(0, I)$ such that

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \bar{\epsilon}_t .
$$

Here $\bar{\epsilon}_t$ is not any one step's noise. It is all the noise added in steps $1, \dots, t$, combined and rescaled to unit variance, so it is a fixed linear combination of $\epsilon_1, \dots, \epsilon_t$. Proof by induction on $t$:

- *Base case $t = 1$.* $\bar{\alpha}_1 = 1 - \beta_1$, so $x_1 = \sqrt{\bar{\alpha}_1}\, x_0 + \sqrt{1-\bar{\alpha}_1}\, \epsilon_1$. Take $\bar{\epsilon}_1 = \epsilon_1$.
- *Inductive step.* Assume the claim for $t-1$, so $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1-\bar{\alpha}_{t-1}}\, \bar{\epsilon}_{t-1}$. Since $\bar{\epsilon}_{t-1}$ is built only from $\epsilon_1, \dots, \epsilon_{t-1}$, it is independent of the new noise $\epsilon_t$. Substitute into the one-step rule:

$$
\begin{aligned}
x_t &= \sqrt{1-\beta_t}\,\Big(\sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1-\bar{\alpha}_{t-1}}\, \bar{\epsilon}_{t-1}\Big) + \sqrt{\beta_t}\, \epsilon_t \\
&= \underbrace{\sqrt{(1-\beta_t)\,\bar{\alpha}_{t-1}}}_{=\sqrt{\bar{\alpha}_t}\ \text{(definition of } \bar{\alpha}_t)}\, x_0 \;+\; \underbrace{\sqrt{1-\beta_t}\sqrt{1-\bar{\alpha}_{t-1}}\; \bar{\epsilon}_{t-1} + \sqrt{\beta_t}\; \epsilon_t}_{\text{noise part}} .
\end{aligned}
$$

  The noise part is a sum of two independent zero-mean Gaussians, so it is a zero-mean Gaussian whose covariance is the sum of the two covariances:

$$
\begin{aligned}
(1-\beta_t)(1-\bar{\alpha}_{t-1}) + \beta_t
&= 1 - \bar{\alpha}_{t-1} - \beta_t + \beta_t\bar{\alpha}_{t-1} + \beta_t && \text{(expand the product)} \\
&= 1 - \bar{\alpha}_{t-1}(1-\beta_t) && \text{(the } \beta_t \text{ terms cancel)} \\
&= 1 - \bar{\alpha}_t && \text{(definition of } \bar{\alpha}_t) .
\end{aligned}
$$

  So the noise part equals $\sqrt{1-\bar{\alpha}_t}\,\bar{\epsilon}_t$ with $\bar{\epsilon}_t \sim \mathcal{N}(0, I)$, which proves the claim for $t$.

Equivalently,

$$
q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar{\alpha}_t}\, x_0,\ (1-\bar{\alpha}_t) I\big), \qquad x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
$$

From here on, $\epsilon$ in this formula means that combined noise $\bar{\epsilon}_t$, and it is exactly the quantity an $\epsilon$-prediction network is trained to recover. This is why training is cheap: pick $t$, draw one $\epsilon$, and form $x_t$ in one shot, with no simulation of the chain.

**The posterior given $x_0$.** The true reverse marginal $q(x_{t-1} \mid x_t)$ depends on the unknown $p_{\text{data}}$, but once we also condition on $x_0$ everything is Gaussian. By Bayes' rule (the Markov property lets us drop $x_0$ from the first factor),

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} \;\propto\; q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0) \quad \text{(as a function of } x_{t-1}) .
$$

Take logs and keep only the terms that depend on $x_{t-1}$:

$$
\log q(x_{t-1} \mid x_t, x_0) = -\frac{1}{2}\left[\frac{\lVert x_t - \sqrt{1-\beta_t}\, x_{t-1}\rVert^2}{\beta_t} + \frac{\lVert x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\, x_0\rVert^2}{1-\bar{\alpha}_{t-1}}\right] + \text{const}.
$$

Expand both squares and collect powers of $x_{t-1}$. The result has the form $-\tfrac12\big(A\lVert x_{t-1}\rVert^2 - 2\,b^\top x_{t-1}\big) + \text{const}$ with

$$
A = \frac{1-\beta_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}, \qquad b = \frac{\sqrt{1-\beta_t}}{\beta_t}\, x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\, x_0 .
$$

A log-density of this form is a Gaussian with variance $1/A$ and mean $b/A$ (complete the square: $A\lVert x\rVert^2 - 2b^\top x = A\lVert x - b/A\rVert^2 - \lVert b\rVert^2/A$). Over a common denominator, and using the identity proved above,

$$
A = \frac{(1-\beta_t)(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t\,(1-\bar{\alpha}_{t-1})} = \frac{1-\bar{\alpha}_t}{\beta_t\,(1-\bar{\alpha}_{t-1})} .
$$

So the variance is $\tilde{\beta}_t = 1/A$, and the mean is $\tilde{\beta}_t\, b$, which we multiply out term by term:

$$
\begin{aligned}
q(x_{t-1} \mid x_t, x_0) &= \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I), \qquad \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t, \\
\tilde{\mu}_t &= \tilde{\beta}_t\, b = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\, x_0 + \frac{\sqrt{1-\beta_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\, x_t .
\end{aligned}
$$

In words, the posterior mean is a fixed weighted average of the clean image and the current noisy image. The weights depend only on the schedule, and the variance $\tilde{\beta}_t$ is smaller than the forward step's $\beta_t$ because $\bar{\alpha}_{t-1} > \bar{\alpha}_t$ (at $t = 1$ it is exactly 0, since $x_0$ is then known). The reverse model will try to match this Gaussian without knowing $x_0$.

::: {.qa asked="Mistral AI SWE screen, PracHub 2026"}
**Explain the forward and reverse processes of a diffusion model.**

The forward process is a fixed Gaussian Markov chain that mixes data with more and more noise; it has the closed form $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ and ends near $\mathcal{N}(0, I)$. The reverse process is a learned chain $p_\theta(x_{t-1} \mid x_t)$, modeled as Gaussian because each forward step is small. The network takes $x_t$, the noise level and any conditioning, and predicts the noise (or $x_0$, or $v$); training is an MSE at a random $t$. Sampling starts from pure noise and applies the learned denoising step repeatedly, often with fewer steps than training via DDIM or an ODE solver. A complete answer also names the design choices: noise schedule, prediction target, sampler, and guidance scale, since each changes quality, diversity and cost.
:::

::: {.qa asked="Prenuvo senior ML scientist, Glassdoor 2024"}
**Explain diffusion models and their applications. Why use one instead of a GAN or a VAE?**

Diffusion models are trained with a stable regression loss, so there is no adversarial game and no mode collapse in the GAN sense, and they cover the data distribution well. A VAE decodes in one step with a simple (often factorized Gaussian) likelihood, so its outputs average over plausible details and look blurry; diffusion splits generation into many small steps, each of which a Gaussian models well, so samples are much sharper. The price is sampling cost: tens to hundreds of network evaluations instead of one, which is why fast solvers and distillation matter. Applications include text-to-image and video generation, super-resolution and inpainting (as conditional generation or as a learned prior for inverse problems), editing via inversion, audio, and molecules. See [GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html) for the comparison in depth.
:::

## The reverse process and the ELBO

If every $\beta_t$ is small, the reverse of a single forward step is close to Gaussian, so we set $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \varsigma_t^2 I)$ with $p(x_T) = \mathcal{N}(0, I)$.

**ELBO.** Start from $-\log p_\theta(x_0) \le \mathbb{E}_q\big[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\big]$. For $t \ge 2$, the Markov property and Bayes give $q(x_t \mid x_{t-1}) = q(x_{t-1} \mid x_t, x_0)\, \frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$. The ratio of marginals telescopes across $t$, leaving

$$
L = \mathbb{E}_q\Big[\underbrace{\mathrm{KL}\big(q(x_T \mid x_0)\,\|\,p(x_T)\big)}_{L_T} + \sum_{t=2}^{T} \underbrace{\mathrm{KL}\big(q(x_{t-1} \mid x_t, x_0)\,\|\,p_\theta(x_{t-1} \mid x_t)\big)}_{L_{t-1}} \underbrace{- \log p_\theta(x_0 \mid x_1)}_{L_0}\Big].
$$

$L_T$ has no parameters. Each $L_{t-1}$ is a KL between Gaussians with fixed covariances, so only the means matter:

$$
L_{t-1} = \mathbb{E}_q\Big[\tfrac{1}{2\varsigma_t^2}\big\lVert \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\big\rVert^2\Big] + C.
$$

**Noise parameterization.** Substitute $x_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon)/\sqrt{\bar{\alpha}_t}$ into $\tilde{\mu}_t$. The $x_t$ coefficients combine to $1/\sqrt{1-\beta_t}$, giving

$$
\tilde{\mu}_t = \frac{1}{\sqrt{1-\beta_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon\Big).
$$

Since $x_t$ is a network input, the only unknown is $\epsilon$, so DDPM uses the same form with $\epsilon_\theta(x_t, t)$ in place of $\epsilon$. Then

$$
L_{t-1} = \mathbb{E}\Big[\frac{\beta_t^2}{2\varsigma_t^2 (1-\beta_t)(1-\bar{\alpha}_t)}\big\lVert \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\ t)\big\rVert^2\Big] + C,
$$

and Ho et al. drop the weight:

$$
L_{\text{simple}} = \mathbb{E}_{t \sim \mathcal{U}\{1..T\},\, x_0,\, \epsilon}\big\lVert \epsilon - \epsilon_\theta(x_t, t)\big\rVert^2 .
$$

The dropped weight is largest at small $t$, so $L_{\text{simple}}$ shifts emphasis toward harder, noisier steps. It gives better samples and worse likelihoods than the true bound. DDPM fixes $\varsigma_t^2$ to $\beta_t$ or $\tilde{\beta}_t$ (similar results). Improved DDPM (Nichol & Dhariwal) learns $\Sigma_\theta = \exp(v \log \beta_t + (1-v)\log \tilde{\beta}_t)$ with $L_{\text{hybrid}} = L_{\text{simple}} + 0.001\, L_{\text{vlb}}$, which improves likelihood and keeps quality with about 100 sampling steps.

::: pitfall
"$L_{\text{simple}}$ is the ELBO." It is a reweighted ELBO. Kingma & Gao show that the common objectives are all the same per-noise-level loss integrated with different weights $w(\lambda)$; only $w(\lambda) = 1$ is the ELBO.
:::

::: qa
**The DDPM chain has 1000 latent steps. Why does its loss reduce to a simple MSE on the noise?**

Three structural facts do the work. The forward marginals $q(x_t \mid x_0)$ are Gaussian in closed form, so any $x_t$ is one reparameterized sample away. The forward posterior $q(x_{t-1} \mid x_t, x_0)$ is also Gaussian, so the ELBO splits into per-step KLs between Gaussians. With the reverse variance fixed, each KL is a weighted squared distance between means, and writing the mean in terms of $\epsilon$ turns it into a weighted $\lVert \epsilon - \epsilon_\theta \rVert^2$. Uniform sampling of $t$ gives an unbiased estimate of the sum, so no chain is ever simulated. Dropping the weights gives $L_{\text{simple}}$.
:::

::: qa
**Why is each reverse step modeled as a Gaussian, and when does that assumption fail?**

The true reverse step $q(x_{t-1} \mid x_t)$ is a mixture over all clean images consistent with $x_t$. When the forward step is small, $x_{t-1}$ is pinned close to $x_t$ and this mixture is close to a Gaussian, so a Gaussian model loses little. With large jumps (few sampling steps), the reverse conditional becomes multimodal, and a Gaussian centered on the conditional mean averages the modes, which produces blur. This is why naive few-step sampling degrades and why few-step generators use distillation or adversarial losses, covered on the [flow matching](flow-matching.html) page.
:::

::: {.qa asked="Photoroom AS, diffusion coding task, Glassdoor 2025"}
**Write the core of a DDPM training step.**

```python
def ddpm_loss(model, x0, alpha_bar):           # alpha_bar: (T,) cumulative products
    B = x0.shape[0]
    t = torch.randint(0, alpha_bar.numel(), (B,), device=x0.device)
    a = alpha_bar[t].view(B, *([1] * (x0.dim() - 1)))
    eps = torch.randn_like(x0)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    return F.mse_loss(model(xt, t), eps)
```

Things interviewers check: broadcasting $\bar{\alpha}_t$ per sample, using $\bar{\alpha}_t$ (not $\beta_t$) and square roots, consistent 0- or 1-based timestep indexing between training and sampling, and, for conditional models, randomly replacing the condition with a null token so classifier-free guidance works later.
:::

## Score matching and Tweedie's formula

The score $\nabla_x \log p(x)$ does not depend on the normalizing constant, so it is a natural target. Plain score matching needs the unknown $\nabla \log p$; denoising score matching (Vincent, 2011) replaces it with something we know.

**DSM identity.** Let $p_\sigma(\tilde{x}) = \int q(\tilde{x} \mid x)\, p(x)\, dx$ with Gaussian $q$. Then

$$
\mathbb{E}_{p_\sigma}\big\lVert s(\tilde{x}) - \nabla \log p_\sigma(\tilde{x})\big\rVert^2 = \mathbb{E}_{p(x)\,q(\tilde{x} \mid x)}\big\lVert s(\tilde{x}) - \nabla_{\tilde{x}} \log q(\tilde{x} \mid x)\big\rVert^2 + C.
$$

Expand both squares. The $\lVert s \rVert^2$ terms match, and the cross terms match because

$$
\mathbb{E}_{p_\sigma}[s^\top \nabla \log p_\sigma] = \int s^\top \nabla p_\sigma\, d\tilde{x} = \iint s^\top \nabla_{\tilde{x}} q(\tilde{x} \mid x)\, p(x)\, dx\, d\tilde{x} = \mathbb{E}[s^\top \nabla_{\tilde{x}} \log q(\tilde{x} \mid x)].
$$

For $x_t = \alpha_t x_0 + \sigma_t \epsilon$, $\nabla \log q(x_t \mid x_0) = -\epsilon/\sigma_t$, so DSM at level $t$ is $\mathbb{E}\lVert s_\theta(x_t) + \epsilon/\sigma_t \rVert^2$. With $s_\theta = -\epsilon_\theta/\sigma_t$ this is $\sigma_t^{-2}\,\mathbb{E}\lVert \epsilon_\theta - \epsilon \rVert^2$: noise prediction is denoising score matching up to a per-level weight.

**Tweedie's formula.** The same exchange of gradient and integral gives $\nabla \log p_t(x_t) = \mathbb{E}[\nabla \log q(x_t \mid x_0) \mid x_t] = -\big(x_t - \alpha_t\, \mathbb{E}[x_0 \mid x_t]\big)/\sigma_t^2$. Rearranging:

$$
\mathbb{E}[x_0 \mid x_t] = \frac{x_t + \sigma_t^2\, \nabla \log p_t(x_t)}{\alpha_t}, \qquad \mathbb{E}[\epsilon \mid x_t] = -\sigma_t\, \nabla \log p_t(x_t).
$$

**Why the denoiser estimates the posterior mean.** For any random pair, $\arg\min_f \mathbb{E}\lVert x_0 - f(x_t)\rVert^2 = \mathbb{E}[x_0 \mid x_t]$. So an MSE-trained $x_0$-predictor converges to the posterior mean, an $\epsilon$-predictor to $\mathbb{E}[\epsilon \mid x_t]$, and by Tweedie both are affine functions of the score of the noisy marginal $p_t$.

::: key
The network cannot recover the exact noise that was added; that is not identifiable from $x_t$. It learns the average noise consistent with $x_t$, which is a rescaled score. One trained denoiser is therefore a score model at every noise level.
:::

::: qa
**Why does a network trained to predict noise give you the score?**

MSE regression converges to a conditional expectation, so at the optimum $\epsilon_\theta(x_t, t) = \mathbb{E}[\epsilon \mid x_t]$. Tweedie's formula for Gaussian corruption gives $\mathbb{E}[\epsilon \mid x_t] = -\sigma_t \nabla \log p_t(x_t)$, so $s_\theta = -\epsilon_\theta/\sigma_t$. Equivalently, the $\epsilon$-loss is denoising score matching up to a per-level weight, and DSM equals explicit score matching on $p_t$ up to a constant. Two caveats: this holds only at the optimum, and it gives the score of the noise-smoothed marginal $p_t$, not of $p_{\text{data}}$, which may not have a score at all if the data sit on a low-dimensional manifold.
:::

::: qa
**One-step $x_0$ predictions at high noise look blurry. Why are final samples sharp?**

At high noise many clean images are consistent with $x_t$, and the MSE-optimal prediction is their average, the posterior mean, which is blurry. The sampler does not output that high-noise average directly. Each step moves only a little toward it and then either re-noises (ancestral sampling) or follows the ODE, so the next prediction is made from a less noisy input where the posterior is narrower. The sequence of small, locally accurate steps commits to one mode gradually. Blur reappears when you take too few steps, because each step then trusts a wide posterior mean.
:::

## The SDE view

Song et al. (2021) take $T \to \infty$, giving a forward SDE $dx = f(x,t)\,dt + g(t)\,dw$.

- **VP SDE** (limit of DDPM): $x_i = \sqrt{1-\beta_i}\,x_{i-1} + \sqrt{\beta_i}\,z$ becomes $dx = -\tfrac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw$. Variance stays bounded.
- **VE SDE** (limit of NCSN): $x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}\,z$ becomes $dx = \sqrt{\tfrac{d[\sigma^2(t)]}{dt}}\,dw$. Here $\alpha_t = 1$ and the variance grows to $\sigma_{\max}^2$.

**Reverse-time SDE** (Anderson, 1982), integrated from $t = T$ down to $0$:

$$
dx = \big[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\big]\,dt + g(t)\,d\bar{w}.
$$

**Probability-flow ODE.** The Fokker-Planck equation of the forward SDE is $\partial_t p_t = -\nabla\cdot(f p_t) + \tfrac{1}{2}g^2 \Delta p_t$. Since $\Delta p = \nabla\cdot(p\,\nabla \log p)$,

$$
\partial_t p_t = -\nabla\cdot\Big[\big(f - \tfrac{1}{2}g^2 \nabla \log p_t\big)\,p_t\Big],
$$

which is a continuity equation with velocity $f - \tfrac{1}{2}g^2\nabla\log p_t$. So the deterministic ODE

$$
\frac{dx}{dt} = f(x,t) - \tfrac{1}{2}g(t)^2\, \nabla_x \log p_t(x)
$$

has exactly the same marginals $p_t$ as the SDE. The same argument gives a whole family: for any $\zeta \ge 0$, the reverse process $dx = [f - \tfrac{1+\zeta^2}{2} g^2 \nabla \log p_t]\,dt + \zeta g\, d\bar{w}$ preserves the marginals ($\zeta = 1$ is the reverse SDE, $\zeta = 0$ the ODE). The ODE gives a deterministic, invertible map between noise and data, exact likelihoods via the instantaneous change-of-variables formula (with a Hutchinson trace estimator), and access to high-order ODE solvers.

In EDM's VE form with $\sigma(t) = t$, Tweedie gives $\nabla \log p(x;\sigma) = (D(x;\sigma) - x)/\sigma^2$ for the denoiser $D$, and the ODE becomes $dx/d\sigma = (x - D(x;\sigma))/\sigma$. All EDM samplers integrate this.

::: qa
**How are DDPM and score-based generative models (NCSN) related?**

They are two discretizations of the same continuous framework: DDPM is a discretized VP SDE and NCSN (annealed Langevin over noise levels) is a discretized VE SDE. DDPM's noise predictor is a scaled score, $s_\theta = -\epsilon_\theta/\sigma_t$, and both are trained by denoising score matching with different per-level weights. Ancestral DDPM sampling is a discretization of the reverse SDE; NCSN's Langevin steps act as a corrector. The SDE view adds the probability-flow ODE (deterministic sampling, likelihoods), predictor-corrector samplers, and a clean way to compare schedules.
:::

::: qa
**The probability-flow ODE injects no noise. How can it produce the same distribution as the reverse SDE?**

The Fokker-Planck equation of the SDE can be rewritten as a pure transport equation by absorbing the diffusion term into the drift: $\tfrac{1}{2}g^2\Delta p = \nabla\cdot(\tfrac{1}{2}g^2 p\, \nabla\log p)$. A deterministic flow with that modified drift moves probability mass exactly as the SDE does, so the marginals match at every $t$, while individual trajectories differ. The equivalence assumes the exact score. With learned scores and discretization error, the two behave differently: the SDE's fresh noise tends to contract accumulated errors, while the ODE carries them forward but converges faster with high-order solvers.
:::

## Parameterizations

Given $x_t$ and $t$, the relation $x_t = \alpha_t x_0 + \sigma_t \epsilon$ leaves one unknown, so every target is an affine function of the others. With variance preserving $\alpha_t^2 + \sigma_t^2 = 1$:

| Target | Definition | Recover $\hat{x}_0$ | Recover $\hat{\epsilon}$ |
|---|---|---|---|
| noise | $\epsilon$ | $(x_t - \sigma_t \hat{\epsilon})/\alpha_t$ | $\hat{\epsilon}$ |
| data | $x_0$ | $\hat{x}_0$ | $(x_t - \alpha_t \hat{x}_0)/\sigma_t$ |
| velocity | $v = \alpha_t \epsilon - \sigma_t x_0$ | $\alpha_t x_t - \sigma_t \hat{v}$ | $\sigma_t x_t + \alpha_t \hat{v}$ |
| score | $\nabla \log p_t(x_t)$ | $(x_t + \sigma_t^2 \hat{s})/\alpha_t$ | $-\sigma_t \hat{s}$ |

With $\alpha_t = \cos\phi$, $\sigma_t = \sin\phi$, $v = dx_t/d\phi$, the velocity along the circle between data and noise. In terms of the $x_0$ error,

$$
\lVert \epsilon - \hat{\epsilon}\rVert^2 = \mathrm{SNR}(t)\,\lVert x_0 - \hat{x}_0\rVert^2, \qquad \lVert v - \hat{v}\rVert^2 = \big(1 + \mathrm{SNR}(t)\big)\lVert x_0 - \hat{x}_0\rVert^2 .
$$

So changing the target does two things: it changes the implied loss weight over noise levels, and it changes how well-conditioned the regression is (how large the target is and how much output errors get amplified).

When each is preferred:

- **$\epsilon$**: unit-variance target at every level; works well for many-step sampling. But $\hat{x}_0 = (x_t - \sigma_t\hat{\epsilon})/\alpha_t$ amplifies errors by $\sigma_t/\alpha_t$ at low SNR, and at zero SNR the task is trivial ($x_T = \epsilon$) and says nothing about data.
- **$x_0$**: well-behaved at high noise (predict a mean image). At low noise the network must reproduce almost its input, and the implied $\epsilon$ is noisy; usually paired with a skip connection.
- **$v$**: behaves like $\epsilon$-prediction at high SNR and like $-x_0$ at low SNR, with a unit-variance target if the data have unit variance. It stays well-defined at zero SNR, which is why progressive distillation and the zero-terminal-SNR fix use it.
- **score**: natural in theory, but it scales like $1/\sigma_t$, so networks rarely output it directly.
- **EDM preconditioning** (next section) mixes these in a $\sigma$-dependent way; with $\sigma_{\text{data}} = 1$ its network input $c_{\text{in}}x$ is exactly the VP sample $x_t$ (with $\sigma_t/\alpha_t = \sigma$) and its effective target is exactly $-v$.

::: qa
**Compare $\epsilon$-, $x_0$- and $v$-prediction. When would you pick each?**

All three carry the same information given $x_t$, so the differences are optimization and numerics. $\epsilon$-prediction has a unit-variance target and is the classic choice for many-step samplers, but implied $\hat{x}_0$ errors are amplified by $\sigma_t/\alpha_t$ at low SNR. $x_0$-prediction is stable at high noise but wasteful at low noise, where it must copy most of the input. $v$-prediction interpolates: like $\epsilon$ when noise is small, like $x_0$ when noise is large, so its errors stay bounded across the range. I would use $v$ (or EDM preconditioning) for zero terminal SNR, few-step distillation or high resolution, and remember that switching targets also changes the implied loss weighting.
:::

::: qa
**Why does $\epsilon$-prediction break down at zero SNR and in one-step distillation?**

At zero SNR, $x_T = \epsilon$, so predicting the noise is the identity map and the loss gives no signal about the data. Converting to an image needs $\hat{x}_0 = (x_t - \sigma_t\hat{\epsilon})/\alpha_t$, which divides by $\alpha_t = 0$. Near zero SNR the division is finite but amplifies tiny output errors enormously. With many steps later updates correct this; with one or few steps the first prediction must already be a good image. Equivalently, the $\epsilon$-loss gives weight $\mathrm{SNR} \to 0$ to $x_0$ reconstruction there. Salimans & Ho switched to $x_0$, mixed, or $v$ targets for this reason.
:::

## Noise schedules and SNR

Think of a schedule as a curve $\lambda_t$ in log-SNR. It decides which noise levels the network trains on and how often, and separately, where the sampler places its steps. Kingma & Gao make the training side precise: changing variables from $t$ to $\lambda$,

$$
\mathcal{L}_w = \tfrac{1}{2}\int_{\lambda_{\min}}^{\lambda_{\max}} w(\lambda)\, \mathbb{E}_\epsilon\lVert \hat{\epsilon}_\theta(x_\lambda;\lambda) - \epsilon\rVert^2\, d\lambda,
$$

which does not depend on the schedule except through its endpoints. The schedule only sets the sampling density $p(\lambda)$ of a Monte Carlo estimate, that is, an importance-sampling distribution. Plain $\epsilon$-MSE with uniform $t$ therefore has implied weight $w(\lambda) = p(\lambda)$.

**Linear and cosine.** DDPM's linear $\beta_t$ destroys information too early at 32 and 64 pixels: Nichol & Dhariwal found that skipping up to the first 20% of the reverse process (the noisiest steps) barely changed FID. Their cosine schedule sets

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\Big(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\Big), \quad s = 0.008,
$$

with $\beta_t$ clipped at $0.999$. In continuous time without the offset this is $\lambda_t = -2\log\tan(\pi t/2)$.

**Resolution-dependent shifting.** Noise is independent per pixel, but image content is strongly correlated across neighboring pixels. Averaging an $s \times s$ patch keeps the signal and divides noise variance by $s^2$, so the coarse-scale SNR is $s^2$ times the per-pixel SNR. At high resolution, the same schedule leaves global layout visible at nominally high noise, so the model has only a short window in which to decide composition. Simple diffusion (Hoogeboom et al.) defines the schedule at a reference resolution of 64 and sets

$$
\mathrm{SNR}^{d\times d}_{\text{shift}}(t) = \mathrm{SNR}^{64\times 64}(t)\cdot (64/d)^2,
$$

a shift of $2\log(64/d)$ in log-SNR. Stable Diffusion 3 uses the same idea for rectified flow, $t_m = \frac{\alpha t_n}{1 + (\alpha - 1)t_n}$ with shift factor $\alpha = \sqrt{m/n}$ for pixel counts $m, n$ (here $\alpha$ is not the signal scale; equivalently $\lambda$ shifted by $-2\log\alpha$), and uses $\alpha = 3$ at $1024^2$. See [flow matching](flow-matching.html) for that parameterization.

**Zero terminal SNR.** The linear, clipped cosine and Stable Diffusion schedules do not reach $\mathrm{SNR} = 0$. Stable Diffusion's schedule ends at $\sqrt{\bar{\alpha}_T} \approx 0.068$, so $x_T \approx 0.068\,x_0 + 0.998\,\epsilon$. During training the model sees this leaked low-frequency signal at $t = T$, including each channel's mean, and learns to keep it. At inference it gets pure noise with zero mean, so outputs stay near medium brightness and prompts like "solid black background" fail. Lin et al. fix the mismatch with four changes: (1) rescale so $\sqrt{\bar{\alpha}_T} = 0$ while keeping $\sqrt{\bar{\alpha}_1}$,

$$
\sqrt{\bar{\alpha}_t}' = \big(\sqrt{\bar{\alpha}_t} - \sqrt{\bar{\alpha}_T}\big)\,\frac{\sqrt{\bar{\alpha}_1}}{\sqrt{\bar{\alpha}_1} - \sqrt{\bar{\alpha}_T}};
$$

(2) train with $v$-prediction, since $\epsilon$-prediction is trivial at $t = T$ (there $v = -x_0$, so the model predicts the prompt-conditional mean); (3) make samplers start at $t = T$ ("trailing" timestep spacing; some DDIM implementations never visit $T$); (4) rescale CFG to avoid over-exposure. VE schedules cannot reach zero SNR because $\alpha_t = 1$.

::: qa
**Why should the noise schedule change with resolution?**

SNR is defined per pixel, but what matters for generation is how much of the image's structure survives. Neighboring pixels share signal while noise is independent, so downsampling by $s$ raises the effective SNR by $s^2$. A schedule tuned at 64 pixels therefore under-noises a 512-pixel image: global structure is still readable at what should be near-pure noise, and the model gets little training at the levels where layout is decided. Shifting log-SNR down by $2\log(d/64)$ (simple diffusion) or using SD3's timestep shift restores the balance. Some papers interpolate between shifts so fine detail still gets enough steps.
:::

::: qa
**What is zero terminal SNR, and which train/test mismatch does it fix?**

Schedules like Stable Diffusion's leave a little signal at $t = T$ ($\sqrt{\bar{\alpha}_T} \approx 0.068$). In training the model learns to read low-frequency information such as the mean color from that leak; at sampling the input is pure zero-mean noise, so outputs stay near medium brightness. Rescaling so $\bar{\alpha}_T = 0$ makes the training input at $t = T$ match the sampling input. It requires $v$- or $x_0$-prediction (at zero SNR, $\epsilon$-prediction is trivial and $\hat{x}_0$ undefined), a sampler that starts at $t = T$, and, per Lin et al., CFG rescaling to prevent over-exposure.
:::

::: pitfall
"The cosine schedule reaches zero SNR." Improved DDPM clips $\beta_t \le 0.999$, so its terminal SNR is small but positive; removing the clip fixes it. Also, the training schedule and the sampling time grid are separate choices.
:::

## Loss weighting and EDM preconditioning

All the objectives above can be written as $\mathbb{E}_t[w(t)\,\lVert x_0 - \hat{x}_0\rVert^2]$ for some weight. The $\epsilon$-loss corresponds to $w = \mathrm{SNR}$, which puts very large weight on low-noise steps that are easy and contribute little to perceptual quality.

**Min-SNR-$\gamma$** (Hang et al.) treats each timestep as a task in a multi-task problem whose gradients conflict, and clamps the weight to $w_t = \min(\mathrm{SNR}(t), \gamma)$ in $x_0$ space, default $\gamma = 5$. For other targets, divide by the implicit factor:

$$
w^{\epsilon}_t = \frac{\min(\mathrm{SNR}, \gamma)}{\mathrm{SNR}} = \min\Big(\frac{\gamma}{\mathrm{SNR}}, 1\Big), \qquad w^{v}_t = \frac{\min(\mathrm{SNR}, \gamma)}{\mathrm{SNR} + 1}.
$$

They report about 3.4 times faster convergence to the same FID on ImageNet 256.

**EDM preconditioning** (Karras et al. 2022). Write the denoiser as

$$
D_\theta(x;\sigma) = c_{\text{skip}}(\sigma)\,x + c_{\text{out}}(\sigma)\,F_\theta\big(c_{\text{in}}(\sigma)\,x;\ c_{\text{noise}}(\sigma)\big),
$$

with $x = x_0 + \sigma\epsilon$ and data standard deviation $\sigma_{\text{data}}$. The loss $\lambda(\sigma)\lVert D_\theta - x_0\rVert^2$ equals $\lambda(\sigma)\,c_{\text{out}}^2\,\lVert F_\theta - (x_0 - c_{\text{skip}}x)/c_{\text{out}}\rVert^2$. The design rules:

- Unit-variance input: $c_{\text{in}}^2(\sigma^2 + \sigma_{\text{data}}^2) = 1$, so $c_{\text{in}} = 1/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$.
- Unit-variance target: $c_{\text{out}}^2 = (1 - c_{\text{skip}})^2\sigma_{\text{data}}^2 + c_{\text{skip}}^2\sigma^2$.
- Amplify network errors as little as possible: minimize $c_{\text{out}}$ over $c_{\text{skip}}$. Setting the derivative to zero gives

$$
c_{\text{skip}} = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}, \qquad c_{\text{out}} = \frac{\sigma\,\sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}} .
$$

  The noise conditioning $c_{\text{noise}} = \tfrac{1}{4}\ln\sigma$ does not follow from these requirements; the paper says it was chosen empirically.

- Equalize the effective weight ($\lambda(\sigma)$ here is EDM's loss weight, not log-SNR): $\lambda(\sigma) = 1/c_{\text{out}}^2 = (\sigma^2 + \sigma_{\text{data}}^2)/(\sigma\,\sigma_{\text{data}})^2$.

As $\sigma \to 0$, $c_{\text{skip}} \to 1$ and $D \approx x$ (the network only adds a small correction); as $\sigma \to \infty$, $c_{\text{skip}} \to 0$ and the network predicts $x_0$ directly. Training noise levels follow a log-normal, $\ln\sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$ with $P_{\text{mean}} = -1.2$, $P_{\text{std}} = 1.2$ and $\sigma_{\text{data}} = 0.5$, because the loss can only be reduced substantially at intermediate noise: at tiny $\sigma$ the noise is hard to detect and irrelevant, and at huge $\sigma$ the best answer is close to the dataset mean.

::: qa
**Derive EDM's $c_{\text{skip}}$, $c_{\text{out}}$ and $c_{\text{in}}$.**

With $x = x_0 + \sigma\epsilon$ and independent $x_0$ of variance $\sigma_{\text{data}}^2$, the input has variance $\sigma^2 + \sigma_{\text{data}}^2$, so $c_{\text{in}} = (\sigma^2 + \sigma_{\text{data}}^2)^{-1/2}$. The network's effective target is $(x_0 - c_{\text{skip}}x)/c_{\text{out}}$; requiring unit variance gives $c_{\text{out}}^2 = (1 - c_{\text{skip}})^2\sigma_{\text{data}}^2 + c_{\text{skip}}^2\sigma^2$. Minimizing this over $c_{\text{skip}}$ (so network errors are scaled up as little as possible) gives $c_{\text{skip}} = \sigma_{\text{data}}^2/(\sigma^2 + \sigma_{\text{data}}^2)$. Substituting back gives $c_{\text{out}} = \sigma\sigma_{\text{data}}/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$. Finally $\lambda = 1/c_{\text{out}}^2$ makes the effective loss weight one at every $\sigma$.
:::

::: qa
**Why does Min-SNR weighting speed up training?**

Under the $\epsilon$-loss, the implied $x_0$-space weight is the SNR, so low-noise timesteps dominate the gradient even though those denoising tasks are easy and matter little for global structure. Hang et al. measure that gradients from different timesteps conflict, and they frame training as multi-task learning. Clamping the weight at $\gamma$ caps the influence of high-SNR steps and leaves noisier steps unchanged, which is close to the Pareto-optimal weighting they compute directly. For an $\epsilon$-model the per-sample weight is $\min(\gamma/\mathrm{SNR}, 1)$, which down-weights only the cleanest timesteps.
:::

## Samplers

**Ancestral DDPM.** Use the learned reverse Gaussian:

$$
x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\Big) + \varsigma_t z, \qquad \varsigma_t^2 \in \{\beta_t, \tilde{\beta}_t\}.
$$

It needs roughly as many steps as training because the Gaussian approximation relies on small steps.

**DDIM.** Look for non-Markovian forward processes with the same marginals $q(x_t \mid x_0)$. Try

$$
q_\varsigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(\sqrt{\bar{\alpha}_{t-1}}\,x_0 + k_t\,(x_t - \sqrt{\bar{\alpha}_t}\,x_0),\ \varsigma_t^2 I\big).
$$

Because $x_t - \sqrt{\bar{\alpha}_t}x_0 = \sqrt{1-\bar{\alpha}_t}\,\epsilon$, the marginal of $x_{t-1}$ has variance $k_t^2(1-\bar{\alpha}_t) + \varsigma_t^2$, which must equal $1 - \bar{\alpha}_{t-1}$. So $k_t = \sqrt{1-\bar{\alpha}_{t-1}-\varsigma_t^2}/\sqrt{1-\bar{\alpha}_t}$. Replacing $x_0$ and $\epsilon$ by network estimates gives

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\varsigma_t^2}\;\hat{\epsilon} + \varsigma_t z, \qquad \varsigma_t(\eta) = \eta\sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}}\sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}.
$$

$\eta = 1$ recovers DDPM's posterior variance $\tilde{\beta}_t$; $\eta = 0$ is deterministic. No retraining is needed because the $\epsilon$-network only depends on the marginals $q(x_t \mid x_0)$, which every member of the family shares (the DDIM paper shows each variational objective is a reweighted $L_{\text{simple}}$, with the same optimum when parameters are not shared across $t$). Steps can be skipped because the derivation only used the marginals at two times, so $(t, t-1)$ can be any pair $(\tau_i, \tau_{i-1})$ from a subsequence.

**DDIM is Euler on the ODE.** With $\eta = 0$, divide by $\alpha_{t-1}$ and set $\bar{x} = x/\alpha$, $\bar{\sigma} = \sigma/\alpha$: $\bar{x}_{t-1} = \bar{x}_t + (\bar{\sigma}_{t-1} - \bar{\sigma}_t)\,\hat{\epsilon}$. This is an Euler step of $d\bar{x}/d\bar{\sigma} = \hat{\epsilon}$, the probability-flow ODE in VE coordinates.

**DPM-Solver.** With $\lambda = \log(\alpha_t/\sigma_t)$ (half the log-SNR in this paper's convention), the linear part of the VP ODE can be solved exactly:

$$
x_t = \frac{\alpha_t}{\alpha_s}\,x_s - \alpha_t\int_{\lambda_s}^{\lambda_t} e^{-\lambda}\,\hat{\epsilon}_\theta(\hat{x}_\lambda, \lambda)\,d\lambda .
$$

Taylor-expanding $\hat{\epsilon}_\theta$ in $\lambda$ gives solvers of order $k$. Order 1 is exactly DDIM; orders 2 and 3 give good samples in about 10 to 20 evaluations. DPM-Solver++ uses the $x_0$-prediction form with multistep updates, which is more stable under large guidance.

**EDM Heun.** Time steps $\sigma_i = \big(\sigma_{\max}^{1/\rho} + \tfrac{i}{N-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho})\big)^\rho$ with $\rho = 7$, $\sigma_{\min} = 0.002$, $\sigma_{\max} = 80$ concentrate steps at low noise. Each step is Euler plus a trapezoidal correction:

$$
\begin{aligned}
d_i &= \frac{x_i - D(x_i;\sigma_i)}{\sigma_i}, \qquad \tilde{x} = x_i + (\sigma_{i+1} - \sigma_i)\,d_i, \\
x_{i+1} &= x_i + (\sigma_{i+1} - \sigma_i)\,\tfrac{1}{2}\Big(d_i + \frac{\tilde{x} - D(\tilde{x};\sigma_{i+1})}{\sigma_{i+1}}\Big),
\end{aligned}
$$

skipping the correction on the final step to $\sigma = 0$, so $N$ steps cost $2N - 1$ evaluations. EDM reached FID 1.79 on class-conditional CIFAR-10 with 35 evaluations.

**Stochastic vs deterministic.** Fresh noise (ancestral sampling, EDM's "churn") acts like a Langevin correction that repairs earlier errors; EDM's stochastic sampler took a pretrained ImageNet-64 model from FID 2.07 to 1.55. For EDM's best CIFAR-10 models, deterministic sampling was better. Quality improves with steps with diminishing returns: ODE solvers do well with tens of steps, but below roughly ten the trajectory's curvature dominates and distillation is the usual answer ([flow matching](flow-matching.html)).

::: qa
**Derive DDIM. Why can it skip steps and reuse a DDPM-trained model?**

Define a family of forward processes whose conditional $q(x_{t-1} \mid x_t, x_0)$ is Gaussian with mean $\sqrt{\bar{\alpha}_{t-1}}x_0 + k_t(x_t - \sqrt{\bar{\alpha}_t}x_0)$ and variance $\varsigma_t^2$, and pick $k_t = \sqrt{1-\bar{\alpha}_{t-1}-\varsigma_t^2}/\sqrt{1-\bar{\alpha}_t}$ so that every marginal $q(x_t \mid x_0)$ matches DDPM's. Replacing $x_0$ and $\epsilon$ with network predictions gives the DDIM update. Since $L_{\text{simple}}$ only involves the marginals, the trained network is valid for every member of the family. The construction only uses the marginals at the two endpoints of a step, so any decreasing subsequence of timesteps works. With $\varsigma_t = 0$ the map is deterministic, which also makes approximate inversion possible.
:::

::: qa
**How do DDIM, the probability-flow ODE and DPM-Solver relate?**

Deterministic DDIM is a first-order solver of the probability-flow ODE: in coordinates $\bar{x} = x/\alpha_t$, $\bar{\sigma} = \sigma_t/\alpha_t$ it is exactly an Euler step of $d\bar{x}/d\bar{\sigma} = \hat{\epsilon}$. DPM-Solver writes the VP ODE as a linear term plus a nonlinear term, solves the linear term exactly, and expresses the rest as an exponentially weighted integral of $\hat{\epsilon}$ over half-log-SNR. Its first-order version coincides with DDIM; higher orders approximate $\hat{\epsilon}$ along $\lambda$ with a Taylor expansion. That is why DDIM already beats naive Euler in $t$, and why second and third order buy another large reduction in steps.
:::

::: qa
**When would you use a stochastic sampler instead of a deterministic one?**

Use stochastic sampling when the model is imperfect and you can afford more steps: the injected noise lets the sampler correct earlier errors, which often gives the best FID at large step counts. Use deterministic ODE solvers when steps are scarce, when you need a reproducible noise-to-image map (inversion, interpolation, editing, consistent seeds), or when you want likelihoods. Too much stochasticity has costs: EDM saw loss of detail and color drift, and the best settings had to be tuned per model. In practice, high-order deterministic solvers dominate the low-step regime and mildly stochastic samplers the high-step regime.
:::

## Guidance

**Classifier guidance** (Dhariwal & Nichol). By Bayes, $\nabla\log p(x_t \mid c) = \nabla\log p(x_t) + \nabla\log p(c \mid x_t)$. Scaling the classifier term:

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, c) - w\,\sigma_t\,\nabla_{x_t}\log p_\phi(c \mid x_t) \approx -\sigma_t\nabla_{x_t}\big[\log p(x_t \mid c) + w\log p_\phi(c \mid x_t)\big],
$$

which targets $p(x_t \mid c)\,p_\phi(c \mid x_t)^w$. It needs a separate classifier trained on noisy inputs, and classifier gradients can be adversarial rather than semantic.

**Classifier-free guidance** (Ho & Salimans). Train one network conditionally and unconditionally by replacing $c$ with a null token with probability $p_{\text{uncond}}$ ($0.1$ to $0.2$ worked best; $0.5$ was worse). Define the implicit classifier $p^i(c \mid x_t) \propto p(x_t \mid c)/p(x_t)$. With exact scores,

$$
\nabla\log p^i(c \mid x_t) = \nabla\log p(x_t \mid c) - \nabla\log p(x_t) = -\frac{1}{\sigma_t}\big(\epsilon^*(x_t, c) - \epsilon^*(x_t)\big).
$$

Plugging this into classifier guidance on the conditional model:

$$
\tilde{\epsilon} = \epsilon(x_t, c) + w\big(\epsilon(x_t, c) - \epsilon(x_t)\big) = (1+w)\,\epsilon(x_t, c) - w\,\epsilon(x_t).
$$

Most code writes $\epsilon_u + s(\epsilon_c - \epsilon_u)$ with $s = 1 + w$ ($s = 1$ means no guidance). Each step costs two forward passes, usually batched. A negative prompt simply replaces $\epsilon_u$ with the prediction under that prompt.

**Why fidelity goes up and diversity down.** The target $p(x \mid c)\,p(c \mid x)^w \propto p(x \mid c)^{1+w}/p(x)^w$ up-weights samples where the condition is much more likely than under the unconditional model: prototypical, unambiguous examples. Raising a density to a power above one also sharpens it, like lowering temperature. On ImageNet, FID was best at small guidance ($w = 0.1$ or $0.3$) and Inception score kept rising with $w$; text-to-image systems use $s$ of roughly 5 to 10 because prompt alignment and perceived quality matter more than coverage.

**Nuance.** The guided score at each noise level is not the score of any noised version of the tilted distribution, so samplers do not actually sample $p(x \mid c)^{1+w}/p(x)^w$, and CFG with DDPM and with DDIM gives different distributions (Bradley & Nakkiran; Karras et al. 2024). In practice, guidance at high noise pulls trajectories toward a few template layouts, and at high scale the extrapolated prediction leaves the range of real data, giving oversaturated, over-exposed images.

**Fixes and refinements.**

- *Dynamic thresholding* (Imagen, pixel space): set $s$ to a high percentile of $\lvert\hat{x}_0\rvert$ per sample; if $s > 1$, clip $\hat{x}_0$ to $[-s, s]$ and divide by $s$.
- *CFG rescale* (Lin et al.): match the standard deviation of the guided output to the conditional one, $x_{\text{rescaled}} = x_{\text{cfg}}\cdot\operatorname{std}(x_{\text{pos}})/\operatorname{std}(x_{\text{cfg}})$, then blend $x_{\text{final}} = \phi\,x_{\text{rescaled}} + (1-\phi)\,x_{\text{cfg}}$ with $\phi = 0.7$.
- *Adaptive projected guidance* (Sadat et al.): split the guidance update into components parallel and orthogonal to the conditional prediction; the parallel part mostly causes oversaturation, so down-weight it.
- *Guidance interval* (Kynkäänniemi et al.): apply guidance only in a middle range of noise levels (details in the question below).
- *Autoguidance* (Karras et al. 2024): guide with a worse version of the same conditional model, $D_w = w\,D_1(x;\sigma,c) + (1-w)\,D_0(x;\sigma,c)$, where $D_0$ is smaller and/or trained for less time on the same task and conditioning. It reached FID 1.01 on ImageNet-64 and 1.25 on ImageNet-512, and it also works for unconditional models. In the EDM2-S ablation the best guide was one size smaller (XS) and trained for 1/16 as long; the 1.25 result guided EDM2-XXL with an M-size model.

::: qa
**Derive classifier-free guidance from classifier guidance.**

Classifier guidance adds $w\nabla\log p(c \mid x_t)$ to the conditional score. Replace the external classifier with the implicit one given by Bayes' rule, $p(c \mid x_t) \propto p(x_t \mid c)/p(x_t)$, whose gradient is the difference of conditional and unconditional scores. In noise units the score is $-\epsilon/\sigma_t$, so the gradient is $-(\epsilon(x_t,c) - \epsilon(x_t))/\sigma_t$. Substituting gives $\tilde{\epsilon} = (1+w)\epsilon(x_t, c) - w\epsilon(x_t)$. One network provides both terms because the condition is dropped at random during training. Ho & Salimans note that with learned networks this difference is generally not the gradient of any classifier.
:::

::: qa
**Why does classifier-free guidance increase fidelity and reduce diversity? What happens if the scale is too high?**

Guidance tilts the target toward $p(x \mid c)^{1+w}/p(x)^w$, which concentrates mass where the condition is most distinguishable from the unconditional distribution, so samples are more typical of the prompt. Raising a density to a power above one also lowers its entropy, so variety falls, and at high noise levels guidance steers many seeds toward the same few layouts. At very high scales the extrapolated prediction overshoots the data range: oversaturated colors, over-exposure and artifacts. Fixes: a lower scale, dynamic thresholding, CFG rescale or APG, or a guidance interval.
:::

::: qa
**What is autoguidance and how does it differ from CFG?**

CFG extrapolates away from an unconditional model, which entangles two effects: better prompt alignment with lower variation, and an image-quality gain. Karras et al. argue the quality gain comes from the two models fitting the data with different errors, while the task mismatch (conditional vs unconditional) is what also pulls samples toward prototypes. Autoguidance keeps only the quality effect by guiding a model with a smaller or less-trained version of itself on the same conditioning: $D_w = wD_1 + (1-w)D_0$. Since both models solve the same task, the difference mainly flags regions the weaker model fits poorly and pushes samples toward the data manifold without collapsing them to prototypes. It needs an extra model and does not by itself improve prompt adherence.
:::

::: qa
**Why apply guidance only in a limited interval of noise levels?**

At high noise, the guided update changes the global layout, and with strong guidance many seeds drift toward a few template images, so variation collapses and FID worsens. At low noise, the conditional and unconditional predictions nearly agree, so guidance changes little. Between these, guidance improves detail and alignment. Kynkäänniemi et al. disable it outside a tuned middle interval and keep the same weight inside; on ImageNet-512 FID improved from 1.81 to 1.40, and steps outside the interval need only one network pass.
:::

::: pitfall
Mixing up guidance conventions. In Ho & Salimans, $w = 0$ is unguided and $\tilde{\epsilon} = (1+w)\epsilon_c - w\epsilon_u$; in most libraries "guidance scale" $s = 1 + w$, so $s = 1$ is unguided and $s = 7.5$ corresponds to $w = 6.5$. State which one you mean.
:::

## Latent diffusion and a note on DiT

**Why latent space.** Likelihood-style models spend most of their capacity on imperceptible high-frequency detail. Latent diffusion (Rombach et al.) splits the job: an autoencoder does perceptual compression and the diffusion model does semantic composition in a latent that is $f$ times smaller per side. The autoencoder is trained with a perceptual loss plus a patch-based adversarial loss and a light KL or VQ regularizer, so reconstructions stay sharp. LDM found $f$ between 4 and 16 a good trade-off: $f = 1, 2$ train slowly, and large $f$ loses too much information. Compute drops roughly with the number of latent positions. The autoencoder caps achievable fidelity (small text, faces, fine texture), and latent scale matters for the schedule: LDM rescales KL latents by $1/\hat{\sigma}$, a single standard deviation estimated over all latent entries of the first training batch, so they have roughly unit variance and the schedule's SNR means what it should. The constant $0.18215$ is not in the LDM paper; it is the `scale_factor` in the Stable Diffusion configs and code, described there as this estimate for SD's $f = 8$ autoencoder. Conditioning on text enters through cross-attention. See [GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html) for the autoencoder.

**DiT.** Diffusion Transformers patchify the latent into tokens and replace the U-Net with a transformer. Conditioning on timestep and class uses adaptive layer norm: an MLP on the sum of the two embeddings regresses a per-block scale $\gamma$ and shift $\beta$. adaLN-Zero also regresses a per-dimension gate $\alpha$ applied before each residual connection and initializes it to zero, so every block starts as the identity. It beat in-context and cross-attention conditioning at lower compute, and DiT-XL/2 reached FID 2.27 on ImageNet 256 with guidance. FID tracked model Gflops closely. Architecture depth (MMDiT, text conditioning) is on [text-to-image and multimodal](text-to-image-multimodal.html).

::: {.qa asked="Wells Fargo applied AI research intern, PracHub 2026"}
**Walk through the architecture of a diffusion model: how data are corrupted, what the network predicts, how it is built and conditioned, and how samples are generated.**

An image is encoded to a latent by a pretrained autoencoder, then corrupted as $z_t = \alpha_t z_0 + \sigma_t\epsilon$ at a random noise level. A U-Net or DiT takes $z_t$, an embedding of the noise level and the condition, and predicts $\epsilon$, $v$ or $z_0$; text enters through cross-attention or joint attention, timestep and class through adaptive normalization. Training is a weighted MSE with the condition dropped about 10% of the time. To sample, start from Gaussian noise, run 20 to 50 steps of a DDIM, DPM-Solver or Heun-type solver with classifier-free guidance, then decode the latent. I would also state the schedule (e.g. zero terminal SNR, resolution shift) and how much each choice costs at inference.
:::

::: qa
**Why train diffusion in a latent space, and what can go wrong?**

Pixel-space diffusion spends compute and capacity on high-frequency detail that a good autoencoder reproduces cheaply, so a latent 8 times smaller per side cuts cost roughly by the reduction in positions while keeping perceptual quality. The risks: reconstruction error is a hard ceiling on detail; too much compression loses information; the latent's scale changes the effective SNR, so latents must be normalized or the schedule retuned; and decoder artifacts can look like generator failures. Pixel-space models avoid the autoencoder at higher compute cost.
:::

## Rapid-fire review

::: {.qa asked="TikTok AI research scientist, Interview Query"}
**Name two activation functions used in diffusion networks and how they differ.**

DDPM and ADM U-Nets use SiLU (Swish), $x\,\sigma(x)$; DiT uses tanh-approximated GELU, $x\,\Phi(x)$, in its MLPs and SiLU before the adaLN projection. Both are smooth, non-monotone near zero, and close in shape; GELU weights by the Gaussian CDF, SiLU by the logistic sigmoid.
:::

::: qa
**Why does the network need the timestep or noise level as input?**

The same $x_t$ can arise from different noise levels, and the optimal denoiser depends on $\sigma$. DDPM uses sinusoidal timestep embeddings followed by an MLP; EDM feeds $c_{\text{noise}} = \tfrac{1}{4}\ln\sigma$.
:::

::: qa
**What is the difference between the noise schedule and the loss weighting?**

In continuous time, the loss is an integral over log-SNR with weight $w(\lambda)$, independent of the schedule except its endpoints. The training schedule acts as an importance-sampling density that affects gradient variance; the sampling schedule controls discretization error.
:::

::: qa
**VP vs VE: what is the practical difference?**

VP keeps $\alpha_t^2 + \sigma_t^2 = 1$ and bounded variance; VE keeps $\alpha_t = 1$ and grows noise up to $\sigma_{\max}$. They are rescalings of each other ($x_{\text{VE}} = x_{\text{VP}}/\alpha_t$), but VE cannot reach zero SNR and its inputs need normalization ($c_{\text{in}}$).
:::

::: qa
**What does learning the reverse variance buy?**

Better likelihoods and robustness to fewer sampling steps. Improved DDPM interpolates between $\beta_t$ and $\tilde{\beta}_t$ in log space and trains it with a small $L_{\text{vlb}}$ term; about 100 steps sufficed.
:::

::: qa
**Can diffusion models give likelihoods?**

Yes: a bound via the ELBO in discrete time, or an exact value via the probability-flow ODE using the change-of-variables formula and a Hutchinson trace estimate. Models tuned for sample quality usually have worse likelihoods.
:::

::: qa
**Why is DDIM inversion only approximate?**

Running the deterministic update forward uses $\hat{\epsilon}(x_t)$ where the exact inverse would need $\hat{\epsilon}(x_{t+1})$. The error grows with large steps and with CFG, which is why editing methods add corrections.
:::

::: qa
**Why does $p_{\text{uncond}}$ matter in CFG training?**

It sets how much capacity goes to the unconditional model. Ho & Salimans found 0.1 and 0.2 about equally good and 0.5 clearly worse across the whole quality-diversity frontier.
:::

::: qa
**What is "trailing" timestep spacing?**

Choosing sampler steps counted back from $t = T$ so the first step uses the pure-noise level the model was trained on. "Leading" spacing never visits $T$, so sampling feeds pure noise at a timestep whose training inputs still contained signal, reintroducing the mismatch that zero terminal SNR removes.
:::

::: qa
**How would you reduce sampling steps without retraining?**

Switch from ancestral sampling to a deterministic high-order solver (DPM-Solver++, Heun), use a good time grid (EDM's $\rho = 7$ or trailing spacing), and limit guidance to a middle interval. Below about 10 steps you usually need distillation, which requires training.
:::

::: qa
**Why use an EMA of the weights?**

Sample quality fluctuates noticeably between nearby checkpoints (for example color shifts), and averaging the weights smooths this out. Song & Ermon (2020) recommended EMA for score models for this reason, DDPM also used it, and it is now standard.
:::

::: qa
**What does $\sigma_{\min}$ control?**

The smallest noise level the sampler steps down to before the final jump to $\sigma = 0$ (and, for some schedules, the lowest level used in training). Too large leaves visible noise; too small spends steps on imperceptible noise. EDM samples down to $\sigma_{\min} = 0.002$ for data in $[-1, 1]$, but its training draws $\ln\sigma \sim \mathcal{N}(-1.2, 1.2^2)$ with no lower cutoff; the VE baseline trained down to 0.02.
:::

::: qa
**Why does CFG roughly double inference cost, and how is it reduced?**

Each step evaluates conditional and unconditional branches. Options: batch both, apply guidance only in an interval, or distill a guided teacher into a student that takes the scale as input ([flow matching](flow-matching.html)).
:::

## References

1. Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics," 2015. <https://arxiv.org/abs/1503.03585>
2. Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models," 2020. <https://arxiv.org/abs/2006.11239>
3. Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution," 2019. <https://arxiv.org/abs/1907.05600>
4. Song & Ermon, "Improved Techniques for Training Score-Based Generative Models," 2020. <https://arxiv.org/abs/2006.09011>
5. Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations," 2021. <https://arxiv.org/abs/2011.13456>
6. Song, Meng, Ermon, "Denoising Diffusion Implicit Models," 2020. <https://arxiv.org/abs/2010.02502>
7. Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models," 2021. <https://arxiv.org/abs/2102.09672>
8. Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis," 2021. <https://arxiv.org/abs/2105.05233>
9. Kingma et al., "Variational Diffusion Models," 2021. <https://arxiv.org/abs/2107.00630>
10. Ho & Salimans, "Classifier-Free Diffusion Guidance," 2022. <https://arxiv.org/abs/2207.12598>
11. Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models," 2022. <https://arxiv.org/abs/2202.00512>
12. Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models," 2022. <https://arxiv.org/abs/2206.00364>
13. Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps," 2022. <https://arxiv.org/abs/2206.00927>
14. Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models," 2022. <https://arxiv.org/abs/2211.01095>
15. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," 2021. <https://arxiv.org/abs/2112.10752>
16. Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Imagen), 2022. <https://arxiv.org/abs/2205.11487>
17. Peebles & Xie, "Scalable Diffusion Models with Transformers," 2022. <https://arxiv.org/abs/2212.09748>
18. Hoogeboom et al., "Simple diffusion: End-to-end diffusion for high resolution images," 2023. <https://arxiv.org/abs/2301.11093>
19. Chen, "On the Importance of Noise Scheduling for Diffusion Models," 2023. <https://arxiv.org/abs/2301.10972>
20. Kingma & Gao, "Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation," 2023. <https://arxiv.org/abs/2303.00848>
21. Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy," 2023. <https://arxiv.org/abs/2303.09556>
22. Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed," 2023. <https://arxiv.org/abs/2305.08891>
23. Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models" (EDM2), 2023. <https://arxiv.org/abs/2312.02696>
24. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3), 2024. <https://arxiv.org/abs/2403.03206>
25. Kynkäänniemi et al., "Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models," 2024. <https://arxiv.org/abs/2404.07724>
26. Karras et al., "Guiding a Diffusion Model with a Bad Version of Itself," 2024. <https://arxiv.org/abs/2406.02507>
27. Bradley & Nakkiran, "Classifier-Free Guidance is a Predictor-Corrector," 2024. <https://arxiv.org/abs/2408.09000>
28. Sadat et al., "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models," 2024. <https://arxiv.org/abs/2410.02416>
29. Vincent, "A Connection Between Score Matching and Denoising Autoencoders," Neural Computation, 2011. <https://doi.org/10.1162/NECO_a_00142>
30. Efron, "Tweedie's Formula and Selection Bias," JASA, 2011. <https://doi.org/10.1198/jasa.2011.tm11181>
31. Anderson, "Reverse-time diffusion equation models," Stochastic Processes and their Applications, 1982. <https://doi.org/10.1016/0304-4149(82)90051-5>
32. Luo, "Understanding Diffusion Models: A Unified Perspective," 2022. <https://arxiv.org/abs/2208.11970>
33. Weng, "What are Diffusion Models?" (blog), 2021. <https://lilianweng.github.io/posts/2021-07-11-diffusion-models/>
34. Song, "Generative Modeling by Estimating Gradients of the Data Distribution" (blog), 2021. <https://yang-song.net/blog/2021/score/>
35. Dieleman, "Guidance: a cheat code for diffusion models" (blog), 2022. <https://sander.ai/2022/05/26/guidance.html>
36. Dieleman, "Noise schedules considered harmful" (blog), 2024. <https://sander.ai/2024/06/14/noise-schedules.html>
37. Interview report (Mistral AI): "Explain the Forward and Reverse Processes of Diffusion Models," PracHub. <https://prachub.com/interview-questions/explain-the-forward-and-reverse-processes-of-diffusion-models>
38. Interview report (Wells Fargo): "Research Deep Dive: Model Details, Coding Habits and Diffusion Architecture," PracHub. <https://prachub.com/interview-questions/research-deep-dive-model-details-coding-habits-and-diffusion-architecture>
39. Interview report (Prenuvo), Glassdoor. <https://www.glassdoor.ca/Interview/Prenuvo-Interview-E2603205-RVW89924656.htm>
40. Interview report (Photoroom), Glassdoor. <https://www.glassdoor.sg/Interview/Photoroom-Interview-E8011752-RVW96817759.htm>
41. Interview guide with candidate reports (TikTok AI research scientist), Interview Query. <https://www.interviewquery.com/guides/tiktok-ai-research-scientist>
