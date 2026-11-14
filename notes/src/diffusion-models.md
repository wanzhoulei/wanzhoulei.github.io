---
title: "Diffusion Models"
subtitle: "A guided, fully derived reading of DDPM, score matching, the SDE view, parameterizations, noise schedules, loss weighting, samplers, guidance and latent diffusion, in one notation, with worked numbers you can check by hand."
order: 1
updated: 2026-11-13
---

A diffusion model learns to reverse a fixed process that slowly corrupts data with Gaussian noise. Training is a plain regression problem at random noise levels, and sampling calls the same network many times, removing a little noise each time. Most current image and video generators are diffusion models or close relatives, and flow matching reuses most of the same ideas. Interviewers like the topic because short derivations separate memorized slogans ("predict the noise") from real understanding: why the loss is a mean-squared error, what the network actually estimates, why samplers can skip steps, and why guidance trades diversity for fidelity.

**Roadmap.** This page is written as a guided reading for someone who knows ML and probability but has not worked through the derivations. It first fixes notation in a table and lists twelve probability and calculus facts ("tools") that every later proof cites by number; tools 1 to 5 have the same numbers as on the [DDIM page](ddim.html). Then it goes in teaching order. "The forward process" builds the noising chain and its two closed forms. "The reverse process and the ELBO" derives the training loss from the likelihood bound, step by step, and shows what dropping the weights does. "Score matching and Tweedie's formula" proves that the noise-prediction loss is denoising score matching and that the trained network is a rescaled score. "The SDE view" takes the number of steps to infinity, states the reverse-time SDE and derives the probability-flow ODE. "Parameterizations" derives every conversion between $\epsilon$, $x_0$, $v$ and the score and says where each is well conditioned. "Noise schedules and SNR" and "Loss weighting" explain what the schedule and the weights really control. "Samplers" covers ancestral DDPM, DDIM, DPM-Solver (derived in full), EDM's Heun sampler and the stochastic versus deterministic choice. "Guidance" derives classifier guidance and classifier-free guidance, explains the diversity loss with a solvable toy, and covers autoguidance. "Latent diffusion and DiT" closes the main text. Each major derivation ends with a short "What this says" paragraph, each major section ends with a spoken interview answer and with interview questions, and a rapid-fire review closes the page. The full paper readings of [DDIM](ddim.html) and [EDM](edm.html) are on their own pages; this page summarizes them so it can stand alone.

## Notation

DDPM writes $\alpha_t = 1-\beta_t$ for the per-step factor. This page never uses that symbol: $\alpha_t$ here always means the signal scale in $x_t = \alpha_t x_0 + \sigma_t\epsilon$. The table is the reference for every symbol on the page.

| Symbol | Meaning in words | Notes |
|---|---|---|
| $d$ | dimension of one data sample (number of pixels times channels) | |
| $x_0$ | a clean data sample, a vector in $\mathbb{R}^d$ | images are scaled to $[-1, 1]$ |
| $p_{\text{data}}$, also $q(x_0)$ | the data distribution | |
| $t$, $T$ | noise-level index and the number of levels | DDPM: integer $t \in \{1, \dots, T\}$, $T = 1000$; SDE view: real $t \in [0, 1]$ |
| $x_t$ | the noisy sample at level $t$ | |
| $q$ | the fixed forward (noising) process; also all its marginals and conditionals | nothing in $q$ is learned |
| $p_\theta$ | the learned reverse (generative) process, with network weights $\theta$ | |
| $\epsilon_1, \dots, \epsilon_T$ | the fresh standard Gaussian noise drawn at each forward step | |
| $\bar\epsilon_t$ | all the noise added in forward steps $1, \dots, t$, combined into one standard Gaussian vector | constructed and proved in "Closed form" below |
| $\epsilon$ | a standard Gaussian vector, $\epsilon \sim \mathcal{N}(0, I)$; in $x_t = \alpha_t x_0 + \sigma_t\epsilon$ it is the combined noise $\bar\epsilon_t$ of the row above | |
| $z$ | a fresh standard Gaussian vector drawn inside one sampler or SDE step | in the latent-diffusion section, $z_0$ and $z_t$ instead mean a clean and a noisy latent |
| $\beta_t$ | variance of the noise added by forward step $t$ | linear schedule: $10^{-4}$ to $0.02$ |
| $\bar\alpha_t = \prod_{s \le t}(1-\beta_s)$ | fraction of the signal's variance left at level $t$ | $\bar\alpha_0 = 1$ |
| $\alpha_t$, $\sigma_t$ | signal scale and noise scale, $x_t = \alpha_t x_0 + \sigma_t\epsilon$ | DDPM: $\alpha_t = \sqrt{\bar\alpha_t}$, $\sigma_t = \sqrt{1-\bar\alpha_t}$ |
| variance preserving (VP) | any schedule with $\alpha_t^2 + \sigma_t^2 = 1$ | DDPM, cosine, most latent models |
| $\mathrm{SNR}(t) = \alpha_t^2/\sigma_t^2$ | signal-to-noise ratio at level $t$ | DDPM: $\bar\alpha_t/(1-\bar\alpha_t)$ |
| $\lambda_t = \log\mathrm{SNR}(t)$ | log-SNR (natural log), decreasing in $t$ | |
| $\ell_t = \log(\alpha_t/\sigma_t) = \lambda_t/2$ | half the log-SNR, the time variable of DPM-Solver | that paper calls it $\lambda_t$ |
| $\tilde\mu_t$, $\tilde\beta_t$ | mean and variance of the forward posterior $q(x_{t-1} \mid x_t, x_0)$ | |
| $\mu_\theta$, $\varsigma_t^2$ | mean and variance of a learned reverse step $p_\theta(x_{t-1} \mid x_t)$ | DDPM and DDIM write $\sigma_t^2$; renamed here to avoid a clash |
| $\epsilon_\theta(x_t, t)$, short $\hat\epsilon$ or $\hat\epsilon_\theta$ | the network's estimate of the noise in $x_t$ | |
| $\hat x_0$, also $\hat x_\theta(x_t, t)$ | the clean sample implied by the network's output | $\hat x_\theta$ stresses that it is a function computed by the network |
| $v = \alpha_t\epsilon - \sigma_t x_0$, $\hat v$ | the "velocity" target and its estimate | |
| $p_t$ | the density of $x_t$ (data pushed through the forward process) | |
| $\nabla\log p_t(x)$, $s_\theta(x, t)$ | the score of $p_t$ and a network that estimates it | gradient with respect to $x$ |
| $\sigma$ (no subscript), $x = x_0 + \sigma\epsilon$ | variance-exploding (VE) noise level and sample, used by EDM | $\sigma = \sigma_t/\alpha_t$ and $x = x_t/\alpha_t$ |
| $D(x; \sigma)$ | a denoiser: an estimate of $x_0$ from the VE input | |
| $\sigma_{\text{data}}$ | standard deviation of the data, per pixel | EDM uses $0.5$ |
| $f(x, t)$, $g(t)$ | drift and diffusion coefficient of a forward SDE | |
| $W_t$, $\bar W_t$ | standard Brownian motion forward in time, and backward in time | |
| $c$, $\varnothing$ | a condition (class label, text) and the null condition | |
| $\epsilon_c$, $\epsilon_u$ | the noise predictions $\epsilon_\theta(x_t, c)$ with the condition and $\epsilon_\theta(x_t, \varnothing)$ without it ("unconditional") | |
| $w$ | guidance weight in Ho and Salimans' convention: $w = 0$ means no guidance | libraries use the scale $s_{\text{cfg}} = 1 + w$ |
| $w(\lambda)$, $w_t$ | loss weight per noise level | always written with an argument or subscript |
| $\eta$ | DDIM's noise knob: $\eta = 0$ deterministic, $\eta = 1$ DDPM | |
| $\mathcal{N}(x; m, vI)$ | the Gaussian density with mean $m$ and covariance $vI$, evaluated at $x$ | |
| $\mathrm{KL}(q \,\Vert\, p)$ | Kullback-Leibler divergence $\mathbb{E}_q[\log q - \log p]$ | |

::: pitfall
Reading papers with the wrong $\alpha$. DDPM's $\alpha_t$ is $1-\beta_t$; DDIM's $\alpha_t$ is DDPM's $\bar\alpha_t$; VDM, the CFG paper and DPM-Solver use $\alpha_t$ for the signal scale $\sqrt{\bar\alpha_t}$, which is this page's convention. Before using any formula from a paper, check which one it means.
:::

## Twelve tools

The derivations below cite these facts by number; everything else they use is ordinary algebra and calculus (Taylor expansion, the chain rule, integration by parts). Each fact is stated once with a short reason.

**Tool 1 (linear-Gaussian marginalization).** If $x \sim \mathcal{N}(\mu, \Sigma)$ and $y \mid x \sim \mathcal{N}(Ax + b, \Lambda)$, then

$$
y \sim \mathcal{N}\big(A\mu + b,\ \Lambda + A\Sigma A^\top\big).
$$

Reason: write $y = Ax + b + \xi$ with $\xi \sim \mathcal{N}(0, \Lambda)$ independent of $x$. A linear function of independent Gaussians is Gaussian, means add, and the covariances of independent parts add. In the isotropic case used on this page ($A = aI$, $\Sigma = vI$, $\Lambda = uI$): mean $a\mu + b$, covariance $(u + a^2 v)I$.

**Tool 2 (Gaussian conditioning).** If $u$ and $v$ are jointly Gaussian with means $m_u, m_v$ and covariance blocks $C$, then

$$
u \mid v \sim \mathcal{N}\big(m_u + C_{uv}C_{vv}^{-1}(v - m_v),\ C_{uu} - C_{uv}C_{vv}^{-1}C_{vu}\big).
$$

Reason: this is Bayes' rule for Gaussians; completing the square (Tool 6) in the joint density gives the same result. In one dimension, $C_{uv}C_{vv}^{-1}$ is the regression slope $\mathrm{Cov}(u, v)/\mathrm{Var}(v)$.

**Tool 3 (KL between Gaussians).** In $d$ dimensions,

$$
\mathrm{KL}\big(\mathcal{N}(\mu_1, \Sigma_1)\,\big\Vert\,\mathcal{N}(\mu_2, \Sigma_2)\big) = \tfrac{1}{2}\Big[\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^\top\Sigma_2^{-1}(\mu_2 - \mu_1) - d + \ln\frac{\det\Sigma_2}{\det\Sigma_1}\Big].
$$

With $\Sigma_1 = a I$ and $\Sigma_2 = b I$ this is $\tfrac{1}{2}\big[d\,a/b + \lVert\mu_1 - \mu_2\rVert^2/b - d + d\ln(b/a)\big]$. If only the means depend on a parameter, every term except $\lVert\mu_1 - \mu_2\rVert^2/(2b)$ is a constant in that parameter, and if $a = b$ those constants are zero.

**Tool 4 (the MSE minimizer is the conditional mean).** For random $X, Y$, $\arg\min_f \mathbb{E}\lVert Y - f(X)\rVert^2 = \mathbb{E}[Y \mid X]$. Reason: add and subtract $\mathbb{E}[Y \mid X]$ inside the norm. The cross term has zero mean by the tower rule, so $\mathbb{E}\lVert Y - f(X)\rVert^2 = \mathbb{E}\lVert Y - \mathbb{E}[Y \mid X]\rVert^2 + \mathbb{E}\lVert \mathbb{E}[Y \mid X] - f(X)\rVert^2$, and only the second term depends on $f$. The first term is the loss floor that no model can beat.

**Tool 5 (the score of a Gaussian-smoothed density).** If $x = \alpha x_0 + \sigma\epsilon$ with $x_0 \sim p_0$ independent of $\epsilon \sim \mathcal{N}(0, I)$, then $\nabla\log p(x) = -\mathbb{E}[\epsilon \mid x]/\sigma$. This is Tweedie's formula; it is proved line by line in the score matching section below.

**Tool 6 (completing the square).** For a scalar $y$ and $A > 0$,

$$
-\tfrac{1}{2}\big(A y^2 - 2by\big) = -\tfrac{1}{2}A\Big(y - \frac{b}{A}\Big)^2 + \frac{b^2}{2A},
$$

so a density proportional to $\exp\big(-\tfrac{1}{2}(Ay^2 - 2by)\big)$ is $\mathcal{N}(b/A, 1/A)$. Reason: expand $A(y - b/A)^2 = Ay^2 - 2by + b^2/A$. The same holds for vectors with $\lVert y\rVert^2$ and $b^\top y$ in place of $y^2$ and $by$, giving $\mathcal{N}(b/A, I/A)$. In words: if a log-density is quadratic in $y$, the precision is the coefficient of $-\tfrac{1}{2}y^2$ and the mean is the coefficient of $y$ divided by the precision.

**Tool 7 (the score of a Gaussian).** $\nabla_x\log\mathcal{N}(x; \mu, s^2 I) = -(x - \mu)/s^2$. Reason: $\log\mathcal{N}(x; \mu, s^2 I) = -\lVert x - \mu\rVert^2/(2s^2) + \text{const}$, and the gradient of $\lVert x - \mu\rVert^2$ is $2(x - \mu)$. The score points from $x$ back toward the mean, and its length grows with the distance.

**Tool 8 (Bayes' rule for scores).** $\nabla_x\log p(x \mid c) = \nabla_x\log p(x) + \nabla_x\log p(c \mid x)$. Reason: $p(x \mid c) = p(c \mid x)\,p(x)/p(c)$; take logs, and $\log p(c)$ does not depend on $x$, so its gradient is zero. More generally, normalizing constants vanish under $\nabla_x\log$, which is why scores are convenient targets.

**Tool 9 (Jensen's inequality for the log).** For a positive random variable $Y$, $\log\mathbb{E}[Y] \ge \mathbb{E}[\log Y]$, with equality only when $Y$ is almost surely constant. Reason: $\log$ is concave.

**Tool 10 (three calculus identities).** For a positive differentiable density $p$ and a vector field $u$ on $\mathbb{R}^d$:

- (a) $\nabla p = p\,\nabla\log p$, because $\nabla\log p = \nabla p/p$.
- (b) $\nabla\cdot(p\,u) = p\,\nabla\cdot u + u\cdot\nabla p$, the product rule for the divergence $\nabla\cdot u = \sum_i \partial u_i/\partial x_i$.
- (c) $\nabla_x\int h(x, y)\,dy = \int\nabla_x h(x, y)\,dy$ when $h$ is smooth and its gradient is integrable, which holds for the Gaussian kernels used here.

Combining (a) with $\Delta p = \nabla\cdot\nabla p$ gives $\Delta p = \nabla\cdot(p\,\nabla\log p)$, where $\Delta$ is the Laplacian.

**Tool 11 (the Fokker-Planck equation).** If $x$ follows the SDE $dx = f(x, t)\,dt + g(t)\,dW_t$, its density $p_t$ satisfies

$$
\partial_t p_t(x) = -\nabla\cdot\big(f(x, t)\,p_t(x)\big) + \tfrac{1}{2}g(t)^2\,\Delta p_t(x).
$$

Intuition: the first term moves probability mass along the velocity field $f$ without creating or destroying it (a continuity equation); the second spreads mass like heat at rate $\tfrac{1}{2}g^2$. A one-line check: with $f = 0$ and constant $g$ in one dimension, $x_t = x_0 + g\sqrt{t}\,z$, so the variance grows by $g^2 t$, and the heat equation $\partial_t p = \tfrac{1}{2}g^2 p''$ is solved by a Gaussian whose variance grows at exactly that rate. With $g = 0$ the equation is the continuity equation of the ODE $dx/dt = f$. This tool is used as a known result, not proved here.

**Tool 12 (variation of constants).** The linear ODE $dx/dt = a(t)\,x + b(t)$ has the solution

$$
x(t) = e^{\int_s^t a(r)\,dr}\,x(s) + \int_s^t e^{\int_\tau^t a(r)\,dr}\,b(\tau)\,d\tau .
$$

Reason: multiply the ODE by the integrating factor $e^{-\int_s^t a}$. The left side becomes $\frac{d}{dt}\big[e^{-\int_s^t a}x(t)\big]$ by the product rule, and the right side is $e^{-\int_s^t a}b(t)$. Integrate from $s$ to $t$ and multiply by $e^{\int_s^t a}$.

## The forward process

### The forward chain

The forward process is a fixed Markov chain that adds a small amount of Gaussian noise at each step and shrinks the signal so the variance stays bounded:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\big).
$$

Nothing is learned here. DDPM uses $T = 1000$ and $\beta_t$ rising linearly from $10^{-4}$ to $0.02$.

### Closed form

Each forward step draws its own fresh noise. Written as a sampling rule,

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

### The posterior given $x_0$

The true reverse marginal $q(x_{t-1} \mid x_t)$ depends on the unknown $p_{\text{data}}$, but once we also condition on $x_0$ everything is Gaussian. By Bayes' rule (the Markov property lets us drop $x_0$ from the first factor),

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} \;\propto\; q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0) \quad \text{(as a function of } x_{t-1}) .
$$

Take logs and keep only the terms that depend on $x_{t-1}$:

$$
\log q(x_{t-1} \mid x_t, x_0) = -\frac{1}{2}\left[\frac{\lVert x_t - \sqrt{1-\beta_t}\, x_{t-1}\rVert^2}{\beta_t} + \frac{\lVert x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\, x_0\rVert^2}{1-\bar{\alpha}_{t-1}}\right] + \text{const}.
$$

Expand both squares with $\lVert a - b\rVert^2 = \lVert a\rVert^2 - 2a^\top b + \lVert b\rVert^2$, and put every term without $x_{t-1}$ into the constant:

$$
\begin{aligned}
\frac{\lVert x_t - \sqrt{1-\beta_t}\, x_{t-1}\rVert^2}{\beta_t} &= \frac{1-\beta_t}{\beta_t}\lVert x_{t-1}\rVert^2 - 2\,\frac{\sqrt{1-\beta_t}}{\beta_t}\,x_t^\top x_{t-1} + \text{const},\\
\frac{\lVert x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\, x_0\rVert^2}{1-\bar{\alpha}_{t-1}} &= \frac{1}{1-\bar{\alpha}_{t-1}}\lVert x_{t-1}\rVert^2 - 2\,\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\,x_0^\top x_{t-1} + \text{const}.
\end{aligned}
$$

Adding the two lines, the bracket has the form $A\lVert x_{t-1}\rVert^2 - 2\,b^\top x_{t-1} + \text{const}$ with

$$
A = \frac{1-\beta_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}, \qquad b = \frac{\sqrt{1-\beta_t}}{\beta_t}\, x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\, x_0 .
$$

By Tool 6 (vector form), a log-density $-\tfrac12\big(A\lVert x_{t-1}\rVert^2 - 2\,b^\top x_{t-1}\big) + \text{const}$ is a Gaussian with variance $1/A$ and mean $b/A$. Over a common denominator, and using the identity proved above,

$$
A = \frac{(1-\beta_t)(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t\,(1-\bar{\alpha}_{t-1})} = \frac{1-\bar{\alpha}_t}{\beta_t\,(1-\bar{\alpha}_{t-1})} .
$$

So the variance is $\tilde{\beta}_t = 1/A$, and the mean is $\tilde{\beta}_t\, b$, which we multiply out term by term:

$$
\begin{aligned}
q(x_{t-1} \mid x_t, x_0) &= \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I), \qquad \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t, \\
\tilde{\mu}_t &= \tilde{\beta}_t\, b = \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}\Big(\frac{\sqrt{1-\beta_t}}{\beta_t}\, x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\, x_0\Big) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\, x_0 + \frac{\sqrt{1-\beta_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\, x_t .
\end{aligned}
$$

In words, the posterior mean is a fixed weighted average of the clean image and the current noisy image. The weights depend only on the schedule, and the variance $\tilde{\beta}_t$ is smaller than the forward step's $\beta_t$ because $\bar{\alpha}_{t-1} > \bar{\alpha}_t$ (at $t = 1$ it is exactly 0, since $x_0$ is then known). The reverse model will try to match this Gaussian without knowing $x_0$.

### Worked example: the linear schedule in numbers

DDPM's schedule is $\beta_t = 10^{-4} + (t-1)\,(0.02 - 10^{-4})/999$ for $t = 1, \dots, 1000$. The table gives the quantities defined above at a few levels (computed in double precision; $\lambda_t$ is the natural log of the SNR).

| $t$ | $\beta_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ | $\mathrm{SNR}(t)$ | $\lambda_t$ | $\tilde\beta_t/\beta_t$ |
|---|---|---|---|---|---|---|
| 1 | $1.000 \times 10^{-4}$ | 0.99990 | 0.99995 | 9999 | 9.21 | 0 |
| 2 | $1.199 \times 10^{-4}$ | 0.99978 | 0.99989 | 4546 | 8.42 | 0.455 |
| 10 | $2.793 \times 10^{-4}$ | 0.99811 | 0.99905 | 527 | 6.27 | 0.853 |
| 100 | $2.072 \times 10^{-3}$ | 0.8970 | 0.9471 | 8.71 | 2.16 | 0.982 |
| 250 | $5.060 \times 10^{-3}$ | 0.5241 | 0.7239 | 1.10 | 0.10 | 0.994 |
| 500 | $1.004 \times 10^{-2}$ | 0.0786 | 0.2803 | 0.0853 | $-2.46$ | 0.9991 |
| 750 | $1.502 \times 10^{-2}$ | $3.35 \times 10^{-3}$ | 0.0579 | $3.36 \times 10^{-3}$ | $-5.70$ | 0.99995 |
| 1000 | $2.000 \times 10^{-2}$ | $4.04 \times 10^{-5}$ | 0.00635 | $4.04 \times 10^{-5}$ | $-10.12$ | 1.0000 |

Three entries you can check by hand:

- *$t = 1$ and $t = 2$.* $\bar\alpha_1 = 1 - \beta_1 = 0.9999$. Then $\beta_2 = 10^{-4} + 0.0199/999 = 1.1992 \times 10^{-4}$, so $\bar\alpha_2 = 0.9999 \times (1 - 0.00011992) = 0.99978$. The posterior variance ratio is $\tilde\beta_2/\beta_2 = (1-\bar\alpha_1)/(1-\bar\alpha_2) = 0.0001/0.00022 = 0.455$, and $\tilde\beta_1 = 0$ because $1 - \bar\alpha_0 = 0$.
- *$t = 1000$.* For small $\beta$, $\log(1-\beta) \approx -\beta - \beta^2/2$. The sum of the $\beta_s$ is $1000 \times (10^{-4} + 0.02)/2 = 10.05$, and the sum of the $\beta_s^2/2$ is about $0.067$, so $\log\bar\alpha_{1000} \approx -10.12$ and $\bar\alpha_{1000} \approx 4.0 \times 10^{-5}$. Hence $x_{1000} = 0.00635\,x_0 + 0.99998\,\epsilon$.
- *Where the SNR crosses 1.* $\mathrm{SNR}(250) = 0.5241/0.4759 = 1.10$; the first step with $\mathrm{SNR} < 1$ is $t = 260$.

**What this says.** Under the linear schedule, roughly the last three quarters of the steps have more noise than signal, and the last quarter has an SNR below $0.004$, which is nearly pure noise. The final level is close to $\mathcal{N}(0, I)$ but not exactly: a fraction $0.00635$ of the image survives, which matters for the zero-terminal-SNR discussion later. The posterior variance $\tilde\beta_t$ equals $\beta_t$ to within 2% for every $t \ge 91$ and differs by more than that only in the first 90 steps, which helps explain why DDPM found that the two fixed reverse variances $\beta_t$ and $\tilde\beta_t$ give similar samples.

::: key
**How to say it in an interview.** "The forward process is a fixed Gaussian Markov chain, $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t$, that shrinks the signal and adds noise so the variance stays at one. Because each step is linear-Gaussian, the noise from all steps combines into one Gaussian, so $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ and training can jump to any level in one shot. Conditioned on $x_0$, one step backwards is also Gaussian, with a mean that blends $x_0$ and $x_t$ and a variance $\tilde\beta_t$ slightly below $\beta_t$; that Gaussian is what the learned reverse step imitates."
:::

::: {.qa asked="Mistral AI SWE screen, PracHub 2026"}
**Explain the forward and reverse processes of a diffusion model.**

The forward process is a fixed Gaussian Markov chain that mixes data with more and more noise; it has the closed form $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ and ends near $\mathcal{N}(0, I)$. The reverse process is a learned chain $p_\theta(x_{t-1} \mid x_t)$, modeled as Gaussian because each forward step is small. The network takes $x_t$, the noise level and any conditioning, and predicts the noise (or $x_0$, or $v$); training is an MSE at a random $t$. Sampling starts from pure noise and applies the learned denoising step repeatedly, often with fewer steps than training via DDIM or an ODE solver. A complete answer also names the design choices: noise schedule, prediction target, sampler, and guidance scale, since each changes quality, diversity and cost.
:::

::: {.qa asked="Prenuvo senior ML scientist, Glassdoor 2024"}
**Explain diffusion models and their applications. Why use one instead of a GAN or a VAE?**

Diffusion models are trained with a stable regression loss, so there is no adversarial game and no mode collapse in the GAN sense, and they cover the data distribution well. A VAE decodes in one step with a simple (often factorized Gaussian) likelihood, so its outputs average over plausible details and look blurry; diffusion splits generation into many small steps, each of which a Gaussian models well, so samples are much sharper. The price is sampling cost: tens to hundreds of network evaluations instead of one, which is why fast solvers and distillation matter. Applications include text-to-image and video generation, super-resolution and inpainting (as conditional generation or as a learned prior for inverse problems), editing via inversion, audio, and molecules. See [GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html) for the comparison in depth.
:::

::: qa
**Derive $q(x_t \mid x_0)$ for the DDPM forward process.**

By induction on $t$. For $t = 1$ the claim is the one-step kernel itself, since $\bar\alpha_1 = 1 - \beta_1$. If $x_{t-1} \mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_{t-1}}x_0, (1-\bar\alpha_{t-1})I)$, then $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t$ with fresh independent noise, so by linear-Gaussian marginalization $x_t \mid x_0$ is Gaussian with mean $\sqrt{1-\beta_t}\sqrt{\bar\alpha_{t-1}}\,x_0 = \sqrt{\bar\alpha_t}\,x_0$ and variance $(1-\beta_t)(1-\bar\alpha_{t-1}) + \beta_t = 1 - \bar\alpha_t$. The last identity is one line of algebra: expand the product and the $\beta_t$ terms cancel. The noise in the closed form, $\bar\epsilon_t$, is a fixed weighted combination of all the per-step noises $\epsilon_1, \dots, \epsilon_t$, rescaled to unit variance, not any single step's noise. This is what makes training cheap: one Gaussian draw gives a sample at any level.
:::

## The reverse process and the ELBO

This section derives DDPM's training loss from the likelihood. The plan: write the model, bound its log-likelihood (Step 1), rewrite the bound as a sum of KL divergences between Gaussians (Steps 2 to 4), reduce each KL to a squared distance between means (Step 5), write the mean through the noise (Step 6), and look at the weights that come out (Steps 7 and 8).

### The model, and why each reverse step is Gaussian

The generative model is a Markov chain run backwards, starting from pure noise:

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t), \qquad p(x_T) = \mathcal{N}(0, I), \qquad p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(\mu_\theta(x_t, t),\ \varsigma_t^2 I\big).
$$

Here $\mu_\theta$ is a neural network and $\varsigma_t^2$ is a fixed number per step (DDPM uses $\beta_t$ or $\tilde\beta_t$). The likelihood of a data point is the marginal $p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T}$, which cannot be computed directly.

Why a Gaussian per step? The true reverse step is $q(x_{t-1} \mid x_t) = \int q(x_{t-1} \mid x_t, x_0)\,q(x_0 \mid x_t)\,dx_0$: an average of the Gaussian posteriors from the forward process section, one for each clean image $x_0$ that could have produced $x_t$. Each component has variance $\tilde\beta_t$, and its mean depends on $x_0$ through the coefficient $c_t = \sqrt{\bar\alpha_{t-1}}\,\beta_t/(1-\bar\alpha_t)$. So the mixture is close to a single Gaussian when $c_t$ times the spread of the plausible $x_0$ is small compared with $\sqrt{\tilde\beta_t}$.

For one step of the linear schedule at $t = 500$, $c_{500} = 0.0031$ while $\sqrt{\tilde\beta_{500}} = 0.100$; even if the plausible clean pixels spanned the whole range $[-1, 1]$, the component means would differ by at most $0.006$, a small fraction of the component width. Large jumps are different. A jump from $t$ straight back to an earlier level $s < t$ has the same algebra as one step, with $\bar\alpha_s$ in place of $\bar\alpha_{t-1}$ and $1 - \bar\alpha_t/\bar\alpha_s$ in place of $\beta_t$ (because $x_t = \sqrt{\bar\alpha_t/\bar\alpha_s}\,x_s + \sqrt{1 - \bar\alpha_t/\bar\alpha_s}\,\epsilon$ by the closed form applied from $s$ to $t$). So its coefficient on $x_0$ is $\sqrt{\bar\alpha_s}\,(1 - \bar\alpha_t/\bar\alpha_s)/(1-\bar\alpha_t)$ and its variance is $(1-\bar\alpha_s)(1 - \bar\alpha_t/\bar\alpha_s)/(1-\bar\alpha_t)$. From $t = 500$ straight to $t = 250$ the coefficient on $x_0$ is $0.67$ and the width (square root of the variance) is $0.66$, and from $t = 250$ straight to $t = 1$ the coefficient is $1.00$ and the width is $0.010$: the step back is essentially "draw a clean image from $q(x_0 \mid x_t)$", which is as multimodal as the data.

**What this says.** A Gaussian reverse step is an accurate model only for small steps. This is why ancestral DDPM sampling needs many steps, and why fast samplers either change the update (DDIM, ODE solvers) or change the training (distillation). The practice question "Why is each reverse step modeled as a Gaussian, and when does that assumption fail?" at the end of this section works this out step by step, with a toy you can compute by hand.

### Step 1: the variational bound

Multiply and divide by the forward process inside the integral, then apply Jensen:

$$
\begin{aligned}
\log p_\theta(x_0) &= \log\int p_\theta(x_{0:T})\,dx_{1:T} && \text{marginalize the latents}\\
&= \log\int q(x_{1:T} \mid x_0)\,\frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\,dx_{1:T} && \text{multiply and divide by } q\\
&= \log\mathbb{E}_{q(x_{1:T} \mid x_0)}\Big[\frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\Big] && \text{definition of the expectation}\\
&\ge \mathbb{E}_{q(x_{1:T} \mid x_0)}\big[\log p_\theta(x_{0:T}) - \log q(x_{1:T} \mid x_0)\big] && \text{Tool 9.}
\end{aligned}
$$

Negate it and average over the data to get a loss that upper-bounds the negative log-likelihood:

$$
L := \mathbb{E}_{q}\big[\log q(x_{1:T} \mid x_0) - \log p_\theta(x_{0:T})\big] \ge \mathbb{E}_{q(x_0)}\big[-\log p_\theta(x_0)\big],
$$

where $\mathbb{E}_q$ now averages over $x_0 \sim q(x_0)$ and $x_{1:T} \sim q(x_{1:T} \mid x_0)$. This is the negative ELBO.

### Step 2: expand both chains

Insert the two factorizations, $q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$ and the model above:

$$
L = \mathbb{E}_q\Big[-\log p(x_T) - \sum_{t=1}^{T}\log p_\theta(x_{t-1} \mid x_t) + \sum_{t=1}^{T}\log q(x_t \mid x_{t-1})\Big].
$$

This form is correct but has high variance: every term is evaluated at one sampled trajectory, and the forward factor $q(x_t \mid x_{t-1})$ points in the wrong direction to be compared with the reverse factor $p_\theta(x_{t-1} \mid x_t)$.

### Step 3: turn the forward steps around with Bayes' rule

For $t \ge 2$, the Markov property gives $q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0)$, and Bayes' rule applied to the three variables $x_{t-1}, x_t$ given $x_0$ gives

$$
q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)\,q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)} .
$$

Take logs and sum over $t = 2, \dots, T$:

$$
\sum_{t=2}^{T}\log q(x_t \mid x_{t-1}) = \sum_{t=2}^{T}\log q(x_{t-1} \mid x_t, x_0) + \sum_{t=2}^{T}\big[\log q(x_t \mid x_0) - \log q(x_{t-1} \mid x_0)\big].
$$

The last sum telescopes. Written out, it is $(\log q(x_2 \mid x_0) - \log q(x_1 \mid x_0)) + (\log q(x_3 \mid x_0) - \log q(x_2 \mid x_0)) + \dots + (\log q(x_T \mid x_0) - \log q(x_{T-1} \mid x_0))$, and every middle term cancels, leaving $\log q(x_T \mid x_0) - \log q(x_1 \mid x_0)$. Adding back the $t = 1$ term $\log q(x_1 \mid x_0)$ of the original sum cancels the leftover:

$$
\sum_{t=1}^{T}\log q(x_t \mid x_{t-1}) = \sum_{t=2}^{T}\log q(x_{t-1} \mid x_t, x_0) + \log q(x_T \mid x_0).
$$

### Step 4: group the terms into KL divergences

This step has two parts. Part (a) is bookkeeping: put the result of Step 3 into Step 2 and pair up terms. Part (b) carries the real idea: the expected log-ratio of two densities of the same variable is a KL divergence, provided you average over that variable first.

**Part (a): regroup the terms.** Start from the bracket in Step 2 and make two edits.

1. Replace the forward sum by the result of Step 3: $\sum_{t=1}^{T}\log q(x_t \mid x_{t-1}) = \sum_{t=2}^{T}\log q(x_{t-1} \mid x_t, x_0) + \log q(x_T \mid x_0)$.
2. Split the model sum into its $t = 1$ term and the rest: $\sum_{t=1}^{T}\log p_\theta(x_{t-1} \mid x_t) = \sum_{t=2}^{T}\log p_\theta(x_{t-1} \mid x_t) + \log p_\theta(x_0 \mid x_1)$.

Then collect the terms:

$$
\begin{aligned}
L &= \mathbb{E}_q\Big[-\log p(x_T) - \sum_{t=2}^{T}\log p_\theta(x_{t-1} \mid x_t) - \log p_\theta(x_0 \mid x_1) + \sum_{t=2}^{T}\log q(x_{t-1} \mid x_t, x_0) + \log q(x_T \mid x_0)\Big] && \text{edits 1 and 2}\\
&= \mathbb{E}_q\Big[\big(\log q(x_T \mid x_0) - \log p(x_T)\big) + \sum_{t=2}^{T}\big(\log q(x_{t-1} \mid x_t, x_0) - \log p_\theta(x_{t-1} \mid x_t)\big) - \log p_\theta(x_0 \mid x_1)\Big] && \text{reorder into pairs}\\
&= \mathbb{E}_q\Big[\log\frac{q(x_T \mid x_0)}{p(x_T)} + \sum_{t=2}^{T}\log\frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)} - \log p_\theta(x_0 \mid x_1)\Big] && \log a - \log b = \log\tfrac{a}{b} .
\end{aligned}
$$

The pairing rule is: put together the two terms that are densities of the same variable, the one written in front of the bar. The first pair is about $x_T$. The forward side $q(x_T \mid x_0)$ is "where the forward process puts $x_T$, given the clean image", and the model side $p(x_T)$ is "where sampling starts". Each middle pair, one for every $t = 2, \dots, T$, is about $x_{t-1}$. The forward side $q(x_{t-1} \mid x_t, x_0)$ is the true one-step-back distribution when the clean image is known, and the model side $p_\theta(x_{t-1} \mid x_t)$ is the learned step that must work without $x_0$. The last term, $-\log p_\theta(x_0 \mid x_1)$, has no partner. The forward term that could have paired with it, $\log q(x_1 \mid x_0)$, was used up in Step 3 to cancel the leftover of the telescoping sum.

**Part (b): why each expected log-ratio is a KL.** Recall the definition from the notation table. For two densities $Q$ and $P$ of the same variable $y$,

$$
\mathrm{KL}(Q \,\Vert\, P) = \int Q(y)\log\frac{Q(y)}{P(y)}\,dy = \mathbb{E}_{y \sim Q}\Big[\log\frac{Q(y)}{P(y)}\Big].
$$

Two features of this formula matter. The log-ratio is averaged over the one variable $y$, and the distribution doing the averaging is $Q$, the density in the numerator. In the last line of Part (a) neither feature is visible yet. There, $\mathbb{E}_q$ averages over the whole trajectory $(x_0, x_1, \dots, x_T)$ at once, and nothing singles out one variable. Three moves expose the KL. Fix one $t \in \{2, \dots, T\}$ and call its middle term

$$
f(x_0, x_t, x_{t-1}) = \log\frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)} .
$$

*Move 1: drop the variables that $f$ does not use.* The expectation $\mathbb{E}_q[f]$ is an integral of $f$ against the joint density $q(x_0, x_1, \dots, x_T)$. But $f$ involves only $x_0$, $x_t$ and $x_{t-1}$. Integrate out every other variable first. They appear only in the density, and integrating a joint density over some of its variables leaves the marginal density of the others. So

$$
\mathbb{E}_q[f] = \iiint f(x_0, x_t, x_{t-1})\;q(x_0, x_t, x_{t-1})\;dx_{t-1}\,dx_t\,dx_0 .
$$

*Move 2: factor the three-variable joint in a chosen order.* The chain rule of probability, $p(a, b, c) = p(a)\,p(b \mid a)\,p(c \mid a, b)$, holds for any ordering of the variables. Choose the order $x_0$, then $x_t$, then $x_{t-1}$:

$$
q(x_0, x_t, x_{t-1}) = q(x_0)\;q(x_t \mid x_0)\;q(x_{t-1} \mid x_t, x_0) .
$$

This order is chosen on purpose, so that the last factor is exactly the numerator of $f$. Every factor is something already known. $q(x_0)$ is the data distribution. $q(x_t \mid x_0)$ is the closed form, $\mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$. $q(x_{t-1} \mid x_t, x_0)$ is the Gaussian posterior $\mathcal{N}(\tilde\mu_t(x_t, x_0), \tilde\beta_t I)$ from the forward process section.

*Move 3: do the innermost integral first.* Substitute the factorization and nest the integrals with $x_{t-1}$ innermost. Each factor goes into the innermost integral that involves its own variable:

$$
\mathbb{E}_q[f] = \int q(x_0)\int q(x_t \mid x_0)\,\underbrace{\Big[\int q(x_{t-1} \mid x_t, x_0)\,\log\frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)}\,dx_{t-1}\Big]}_{\text{inner bracket}}\,dx_t\,dx_0 .
$$

Inside the bracket, $x_0$ and $x_t$ are held fixed, and only $x_{t-1}$ varies. Rename things to match the definition:

- $y = x_{t-1}$;
- $Q(y) = q(y \mid x_t, x_0)$, the posterior Gaussian;
- $P(y) = p_\theta(y \mid x_t)$, the model's Gaussian step $\mathcal{N}(\mu_\theta(x_t, t), \varsigma_t^2 I)$.

Both are densities of the same variable $y$. The bracket is $\int Q(y)\log\frac{Q(y)}{P(y)}\,dy$, so it equals $\mathrm{KL}(Q \,\Vert\, P)$ by definition. This KL is a single number for each fixed pair $(x_0, x_t)$. The two outer integrals then average that number over $x_0 \sim q(x_0)$ and $x_t \sim q(x_t \mid x_0)$:

$$
\mathbb{E}_q\Big[\log\frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)}\Big] = \mathbb{E}_{q(x_0,\, x_t)}\Big[\mathrm{KL}\big(q(x_{t-1} \mid x_t, x_0)\,\Vert\,p_\theta(x_{t-1} \mid x_t)\big)\Big].
$$

In probability language, Moves 1 to 3 are the tower rule, $\mathbb{E}[f] = \mathbb{E}\big[\,\mathbb{E}[f \mid x_0, x_t]\,\big]$, and the inner conditional expectation is the KL. This also explains the direction of the KL, with the forward posterior first. The variable $x_{t-1}$ is averaged under $q(x_{t-1} \mid x_t, x_0)$, because that is the conditional law of $x_{t-1}$ in the forward process, and the averaging density is always the first argument of a KL.

*The first term, by the same three moves.* $\log\frac{q(x_T \mid x_0)}{p(x_T)}$ involves only $x_0$ and $x_T$.

- Move 1 reduces the expectation to the pair $(x_0, x_T)$.
- Move 2 factors their joint as $q(x_0)\,q(x_T \mid x_0)$.
- Move 3 fixes $x_0$ and integrates over $x_T$:

$$
\int q(x_T \mid x_0)\log\frac{q(x_T \mid x_0)}{p(x_T)}\,dx_T = \mathrm{KL}\big(q(x_T \mid x_0)\,\Vert\,p(x_T)\big),
$$

a KL between $Q = \mathcal{N}(\sqrt{\bar\alpha_T}\,x_0, (1-\bar\alpha_T)I)$ and $P = \mathcal{N}(0, I)$. The outer integral averages it over $x_0 \sim q(x_0)$.

*The last term stays as it is.* $-\log p_\theta(x_0 \mid x_1)$ contains no $q$ density, so there is no KL to form. Move 1 still applies: it is an expected log-likelihood over the pair $x_0 \sim q(x_0)$, $x_1 \sim q(x_1 \mid x_0)$. It measures how well the final model step reconstructs the clean image from the least noisy one.

**A one-dimensional check that the average of a log-ratio is the KL.** Take $Q = \mathcal{N}(0, 1)$ and $P = \mathcal{N}(1, 1)$.

- *The log-ratio.* Both densities have the same normalizing constant $1/\sqrt{2\pi}$, which cancels in the ratio, so $\log\frac{Q(y)}{P(y)} = -\frac{y^2}{2} + \frac{(y-1)^2}{2} = \frac{1}{2} - y$.
- *Its average under $Q$.* $y$ has mean $0$ under $Q$, so the average is $\frac{1}{2}$.
- *Tool 3 agrees.* With $d = 1$, $a = b = 1$ and $\lVert\mu_1 - \mu_2\rVert^2 = 1$, it gives $\frac{1}{2}[1 + 1 - 1 + \ln 1] = \frac{1}{2}$.
- *Single samples.* $y = 0.8$ gives $-0.3$, a negative value. $y = -1.2$ gives $1.7$. In general the single-sample value $\frac{1}{2} - y$ has standard deviation $1$, twice the KL itself.

So one log-ratio is a noisy number that can even be negative. Only its average under the numerator density is the KL, which is never negative. That is what Part (b) buys. The trajectory form of Step 2, evaluated at one sampled $x_{t-1}$, is like using a single $y$. The KL form does the $x_{t-1}$ average exactly, in closed form.

Putting the three pieces back into $L$ gives DDPM's decomposition:

$$
L = \mathbb{E}_q\Big[\underbrace{\mathrm{KL}\big(q(x_T \mid x_0)\,\Vert\,p(x_T)\big)}_{L_T} + \sum_{t=2}^{T}\underbrace{\mathrm{KL}\big(q(x_{t-1} \mid x_t, x_0)\,\Vert\,p_\theta(x_{t-1} \mid x_t)\big)}_{L_{t-1}}\ \underbrace{-\ \log p_\theta(x_0 \mid x_1)}_{L_0}\Big].
$$

The single $\mathbb{E}_q$ in front is DDPM's shorthand. By Move 1, each term needs only the variables it still contains:

- $L_T$ averages over $x_0$;
- each $L_{t-1}$ averages over the pair $(x_0, x_t)$, because $x_{t-1}$ has already been integrated out inside the KL;
- $L_0$ averages over $(x_0, x_1)$.

**What this says.** The bound is a sum of comparisons, one per step, between the learned reverse step and the Gaussian posterior derived in the forward process section. All three kinds of terms are cheap:

- $L_T$ has no parameters. With the linear schedule it is about $10^{-5}$ bits per dimension, because $q(x_T \mid x_0)$ is already almost $\mathcal{N}(0, I)$.
- Each $L_{t-1}$ is a KL between two Gaussians, available in closed form (Tool 3). DDPM calls this "Rao-Blackwellized": an exact expectation replaces a noisy sample.
- $L_0$ is a reconstruction term. DDPM makes it a discrete decoder: the Gaussian $\mathcal{N}(\mu_\theta(x_1, 1), \varsigma_1^2 I)$ integrated over each pixel's bin of width $2/255$, so it gives a proper discrete likelihood for 8-bit images.

To evaluate $L_{t-1}$ you need only a pair $(x_0, x_t)$, drawn as $x_0 \sim q(x_0)$ and $x_t \sim q(x_t \mid x_0)$ in one shot, plus the formula for the posterior.

::: key
**How to say it in an interview.** "After the Bayes-rule rewrite, each term of the negative ELBO compares two densities of the same variable. For $x_{t-1}$, those are the forward posterior $q(x_{t-1} \mid x_t, x_0)$ and the model step $p_\theta(x_{t-1} \mid x_t)$. If you take the expectation over $x_{t-1}$ first, holding $x_0$ and $x_t$ fixed, the log-ratio averaged under the posterior is exactly their KL. Both are Gaussians, so the KL has a closed form. The loss becomes a parameter-free prior term, a sum over $t$ of Gaussian KLs, and a reconstruction term. Integrating $x_{t-1}$ out exactly instead of sampling it is why DDPM calls the terms Rao-Blackwellized."
:::

### Step 5: each KL is a squared distance between means

Apply Tool 3 with $\Sigma_1 = \tilde\beta_t I$ (the posterior) and $\Sigma_2 = \varsigma_t^2 I$ (the model):

$$
L_{t-1} = \mathbb{E}_{q(x_0, x_t)}\Big[\frac{1}{2\varsigma_t^2}\big\lVert\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\big\rVert^2\Big] + C_t, \qquad C_t = \frac{d}{2}\Big(\frac{\tilde\beta_t}{\varsigma_t^2} - 1 + \ln\frac{\varsigma_t^2}{\tilde\beta_t}\Big).
$$

$C_t$ does not involve $\theta$, and it is exactly zero when $\varsigma_t^2 = \tilde\beta_t$.

### Step 6: write the mean through the noise

The posterior mean from the forward process section is

$$
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t .
$$

The network sees $x_t$ but not $x_0$. Solve the closed form for $x_0$, namely $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$, and substitute:

$$
\begin{aligned}
\tilde\mu_t &= \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\cdot\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}} + \frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t && \text{substitute } x_0\\
&= \frac{\beta_t}{(1-\bar\alpha_t)\sqrt{1-\beta_t}}\big(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\big) + \frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t && \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} = \frac{1}{\sqrt{1-\beta_t}} .
\end{aligned}
$$

Collect the coefficient of $x_t$ over the common denominator $(1-\bar\alpha_t)\sqrt{1-\beta_t}$:

$$
\begin{aligned}
\frac{\beta_t + (1-\beta_t)(1-\bar\alpha_{t-1})}{(1-\bar\alpha_t)\sqrt{1-\beta_t}} &= \frac{1-\bar\alpha_t}{(1-\bar\alpha_t)\sqrt{1-\beta_t}} && \text{the identity proved in "Closed form"}\\
&= \frac{1}{\sqrt{1-\beta_t}} .
\end{aligned}
$$

The coefficient of $\epsilon$ is $-\beta_t\sqrt{1-\bar\alpha_t}/\big((1-\bar\alpha_t)\sqrt{1-\beta_t}\big) = -\beta_t/\big(\sqrt{1-\bar\alpha_t}\sqrt{1-\beta_t}\big)$. Together:

$$
\tilde\mu_t = \frac{1}{\sqrt{1-\beta_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big).
$$

Since $x_t$ is an input, the only unknown is $\epsilon$. DDPM therefore gives the model mean the same form with a network in place of the noise:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\Big).
$$

Equivalently, $\mu_\theta(x_t, t) = \tilde\mu_t(x_t, \hat x_0)$ with $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$: plug the network's clean-image guess into the true posterior mean. Subtract the two means; the $x_t$ terms cancel:

$$
\tilde\mu_t - \mu_\theta = -\frac{\beta_t}{\sqrt{1-\beta_t}\,\sqrt{1-\bar\alpha_t}}\,\big(\epsilon - \epsilon_\theta(x_t, t)\big).
$$

Square, divide by $2\varsigma_t^2$ (Step 5), and write $x_t$ out:

$$
L_{t-1} - C_t = \mathbb{E}_{x_0, \epsilon}\Big[\frac{\beta_t^2}{2\varsigma_t^2(1-\beta_t)(1-\bar\alpha_t)}\big\lVert\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big)\big\rVert^2\Big].
$$

**What this says.** Each term of the bound is a weighted noise-regression loss. The network is trained to output the noise that was mixed into its input, and the "reverse diffusion step" is just a fixed affine function of that output. This is DDPM's eq. (12), with its $\alpha_t$ written as $1-\beta_t$.

### Step 7: what the weights look like

With $\varsigma_t^2 = \tilde\beta_t = (1-\bar\alpha_{t-1})\beta_t/(1-\bar\alpha_t)$, the weight simplifies:

$$
\frac{\beta_t^2}{2\tilde\beta_t(1-\beta_t)(1-\bar\alpha_t)} = \frac{\beta_t^2\,(1-\bar\alpha_t)}{2(1-\bar\alpha_{t-1})\beta_t(1-\beta_t)(1-\bar\alpha_t)} = \frac{\beta_t}{2(1-\bar\alpha_{t-1})(1-\beta_t)} .
$$

Under the linear schedule this weight is $0.600$ at $t = 2$, $0.086$ at $t = 10$, $0.0103$ at $t = 100$, $0.0055$ at $t = 500$ and $0.0102$ at $t = 1000$. The true bound puts about 59 times more weight on the $t = 2$ term than on the $t = 1000$ term, and the large weights sit at the smallest noise levels.

The same weight is more illuminating in units of the clean image. The noise error and the clean-image error are proportional: from $\epsilon = (x_t - \sqrt{\bar\alpha_t}x_0)/\sqrt{1-\bar\alpha_t}$ and the same formula for $\hat\epsilon$ and $\hat x_0$, $\epsilon - \hat\epsilon = -\sqrt{\mathrm{SNR}(t)}\,(x_0 - \hat x_0)$, so $\lVert\epsilon - \hat\epsilon\rVert^2 = \mathrm{SNR}(t)\,\lVert x_0 - \hat x_0\rVert^2$ (derived in full in the parameterization section). Multiply the weight by $\mathrm{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$:

$$
\begin{aligned}
\frac{\beta_t}{2(1-\bar\alpha_{t-1})(1-\beta_t)}\cdot\frac{\bar\alpha_t}{1-\bar\alpha_t} &= \frac{\beta_t\,\bar\alpha_{t-1}}{2(1-\bar\alpha_{t-1})(1-\bar\alpha_t)} && \bar\alpha_t/(1-\beta_t) = \bar\alpha_{t-1}.
\end{aligned}
$$

Now compute the drop in SNR across the step:

$$
\begin{aligned}
\mathrm{SNR}(t-1) - \mathrm{SNR}(t) &= \frac{\bar\alpha_{t-1}}{1-\bar\alpha_{t-1}} - \frac{\bar\alpha_t}{1-\bar\alpha_t} && \text{definition}\\
&= \frac{\bar\alpha_{t-1}(1-\bar\alpha_t) - \bar\alpha_t(1-\bar\alpha_{t-1})}{(1-\bar\alpha_{t-1})(1-\bar\alpha_t)} && \text{common denominator}\\
&= \frac{\bar\alpha_{t-1} - \bar\alpha_t}{(1-\bar\alpha_{t-1})(1-\bar\alpha_t)} && \text{the products } \bar\alpha_{t-1}\bar\alpha_t \text{ cancel}\\
&= \frac{\bar\alpha_{t-1}\,\beta_t}{(1-\bar\alpha_{t-1})(1-\bar\alpha_t)} && \bar\alpha_{t-1} - \bar\alpha_t = \bar\alpha_{t-1}\beta_t .
\end{aligned}
$$

Comparing the two lines,

$$
L_{t-1} = \tfrac{1}{2}\big(\mathrm{SNR}(t-1) - \mathrm{SNR}(t)\big)\,\mathbb{E}\big\lVert x_0 - \hat x_0(x_t, t)\big\rVert^2 \qquad (\varsigma_t^2 = \tilde\beta_t).
$$

Check at $t = 2$: $\tfrac{1}{2}(9999 - 4546.4) = 2726.3$, and $0.5997 \times 4546.4 = 2726.3$.

**What this says.** The ELBO is a sum of clean-image reconstruction errors, each weighted by half the SNR that its step destroys. This is the discrete form of the loss in Variational Diffusion Models (Kingma et al. 2021). Letting the number of steps go to infinity turns the sum into $-\tfrac{1}{2}\int_0^1 \mathrm{SNR}'(t)\,\mathbb{E}\lVert x_0 - \hat x_0\rVert^2\,dt$; changing the variable of integration from $t$ to the SNR shows that the shape of the schedule drops out and only its two endpoints matter. The noise schedule section builds on this.

### Step 8: $L_{\text{simple}}$ drops the weights

Ho et al. train with

$$
L_{\text{simple}} = \mathbb{E}_{t \sim \mathcal{U}\{1, \dots, T\},\ x_0,\ \epsilon}\big\lVert\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big)\big\rVert^2 .
$$

Every step gets weight one, the $t = 1$ term stands in for $L_0$, and $L_T$ is dropped because it has no parameters. Relative to the bound, which weights the smallest noise levels most (Step 7), $L_{\text{simple}}$ down-weights the low-noise terms. The paper's reason: those terms train the network to remove tiny amounts of noise, which is easy, so it is better to spend capacity on the harder denoising tasks at larger $t$. On CIFAR-10 the trade is large. With fixed variances and the noise parameterization, training on the true bound gave FID 13.51, and $L_{\text{simple}}$ gave FID 3.17; the likelihood bound moved the other way, from at most 3.70 to at most 3.75 bits per dimension.

**Fixed versus learned variance.** DDPM fixed $\varsigma_t^2$ to $\beta_t$ or to $\tilde\beta_t$ and saw similar samples; these are the reverse-variance choices that are optimal when the data is $\mathcal{N}(0, I)$ and when it is a single point. Improved DDPM (Nichol and Dhariwal) learns the variance instead. Because $\tilde\beta_t$ and $\beta_t$ are almost equal except near $t = 0$ (the last column of the table above), predicting the variance directly is hard, so the network outputs an interpolation number $r$ per dimension (the paper calls it $v$; renamed here because $v$ is the velocity on this page) and

$$
\Sigma_\theta(x_t, t) = \exp\big(r\log\beta_t + (1-r)\log\tilde\beta_t\big).
$$

It is trained with $L_{\text{hybrid}} = L_{\text{simple}} + 0.001\,L_{\text{vlb}}$, with a stop-gradient on $\mu_\theta$ inside $L_{\text{vlb}}$ so that the bound only steers the variance. The motivation is that the first few steps contribute most of the bound, which is exactly where the variance choice matters. Models trained this way (with $T = 4000$) gave near-optimal FID with 100 sampling steps, where fixed-variance models degraded much more.

::: pitfall
"$L_{\text{simple}}$ is the ELBO." It is a reweighted ELBO. Kingma & Gao show that the common objectives are all the same per-noise-level loss integrated with different weights $w(\lambda)$; only $w(\lambda) = 1$ is the ELBO of the data itself. (They also show that any weight that is monotone in the noise level equals the ELBO of noise-augmented data.)
:::

::: key
**How to say it in an interview.** "Start from the variational bound on $-\log p_\theta(x_0)$. Using Bayes' rule on each forward step, the bound telescopes into a prior term with no parameters, a reconstruction term, and one KL per step between the Gaussian posterior $q(x_{t-1} \mid x_t, x_0)$ and the model step. With a fixed reverse variance, each KL is a squared distance between means. Writing the posterior mean in terms of $x_t$ and the noise turns that into a weighted $\lVert\epsilon - \epsilon_\theta\rVert^2$, whose weight in clean-image units is half the SNR lost in that step. DDPM drops the weights, which is a reweighted bound that de-emphasizes the easy low-noise steps and gives better samples."
:::

::: qa
**The DDPM chain has 1000 latent steps. Why does its loss reduce to a simple MSE on the noise?**

Three structural facts do the work. The forward marginals $q(x_t \mid x_0)$ are Gaussian in closed form, so any $x_t$ is one reparameterized sample away. The forward posterior $q(x_{t-1} \mid x_t, x_0)$ is also Gaussian, so the ELBO splits into per-step KLs between Gaussians. With the reverse variance fixed, each KL is a weighted squared distance between means, and writing the mean in terms of $\epsilon$ turns it into a weighted $\lVert \epsilon - \epsilon_\theta \rVert^2$. Uniform sampling of $t$ gives an unbiased estimate of the sum, so no chain is ever simulated. Dropping the weights gives $L_{\text{simple}}$.
:::

::: qa
**Why is each reverse step modeled as a Gaussian, and when does that assumption fail?**

**Short answer.** One true reverse step is the sum of two independent pieces: a scaled copy of "which clean image was it?" plus Gaussian noise. For a small step the noise piece dominates, so the step is almost exactly Gaussian. For a large jump the clean-image piece dominates, so the step is as multimodal as the data. A Gaussian model then sits at the average of the modes, and the average of several sharp images is a blurry image. The steps below show this with formulas and then with numbers.

**Step 1: the true reverse step is a mixture of Gaussians.** Bring $x_0$ in and integrate it back out. The second equality is the chain rule of probability:

$$
q(x_{t-1} \mid x_t) = \int q(x_{t-1}, x_0 \mid x_t)\,dx_0 = \int \underbrace{q(x_{t-1} \mid x_t, x_0)}_{\text{component}}\;\underbrace{q(x_0 \mid x_t)}_{\text{weight}}\,dx_0 .
$$

The weight $q(x_0 \mid x_t)$ says how plausible each clean image is, given the noisy $x_t$. Each component is the Gaussian posterior derived in "The posterior given $x_0$". Name its two mean coefficients $c_t$ (on $x_0$) and $c_t'$ (on $x_t$):

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(c_t'\,x_t + c_t\,x_0,\ \tilde\beta_t I\big), \qquad c_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}, \quad c_t' = \frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}, \quad \tilde\beta_t = \frac{(1-\bar\alpha_{t-1})\,\beta_t}{1-\bar\alpha_t} .
$$

So every component has the same width $\sqrt{\tilde\beta_t}$. The components differ only in their centers, and a center moves with $x_0$ at rate $c_t$.

**Step 2: write the mixture as a sum of two independent pieces.** To sample from a mixture, first draw a component from the weights, then draw from that component. Here that means:

$$
x_{t-1} = c_t'\,x_t + \underbrace{c_t\,x_0}_{x_0 \,\sim\, q(x_0 \mid x_t)} + \underbrace{\sqrt{\tilde\beta_t}\,z}_{z \,\sim\, \mathcal{N}(0, I)}, \qquad z \text{ independent of } x_0 .
$$

For independent pieces, means add and covariances add, as in Tool 1. So

$$
\mathbb{E}[x_{t-1} \mid x_t] = c_t'\,x_t + c_t\,\mathbb{E}[x_0 \mid x_t], \qquad \mathrm{Cov}[x_{t-1} \mid x_t] = \tilde\beta_t I + c_t^2\,\mathrm{Cov}[x_0 \mid x_t] .
$$

The shape of the sum depends on which piece is larger. Compare the two spreads, per pixel:

$$
\underbrace{c_t \cdot \operatorname{std}(x_0 \mid x_t)}_{\text{spread from not knowing } x_0} \qquad\text{versus}\qquad \underbrace{\sqrt{\tilde\beta_t}}_{\text{spread of the Gaussian noise}} .
$$

- **Left side much smaller.** The Gaussian noise swamps the clean-image piece. The sum is almost exactly $\mathcal{N}\big(\mathbb{E}[x_{t-1} \mid x_t],\ \tilde\beta_t I\big)$, so a Gaussian model loses almost nothing.
- **Left side much larger.** The noise is only a thin blur on a scaled copy of $q(x_0 \mid x_t)$. The step looks like the distribution over clean images, which has a separate mode for each plausible image.

The criterion uses $\operatorname{std}(x_0 \mid x_t)$, not the range of the data. Late in sampling, at small $t$, $c_t$ is large: $c_{10} = 0.15$ against $\sqrt{\tilde\beta_{10}} = 0.015$. But there $x_t$ is only about $0.04$ of noise away from $x_0$. So $\operatorname{std}(x_0 \mid x_t)$ is about that small, $c_t \cdot \operatorname{std} \approx 0.15 \times 0.04 = 0.006$, and the step stays close to Gaussian.

**Step 3: a large jump is the same formula with different coefficients.** A fast sampler jumps from level $t$ straight to an earlier level $s < t$. The closed form applied from $s$ to $t$ is $x_t = \sqrt{\bar\alpha_t/\bar\alpha_s}\,x_s + \sqrt{1-\bar\alpha_t/\bar\alpha_s}\,\epsilon$. So the jump is "one step" whose noise variance is $\beta' = 1 - \bar\alpha_t/\bar\alpha_s$, starting from level $s$. Redo the posterior with $\bar\alpha_s$ in place of $\bar\alpha_{t-1}$ and $\beta'$ in place of $\beta_t$:

$$
q(x_s \mid x_t, x_0) = \mathcal{N}\big(c_{t \to s}'\,x_t + c_{t \to s}\,x_0,\ \tilde\beta_{t \to s} I\big), \qquad c_{t \to s} = \frac{\sqrt{\bar\alpha_s}\,\beta'}{1-\bar\alpha_t}, \quad \tilde\beta_{t \to s} = \frac{(1-\bar\alpha_s)\,\beta'}{1-\bar\alpha_t} .
$$

Here $c_{t \to s}' = \sqrt{1-\beta'}\,(1-\bar\alpha_s)/(1-\bar\alpha_t)$. With $s = t - 1$, $\beta' = \beta_t$ and you get back Step 1. Now push $s$ toward the clean end. Then $\bar\alpha_s \to 1$, so $\beta' \to 1 - \bar\alpha_t$, which gives $c_{t \to s} \to 1$ and $\tilde\beta_{t \to s} \to 0$. The jump becomes $x_s \approx x_0$ with $x_0 \sim q(x_0 \mid x_t)$: "draw a clean image that fits $x_t$", which is exactly as multimodal as the data.

**Step 4: a toy you can compute by hand.** Let each pixel of the clean data be $-1$ or $+1$ with probability $\tfrac{1}{2}$, so there are two "images". Look at the noisy value $x_t = 0$. It sits halfway between them, so by symmetry $q(x_0 = +1 \mid x_t = 0) = q(x_0 = -1 \mid x_t = 0) = \tfrac{1}{2}$. Because $x_t = 0$, the $c'$ term vanishes, and Step 1 gives

$$
q(x_s \mid x_t = 0) = \tfrac{1}{2}\,\mathcal{N}\big(+c,\ \tilde\beta\big) + \tfrac{1}{2}\,\mathcal{N}\big(-c,\ \tilde\beta\big),
$$

with $c$ and $\tilde\beta$ the coefficients of the step or jump in question. This is two bumps of width $\sqrt{\tilde\beta}$, centered $2c$ apart. When is it one bump or two? Each Gaussian $\mathcal{N}(\pm c, \tilde\beta)$ has second derivative $\mathcal{N}(0; \pm c, \tilde\beta)\,(c^2 - \tilde\beta)/\tilde\beta^2$ at $0$. So the middle of the mixture turns from a peak into a dip exactly when $c > \sqrt{\tilde\beta}$. That makes $c/\sqrt{\tilde\beta}$ the number to look at. With the linear schedule:

| Transition | $c$ | width $\sqrt{\tilde\beta}$ | $c/\sqrt{\tilde\beta}$ | Shape of $q(x_s \mid x_t = 0)$ |
|---|---|---|---|---|
| one step, $t = 500 \to 499$ | $0.0031$ | $0.100$ | $0.03$ | one bump; the variance is only 0.09% above $\tilde\beta$, so it is Gaussian to the eye |
| jump, $1000 \to 500$ | $0.28$ | $0.96$ | $0.29$ | one bump, close to Gaussian |
| jump, $500 \to 250$ | $0.67$ | $0.66$ | $1.01$ | flat top, right at the point of splitting in two |
| jump, $250 \to 1$ | $1.00$ | $0.010$ | $100$ | two sharp spikes at $\pm 1$, 200 widths apart |

The value $x_t = 0$ is a typical one at these levels, not a corner case. At $t = 500$ the two possible noise-free values of $x_t$ are $\pm 0.28$, and the noise has standard deviation $0.96$. At $t = 250$ they are $\pm 0.72$, with noise standard deviation $0.69$.

**Step 5: what a Gaussian model does with two spikes.** The model is $p_\theta(x_s \mid x_t) = \mathcal{N}(\mu_\theta, \varsigma^2 I)$ with a fixed variance. Step 5 of the ELBO derivation shows that training minimizes $\mathbb{E}\lVert\tilde\mu - \mu_\theta\rVert^2/(2\varsigma^2)$. By Tool 4, the best mean is the conditional mean:

$$
\mu^* = \mathbb{E}[\tilde\mu \mid x_t] = c'\,x_t + c\,\mathbb{E}[x_0 \mid x_t],
$$

which is the mean of the mixture. In the toy with the jump $250 \to 1$, $\mu^* = 0$: exactly between the two spikes, where the true step has essentially no mass.

- *With the matched width* $\varsigma = \sqrt{\tilde\beta} = 0.01$, every sample lands within about $\pm 0.03$ of $0$. That is a gray value halfway between the two real pixel values, never $-1$ or $+1$.
- *With the moment-matched variance* $\tilde\beta + c^2 \approx 1.0$, the samples spread out. But 38% of them land in $|x| < 0.5$, where the true step has none, and many land beyond $\pm 1$.

Either way the Gaussian step cannot pick a mode. For real images, $\mathbb{E}[x_0 \mid x_t]$ at high noise is the average of all clean images consistent with $x_t$, and that average is blurry.

**Conclusion.** Small steps keep $c_t \cdot \operatorname{std}(x_0 \mid x_t)$ below $\sqrt{\tilde\beta_t}$, so Gaussian steps are accurate. This is why ancestral DDPM sampling needs hundreds of steps. Few large jumps push $c \to 1$ and $\sqrt{\tilde\beta} \to 0$, so the true step becomes "pick one clean image". A Gaussian step can only return the blurry average. That is why naive few-step sampling degrades, and why few-step generators change the training instead, with distillation or adversarial losses (covered on the [flow matching](flow-matching.html) page).
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

::: qa
**Derive the DDPM ELBO and its decomposition into $L_T$, $L_{t-1}$ and $L_0$.**

Jensen's inequality on $\log\int q(x_{1:T} \mid x_0)\,\frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\,dx_{1:T}$ gives $-\log p_\theta(x_0) \le \mathbb{E}_q[\log q(x_{1:T} \mid x_0) - \log p_\theta(x_{0:T})]$. Expand both chains into products. For $t \ge 2$, rewrite each forward factor with the Markov property and Bayes' rule as $q(x_{t-1} \mid x_t, x_0)\,q(x_t \mid x_0)/q(x_{t-1} \mid x_0)$; the ratio of marginals telescopes across $t$ to $q(x_T \mid x_0)/q(x_1 \mid x_0)$, and the denominator cancels the $t = 1$ forward factor. Pairing terms gives $\mathrm{KL}(q(x_T \mid x_0) \Vert p(x_T))$, which has no parameters, the per-step terms $\mathrm{KL}(q(x_{t-1} \mid x_t, x_0) \Vert p_\theta(x_{t-1} \mid x_t))$, and the decoder term $-\log p_\theta(x_0 \mid x_1)$. Each log-ratio becomes a KL because, under $q$, the conditional law of $x_{t-1}$ given $(x_t, x_0)$ is exactly the posterior in the numerator. The point of the rewrite is variance reduction: every per-step term compares two Gaussians in closed form.
:::

::: qa
**In units of the clean image, what weight does the true ELBO put on each noise level?**

With the reverse variance $\tilde\beta_t$, the per-step term is $\frac{\beta_t}{2(1-\bar\alpha_{t-1})(1-\beta_t)}\mathbb{E}\lVert\epsilon - \hat\epsilon\rVert^2$. Since $\lVert\epsilon - \hat\epsilon\rVert^2 = \mathrm{SNR}(t)\lVert x_0 - \hat x_0\rVert^2$, this equals $\tfrac{1}{2}(\mathrm{SNR}(t-1) - \mathrm{SNR}(t))\,\mathbb{E}\lVert x_0 - \hat x_0\rVert^2$: half the SNR destroyed by the step. In continuous time it becomes $-\tfrac{1}{2}\int\mathrm{SNR}'(t)\,\mathbb{E}\lVert x_0 - \hat x_0\rVert^2\,dt$, which after a change of variables depends on the schedule only through its endpoints (VDM). In noise units this is a constant weight per unit of log-SNR, which is what "$w(\lambda) = 1$ is the ELBO" means. The practical consequence: the ELBO spends most of its weight on very low noise levels, where errors are invisible, which is why sample-quality objectives reweight.
:::

## Score matching and Tweedie's formula

The ELBO view says "the network predicts the noise". The score view says "the network estimates the gradient of the log-density of noisy data". This section proves they are the same statement, and shows exactly what the trained network computes.

### The score, and why it is a convenient target

The score of a density $p$ is $\nabla_x\log p(x)$, a vector field that points toward higher density. Two properties make it attractive. First, it ignores normalization: if $p = \tilde p/Z$, then $\nabla\log p = \nabla\log\tilde p$ because $\log Z$ is a constant (Tool 8's argument), so a model of the score never needs the intractable $Z$. Second, the score is all a sampler needs: Langevin dynamics, the reverse SDE and the probability-flow ODE below are all driven by it.

The natural objective, explicit score matching, compares a model $s(x)$ with the true score:

$$
J_{\text{ESM}}(s) = \mathbb{E}_{x \sim p}\big\lVert s(x) - \nabla\log p(x)\big\rVert^2 .
$$

It cannot be computed, because $\nabla\log p$ is the unknown. An integration-by-parts trick removes it but leaves the trace of the Jacobian of $s$, which costs $d$ backward passes. Denoising score matching avoids both problems for noisy data.

### Denoising score matching (Vincent, 2011)

Corrupt clean data $x \sim p(x)$ with a known kernel $q(\tilde x \mid x)$, and call the marginal of the corrupted variable $p_\sigma(\tilde x) = \int q(\tilde x \mid x)\,p(x)\,dx$. Define

$$
J_{\text{ESM}}(s) = \mathbb{E}_{p_\sigma(\tilde x)}\big\lVert s(\tilde x) - \nabla\log p_\sigma(\tilde x)\big\rVert^2, \qquad J_{\text{DSM}}(s) = \mathbb{E}_{p(x)\,q(\tilde x \mid x)}\big\lVert s(\tilde x) - \nabla_{\tilde x}\log q(\tilde x \mid x)\big\rVert^2 .
$$

**Claim.** $J_{\text{ESM}}(s) = J_{\text{DSM}}(s) + C$, where $C$ does not depend on $s$. So both objectives have the same minimizer, and the second one only needs the gradient of the known kernel.

**Proof.** Expand both squares, $\lVert a - b\rVert^2 = \lVert a\rVert^2 - 2a^\top b + \lVert b\rVert^2$, and compare the three kinds of terms.

*The $\lVert s\rVert^2$ terms are equal.* In $J_{\text{DSM}}$ the term is $\mathbb{E}_{p(x)q(\tilde x \mid x)}\lVert s(\tilde x)\rVert^2$. The integrand depends only on $\tilde x$, and the marginal of $\tilde x$ under $p(x)\,q(\tilde x \mid x)$ is $p_\sigma$ by definition, so this is $\mathbb{E}_{p_\sigma}\lVert s(\tilde x)\rVert^2$, the term in $J_{\text{ESM}}$.

*The cross terms are equal.* Line by line:

$$
\begin{aligned}
\mathbb{E}_{p_\sigma}\big[s(\tilde x)^\top\nabla\log p_\sigma(\tilde x)\big] &= \int p_\sigma(\tilde x)\,s(\tilde x)^\top\frac{\nabla p_\sigma(\tilde x)}{p_\sigma(\tilde x)}\,d\tilde x && \text{Tool 10(a)}\\
&= \int s(\tilde x)^\top\,\nabla_{\tilde x}\Big[\int q(\tilde x \mid x)\,p(x)\,dx\Big]\,d\tilde x && p_\sigma \text{ cancels; insert its definition}\\
&= \iint s(\tilde x)^\top\,\nabla_{\tilde x}q(\tilde x \mid x)\,p(x)\,dx\,d\tilde x && \text{Tool 10(c)}\\
&= \iint s(\tilde x)^\top\,\nabla_{\tilde x}\log q(\tilde x \mid x)\;q(\tilde x \mid x)\,p(x)\,dx\,d\tilde x && \text{Tool 10(a) for } q\\
&= \mathbb{E}_{p(x)\,q(\tilde x \mid x)}\big[s(\tilde x)^\top\nabla_{\tilde x}\log q(\tilde x \mid x)\big] && \text{definition of the expectation.}
\end{aligned}
$$

*The remaining terms do not involve $s$.* They are $\mathbb{E}_{p_\sigma}\lVert\nabla\log p_\sigma\rVert^2$ in $J_{\text{ESM}}$ and $\mathbb{E}\lVert\nabla_{\tilde x}\log q(\tilde x \mid x)\rVert^2$ in $J_{\text{DSM}}$, so $C = J_{\text{ESM}} - J_{\text{DSM}} = \mathbb{E}_{p_\sigma}\lVert\nabla\log p_\sigma\rVert^2 - \mathbb{E}\lVert\nabla_{\tilde x}\log q(\tilde x \mid x)\rVert^2$. $\square$

**What this says.** Regressing onto the score of the corruption kernel, which is known in closed form, is as good as regressing onto the unknown score of the noisy data distribution, up to a constant. The target $\nabla_{\tilde x}\log q(\tilde x \mid x)$ is noisy (it depends on the particular $x$), but its average over all $x$ that could have produced $\tilde x$ is the true score, which is why the regression works.

### Noise prediction is denoising score matching

For the diffusion kernel $q(x_t \mid x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 I)$, Tool 7 gives the target:

$$
\nabla_{x_t}\log q(x_t \mid x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2} = -\frac{\sigma_t\epsilon}{\sigma_t^2} = -\frac{\epsilon}{\sigma_t} .
$$

So denoising score matching at level $t$ is $\mathbb{E}\lVert s_\theta(x_t, t) + \epsilon/\sigma_t\rVert^2$. Parameterize the score model through a noise predictor, $s_\theta = -\epsilon_\theta/\sigma_t$:

$$
\mathbb{E}\Big\lVert -\frac{\epsilon_\theta}{\sigma_t} + \frac{\epsilon}{\sigma_t}\Big\rVert^2 = \frac{1}{\sigma_t^2}\,\mathbb{E}\big\lVert\epsilon_\theta(x_t, t) - \epsilon\big\rVert^2 .
$$

**What this says.** The DDPM loss at level $t$ equals $\sigma_t^2$ times the denoising score matching loss. The score SDE paper's recommended per-level weight, $w_t \propto 1/\mathbb{E}\lVert\nabla\log q(x_t \mid x_0)\rVert^2 = 1/\mathbb{E}\lVert\epsilon/\sigma_t\rVert^2 = \sigma_t^2/d$ (the paper writes this weight as $\lambda(t)$, which is not the log-SNR), is exactly this factor up to a constant, so "train a score model with DSM" and "train a noise predictor with $L_{\text{simple}}$" are the same procedure written in two units.

### The constant is not zero: the loss has a floor

The constant $C$ is the reason a well-trained diffusion model's loss stays well above zero. Rearranged, the claim says $J_{\text{DSM}}(s) = J_{\text{ESM}}(s) - C$. The best possible score, $s = \nabla\log p_\sigma$, makes $J_{\text{ESM}} = 0$, so the smallest value the DSM loss can reach is $-C$, and $-C \ge 0$ (so $C \le 0$). By Tool 4 this floor is the average conditional variance of the DSM target given the input; for Gaussian noise the target is $-\epsilon/\sigma_t$, so the floor is $\mathbb{E}\big[\lVert\epsilon/\sigma_t - \mathbb{E}[\epsilon \mid x_t]/\sigma_t\rVert^2\big]$. It is positive because the noise in $x_t$ cannot be recovered exactly: many pairs $(x_0, \epsilon)$ produce the same $x_t$. The worked example below puts a number on it.

### Tweedie's formula

**Claim.** For $x_t = \alpha_t x_0 + \sigma_t\epsilon$ with $x_0 \sim p_0$ independent of $\epsilon \sim \mathcal{N}(0, I)$,

$$
\mathbb{E}[x_0 \mid x_t] = \frac{x_t + \sigma_t^2\,\nabla\log p_t(x_t)}{\alpha_t}, \qquad \mathbb{E}[\epsilon \mid x_t] = -\sigma_t\,\nabla\log p_t(x_t).
$$

**Proof.** Write the density of $x_t$ as a Gaussian blur of the data and differentiate:

$$
\begin{aligned}
p_t(x) &= \int p_0(x_0)\,\mathcal{N}(x;\,\alpha_t x_0,\,\sigma_t^2 I)\,dx_0 && \text{definition of } p_t\\
\nabla p_t(x) &= \int p_0(x_0)\,\mathcal{N}(x;\,\alpha_t x_0,\,\sigma_t^2 I)\Big(-\frac{x - \alpha_t x_0}{\sigma_t^2}\Big)\,dx_0 && \text{Tools 10(c) and 7}\\
\nabla\log p_t(x) &= \int\underbrace{\frac{p_0(x_0)\,\mathcal{N}(x;\,\alpha_t x_0,\,\sigma_t^2 I)}{p_t(x)}}_{=\,p(x_0 \mid x_t = x)\ \text{(Bayes)}}\Big(-\frac{x - \alpha_t x_0}{\sigma_t^2}\Big)\,dx_0 && \text{divide by } p_t(x)\\
&= -\frac{x - \alpha_t\,\mathbb{E}[x_0 \mid x_t = x]}{\sigma_t^2} && \text{the integral is a posterior mean.}
\end{aligned}
$$

Multiply by $\sigma_t^2$, move $x$ across and divide by $\alpha_t$ to get the first formula. For the second, $\epsilon = (x_t - \alpha_t x_0)/\sigma_t$ exactly, so $\mathbb{E}[\epsilon \mid x_t] = (x_t - \alpha_t\mathbb{E}[x_0 \mid x_t])/\sigma_t = -\sigma_t\nabla\log p_t(x_t)$ by the last line. $\square$

The second formula is Tool 5. With $\alpha_t = 1$ the first formula is the classical statement of Tweedie's formula (Efron 2011): the posterior mean under Gaussian noise is the observation plus the noise variance times the score of the observed density.

### What the network converges to

Combine Tweedie with Tool 4. The $\epsilon$-loss at level $t$ is an MSE regression of $\epsilon$ on $x_t$, so its minimizer is $\epsilon^*(x_t, t) = \mathbb{E}[\epsilon \mid x_t]$. By Tweedie,

$$
\epsilon^*(x_t, t) = -\sigma_t\,\nabla\log p_t(x_t), \qquad \hat x_0^*(x_t, t) = \mathbb{E}[x_0 \mid x_t] = \frac{x_t - \sigma_t\,\epsilon^*(x_t, t)}{\alpha_t} .
$$

**What this says.** A trained noise predictor is a rescaled score, $s_\theta = -\epsilon_\theta/\sigma_t$, and the clean image it implies is the posterior mean, not a sample. One network trained on all levels is a score model for every noisy marginal $p_t$. Two caveats: this holds only at the optimum, and it is the score of the smoothed marginal $p_t$, not of $p_{\text{data}}$ itself, which may not even have a density if images lie near a low-dimensional manifold.

### Worked example: Gaussian data, where everything has a closed form

Take one-dimensional data $x_0 \sim \mathcal{N}(2, 1)$ and one noise level with $\alpha = 0.6$, $\sigma = 0.8$. This is variance preserving ($0.36 + 0.64 = 1$) with $\mathrm{SNR} = 0.36/0.64 = 0.5625$.

1. *The noisy marginal* (Tool 1): $x_t \sim \mathcal{N}(0.6 \times 2,\ 0.36 \times 1 + 0.64) = \mathcal{N}(1.2, 1)$.
2. *Its score* (Tool 7): $\nabla\log p_t(x) = -(x - 1.2)$. At the observation $x_t = 2.0$, the score is $-0.8$.
3. *Tweedie:* $\mathbb{E}[x_0 \mid x_t = 2] = (2.0 + 0.64 \times (-0.8))/0.6 = 1.488/0.6 = 2.48$.
4. *Direct check by Gaussian conditioning* (Tool 2): $\mathrm{Cov}(x_0, x_t) = \alpha\,\mathrm{Var}(x_0) = 0.6$ and $\mathrm{Var}(x_t) = 1$, so $\mathbb{E}[x_0 \mid x_t] = 2 + 0.6 \times (2.0 - 1.2) = 2.48$. The posterior variance is $1 - 0.6^2 = 0.64$.
5. *The noise:* $\mathbb{E}[\epsilon \mid x_t] = -\sigma \times \text{score} = 0.8 \times 0.8 = 0.64$. Directly, $\mathrm{Cov}(\epsilon, x_t) = \sigma = 0.8$, so $\mathbb{E}[\epsilon \mid x_t] = 0.8 \times 0.8 = 0.64$, with posterior variance $1 - 0.64 = 0.36$.
6. *Consistency:* $\alpha\,\mathbb{E}[x_0 \mid x_t] + \sigma\,\mathbb{E}[\epsilon \mid x_t] = 0.6 \times 2.48 + 0.8 \times 0.64 = 1.488 + 0.512 = 2.0 = x_t$.

The optimal denoiser here is the straight line $\hat x_0(x) = 2 + 0.6\,(x - 1.2)$: it shrinks the observation toward the data mean, more strongly the noisier the input. The minimum achievable losses are the posterior variances: $0.64$ for $x_0$-prediction and $0.36$ for $\epsilon$-prediction. Their ratio is $0.36/0.64 = 0.5625 = \mathrm{SNR}$, a preview of the parameterization section. The denoising score matching loss at the true score equals $0.36/\sigma^2 = 0.5625$, not zero; a Monte Carlo run with two million samples gives $0.5626$.

**What this says.** In this example you can see every claim of the section at once: the score is linear, Tweedie's formula reproduces Bayesian conditioning, the noise estimate is the rescaled score, and the irreducible loss is the posterior variance.

### Why the final samples are sharp

At high noise, many clean images are consistent with $x_t$, and the MSE-optimal prediction is their average, which is blurry. In the example above, as $\sigma \to 1$ the prediction $\hat x_0$ collapses to the data mean 2 for every input. A sampler never outputs that high-noise average directly. Each step moves only part of the way toward it, then either re-noises (ancestral sampling) or follows the ODE, so the next prediction is made from a less noisy input, where the posterior is narrower. The sampler section shows with numbers what happens when this is skipped: one deterministic step from pure noise returns the posterior mean.

::: key
The network cannot recover the exact noise that was added; that is not identifiable from $x_t$. It learns the average noise consistent with $x_t$, which is a rescaled score. One trained denoiser is therefore a score model at every noise level.
:::

::: key
**How to say it in an interview.** "Denoising score matching says that regressing onto the score of the noising kernel, $-\epsilon/\sigma_t$, has the same minimizer as regressing onto the true score of the noisy data, because the cross terms agree after swapping the gradient and the integral. So the $\epsilon$-loss is DSM up to a weight of $\sigma_t^2$. By Tweedie's formula, the MSE-optimal noise predictor is $\mathbb{E}[\epsilon \mid x_t] = -\sigma_t\nabla\log p_t(x_t)$, and the implied clean image is the posterior mean. That is why the loss has a positive floor, why one-step predictions are blurry at high noise, and why one network gives the score at every level."
:::

::: qa
**Why does a network trained to predict noise give you the score?**

MSE regression converges to a conditional expectation, so at the optimum $\epsilon_\theta(x_t, t) = \mathbb{E}[\epsilon \mid x_t]$. Tweedie's formula for Gaussian corruption gives $\mathbb{E}[\epsilon \mid x_t] = -\sigma_t \nabla \log p_t(x_t)$, so $s_\theta = -\epsilon_\theta/\sigma_t$. Equivalently, the $\epsilon$-loss is denoising score matching up to a per-level weight, and DSM equals explicit score matching on $p_t$ up to a constant. Two caveats: this holds only at the optimum, and it gives the score of the noise-smoothed marginal $p_t$, not of $p_{\text{data}}$, which may not have a score at all if the data sit on a low-dimensional manifold.
:::

::: qa
**One-step $x_0$ predictions at high noise look blurry. Why are final samples sharp?**

At high noise many clean images are consistent with $x_t$, and the MSE-optimal prediction is their average, the posterior mean, which is blurry. The sampler does not output that high-noise average directly. Each step moves only a little toward it and then either re-noises (ancestral sampling) or follows the ODE, so the next prediction is made from a less noisy input where the posterior is narrower. The sequence of small, locally accurate steps commits to one mode gradually. Blur reappears when you take too few steps, because each step then trusts a wide posterior mean.
:::

::: qa
**State and prove the denoising score matching identity.**

Let $p_\sigma(\tilde x) = \int q(\tilde x \mid x)\,p(x)\,dx$. Then $\mathbb{E}_{p_\sigma}\lVert s - \nabla\log p_\sigma\rVert^2 = \mathbb{E}_{p(x)q(\tilde x \mid x)}\lVert s - \nabla_{\tilde x}\log q(\tilde x \mid x)\rVert^2 + C$. Expand both squares. The $\lVert s\rVert^2$ terms match because the marginal of $\tilde x$ is $p_\sigma$. For the cross term, $\mathbb{E}_{p_\sigma}[s^\top\nabla\log p_\sigma] = \int s^\top\nabla p_\sigma\,d\tilde x$; substitute $p_\sigma = \int q\,p\,dx$, move the gradient inside, and write $\nabla q = q\,\nabla\log q$ to get $\mathbb{E}_{p(x)q(\tilde x \mid x)}[s^\top\nabla\log q]$. The leftover squared-target terms do not involve $s$. For Gaussian corruption the target is $-\epsilon/\sigma_t$, so with $s = -\epsilon_\theta/\sigma_t$ the DSM loss is $\sigma_t^{-2}\mathbb{E}\lVert\epsilon_\theta - \epsilon\rVert^2$. The constant is $C = \mathbb{E}\lVert\nabla\log p_\sigma\rVert^2 - \mathbb{E}\lVert\nabla_{\tilde x}\log q\rVert^2 \le 0$: at the true score the ESM loss is zero, so the DSM loss stops at $-C \ge 0$, the average conditional variance of the target. That is why the training loss plateaus above zero.
:::

::: qa
**Derive Tweedie's formula.**

Write $p_t(x) = \int p_0(x_0)\,\mathcal{N}(x; \alpha_t x_0, \sigma_t^2 I)\,dx_0$ and differentiate under the integral: the Gaussian's gradient is $-(x - \alpha_t x_0)/\sigma_t^2$ times the Gaussian. Divide by $p_t(x)$; the weight $p_0(x_0)\mathcal{N}(x; \alpha_t x_0, \sigma_t^2 I)/p_t(x)$ is the posterior $p(x_0 \mid x_t = x)$ by Bayes' rule. So $\nabla\log p_t(x) = -(x - \alpha_t\mathbb{E}[x_0 \mid x])/\sigma_t^2$, which rearranges to $\mathbb{E}[x_0 \mid x_t] = (x_t + \sigma_t^2\nabla\log p_t(x_t))/\alpha_t$ and $\mathbb{E}[\epsilon \mid x_t] = -\sigma_t\nabla\log p_t(x_t)$. A check with Gaussian data $\mathcal{N}(2, 1)$, $\alpha = 0.6$, $\sigma = 0.8$ and $x_t = 2$: the score is $-0.8$ and both Tweedie and direct Gaussian conditioning give $\mathbb{E}[x_0 \mid x_t] = 2.48$.
:::

## The SDE view

Song et al. (2021) let the number of noise levels go to infinity. The discrete chain becomes a stochastic differential equation (SDE), and three things come out of it: a common language for DDPM and score-based models, a reverse-time SDE for sampling, and a deterministic ODE with the same marginals, which is the basis of fast samplers and exact likelihoods.

### What an SDE means here

The forward SDE is written

$$
dx = f(x, t)\,dt + g(t)\,dW_t ,
$$

with a drift $f$, a scalar diffusion coefficient $g$ and Brownian motion $W_t$. Operationally it means: over a short time $\Delta t$,

$$
x_{t + \Delta t} \approx x_t + f(x_t, t)\,\Delta t + g(t)\sqrt{\Delta t}\,z, \qquad z \sim \mathcal{N}(0, I),
$$

which is the Euler-Maruyama step. The noise scales with $\sqrt{\Delta t}$, not $\Delta t$, because the variances of independent increments add.

### The VP SDE is the limit of DDPM

Take $N$ steps on $t \in [0, 1]$ with $\Delta t = 1/N$, and write each DDPM variance as $\beta_i = \beta(t + \Delta t)\,\Delta t$, where $\beta(\cdot)$ is a fixed function (so $\beta(i/N) = N\beta_i$). One DDPM step is

$$
\begin{aligned}
x(t + \Delta t) &= \sqrt{1 - \beta(t + \Delta t)\Delta t}\;x(t) + \sqrt{\beta(t + \Delta t)\Delta t}\;z && \text{DDPM step}\\
&\approx \Big(1 - \tfrac{1}{2}\beta(t + \Delta t)\Delta t\Big)x(t) + \sqrt{\beta(t + \Delta t)\Delta t}\;z && \sqrt{1 - u} = 1 - \tfrac{u}{2} + O(u^2)\\
&\approx x(t) - \tfrac{1}{2}\beta(t)\,x(t)\,\Delta t + \sqrt{\beta(t)\Delta t}\;z && \beta(t + \Delta t) = \beta(t) + O(\Delta t).
\end{aligned}
$$

Matching this to the Euler-Maruyama step reads off $f = -\tfrac{1}{2}\beta(t)x$ and $g = \sqrt{\beta(t)}$:

$$
dx = -\tfrac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dW_t \qquad \text{(variance preserving, VP).}
$$

DDPM's linear schedule with $N = 1000$ becomes $\beta(t) = 0.1 + 19.9\,t$, since $1000 \times 10^{-4} = 0.1$ and $1000 \times 0.02 = 20$.

*Why "variance preserving".* Take the variance of both sides of the step. The two parts are independent, so variances add (Tool 1): $\mathrm{Var}\,x(t + \Delta t) = (1 - \beta\Delta t)\,\mathrm{Var}\,x(t) + \beta\Delta t$. Subtract $\mathrm{Var}\,x(t)$ and divide by $\Delta t$:

$$
\frac{d}{dt}\mathrm{Var}\,x(t) = \beta(t)\big(1 - \mathrm{Var}\,x(t)\big).
$$

If the data has unit variance, the right side is zero and the variance stays at one forever. Otherwise the gap to one shrinks: $\mathrm{Var}\,x(t) = 1 + e^{-\int_0^t\beta}\,(\mathrm{Var}\,x(0) - 1)$.

*The signal scale.* The mean obeys $\frac{d}{dt}\mathbb{E}x = -\tfrac{1}{2}\beta(t)\,\mathbb{E}x$, so $\alpha(t) = \exp\big(-\tfrac{1}{2}\int_0^t\beta(s)\,ds\big)$. For the linear $\beta(t)$ the integral is $0.1\,t + 9.95\,t^2$, so

$$
\alpha(t) = \exp\big(-\tfrac{1}{4}t^2(\bar\beta_{\max} - \bar\beta_{\min}) - \tfrac{1}{2}t\,\bar\beta_{\min}\big), \qquad \bar\beta_{\min} = 0.1,\ \bar\beta_{\max} = 20 .
$$

*Check against the discrete chain.* At $t = 0.5$: $\alpha = \exp(-\tfrac{1}{4}(0.25)(19.9) - 0.025) = \exp(-1.269) = 0.2812$, against $\sqrt{\bar\alpha_{500}} = 0.2803$ from the table. At $t = 1$: $\exp(-4.975 - 0.05) = 0.00657$, against $0.00635$. The small gaps come mostly from $\log(1 - \beta) \approx -\beta - \beta^2/2$ rather than $-\beta$. At $t = 1$ that is the whole gap, because $\sum_{i=1}^{1000}\beta_i = 10.05 = \int_0^1\beta(t)\,dt$ exactly; at $t = 0.5$ a smaller part comes from the sum $\sum_{i \le 500}\beta_i = 2.535$ differing from the integral $2.5375$.

**What this says.** DDPM is a discretization of one linear SDE. The SDE's drift shrinks the signal at rate $\tfrac{1}{2}\beta(t)$ while its noise refills the variance at rate $\beta(t)$, and the two balance exactly at unit variance.

### The VE SDE is the limit of NCSN

Score-based models (NCSN, Song and Ermon) add independent noise without shrinking the signal: $x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}\,z$, which gives $x_i \mid x_0 \sim \mathcal{N}(x_0, \sigma_i^2 I)$ when $\sigma_0 = 0$. With $\sigma_i = \sigma(i/N)$, the increment variance is $\sigma^2(t + \Delta t) - \sigma^2(t) \approx \frac{d\sigma^2}{dt}\Delta t$, so

$$
dx = \sqrt{\frac{d\,[\sigma^2(t)]}{dt}}\;dW_t \qquad \text{(variance exploding, VE).}
$$

Here $f = 0$ and $\alpha_t = 1$: the signal is never shrunk and the variance grows to $\mathrm{Var}(x_0) + \sigma_{\max}^2$. NCSN's geometric noise levels become $\sigma(t) = \sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t$, and since $\frac{d\sigma^2}{dt} = 2\sigma^2\log(\sigma_{\max}/\sigma_{\min})$, the diffusion coefficient is $g(t) = \sigma(t)\sqrt{2\log(\sigma_{\max}/\sigma_{\min})}$.

### Any schedule $(\alpha_t, \sigma_t)$ is an SDE

The fast samplers below need the reverse question: given a schedule $x_t = \alpha_t x_0 + \sigma_t\epsilon$, which linear SDE $dx = f(t)\,x\,dt + g(t)\,dW_t$ produces it? Over one short step the mean is multiplied by $1 + f\Delta t$ and the noise variance becomes $(1 + f\Delta t)^2\sigma_t^2 + g^2\Delta t$. Matching both with the schedule:

$$
\begin{aligned}
\frac{d\alpha_t}{dt} = f(t)\,\alpha_t \quad &\Longrightarrow\quad f(t) = \frac{d\log\alpha_t}{dt},\\
\frac{d\sigma_t^2}{dt} = 2f(t)\,\sigma_t^2 + g(t)^2 \quad &\Longrightarrow\quad g(t)^2 = \frac{d\sigma_t^2}{dt} - 2\frac{d\log\alpha_t}{dt}\,\sigma_t^2 .
\end{aligned}
$$

One more rewrite is useful. Factor out $2\sigma_t^2$ and use $\frac{d\sigma_t^2}{dt} = 2\sigma_t^2\frac{d\log\sigma_t}{dt}$:

$$
g(t)^2 = 2\sigma_t^2\Big(\frac{d\log\sigma_t}{dt} - \frac{d\log\alpha_t}{dt}\Big) = -2\sigma_t^2\,\frac{d\ell_t}{dt}, \qquad \ell_t = \log\frac{\alpha_t}{\sigma_t} .
$$

Check with VP: $\alpha^2 = e^{-\int\beta}$ gives $f = -\tfrac{1}{2}\beta$, and $\sigma^2 = 1 - \alpha^2$ gives $\frac{d\sigma^2}{dt} = \beta\alpha^2$, so $g^2 = \beta\alpha^2 + \beta\sigma^2 = \beta$.

**What this says.** The VP and VE SDEs are two members of one family. Any schedule $(\alpha_t, \sigma_t)$ fixes $f$ and $g$, and $g^2$ is proportional to how fast the log-SNR falls. This is the form DPM-Solver uses.

### The reverse-time SDE

Anderson (1982) showed that a diffusion run backwards in time is again a diffusion. For the forward SDE above, the process that starts at $x_T \sim p_T$ and runs from $t = T$ down to $0$,

$$
dx = \big[f(x, t) - g(t)^2\,\nabla_x\log p_t(x)\big]\,dt + g(t)\,d\bar W_t ,
$$

has the same marginal $p_t$ at every time. Here $dt$ is negative and $\bar W_t$ is a Brownian motion running backwards. Everything in it is known except the score, which is what the network estimates. This form uses the fact that the diffusion coefficient $g(t)$ does not depend on $x$; when it does, Anderson's reverse drift has an extra term, the divergence of $g g^\top$. Like Tool 11, Anderson's theorem is used here as a known result; the heuristic below shows where its terms come from.

*Intuition.* Forward diffusion spreads mass out. To undo that, the reverse process needs an extra drift that pushes mass back toward high-density regions, and the push must be strong enough to beat the fresh noise it still injects. The score points uphill in density, and $g^2$ is the rate at which noise is injected, so $g^2\nabla\log p_t$ is the natural correction.

*A heuristic derivation for one small step* (not Anderson's proof, but it shows where the terms come from). By Bayes, the backward step is

$$
q(x_{t-\Delta t} = y \mid x_t = x) \propto \underbrace{\mathcal{N}\big(x;\ y + f\Delta t,\ g^2\Delta t\,I\big)}_{\text{forward step}}\;\underbrace{p_{t-\Delta t}(y)}_{\text{prior on } y} .
$$

The Gaussian factor confines $y$ to within about $g\sqrt{\Delta t}$ of $x$, so expand the prior to first order around $x$: $\log p_{t - \Delta t}(y) \approx \text{const} + \nabla\log p_t(x)^\top(y - x)$, where the change from $t - \Delta t$ to $t$ and the second-order terms only contribute at higher order in $\Delta t$. Also treat $f$ as evaluated at $x$. Then

$$
\log q(y \mid x) = -\frac{\lVert y - (x - f\Delta t)\rVert^2}{2g^2\Delta t} + \nabla\log p_t(x)^\top y + \text{const}.
$$

This is quadratic in $y$ with precision $1/(g^2\Delta t)$, and by Tool 6 the mean is the linear coefficient divided by the precision:

$$
y \sim \mathcal{N}\Big(x - \big[f - g^2\nabla\log p_t(x)\big]\Delta t,\ \ g^2\Delta t\,I\Big).
$$

That is one Euler-Maruyama step of the reverse SDE with $dt = -\Delta t$. The same Taylor-and-complete-the-square trick reappears in classifier guidance.

**What this says.** Sampling from a diffusion model is simulating the reverse SDE with the learned score $s_\theta \approx \nabla\log p_t$. DDPM's ancestral sampler is one such discretization: written with $s_\theta = -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$, its step is $x_{t-1} = (x_t + \beta_t s_\theta)/\sqrt{1-\beta_t} + \sqrt{\beta_t}\,z$, the score SDE paper's form of the same update. NCSN's Langevin updates, $x \leftarrow x + \delta\,s_\theta(x) + \sqrt{2\delta}\,z$ at a fixed noise level, sample $p_t$ itself rather than moving between levels; the score SDE paper reuses them as a "corrector" after each reverse step ("predictor-corrector" samplers).

### The probability-flow ODE, derived from Fokker-Planck

The marginals $p_t$ of the forward SDE satisfy the Fokker-Planck equation (Tool 11). Rewrite its diffusion term as a transport term:

$$
\begin{aligned}
\partial_t p_t &= -\nabla\cdot(f\,p_t) + \tfrac{1}{2}g^2\,\Delta p_t && \text{Tool 11}\\
&= -\nabla\cdot(f\,p_t) + \tfrac{1}{2}g^2\,\nabla\cdot\big(p_t\,\nabla\log p_t\big) && \Delta p = \nabla\cdot(p\,\nabla\log p)\text{, Tool 10}\\
&= -\nabla\cdot\Big[\big(f - \tfrac{1}{2}g^2\,\nabla\log p_t\big)\,p_t\Big] && g \text{ does not depend on } x\text{; collect under one divergence.}
\end{aligned}
$$

The last line is the Fokker-Planck equation of a process with drift $\tilde f = f - \tfrac{1}{2}g^2\nabla\log p_t$ and no diffusion at all, that is, the continuity equation of the ODE

$$
\frac{dx}{dt} = f(x, t) - \tfrac{1}{2}g(t)^2\,\nabla_x\log p_t(x).
$$

If $x(0) \sim p_0$ and $x$ moves by this ODE, its density solves the same linear equation with the same initial condition as the SDE's marginals, so (assuming the solution is unique) it equals $p_t$ at every $t$. Run backwards from $x(T) \sim p_T$, the ODE produces samples of $p_0$.

**What this says.** The diffusion term can be traded for a drift that moves mass exactly as the noise would have spread it. Individual paths are completely different (smooth and deterministic instead of jagged), but the distribution at every time is the same. The ODE gives a deterministic map from noise to data, so it can be inverted (encode an image), interpolated, solved with high-order ODE methods, and used for exact likelihoods.

### A whole family with the same marginals

The split above can be done partially. For any $\zeta \ge 0$, split the diffusion term into two parts and rewrite only the second one with $\Delta p = \nabla\cdot(p\,\nabla\log p)$:

$$
\begin{aligned}
\partial_t p_t &= -\nabla\cdot(f\,p_t) + \tfrac{1}{2}\zeta^2 g^2\,\Delta p_t + \tfrac{1}{2}(1 - \zeta^2)\,g^2\,\nabla\cdot\big(p_t\,\nabla\log p_t\big) && \tfrac12 g^2 = \tfrac12\zeta^2 g^2 + \tfrac12(1-\zeta^2)g^2\\
&= -\nabla\cdot\Big[\big(f - \tfrac{1}{2}(1-\zeta^2)\,g^2\,\nabla\log p_t\big)\,p_t\Big] + \tfrac{1}{2}(\zeta g)^2\,\Delta p_t && \text{collect the divergences.}
\end{aligned}
$$

The last line is the Fokker-Planck equation (Tool 11) of the SDE with drift $f - \tfrac{1}{2}(1-\zeta^2)g^2\nabla\log p_t$ and diffusion coefficient $\zeta g$, so that SDE also has marginals $p_t$. Its time reversal, by Anderson's formula (forward drift minus the squared diffusion coefficient $\zeta^2 g^2$ times the score), subtracts another $\zeta^2 g^2\nabla\log p_t$ from the drift:

$$
dx = \Big[f - \frac{1 + \zeta^2}{2}\,g^2\,\nabla\log p_t\Big]dt + \zeta\,g\,d\bar W_t .
$$

$\zeta = 1$ is the reverse SDE and $\zeta = 0$ is the probability-flow ODE. The extra noise in between is balanced by extra pull toward high density; EDM's stochastic sampler and its "churn" are this idea with $\zeta$ chosen per noise level.

### The ODE in VP and VE coordinates

*VP.* With $f = -\tfrac{1}{2}\beta x$, $g^2 = \beta$ and the score written through the noise predictor, $\nabla\log p_t = -\epsilon_\theta/\sigma_t$:

$$
\frac{dx}{dt} = -\tfrac{1}{2}\beta(t)\Big[x - \frac{\epsilon_\theta(x, t)}{\sigma_t}\Big].
$$

*VE, as in EDM.* With $\sigma(t) = t$, $f = 0$ and $g^2 = d\sigma^2/dt = 2\sigma$, the ODE is $dx/d\sigma = -\sigma\,\nabla\log p(x; \sigma)$. Tweedie with $\alpha = 1$ gives $\nabla\log p(x; \sigma) = (D(x; \sigma) - x)/\sigma^2$, where $D$ is the posterior-mean denoiser, so

$$
\frac{dx}{d\sigma} = \frac{x - D(x; \sigma)}{\sigma} .
$$

The right side is the estimated noise $\hat\epsilon$ in VE units. Every EDM sampler integrates this equation; see the [EDM page](edm.html) for the version with a general scale schedule.

### Worked example: the ODE for Gaussian data

Take one-dimensional data $x_0 \sim \mathcal{N}(m, s^2)$ in VE coordinates. By Tool 1, $p(x; \sigma) = \mathcal{N}(m, s^2 + \sigma^2)$, and by Tool 7 its score is $-(x - m)/(s^2 + \sigma^2)$. The ODE is

$$
\frac{dx}{d\sigma} = \frac{\sigma\,(x - m)}{s^2 + \sigma^2} .
$$

Write $y = x - m$. Then $\frac{dy}{y} = \frac{\sigma\,d\sigma}{s^2 + \sigma^2} = \tfrac{1}{2}\,d\log(s^2 + \sigma^2)$, and integrating from $\sigma_{\max}$ down to $\sigma$,

$$
x(\sigma) - m = \big(x(\sigma_{\max}) - m\big)\sqrt{\frac{s^2 + \sigma^2}{s^2 + \sigma_{\max}^2}} .
$$

At $\sigma = 0$ the deviation from the mean is multiplied by $s/\sqrt{s^2 + \sigma_{\max}^2}$. If $x(\sigma_{\max}) \sim \mathcal{N}(m, s^2 + \sigma_{\max}^2)$, the output is $\mathcal{N}(m, s^2)$: exactly the data. With $s = 0.5$ and $\sigma_{\max} = 80$ the factor is $0.5/80.0016 = 0.00625$. Every sample is a deterministic, monotone rescaling of its starting noise. The reverse SDE from the same start also ends at $\mathcal{N}(m, s^2)$, but two runs from the same $x(\sigma_{\max})$ end at different points.

### Exact likelihoods through the ODE

Let $\tilde f$ be the ODE's velocity. Along a trajectory $x(t)$, the density changes as

$$
\begin{aligned}
\frac{d}{dt}\,p_t\big(x(t)\big) &= \partial_t p_t + \nabla p_t\cdot\tilde f && \text{chain rule}\\
&= -\nabla\cdot(\tilde f\,p_t) + \tilde f\cdot\nabla p_t && \text{continuity equation (Tool 11 with } g = 0)\\
&= -p_t\,\nabla\cdot\tilde f && \text{Tool 10(b).}
\end{aligned}
$$

Dividing by $p_t$ gives $\frac{d}{dt}\log p_t(x(t)) = -\nabla\cdot\tilde f$, and integrating from $0$ to $T$,

$$
\log p_0\big(x(0)\big) = \log p_T\big(x(T)\big) + \int_0^T\nabla\cdot\tilde f\big(x(t), t\big)\,dt .
$$

The divergence is the trace of a $d \times d$ Jacobian, too expensive to form, so it is estimated without bias by Hutchinson's trick, $\nabla\cdot\tilde f = \mathbb{E}_e[e^\top(\partial\tilde f/\partial x)\,e]$ with $\mathbb{E}[ee^\top] = I$, one vector-Jacobian product per sample. With this, the score SDE paper reported 2.99 bits per dimension on uniformly dequantized CIFAR-10, without training on the likelihood.

**What this says.** Because the ODE is an invertible flow, its log-density changes along a path by minus the divergence of the velocity, exactly as in a continuous normalizing flow. A diffusion model trained by regression therefore also gives exact likelihoods, at the price of an ODE solve with a trace estimate.

::: key
**How to say it in an interview.** "In the continuous limit, DDPM is the variance-preserving SDE $dx = -\tfrac{1}{2}\beta x\,dt + \sqrt{\beta}\,dW$ and NCSN is the variance-exploding SDE $dx = \sqrt{d\sigma^2/dt}\,dW$. Anderson's theorem says the time reversal is again an SDE whose drift is the forward drift minus $g^2$ times the score, so a learned score is all you need to sample. Rewriting the Fokker-Planck equation with $\Delta p = \nabla\cdot(p\nabla\log p)$ turns the diffusion term into a drift, which gives the probability-flow ODE $dx/dt = f - \tfrac{1}{2}g^2\nabla\log p_t$ with the same marginals. The ODE is deterministic and invertible, which is what fast solvers, inversion and exact likelihoods use."
:::

::: qa
**How are DDPM and score-based generative models (NCSN) related?**

They are two discretizations of the same continuous framework: DDPM is a discretized VP SDE and NCSN (annealed Langevin over noise levels) is a discretized VE SDE. DDPM's noise predictor is a scaled score, $s_\theta = -\epsilon_\theta/\sigma_t$, and both are trained by denoising score matching with different per-level weights. Ancestral DDPM sampling is a discretization of the reverse SDE; NCSN's Langevin steps act as a corrector. The SDE view adds the probability-flow ODE (deterministic sampling, likelihoods), predictor-corrector samplers, and a clean way to compare schedules.
:::

::: qa
**The probability-flow ODE injects no noise. How can it produce the same distribution as the reverse SDE?**

The Fokker-Planck equation of the SDE can be rewritten as a pure transport equation by absorbing the diffusion term into the drift: $\tfrac{1}{2}g^2\Delta p = \nabla\cdot(\tfrac{1}{2}g^2 p\, \nabla\log p)$. A deterministic flow with that modified drift moves probability mass exactly as the SDE does, so the marginals match at every $t$, while individual trajectories differ. The equivalence assumes the exact score. With learned scores and discretization error, the two behave differently: the SDE's fresh noise tends to contract accumulated errors, while the ODE carries them forward but converges faster with high-order solvers.
:::

::: qa
**Derive the probability-flow ODE from the forward SDE.**

Start from the Fokker-Planck equation of $dx = f\,dt + g\,dW$: $\partial_t p_t = -\nabla\cdot(fp_t) + \tfrac{1}{2}g^2\Delta p_t$. Because $\nabla p = p\nabla\log p$, the Laplacian is $\Delta p = \nabla\cdot(p\nabla\log p)$, and since $g$ does not depend on $x$ the right side becomes $-\nabla\cdot[(f - \tfrac{1}{2}g^2\nabla\log p_t)\,p_t]$. That is the continuity equation of the deterministic ODE $dx/dt = f - \tfrac{1}{2}g^2\nabla\log p_t$, so the ODE started from $p_0$ has marginal $p_t$ at every time (given uniqueness of the PDE solution). In VP form it reads $dx/dt = -\tfrac{1}{2}\beta(t)[x - \epsilon_\theta/\sigma_t]$; in EDM's VE form, $dx/d\sigma = (x - D(x;\sigma))/\sigma$. Splitting only part of the diffusion term gives a family of SDEs with noise $\zeta g$ and drift $f - \tfrac{1+\zeta^2}{2}g^2\nabla\log p_t$ that all share the marginals.
:::

::: qa
**What is the reverse-time SDE, and why does the score appear in it?**

For $dx = f\,dt + g\,dW$, Anderson's result is that the time-reversed process satisfies $dx = [f - g^2\nabla\log p_t]\,dt + g\,d\bar W$, integrated from $T$ down to $0$, and has the same marginals. Heuristically: one backward step is proportional to the forward Gaussian kernel times the prior $p_{t-\Delta t}(y)$. Expanding $\log p$ to first order around the current point and completing the square gives a Gaussian with variance $g^2\Delta t$ and mean shifted by $g^2\Delta t\,\nabla\log p_t$ against the forward drift. The score is the only unknown, so learning it is enough to sample. The forward noise spreads mass out, and the $g^2\nabla\log p_t$ drift pushes it back uphill at the matching rate.
:::

## Parameterizations

The network can be asked to output the noise, the clean image, the velocity $v$ or the score. Given $x_t$ they carry the same information, so the choice does not change what the optimum represents. It changes two practical things: how the loss is weighted across noise levels, and how well conditioned the regression is. This section derives every conversion and then says where each target is well behaved.

### One equation, one unknown

At a fixed level, $x_t = \alpha_t x_0 + \sigma_t\epsilon$ ties together three vectors. The network sees $x_t$ and $t$, so once it commits to one of $x_0$ or $\epsilon$, the other follows. The conversions, line by line:

$$
\begin{aligned}
\hat x_0 &= \frac{x_t - \sigma_t\hat\epsilon}{\alpha_t} && \text{solve } x_t = \alpha_t\hat x_0 + \sigma_t\hat\epsilon \text{ for } \hat x_0\\
\hat\epsilon &= \frac{x_t - \alpha_t\hat x_0}{\sigma_t} && \text{solve the same equation for } \hat\epsilon\\
\hat x_0 &= \frac{x_t + \sigma_t^2\,\hat s}{\alpha_t}, \quad \hat\epsilon = -\sigma_t\,\hat s && \text{Tweedie, with } \hat s \text{ a score estimate.}
\end{aligned}
$$

### The velocity $v$, derived

Define $v = \alpha_t\epsilon - \sigma_t x_0$ and assume variance preservation, $\alpha_t^2 + \sigma_t^2 = 1$. Then $x_0$ and $\epsilon$ are recovered from $(x_t, v)$ as follows:

$$
\begin{aligned}
\alpha_t x_t - \sigma_t v &= \alpha_t(\alpha_t x_0 + \sigma_t\epsilon) - \sigma_t(\alpha_t\epsilon - \sigma_t x_0) && \text{substitute both definitions}\\
&= \alpha_t^2 x_0 + \alpha_t\sigma_t\epsilon - \sigma_t\alpha_t\epsilon + \sigma_t^2 x_0 && \text{expand}\\
&= (\alpha_t^2 + \sigma_t^2)\,x_0 = x_0 && \text{the } \epsilon \text{ terms cancel; VP,}
\end{aligned}
$$

$$
\begin{aligned}
\sigma_t x_t + \alpha_t v &= \sigma_t(\alpha_t x_0 + \sigma_t\epsilon) + \alpha_t(\alpha_t\epsilon - \sigma_t x_0) && \text{substitute}\\
&= (\sigma_t^2 + \alpha_t^2)\,\epsilon = \epsilon && \text{the } x_0 \text{ terms cancel; VP.}
\end{aligned}
$$

*Why "velocity".* Put $\alpha_t = \cos\phi$ and $\sigma_t = \sin\phi$, so $\phi = \arctan(\sigma_t/\alpha_t)$ runs from $0$ (clean) to $\pi/2$ (pure noise). Then $x_\phi = \cos\phi\,x_0 + \sin\phi\,\epsilon$ moves on a circle, and $\frac{dx_\phi}{d\phi} = -\sin\phi\,x_0 + \cos\phi\,\epsilon = v$. Predicting $v$ is predicting the direction of motion along that arc (Salimans and Ho, Appendix D).

*Numeric check.* With $\alpha = 0.6$, $\sigma = 0.8$, $x_0 = 1$ and $\epsilon = 0.5$: $x_t = 0.6 + 0.4 = 1.0$ and $v = 0.3 - 0.8 = -0.5$. Then $\alpha x_t - \sigma v = 0.6 + 0.4 = 1.0 = x_0$ and $\sigma x_t + \alpha v = 0.8 - 0.3 = 0.5 = \epsilon$.

**What this says.** $v$ is a rotation of the pair $(x_0, \epsilon)$: the matrix taking $(x_0, \epsilon)$ to $(x_t, v)$ is $\begin{pmatrix}\alpha_t & \sigma_t\\ -\sigma_t & \alpha_t\end{pmatrix}$, and inverting a rotation is transposing it. That is why the conversions are so clean, and why they need $\alpha_t^2 + \sigma_t^2 = 1$.

| Target | Definition | Recover $\hat{x}_0$ | Recover $\hat{\epsilon}$ |
|---|---|---|---|
| noise | $\epsilon$ | $(x_t - \sigma_t \hat{\epsilon})/\alpha_t$ | $\hat{\epsilon}$ |
| data | $x_0$ | $\hat{x}_0$ | $(x_t - \alpha_t \hat{x}_0)/\sigma_t$ |
| velocity | $v = \alpha_t \epsilon - \sigma_t x_0$ | $\alpha_t x_t - \sigma_t \hat{v}$ | $\sigma_t x_t + \alpha_t \hat{v}$ |
| score | $\nabla \log p_t(x_t)$ | $(x_t + \sigma_t^2 \hat{s})/\alpha_t$ | $-\sigma_t \hat{s}$ |

The velocity column assumes $\alpha_t^2 + \sigma_t^2 = 1$; without it, divide the $\hat x_0$ and $\hat\epsilon$ entries by $\alpha_t^2 + \sigma_t^2$.

### Errors convert with fixed factors

Compare the losses through the clean-image error $x_0 - \hat x_0$. For the noise:

$$
\begin{aligned}
\epsilon - \hat\epsilon &= \frac{x_t - \alpha_t x_0}{\sigma_t} - \frac{x_t - \alpha_t\hat x_0}{\sigma_t} && \text{both from } x_t = \alpha_t(\cdot) + \sigma_t(\cdot)\\
&= -\frac{\alpha_t}{\sigma_t}\,(x_0 - \hat x_0) && \text{the } x_t \text{ terms cancel.}
\end{aligned}
$$

For the velocity, substitute this into $v - \hat v = \alpha_t(\epsilon - \hat\epsilon) - \sigma_t(x_0 - \hat x_0)$:

$$
v - \hat v = -\Big(\frac{\alpha_t^2}{\sigma_t} + \sigma_t\Big)(x_0 - \hat x_0) = -\frac{\alpha_t^2 + \sigma_t^2}{\sigma_t}\,(x_0 - \hat x_0) = -\frac{1}{\sigma_t}\,(x_0 - \hat x_0) \quad \text{(VP)}.
$$

Squaring, and using $1/\sigma_t^2 = (\alpha_t^2 + \sigma_t^2)/\sigma_t^2 = 1 + \mathrm{SNR}(t)$:

$$
\lVert \epsilon - \hat{\epsilon}\rVert^2 = \mathrm{SNR}(t)\,\lVert x_0 - \hat{x}_0\rVert^2, \qquad \lVert v - \hat{v}\rVert^2 = \big(1 + \mathrm{SNR}(t)\big)\lVert x_0 - \hat{x}_0\rVert^2 .
$$

**Check with the Gaussian example** ($\mathrm{SNR} = 0.5625$). The minimum $x_0$-loss is the posterior variance $0.64$. The formulas predict a minimum $\epsilon$-loss of $0.5625 \times 0.64 = 0.36$, which is the noise posterior variance found earlier, and a minimum $v$-loss of $1.5625 \times 0.64 = 1.00$. A Monte Carlo run gives $0.3600$ and $1.0001$.

**What this says.** Switching the target is the same as reweighting the $x_0$-loss per noise level: by $\mathrm{SNR}$ for $\epsilon$, by $1 + \mathrm{SNR}$ for $v$, by $1$ for $x_0$. Each choice therefore comes with an implied loss weighting, which the loss weighting section makes explicit.

### Where each target is well conditioned

Two separate questions decide whether a target is easy to learn.

*How big is the target?* $\epsilon$ always has unit variance. $x_0$ has the data's variance $\sigma_{\text{data}}^2$ at every level. For $v$, $\mathrm{Var}(v) = \alpha_t^2 + \sigma_t^2\sigma_{\text{data}}^2$ per coordinate (the two parts are independent), which is exactly 1 when the data has unit variance.

*How much is an output error amplified?* An error $\delta$ in the network's output changes the implied clean image and noise by:

| Network outputs | Effect on $\hat x_0$ | Effect on $\hat\epsilon$ |
|---|---|---|
| $\hat\epsilon + \delta$ | $-(\sigma_t/\alpha_t)\,\delta$ | $\delta$ |
| $\hat x_0 + \delta$ | $\delta$ | $-(\alpha_t/\sigma_t)\,\delta$ |
| $\hat v + \delta$ | $-\sigma_t\,\delta$ | $\alpha_t\,\delta$ |

Under the linear schedule, the factor $\sigma_t/\alpha_t$ is $0.34$ at $t = 100$, $3.4$ at $t = 500$, $17$ at $t = 750$ and $157$ at $t = 1000$. So with $\epsilon$-prediction, an output error of $0.01$ at the last step becomes an error of $1.6$ in $\hat x_0$, larger than the pixel range. With $x_0$-prediction the mirror problem appears at low noise: $\alpha_t/\sigma_t = 100$ at $t = 1$, so small errors become large errors in the implied noise, and the network must reproduce almost all of its input exactly. With $v$-prediction both factors, $\sigma_t$ and $\alpha_t$, are at most 1, at every level.

When each is preferred:

- **$\epsilon$**: unit-variance target at every level; works well for many-step sampling. But $\hat{x}_0 = (x_t - \sigma_t\hat{\epsilon})/\alpha_t$ amplifies errors by $\sigma_t/\alpha_t$ at low SNR, and at zero SNR the task is trivial ($x_T = \epsilon$) and says nothing about data.
- **$x_0$**: well-behaved at high noise (predict a mean image). At low noise the network must reproduce almost its input, and the implied $\epsilon$ is noisy; usually paired with a skip connection.
- **$v$**: behaves like $\epsilon$-prediction at high SNR and like $-x_0$ at low SNR, with a unit-variance target if the data have unit variance. It stays well-defined at zero SNR, which is why progressive distillation and the zero-terminal-SNR fix use it.
- **score**: natural in theory, but it scales like $1/\sigma_t$, so networks rarely output it directly.
- **EDM preconditioning** (next sections and the [EDM page](edm.html)) mixes these in a $\sigma$-dependent way; with $\sigma_{\text{data}} = 1$ its network input $c_{\text{in}}x$ is exactly the VP sample $x_t$ (with $\sigma_t/\alpha_t = \sigma$) and its effective target is exactly $-v$.

*Checking the last bullet.* With $\sigma_{\text{data}} = 1$, EDM uses $c_{\text{skip}} = 1/(1 + \sigma^2)$ and $c_{\text{out}} = \sigma/\sqrt{1 + \sigma^2}$, and its network is trained toward $(x_0 - c_{\text{skip}}x)/c_{\text{out}}$ with $x = x_0 + \sigma\epsilon$:

$$
\begin{aligned}
x_0 - \frac{x_0 + \sigma\epsilon}{1 + \sigma^2} &= \frac{\sigma^2 x_0 - \sigma\epsilon}{1 + \sigma^2} && \text{common denominator}\\
\frac{1}{c_{\text{out}}}\cdot\frac{\sigma(\sigma x_0 - \epsilon)}{1 + \sigma^2} &= \frac{\sigma x_0 - \epsilon}{\sqrt{1 + \sigma^2}} && \text{divide by } c_{\text{out}} .
\end{aligned}
$$

With $\alpha_t = 1/\sqrt{1 + \sigma^2}$ and $\sigma_t = \sigma/\sqrt{1 + \sigma^2}$ this is $\sigma_t x_0 - \alpha_t\epsilon = -v$.

::: key
**How to say it in an interview.** "Given $x_t$, the noise, the clean image, $v = \alpha\epsilon - \sigma x_0$ and the score are affine functions of each other, so the targets differ only in weighting and conditioning. In clean-image units the $\epsilon$-loss weights each level by the SNR and the $v$-loss by $1 + \mathrm{SNR}$. $\epsilon$-prediction amplifies errors by $\sigma/\alpha$ when converting to an image, which blows up near zero SNR; $x_0$-prediction has the mirror problem at low noise; $v$ keeps both amplification factors below one, which is why it is used for distillation, zero terminal SNR and many large models."
:::

::: qa
**Compare $\epsilon$-, $x_0$- and $v$-prediction. When would you pick each?**

All three carry the same information given $x_t$, so the differences are optimization and numerics. $\epsilon$-prediction has a unit-variance target and is the classic choice for many-step samplers, but implied $\hat{x}_0$ errors are amplified by $\sigma_t/\alpha_t$ at low SNR. $x_0$-prediction is stable at high noise but wasteful at low noise, where it must copy most of the input. $v$-prediction interpolates: like $\epsilon$ when noise is small, like $x_0$ when noise is large, so its errors stay bounded across the range. I would use $v$ (or EDM preconditioning) for zero terminal SNR, few-step distillation or high resolution, and remember that switching targets also changes the implied loss weighting.
:::

::: qa
**Why does $\epsilon$-prediction break down at zero SNR and in one-step distillation?**

At zero SNR, $x_T = \epsilon$, so predicting the noise is the identity map and the loss gives no signal about the data. Converting to an image needs $\hat{x}_0 = (x_t - \sigma_t\hat{\epsilon})/\alpha_t$, which divides by $\alpha_t = 0$. Near zero SNR the division is finite but amplifies tiny output errors enormously. With many steps later updates correct this; with one or few steps the first prediction must already be a good image. Equivalently, the $\epsilon$-loss gives weight $\mathrm{SNR} \to 0$ to $x_0$ reconstruction there. Salimans & Ho switched to $x_0$, mixed, or $v$ targets for this reason.
:::

::: qa
**Derive the conversions between $v$, $x_0$ and $\epsilon$, and the loss weight that $v$-prediction implies.**

With $v = \alpha_t\epsilon - \sigma_t x_0$ and $x_t = \alpha_t x_0 + \sigma_t\epsilon$, compute $\alpha_t x_t - \sigma_t v = (\alpha_t^2 + \sigma_t^2)x_0$ and $\sigma_t x_t + \alpha_t v = (\alpha_t^2 + \sigma_t^2)\epsilon$; with variance preservation both factors are one, so $x_0 = \alpha_t x_t - \sigma_t v$ and $\epsilon = \sigma_t x_t + \alpha_t v$. Writing $\alpha_t = \cos\phi$, $\sigma_t = \sin\phi$ shows $v = dx_t/d\phi$. For the loss, $\epsilon - \hat\epsilon = -(\alpha_t/\sigma_t)(x_0 - \hat x_0)$, so $v - \hat v = \alpha_t(\epsilon - \hat\epsilon) - \sigma_t(x_0 - \hat x_0) = -(x_0 - \hat x_0)/\sigma_t$, and $\lVert v - \hat v\rVert^2 = (1 + \mathrm{SNR})\lVert x_0 - \hat x_0\rVert^2$. Compared with the $\epsilon$-loss's weight $\mathrm{SNR}$, $v$ keeps a weight of at least one at high noise, which is why it does not ignore the zero-SNR end.
:::

## Noise schedules and SNR

Think of a schedule as a curve $\lambda_t$ in log-SNR. It decides which noise levels the network trains on and how often, and separately, where the sampler places its steps. This section shows that, for training, the schedule acts only as a sampling distribution over noise levels, and then covers the three schedule fixes interviewers ask about: cosine, resolution shift and zero terminal SNR.

### For training, the schedule is an importance-sampling distribution

Kingma and Gao (2023) write every common objective in one form. In continuous time with $t \sim \mathcal{U}(0, 1)$ (uniform on $[0, 1]$), and writing the noise prediction as $\hat\epsilon_\theta(x_t; \lambda_t)$ to stress that the network is told the log-SNR, a weighted noise-prediction loss is

$$
\mathcal{L}_w = \tfrac{1}{2}\,\mathbb{E}_{t \sim \mathcal{U}(0,1),\,\epsilon}\Big[w(\lambda_t)\,\Big(-\frac{d\lambda_t}{dt}\Big)\,\big\lVert\hat\epsilon_\theta(x_t; \lambda_t) - \epsilon\big\rVert^2\Big].
$$

Change the variable of integration from $t$ to $\lambda$ (the schedule $\lambda_t$ is decreasing, so it is invertible):

$$
\begin{aligned}
\mathcal{L}_w &= \tfrac{1}{2}\int_0^1 w(\lambda_t)\Big(-\frac{d\lambda_t}{dt}\Big)\,\mathbb{E}_\epsilon\lVert\hat\epsilon_\theta - \epsilon\rVert^2\,dt && \text{write the expectation over } t \text{ as an integral}\\
&= \tfrac{1}{2}\int_{\lambda_{\min}}^{\lambda_{\max}} w(\lambda)\,\mathbb{E}_\epsilon\big\lVert\hat\epsilon_\theta(x_\lambda; \lambda) - \epsilon\big\rVert^2\,d\lambda && \text{substitute } \lambda = \lambda_t\text{; the minus sign flips the limits.}
\end{aligned}
$$

Here $x_\lambda$ is the noisy sample at the level whose log-SNR is $\lambda$. The second line uses $d\lambda = \frac{d\lambda_t}{dt}dt$: as $t$ runs from 0 to 1, $\lambda$ runs down from $\lambda_0$ to $\lambda_1$, and $\int_0^1 (\cdot)\big(-\frac{d\lambda_t}{dt}\big)dt = -\int_{\lambda_0}^{\lambda_1}(\cdot)\,d\lambda = \int_{\lambda_1}^{\lambda_0}(\cdot)\,d\lambda$.

The integral no longer mentions the schedule, except through its endpoints $\lambda_{\min} = \lambda_1$ and $\lambda_{\max} = \lambda_0$. What the schedule does change is the Monte Carlo estimate: drawing $t$ uniformly draws $\lambda$ from the density $p(\lambda) = -dt/d\lambda$, so the same integral is estimated as $\tfrac{1}{2}\mathbb{E}_{\lambda \sim p}[w(\lambda)/p(\lambda)\cdot\lVert\hat\epsilon_\theta - \epsilon\rVert^2]$. The schedule is an importance-sampling distribution: it changes the variance of the gradient, not the objective. Conversely, plain $\epsilon$-MSE with uniform $t$ is $\int p(\lambda)\,\mathbb{E}\lVert\hat\epsilon_\theta - \epsilon\rVert^2\,d\lambda$, so its implied weight is $w(\lambda) \propto p(\lambda)$.

**Worked example: the cosine schedule's implied weight.** Without the small offset, the cosine schedule has $\bar\alpha_t = \cos^2(\pi t/2)$, so $\mathrm{SNR} = \cot^2(\pi t/2)$ and $\lambda_t = -2\log\tan(\pi t/2)$. Invert: $\tan(\pi t/2) = e^{-\lambda/2}$, so $t = \frac{2}{\pi}\arctan(e^{-\lambda/2})$. Differentiate:

$$
\begin{aligned}
\frac{dt}{d\lambda} &= \frac{2}{\pi}\cdot\frac{1}{1 + e^{-\lambda}}\cdot\Big(-\tfrac{1}{2}e^{-\lambda/2}\Big) && \text{chain rule, } \tfrac{d}{du}\arctan u = \tfrac{1}{1 + u^2}\\
&= -\frac{1}{\pi}\cdot\frac{1}{e^{\lambda/2} + e^{-\lambda/2}} && \text{multiply top and bottom by } e^{\lambda/2}\\
&= -\frac{1}{2\pi}\operatorname{sech}(\lambda/2) .
\end{aligned}
$$

So $p(\lambda) = \operatorname{sech}(\lambda/2)/(2\pi)$: $0.159$ at $\lambda = 0$ and $0.042$ at $\lambda = \pm 4$. $\epsilon$-prediction trained with the cosine schedule therefore weights noise levels by $\operatorname{sech}(\lambda/2)$, peaked at $\mathrm{SNR} = 1$, which is the entry Kingma and Gao list for improved DDPM.

**What this says.** In continuous time, "which schedule" and "which loss weight" are not independent design choices: only the product matters for the objective, and the split between them only matters for gradient noise. For sampling, the schedule is a separate choice again: it decides where the solver's steps land.

### Why the linear schedule was replaced: the cosine schedule

Nichol and Dhariwal found that DDPM's linear schedule wastes steps at low resolution: the end of the forward process is so noisy that it contributes little to sample quality. On ImageNet $64 \times 64$, a model trained with the linear schedule barely got worse in FID when up to the first 20% of the reverse process was skipped. They found the linear schedule sub-optimal at $64 \times 64$ and $32 \times 32$ but good at high resolution. Their replacement defines $\bar\alpha_t$ directly:

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\Big(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\Big), \quad s = 0.008,
$$

with $\beta_t = 1 - \bar\alpha_t/\bar\alpha_{t-1}$ clipped at $0.999$ to avoid a singularity at $t = T$. The offset $s$ keeps the first steps from being too small: it was chosen so that $\sqrt{\beta_1}$ is slightly below the pixel bin size $1/127.5 = 0.0078$ (here $\sqrt{\beta_1} = 0.0064$). In continuous time without the offset this is $\lambda_t = -2\log\tan(\pi t/2)$.

**Worked comparison ($T = 1000$).**

| $t$ | linear $\bar\alpha_t$ | linear $\lambda_t$ | cosine $\bar\alpha_t$ | cosine $\lambda_t$ | cosine $\tilde\beta_t/\beta_t$ |
|---|---|---|---|---|---|
| 1 | 0.99990 | 9.21 | 0.99996 | 10.10 | 0 |
| 10 | 0.99811 | 6.27 | 0.99937 | 7.37 | 0.865 |
| 100 | 0.8970 | 2.16 | 0.9721 | 3.55 | 0.982 |
| 250 | 0.5241 | 0.10 | 0.8470 | 1.71 | 0.993 |
| 500 | 0.0786 | $-2.46$ | 0.4938 | $-0.02$ | 0.997 |
| 750 | $3.35 \times 10^{-3}$ | $-5.70$ | 0.1443 | $-1.78$ | 0.9987 |
| 900 | $2.75 \times 10^{-4}$ | $-8.20$ | 0.0241 | $-3.70$ | 0.9995 |
| 1000 | $4.04 \times 10^{-5}$ | $-10.12$ | $2.43 \times 10^{-9}$ | $-19.84$ | 1.0000 |

Two hand checks. At $t = 500$ the argument is $(0.5 + 0.008)/1.008 = 0.50397$, so the cosine schedule gives $f(500)/f(0) = \cos^2(0.50397 \times \pi/2)/\cos^2(0.00794 \times \pi/2) = 0.49377/0.99984 = 0.4938$, an SNR of almost exactly one at the midpoint, as designed. At $t = T$ only one $\beta$ is clipped (the last one), which leaves $\sqrt{\bar\alpha_T} = 4.9 \times 10^{-5}$ instead of zero.

**What this says.** The linear schedule crosses $\mathrm{SNR} = 1$ at $t = 260$ and spends the last 250 steps below $\lambda = -5.7$, where the image is nearly invisible. The cosine schedule crosses at $t \approx 497$ and changes log-SNR almost linearly through the middle, so steps are spread more evenly over the levels where structure is decided. Its last few steps fall off a cliff to $\lambda \approx -20$, which is harmless.

### Resolution: the same SNR means less noise for bigger images

Noise is independent per pixel, but image content is strongly correlated across neighboring pixels. Average an $s \times s$ patch where the image is roughly constant: the signal average is unchanged, while the noise average has variance $\sigma_t^2/s^2$ (Tool 1 with $s^2$ independent terms). So the SNR of the coarse image is $s^2$ times the per-pixel SNR. At high resolution, a schedule tuned at low resolution leaves the global layout readable at what is nominally heavy noise, and the model has only a short window of levels in which to decide composition.

Simple diffusion (Hoogeboom et al.) defines the schedule at a reference resolution of 64 and shifts it:

$$
\mathrm{SNR}^{d\times d}_{\text{shift}}(t) = \mathrm{SNR}^{64\times 64}(t)\cdot (64/d)^2,
$$

a shift of $2\log(64/d)$ in log-SNR: $-1.39$ at $128$, $-2.77$ at $256$ and $-4.16$ at $512$. Since the shifted schedule generates fine detail late, they also propose interpolating between two shifts, and they recommend the interpolated version when sampling with guidance.

Stable Diffusion 3 derives a similar shift for rectified flow ($x_t = (1-t)x_0 + t\epsilon$). Consider a constant image: the $n$ pixels give $n$ independent noisy observations of the same value, so the uncertainty about it scales as $\frac{t}{1-t}\cdot\frac{1}{\sqrt n}$. Requiring equal uncertainty at resolutions with $n$ and $m$ pixels gives

$$
t_m = \frac{\alpha t_n}{1 + (\alpha - 1)t_n}, \qquad \alpha = \sqrt{m/n},
$$

where $\alpha$ is a shift factor, not a signal scale. With $\lambda_t = 2\log\frac{1-t}{t}$ for rectified flow, substitution gives $\lambda_{t_m} = \lambda_{t_n} - 2\log\alpha$, the same kind of log-SNR shift as simple diffusion. Worked example with SD3's choice $\alpha = 3$ at $1024^2$: $t_n = 0.5$ maps to $t_m = 1.5/2 = 0.75$, and $\lambda$ moves from $0$ to $-2.20 = -2\log 3$. See [flow matching](flow-matching.html) for that parameterization.

**What this says.** The "right" amount of noise depends on how much the image can average away. Doubling the resolution in each direction multiplies the effective SNR by four, so the schedule should move about $\log 4 = 1.39$ toward noise in log-SNR; both simple diffusion and SD3 implement that as a constant shift.

### Zero terminal SNR (Lin et al.)

Neither the linear, the clipped cosine nor Stable Diffusion's schedule reaches $\mathrm{SNR} = 0$. Stable Diffusion uses $\beta_t = \big(\sqrt{0.00085}\,(1 - i) + \sqrt{0.012}\,i\big)^2$ with $i = (t-1)/(T-1)$, which ends at

$$
x_T = 0.068265\,x_0 + 0.997667\,\epsilon, \qquad \mathrm{SNR}(T) = 0.00468 .
$$

During training the model sees this leaked low-frequency signal at $t = T$, including each channel's mean, and learns to keep it. At inference it gets pure noise with zero mean, so outputs stay near medium brightness, and prompts like "solid black background" fail. Lin et al. fix the mismatch with four changes:

1. Rescale the schedule so $\sqrt{\bar\alpha_T} = 0$ while keeping $\sqrt{\bar\alpha_1}$: shift $\sqrt{\bar\alpha_t}$ down by its last value and stretch it back up,

   $$
   \sqrt{\bar{\alpha}_t}' = \big(\sqrt{\bar{\alpha}_t} - \sqrt{\bar{\alpha}_T}\big)\,\frac{\sqrt{\bar{\alpha}_1}}{\sqrt{\bar{\alpha}_1} - \sqrt{\bar{\alpha}_T}} .
   $$

   At $t = 1$ the factor and the shift cancel exactly, and at $t = T$ the bracket is zero. They rescale in $\sqrt{\bar\alpha}$ rather than in SNR because it preserves the shape of the curve better.
2. Train with $v$-prediction, since $\epsilon$-prediction is trivial at $t = T$. There $v = \alpha_T\epsilon - \sigma_T x_0 = -x_0$, so the model learns to predict the prompt-conditional mean image from pure noise.
3. Make samplers start at $t = T$ ("trailing" timestep spacing; some DDIM implementations never visit $T$).
4. Rescale classifier-free guidance to avoid over-exposure (see the guidance section).

**Worked example: rescaling Stable Diffusion's schedule.**

| $t$ | old $\sqrt{\bar\alpha_t}$ | new $\sqrt{\bar\alpha_t}$ | old SNR | new SNR |
|---|---|---|---|---|
| 1 | 0.99957 | 0.99957 | 1175 | 1175 |
| 500 | 0.5269 | 0.4923 | 0.384 | 0.320 |
| 900 | 0.1196 | 0.0551 | 0.0145 | 0.00304 |
| 999 | 0.0687 | 0.00044 | 0.00474 | $2.0 \times 10^{-7}$ |
| 1000 | 0.0683 | 0 | 0.00468 | 0 |

The stretch factor is $0.99957/(0.99957 - 0.06826) = 1.0733$; for example at $t = 500$, $(0.5269 - 0.0683) \times 1.0733 = 0.4923$. The change is small in the middle and decisive at the end. VE schedules cannot reach zero SNR, because their signal scale is fixed at $\alpha_t = 1$.

**What this says.** A terminal SNR that is small but not zero is enough to leak the image's mean color into training. The fix makes the model's first input at sampling time identical to its training input at $t = T$, and it forces a target, $v$ or $x_0$, that is still informative when the input is pure noise.

::: key
**How to say it in an interview.** "A schedule is a curve in log-SNR. For training, it only sets the density of noise levels you sample, which is importance sampling for a fixed weighted objective, so its interaction with the loss weight is what matters. The cosine schedule fixed the linear schedule's waste of steps at near-pure noise. At higher resolution, neighboring pixels average away noise, so the effective SNR is higher and the schedule should be shifted toward more noise, by $2\log(64/d)$ in simple diffusion or SD3's timestep shift. And the last level must be pure noise, otherwise training and sampling see different inputs at $t = T$; that fix needs $v$- or $x_0$-prediction and a sampler that starts at $T$."
:::

::: qa
**Why should the noise schedule change with resolution?**

SNR is defined per pixel, but what matters for generation is how much of the image's structure survives. Neighboring pixels share signal while noise is independent, so downsampling by $s$ raises the effective SNR by $s^2$. A schedule tuned at 64 pixels therefore under-noises a 512-pixel image: global structure is still readable at what should be near-pure noise, and the model gets little training at the levels where layout is decided. Shifting log-SNR down by $2\log(d/64)$ (simple diffusion) or using SD3's timestep shift restores the balance. Some papers interpolate between shifts so fine detail still gets enough steps.
:::

::: qa
**What is zero terminal SNR, and which train/test mismatch does it fix?**

Schedules like Stable Diffusion's leave a little signal at $t = T$ ($\sqrt{\bar{\alpha}_T} \approx 0.068$). In training the model learns to read low-frequency information such as the mean color from that leak; at sampling the input is pure zero-mean noise, so outputs stay near medium brightness. Rescaling so $\bar{\alpha}_T = 0$ makes the training input at $t = T$ match the sampling input. It requires $v$- or $x_0$-prediction (at zero SNR, $\epsilon$-prediction is trivial and $\hat{x}_0$ undefined), a sampler that starts at $t = T$, and, per Lin et al., CFG rescaling to prevent over-exposure.
:::

::: qa
**What problem did the cosine schedule fix, and how?**

Under the linear schedule ($\beta$ from $10^{-4}$ to $0.02$ over 1000 steps), the SNR falls below one at step 260 and below $0.004$ for the last quarter of the steps, so a large share of training and sampling happens on almost pure noise. Nichol and Dhariwal showed that skipping up to the first 20% of the reverse process barely changed FID at low resolution. The cosine schedule sets $\bar\alpha_t = f(t)/f(0)$ with $f(t) = \cos^2\big(\frac{t/T + 0.008}{1.008}\cdot\frac{\pi}{2}\big)$, which makes log-SNR fall roughly linearly through the middle and puts $\mathrm{SNR} = 1$ near $t = T/2$. The offset keeps $\sqrt{\beta_1}$ just below one pixel bin, and $\beta_t$ is clipped at $0.999$, so it does not reach zero SNR unless the clip is removed. With $\epsilon$-prediction, its implied weight over log-SNR is $\operatorname{sech}(\lambda/2)$, peaked at $\mathrm{SNR} = 1$.
:::

::: pitfall
"The cosine schedule reaches zero SNR." Improved DDPM clips $\beta_t \le 0.999$, so its terminal SNR is small but positive; removing the clip fixes it. Also, the training schedule and the sampling time grid are separate choices.
:::

## Loss weighting and EDM preconditioning

Every objective above can be written as $\mathbb{E}_t\big[w_t\,\lVert x_0 - \hat x_0\rVert^2\big]$ for some weight $w_t$. Writing them all in this common unit shows what each objective really prioritizes, and makes Min-SNR and EDM's weighting easy to state.

### Every objective as a weighted clean-image loss

The conversion factors come from the parameterization section: $\lVert\epsilon - \hat\epsilon\rVert^2 = \mathrm{SNR}\,\lVert x_0 - \hat x_0\rVert^2$ and $\lVert v - \hat v\rVert^2 = (1 + \mathrm{SNR})\,\lVert x_0 - \hat x_0\rVert^2$. Each row of the table is one line of algebra from them.

| Objective | Weight on $\lVert x_0 - \hat x_0\rVert^2$ | Weight on $\lVert\epsilon - \hat\epsilon\rVert^2$ |
|---|---|---|
| $x_0$-MSE | $1$ | $1/\mathrm{SNR}$ |
| $\epsilon$-MSE ($L_{\text{simple}}$) | $\mathrm{SNR}$ | $1$ |
| $v$-MSE | $1 + \mathrm{SNR}$ | $(1 + \mathrm{SNR})/\mathrm{SNR}$ |
| DSM with weight $\sigma_t^2$ (score SDE) | $\mathrm{SNR}$ | $1$ |
| ELBO, discrete, $\varsigma_t^2 = \tilde\beta_t$ | $\tfrac{1}{2}\big(\mathrm{SNR}(t-1) - \mathrm{SNR}(t)\big)$ | $\beta_t/\big(2(1-\bar\alpha_{t-1})(1-\beta_t)\big)$ |
| ELBO, continuous, per unit of $\lambda$ | $\tfrac{1}{2}\mathrm{SNR}$ | $\tfrac{1}{2}$ |
| truncated SNR (Salimans and Ho) | $\max(\mathrm{SNR}, 1)$ | $\max(1, 1/\mathrm{SNR})$ |
| Min-SNR-$\gamma$ (Hang et al.) | $\min(\mathrm{SNR}, \gamma)$ | $\min(1, \gamma/\mathrm{SNR})$ |
| EDM, with $\mathrm{SNR} = 1/\sigma^2$ | $(\sigma^2 + \sigma_{\text{data}}^2)/(\sigma\,\sigma_{\text{data}})^2$ | $(\sigma^2 + \sigma_{\text{data}}^2)/\sigma_{\text{data}}^2$ |

Three rows deserve a derivation.

- *DSM.* The score SDE paper weights DSM by $\sigma_t^2$: $\sigma_t^2\lVert s_\theta + \epsilon/\sigma_t\rVert^2 = \lVert\epsilon - \epsilon_\theta\rVert^2$ with $s_\theta = -\epsilon_\theta/\sigma_t$, so it is the $\epsilon$-loss exactly.
- *Continuous ELBO.* VDM's loss is $-\tfrac{1}{2}\int\mathrm{SNR}'(t)\,\mathbb{E}\lVert x_0 - \hat x_0\rVert^2\,dt$. Since $\mathrm{SNR} = e^\lambda$, $\mathrm{SNR}'(t)\,dt = e^{\lambda}\,d\lambda$, so per unit of $\lambda$ the clean-image weight is $\tfrac{1}{2}e^\lambda = \tfrac{1}{2}\mathrm{SNR}$, and dividing by $\mathrm{SNR}$ gives the constant $\tfrac{1}{2}$ in noise units. This is Kingma and Gao's statement that the ELBO is the weighting $w(\lambda) = 1$.
- *EDM.* In VE form, $x = x_0 + \sigma\epsilon$ and $\hat\epsilon = (x - D)/\sigma$, so $\epsilon - \hat\epsilon = (D - x_0)/\sigma$ and $\lVert x_0 - D\rVert^2 = \sigma^2\lVert\epsilon - \hat\epsilon\rVert^2$. EDM's loss weight times $\sigma^2$ is $(\sigma^2 + \sigma_{\text{data}}^2)/\sigma_{\text{data}}^2$.

Remember that the weight seen in training is the table entry times how often that level is sampled. EDM draws $\ln\sigma \sim \mathcal{N}(-1.2, 1.2^2)$, and since $\lambda = \log(1/\sigma^2) = -2\ln\sigma$, that is $\lambda \sim \mathcal{N}(2.4, 2.4^2)$. Multiplying by the noise-unit weight, which is proportional to $\sigma^2 + \sigma_{\text{data}}^2 = e^{-\lambda} + 0.5^2$, gives EDM's total implied weighting $\mathcal{N}(\lambda; 2.4, 2.4^2)\cdot(e^{-\lambda} + 0.5^2)$, exactly the entry in Kingma and Gao's table.

### Why the $\epsilon$-loss over-weights low noise

In clean-image units, $L_{\text{simple}}$ weights each level by its SNR. Under the linear schedule that is $527$ at $t = 10$, $8.7$ at $t = 100$, $1.1$ at $t = 250$ and $0.085$ at $t = 500$: the $t = 10$ term counts about 6000 times as much as the $t = 500$ term. Low-noise denoising is easy and changes only fine detail, while the levels that decide layout and object shape get little weight. The true ELBO is even more extreme. This is the motivation for clamping.

### Min-SNR-$\gamma$

Hang et al. treat the denoising task at each timestep as one task of a multi-task problem and measure that gradients from different timesteps conflict. Solving for Pareto-optimal per-step weights during training is expensive, so they propose a fixed rule that comes close to it in their Pareto objective: clamp the clean-image weight,

$$
w_t = \min\big(\mathrm{SNR}(t),\ \gamma\big), \qquad \gamma = 5 \text{ by default}.
$$

For a network that outputs $\epsilon$ or $v$, divide by the factor the target already carries:

$$
w^{\epsilon}_t = \frac{\min(\mathrm{SNR}, \gamma)}{\mathrm{SNR}} = \min\Big(\frac{\gamma}{\mathrm{SNR}}, 1\Big), \qquad w^{v}_t = \frac{\min(\mathrm{SNR}, \gamma)}{\mathrm{SNR} + 1}.
$$

**Worked numbers (linear schedule, $\gamma = 5$).**

| $t$ | $\mathrm{SNR}$ | $w_t$ ($x_0$ units) | $w^\epsilon_t$ | $w^v_t$ |
|---|---|---|---|---|
| 10 | 527 | 5 | 0.0095 | 0.0095 |
| 100 | 8.71 | 5 | 0.574 | 0.515 |
| 250 | 1.10 | 1.10 | 1 | 0.524 |
| 500 | 0.0853 | 0.0853 | 1 | 0.0786 |
| 1000 | $4.0 \times 10^{-5}$ | $4.0 \times 10^{-5}$ | 1 | $4.0 \times 10^{-5}$ |

For an $\epsilon$-model the rule leaves every step with $\mathrm{SNR} \le 5$ untouched and down-weights only the cleanest ones. They report reaching FID 10 on ImageNet $256 \times 256$ about 3.4 times faster than the baseline weighting with a ViT-B, results that are not sensitive to $\gamma \in \{1, 5, 10, 20\}$, and that an $\epsilon$-output network trained with a constant clean-image weight or with the max-SNR weighting diverged.

### EDM preconditioning

This is a summary; the [EDM page](edm.html) reads the paper section by section, with the training ablations. Karras et al. (2022) write the denoiser as

$$
D_\theta(x;\sigma) = c_{\text{skip}}(\sigma)\,x + c_{\text{out}}(\sigma)\,F_\theta\big(c_{\text{in}}(\sigma)\,x;\ c_{\text{noise}}(\sigma)\big),
$$

with $x = x_0 + \sigma\epsilon$ and data standard deviation $\sigma_{\text{data}}$. The loss $\lambda(\sigma)\lVert D_\theta - x_0\rVert^2$ equals $\lambda(\sigma)\,c_{\text{out}}^2\,\lVert F_\theta - (x_0 - c_{\text{skip}}x)/c_{\text{out}}\rVert^2$, so $F_\theta$ is regressed onto the effective target $(x_0 - c_{\text{skip}}x)/c_{\text{out}}$ with effective weight $\lambda c_{\text{out}}^2$. ($\lambda(\sigma)$ here is EDM's loss weight, not the log-SNR.) The design rules:

- Unit-variance input: $\mathrm{Var}(x) = \sigma_{\text{data}}^2 + \sigma^2$, so $c_{\text{in}}^2(\sigma^2 + \sigma_{\text{data}}^2) = 1$ and $c_{\text{in}} = 1/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$.
- Unit-variance target: the target is $\big((1 - c_{\text{skip}})x_0 - c_{\text{skip}}\sigma\epsilon\big)/c_{\text{out}}$, so $c_{\text{out}}^2 = (1 - c_{\text{skip}})^2\sigma_{\text{data}}^2 + c_{\text{skip}}^2\sigma^2$.
- Amplify network errors as little as possible: minimize $c_{\text{out}}$ over $c_{\text{skip}}$. The derivative of $c_{\text{out}}^2$ is $-2(1 - c_{\text{skip}})\sigma_{\text{data}}^2 + 2c_{\text{skip}}\sigma^2$; setting it to zero gives $c_{\text{skip}}(\sigma^2 + \sigma_{\text{data}}^2) = \sigma_{\text{data}}^2$. Substituting back, $1 - c_{\text{skip}} = \sigma^2/(\sigma^2 + \sigma_{\text{data}}^2)$ and $c_{\text{out}}^2 = \frac{\sigma^4\sigma_{\text{data}}^2 + \sigma_{\text{data}}^4\sigma^2}{(\sigma^2 + \sigma_{\text{data}}^2)^2} = \frac{\sigma^2\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}$:

  $$
  c_{\text{skip}} = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}, \qquad c_{\text{out}} = \frac{\sigma\,\sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}} .
  $$

  The noise conditioning $c_{\text{noise}} = \tfrac{1}{4}\ln\sigma$ does not follow from these requirements; the paper says it was chosen empirically.

- Equalize the effective weight: $\lambda(\sigma) = 1/c_{\text{out}}^2 = (\sigma^2 + \sigma_{\text{data}}^2)/(\sigma\,\sigma_{\text{data}})^2$.

*Worked example.* At $\sigma = \sigma_{\text{data}} = 0.5$: $c_{\text{skip}} = 0.25/0.5 = 0.5$, $c_{\text{out}} = 0.25/0.7071 = 0.354$, $c_{\text{in}} = 1/0.7071 = 1.414$ and $\lambda = 8$. The denoiser keeps half of its input and lets the network supply the rest.

As $\sigma \to 0$, $c_{\text{skip}} \to 1$ and $D \approx x$ (the network only adds a small correction); as $\sigma \to \infty$, $c_{\text{skip}} \to 0$ and the network predicts $x_0$ directly. Training noise levels follow a log-normal, $\ln\sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$ with $P_{\text{mean}} = -1.2$, $P_{\text{std}} = 1.2$ and $\sigma_{\text{data}} = 0.5$, because the loss can only be reduced substantially at intermediate noise: at tiny $\sigma$ the noise is hard to detect and irrelevant, and at huge $\sigma$ the best answer is close to the dataset mean.

**What this says.** EDM separates three things older papers bundled: how the network's input and output are scaled (preconditioning), how much each level counts once it is sampled ($\lambda(\sigma)$, set so the effective weight is one), and how often each level is sampled (the training distribution of noise levels, which EDM calls $p_{\text{train}}$: the log-normal above, and where the real emphasis lives).

::: key
**How to say it in an interview.** "Put every objective in clean-image units. The $\epsilon$-loss weights a level by its SNR, the $v$-loss by $1 + \mathrm{SNR}$, the ELBO by half the SNR drop per step, and the $x_0$-loss uniformly. The $\epsilon$-loss therefore spends most of its weight on nearly clean inputs; Min-SNR caps the weight at $\gamma = 5$, which for an $\epsilon$-model means multiplying by $\min(1, \gamma/\mathrm{SNR})$. EDM normalizes the network's input and target to unit variance, picks the skip connection that minimizes error amplification, sets the per-level weight to make the effective weight one, and puts the emphasis in the log-normal sampling of noise levels."
:::

::: qa
**Derive EDM's $c_{\text{skip}}$, $c_{\text{out}}$ and $c_{\text{in}}$.**

With $x = x_0 + \sigma\epsilon$ and independent $x_0$ of variance $\sigma_{\text{data}}^2$, the input has variance $\sigma^2 + \sigma_{\text{data}}^2$, so $c_{\text{in}} = (\sigma^2 + \sigma_{\text{data}}^2)^{-1/2}$. The network's effective target is $(x_0 - c_{\text{skip}}x)/c_{\text{out}}$; requiring unit variance gives $c_{\text{out}}^2 = (1 - c_{\text{skip}})^2\sigma_{\text{data}}^2 + c_{\text{skip}}^2\sigma^2$. Minimizing this over $c_{\text{skip}}$ (so network errors are scaled up as little as possible) gives $c_{\text{skip}} = \sigma_{\text{data}}^2/(\sigma^2 + \sigma_{\text{data}}^2)$. Substituting back gives $c_{\text{out}} = \sigma\sigma_{\text{data}}/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$. Finally $\lambda = 1/c_{\text{out}}^2$ makes the effective loss weight one at every $\sigma$.
:::

::: qa
**Why does Min-SNR weighting speed up training?**

Under the $\epsilon$-loss, the implied $x_0$-space weight is the SNR, so low-noise timesteps dominate the gradient even though those denoising tasks are easy and matter little for global structure. Hang et al. measure that gradients from different timesteps conflict, and they frame training as multi-task learning. Clamping the weight at $\gamma$ caps the influence of high-SNR steps and leaves noisier steps unchanged, which is close to the Pareto-optimal weighting they compute directly. For an $\epsilon$-model the per-sample weight is $\min(\gamma/\mathrm{SNR}, 1)$, which down-weights only the cleanest timesteps.
:::

::: qa
**Write the $\epsilon$-loss, the $v$-loss and the ELBO as weighted clean-image losses.**

From $x_t = \alpha_t x_0 + \sigma_t\epsilon$, the noise error is $\epsilon - \hat\epsilon = -(\alpha_t/\sigma_t)(x_0 - \hat x_0)$, so the $\epsilon$-loss is the $x_0$-loss weighted by $\mathrm{SNR}$. With variance preservation, $v - \hat v = -(x_0 - \hat x_0)/\sigma_t$, so the $v$-loss carries weight $1/\sigma_t^2 = 1 + \mathrm{SNR}$. The discrete ELBO with the posterior variance gives weight $\tfrac{1}{2}(\mathrm{SNR}(t-1) - \mathrm{SNR}(t))$ per step, and in continuous time $\tfrac{1}{2}\mathrm{SNR}$ per unit of log-SNR, which is a constant $\tfrac{1}{2}$ in noise units. The weight that matters in practice is this table weight times the density with which training samples each level, which is why the schedule and the weighting must be discussed together.
:::

## Samplers

A sampler turns a trained denoiser into images. All the samplers here use the same network; they differ in how they step from one noise level to the next: with fresh noise or without, with one network call per step or several, and on which grid of levels.

### Ancestral DDPM sampling

Use the learned reverse Gaussian directly. Start at $x_T \sim \mathcal{N}(0, I)$ and for $t = T, \dots, 1$:

$$
x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\Big) + \varsigma_t z, \qquad z \sim \mathcal{N}(0, I),\quad \varsigma_t^2 \in \{\beta_t, \tilde{\beta}_t\},
$$

with no noise added at the final step. The mean is $\mu_\theta$ from the ELBO section, and the step is a discretization of the reverse SDE (the SDE section showed the same update written with the score). It needs roughly as many steps as training, $T = 1000$ network calls, because the Gaussian form of each step is only accurate for small steps.

### DDIM, in brief

*The full derivation, Theorem 1 and the experiments are on the [DDIM page](ddim.html); this is the self-contained summary.*

**The observation.** $L_{\text{simple}}$ uses only the marginals $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$, one $t$ at a time. It never uses how consecutive levels are linked. So any other "forward process" with the same marginals has the same loss and the same trained network, and its reverse process is an equally valid sampler.

**The family.** DDIM defines the process backwards, with one free noise scale $\varsigma_t$ per step:

$$
q_\varsigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(\sqrt{\bar{\alpha}_{t-1}}\,x_0 + k_t\,(x_t - \sqrt{\bar{\alpha}_t}\,x_0),\ \varsigma_t^2 I\big).
$$

Choose $k_t$ so the marginals stay DDPM's. Suppose $x_t \mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$. By Tool 1 (with $a = k_t$), $x_{t-1} \mid x_0$ is Gaussian with

$$
\begin{aligned}
\text{mean} &= \sqrt{\bar\alpha_{t-1}}\,x_0 + k_t\big(\sqrt{\bar\alpha_t}\,x_0 - \sqrt{\bar\alpha_t}\,x_0\big) = \sqrt{\bar\alpha_{t-1}}\,x_0 && \text{the bracket has mean zero}\\
\text{variance} &= \varsigma_t^2 + k_t^2(1 - \bar\alpha_t) && \text{Tool 1: } u + a^2 v .
\end{aligned}
$$

The mean is already right. Setting the variance equal to $1 - \bar\alpha_{t-1}$ gives $k_t = \sqrt{1-\bar{\alpha}_{t-1}-\varsigma_t^2}/\sqrt{1-\bar{\alpha}_t}$. Starting from $q_\varsigma(x_T \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_T}x_0, (1-\bar\alpha_T)I)$ and inducting downwards, every marginal matches.

**The update.** Replace the unknown $x_0$ by the network's $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$. Then $x_t - \sqrt{\bar\alpha_t}\hat x_0 = \sqrt{1-\bar\alpha_t}\,\hat\epsilon$, the factor $\sqrt{1-\bar\alpha_t}$ cancels against $k_t$'s denominator, and

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\varsigma_t^2}\;\hat{\epsilon} + \varsigma_t z, \qquad \varsigma_t(\eta) = \eta\sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}}\sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}.
$$

In words: predict the clean image, scale it to the next signal level, re-add the predicted noise direction for part of the next noise level, and fresh noise for the rest. Since $1 - \bar\alpha_t/\bar\alpha_{t-1} = \beta_t$, $\eta = 1$ gives $\varsigma_t^2 = \tilde\beta_t$, which is exactly DDPM's ancestral step. $\eta = 0$ is deterministic: that is DDIM.

**Why no retraining.** The DDIM paper's Theorem 1 shows that the variational objective of every member of the family equals a weighted $L_{\text{simple}}$ plus a constant. If each $t$ has its own parameters, a positive weight per term does not change each term's minimizer, so the $L_{\text{simple}}$ network is optimal for all of them.

**Why it can skip steps.** The construction only used the marginals at the two ends of a step, so $(t, t-1)$ can be replaced by any pair $(\tau_i, \tau_{i-1})$ from an increasing subsequence $\tau$ of $\{1, \dots, T\}$. Sampling then costs one network call per element of $\tau$.

**DDIM is Euler on the probability-flow ODE.** Set $\eta = 0$, insert $\hat x_0$, and divide by $\sqrt{\bar\alpha_{t-1}}$:

$$
\frac{x_{t-1}}{\sqrt{\bar\alpha_{t-1}}} = \frac{x_t}{\sqrt{\bar\alpha_t}} + \Big(\sqrt{\frac{1-\bar\alpha_{t-1}}{\bar\alpha_{t-1}}} - \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\Big)\hat\epsilon .
$$

With $\bar x = x/\sqrt{\bar\alpha}$ and $\bar\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha}$ this is $\bar x_{t-1} = \bar x_t + (\bar\sigma_{t-1} - \bar\sigma_t)\,\hat\epsilon$, one Euler step of $d\bar x/d\bar\sigma = \hat\epsilon$. In VE coordinates, $\hat\epsilon = (\bar x - D)/\bar\sigma$, so this is the ODE $dx/d\sigma = (x - D(x; \sigma))/\sigma$ from the SDE section.

**What this says.** On CIFAR-10 the deterministic sampler reached FID 13.36 with 10 steps where the $\eta = 1$ (DDPM) sampler gave 41.07, and 4.16 against 5.78 with 100 steps. Determinism also gives a fixed noise-to-image map, used for interpolation and (approximate) inversion.

### DPM-Solver: solve the linear part exactly

DDIM is first order. DPM-Solver (Lu et al. 2022) explains why it already works well and how to go to higher order. Write a VP-type schedule $x_t = \alpha_t x_0 + \sigma_t\epsilon$ (not necessarily $\alpha_t^2 + \sigma_t^2 = 1$), use $\ell_t = \log(\alpha_t/\sigma_t)$, half the log-SNR, and step from time $s$ to an earlier time $t < s$.

**Step 1: the ODE in noise form.** The SDE section showed the schedule corresponds to $f(t) = \frac{d\log\alpha_t}{dt}$ and $g(t)^2 = -2\sigma_t^2\frac{d\ell_t}{dt}$. Insert the score $\nabla\log p_t = -\hat\epsilon_\theta/\sigma_t$ into the probability-flow ODE:

$$
\frac{dx_t}{dt} = f(t)\,x_t - \tfrac{1}{2}g(t)^2\Big(-\frac{\hat\epsilon_\theta(x_t, t)}{\sigma_t}\Big) = f(t)\,x_t + \frac{g(t)^2}{2\sigma_t}\,\hat\epsilon_\theta(x_t, t) .
$$

The right side is linear in $x_t$ plus a nonlinear term (the network). A generic solver approximates both. DPM-Solver integrates the linear part exactly.

**Step 2: variation of constants.** Apply Tool 12 with $a = f$ and $b = \frac{g^2}{2\sigma}\hat\epsilon_\theta$:

$$
x_t = e^{\int_s^t f(r)\,dr}\,x_s + \int_s^t e^{\int_\tau^t f(r)\,dr}\,\frac{g(\tau)^2}{2\sigma_\tau}\,\hat\epsilon_\theta(x_\tau, \tau)\,d\tau .
$$

**Step 3: simplify the coefficients.** Since $f = \frac{d\log\alpha}{dt}$, $e^{\int_\tau^t f} = \alpha_t/\alpha_\tau$, and in particular $e^{\int_s^t f} = \alpha_t/\alpha_s$. Since $g^2 = -2\sigma^2\frac{d\ell}{d\tau}$, $\frac{g^2}{2\sigma} = -\sigma_\tau\frac{d\ell_\tau}{d\tau}$. So the integrand is

$$
\frac{\alpha_t}{\alpha_\tau}\Big(-\sigma_\tau\frac{d\ell_\tau}{d\tau}\Big)\hat\epsilon_\theta = -\alpha_t\,\frac{\sigma_\tau}{\alpha_\tau}\,\frac{d\ell_\tau}{d\tau}\,\hat\epsilon_\theta = -\alpha_t\,e^{-\ell_\tau}\,\frac{d\ell_\tau}{d\tau}\,\hat\epsilon_\theta ,
$$

using $\sigma_\tau/\alpha_\tau = e^{-\ell_\tau}$.

**Step 4: change variables to $\ell$.** $\ell_t$ is monotone in $t$, so substitute $\ell = \ell_\tau$, $d\ell = \frac{d\ell_\tau}{d\tau}d\tau$. Write $\hat x_\ell$ for the ODE solution at the time whose half-log-SNR is $\ell$ (DPM-Solver's notation; this hat does not mean a network estimate), and $\hat\epsilon_\theta(\hat x_\ell, \ell)$ for the network evaluated there:

$$
x_t = \frac{\alpha_t}{\alpha_s}\,x_s - \alpha_t\int_{\ell_s}^{\ell_t} e^{-\ell}\,\hat\epsilon_\theta(\hat x_\ell, \ell)\,d\ell .
$$

This is exact (the paper's Proposition 3.1). All the approximation is in one integral of the network output, weighted by $e^{-\ell}$ ("exponentially weighted").

**Step 5: first order is DDIM.** Let $h = \ell_t - \ell_s > 0$ and freeze the network at its value at $s$, $\hat\epsilon_s = \hat\epsilon_\theta(x_s, s)$:

$$
\begin{aligned}
\int_{\ell_s}^{\ell_t} e^{-\ell}\,d\ell &= e^{-\ell_s} - e^{-\ell_t} = e^{-\ell_t}\big(e^{h} - 1\big) && \text{integrate; factor out } e^{-\ell_t}\\
\alpha_t\,e^{-\ell_t}\big(e^h - 1\big) &= \sigma_t\big(e^h - 1\big) && \alpha_t e^{-\ell_t} = \sigma_t ,
\end{aligned}
$$

so DPM-Solver-1 is

$$
x_t = \frac{\alpha_t}{\alpha_s}\,x_s - \sigma_t\big(e^{h} - 1\big)\,\hat\epsilon_s .
$$

Compare DDIM with $\eta = 0$: $x_t = \alpha_t\hat x_0 + \sigma_t\hat\epsilon_s$ with $\hat x_0 = (x_s - \sigma_s\hat\epsilon_s)/\alpha_s$, which is $\frac{\alpha_t}{\alpha_s}x_s - \big(\frac{\alpha_t\sigma_s}{\alpha_s} - \sigma_t\big)\hat\epsilon_s$. Since $\frac{\alpha_t\sigma_s}{\alpha_s\sigma_t} = e^{\ell_t - \ell_s} = e^h$, the bracket is $\sigma_t(e^h - 1)$. The two updates are identical.

**Step 6: why this beats plain Euler.** Euler in $t$ freezes the whole right side, $f x + \frac{g^2}{2\sigma}\hat\epsilon$, over the step, so it makes an error on the linear part too, and that part changes by orders of magnitude between noise levels. DPM-Solver makes no error on the linear part, and the only frozen quantity is $\hat\epsilon_\theta$, which has unit scale and varies slowly in $\ell$. This is why DDIM is much better than Euler in $t$ at the same number of steps.

**Step 7: second order.** Expand the network to first order in $\ell$: $\hat\epsilon_\theta(\hat x_\ell, \ell) \approx \hat\epsilon_s + (\ell - \ell_s)\,\hat\epsilon^{(1)}_s$, where $\hat\epsilon^{(1)}$ is the total derivative along the path. The new term needs one more integral; with $\kappa = \ell - \ell_s$ and integration by parts,

$$
\begin{aligned}
\int_{\ell_s}^{\ell_t} e^{-\ell}(\ell - \ell_s)\,d\ell &= e^{-\ell_s}\int_0^h \kappa\,e^{-\kappa}\,d\kappa && \text{substitute } \ell = \ell_s + \kappa\\
&= e^{-\ell_s}\Big[-\kappa\,e^{-\kappa}\Big]_0^h + e^{-\ell_s}\int_0^h e^{-\kappa}\,d\kappa && \text{integration by parts}\\
&= e^{-\ell_s}\big(1 - e^{-h} - h\,e^{-h}\big) && \text{evaluate.}
\end{aligned}
$$

Multiplying by $\alpha_t$ and using $\alpha_t e^{-\ell_s} = \alpha_t e^{-\ell_t}e^{h} = \sigma_t e^{h}$ gives $\sigma_t(e^h - 1 - h)$, so the update that keeps both terms of the expansion is

$$
x_t \approx \frac{\alpha_t}{\alpha_s}\,x_s - \sigma_t\big(e^h - 1\big)\,\hat\epsilon_s - \sigma_t\big(e^h - 1 - h\big)\,\hat\epsilon^{(1)}_s .
$$

The derivative is unknown, so evaluate the network once more at the midpoint $\ell_s + h/2$: first take a DPM-Solver-1 step of size $h/2$ to get a point $u$ there, then use $\hat\epsilon^{(1)}_s \approx (\hat\epsilon_\theta(u) - \hat\epsilon_s)/(h/2)$. DPM-Solver-2 uses the slightly simpler combination $-\sigma_t(e^h - 1)\,\hat\epsilon_\theta(u)$. Since $\hat\epsilon_\theta(u) \approx \hat\epsilon_s + (h/2)\,\hat\epsilon^{(1)}_s$, this equals the line above except that the derivative's coefficient $(e^h - 1 - h)$ is replaced by $(e^h - 1)\,h/2$. The paper's appendix shows that the difference of the two coefficients, $(e^h - 1 - h) - (e^h - 1)h/2 = (2e^h - h - 2 - h e^h)/2$, is $O(h^3)$. Inserting $e^h = 1 + h + h^2/2 + h^3/6 + O(h^4)$ gives $\big(h^2/2 + h^3/6\big) - \big(h^2/2 + h^3/4\big) + O(h^4) = -h^3/12 + O(h^4)$; for example, at $h = 0.1$ it is $-8.76 \times 10^{-5}$ against $-h^3/12 = -8.33 \times 10^{-5}$. That is small enough to keep the method second order. The final update is

$$
u = \frac{\alpha_{t_{\text{mid}}}}{\alpha_s}\,x_s - \sigma_{t_{\text{mid}}}\big(e^{h/2} - 1\big)\,\hat\epsilon_s, \qquad x_t = \frac{\alpha_t}{\alpha_s}\,x_s - \sigma_t\big(e^{h} - 1\big)\,\hat\epsilon_\theta(u, t_{\text{mid}}),
$$

where $t_{\text{mid}}$ is the time with $\ell_{t_{\text{mid}}} = \ell_s + h/2$ (the paper calls it $s_i$; the $u$ formula is DPM-Solver-1 from $s$ to $t_{\text{mid}}$). Each step costs two network calls; DPM-Solver-3 costs three. The paper reports FID 4.70 with 10 network calls and 2.87 with 20 on CIFAR-10.

**What this says.** The probability-flow ODE of a diffusion model is semi-linear. Solving the linear part in closed form and expanding only the network output in half-log-SNR gives DDIM at first order and accurate 10 to 20 step samplers at second and third order, with no training.

### DPM-Solver++: the same idea with $\hat x_0$

With classifier-free guidance at large scales, high-order solvers built on $\hat\epsilon$ become unstable: guidance amplifies the network output and its derivatives, and the extrapolated predictions leave the range of real data. DPM-Solver++ (Lu et al. 2022b) rewrites the ODE in terms of the data prediction. Substitute $\hat\epsilon = (x - \alpha_t\hat x_0)/\sigma_t$ into the ODE of Step 1:

$$
\begin{aligned}
\frac{dx}{dt} &= f x + \frac{g^2}{2\sigma_t^2}\big(x - \alpha_t\hat x_0\big) && \text{substitute}\\
&= \Big(\frac{d\log\alpha_t}{dt} - \frac{d\ell_t}{dt}\Big)x + \frac{d\ell_t}{dt}\,\alpha_t\,\hat x_0 && \frac{g^2}{2\sigma^2} = -\frac{d\ell}{dt}\\
&= \frac{d\log\sigma_t}{dt}\,x + \frac{d\ell_t}{dt}\,\alpha_t\,\hat x_0 && \ell = \log\alpha - \log\sigma .
\end{aligned}
$$

Tool 12 with $a = \frac{d\log\sigma_t}{dt}$ and $b = \frac{d\ell_t}{dt}\alpha_t\hat x_\theta$, using $e^{\int_\tau^t d\log\sigma} = \sigma_t/\sigma_\tau$ and $\frac{\sigma_t}{\sigma_\tau}\alpha_\tau = \sigma_t e^{\ell_\tau}$, and then changing variables to $\ell$ as in Step 4, gives the exact solution

$$
x_t = \frac{\sigma_t}{\sigma_s}\,x_s + \sigma_t\int_{\ell_s}^{\ell_t} e^{\ell}\,\hat x_\theta(\hat x_\ell, \ell)\,d\ell .
$$

Freezing $\hat x_\theta$ gives the first-order step: $\sigma_t\int_{\ell_s}^{\ell_t}e^{\ell}\,d\ell = \sigma_t\big(e^{\ell_t} - e^{\ell_s}\big) = \sigma_t e^{\ell_t}\big(1 - e^{-h}\big) = \alpha_t\big(1 - e^{-h}\big)$, so $x_t = \frac{\sigma_t}{\sigma_s}x_s - \alpha_t(e^{-h} - 1)\,\hat x_\theta(x_s, s)$, which is again DDIM. The second-order multistep version, DPM-Solver++(2M), reuses the previous step's prediction instead of a new midpoint evaluation. Write the sampling times as $t_0 > t_1 > \dots > t_M$, $\tilde x_{t_i}$ for the solver's approximation of $x$ at time $t_i$, $h_i = \ell_{t_i} - \ell_{t_{i-1}}$ for the step in half-log-SNR, and $r_i = h_{i-1}/h_i$ for the ratio of consecutive steps. Then

$$
D_i = \Big(1 + \frac{1}{2r_i}\Big)\hat x_\theta(\tilde x_{t_{i-1}}, t_{i-1}) - \frac{1}{2r_i}\,\hat x_\theta(\tilde x_{t_{i-2}}, t_{i-2}), \qquad \tilde x_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}\,\tilde x_{t_{i-1}} - \alpha_{t_i}\big(e^{-h_i} - 1\big)\,D_i .
$$

$D_i$ (DPM-Solver++'s name; it is an estimate of the clean image, like EDM's denoiser output) is a linear extrapolation of the clean-image prediction to the middle of the step: writing $\hat x_{\theta, j} = \hat x_\theta(\tilde x_{t_j}, t_j)$, the slope of $\hat x_\theta$ in $\ell$ over the previous step is about $(\hat x_{\theta, i-1} - \hat x_{\theta, i-2})/h_{i-1}$, and moving half a step, $h_i/2$, along it adds $\frac{h_i}{2h_{i-1}} = \frac{1}{2r_i}$ times the difference. The first step, which has no previous prediction, uses the first-order update. Two advantages under guidance: $\hat x_0$ is bounded, so it can be clipped or dynamically thresholded, and a multistep method costs one network call per step, so for a fixed budget its steps are smaller. The paper reports good guided samples in 15 to 20 steps for pixel and latent models.

**What this says.** The noise form and the data form are the same ODE, solved exactly in their linear parts; they differ in which network output is assumed to vary slowly. Under strong guidance $\hat x_0$ is the better-behaved quantity, and reusing past evaluations (multistep) gets second order at the cost of one call per step.

### EDM's Heun sampler, in brief

*Derivations and ablations are on the [EDM page](edm.html).* EDM integrates $dx/d\sigma = (x - D(x; \sigma))/\sigma$ on a grid that concentrates steps at low noise:

$$
\sigma_i = \Big(\sigma_{\max}^{1/\rho} + \tfrac{i}{N-1}\big(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\big)\Big)^\rho, \qquad \rho = 7,\ \sigma_{\min} = 0.002,\ \sigma_{\max} = 80,
$$

followed by $\sigma_N = 0$. Each step is Euler plus a trapezoidal correction:

$$
\begin{aligned}
d_i &= \frac{x_i - D(x_i;\sigma_i)}{\sigma_i}, \qquad \tilde{x} = x_i + (\sigma_{i+1} - \sigma_i)\,d_i, \\
x_{i+1} &= x_i + (\sigma_{i+1} - \sigma_i)\,\tfrac{1}{2}\Big(d_i + \frac{\tilde{x} - D(\tilde{x};\sigma_{i+1})}{\sigma_{i+1}}\Big),
\end{aligned}
$$

skipping the correction on the final step to $\sigma = 0$, so $N$ steps cost $2N - 1$ evaluations. EDM reached FID 1.79 on class-conditional CIFAR-10 with 35 evaluations. Heun is second order in the step size; the cost is two network calls per step.

### Worked example: deterministic samplers on Gaussian data

The Gaussian example of the SDE section can be run through the samplers exactly. Data $\mathcal{N}(m, 0.5^2)$; the exact denoiser is $D(x; \sigma) = m + \frac{0.25}{0.25 + \sigma^2}(x - m)$, so the ODE's slope is $d(x; \sigma) = \sigma(x - m)/(0.25 + \sigma^2)$. Start from the exact noisy marginal, $x(80) \sim \mathcal{N}(m, 0.25 + 80^2)$. Every sampler maps $x - m$ to a multiple of itself, so the output standard deviation can be computed exactly; the true answer is $0.5$. The table uses EDM's $\rho = 7$ grid with $N$ levels from $80$ to $0.002$, then a final step to $0$. All numbers in this subsection are for this one-dimensional toy, not measurements on image models.

| Levels $N$ | DDIM (Euler in $\sigma$): output std | network calls | Heun: output std | network calls |
|---|---|---|---|---|
| 1 | 0.0031 | 1 | 0.0031 | 1 |
| 3 | 0.098 | 3 | 2.05 | 5 |
| 5 | 0.234 | 5 | 1.26 | 9 |
| 10 | 0.365 | 10 | 0.610 | 19 |
| 18 | 0.423 | 18 | 0.528 | 35 |
| 50 | 0.472 | 50 | 0.503 | 99 |

Read it in three parts.

- *One step returns the posterior mean.* With $N = 1$ the only step goes from $\sigma = 80$ to $0$: $x_0 = x + (0 - 80)\,(x - D)/80 = D(x; 80)$. For this data $D(x; 80) - m = \frac{0.25}{6400.25}(x - m)$, so a typical input with $x - m = 80$ lands $0.0031$ from the mean. Every sample is almost the dataset mean: the one-dimensional version of a blurry one-step image.
- *Euler is first order.* Its error falls roughly in proportion to $1/N$ (0.077 at 18 levels, 0.028 at 50), and in this example it undershoots at every $N$. The reason can be checked by hand. One Euler step from $\sigma_i$ to $\sigma_{i+1}$ multiplies $x - m$ by $1 - (\sigma_i - \sigma_{i+1})\,\sigma_i/(0.25 + \sigma_i^2) = (0.25 + \sigma_i\sigma_{i+1})/(0.25 + \sigma_i^2)$, while the exact solution of the ODE (the formula in the SDE section's Gaussian example) multiplies it by $\sqrt{(0.25 + \sigma_{i+1}^2)(0.25 + \sigma_i^2)}\,/\,(0.25 + \sigma_i^2)$. By the Cauchy-Schwarz inequality, $0.5 \cdot 0.5 + \sigma_i\sigma_{i+1} \le \sqrt{(0.25 + \sigma_i^2)(0.25 + \sigma_{i+1}^2)}$, so every Euler step shrinks the sample toward the mean a little more than it should.
- *Heun is second order, but needs small enough steps.* With very few levels, the corrector evaluates the slope at a tiny $\sigma$ where it is nearly zero, averages it with the large slope, and moves only half way: 3 levels overshoot to a standard deviation of 2.05. From about 18 levels on, the error falls roughly as $1/N^2$ (0.028 at 18, 0.003 at 50). At a matched budget of about 35 network calls, Euler with 35 levels gives 0.460 (error 0.040) and Heun with 18 gives 0.528 (error 0.028); at about 99 calls the errors are 0.014 and 0.003. DPM-Solver-2 also fits in 35 calls (17 two-call steps on the 18 levels, then a one-call final step to 0) and gives 0.516. In VE coordinates ($\alpha_t = 1$, $\ell = -\log\sigma$) it reduces to the midpoint method in $\sigma$ with the extra evaluation at the geometric mean $\sqrt{\sigma_i\sigma_{i+1}}$, the midpoint in $\log\sigma$.

**What this says.** In this toy, few-step blur is not a network failure (the denoiser is exact): it is what a deterministic solver does with large steps, because each step is aimed at a posterior mean. Higher-order solvers fix it only once the steps are small enough for the slope to change little across a step. The same mechanism is one reason why, for real image models, solvers degrade sharply below roughly 10 steps and distillation is used instead.

### Stochastic versus deterministic sampling

Fresh noise (ancestral sampling, EDM's "churn") acts like a Langevin correction that repairs earlier errors; EDM's stochastic sampler took a pretrained ImageNet-64 model from FID 2.07 to 1.55. For EDM's best CIFAR-10 models, deterministic sampling was better. Quality improves with steps with diminishing returns: ODE solvers do well with tens of steps, but below roughly ten the trajectory's curvature dominates and distillation is the usual answer ([flow matching](flow-matching.html)).

Why noise helps: with an imperfect score, the sample drifts off the distribution the network was trained on. The $\zeta$-family of the SDE section shows that a reverse SDE adds an extra pull $\tfrac{\zeta^2}{2}g^2\nabla\log p_t$ toward high density together with matching noise; that combination is a Langevin step at the current level, which moves samples back toward $p_t$ and so forgets part of the earlier error. Why it costs: the noise must be removed again, which needs smaller steps, and too much of it erodes detail. EDM found loss of detail and color drift with excessive churn, so the amount had to be tuned per model.

::: key
**How to say it in an interview.** "Ancestral DDPM samples the learned Gaussian reverse steps and needs about $T$ steps. DDIM keeps DDPM's marginals but makes the reverse step deterministic, so the same network samples in tens of steps; it is the Euler method on the probability-flow ODE in rescaled coordinates. DPM-Solver writes that ODE as a linear part plus the network term, solves the linear part exactly with variation of constants, and expands only $\hat\epsilon$ in half-log-SNR; its first order is DDIM and its second order gives good samples in 10 to 20 calls, and DPM-Solver++ does the same with $\hat x_0$, which behaves better under guidance. EDM uses Heun on a $\rho = 7$ grid. Stochastic samplers correct accumulated errors and win when steps are plentiful; deterministic high-order solvers win when steps are scarce or when you need inversion."
:::

::: qa
**Derive DDIM. Why can it skip steps and reuse a DDPM-trained model?**

Define a family of forward processes whose conditional $q(x_{t-1} \mid x_t, x_0)$ is Gaussian with mean $\sqrt{\bar{\alpha}_{t-1}}x_0 + k_t(x_t - \sqrt{\bar{\alpha}_t}x_0)$ and variance $\varsigma_t^2$, and pick $k_t = \sqrt{1-\bar{\alpha}_{t-1}-\varsigma_t^2}/\sqrt{1-\bar{\alpha}_t}$ so that every marginal $q(x_t \mid x_0)$ matches DDPM's. Replacing $x_0$ and $\epsilon$ with network predictions gives the DDIM update. Since $L_{\text{simple}}$ only involves the marginals, the trained network is valid for every member of the family. The construction only uses the marginals at the two endpoints of a step, so any decreasing subsequence of timesteps works. With $\varsigma_t = 0$ the map is deterministic, which also makes approximate inversion possible.
:::

::: qa
**How do DDIM, the probability-flow ODE and DPM-Solver relate?**

Deterministic DDIM is a first-order solver of the probability-flow ODE: in coordinates $\bar{x} = x/\alpha_t$, $\bar{\sigma} = \sigma_t/\alpha_t$ it is exactly an Euler step of $d\bar{x}/d\bar{\sigma} = \hat{\epsilon}$. DPM-Solver writes the VP ODE as a linear term plus a nonlinear term, solves the linear term exactly, and expresses the rest as an exponentially weighted integral of $\hat{\epsilon}$ over half-log-SNR. Its first-order version coincides with DDIM; higher orders approximate $\hat{\epsilon}$ along the half-log-SNR $\ell_t = \log(\alpha_t/\sigma_t)$ with a Taylor expansion. That is why DDIM already beats naive Euler in $t$, and why second and third order buy another large reduction in steps.
:::

::: qa
**When would you use a stochastic sampler instead of a deterministic one?**

Use stochastic sampling when the model is imperfect and you can afford more steps: the injected noise lets the sampler correct earlier errors, which often gives the best FID at large step counts. Use deterministic ODE solvers when steps are scarce, when you need a reproducible noise-to-image map (inversion, interpolation, editing, consistent seeds), or when you want likelihoods. Too much stochasticity has costs: EDM saw loss of detail and color drift, and the best settings had to be tuned per model. In practice, high-order deterministic solvers dominate the low-step regime and mildly stochastic samplers the high-step regime.
:::

::: qa
**Derive DPM-Solver-1 and show that it is DDIM.**

Write the probability-flow ODE in noise form, $dx/dt = f(t)x + \frac{g(t)^2}{2\sigma_t}\hat\epsilon_\theta$, with $f = d\log\alpha_t/dt$ and $g^2 = -2\sigma_t^2\,d\ell_t/dt$, where $\ell_t = \log(\alpha_t/\sigma_t)$. Variation of constants solves the linear part exactly: $x_t = \frac{\alpha_t}{\alpha_s}x_s - \alpha_t\int_{\ell_s}^{\ell_t}e^{-\ell}\hat\epsilon_\theta\,d\ell$. Freezing $\hat\epsilon_\theta$ at its value at $s$ and integrating gives $x_t = \frac{\alpha_t}{\alpha_s}x_s - \sigma_t(e^h - 1)\hat\epsilon_s$ with $h = \ell_t - \ell_s$. DDIM's $x_t = \alpha_t\hat x_0 + \sigma_t\hat\epsilon_s$ with $\hat x_0 = (x_s - \sigma_s\hat\epsilon_s)/\alpha_s$ expands to the same thing because $\alpha_t\sigma_s/(\alpha_s\sigma_t) = e^h$. Higher orders Taylor-expand $\hat\epsilon_\theta$ in $\ell$ and estimate the derivatives with extra evaluations.
:::

::: qa
**Why does ancestral DDPM sampling need so many steps, and what does a single deterministic step from pure noise produce?**

Each ancestral step samples a Gaussian, and the true reverse step is close to Gaussian only when the step is small; with large jumps it is a mixture over all plausible clean images, and a Gaussian at its mean blurs the modes. A deterministic step aims at the posterior mean: a DDIM step from $\sigma_{\max}$ straight to zero returns exactly $D(x; \sigma_{\max})$, the average of everything consistent with nearly pure noise, which is close to the dataset mean. For Gaussian data with standard deviation 0.5, that single step gives samples with standard deviation 0.003; 10 Euler steps give 0.37 and 50 give 0.47. More steps, a higher-order solver or distillation are the remedies.
:::

## Guidance

A conditional diffusion model samples $p(x \mid c)$. In practice that is not what people want: samples from the exact conditional are diverse but often only loosely match the condition and include many low-quality images. Guidance changes the sampler to trade some diversity for fidelity. This section derives classifier guidance from Bayes' rule, derives classifier-free guidance from it, shows with a solvable toy why diversity drops and what is actually sampled, and covers the fixes for high guidance scales and autoguidance.

### Classifier guidance, from Bayes' rule

By Tool 8, the conditional score splits into an unconditional score and a classifier gradient:

$$
\nabla_{x_t}\log p(x_t \mid c) = \nabla_{x_t}\log p(x_t) + \nabla_{x_t}\log p(c \mid x_t).
$$

Dhariwal and Nichol (2021) train a classifier $p_\phi(c \mid x_t)$ with parameters $\phi$ on noisy images at all levels and add its gradient, multiplied by a "gradient scale" $s$, to a diffusion model's score. The CFG paper writes the same construction with weight $w$ and starts from a conditional model, so the guided score is $\nabla\log p(x_t \mid c) + w\nabla\log p_\phi(c \mid x_t)$. Convert to noise units with $\epsilon = -\sigma_t\times\text{score}$:

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, c) - w\,\sigma_t\,\nabla_{x_t}\log p_\phi(c \mid x_t) \approx -\sigma_t\nabla_{x_t}\big[\log p(x_t \mid c) + w\log p_\phi(c \mid x_t)\big],
$$

which targets $p(x_t \mid c)\,p_\phi(c \mid x_t)^w$. Because $p(x_t \mid c) \propto p(x_t)\,p(c \mid x_t)$, this equals $p(x_t)\,p_\phi(c \mid x_t)^{1+w}$ up to normalization: guiding a conditional model with weight $w$ is in theory the same as guiding an unconditional model with weight $1 + w$. In ADM's DDIM sampler, the guided noise is $\hat\epsilon - \sqrt{1-\bar\alpha_t}\,s\,\nabla\log p_\phi(c \mid x_t)$ with scale $s$, the same formula with $\sigma_t = \sqrt{1-\bar\alpha_t}$.

*ADM's derivation for the ancestral sampler* uses the same Taylor-and-complete-the-square trick as the reverse-SDE heuristic. A conditional reverse step is proportional to $p_\theta(x_{t-1} \mid x_t)\,p_\phi(c \mid x_{t-1})$. Write the model step as $\mathcal{N}(\mu, \Sigma)$, with $\Sigma$ diagonal as in DDPM and ADM, and expand $\log p_\phi(c \mid x_{t-1}) \approx \text{const} + (x_{t-1} - \mu)^\top g_\phi$ around $\mu$, with $g_\phi = \nabla\log p_\phi(c \mid x)\vert_{x = \mu}$ the classifier gradient (ADM calls it $g$; renamed here because $g(t)$ is the SDE's diffusion coefficient); this is accurate when $\Sigma$ is small. Then

$$
\begin{aligned}
\log\big[p_\theta(x_{t-1} \mid x_t)\,p_\phi(c \mid x_{t-1})\big] &\approx -\tfrac{1}{2}(x_{t-1} - \mu)^\top\Sigma^{-1}(x_{t-1} - \mu) + (x_{t-1} - \mu)^\top g_\phi + \text{const}\\
&= -\tfrac{1}{2}(x_{t-1} - \mu - \Sigma g_\phi)^\top\Sigma^{-1}(x_{t-1} - \mu - \Sigma g_\phi) + \text{const} && \text{Tool 6, one coordinate at a time,}
\end{aligned}
$$

so the guided step is $\mathcal{N}(\mu + \Sigma g_\phi, \Sigma)$: the same Gaussian with its mean shifted by the covariance times the classifier gradient. ADM samples $\mathcal{N}(\mu + s\,\Sigma\,\nabla\log p_\phi(c \mid x_t), \Sigma)$.

**What this says.** Whether you shift the score (deterministic samplers) or shift the mean of each Gaussian step by $\Sigma g_\phi$ (ancestral sampler), classifier guidance moves each step uphill on the classifier's log-probability, scaled by the step's own noise level.

*Why the scale matters.* $s\,\nabla\log p_\phi = \nabla\log\big(p_\phi^s/Z\big)$, so the scale sharpens the classifier's distribution. On unconditional ImageNet models, a scale of 1 gave samples to which the classifier assigned about 50% probability for the requested class but which did not look like that class; a scale of 10 gave class-consistent images. Raising the scale traded recall (diversity) for precision and Inception score. Their guided ImageNet $256 \times 256$ model reached FID 4.59 with 250 steps (3.94 with an upsampling stack). The drawbacks: a separate classifier must be trained on noisy inputs, and following its gradients can behave like an adversarial attack on it rather than adding meaningful features.

### Classifier-free guidance

Ho and Salimans (2022) remove the classifier.

**Training.** Train one network both conditionally and unconditionally: for each example, replace $c$ by a null token $\varnothing$ with probability $p_{\text{uncond}}$. Then $\epsilon_\theta(x_t, c)$ estimates the conditional noise and $\epsilon_\theta(x_t, \varnothing)$, written $\epsilon_\theta(x_t)$, the unconditional one. They tried $p_{\text{uncond}} \in \{0.1, 0.2, 0.5\}$; $0.1$ and $0.2$ worked about equally well, and $0.5$ was worse across the whole FID/IS trade-off.

**The implicit classifier.** Bayes' rule gives $p(c \mid x_t) = p(x_t \mid c)\,p(c)/p(x_t)$. Define the implicit classifier $p^i(c \mid x_t) \propto p(x_t \mid c)/p(x_t)$; the factor $p(c)$ does not depend on $x_t$. With exact noise predictors $\epsilon^*$, each score is $-\epsilon^*/\sigma_t$, so

$$
\nabla\log p^i(c \mid x_t) = \nabla\log p(x_t \mid c) - \nabla\log p(x_t) = -\frac{1}{\sigma_t}\big(\epsilon^*(x_t, c) - \epsilon^*(x_t)\big).
$$

**The guided prediction.** Plug this gradient into classifier guidance on the conditional model:

$$
\begin{aligned}
\tilde{\epsilon} &= \epsilon(x_t, c) - w\,\sigma_t\Big(-\frac{1}{\sigma_t}\big(\epsilon(x_t, c) - \epsilon(x_t)\big)\Big) && \text{substitute the implicit gradient}\\
&= \epsilon(x_t, c) + w\big(\epsilon(x_t, c) - \epsilon(x_t)\big) && \text{the } \sigma_t \text{ cancels}\\
&= (1+w)\,\epsilon(x_t, c) - w\,\epsilon(x_t) && \text{collect terms.}
\end{aligned}
$$

In score form, the guided score is $(1 + w)\nabla\log p(x_t \mid c) - w\nabla\log p(x_t)$, the score of $p(x_t \mid c)^{1+w}/p(x_t)^w$. With $\epsilon_c = \epsilon_\theta(x_t, c)$ and $\epsilon_u = \epsilon_\theta(x_t, \varnothing)$, most code writes $\epsilon_u + s_{\text{cfg}}(\epsilon_c - \epsilon_u)$ with $s_{\text{cfg}} = 1 + w$ ($s_{\text{cfg}} = 1$ means no guidance); expanding gives $(1 + w)\epsilon_c - w\epsilon_u$, the same thing. Each step costs two forward passes, usually batched. A negative prompt simply replaces $\epsilon_u$ with the prediction under that prompt.

**The same formula works for any target.** The guided combination has weights $(1 + w)$ and $-w$, which sum to one, so it commutes with any affine conversion that uses the same $x_t$. For example, for the clean image:

$$
(1+w)\,\frac{x_t - \sigma_t\epsilon_c}{\alpha_t} - w\,\frac{x_t - \sigma_t\epsilon_u}{\alpha_t} = \frac{x_t - \sigma_t\big[(1+w)\epsilon_c - w\epsilon_u\big]}{\alpha_t} .
$$

So guiding $\hat x_0$, $v$ or EDM's $D$ with the same weights gives the same sampler.

**What this says.** Classifier-free guidance is classifier guidance with the classifier replaced by the ratio of the model's own conditional and unconditional densities. Its update is an extrapolation from the unconditional prediction through the conditional one, by a factor $1 + w$.

**A caveat in the paper.** With learned networks, $\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)$ is in general not the gradient of any classifier, because unconstrained networks do not output conservative vector fields. CFG is "inspired by" an implicit classifier rather than equal to one.

**Results.** On ImageNet, the best FID came from small guidance ($w = 0.1$ at $64 \times 64$, $w = 0.3$ at $128 \times 128$), while Inception score kept rising up to the largest weight tried, $w = 4$. At $64 \times 64$ with $p_{\text{uncond}} = 0.1$, FID went from 1.80 unguided to 1.55 at $w = 0.1$ and up to 12.6 at $w = 1$. Text-to-image systems use $s_{\text{cfg}}$ of roughly 5 to 10 because prompt alignment and perceived quality matter more than coverage.

### Why fidelity goes up and diversity down

The target $p(x \mid c)\,p(c \mid x)^w \propto p(x \mid c)^{1+w}/p(x)^w$ up-weights samples where the condition is much more likely than under the unconditional model: prototypical, unambiguous examples. Raising a density to a power above one also sharpens it, like lowering temperature. Both effects increase how recognizable each sample is (fidelity, Inception score, prompt alignment) and reduce how much of the conditional distribution is covered (diversity, recall), which is why FID, which needs both, is best at a small weight.

**Worked example: a solvable toy.** Let the conditional model be $\mathcal{N}(1, 1)$ and the unconditional model $\mathcal{N}(0, 2^2)$ (it spreads over all classes). The tilted target at weight $w$ has log-density

$$
(1+w)\Big(-\frac{(x - 1)^2}{2}\Big) - w\Big(-\frac{x^2}{8}\Big) + \text{const} = -\frac{1}{2}\Big[\Big((1+w) - \frac{w}{4}\Big)x^2 - 2(1+w)\,x\Big] + \text{const}.
$$

To get the right side, expand $(x - 1)^2 = x^2 - 2x + 1$, drop the constant, and collect: the coefficient of $x^2$ inside the bracket is $(1 + w) - w/4$ and the coefficient of $-2x$ is $(1 + w)$. By Tool 6 it is Gaussian with precision $(1 + w) - w/4$ and mean $(1 + w)$ divided by that precision. At $w = 3$ ($s_{\text{cfg}} = 4$): precision $4 - 0.75 = 3.25$, standard deviation $1/\sqrt{3.25} = 0.555$, mean $4/3.25 = 1.23$. In this toy, guidance narrows the distribution (from 1 to 0.555) and pushes it away from where the unconditional model puts mass (from 1 to 1.23).

### The nuance: guided sampling does not sample the tilted distribution

The guided score at noise level $\sigma$ is $(1+w)\nabla\log p_\sigma(x \mid c) - w\nabla\log p_\sigma(x)$, built from the noised conditional and unconditional densities. That is not the score of the noised tilted distribution, because noising does not commute with taking powers and ratios of densities. So the sampler does not follow the tilted family, and different samplers give different answers.

The toy shows it exactly. In VE coordinates ($x = x_0 + \sigma\epsilon$, time $t = \sigma$, so $g^2 = 2\sigma$ as in the SDE section) the noised models are $\mathcal{N}(1, 1 + \sigma^2)$ and $\mathcal{N}(0, 4 + \sigma^2)$ by Tool 1. By Tool 7 the guided score is linear in $x$:

$$
(1+w)\Big(-\frac{x - 1}{1 + \sigma^2}\Big) - w\Big(-\frac{x}{4 + \sigma^2}\Big) = -a(\sigma)\,x + b(\sigma), \qquad a = \frac{1+w}{1+\sigma^2} - \frac{w}{4+\sigma^2},\quad b = \frac{1+w}{1+\sigma^2}.
$$

A linear update keeps Gaussians Gaussian, so it is enough to track the mean $\bar m$ and variance $V$ of the samples while $\sigma$ decreases. The guided probability-flow ODE, $dx/d\sigma = -\sigma(-a x + b)$, gives $d\bar m/d\sigma = \sigma(a\bar m - b)$ and $dV/d\sigma = 2\sigma a V$. One step of the guided reverse SDE from $\sigma$ down to $\sigma - \delta$ is $x \leftarrow x + 2\sigma(-a x + b)\,\delta + \sqrt{2\sigma\delta}\,z$, which gives $d\bar m/d\sigma = 2\sigma(a\bar m - b)$ and $dV/d\sigma = 4\sigma a V - 2\sigma$ (the last term is the injected noise). Both start from the conditional model's marginal at $\sigma = 80$, $\mathcal{N}(1, 1 + 80^2)$. Integrating these equations numerically down to $\sigma = 0$:

| Guidance | Target $p(x \mid c)^{1+w}/p(x)^w$ | Guided reverse SDE | Guided ODE |
|---|---|---|---|
| $w = 0$ | mean 1.00, std 1.00 | mean 1.00, std 1.00 | mean 1.00, std 1.00 |
| $w = 1$ ($s_{\text{cfg}} = 2$) | mean 1.14, std 0.76 | mean 1.25, std 0.66 | mean 1.37, std 0.50 |
| $w = 3$ ($s_{\text{cfg}} = 4$) | mean 1.23, std 0.55 | mean 1.33, std 0.44 | mean 1.51, std 0.13 |

Without guidance all three agree, as they must. With guidance, in this toy the stochastic and deterministic samplers both land on distributions that are narrower and further from the unconditional mass than the tilted target, and they disagree with each other. (Method: the four linear equations above, integrated with a stiff solver at tight tolerance; at $w = 0$ the integration reproduces $\mathcal{N}(1, 1)$, a check on the setup, and a direct simulation of the guided SDE with 200,000 samples gives mean 1.33 and standard deviation 0.44 at $w = 3$.) The direction and size of these gaps are specific to this toy; the general statement is the one below. Bradley and Nakkiran (2024) prove the general version: CFG with DDPM and with DDIM sample different distributions, and neither is the "gamma-powered" target; in the SDE limit CFG equals a DDIM step for the conditional model followed by Langevin steps toward a sharpened distribution. Karras et al. (2024) make the same point: the guided density at noise level $\sigma$ is not a valid diffusion of the guided density at zero noise, so trajectories are distorted and truncation is exaggerated. In practice, guidance at high noise pulls many seeds toward a few template layouts, and at high scale the extrapolated prediction leaves the range of real data, giving oversaturated, over-exposed images.

### Fixes and refinements

- *Dynamic thresholding* (Imagen, pixel space): set $s$ to a high percentile of $\lvert\hat{x}_0\rvert$ per sample; if $s > 1$, clip $\hat{x}_0$ to $[-s, s]$ and divide by $s$.
- *CFG rescale* (Lin et al.): with $x_{\text{pos}}$ the network's prediction under the prompt and $x_{\text{cfg}}$ the guided prediction, match the standard deviation (over all entries of one sample) of the guided output to the conditional one, $x_{\text{rescaled}} = x_{\text{cfg}}\cdot\operatorname{std}(x_{\text{pos}})/\operatorname{std}(x_{\text{cfg}})$, then blend $x_{\text{final}} = \phi\,x_{\text{rescaled}} + (1-\phi)\,x_{\text{cfg}}$ with $\phi = 0.7$ (a blend factor, unrelated to the classifier parameters $\phi$).
- *Adaptive projected guidance* (Sadat et al.): with $D_c$ and $D_u$ the conditional and unconditional denoised predictions, split the guidance update $D_c - D_u$ into components parallel and orthogonal to $D_c$; the parallel part mostly increases saturation, like a gain on the image, so down-weight it.
- *Guidance interval* (Kynkäänniemi et al.): apply guidance only in a middle range of noise levels (details in the question below).
- *Autoguidance* (Karras et al. 2024): guide with a worse version of the same conditional model, $D_w = w\,D_1(x;\sigma,c) + (1-w)\,D_0(x;\sigma,c)$, where $D_0$ is smaller and/or trained for less time on the same task and conditioning. It reached FID 1.01 on ImageNet-64 and 1.25 on ImageNet-512, and it also works for unconditional models. In the EDM2-S ablation the best guide was one size smaller (XS) and trained for 1/16 as long; the 1.25 result guided EDM2-XXL with an M-size model.

### Autoguidance, in more detail

Karras et al. write any guidance as an extrapolation between two denoisers, $D_w = w D_1 + (1 - w) D_0$. Note the convention: here $w = 1$ means no guidance, so their $w$ is $s_{\text{cfg}}$, not Ho and Salimans' $w$. Let $p_1(x \mid c; \sigma)$ and $p_0(x \mid c; \sigma)$ be the noisy densities that $D_1$ and $D_0$ implicitly model, so that $D_i \approx x + \sigma^2\nabla\log p_i$ (Tweedie in VE form). Then $D_w = x + \sigma^2\big[w\nabla\log p_1 + (1 - w)\nabla\log p_0\big]$, and with $wa + (1-w)b = a + (w-1)(a - b)$ the bracket, the score that the guided sampler follows, is

$$
\nabla\log p_1(x \mid c; \sigma) + (w - 1)\,\nabla\log\frac{p_1(x \mid c; \sigma)}{p_0(x \mid c; \sigma)} .
$$

The second term pushes samples toward where the main model's density is high relative to the guiding model's. With CFG, $p_0$ is the unconditional model's density $p_0(x; \sigma)$, so this term mixes two effects: it raises the class likelihood (narrowing toward prototypes), and, because the unconditional model is a worse fit to the data (it has a harder task and a small share of the training), it also pulls samples away from regions both models fit poorly, toward the core of the data manifold. Karras et al. argue that the second effect is what improves image quality.

Autoguidance keeps only that effect: $D_0$ is a weaker model trained on the same task and conditioning, so the ratio $p_1/p_0$ mostly flags regions the weak model gets wrong, not class boundaries. Results on ImageNet-512 with EDM2: FID 2.56 to 1.34 for EDM2-S (guided by an XS model trained $1/16$ as long, an extra 3.6% of training cost), 1.25 for EDM2-XXL guided by an M model, and 1.01 on ImageNet-64. For an unconditional EDM2-S model, FID fell from 11.67 to 3.86. Autoguidance needs a second network, does not by itself improve prompt adherence, and its guidance weight and EMA lengths were tuned per setup by grid search.

::: key
**How to say it in an interview.** "Bayes' rule splits the conditional score into an unconditional score plus a classifier gradient; classifier guidance scales that gradient, which targets $p(x \mid c)p(c \mid x)^w$. Classifier-free guidance replaces the classifier by the implicit one, $p(x \mid c)/p(x)$, whose gradient is the difference of conditional and unconditional scores; substituting gives $\tilde\epsilon = (1+w)\epsilon_c - w\epsilon_u$, with both terms from one network trained with label dropout. The tilted target is sharper and more prototypical, so fidelity goes up and diversity down, but the sampler does not exactly sample it: the guided scores are not scores of a noised tilted density, and DDPM and DDIM give different results. At high scales the extrapolation oversaturates, which is what thresholding, CFG rescale, APG and guidance intervals address."
:::

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

::: qa
**Derive classifier guidance from Bayes' rule, as Dhariwal and Nichol do for the ancestral sampler.**

A conditional reverse step is proportional to the unconditional step times the classifier, $p_\theta(x_{t-1} \mid x_t)\,p_\phi(c \mid x_{t-1})$. The model step is $\mathcal{N}(\mu, \Sigma)$. When $\Sigma$ is small, expand $\log p_\phi(c \mid x_{t-1})$ to first order around $\mu$: $(x_{t-1} - \mu)^\top g_\phi$ plus a constant, with $g_\phi$ the classifier gradient at $\mu$. Adding this linear term to the Gaussian's quadratic and completing the square gives $\mathcal{N}(\mu + \Sigma g_\phi, \Sigma)$. ADM scales the gradient by $s$, which corresponds to the sharpened classifier $p_\phi^s$. For deterministic samplers the score form is used instead: $\tilde\epsilon = \hat\epsilon - \sigma_t\,s\,\nabla\log p_\phi(c \mid x_t)$, from $\nabla\log[p(x_t)p(c \mid x_t)^s] = -\hat\epsilon/\sigma_t + s\nabla\log p_\phi$.
:::

::: qa
**Does a CFG sampler actually draw from $p(x \mid c)^{1+w}/p(x)^w$?**

No. The guided score at each noise level is $(1+w)\nabla\log p_\sigma(x \mid c) - w\nabla\log p_\sigma(x)$, and noising does not commute with taking powers and ratios of densities, so this is not the score of any noised version of the tilted target. Different samplers then land on different distributions. In a solvable toy with conditional $\mathcal{N}(1, 1)$, unconditional $\mathcal{N}(0, 4)$ and $w = 3$, the tilted target is $\mathcal{N}(1.23, 0.55^2)$, the guided reverse SDE gives $\mathcal{N}(1.33, 0.44^2)$ and the guided ODE gives $\mathcal{N}(1.51, 0.13^2)$. Bradley and Nakkiran show in general that CFG with DDPM and with DDIM differ and that neither samples the powered distribution. The practical lesson is that the guidance scale's effect depends on the sampler and step count, so it has to be re-tuned when either changes.
:::

::: pitfall
Mixing up guidance conventions. In Ho & Salimans, $w = 0$ is unguided and $\tilde{\epsilon} = (1+w)\epsilon_c - w\epsilon_u$; in most libraries "guidance scale" $s = 1 + w$, so $s = 1$ is unguided and $s = 7.5$ corresponds to $w = 6.5$. State which one you mean.
:::

## Latent diffusion and a note on DiT

### Why diffuse in a latent space

Likelihood-style models spend most of their capacity on imperceptible high-frequency detail. Latent diffusion (Rombach et al.) splits the job: an autoencoder does perceptual compression and the diffusion model does semantic composition in a latent that is $f$ times smaller per side ($f$ is LDM's downsampling factor, unrelated to the SDE drift $f$). Write $z_0$ for the latent of an image and $z_t = \alpha_t z_0 + \sigma_t\epsilon$ for its noisy version; diffusion runs on $z$ exactly as it ran on $x$. The autoencoder is trained with a perceptual loss plus a patch-based adversarial loss and a light regularizer, either a KL term toward $\mathcal{N}(0, I)$ weighted by about $10^{-6}$ or a vector-quantization layer, so reconstructions stay sharp.

LDM compared $f \in \{1, 2, 4, 8, 16, 32\}$ ($f = 1$ is pixel space) on one A100 each, with the same number of training steps and parameters, and found $f$ between 4 and 16 a good trade-off: $f = 1, 2$ train slowly, because the diffusion model still has to do the perceptual compression itself, and large $f$ loses too much information, so fidelity stalls early. After 2M training steps on ImageNet, the FID gap between pixel-space diffusion and $f = 8$ was 38.

**Worked numbers.** A $256 \times 256 \times 3$ image has 196,608 values and 65,536 spatial positions. Stable Diffusion's $f = 8$ autoencoder maps it to a $32 \times 32 \times 4$ latent: 4,096 values (48 times fewer) and 1,024 positions (64 times fewer). Convolution and attention costs scale with the number of positions (attention quadratically), which is where the savings come from. The autoencoder caps achievable fidelity (small text, faces, fine texture).

### Latent scale is part of the noise schedule

The schedule's SNR assumes the signal has roughly unit variance. If the latent has variance $v$, the effective SNR at level $t$ is $v\,\bar\alpha_t/(1-\bar\alpha_t)$, a shift of $\log v$ in log-SNR. LDM therefore rescales KL-regularized latents by $1/\hat{\sigma}$, a single standard deviation estimated over all latent entries of the first training batch, so they have roughly unit variance; VQ latents already had variance close to one. LDM's appendix notes that without this rescaling the SNR is very high, and the model allocates much of the semantic detail early in the reverse process. The constant $0.18215$ is not in the LDM paper; it is the `scale_factor` in the Stable Diffusion v1 configs, playing the role of $1/\hat\sigma$ for that autoencoder. Without it the latent variance would be about $1/0.18215^2 = 30$, a log-SNR shift of $+3.4$, larger than the resolution shift between $64$ and $256$ pixels ($2.8$).

### Conditioning by cross-attention

For text and other modalities, a domain-specific encoder $\tau_\theta$ maps the condition $y$ to a sequence of tokens, and the U-Net's intermediate features attend to them: $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$ with $Q = W_Q\,\varphi_i(z_t)$ from the U-Net's intermediate features $\varphi_i(z_t)$ at layer $i$, and $K = W_K\,\tau_\theta(y)$, $V = W_V\,\tau_\theta(y)$ from the condition; $W_Q, W_K, W_V$ are learned projections and $d_k$ is the width of $Q$ and $K$. (In this formula $V$ is the attention "value", not a variance, and $\varphi_i$ is unrelated to the classifier parameters $\phi$.) Each spatial position picks the parts of the prompt it needs. See [GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html) for the autoencoder.

### DiT

**Architecture.** Diffusion Transformers (Peebles and Xie) patchify the latent into tokens and replace the U-Net with a transformer. With patch size $p$, an $I \times I \times C$ latent becomes $(I/p)^2$ tokens; for the $32 \times 32 \times 4$ latent of a $256^2$ image and $p = 2$ that is 256 tokens. Halving $p$ quadruples the tokens and at least quadruples the Gflops, while barely changing the parameter count.

**Conditioning on timestep and class.** DiT compared four ways: in-context tokens, cross-attention, adaptive layer norm (adaLN) and adaLN-Zero. In adaLN, an MLP on the sum of the timestep and class embeddings regresses a per-block scale $\gamma$ and shift $\beta$ for each layer norm (DiT's names; unrelated to the schedule's $\beta_t$ and to Min-SNR's $\gamma$). adaLN-Zero also regresses a per-dimension gate $\alpha$ (unrelated to the signal scale $\alpha_t$) applied before each residual connection and initializes the MLP so that every $\alpha$ is zero, which makes every block start as the identity function, the transformer analogue of zero-initializing the last layer of each residual block. adaLN-Zero gave the lowest FID of the four, with nearly half the FID of in-context conditioning at 400K iterations, and outperformed vanilla adaLN by a wide margin at the same cost; the two adaLN variants were also the cheapest (cross-attention added about 15% Gflops).

**Scaling.** FID tracked model Gflops closely: deeper or wider models, or smaller patches, all lowered FID, and at a fixed parameter count, more tokens helped. DiT-XL/2 reached FID 2.27 on ImageNet $256 \times 256$ with classifier-free guidance at scale 1.5 (library convention) and 250 DDPM sampling steps, and 3.04 at $512 \times 512$. It kept ADM's diffusion settings: the linear schedule from $10^{-4}$ to $0.02$, learned variances, and Stable Diffusion's $f = 8$ autoencoder. Architecture depth (MMDiT, text conditioning) is on [text-to-image and multimodal](text-to-image-multimodal.html).

::: key
**How to say it in an interview.** "Latent diffusion trains an autoencoder for perceptual compression and runs diffusion in its latent, typically 8 times smaller per side, so a $256^2$ image becomes a $32 \times 32 \times 4$ latent with 64 times fewer positions. The latent must be rescaled to unit variance, or the schedule's SNR is wrong. Text enters through cross-attention. DiT replaces the U-Net with a transformer on latent patches, conditions on timestep and class through adaLN-Zero, which starts every block as the identity, and its FID improves predictably with Gflops."
:::

::: {.qa asked="Wells Fargo applied AI research intern, PracHub 2026"}
**Walk through the architecture of a diffusion model: how data are corrupted, what the network predicts, how it is built and conditioned, and how samples are generated.**

An image is encoded to a latent by a pretrained autoencoder, then corrupted as $z_t = \alpha_t z_0 + \sigma_t\epsilon$ at a random noise level. A U-Net or DiT takes $z_t$, an embedding of the noise level and the condition, and predicts $\epsilon$, $v$ or $z_0$; text enters through cross-attention or joint attention, timestep and class through adaptive normalization. Training is a weighted MSE with the condition dropped about 10% of the time. To sample, start from Gaussian noise, run 20 to 50 steps of a DDIM, DPM-Solver or Heun-type solver with classifier-free guidance, then decode the latent. I would also state the schedule (e.g. zero terminal SNR, resolution shift) and how much each choice costs at inference.
:::

::: qa
**Why train diffusion in a latent space, and what can go wrong?**

Pixel-space diffusion spends compute and capacity on high-frequency detail that a good autoencoder reproduces cheaply, so a latent 8 times smaller per side cuts cost roughly by the reduction in positions while keeping perceptual quality. The risks: reconstruction error is a hard ceiling on detail; too much compression loses information; the latent's scale changes the effective SNR, so latents must be normalized or the schedule retuned; and decoder artifacts can look like generator failures. Pixel-space models avoid the autoencoder at higher compute cost.
:::

::: qa
**Why does DiT use adaLN-Zero, and what does the "Zero" do?**

The timestep and class must modulate every block. adaLN does this cheaply: an MLP on the sum of the two embeddings outputs a scale and shift for each layer norm, the same for all tokens, which costs almost no compute compared with extra tokens or cross-attention. adaLN-Zero also outputs a per-channel gate on each residual branch and initializes the MLP so all gates are zero, so at initialization every block is the identity and the network starts as a well-conditioned shallow map, as with zero-initialized residual branches in ResNets and diffusion U-Nets. In DiT's comparison it gave the lowest FID of the four conditioning designs, beat plain adaLN at identical cost, and added almost no Gflops.
:::

## Rapid-fire review

::: {.qa asked="TikTok AI research scientist, Interview Query"}
**Name two activation functions used in diffusion networks and how they differ.**

DDPM and ADM U-Nets use SiLU (Swish), $x \cdot \mathrm{sigmoid}(x) = x/(1 + e^{-x})$; DiT uses tanh-approximated GELU, $x\,\Phi(x)$ with $\Phi$ the standard normal CDF, in its MLPs and SiLU before the adaLN projection. Both are smooth, non-monotone near zero (each dips slightly below zero for negative inputs), and close in shape; GELU weights $x$ by the Gaussian CDF, SiLU by the logistic sigmoid.
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

It sets how much capacity goes to the unconditional model. Ho & Salimans found 0.1 and 0.2 about equally good and 0.5 consistently worse across the whole quality-diversity frontier.
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

::: qa
**Why does a well-trained diffusion model's training loss stay well above zero?**

The target noise is not determined by $x_t$: many pairs $(x_0, \epsilon)$ give the same noisy input. By the conditional-mean argument, the best possible $\epsilon$-loss at a level is the average posterior variance $\mathbb{E}\,\mathrm{Var}(\epsilon \mid x_t)$, which is large at intermediate SNR. In the 1-D Gaussian example with $\mathrm{SNR} = 0.5625$ it is $0.36$ per dimension. So the loss value says little about sample quality; compare models by FID or by loss at matched noise levels.
:::

::: qa
**What does Stable Diffusion's latent scale factor 0.18215 do?**

It multiplies the VAE latent so that its overall standard deviation is about one, which is what the noise schedule's SNR assumes. LDM's recipe estimates $1/\hat\sigma$ from the first training batch; 0.18215 is that constant in the Stable Diffusion v1 config. Leaving it out would make the latent variance about 30, a log-SNR shift of about $+3.4$, so every timestep would be far less noisy than intended and training and sampling would be mismatched.
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
