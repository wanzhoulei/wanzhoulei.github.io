---
title: "EDM: The Design Space of Diffusion Models"
subtitle: "A guided reading of Karras et al. (2022): the common framework, the Heun sampler and step grid, stochastic churn, preconditioning and training, with every step derived, plus the EDM2 follow-up."
order: 1.5
updated: 2026-11-08
---

"Elucidating the Design Space of Diffusion-Based Generative Models" by Karras, Aittala, Aila and Laine (NeurIPS 2022), usually called EDM, takes diffusion models apart and tunes each part on its own. The parts are: how the noise level grows with time and whether the signal is rescaled, which ODE or SDE solver generates samples, where the solver places its steps, how the network's inputs and outputs are scaled, which noise levels the network sees during training, and how the loss is weighted across noise levels. Earlier papers derived all of these from one theoretical starting point, so each model looked like a package in which nothing could be swapped. EDM shows the parts are independent, writes three earlier model families in one table, and improves each part with a short argument and an ablation. The result was a record FID of 1.79 on class-conditional CIFAR-10 at 35 network evaluations, and several of its choices (the Heun sampler, the $\rho = 7$ step grid, the $c_{\text{skip}}$ preconditioning, the log-normal training noise) became common defaults. Interviewers like the paper because it tests whether you can move between DDPM, score-SDE and flow notation, derive a sampler or a loss weight from first principles, and explain why a design choice works rather than reciting it.

**Roadmap.** The page follows the paper's order. A notation table first maps EDM's symbols to DDPM's. Section 1 states the paper's main claim. Section 2 builds the common framework: the probability-flow ODE written with a noise schedule $\sigma(t)$ and a scale schedule $s(t)$, the identity that turns a denoiser into a score, and the paper's Table 1, which writes VP, VE, iDDPM and EDM in one language. Section 3 covers deterministic sampling: Heun's method, the $\rho$ step grid, the choice $\sigma(t) = t$, and the paper's numbers on three pretrained models. Section 4 covers stochastic sampling: the SDE written as the ODE plus Langevin dynamics, Algorithm 2 and its four knobs, and when noise injection helps. Section 5 derives the preconditioning and the loss weight, explains the log-normal training distribution and the augmentation, and walks through the A-to-F training ablation. Section 6 collects the final results. Section 7 connects EDM to $\epsilon$-prediction models, rectified flow and SD3. Section 8 covers EDM2, the follow-up on training dynamics. Most sections end with a "What this says" paragraph, a spoken-answer box, and interview questions. DDPM, DDIM and the SDE view are covered on the [diffusion models](diffusion-models.html) page; this page is self-contained for EDM.

## Notation and the map to DDPM

This page uses the paper's own symbols. The [diffusion models](diffusion-models.html) page writes the same noisy image as $x = x_0 + \sigma\epsilon$; here it is $x = y + n$, with $y = x_0$ the clean image and $n = \sigma\epsilon$ the noise.

| EDM symbol | Meaning in words | DDPM / VP equivalent |
|---|---|---|
| $y$ | a clean training image; each pixel has standard deviation $\sigma_{\text{data}}$ over the dataset | $x_0$ |
| $n$ | Gaussian noise, $n \sim \mathcal{N}(0, \sigma^2 I)$ | $\sigma\epsilon$, with $\epsilon \sim \mathcal{N}(0, I)$ |
| $\sigma$ | noise standard deviation of the unscaled noisy image $y + n$ | $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ |
| $s(t)$ | scale applied to the whole noisy image at time $t$ | $\sqrt{\bar\alpha_t} = 1/\sqrt{1+\sigma^2}$ |
| $x = s(t)\,(y + n)$ | the variable the ODE moves | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ |
| $s(t)\,\sigma(t)$ | total noise standard deviation in $x$ | $\sqrt{1-\bar\alpha_t}$ |
| $1/\sigma^2$ | signal-to-noise ratio (per unit signal variance) | $\mathrm{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$ |
| $-2\ln\sigma$ | log-SNR | $\lambda_t$ on the diffusion page |
| $p(x;\sigma)$ | data density blurred by $\mathcal{N}(0, \sigma^2 I)$ | density of $x_t/\sqrt{\bar\alpha_t}$ |
| $D(x;\sigma)$ | ideal denoiser, the mean of $y$ given $x$ | $\hat{x}_0 = \mathbb{E}[x_0 \mid x_t]$ |
| $D_\theta$, $F_\theta$ | trained denoiser; raw network inside it | $\epsilon_\theta$ is $F_\theta$ for VP models |
| $t$ | ODE time; with EDM's choice $\sigma(t) = t$ it equals the noise level | DDPM's integer step $t$ (a different object) |
| $\lambda(\sigma)$ | loss weight at noise level $\sigma$ | not the log-SNR |
| $p_{\text{train}}(\sigma)$ | distribution of noise levels used in training | uniform over timesteps in DDPM |
| $x_i$, $i = 0, \dots, N$ | sampler iterate; $x_0$ is pure noise, $x_N$ the output | reversed: DDPM's $x_0$ is the data |

The VP row follows from one division. A DDPM sample is $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$. Divide by $\sqrt{\bar\alpha_t}$:

$$
\frac{x_t}{\sqrt{\bar\alpha_t}} = x_0 + \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\;\epsilon ,
$$

which is $y + n$ with $\sigma = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$. To invert,

$$
\begin{aligned}
\sigma^2 &= \frac{1-\bar\alpha_t}{\bar\alpha_t} && \text{(square the definition)}\\
\sigma^2\bar\alpha_t &= 1 - \bar\alpha_t && \text{(multiply by } \bar\alpha_t)\\
\bar\alpha_t\,(1+\sigma^2) &= 1 && \text{(collect } \bar\alpha_t)\\
\bar\alpha_t &= \frac{1}{1+\sigma^2} && \text{(divide).}
\end{aligned}
$$

So a DDPM timestep and an EDM noise level carry the same information, and the DDPM image is the EDM image times $s = \sqrt{\bar\alpha_t} = 1/\sqrt{1+\sigma^2}$.

::: pitfall
Two symbol clashes cause most confusion. EDM's $x_0$ is the starting noise image, not the data; the image comes out as $x_N$. And EDM's $\lambda(\sigma)$ is a loss weight, while many diffusion papers use $\lambda$ for log-SNR.
:::

## 1. The main move: a design space with independent parts

### The problem the paper points at

Before EDM, each family of diffusion models came with its own derivation. DDPM started from a discrete Markov chain, NCSN from annealed Langevin dynamics, the score-SDE paper from a stochastic differential equation. Each derivation produced a noise schedule, a network parameterization, a loss and a sampler together. Because they came out of one derivation, they looked as if they depended on each other, and it was hard to tell which ones mattered.

EDM's view is practical. Look at the objects that actually appear in code: the noise levels the sampler visits, the solver, the scaling around the network, the noise levels drawn during training, the loss weight. Write every earlier model in terms of those objects. Then ask, one object at a time, whether a different choice is better.

### The components

The paper separates these choices, each handled in a later section of this page:

- **Noise schedule $\sigma(t)$ and scale schedule $s(t)$**: which noise level the ODE is at, at time $t$, and whether the image is shrunk as noise grows (Section 3.5).
- **Solver**: Euler, Heun, or a higher-order method (Section 3.2).
- **Time discretization $\{t_i\}$**: where the $N$ solver steps go (Section 3.4).
- **Stochasticity**: how much noise the sampler adds and removes on the way (Section 4).
- **Preconditioning**: the scalings $c_{\text{in}}, c_{\text{out}}, c_{\text{skip}}, c_{\text{noise}}$ that turn a raw network $F_\theta$ into a denoiser $D_\theta$ (Sections 5.2 to 5.5).
- **Training noise distribution $p_{\text{train}}(\sigma)$ and loss weight $\lambda(\sigma)$** (Sections 5.4 and 5.7).
- **Regularization and architecture details**: augmentation, dropout, channel layout (Sections 5.8 and 5.9).

The paper's claim is that, within reason, any combination of these choices gives a working model, because changing one does not break the property that the sampler converges to the data distribution. Some combinations work better than others, and the rest of the paper finds good ones.

### The hypothesis tested first

The paper tests separability directly in Section 3: if sampling is independent of training, then a better sampler should improve models that were trained by other people, with other conventions, without retraining. It does, by large factors (Section 3.6). Training improvements are then tested with the improved sampler fixed (Section 5).

**What this says.** EDM is not a new architecture or a new generative principle. It is a common coordinate system for diffusion models plus a set of tuned choices in that system. Most of its value for an interview is the coordinate system: once you can place a model in Table 1, you can reason about any of its parts in isolation.

::: key
*How to say it in an interview:* "EDM's point is that a diffusion model is a set of separable choices: noise schedule and scaling, solver, step placement, network preconditioning, training noise distribution and loss weight. They rewrite VP, VE and DDPM-style models in one table, show the sampler is independent of training by speeding up other people's pretrained models up to 300 times in step count, and then fix each training component with a first-principles argument. The defaults that came out of it, Heun with a $\rho = 7$ grid, $c_{\text{skip}}$ preconditioning and log-normal noise levels, are now widely used."
:::

## 2. The common framework

### 2.1 Noise levels instead of time

Start from the data distribution $p_{\text{data}}(y)$, with per-pixel standard deviation $\sigma_{\text{data}}$. Adding independent Gaussian noise of standard deviation $\sigma$ to every sample gives the blurred, or "mollified", distribution

$$
p(x;\sigma) = \int p_{\text{data}}(y)\, \mathcal{N}(x;\, y,\, \sigma^2 I)\, dy ,
$$

the convolution of the data density with a Gaussian. When $\sigma_{\max} \gg \sigma_{\text{data}}$, $p(x;\sigma_{\max})$ is practically $\mathcal{N}(0, \sigma_{\max}^2 I)$, because the data's spread is negligible next to the noise.

Generation then has a simple description. Draw $x_0 \sim \mathcal{N}(0, \sigma_{\max}^2 I)$. Pick decreasing noise levels $\sigma_0 = \sigma_{\max} > \sigma_1 > \dots > \sigma_N = 0$. Transform the image step by step so that at every step $x_i \sim p(x_i; \sigma_i)$. The last iterate then has the data distribution.

Note what is missing: there is no forward Markov chain and no time variable yet. The family $p(x;\sigma)$ is what matters; time will be a way to parameterize the path through it.

### 2.2 The probability-flow ODE when the image is not rescaled

We want an ODE that moves a sample along the family: if $x$ at time $t_a$ is distributed as $p(x;\sigma(t_a))$, then after integrating to $t_b$ it is distributed as $p(x;\sigma(t_b))$. Here $\sigma(t)$ is any increasing function, the noise schedule. The paper gets the ODE from the score-SDE paper's formula (Section 2.4 below re-derives that route). A shorter route goes through the heat equation, and the paper uses the same equation in its SDE appendix.

**Step 1: the Gaussian kernel solves a heat equation.** Let $G_\sigma(x) = (2\pi\sigma^2)^{-d/2}\exp(-\lVert x\rVert^2/2\sigma^2)$ be the density of $\mathcal{N}(0, \sigma^2 I)$ in $d$ dimensions.

$$
\begin{aligned}
\ln G_\sigma &= -\tfrac{d}{2}\ln(2\pi\sigma^2) - \frac{\lVert x\rVert^2}{2\sigma^2} && \text{(take the log)}\\
\frac{\partial G_\sigma}{\partial\sigma} &= G_\sigma\Big(-\frac{d}{\sigma} + \frac{\lVert x\rVert^2}{\sigma^3}\Big) && \text{(differentiate the log in } \sigma)\\
\nabla G_\sigma &= -\frac{x}{\sigma^2}\,G_\sigma && \text{(gradient of the exponent)}\\
\Delta G_\sigma = \nabla\cdot\nabla G_\sigma &= -\frac{d}{\sigma^2}G_\sigma + \frac{\lVert x\rVert^2}{\sigma^4}G_\sigma && (\nabla\cdot x = d \text{, plus product rule)}\\
\frac{\partial G_\sigma}{\partial\sigma} &= \sigma\,\Delta G_\sigma && \text{(compare the two brackets).}
\end{aligned}
$$

**Step 2: the blurred data density solves the same equation.** With $\sigma = \sigma(t)$, the chain rule gives $\partial_t G_{\sigma(t)} = \dot\sigma\,\partial_\sigma G = \dot\sigma\sigma\,\Delta G$, where the dot is a time derivative. The density $p(x;\sigma(t)) = \int p_{\text{data}}(y)\,G_{\sigma(t)}(x - y)\,dy$ is a linear combination of shifted kernels, so the time derivative and the Laplacian pass through the integral:

$$
\frac{\partial p}{\partial t} = \dot\sigma(t)\,\sigma(t)\,\Delta p .
$$

This is heat diffusion with a time-varying rate $\dot\sigma\sigma$. (The paper notes that $\sigma(t) \propto \sqrt{t}$ makes the rate constant, which is why it is the "mathematically natural" choice, and then argues against it in Section 3.)

**Step 3: rewrite diffusion as transport.**

$$
\begin{aligned}
\Delta p &= \nabla\cdot(\nabla p) && \text{(definition of the Laplacian)}\\
\nabla p &= p\,\nabla\log p && \text{(since } \nabla\log p = \nabla p / p)\\
\frac{\partial p}{\partial t} &= \nabla\cdot\big(\dot\sigma\sigma\, p\,\nabla\log p\big) && \text{(substitute into Step 2)}\\
&= -\nabla\cdot\big(p\; v\big), \qquad v(x,t) = -\dot\sigma(t)\,\sigma(t)\,\nabla_x\log p\big(x;\sigma(t)\big) && \text{(pull out a minus sign).}
\end{aligned}
$$

The last line is a continuity equation: it is exactly the equation satisfied by the density of points that each move with velocity $v(x,t)$. So the ODE

$$
dx = -\dot\sigma(t)\,\sigma(t)\,\nabla_x \log p\big(x;\sigma(t)\big)\,dt
$$

carries samples of $p(x;\sigma(t_a))$ to samples of $p(x;\sigma(t_b))$, forward or backward in time. This is the paper's Eq. 1.

**Reading it.** $\nabla_x\log p(x;\sigma)$, the score, points toward higher density, that is, toward the data. Moving forward in time ($dt > 0$) the minus sign pushes the sample away from the data, at a speed set by how fast the noise level changes. Moving backward pushes it toward the data. The only randomness in generation is the starting noise.

### 2.3 Adding a scale schedule $s(t)$

VP models (and DDPM) also shrink the image as noise grows, so that its total variance stays bounded. EDM handles this with a second function, $s(t)$, and defines the ODE variable as a scaled copy, $x = s(t)\,\hat{x}$, where $\hat{x}$ is the unscaled variable that follows the ODE of Section 2.2. Then $x$ has the distribution of $s(t)\,(y + n)$ with $n \sim \mathcal{N}(0, \sigma(t)^2 I)$, so its density is $s(t)^{-d}\,p\big(x/s(t);\sigma(t)\big)$.

Derive the ODE for $x$. Two notations for the score appear below, and the difference matters. $(\nabla\log p)(u;\sigma)$ is the gradient of $\log p$ evaluated at the point $u$. $\nabla_x\big[\log p(x/s;\sigma)\big]$ is the gradient with respect to $x$ of the composite function $x \mapsto \log p(x/s;\sigma)$. The chain rule says the second is the first, at $u = x/s$, times $1/s$. The paper's Eq. 4 is written with the second.

$$
\begin{aligned}
\frac{dx}{dt} &= \dot s\,\hat{x} + s\,\frac{d\hat{x}}{dt} && \text{(product rule on } x = s\hat{x})\\
&= \frac{\dot s}{s}\,x + s\,\frac{d\hat{x}}{dt} && (\hat{x} = x/s)\\
&= \frac{\dot s}{s}\,x - s\,\dot\sigma\sigma\,\big(\nabla\log p\big)(\hat{x};\sigma) && \text{(ODE of Section 2.2 for } \hat{x})\\
\nabla_x\Big[\log p\big(x/s;\sigma\big)\Big] &= \frac{1}{s}\,\big(\nabla\log p\big)(x/s;\sigma) && \text{(chain rule: inner map } x \mapsto x/s)\\
\frac{dx}{dt} &= \frac{\dot s(t)}{s(t)}\,x - s(t)^2\,\dot\sigma(t)\,\sigma(t)\,\nabla_x\log p\Big(\frac{x}{s(t)};\sigma(t)\Big) && \text{(replace } (\nabla\log p)(\hat x) \text{ by } s\,\nabla_x[\cdot]).
\end{aligned}
$$

This is the paper's Eq. 4, the general ODE. The first term is pure rescaling. The second is the same transport as before, evaluated on the unscaled image $x/s$ and scaled back. The paper notes that the score is always evaluated on $x/s$, so that $p(x;\sigma)$ never depends on $s$. Changing $\sigma(t)$ only reparameterizes time; changing $s(t)$ only reparameterizes space. Every probability-flow ODE in the table below is the same canonical ODE seen through these two reparameterizations.

### 2.4 Cross-check with the score-SDE formula

Song et al. write a forward SDE $dx = f(t)\,x\,dt + g(t)\,d\omega_t$ with drift coefficient $f$, diffusion coefficient $g$ and Wiener process $\omega_t$, and the probability-flow ODE $dx = [f(t)\,x - \tfrac12 g(t)^2\nabla_x\log p_t(x)]\,dt$. EDM's appendix converts $f, g$ into $s, \sigma$. Work with one coordinate and start from a clean value $y$.

$$
\begin{aligned}
\frac{d}{dt}\mathbb{E}[x] &= f\,\mathbb{E}[x] && \text{(take expectations; } d\omega \text{ has mean 0)}\\
d(x^2) &= 2x\,dx + g^2\,dt && \text{(Itô's rule)}\\
\frac{d}{dt}\mathbb{E}[x^2] &= 2f\,\mathbb{E}[x^2] + g^2 && \text{(substitute } dx\text{, take expectations)}\\
\frac{d}{dt}\mathrm{Var}[x] &= 2f\,\mathrm{Var}[x] + g^2 && \text{(subtract } \tfrac{d}{dt}\mathbb{E}[x]^2 = 2f\,\mathbb{E}[x]^2).
\end{aligned}
$$

Now require the mean to be $s(t)\,y$ and the variance to be $s(t)^2\sigma(t)^2$:

$$
\begin{aligned}
\dot s\,y &= f\,s\,y \;\Rightarrow\; f = \dot s/s && \text{(mean equation)}\\
2s\dot s\sigma^2 + 2s^2\sigma\dot\sigma &= 2f\,s^2\sigma^2 + g^2 && \text{(variance equation, product rule on the left)}\\
2s^2\sigma\dot\sigma &= g^2 \;\Rightarrow\; g = s\sqrt{2\dot\sigma\sigma} && (2s\dot s\sigma^2 = 2f s^2\sigma^2 \text{ cancels}).
\end{aligned}
$$

The marginal is $p_t(x) = s^{-d}\,p(x/s;\sigma)$; the constant $s^{-d}$ has zero gradient, so $\nabla_x\log p_t(x) = \nabla_x\log p(x/s;\sigma)$. Substitute $f$ and $g$ into Song's ODE: $f x = (\dot s/s)\,x$ and $\tfrac12 g^2 = s^2\dot\sigma\sigma$, which reproduces Eq. 4 exactly. The paper's point is that $f$ and $g$ only describe the marginals indirectly, while $s$ and $\sigma$ name them directly.

### 2.5 The score is a denoiser in disguise

The ODE needs the score of $p(x;\sigma)$, which is intractable. EDM uses the classical link to denoising (Tweedie's formula). Let $D(x;\sigma)$ be the function that minimizes the expected squared denoising error at noise level $\sigma$:

$$
D = \arg\min_D \; \mathbb{E}_{y \sim p_{\text{data}}}\,\mathbb{E}_{n \sim \mathcal{N}(0,\sigma^2 I)}\,\lVert D(y + n;\sigma) - y\rVert_2^2 .
$$

**The minimizer is the posterior mean.** Group the expectation by the noisy input $x = y + n$: the loss is $\mathbb{E}_x\,\mathbb{E}[\lVert D(x) - y\rVert^2 \mid x]$. For each $x$ separately the inner term is a convex quadratic in the vector $D(x)$; its gradient is $2(D(x) - \mathbb{E}[y \mid x])$, which vanishes at $D(x;\sigma) = \mathbb{E}[y \mid x]$.

**The score equals the posterior mean minus $x$, over $\sigma^2$.**

$$
\begin{aligned}
p(x;\sigma) &= \int p_{\text{data}}(y)\,\mathcal{N}(x; y, \sigma^2 I)\,dy && \text{(definition)}\\
\nabla_x\,\mathcal{N}(x; y, \sigma^2 I) &= \mathcal{N}(x; y, \sigma^2 I)\,\frac{y - x}{\sigma^2} && \text{(gradient of a Gaussian, as in 2.2)}\\
\nabla_x\,p(x;\sigma) &= \int p_{\text{data}}(y)\,\mathcal{N}(x; y, \sigma^2 I)\,\frac{y - x}{\sigma^2}\,dy && \text{(differentiate under the integral)}\\
\nabla_x\log p(x;\sigma) &= \int \underbrace{\frac{p_{\text{data}}(y)\,\mathcal{N}(x; y, \sigma^2 I)}{p(x;\sigma)}}_{p(y \mid x) \text{ by Bayes}}\,\frac{y - x}{\sigma^2}\,dy && \text{(divide by } p)\\
&= \frac{\mathbb{E}[y \mid x] - x}{\sigma^2} = \frac{D(x;\sigma) - x}{\sigma^2} && \text{(the posterior integrates to 1).}
\end{aligned}
$$

This is the paper's Eq. 3. In words: the score points from the noisy image toward its best clean estimate, and $D - x$ is minus the expected noise, so the score "isolates the noise component" of $x$. The practical consequence is the paper's key observation: train a network $D_\theta$ with the plain $L_2$ denoising loss, and you have a score model.

For a finite training set $\{y_1, \dots, y_Y\}$, the same derivation gives a closed form: $D(x;\sigma) = \sum_i \mathcal{N}(x; y_i, \sigma^2 I)\,y_i \,/\, \sum_i \mathcal{N}(x; y_i, \sigma^2 I)$, a softmax-weighted average of training images. The paper's Figure 1 shows this ideal denoiser on CIFAR-10: at small $\sigma$ it returns a nearby training image, and as $\sigma$ grows its output fades toward the dataset mean. (A model that matched this formula exactly would only reproduce training images; generalization comes from the network not fitting it exactly.)

### 2.6 The ODE in terms of the denoiser

Substitute the denoiser into Eq. 4. First rewrite the score term:

$$
\begin{aligned}
\nabla_x\log p\big(x/s;\sigma\big) &= \frac{1}{s}\,\big(\nabla\log p\big)(x/s;\sigma) && \text{(chain rule, as in 2.3)}\\
&= \frac{1}{s\,\sigma^2}\Big(D\big(x/s;\sigma\big) - \frac{x}{s}\Big) && \text{(Section 2.5 at the point } x/s).
\end{aligned}
$$

Then

$$
\begin{aligned}
\frac{dx}{dt} &= \frac{\dot s}{s}\,x - s^2\dot\sigma\sigma\cdot\frac{1}{s\sigma^2}\Big(D(x/s;\sigma) - \frac{x}{s}\Big) && \text{(substitute)}\\
&= \frac{\dot s}{s}\,x - \frac{s\,\dot\sigma}{\sigma}\,D(x/s;\sigma) + \frac{\dot\sigma}{\sigma}\,x && \text{(expand; } s\cdot x/s = x)\\
&= \Big(\frac{\dot\sigma(t)}{\sigma(t)} + \frac{\dot s(t)}{s(t)}\Big)\,x - \frac{\dot\sigma(t)\,s(t)}{\sigma(t)}\,D_\theta\Big(\frac{x}{s(t)};\sigma(t)\Big) && \text{(collect terms in } x;\ D \to D_\theta).
\end{aligned}
$$

This is the derivative that Algorithm 1 evaluates for any $\sigma(t)$ and $s(t)$. EDM's own choice is $\sigma(t) = t$ and $s(t) = 1$, so $\dot\sigma = 1$ and $\dot s = 0$, and it collapses to

$$
\frac{dx}{dt} = \frac{x - D_\theta(x;t)}{t}, \qquad \text{with } t \text{ and } \sigma \text{ now interchangeable}.
$$

One Euler step from $(x, t)$ all the way to $t = 0$ gives $x + (0 - t)\,\frac{x - D_\theta}{t} = D_\theta(x;t)$: the tangent of every trajectory points straight at the current denoised estimate. Section 3.5 builds on this.

### 2.7 Table 1: four model families in one language

The paper's Table 1 writes the deterministic versions of three earlier methods and EDM in the coordinates above. The denoiser is always $D_\theta(x;\sigma) = c_{\text{skip}}(\sigma)\,x + c_{\text{out}}(\sigma)\,F_\theta\big(c_{\text{in}}(\sigma)\,x;\ c_{\text{noise}}(\sigma)\big)$, with $F_\theta$ the raw network. $N$ is the number of solver steps, $t_N = 0$, and $M$ and $u_j$ are the original number of training timesteps and their noise levels when a model was trained on a discrete set.

| | VP (Song et al.) | VE (Song et al.) | iDDPM + DDIM | EDM |
|---|---|---|---|---|
| ODE solver | Euler | Euler | Euler | Heun (2nd order) |
| Time steps $t_{i<N}$ | $1 + \frac{i}{N-1}(\epsilon_s - 1)$ | $\sigma_{\max}^2\,(\sigma_{\min}^2/\sigma_{\max}^2)^{\frac{i}{N-1}}$ | $u_{\lfloor j_0 + \frac{M-1-j_0}{N-1}\,i + \frac12\rfloor}$ | $\big(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho})\big)^\rho$ |
| Schedule $\sigma(t)$ | $\sqrt{e^{\frac12\beta_d t^2 + \beta_{\min} t} - 1}$ | $\sqrt{t}$ | $t$ | $t$ |
| Scaling $s(t)$ | $1/\sqrt{e^{\frac12\beta_d t^2 + \beta_{\min} t}}$ | $1$ | $1$ | $1$ |
| Network $F_\theta$ | DDPM++ | NCSN++ | DDPM (ADM) | any |
| $c_{\text{skip}}(\sigma)$ | $1$ | $1$ | $1$ | $\sigma_{\text{data}}^2/(\sigma^2 + \sigma_{\text{data}}^2)$ |
| $c_{\text{out}}(\sigma)$ | $-\sigma$ | $\sigma$ | $-\sigma$ | $\sigma\,\sigma_{\text{data}}/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$ |
| $c_{\text{in}}(\sigma)$ | $1/\sqrt{\sigma^2+1}$ | $1$ | $1/\sqrt{\sigma^2+1}$ | $1/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$ |
| $c_{\text{noise}}(\sigma)$ | $(M-1)\,\sigma^{-1}(\sigma)$ | $\ln(\frac12\sigma)$ | $M - 1 - \arg\min_j \lvert u_j - \sigma\rvert$ | $\frac14\ln\sigma$ |
| $p_{\text{train}}(\sigma)$ | $\sigma^{-1}(\sigma) \sim \mathcal{U}(\epsilon_t, 1)$ | $\ln\sigma \sim \mathcal{U}(\ln\sigma_{\min}, \ln\sigma_{\max})$ | $\sigma = u_j,\ j \sim \mathcal{U}\{0, \dots, M-1\}$ | $\ln\sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$ |
| $\lambda(\sigma)$ | $1/\sigma^2$ | $1/\sigma^2$ | $1/\sigma^2$ (iDDPM adds $L_{\text{vlb}}$) | $(\sigma^2 + \sigma_{\text{data}}^2)/(\sigma\,\sigma_{\text{data}})^2$ |
| Parameters | $\beta_d = 19.9$, $\beta_{\min} = 0.1$, $\epsilon_s = 10^{-3}$, $\epsilon_t = 10^{-5}$, $M = 1000$ | $\sigma_{\min} = 0.02$, $\sigma_{\max} = 100$ | $\bar\alpha_j = \sin^2\big(\frac{\pi}{2}\frac{j}{M(C_2+1)}\big)$, $C_1 = 0.001$, $C_2 = 0.008$, $M = 1000$, $j_0 = 8$ | $\sigma_{\min} = 0.002$, $\sigma_{\max} = 80$, $\sigma_{\text{data}} = 0.5$, $\rho = 7$, $P_{\text{mean}} = -1.2$, $P_{\text{std}} = 1.2$ |

For iDDPM the levels $u_j$ come from the recursion $u_M = 0$, $u_{j-1} = \sqrt{(u_j^2 + 1)/\max(\bar\alpha_{j-1}/\bar\alpha_j,\, C_1) - 1}$, which encodes the cosine schedule with its $\beta \le 0.999$ clamp. All images are taken to lie in $[-1, 1]$; the paper converts every model to this range. The following subsections read the table row by row.

#### Solver and time steps

All three earlier deterministic samplers are Euler methods; EDM uses Heun. The time grids differ. VP takes $N$ uniform times in $t$ between $1$ and $\epsilon_s$. VE takes times whose noise levels $\sigma = \sqrt{t}$ are spaced geometrically between $\sigma_{\max}$ and $\sigma_{\min}$. iDDPM can only be evaluated at its 1000 trained noise levels, so it picks every so-many of them; the offset $j_0 = 8$ skips the first eight levels, whose $\sigma$ (up to about 20,000) is far above EDM's preferred $\sigma_{\max} = 80$. The paper found $j_0 = 8$ gave better FID than iDDPM's $j_0 = 0$. EDM uses the $\rho$ grid of Section 3.4.

#### Where VP's $\sigma(t)$ and $s(t)$ come from

The VP SDE has $f(t) = -\tfrac12\beta(t)$ and $g(t) = \sqrt{\beta(t)}$ with $\beta(t) = \beta_{\min} + t\,(\beta_{\max} - \beta_{\min})$ for $t \in [0, 1]$. Let $\alpha(t) = \int_0^t\beta = \tfrac12\beta_d t^2 + \beta_{\min} t$, with $\beta_d = \beta_{\max} - \beta_{\min}$ (this $\alpha$ is the paper's notation for the integral, not DDPM's $\alpha_t$). Using Section 2.4 in reverse:

$$
\begin{aligned}
s(t) &= \exp\Big(\int_0^t f\Big) = \exp\big(-\tfrac12\alpha(t)\big) = 1\big/\sqrt{e^{\alpha(t)}} && \text{(from } f = \dot s/s)\\
\sigma(t)^2 &= \int_0^t \frac{g(\xi)^2}{s(\xi)^2}\,d\xi = \int_0^t \beta(\xi)\,e^{\alpha(\xi)}\,d\xi && \text{(from } g = s\sqrt{2\dot\sigma\sigma}\text{, integrated)}\\
&= \int_0^t \dot\alpha(\xi)\,e^{\alpha(\xi)}\,d\xi = e^{\alpha(t)} - 1 && (\dot\alpha = \beta,\ \alpha(0) = 0)\\
s(t) &= 1/\sqrt{\sigma(t)^2 + 1} && \text{(combine the two lines).}
\end{aligned}
$$

The last line is the notation table's $\sqrt{\bar\alpha_t} = 1/\sqrt{1+\sigma^2}$ again. **This is how DDPM becomes a special case.** The VP SDE is the continuous limit of DDPM: Song et al. relate the discrete $\beta_i$ to $\beta(t)$ by $\beta_i \approx \beta(i/T)/T$, and $\beta_{\min} = 0.1$, $\beta_{\max} = 20$ with $T = 1000$ give $0.1/1000 = 10^{-4}$ and $20/1000 = 0.02$, DDPM's linear schedule endpoints. In the table the remaining DDPM choices are $\epsilon$-prediction ($c_{\text{skip}} = 1$, $c_{\text{out}} = -\sigma$, $c_{\text{in}} = 1/\sqrt{\sigma^2+1}$), uniform timesteps for training, and weight $1/\sigma^2$, derived below.

#### Where VE's $\sigma(t) = \sqrt{t}$ comes from

VE has no scaling, $s = 1$. Its original sampler used noise levels $\bar\sigma_i = \sigma_{\min}(\sigma_{\max}/\sigma_{\min})^{1 - i/(N-1)}$ and the update $x_{i+1} = x_i + \tfrac12(\bar\sigma_i^2 - \bar\sigma_{i+1}^2)\nabla\log p$. The paper shows this is exactly Euler on Eq. 1 with $\sigma(t) = \sqrt{t}$ and $t_i = \bar\sigma_i^2$:

$$
\begin{aligned}
x_{i+1} &= x_i - (t_{i+1} - t_i)\,\dot\sigma\sigma\,\nabla\log p && \text{(Euler step on Eq. 1)}\\
\dot\sigma\,\sigma &= \frac{1}{2\sqrt{t}}\cdot\sqrt{t} = \frac12 && (\sigma = \sqrt t)\\
x_{i+1} &= x_i + \tfrac12\,(t_i - t_{i+1})\,\nabla\log p = x_i + \tfrac12\,(\bar\sigma_i^2 - \bar\sigma_{i+1}^2)\,\nabla\log p && (t_i = \bar\sigma_i^2).
\end{aligned}
$$

The VE parameters are doubled from Song et al.'s ($0.01$ and $50$) because those were defined for images in $[0, 1]$, and so is the $\frac12$ inside $c_{\text{noise}} = \ln(\frac12\sigma)$: the original network took $\ln\sigma$ measured in $[0, 1]$ units.

#### Where iDDPM + DDIM's $\sigma(t) = t$ comes from

Deterministic DDIM, written in the variable $x_t/\sqrt{\bar\alpha_t}$, is an Euler step of $dx = \hat\epsilon\,d\sigma$ (the [diffusion models](diffusion-models.html) page derives this). Since $\hat\epsilon = (x - D)/\sigma = -\sigma\nabla\log p$, that is $dx = -\sigma\nabla\log p\,d\sigma$, which is Eq. 1 with $\sigma(t) = t$. So DDIM already used EDM's preferred schedule; it just used Euler and iDDPM's time grid.

#### The preconditioning rows

Read each column as "how does the code turn the network output into a clean image?" For VP and iDDPM the network predicts unit-variance noise from an input scaled to unit variance, and the clean image is $D = x - \sigma F_\theta(x/\sqrt{\sigma^2+1})$; this is $\epsilon$-prediction, with $c_{\text{skip}} = 1$ and $c_{\text{out}} = -\sigma$. For VE the network output times $1/\sigma$ is the score, so $D = x + \sigma F_\theta(x)$, with no input scaling at all. EDM's column is derived in Section 5. The $c_{\text{noise}}$ row is how the noise level is fed to the network: VP passes the continuous time scaled to DDPM's index range, $999\,t$; VE passes a log of $\sigma$; iDDPM passes the integer timestep of the nearest trained level (indices run in the opposite direction, hence $M - 1 - j$); EDM passes $\frac14\ln\sigma$.

#### The training rows, and why all three predecessors share $\lambda = 1/\sigma^2$

All three predecessors train the network to predict noise with a plain MSE. Rewrite that loss in terms of $D$:

$$
\begin{aligned}
\epsilon &= n/\sigma = (x - y)/\sigma && (x = y + n)\\
F_\theta &= (x - D_\theta)/\sigma && \text{(invert } D_\theta = x - \sigma F_\theta)\\
F_\theta - \epsilon &= (y - D_\theta)/\sigma && \text{(subtract; } x \text{ cancels)}\\
\lVert F_\theta - \epsilon\rVert^2 &= \frac{1}{\sigma^2}\,\lVert D_\theta - y\rVert^2 && \text{(square).}
\end{aligned}
$$

(The VE case is the same with the sign of $F$ flipped.) So in the common language all three have $\lambda(\sigma) = 1/\sigma^2$, and their training rows differ only in $p_{\text{train}}$: uniform in VP time, log-uniform in $\sigma$ for VE, uniform over 1000 discrete levels for iDDPM.

#### What the paper gained just by reimplementing

Running each method's own choices through EDM's Algorithm 1 matched or slightly beat the original code (Figure 2, blue versus orange curves). The appendix traces the gaps to oversights: the VP code multiplied the Euler derivative by $-1/N$ instead of $(\epsilon_s - 1)/(N-1)$ and over- or undershot the last step; the VE code accumulated floating-point round-off at thousands of steps; for iDDPM the difference is the $j_0 = 8$ choice. EDM runs the sampler state in float64 and the network in float32.

**What this says.** The table turns "VP versus VE versus DDPM" from three theories into a handful of different entries in the same rows. DDPM is VP with $\epsilon$-prediction, $\lambda = 1/\sigma^2$ and uniform timesteps; DDIM is that network sampled by Euler with $\sigma(t) = t$. Once written this way, nothing forces the entries in one row to follow from another, which is what lets Sections 3 to 5 change them one at a time.

::: key
*How to say it in an interview:* "EDM writes the probability-flow ODE in terms of the marginals: a noise schedule $\sigma(t)$ and a scale $s(t)$, $dx/dt = (\dot s/s)\,x - s^2\dot\sigma\sigma\,\nabla\log p(x/s;\sigma)$. The score comes from a denoiser through Tweedie, $\nabla\log p = (D - x)/\sigma^2$. With that, any model is a row of choices: schedule, scaling, solver, time grid, network pre- and post-scaling, training noise distribution and loss weight. DDPM is VP with $\epsilon$-prediction, $\lambda = 1/\sigma^2$ and uniform timesteps, and DDIM is Euler on the $\sigma(t) = t$ ODE."
:::

::: qa
**Derive the probability-flow ODE in EDM's form, with a noise schedule $\sigma(t)$ and a scale schedule $s(t)$.**

Start without scaling. The blurred density $p(x;\sigma) = p_{\text{data}} * \mathcal{N}(0, \sigma^2 I)$ satisfies a heat equation, because the Gaussian kernel does: $\partial_\sigma G_\sigma = \sigma\,\Delta G_\sigma$, so with $\sigma = \sigma(t)$, $\partial_t p = \dot\sigma\sigma\,\Delta p$. Writing $\Delta p = \nabla\cdot(p\nabla\log p)$ turns this into a continuity equation with velocity $-\dot\sigma\sigma\nabla\log p$, giving $dx = -\dot\sigma\sigma\nabla\log p(x;\sigma)\,dt$. For a scaled variable $x = s\hat{x}$, the product rule gives $\dot x = (\dot s/s)\,x + s\,\dot{\hat x}$, and the chain rule gives $(\nabla\log p)(\hat x) = s\nabla_x\log p(x/s;\sigma)$, so $\dot x = (\dot s/s)\,x - s^2\dot\sigma\sigma\nabla_x\log p(x/s;\sigma)$. Matching means and variances of Song's linear SDE gives the same result with $f = \dot s/s$ and $g = s\sqrt{2\dot\sigma\sigma}$.
:::

::: qa
**Prove that $\nabla_x\log p(x;\sigma) = (D(x;\sigma) - x)/\sigma^2$ for the optimal denoiser.**

The $L_2$-optimal denoiser is the posterior mean $\mathbb{E}[y \mid x]$, since for each $x$ the expected loss is a quadratic in $D(x)$ minimized at the conditional mean. Differentiate $p(x;\sigma) = \int p_{\text{data}}(y)\,\mathcal{N}(x; y, \sigma^2 I)\,dy$: the Gaussian's gradient is $\mathcal{N}\cdot(y - x)/\sigma^2$. Dividing by $p(x;\sigma)$ turns $p_{\text{data}}(y)\mathcal{N}(x;y,\sigma^2 I)/p(x;\sigma)$ into the posterior $p(y \mid x)$, so the score is $\mathbb{E}[(y - x)/\sigma^2 \mid x] = (\mathbb{E}[y \mid x] - x)/\sigma^2$. The practical consequence is that a network trained with a plain MSE denoising loss is a score model, with no normalizing constant involved.
:::

::: qa
**Where does DDPM sit in EDM's Table 1, and why do VP, VE and DDPM all have $\lambda(\sigma) = 1/\sigma^2$?**

DDPM is the VP column: $s = 1/\sqrt{1+\sigma^2}$ (that is, $\sqrt{\bar\alpha_t}$), $\epsilon$-prediction, which in EDM's terms is $c_{\text{skip}} = 1$, $c_{\text{out}} = -\sigma$, $c_{\text{in}} = 1/\sqrt{\sigma^2+1}$, and training on uniformly sampled timesteps. Its noise MSE equals the denoising MSE divided by $\sigma^2$: with $D = x - \sigma F$ and $\epsilon = (x - y)/\sigma$, $F - \epsilon = (y - D)/\sigma$. VE predicts a scaled score, which gives the same identity with a sign flip. So the three families share $\lambda = 1/\sigma^2$ and differ in $p_{\text{train}}$, the time grid and the schedule. DDIM is the same network sampled by Euler with $\sigma(t) = t$, $s = 1$.
:::

## 3. Deterministic sampling

### 3.1 Setup: three pretrained models as black boxes

The hypothesis is that sampling choices ($\sigma(t)$, $s(t)$, the time grid, the solver) are independent of how the network was trained; to the sampler, $D_\theta$ is a black box. The paper tests this on three public checkpoints, each from a different theory:

- **VP**: "DDPM++ cont. (VP)" from Song et al., unconditional CIFAR-10 at 32×32, 62M parameters, trained on $\sigma$ in about $[0.001, 152]$.
- **VE**: "NCSN++ cont. (VE)" from Song et al., same dataset, 63M parameters, trained on $\sigma \in [0.02, 100]$. Because the model never saw $\sigma < 0.02$, these experiments use $\sigma_{\min} = 0.02$.
- **ADM**: the "ADM (dropout)" model of Dhariwal and Nichol, class-conditional ImageNet at 64×64, 296M parameters, trained on $M = 1000$ discrete noise levels from about 20,291 down to 0.0064. Sampling with it requires rounding each $\sigma_i$ to the nearest trained level, and $\sigma_{\min} = 0.0064$.

In general the paper sets $\sigma_{\min} = \max(\sigma_{\text{lo}}, 0.002)$ and $\sigma_{\max} = \min(\sigma_{\text{hi}}, 80)$, where $\sigma_{\text{lo}}, \sigma_{\text{hi}}$ are the lowest and highest levels a model supports.

Quality is FID between 50,000 generated images and all real images, computed three times and the minimum reported (run-to-run variation is about ±2%). Cost is NFE, the number of evaluations of $D_\theta$ per image, which dominates sampling time. To summarize an FID-versus-NFE curve by two numbers, the paper reports the lowest FID on the curve and the smallest NFE whose FID is within 3% of that lowest value.

### 3.2 Euler versus Heun: truncation error

An ODE solver approximates the true trajectory. Write the ODE as $\dot x = f(x, t)$ and a step as $h = t_{i+1} - t_i$ (negative when sampling, since time goes down; only $\lvert h\rvert$ matters below). The local truncation error $\tau_i$ is the error of one step started from the exact value.

**Euler.** Taylor-expand the true solution:

$$
\begin{aligned}
x(t+h) &= x + h\,\dot x + \tfrac{h^2}{2}\,\ddot x + O(h^3) && \text{(Taylor in } h)\\
\text{Euler: } x_{\text{E}} &= x + h\,f(x,t) = x + h\,\dot x && \text{(one slope evaluation)}\\
\tau_{\text{Euler}} &= x(t+h) - x_{\text{E}} = \tfrac{h^2}{2}\,\ddot x + O(h^3) = O(h^2) && \text{(subtract).}
\end{aligned}
$$

**Heun** (also called improved Euler or the explicit trapezoidal rule) takes the Euler step, evaluates the slope again at the end, and averages the two slopes:

$$
\begin{aligned}
x_{\text{H}} &= x + \tfrac{h}{2}\Big(f(x,t) + f\big(x + h f(x,t),\; t + h\big)\Big) && \text{(definition)}\\
f(x + hf,\, t + h) &= f + h\,\big(\partial_t f + (\partial_x f)\,f\big) + O(h^2) && \text{(Taylor in both arguments)}\\
\partial_t f + (\partial_x f)\,f &= \frac{d}{dt}f\big(x(t),t\big) = \ddot x && \text{(chain rule along the trajectory)}\\
x_{\text{H}} &= x + \tfrac{h}{2}\big(2\dot x + h\,\ddot x + O(h^2)\big) = x + h\dot x + \tfrac{h^2}{2}\ddot x + O(h^3) && \text{(substitute)}\\
\tau_{\text{Heun}} &= x(t+h) - x_{\text{H}} = O(h^3) && \text{(matches Taylor through } h^2).
\end{aligned}
$$

So halving the step cuts Euler's local error by 4 and Heun's by 8. Errors accumulate over the $N$ steps into a global error $e_N$; if $D_\theta$ is Lipschitz (true for the networks used), $\lVert e_N\rVert \le E\,\max_i\lVert\tau_i\rVert$ for a constant $E$ depending on $N$, the endpoints and the Lipschitz constant. Reducing the global error at fixed $N$, and so reducing $N$ itself, comes down to choosing the solver and the grid so the largest local error is small.

Heun costs one extra evaluation of $D_\theta$ per step. Higher-order Runge-Kutta methods shrink the error faster but need more evaluations per step, and linear multistep methods had also been proposed for diffusion sampling; after extensive tests the paper found Heun gave an excellent trade-off between truncation error and NFE. An adaptive RK45 solver (Dormand-Prince) with EDM's schedule was also tested and its overhead outweighed its benefit (Section 3.6).

**The second-order family.** Heun is one member of the explicit two-stage second-order Runge-Kutta methods, $x_{i+1} = x_i + h\big[(1 - \tfrac{1}{2\alpha})\,d_i + \tfrac{1}{2\alpha}\,f(x_i + \alpha h d_i,\, t_i + \alpha h)\big]$: $\alpha = 1$ is Heun, $\alpha = \frac12$ the midpoint method, $\alpha = \frac23$ Ralston's. All cost the same. In a separate experiment $\alpha = 1.1$ was slightly better than $\alpha = 1$, which the authors could not explain (it overshoots $t_{i+1}$), so they kept $\alpha = 1$. Heun has a practical advantage: its second evaluation is exactly at $t_{i+1}$, a grid point, so a model trained only on discrete noise levels (like ADM) can be used by rounding the grid points alone.

### 3.3 Algorithm 1, step by step

The paper's Algorithm 1, for general $\sigma(t)$ and $s(t)$:

1. Sample $x_0 \sim \mathcal{N}\big(0,\ \sigma(t_0)^2\,s(t_0)^2\,I\big)$. This is the starting noise at time $t_0$, with the total noise standard deviation $s\sigma$.
2. For $i = 0, \dots, N-1$:
3. Evaluate the slope $d_i = \big(\frac{\dot\sigma(t_i)}{\sigma(t_i)} + \frac{\dot s(t_i)}{s(t_i)}\big)\,x_i - \frac{\dot\sigma(t_i)\,s(t_i)}{\sigma(t_i)}\,D_\theta\big(\frac{x_i}{s(t_i)};\sigma(t_i)\big)$, the ODE of Section 2.6.
4. Take an Euler step: $x_{i+1} \leftarrow x_i + (t_{i+1} - t_i)\,d_i$.
5. If $\sigma(t_{i+1}) \ne 0$, evaluate the slope $d_i'$ at $(x_{i+1}, t_{i+1})$ with the same formula, and replace the step by the trapezoidal average: $x_{i+1} \leftarrow x_i + (t_{i+1} - t_i)\,\big(\tfrac12 d_i + \tfrac12 d_i'\big)$.
6. Return $x_N$.

Dropping step 5 gives Euler's method. With EDM's $\sigma(t) = t$, $s = 1$, step 3 becomes $d_i = (x_i - D_\theta(x_i; t_i))/t_i$.

**Why the last step is Euler.** The final step goes to $t_N = 0$, where the slope $(x - D_\theta)/t$ divides by zero, so step 5 is skipped. Nothing is lost: an Euler step from $t_{N-1} = \sigma_{\min}$ to $0$ outputs exactly $x_N = D_\theta(x_{N-1};\sigma_{\min})$ (Section 2.6), the denoised estimate at the smallest noise level.

**Counting NFE.** Steps $0$ to $N-2$ each cost two evaluations, and the last step costs one, so NFE $= 2(N-1) + 1 = 2N - 1$. The paper's settings are $N = 18$ for CIFAR-10 (35 NFE) and $N = 40$ at 64×64 (79 NFE).

```python
import numpy as np

def heun_sampler(D, shape, sigmas, rng):          # sigmas: sigma_0 > ... > sigma_N = 0
    x = rng.standard_normal(shape) * sigmas[0]     # x_0 ~ N(0, sigma_max^2 I)
    for s_cur, s_next in zip(sigmas[:-1], sigmas[1:]):
        d = (x - D(x, s_cur)) / s_cur              # slope dx/dsigma at s_cur
        x_next = x + (s_next - s_cur) * d          # Euler step
        if s_next > 0:                             # 2nd-order correction except last step
            d2 = (x_next - D(x_next, s_next)) / s_next
            x_next = x + (s_next - s_cur) * 0.5 * (d + d2)
        x = x_next
    return x                                       # NFE = 2N - 1
```

Here `D(x, sigma)` is the trained denoiser $D_\theta$; with $\sigma(t) = t$ the time grid is the list of noise levels. The official sampler does the same in float64.

### 3.4 Where to put the steps: the $\rho$ grid

The time steps decide how the step sizes, and so the truncation errors, are spread over noise levels. The paper first measured local errors on the VE model (below), concluded that the step size should shrink monotonically as $\sigma$ decreases and need not vary per sample, and then chose a one-parameter family.

**Deriving the formula.** Make $\sigma_i^{1/\rho}$ linear in $i$, so $\sigma_{i<N} = (A i + B)^\rho$ with $\sigma_N = 0$ appended. Fix $A, B$ by the endpoints:

$$
\begin{aligned}
\sigma_0 = \sigma_{\max} &\;\Rightarrow\; B = \sigma_{\max}^{1/\rho} && (i = 0)\\
\sigma_{N-1} = \sigma_{\min} &\;\Rightarrow\; A(N-1) + \sigma_{\max}^{1/\rho} = \sigma_{\min}^{1/\rho} && (i = N-1)\\
A &= \frac{\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}}{N-1} && \text{(solve)}\\
\sigma_{i<N} &= \Big(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}\big(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\big)\Big)^\rho, \qquad \sigma_N = 0 && \text{(substitute).}
\end{aligned}
$$

This is the paper's Eq. 5. The paper calls it a linear resampling of a warp function $w(z) = z^\rho$.

**How the step size depends on $\sigma$.** Let $u = \sigma^{1/\rho}$; the grid moves $u$ by the constant amount $\lvert A\rvert$ per step. Then

$$
\begin{aligned}
\sigma &= u^\rho && \text{(definition of } u)\\
\frac{d\sigma}{du} &= \rho\,u^{\rho-1} = \rho\,\sigma^{(\rho-1)/\rho} && \text{(differentiate; } u^{\rho-1} = \sigma^{(\rho-1)/\rho})\\
h(\sigma) \approx \lvert A\rvert\,\frac{d\sigma}{du} &\propto \sigma^{1 - 1/\rho} && \text{(step in } \sigma \text{ for a step } \lvert A\rvert \text{ in } u).
\end{aligned}
$$

So $\rho = 1$ gives uniform steps; larger $\rho$ gives steps that shrink faster as $\sigma \to 0$.

**The limit $\rho \to \infty$ is the geometric grid of the VE sampler.** Write $r = \sigma_{\min}/\sigma_{\max}$ and $a = i/(N-1)$:

$$
\begin{aligned}
\sigma_i &= \sigma_{\max}\,\big(1 + a\,(r^{1/\rho} - 1)\big)^\rho && \text{(factor out } \sigma_{\max}^{1/\rho})\\
r^{1/\rho} - 1 &= e^{(\ln r)/\rho} - 1 = \frac{\ln r}{\rho} + O(\rho^{-2}) && \text{(expand the exponential)}\\
\sigma_i &= \sigma_{\max}\Big(1 + \frac{a\ln r}{\rho} + O(\rho^{-2})\Big)^\rho \;\to\; \sigma_{\max}\,e^{a\ln r} = \sigma_{\max}\,r^{\,i/(N-1)} && \big((1 + c/\rho)^\rho \to e^c\big).
\end{aligned}
$$

So EDM's grid generalizes the original VE discretization; $\rho$ interpolates between uniform and geometric spacing.

**What the paper measured.** On the VE CIFAR-10 model, the paper estimated the local error at each $\sigma_i$ by comparing one solver step with 200 small Euler steps over the same interval, averaged over 200 samples, as a per-pixel RMSE:

- With Euler and uniform steps of $1.25$ in $\sigma$, the error is very large (RMSE about 0.56) for $\sigma_i \le 0.5$ and much smaller at high noise. Its spread across samples is tiny, so a per-sample adaptive grid would not help.
- With Euler and the $\rho$ grid, raising $\rho$ lowers the error below $\sigma \approx 10$ and raises it above. Rough balance occurs at $\rho = 2$, but the RMSE stays about 0.03, several quantization levels per step.
- With Heun and $\rho = 3$, the RMSE is almost constant, between 0.0030 and 0.0045, across all noise levels. So Heun with $\rho = 3$ nearly equalizes the local truncation error.

**A rule of thumb for the values 2 and 3 (not from the paper).** The paper reports these values as measurements and does not explain them. A scaling argument reproduces them, but only under an assumption chosen to make it work, so read it as a memory aid, not a derivation.

With $\sigma(t) = t$ the trajectory's slope is $x'(\sigma) = (x - D)/\sigma = \hat\epsilon$, the network's estimate of the unit noise, which is of order one per pixel. (Primes denote derivatives in $\sigma$ along the trajectory.) Assume $\hat\epsilon$ varies on a logarithmic scale: it changes by an amount of order one when $\sigma$ changes by a constant *factor*, not by a constant *amount*. Then $d\hat\epsilon/d\ln\sigma \sim 1$, and since $d/d\sigma = (1/\sigma)\,d/d\ln\sigma$, each further $\sigma$-derivative brings one factor $1/\sigma$:

$$
\begin{aligned}
x' &= \hat\epsilon \sim 1, \qquad x'' = \frac{1}{\sigma}\frac{d\hat\epsilon}{d\ln\sigma} \sim \sigma^{-1}, \qquad x^{(k)} \sim \sigma^{1-k} && \text{(the assumption)}\\
\tau &\sim h^{p+1}\,\lvert x^{(p+1)}\rvert \sim h^{p+1}\,\sigma^{-p} && \text{(local error of an order-}p\text{ method, from 3.2)}\\
\tau = \text{const} &\;\Rightarrow\; h \propto \sigma^{p/(p+1)} && \text{(solve for } h)\\
1 - \tfrac{1}{\rho} = \tfrac{p}{p+1} &\;\Rightarrow\; \rho = p + 1 && \text{(match the grid's } h \propto \sigma^{1 - 1/\rho}).
\end{aligned}
$$

Euler ($p = 1$) gives $\rho = 2$ and Heun ($p = 2$) gives $\rho = 3$, the two values the paper measured. Three caveats:

- *The answer is set by the assumption.* Since $D = x - \sigma\hat\epsilon$ and $x' = \hat\epsilon$, the product rule gives $dD/d\ln\sigma = -\sigma\,d\hat\epsilon/d\ln\sigma$. So the assumption says the denoised image moves by an amount proportional to $\sigma$ per factor-of-$e$ change in $\sigma$. If instead it moved by a fixed amount per factor of $e$, the same steps would give $x^{(k)} \sim \sigma^{-k}$ for $k \ge 2$, then $h \propto \sigma$ for every solver, which is the geometric grid ($\rho \to \infty$). Nothing in the paper measures which scaling holds.
- *The Taylor estimate needs $h \ll \sigma$.* Near $\sigma_{\min}$ these grids take steps comparable to $\sigma$ itself; in the $\rho = 7$ table below, the step from $\sigma_{16}$ to $\sigma_{17}$ shrinks $\sigma$ by a factor of 3.8.
- *One model.* The measurements behind 2 and 3 come from a single network (VE on CIFAR-10).

**Why the paper uses $\rho = 7$ anyway.** Equal raw error is the right target when the error itself matters, for example when an image is encoded to noise by running the ODE forward and then decoded back; the paper recommends $\rho = 3$ there. For generating new images, errors at different noise levels do not matter equally. Sweeping $\rho$ against FID for several models and step counts, $\rho = 3$ is good but $\rho$ between 5 and 10 is better, and $\rho = 7$ "performs reasonably well in all cases". A larger $\rho$ deliberately accepts more error at high noise to reduce it at low noise. The paper's explanation is that $\sigma_{\max}$ is somewhat arbitrary: changing it can change the global error a lot while barely changing the distribution of final images. Errors near $\sigma_{\min}$ end up directly in the visible detail.

#### Worked example: the 18-step grids

Take $N = 18$, $\sigma_{\min} = 0.002$, $\sigma_{\max} = 80$ (the CIFAR-10 setting) and compute $\sigma_0, \dots, \sigma_{17}$ for three values of $\rho$; $\sigma_{18} = 0$ in every case. You can reproduce any entry by hand from Eq. 5.

| $i$ | $\rho = 1$ | $\rho = 3$ | $\rho = 7$ |
|---|---|---|---|
| 0 | 80 | 80 | 80 |
| 1 | 75.29 | 67.06 | 57.59 |
| 2 | 70.59 | 55.60 | 40.79 |
| 3 | 65.88 | 45.53 | 28.37 |
| 4 | 61.18 | 36.75 | 19.35 |
| 5 | 56.47 | 29.18 | 12.91 |
| 6 | 51.77 | 22.73 | 8.401 |
| 7 | 47.06 | 17.30 | 5.315 |
| 8 | 42.35 | 12.82 | 3.257 |
| 9 | 37.65 | 9.187 | 1.923 |
| 10 | 32.94 | 6.315 | 1.088 |
| 11 | 28.24 | 4.114 | 0.5853 |
| 12 | 23.53 | 2.495 | 0.2964 |
| 13 | 18.83 | 1.368 | 0.1395 |
| 14 | 14.12 | 0.6453 | 0.05995 |
| 15 | 9.414 | 0.2361 | 0.02293 |
| 16 | 4.708 | 0.0515 | 0.007528 |
| 17 | 0.002 | 0.002 | 0.002 |

For example, with $\rho = 7$ and $i = 1$: $80^{1/7} = 1.8701$ and $0.002^{1/7} = 0.4116$, so $\sigma_1 = (1.8701 - \tfrac{1}{17}\times 1.4586)^7 = 1.7843^7 \approx 57.6$.

How to read it. With $\rho = 1$, sixteen of the seventeen intervals are spent above $\sigma = 4.7$, and the last step jumps from $4.7$ to $0.002$, where all the fine detail is decided; that grid would be useless. With $\rho = 7$, seven of the eighteen levels are below $\sigma = 1$ and four are below $0.1$, while only six are above 10. The ratio between consecutive levels rises smoothly from $1.39$ at the top to $3.76$ at the bottom. For comparison, a geometric grid with the same endpoints has a constant ratio of $(80/0.002)^{1/17} \approx 1.87$. The $\rho = 3$ column sits in between.

```python
def edm_sigmas(N=18, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    i = np.arange(N)
    s = (sigma_max ** (1 / rho) + i / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return np.append(s, 0.0)                       # sigma_N = 0
```

### 3.5 Choosing $\sigma(t)$ and $s(t)$: straight trajectories

The grid decides where to step; the schedule decides the shape of the path being stepped along. Truncation error grows with the curvature of the trajectory, so the best $\sigma(t)$ and $s(t)$ are those that make trajectories as straight as possible. The paper argues for $\sigma(t) = t$ and $s(t) = 1$, the choice DDIM already made.

**The tangent points at the denoiser output.** With this choice the ODE is $dx/dt = (x - D(x;t))/t$, and (Section 2.6) one Euler step from anywhere to $t = 0$ lands exactly on $D(x;t)$. The trajectory's direction at every point is the line toward the current denoised estimate. If $D$ changes slowly as the noise level drops, the direction changes slowly and the path is nearly straight. The paper's Figure 1 shows that, for real images, the denoiser's output changes mainly within a fairly narrow band of $\sigma$; above it the output is close to the dataset mean, below it close to the final image. So the curvature is concentrated in that band, and trajectories are close to straight lines at both high and low noise.

**Worked example: Gaussian data.** Suppose the data are one-dimensional Gaussian, $y \sim \mathcal{N}(0, \sigma_{\text{data}}^2)$, so the optimal denoiser is linear, the standard Gaussian shrinkage $D(x;\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma_{\text{data}}^2 + \sigma^2}\,x$. Solve EDM's ODE with $t = \sigma$:

$$
\begin{aligned}
x - D &= \Big(1 - \frac{\sigma_{\text{data}}^2}{\sigma_{\text{data}}^2 + \sigma^2}\Big)x = \frac{\sigma^2}{\sigma_{\text{data}}^2 + \sigma^2}\,x && \text{(subtract)}\\
\frac{dx}{d\sigma} &= \frac{x - D}{\sigma} = \frac{\sigma}{\sigma_{\text{data}}^2 + \sigma^2}\,x && \text{(the ODE)}\\
\frac{d\ln x}{d\sigma} &= \frac{\sigma}{\sigma_{\text{data}}^2 + \sigma^2} = \frac{d}{d\sigma}\,\tfrac12\ln(\sigma_{\text{data}}^2 + \sigma^2) && \text{(divide by } x)\\
x(\sigma) &= K\sqrt{\sigma_{\text{data}}^2 + \sigma^2} && \text{(integrate; } K \text{ constant per trajectory).}
\end{aligned}
$$

Each trajectory is a hyperbola. For $\sigma \gg \sigma_{\text{data}}$, $x \approx K\sigma$, a straight line aimed at $0$, the data mean. For $\sigma \ll \sigma_{\text{data}}$, $x \approx K\sigma_{\text{data}}$, flat, already at its final value. All the bending happens near $\sigma \approx \sigma_{\text{data}}$. This is the behavior the paper sketches for a two-point dataset in its Figure 3c, where trajectories approach straight lines toward the data mean at large $\sigma$ and become linear toward the data near $\sigma = 0$.

**The same example in VP coordinates.** VP's ODE variable is $x_{\text{VP}} = s\,x = x/\sqrt{1 + \sigma^2}$ (Section 2.7), so here $x_{\text{VP}} = K\sqrt{\sigma_{\text{data}}^2 + \sigma^2}/\sqrt{1 + \sigma^2} \to K$ as $\sigma \to \infty$. At high noise the VP trajectories flatten into horizontal lines whose slopes say nothing about the data; the direction toward the data only appears at small $\sigma$. That matches Figure 3a. The VE schedule $\sigma = \sqrt{t}$ squeezes the region $\sigma \lesssim \sigma_{\text{data}}$ into a tiny interval $t \lesssim \sigma_{\text{data}}^2$ and makes $x \approx K\sqrt{t}$ curved at large $t$, so its trajectories are curved everywhere and sharply curved near the data (Figure 3b).

**What this says.** Straightness is a property of the coordinates, not of the model. The same trained denoiser gives curved or nearly straight trajectories depending on $\sigma(t)$ and $s(t)$, and changing them is free at sampling time.

### 3.6 Results on the pretrained models

The paper's Figure 2 plots FID against NFE for each model as each change is added; the appendix summarizes each curve by its lowest FID and the smallest NFE within 3% of it.

| Deterministic sampler | VP FID | VP NFE | VE FID | VE NFE | ImageNet-64 FID | ImageNet-64 NFE |
|---|---|---|---|---|---|---|
| Original sampler | 2.85 | 256 | 5.45 | 8192 | 2.85 | 250 |
| Reimplementation in Algorithm 1 (Euler) | 2.79 | 512 | 4.78 | 8192 | 2.73 | 384 |
| + Heun and the $\rho$ grid | 2.88 | 255 | 4.23 | 191 | 2.64 | 79 |
| + $\sigma(t) = t$, $s(t) = 1$ | 2.93 | 35 | 3.73 | 27 | (same as above) | (same) |
| Adaptive RK45 with EDM's schedule | 2.94 | 115 | 3.69 | 93 | 2.66 | 131 |

What to read from it:

- **The main gain is in NFE.** The final rows reach near-best quality with 35 instead of 256 evaluations for VP (7.3 times fewer), 27 instead of 8192 for VE (about 300 times), and 79 instead of 250 for ImageNet-64 (3.2 times). These are the factors quoted in the paper. For VE, best FID also improves, from 5.45 to 3.73.
- **The ImageNet-64 last row adds nothing** because DDIM already used $\sigma(t) = t$, $s = 1$; Heun and the grid did all the work there.
- **VP's best FID does not improve** (2.85 with the original sampler, 2.93 at the end); the gain is that near-best quality is reached with far fewer steps.
- **Adaptive RK45 does not pay off**: it reaches similar FID but needs 115, 93 and 131 evaluations, because the cost of error control outweighs its benefit here.

In wall-clock terms, the paper generates 26.3 high-quality CIFAR-10 images per second on one V100. The consistency of the gains across three unrelated models is the evidence for the paper's separability hypothesis.

**What this says.** For a pretrained model, sampler choices alone are worth large factors in compute. The three ingredients are a second-order solver, a grid that spends steps at low noise, and a schedule whose trajectories are nearly straight. None of them touches the weights.

::: key
*How to say it in an interview:* "EDM's deterministic sampler is Heun's method on $dx/d\sigma = (x - D(x;\sigma))/\sigma$. Heun averages the slope at both ends of the step, which makes the local error $O(h^3)$ instead of Euler's $O(h^2)$, at the price of a second evaluation, and the last step to $\sigma = 0$ is plain Euler, so $N$ steps cost $2N - 1$ evaluations. Steps follow $\sigma_i^{1/\rho}$ uniform in $i$ with $\rho = 7$, which packs steps at low noise; $\rho = 3$ would equalize the raw truncation error, but $\rho = 7$ gives better FID because low-noise errors show up in the image. With $\sigma(t) = t$ the tangent always points at the denoised image, so trajectories are nearly straight. On pretrained VP, VE and ADM models this cut the NFE needed by 7, 300 and 3 times."
:::

::: pitfall
"$\rho = 7$ minimizes truncation error." It does not; with Heun, $\rho \approx 3$ makes the measured local error nearly constant across noise levels. $\rho = 7$ was chosen by FID, deliberately shifting error toward high noise, where it matters less for the final image.
:::

::: qa
**Why is a Heun step worth two network evaluations, and why is EDM's last step Euler?**

Euler assumes the slope is constant over the step, so its local error is $\tfrac{h^2}{2}\ddot x = O(h^2)$. Heun re-evaluates the slope at the Euler endpoint and averages: expanding $f(x + hf, t + h) = f + h\ddot x + O(h^2)$ shows the update matches the Taylor series through $h^2$, so the local error is $O(h^3)$. Halving the step then cuts the error by 8 instead of 4, so fewer, larger steps reach the same accuracy, which more than pays for the extra evaluation; the paper found Heun the best trade-off among the solvers it tried. The final step goes to $\sigma = 0$, where the slope $(x - D)/\sigma$ is undefined, so the correction is skipped; the Euler step there simply outputs $D_\theta(x;\sigma_{\min})$. Total cost: $2N - 1$ evaluations, for example 35 for $N = 18$.
:::

::: qa
**Why does EDM use $\rho = 7$ for its step grid when $\rho = 3$ equalizes the truncation error?**

The grid makes $\sigma_i^{1/\rho}$ linear in $i$, so the step size scales like $\sigma^{1 - 1/\rho}$: larger $\rho$ means relatively smaller steps at low noise. The paper measured Heun's local error on a VE model and found $\rho = 3$ makes it nearly constant (RMSE 0.0030 to 0.0045 at every $\sigma$), which is ideal when raw numerical accuracy matters, for example for ODE inversion. For generating images, FID was better for $\rho$ between 5 and 10, so they intentionally accept more error at high noise, where $\sigma_{\max}$ is somewhat arbitrary and errors barely change the output distribution, to reduce error at low noise, where it becomes visible detail. $\rho = 7$ worked well on every model they tried. The values 3 (Heun) and 2 (Euler) are measurements; the paper does not derive them.
:::

::: qa
**Why does EDM choose $\sigma(t) = t$ and $s(t) = 1$?**

Truncation error scales with trajectory curvature, and this schedule gives nearly straight trajectories. The ODE becomes $dx/dt = (x - D(x;t))/t$, so an Euler step to $t = 0$ lands exactly on $D(x;t)$: the tangent always points at the current denoised image. The denoiser's output changes mostly within a narrow band of $\sigma$ (near the dataset mean above it, near the final image below it), so the curvature is confined to that band. For Gaussian data the trajectories are exactly $x = K\sqrt{\sigma_{\text{data}}^2 + \sigma^2}$: straight toward the mean at high noise, flat at low noise. By contrast, VP trajectories go flat at high noise and VE trajectories are curved everywhere. DDIM already used this schedule, which is one reason it does well with few steps; on the ADM model, EDM's change of schedule therefore added nothing.
:::

::: qa
**What does a well-tuned sampler buy for a pretrained model? Give numbers.**

A lot, without touching the weights. EDM replaced the samplers of three public models with Heun, the $\rho = 7$ grid and $\sigma(t) = t$, $s = 1$: near-best FID went from 256 to 35 evaluations for the VP CIFAR-10 model, from 8192 to 27 for the VE model (with best FID improving from 5.45 to 3.73), and from 250 to 79 for ADM on ImageNet-64. With a tuned stochastic sampler the same ImageNet-64 model went from its reported FID of 2.07 to 1.55, close to the state of the art at the time. The lesson for practice is to compare models only under a fixed, good sampler, and to try sampler changes before retraining. The paper also warns the reverse: tuning stochastic sampling per model can hide or inflate differences between training setups.
:::

## 4. Stochastic sampling

### 4.1 Why add noise if the ODE is exact?

The ODE and the reverse SDE produce the same marginals in theory, and the ODE has real advantages: it is deterministic, invertible (an image can be mapped to its latent noise and back), and needs fewer steps. Yet earlier work found stochastic samplers, which inject fresh noise at each step, gave better final quality. The paper asks what the noise actually does, and answers by rewriting the SDE so the role of the noise is visible.

### 4.2 A family of SDEs with the same marginals

Keep $s(t) = 1$ (the scaled case follows the same way, and the paper omits it). We want stochastic processes whose density at time $t$ is $p(x;\sigma(t))$. Section 2.2 showed this density obeys $\partial_t p = \dot\sigma\sigma\,\Delta p$. An SDE $dx = f(x,t)\,dt + g(t)\,d\omega_t$, with drift $f$, scalar noise strength $g$ and Wiener process $\omega_t$, has a density that obeys the Fokker-Planck equation $\partial_t p = -\nabla\cdot(f\,p) + \tfrac12 g^2\,\Delta p$. Match the two:

$$
\begin{aligned}
-\nabla\cdot(f p) + \tfrac12 g^2\Delta p &= \dot\sigma\sigma\,\Delta p && \text{(require the same evolution)}\\
\nabla\cdot(f p) &= \big(\tfrac12 g^2 - \dot\sigma\sigma\big)\,\Delta p && \text{(rearrange)}\\
\text{try } f\,p = \upsilon(t)\,\nabla p:\quad \nabla\cdot(\upsilon\nabla p) &= \upsilon\,\Delta p && (\nabla\cdot\nabla = \Delta)\\
\upsilon(t) &= \tfrac12 g(t)^2 - \dot\sigma(t)\sigma(t) && \text{(compare coefficients of } \Delta p)\\
f(x,t) &= \upsilon\,\frac{\nabla p}{p} = \big(\tfrac12 g^2 - \dot\sigma\sigma\big)\,\nabla_x\log p(x;\sigma) && \text{(divide by } p).
\end{aligned}
$$

So for any choice of $g(t)$,

$$
dx = \big(\tfrac12 g(t)^2 - \dot\sigma(t)\sigma(t)\big)\,\nabla_x\log p\big(x;\sigma(t)\big)\,dt + g(t)\,d\omega_t
$$

has the right marginals; $g = 0$ is the ODE of Section 2.2. The paper re-parameterizes the free function as $g(t) = \sqrt{2\beta(t)}\,\sigma(t)$, so $\tfrac12 g^2 = \beta\sigma^2$, which gives the forward SDE

$$
dx_+ = -\dot\sigma\sigma\,\nabla_x\log p\,dt + \beta(t)\,\sigma(t)^2\,\nabla_x\log p\,dt + \sqrt{2\beta(t)}\,\sigma(t)\,d\omega_t .
$$

**Reversing time.** For a forward SDE $dx = F\,dt + g\,d\omega$, Anderson's time-reversal formula gives the backward process $dx = (F - g^2\nabla\log p)\,dt + g\,d\bar\omega$, run with $dt < 0$, where $\bar\omega$ is a Wiener process running backward in time. Here

$$
\begin{aligned}
F - g^2\nabla\log p &= \big(-\dot\sigma\sigma + \beta\sigma^2 - 2\beta\sigma^2\big)\nabla\log p && (g^2 = 2\beta\sigma^2)\\
&= -\dot\sigma\sigma\,\nabla\log p - \beta\sigma^2\,\nabla\log p && \text{(collect).}
\end{aligned}
$$

So reversal only flips the sign of the middle term. The paper writes both directions as one equation (its Eq. 6):

$$
dx_\pm = \underbrace{-\dot\sigma(t)\sigma(t)\,\nabla_x\log p\big(x;\sigma(t)\big)\,dt}_{\text{probability-flow ODE}} \;\pm\; \underbrace{\underbrace{\beta(t)\sigma(t)^2\,\nabla_x\log p\big(x;\sigma(t)\big)\,dt}_{\text{deterministic noise decay}} + \underbrace{\sqrt{2\beta(t)}\,\sigma(t)\,d\omega_t}_{\text{noise injection}}}_{\text{Langevin diffusion}} .
$$

### 4.3 Reading the SDE: the ODE plus Langevin dynamics

**The bracketed pair is Langevin dynamics, which leaves $p(x;\sigma)$ unchanged at a frozen noise level.** Langevin dynamics for a density $q$ is $dz = \nabla\log q\,d\tau + \sqrt{2}\,d\omega_\tau$. Its Fokker-Planck equation is $\partial_\tau q = -\nabla\cdot(q\,\nabla\log q) + \Delta q = -\Delta q + \Delta q = 0$ (using $q\nabla\log q = \nabla q$), so $q$ is stationary. Rescale its clock by $d\tau = \beta\sigma^2\,dt$; a Wiener increment over $d\tau$ has the distribution of $\sqrt{\beta\sigma^2}$ times one over $dt$, so $dz = \beta\sigma^2\nabla\log q\,dt + \sqrt{2\beta}\,\sigma\,d\omega_t$, exactly the bracket.

**The two halves of the bracket cancel in noise level.** Use Section 2.5 to write the decay term as $\beta\sigma^2\nabla\log p\,dt = \beta\,(D - x)\,dt$. Since $x - D$ is the estimated noise in $x$, this removes a fraction $\beta\,dt$ of the noise. Meanwhile the injection adds independent noise of variance $2\beta\sigma^2\,dt$. Track the variance of the noise component (in the backward direction, where the decay is applied):

$$
\begin{aligned}
\text{after decay: } & (1 - \beta\,dt)^2\,\sigma^2 = \sigma^2 - 2\beta\sigma^2\,dt + O(dt^2) && \text{(scale the noise by } 1 - \beta\,dt)\\
\text{after injection: } & \sigma^2 - 2\beta\sigma^2\,dt + 2\beta\sigma^2\,dt + O(dt^2) = \sigma^2 + O(dt^2) && \text{(independent variances add).}
\end{aligned}
$$

So the Langevin part replaces old noise with new noise at relative rate $\beta(t)$ without changing the noise level; the ODE part is what moves the noise level. The paper calls $\beta(t)$ "the relative rate at which existing noise is replaced with new noise".

**Song et al.'s SDE is one particular $\beta$.** Choosing $\beta = \dot\sigma/\sigma$ makes $\beta\sigma^2 = \dot\sigma\sigma$, so the score terms cancel in the forward SDE, leaving $dx_+ = \sqrt{2\dot\sigma\sigma}\,d\omega_t$: forward diffusion becomes pure noise addition. The paper finds nothing special about this choice for sampling; it is simply the one that makes the forward process easy to write.

### 4.4 Why the Langevin part corrects errors, and why it is not free

During sampling the network is imperfect and each step has truncation error, so after a few steps the samples are distributed slightly differently from $p(x;\sigma_i)$. The ODE has no mechanism to notice this; it carries the error forward. The Langevin term, by contrast, drives whatever distribution it acts on toward its stationary distribution $p(x;\sigma(t))$, so it actively pulls the samples back to the correct marginal at each noise level. That is the paper's explanation of why stochastic samplers can beat deterministic ones.

The catch is that the Langevin term is itself approximated by discrete steps with an imperfect denoiser, which adds error of its own. Since earlier work gives no principled $\beta(t)$, the right amount of stochasticity has to be found empirically, and Algorithm 2 is built to make that search easy.

### 4.5 Algorithm 2, step by step

EDM's stochastic sampler combines the Heun ODE step with an explicit "churn" of adding and removing noise. It assumes $\sigma(t) = t$ and $s(t) = 1$, so $t$ is the noise level. Inputs: the denoiser $D_\theta$, the time grid $t_0 > \dots > t_N = 0$ (the $\rho$ grid), and four parameters $S_{\text{churn}}, S_{t\min}, S_{t\max}, S_{\text{noise}}$.

1. Sample $x_0 \sim \mathcal{N}(0, t_0^2 I)$.
2. For $i = 0, \dots, N-1$:
3. Set $\gamma_i = \min\big(S_{\text{churn}}/N,\ \sqrt2 - 1\big)$ if $t_i \in [S_{t\min}, S_{t\max}]$, else $\gamma_i = 0$.
4. Sample $\epsilon_i \sim \mathcal{N}(0, S_{\text{noise}}^2 I)$.
5. Raise the noise level temporarily: $\hat t_i = t_i + \gamma_i t_i$.
6. Add fresh noise to reach that level: $\hat x_i = x_i + \sqrt{\hat t_i^2 - t_i^2}\;\epsilon_i$.
7. Evaluate the slope at the new point: $d_i = \big(\hat x_i - D_\theta(\hat x_i;\hat t_i)\big)/\hat t_i$.
8. Euler step from $\hat t_i$ down to $t_{i+1}$: $x_{i+1} = \hat x_i + (t_{i+1} - \hat t_i)\,d_i$.
9. If $t_{i+1} \ne 0$, apply the Heun correction: $d_i' = \big(x_{i+1} - D_\theta(x_{i+1}; t_{i+1})\big)/t_{i+1}$ and $x_{i+1} = \hat x_i + (t_{i+1} - \hat t_i)\big(\tfrac12 d_i + \tfrac12 d_i'\big)$.
10. Return $x_N$.

With $S_{\text{churn}} = 0$ every $\gamma_i = 0$ and this is Algorithm 1.

**Why each sub-step keeps the right distribution.** Steps 5 and 6 are exact. If $x_i \sim p(x;t_i)$, its noise is $\mathcal{N}(0, t_i^2 I)$, and adding independent noise of variance $\hat t_i^2 - t_i^2$ gives noise of variance $t_i^2 + \hat t_i^2 - t_i^2 = \hat t_i^2$, so $\hat x_i \sim p(x;\hat t_i)$ (with $S_{\text{noise}} = 1$). Steps 7 to 9 are an ODE step, exact up to truncation error. The paper stresses that this is not a general SDE solver, but a procedure whose correctness comes from alternating two sub-steps that each preserve the marginal. Song et al.'s predictor-corrector sampler has a similar structure.

**Why $\gamma_i$ is capped at $\sqrt2 - 1$.** The added noise has standard deviation

$$
\begin{aligned}
\sqrt{\hat t_i^2 - t_i^2} &= \sqrt{(1+\gamma_i)^2 t_i^2 - t_i^2} = t_i\sqrt{(1+\gamma_i)^2 - 1} && (\hat t_i = (1+\gamma_i)t_i).
\end{aligned}
$$

The paper wants to "never introduce more new noise than what is already present", that is, new variance at most $t_i^2$:

$$
\begin{aligned}
t_i^2\big((1+\gamma_i)^2 - 1\big) &\le t_i^2 && \text{(new variance} \le \text{existing variance)}\\
(1+\gamma_i)^2 &\le 2 && \text{(divide by } t_i^2 \text{, add 1)}\\
\gamma_i &\le \sqrt2 - 1 \approx 0.414 && \text{(take the positive root).}
\end{aligned}
$$

At the cap, the injection doubles the noise variance (from $t_i^2$ to $2t_i^2$), and the ODE step that follows brings it down to $t_{i+1}^2$.

**Why $S_{\text{churn}}$ is divided by $N$.** With the cap inactive, the churn summed over the steps inside $[S_{t\min}, S_{t\max}]$ is $S_{\text{churn}}$ times the fraction of steps in that range, so $S_{\text{churn}}$ sets the overall amount of stochasticity roughly independently of the number of steps (this reading is mine; the paper says $S_{\text{churn}}$ "controls the overall amount of stochasticity").

**How it differs from Euler-Maruyama.** An Euler-Maruyama step on Eq. 6 can be read as adding noise and then taking an ODE step whose slope is evaluated at the state before the noise was added, $(x_i, t_i)$. Algorithm 2 evaluates the denoiser after the injection, at $(\hat x_i, \hat t_i)$, so the slope is consistent with the point it is applied to. The two agree as the step size goes to zero, but the paper finds the difference matters with large steps and few evaluations.

**Cost.** Each step still evaluates $D_\theta$ once at $\hat t_i$ and once at $t_{i+1}$ (skipped on the last step), so NFE stays $2N - 1$. For example, 256 steps cost 511 evaluations.

```python
def stochastic_sampler(D, shape, sigmas, rng, S_churn=40, S_tmin=0.05, S_tmax=50, S_noise=1.003):
    N = len(sigmas) - 1
    x = rng.standard_normal(shape) * sigmas[0]
    for t, t_next in zip(sigmas[:-1], sigmas[1:]):
        gamma = min(S_churn / N, 2 ** 0.5 - 1) if S_tmin <= t <= S_tmax else 0.0
        t_hat = t * (1 + gamma)                                    # raise noise level
        x_hat = x + (t_hat ** 2 - t ** 2) ** 0.5 * S_noise * rng.standard_normal(shape)
        d = (x_hat - D(x_hat, t_hat)) / t_hat                      # slope at the new point
        x_next = x_hat + (t_next - t_hat) * d                      # Euler step
        if t_next > 0:
            d2 = (x_next - D(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * 0.5 * (d + d2)     # Heun correction
        x = x_next
    return x
```

The default arguments are the paper's settings for its retrained ImageNet-64 model.

### 4.6 What goes wrong with too much churn

More stochasticity corrects more error, but the paper observed two failure modes with every dataset and network when noise was added and removed too often:

- **Gradual loss of detail.** To isolate this, the authors ran Algorithm 2 at a fixed noise level with $\gamma_i = \sqrt2 - 1$ for thousands of iterations. In theory Langevin iteration should keep the samples at $p(x;\sigma)$. In practice the images degraded. At high noise with $S_{\text{noise}} = 1$, they became more abstract and lost color; on CIFAR-10 after 10,000 iterations they were mostly black and white with no backgrounds.
- **Drift toward oversaturated colors at very low and very high noise levels.** At noise levels below about 0.2, images started to oversaturate after about 2,000 iterations and became fully corrupted later.

The paper suspects the cause is that a trained denoiser defines a slightly non-conservative vector field through $(D_\theta - x)/\sigma^2$: it is not exactly the gradient of any log-density, which breaks the assumption behind Langevin dynamics. With analytical (ideal) denoisers they saw no degradation. Since the flaw is in $D_\theta$, it can only be patched heuristically at sampling time:

- **$S_{t\min}$ and $S_{t\max}$** restrict churn to a middle range of noise levels, which addresses the color drift at the extremes. In particular $S_{t\min} > 0$ turns stochasticity off at very low noise.
- **$S_{\text{noise}}$ slightly above 1** inflates the standard deviation of the added noise (by 0.3% to 0.7% in the paper's settings). This counteracts the loss of detail at high noise, though it does not fix the low-noise oversaturation, which suggests more than one cause.

**Why inflating the noise helps.** That $S_{\text{noise}} > 1$ works suggests the trained denoiser removes slightly too much noise on each cycle. The paper attributes this, "most likely", to regression toward the mean in $L_2$-trained denoisers. One way to picture it (our reading): a real network cannot resolve all the fine detail the ideal posterior mean would keep, so its output is a little smoother than it should be, and $x - D_\theta$, the part it treats as noise, contains some real detail. Every add-and-remove cycle then strips a little detail, and adding slightly more noise than nominal offsets the over-removal. This is a property of trained networks: the exact denoiser of Section 2.5 makes Langevin dynamics exact, and the paper saw no degradation with it.

### 4.7 Results

The four parameters were found per model by grid search:

| Parameter | CIFAR-10 VP | CIFAR-10 VE | ImageNet-64, pretrained | ImageNet-64, EDM-trained | Grid searched |
|---|---|---|---|---|---|
| $S_{\text{churn}}$ | 30 | 80 | 80 | 40 | 0, 10, 20, ..., 100 |
| $S_{t\min}$ | 0.01 | 0.05 | 0.05 | 0.05 | 0, 0.005, 0.01, 0.02, ..., 1, 2, 5, 10 |
| $S_{t\max}$ | 1 | 1 | 50 | 50 | 0.2, 0.5, 1, 2, ..., 10, 20, 50, 80 |
| $S_{\text{noise}}$ | 1.007 | 1.007 | 1.003 | 1.003 | 1.000, 1.001, ..., 1.010 |

**What to read from it.** The optima differ by model. The CIFAR-10 models want churn only in a narrow band of fairly low noise, $[0.01$ or $0.05,\ 1]$, while the ImageNet model wants it over almost the whole range up to 50. The noise inflation is tiny in every case, but the next table shows it matters.

The paper's Figure 4 then compares, at each NFE, the deterministic sampler, Algorithm 2 with some heuristics disabled, the tuned Algorithm 2, and the original stochastic samplers (Euler-Maruyama for VP, predictor-corrector for VE, iDDPM's ancestral sampler for ImageNet-64). The appendix summarizes each curve the same way as in Section 3.6: the lowest FID on the curve, and the smallest NFE whose FID is within 3% of it (so the FID and NFE in one cell need not come from the same run):

| Sampler | VP FID | VP NFE | VE FID | VE NFE | ImageNet-64 FID | ImageNet-64 NFE |
|---|---|---|---|---|---|---|
| Deterministic (Algorithm 1) | 2.93 | 35 | 3.73 | 27 | 2.64 | 79 |
| Algorithm 2, churn at all levels, $S_{\text{noise}} = 1$ | 2.69 | 95 | 2.97 | 383 | 1.86 | 383 |
| Algorithm 2, churn at all levels | 2.54 | 127 | 2.51 | 511 | 1.63 | 767 |
| Algorithm 2, $S_{\text{noise}} = 1$ | 2.52 | 95 | 2.84 | 191 | 1.84 | 255 |
| Algorithm 2, optimal settings | **2.27** | 383 | **2.23** | 767 | **1.55** | 511 |
| Original stochastic samplers | 2.55 | 768 | 2.46 | 1024 | 2.01 | 384 |

**What to read from it.**

- **Stochasticity improves all three pretrained models** over the deterministic sampler, at the cost of more steps: VP from 2.93 to 2.27, VE from 3.73 to 2.23, ImageNet-64 from 2.64 to 1.55.
- **Both heuristics matter.** Removing either the noise-level range or the noise inflation costs a clear amount of FID; which one matters more depends on the model.
- **The tuned sampler beats the original samplers** in every case, with fewer evaluations (for VP, best FID 2.27, already within 3% of it at 383 evaluations, versus 2.55 for the original sampler, which needed 768), and the paper reports the margin is largest at low step counts.
- **The ImageNet-64 headline.** The ADM model had a reported FID of 2.07 with its own sampler; EDM's sampler alone brings it to 1.55, close to the best results then reported (1.48 for cascaded diffusion, 1.55 for classifier-free guidance, 1.52 for StyleGAN-XL).

The paper's caveat is as important as the numbers. The best results required heuristic choices tuned per model by grid search. If stochastic sampling is the main way models are evaluated, that tuning can leak into, and distort, conclusions about architectures and training. The paper also notes it is an open question whether an adaptive SDE solver could beat a well-tuned fixed schedule; the adaptive solver of Jolicoeur-Martineau et al. is their baseline for such methods, and it does worse here.

**What this says.** Stochastic sampling is error correction, not a different model. Its benefit depends on how much error the network makes (Section 5.11 shows it shrinks as the network improves), and too much of it creates new errors, so it is a tuned trade-off.

::: key
*How to say it in an interview:* "EDM writes the reverse SDE as the probability-flow ODE plus a Langevin term, $\beta\sigma^2\nabla\log p\,dt + \sqrt{2\beta}\,\sigma\,d\omega$, which replaces old noise with new noise at rate $\beta$ without changing the noise level. Langevin dynamics pulls samples back toward the correct marginal, so it corrects errors made by earlier steps, but its discretization with an imperfect denoiser adds errors of its own: too much of it loses detail and oversaturates colors. Their sampler does it explicitly: raise the noise from $t_i$ to $(1+\gamma_i)t_i$ by adding fresh noise, then take a Heun step down to $t_{i+1}$, with $\gamma_i = \min(S_{\text{churn}}/N, \sqrt2 - 1)$ inside a noise range and the new noise slightly inflated. On pretrained ImageNet-64 this took FID from 2.07 to 1.55."
:::

::: qa
**Write EDM's SDE and explain each term. Why does adding and removing noise correct errors?**

$dx_\pm = -\dot\sigma\sigma\nabla\log p\,dt \pm \beta\sigma^2\nabla\log p\,dt + \sqrt{2\beta}\,\sigma\,d\omega$. The first term is the probability-flow ODE, which changes the noise level. The other two form Langevin dynamics with a rescaled clock: the decay term $\beta(D - x)\,dt$ removes a fraction $\beta\,dt$ of the noise and the injection adds noise of variance $2\beta\sigma^2\,dt$, so the noise level is unchanged to first order and $p(x;\sigma)$ is stationary under this pair. It can be derived from the Fokker-Planck equation by asking which drift $f$ and noise $g$ reproduce the heat equation $\partial_t p = \dot\sigma\sigma\Delta p$; the answer is $f = (\tfrac12 g^2 - \dot\sigma\sigma)\nabla\log p$, and $g = \sqrt{2\beta}\sigma$ gives the form above. Because Langevin dynamics drives any distribution toward its stationary one, it pulls samples that drifted off $p(x;\sigma_i)$ (from network or truncation errors) back toward it, while the ODE alone would carry those errors forward. Song et al.'s SDE is the special case $\beta = \dot\sigma/\sigma$.
:::

::: qa
**What does $S_{\text{churn}}$ do in EDM's sampler, and why is $\gamma_i$ capped at $\sqrt2 - 1$?**

At each step the sampler raises the noise level from $t_i$ to $\hat t_i = (1+\gamma_i)t_i$ by adding fresh Gaussian noise of standard deviation $\sqrt{\hat t_i^2 - t_i^2}$, then takes a Heun step from $\hat t_i$ down to $t_{i+1}$. $\gamma_i = S_{\text{churn}}/N$ inside $[S_{t\min}, S_{t\max}]$ and 0 outside, so $S_{\text{churn}}$ is the total amount of noise replacement, roughly independent of the step count; $S_{\text{churn}} = 0$ recovers the deterministic sampler. The added variance is $t_i^2((1+\gamma)^2 - 1)$, and the cap keeps it at most $t_i^2$, the noise already present: $(1+\gamma)^2 \le 2$ gives $\gamma \le \sqrt2 - 1$. The noise-adding step is exact, since independent Gaussian variances add, so the sampler alternates two steps that each preserve the marginal.
:::

::: qa
**Why does EDM inflate the injected noise ($S_{\text{noise}}$ slightly above 1) and restrict churn to $[S_{t\min}, S_{t\max}]$?**

Long runs of adding and removing noise degrade images with real denoisers: detail fades and images go gray at high noise, and colors oversaturate at low noise. The paper attributes this to the trained denoiser defining a slightly non-conservative field, so the Langevin assumption fails; ideal denoisers did not show it. Inflating the new noise by 0.3% to 0.7% counters the detail loss, which suggests the trained denoiser removes slightly too much noise each cycle; the paper attributes this to regression toward the mean in $L_2$-trained networks, whose slightly too smooth output makes $x - D_\theta$ include some real detail. Restricting churn to a middle range of noise levels avoids the color drift, which inflation does not fix. Both heuristics measurably help FID, and both had to be tuned per model.
:::

::: {.qa asked="Image-gen vision intern, Nowcoder 2024"}
**Compare the DDPM, DDIM and EDM samplers.**

DDPM ancestral sampling draws each $x_{t-1}$ from the learned Gaussian reverse transition, written with the reparameterization $x_{t-1} = \mu_\theta(x_t, t) + \varsigma_t z$; it is stochastic and needs roughly as many steps as training used, because the Gaussian approximation only holds for small steps. DDIM uses the same $\epsilon$-network but a non-Markovian process with the same marginals; with $\eta = 0$ it is deterministic and is exactly Euler's method on the probability-flow ODE in the coordinates $\sigma(t) = t$, $s = 1$, so it can skip steps and invert images. EDM keeps those coordinates but replaces Euler with Heun (local error $O(h^3)$, $2N - 1$ evaluations), places steps with $\sigma_i^{1/7}$ uniform so they concentrate at low noise, and optionally adds explicit churn: add fresh noise to raise $\sigma$ by a factor $1 + \gamma_i$, then take the Heun step, with the churn limited to a range of noise levels and the noise slightly inflated. With EDM's training recipe as well, deterministic sampling reached FID 1.79 on class-conditional CIFAR-10 at 35 evaluations; stochastic EDM is best when the model is imperfect and steps are plentiful, for example 511 evaluations on ImageNet-64.
:::

::: qa
**How is EDM's stochastic step different from an Euler-Maruyama step on the reverse SDE?**

Euler-Maruyama adds the noise and the drift in one update whose drift is evaluated at the current state $(x_i, t_i)$, so it is like injecting noise and then taking an ODE step with a slope computed before the injection. EDM performs the two sub-steps in sequence: it first adds noise to reach $(\hat x_i, \hat t_i)$, then evaluates $D_\theta(\hat x_i;\hat t_i)$ and takes a Heun step from there. The slope is therefore consistent with the point where it is applied. The two agree as steps shrink, but the paper reports the difference becomes significant at low NFE with large steps, and its sampler beats Euler-Maruyama, predictor-corrector and iDDPM samplers at fewer evaluations.
:::

## 5. Preconditioning and training

Sections 3 and 4 treated the network as a black box. Section 5 of the paper opens the box: how to wrap a raw network so that training is well-conditioned, which noise levels to train on, and how to weight them. The evaluation uses the deterministic sampler of Section 3 throughout, so that sampler tuning cannot mask training differences.

### 5.1 Why not train $D$ directly, and what is wrong with $\epsilon$-prediction

Standard practice in supervised learning is to keep a network's inputs and targets at roughly unit variance and to avoid large differences in gradient magnitude between samples. A network that outputs $D(x;\sigma)$ directly from $x$ violates this badly: the input $x = y + n$ has standard deviation $\sqrt{\sigma_{\text{data}}^2 + \sigma^2}$, which ranges from 0.5 to about 80 over the sampling range. So the network $F_\theta$ is wrapped in scalings.

The earlier models (Table 1) scale the input to unit variance and train $F_\theta$ to predict the unit-variance noise, recovering $D_\theta = x - \sigma F_\theta$. The paper points out the weakness: at large $\sigma$ the network has to output a very precise copy of the noise so that subtracting it cancels the noise in $x$ and leaves an image at the right scale, and whatever error $F_\theta$ makes is multiplied by $\sigma$ (up to 80) in $D_\theta$. At high noise it would be easier to predict the expected clean image directly. Conversely, at low noise, predicting $D$ directly forces the network to copy its input almost exactly, which is wasteful. The paper's answer is a skip connection whose strength depends on $\sigma$, so the network can predict the noise, the signal, or a mix.

### 5.2 The preconditioned denoiser and its effective target

The paper writes

$$
D_\theta(x;\sigma) = c_{\text{skip}}(\sigma)\,x + c_{\text{out}}(\sigma)\,F_\theta\big(c_{\text{in}}(\sigma)\,x;\ c_{\text{noise}}(\sigma)\big),
$$

where $c_{\text{skip}}$ sets how much of the noisy input is passed straight through, $c_{\text{in}}$ and $c_{\text{out}}$ scale the input and output magnitudes, and $c_{\text{noise}}$ maps the noise level to the conditioning value the network sees. The training loss is the denoising loss of Section 2.5, weighted by $\lambda(\sigma)$ and averaged over noise levels drawn from $p_{\text{train}}$:

$$
\mathcal{L} = \mathbb{E}_{\sigma \sim p_{\text{train}},\ y \sim p_{\text{data}},\ n \sim \mathcal{N}(0,\sigma^2 I)}\Big[\lambda(\sigma)\,\lVert D_\theta(y + n;\sigma) - y\rVert_2^2\Big].
$$

Rewrite it in terms of the raw network, to see what $F_\theta$ is actually trained to output:

$$
\begin{aligned}
D_\theta - y &= c_{\text{skip}}(y+n) + c_{\text{out}}F_\theta - y && \text{(substitute the wrapper)}\\
&= c_{\text{out}}F_\theta - \big(y - c_{\text{skip}}(y+n)\big) && \text{(regroup)}\\
&= c_{\text{out}}\Big(F_\theta - \tfrac{1}{c_{\text{out}}}\big(y - c_{\text{skip}}(y+n)\big)\Big) && \text{(factor out } c_{\text{out}})\\
\lambda\,\lVert D_\theta - y\rVert^2 &= \underbrace{\lambda(\sigma)\,c_{\text{out}}(\sigma)^2}_{\text{effective weight } w(\sigma)}\;\Big\lVert F_\theta\big(c_{\text{in}}(y+n); c_{\text{noise}}\big) - \underbrace{\tfrac{1}{c_{\text{out}}(\sigma)}\big(y - c_{\text{skip}}(\sigma)(y+n)\big)}_{\text{effective target } F_{\text{target}}}\Big\rVert^2 && \text{(square).}
\end{aligned}
$$

This is the paper's Eq. 8. Training $D_\theta$ is ordinary supervised regression of $F_\theta$ onto $F_{\text{target}}$ with per-sample weight $w(\sigma)$. Every scaling can now be chosen by asking what we want of an ordinary regression problem.

### 5.3 Deriving $c_{\text{in}}$, $c_{\text{out}}$ and $c_{\text{skip}}$

The paper states three requirements: unit-variance network inputs, unit-variance training targets, and the smallest possible amplification of the network's errors. Throughout, $y$ and $n$ are independent, each pixel of $y$ has variance $\sigma_{\text{data}}^2$ and each pixel of $n$ has variance $\sigma^2$.

**Requirement 1: unit-variance input.**

$$
\begin{aligned}
\mathrm{Var}\big[c_{\text{in}}(y + n)\big] &= 1 && \text{(requirement)}\\
c_{\text{in}}^2\,\big(\sigma_{\text{data}}^2 + \sigma^2\big) &= 1 && \text{(variances of independent terms add)}\\
c_{\text{in}}(\sigma) &= \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}} && \text{(take the positive root).}
\end{aligned}
$$

**Requirement 2: unit-variance target.** This gives $c_{\text{out}}$ as a function of the still-free $c_{\text{skip}}$:

$$
\begin{aligned}
\mathrm{Var}\Big[\tfrac{1}{c_{\text{out}}}\big(y - c_{\text{skip}}(y + n)\big)\Big] &= 1 && \text{(requirement)}\\
c_{\text{out}}^2 &= \mathrm{Var}\big[(1 - c_{\text{skip}})\,y - c_{\text{skip}}\,n\big] && \text{(pull out } 1/c_{\text{out}}^2 \text{, regroup)}\\
c_{\text{out}}^2 &= (1 - c_{\text{skip}})^2\,\sigma_{\text{data}}^2 + c_{\text{skip}}^2\,\sigma^2 && \text{(independent terms).}
\end{aligned}
$$

**Requirement 3: amplify network errors as little as possible.** An error $\delta$ in $F_\theta$ becomes an error $c_{\text{out}}\delta$ in $D_\theta$, so choose $c_{\text{skip}}$ to make $c_{\text{out}}$, equivalently $c_{\text{out}}^2$, as small as possible. $c_{\text{out}}^2$ is a convex quadratic in $c_{\text{skip}}$:

$$
\begin{aligned}
0 &= \frac{d\,c_{\text{out}}^2}{d\,c_{\text{skip}}} = -2(1 - c_{\text{skip}})\,\sigma_{\text{data}}^2 + 2c_{\text{skip}}\,\sigma^2 && \text{(differentiate)}\\
0 &= c_{\text{skip}}\,(\sigma^2 + \sigma_{\text{data}}^2) - \sigma_{\text{data}}^2 && \text{(divide by 2, collect)}\\
c_{\text{skip}}(\sigma) &= \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2} && \text{(solve).}
\end{aligned}
$$

Substitute back into Requirement 2, using $1 - c_{\text{skip}} = \sigma^2/(\sigma^2 + \sigma_{\text{data}}^2)$:

$$
\begin{aligned}
c_{\text{out}}^2 &= \frac{\sigma^4\,\sigma_{\text{data}}^2}{(\sigma^2 + \sigma_{\text{data}}^2)^2} + \frac{\sigma_{\text{data}}^4\,\sigma^2}{(\sigma^2 + \sigma_{\text{data}}^2)^2} && \text{(square both coefficients)}\\
&= \frac{\sigma^2\sigma_{\text{data}}^2\,(\sigma^2 + \sigma_{\text{data}}^2)}{(\sigma^2 + \sigma_{\text{data}}^2)^2} && \text{(factor out } \sigma^2\sigma_{\text{data}}^2)\\
c_{\text{out}}(\sigma) &= \frac{\sigma\,\sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}} && \text{(cancel, take the root).}
\end{aligned}
$$

Because $\sigma^2 + \sigma_{\text{data}}^2$ is at least $\sigma^2$ and at least $\sigma_{\text{data}}^2$, we get $c_{\text{out}} \le \min(\sigma, \sigma_{\text{data}})$: errors are never amplified by more than $\sigma_{\text{data}} = 0.5$, compared with up to 80 for $\epsilon$-prediction.

**What the network ends up predicting.** Substitute the coefficients into $F_{\text{target}}$:

$$
\begin{aligned}
(1 - c_{\text{skip}})\,y - c_{\text{skip}}\,n &= \frac{\sigma^2\,y - \sigma_{\text{data}}^2\,n}{\sigma^2 + \sigma_{\text{data}}^2} && \text{(common denominator)}\\
F_{\text{target}} &= \frac{\sigma^2\,y - \sigma_{\text{data}}^2\,n}{\sigma^2 + \sigma_{\text{data}}^2}\cdot\frac{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}{\sigma\,\sigma_{\text{data}}} = \frac{(\sigma/\sigma_{\text{data}})\,y - (\sigma_{\text{data}}/\sigma)\,n}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}} && \text{(divide by } c_{\text{out}}).
\end{aligned}
$$

For $\sigma \ll \sigma_{\text{data}}$ this is about $-n/\sigma = -\epsilon$: the network predicts the (negated) unit noise and the skip passes the image through, like $\epsilon$-prediction. For $\sigma \gg \sigma_{\text{data}}$ it is about $y/\sigma_{\text{data}}$: the network predicts the clean image and the skip is nearly off, like $x_0$-prediction. At $\sigma = \sigma_{\text{data}}$ the two are mixed equally ($c_{\text{skip}} = \frac12$). With $\sigma_{\text{data}} = 1$ the target is exactly $-v$, the negated velocity of $v$-prediction (Section 7.2). The variance check: $\big(\sigma^2 + \sigma_{\text{data}}^2\big)/\big(\sigma^2 + \sigma_{\text{data}}^2\big) = 1$, since $\mathrm{Var}[(\sigma/\sigma_{\text{data}})y] = \sigma^2$ and $\mathrm{Var}[(\sigma_{\text{data}}/\sigma)n] = \sigma_{\text{data}}^2$.

### 5.4 The loss weight $\lambda(\sigma)$ and the loss at initialization

**Requirement 4: uniform effective weight.** Section 5.2 showed each sample's gradient is scaled by $w(\sigma) = \lambda(\sigma)\,c_{\text{out}}(\sigma)^2$. To balance noise levels, set it to one:

$$
\begin{aligned}
\lambda(\sigma)\,c_{\text{out}}(\sigma)^2 &= 1 && \text{(requirement)}\\
\lambda(\sigma) &= \frac{1}{c_{\text{out}}(\sigma)^2} = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma\,\sigma_{\text{data}})^2} && \text{(invert the result of 5.3).}
\end{aligned}
$$

**The loss at initialization is exactly 1 at every noise level.** The paper zero-initializes the output layer, as earlier work did, so $F_\theta = 0$ and $D_\theta = c_{\text{skip}}\,x$ at the start. Per pixel:

$$
\begin{aligned}
\mathbb{E}\big[\lambda\,(c_{\text{skip}}(y + n) - y)^2\big] &= \lambda\,\mathbb{E}\big[(c_{\text{skip}}\,n - (1 - c_{\text{skip}})\,y)^2\big] && \text{(regroup)}\\
&= \lambda\,\big(c_{\text{skip}}^2\sigma^2 + (1 - c_{\text{skip}})^2\sigma_{\text{data}}^2\big) && (y, n \text{ independent, zero mean)}\\
&= \lambda\,c_{\text{out}}^2 = 1 && \text{(Requirement 2, then Requirement 4).}
\end{aligned}
$$

(The zero-mean step assumes centered data; the paper's derivation uses the variance of $y$ in the same way.) This is the flat green "loss after init" curve in the paper's Figure 5a.

### 5.5 What about $c_{\text{noise}}$?

The paper is explicit: "The formula for $c_{\text{noise}}$ is chosen empirically." It does not follow from the requirements above, and the paper gives no reason for the logarithm or for the factor $\frac14$. Two observations make the choice unsurprising; they are ours, not the paper's. First, the noise levels span more than four orders of magnitude, $[0.002, 80]$, and a logarithm keeps the many small levels from being squeezed together near zero. Second, $p_{\text{train}}$ (Section 5.7) is Gaussian in $\ln\sigma$, so $\ln\sigma$ is the coordinate in which training noise levels are spread evenly. The factor $\frac14$ only keeps the number small: $\frac14\ln 0.002 = -1.55$, $\frac14\ln 80 = 1.10$, and under $p_{\text{train}}$ the value has mean $-0.3$ and standard deviation $0.3$. The value is fed to the network's usual noise embedding (positional or Fourier features). EDM2 keeps the same $c_{\text{noise}}$.

### 5.6 Worked example: the coefficients at a few noise levels

With $\sigma_{\text{data}} = 0.5$ every entry can be checked with a calculator. For instance at $\sigma = 0.5$: $c_{\text{skip}} = 0.25/0.5 = 0.5$, $c_{\text{out}} = 0.25/\sqrt{0.5} = 0.354$, $c_{\text{in}} = 1/\sqrt{0.5} = 1.414$ and $\lambda = 0.5/0.0625 = 8$.

| $\sigma$ | $c_{\text{in}}$ | $c_{\text{skip}}$ | $c_{\text{out}}$ | $\lambda = 1/c_{\text{out}}^2$ | $c_{\text{noise}}$ | $F_\theta$ mostly predicts |
|---|---|---|---|---|---|---|
| 0.002 | 2.000 | 1.000 | 0.002 | 250,000 | -1.55 | noise |
| 0.02 | 1.998 | 0.998 | 0.0200 | 2,504 | -0.98 | noise |
| 0.1 | 1.961 | 0.962 | 0.0981 | 104 | -0.58 | noise |
| 0.5 | 1.414 | 0.500 | 0.354 | 8 | -0.17 | equal mix |
| 1 | 0.894 | 0.200 | 0.447 | 5 | 0 | mostly image |
| 2 | 0.485 | 0.059 | 0.485 | 4.25 | 0.17 | image |
| 10 | 0.0999 | 0.0025 | 0.499 | 4.01 | 0.58 | image |
| 80 | 0.0125 | 0.00004 | 0.500 | 4.00 | 1.10 | image |

**Reading it.** $c_{\text{out}}$ rises from $\sigma$ to a ceiling of $\sigma_{\text{data}} = 0.5$, so the network's error is never amplified much. $c_{\text{in}}$ shrinks the input by up to 80 times so the network always sees unit variance. $\lambda$ is enormous at low noise ($2.5 \times 10^5$ at $\sigma = 0.002$) and flat at 4 at high noise; this is not a choice to emphasize low noise, it only undoes the tiny $c_{\text{out}}^2$ there, so that the gradient on $F_\theta$ has the same scale at every $\sigma$. How much each noise level is emphasized is decided separately by $p_{\text{train}}$.

### 5.7 Where to spend training: the log-normal $p_{\text{train}}$

With $w(\sigma) = 1$, every noise level contributes equally per sample, so the remaining question is how often to sample each level. The paper looks at the loss per noise level after training (Figure 5a, blue for a 32×32 CIFAR-10 model and orange for a 64×64 FFHQ model) and compares it with the initial value of 1. The loss drops substantially only at intermediate noise levels.

- **At very low $\sigma$** the target is essentially the unit noise $-n/\sigma$, and the noise is vanishingly small compared with the image's own fine texture. Separating it is both hard and irrelevant: the loss stays close to its initial value, and whatever the network learns there barely changes the generated images.
- **At very high $\sigma$** the best possible answer is close to the dataset average, so the training target (a specific image) is always far from it. The loss stays high no matter what the network does, because the error is irreducible.

Training effort spent at either extreme is mostly wasted, so the paper samples $\ln\sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$ with $P_{\text{mean}} = -1.2$ and $P_{\text{std}} = 1.2$, placing the mass on the range where the loss can actually be reduced.

**What the numbers mean.** The median training noise level is $e^{-1.2} \approx 0.30$. About 68% of samples fall in $[e^{-2.4}, e^{0}] \approx [0.09, 1.0]$ and about 95% in $[e^{-3.6}, e^{1.2}] \approx [0.027, 3.3]$. Only about 16% have $\sigma > 1$ and about 7% have $\sigma < 0.05$. The sampler still visits levels up to 80, which training rarely sees; there the preconditioning makes the task easy ($c_{\text{skip}} \approx 0$, predict a mean-like image), and the network is evaluated there on only a few steps of the $\rho$ grid.

**Context.** Concurrent work by Choi et al. (P2 weighting) also prioritizes the noise levels where perceptually recognizable content forms, but it changes only $\lambda$; the paper notes that changing $\lambda$ in isolation gives a smaller improvement than redesigning $p_{\text{train}}$ and $\lambda$ together. In EDM's coordinates the split is clean: $\lambda$ normalizes gradient scale, $p_{\text{train}}$ decides emphasis. Later systems shift $P_{\text{mean}}$ upward for higher resolutions or latents (Sections 7.5 and 8.2).

### 5.8 Non-leaking augmentation

Diffusion models on small datasets overfit, so the paper borrows the augmentation pipeline from ADA, the authors' earlier GAN work. Each training image goes through a combined geometric transform before noise is added. There are six augmentations: x-flip, y-flip, isotropic scaling, rotation, anisotropic scaling, and translation. Each is enabled by a weighted coin toss, with probability 12% for CIFAR-10 and 15% for FFHQ and AFHQv2, except x-flip, which is always enabled. The eight random parameters (zero for disabled augmentations) are converted into a 9-dimensional vector, for example rotation by angle $a$ contributes $\cos a - 1$ and $\sin a$, designed so that the vector is zero when no augmentation is applied.

**Why it does not leak.** The vector is given to the network as an extra conditioning input, next to the noise level and class label. The network therefore learns a whole family of tasks, "denoise images that were transformed by $a$", and the true task is the member $a = 0$. At sampling time the conditioning is set to zero, so the model is asked for untransformed images. The auxiliary tasks provide many distinct training samples and prevent memorizing individual images, and they seem to help the main task because denoising is similar for every $a$. The paper observed no leakage, no rotated or flipped artifacts in outputs, even with augmentation probability at 100%. A side benefit: the x-flip augmentation is enabled for every image (flipping half of them, with the flip recorded in the conditioning) and the usual dataset x-flips are turned off, so the model gets the benefit of flips without being taught that mirrored text or logos are normal. Color and image-space filtering augmentations were consistently harmful for diffusion models, so only geometric ones are used.

### 5.9 Other training changes: configs B and C

Two more changes come before the preconditioning in the ablation. They are not part of "EDM" as a formulation, but they are part of the reported numbers.

**Config B, adjusted hyperparameters**, for faster training and a fairer baseline: 8 GPUs instead of 4; batch size 512 at 32×32 and 256 at 64×64 instead of 128; gradient clipping removed (it gave no benefit); for CIFAR-10 the learning rate raised from $2 \times 10^{-4}$ to $10^{-3}$, with a 10-million-image ramp-up; the EMA half-life standardized to 0.5 million images; and dropout tuned per dataset by grid search at 1% increments (13% for CIFAR-10, 5% for FFHQ, 25% for AFHQv2, against 10% in the baseline). Every model trains on 200 million images, about 400,000 iterations at batch 512, taking roughly 2 days at 32×32 and 4 days at 64×64 on 8 V100 GPUs.

**Config C, redistributed capacity**: remove the 4×4 resolution layers, which mainly contributed to overfitting, and double the capacity of a higher-resolution level, which the paper found critical for quality. The channel counts change from 128 at 64×64 and 32×32 and 256 at 16, 8 and 4, to 128 at 64×64 and 256 at 32, 16 and 8; that is, the 32×32 layers go from 128 to 256 channels. (The appendix text names the 16×16 layers, but its channel counts show the doubling at 32×32.) This slightly reduces the parameter count, to about 56M at 32×32 and 62M at 64×64. The 64×64 resolution exists only in the FFHQ and AFHQv2 models.

The architectures themselves are Song et al.'s DDPM++ (for VP) and NCSN++ (for VE), which differ in three ways: box versus bilinear resampling filters, positional versus random Fourier noise embeddings, and NCSN++'s extra residual connections from the input image into the encoder. Class labels enter as one-hot vectors scaled by $\sqrt{C}$ for $C$ classes, through a fully-connected layer, added to the noise embedding; the augmentation vector enters the same way.

### 5.10 The training ablation, configs A to F

Table 2 of the paper adds one change at a time. Config A is Song et al.'s original training; all rows are evaluated with EDM's deterministic sampler (35 NFE at 32×32, 79 at 64×64). VP and VE refer to the original setups; by config E they differ only in architecture (DDPM++ versus NCSN++).

| Training configuration | CIFAR-10 cond. VP | cond. VE | CIFAR-10 uncond. VP | uncond. VE | FFHQ-64 VP | FFHQ-64 VE | AFHQv2-64 VP | AFHQv2-64 VE |
|---|---|---|---|---|---|---|---|---|
| A: baseline (Song et al.) | 2.48 | 3.11 | 3.01 | 3.77 | 3.39 | 25.95 | 2.58 | 18.52 |
| B: + adjusted hyperparameters | 2.18 | 2.48 | 2.51 | 2.94 | 3.13 | 22.53 | 2.43 | 23.12 |
| C: + redistributed capacity | 2.08 | 2.52 | 2.31 | 2.83 | 2.78 | 41.62 | 2.54 | 15.04 |
| D: + EDM preconditioning | 2.09 | 2.64 | 2.29 | 3.10 | 2.94 | 3.39 | 2.79 | 3.81 |
| E: + EDM loss ($p_{\text{train}}$, $\lambda$) | 1.88 | 1.86 | 2.05 | 1.99 | 2.60 | 2.81 | 2.29 | 2.28 |
| F: + non-leaky augmentation | **1.79** | **1.79** | **1.97** | **1.98** | **2.39** | **2.53** | **1.96** | **2.16** |
| NFE | 35 | 35 | 35 | 35 | 79 | 79 | 79 | 79 |

The unconditional CIFAR-10 config A entries are the original pretrained checkpoints.

**What to read from it.**

- **Configs B and C** are routine engineering and give steady gains for VP (for example conditional CIFAR-10 2.48 to 2.08). VE at 64×64 is unstable throughout A to C (FFHQ 25.95, 22.53, 41.62).
- **Config D (preconditioning) leaves FID roughly unchanged, except that VE at 64×64 is rescued** (FFHQ 41.62 to 3.39, AFHQv2 15.04 to 3.81). The paper's interpretation: the preconditioning's main benefit is not FID by itself but robustness, which makes it safe to redesign the loss next.
- **Config E (the loss) improves every column**, often by a lot: for example unconditional CIFAR-10 VE 3.10 to 1.99, conditional VE 2.64 to 1.86, AFHQv2 VP 2.79 to 2.29. It is the largest single step for the VE CIFAR-10 models and AFHQv2 VP, but not everywhere: for VP on CIFAR-10 the hyperparameter retuning of config B gained more (unconditional 3.01 to 2.51). The paper stresses that E works this well only together with D.
- **Config F (augmentation)** gives a consistent further gain and produced the records: 1.79 for conditional CIFAR-10 and 1.97 for unconditional, against previous records of 1.85 (StyleGAN-XL) and 2.10 (LSGM).
- **After E, VP and VE end up close** (within about 0.2 FID in every dataset at config F), although at config A they differed a lot. By then the two setups differ only in architecture, which suggests the training choices bundled with each family mattered more than the VP/VE label itself.

### 5.11 Stochastic sampling revisited, and retraining ImageNet-64

**Better networks need less stochasticity.** The paper's Figure 5b sweeps $S_{\text{churn}}$ for unconditional CIFAR-10 at 256 steps (511 NFE). For the original Song et al. models, stochastic sampling helps a lot and deterministic sampling is relatively poor. For the EDM-trained models the picture flips: deterministic sampling is best and any amount of churn hurts. This fits Section 4.4: churn corrects network errors, and a better network has fewer errors to correct, while churn's own errors remain.

**ImageNet-64 from scratch.** As a final test, the authors trained the ADM architecture, unchanged (about 296M parameters, same dropout as ADM), with config E. They did not use augmentation, since overfitting was not a concern at this dataset size. The run used 32 NVIDIA Ampere GPUs, batch size 4096, about 2.5 billion training images (about 600,000 iterations, two weeks), learning rate $10^{-4}$, an EMA half-life of 50 million images, and mixed precision (FP16 except in the embedding and self-attention layers, where FP16's range caused occasional instability). Figure 5c sweeps $S_{\text{churn}}$ at 256 steps:

| ImageNet-64, 511 NFE | Deterministic ($S_{\text{churn}} = 0$) | Best stochastic |
|---|---|---|
| Pretrained ADM, EDM sampler | 2.66 | 1.57 (at $S_{\text{churn}} = 75$) |
| ADM retrained with EDM's config E | 2.22 | **1.36** (at $S_{\text{churn}} = 40$) |

(The figure's 1.57 for the pretrained model and the 1.55 of Section 4.7 are both at 511 evaluations with the grid-searched settings; they come from different sweeps, one over $S_{\text{churn}}$ and one over NFE at $S_{\text{churn}} = 80$, and differ by under 2%, within FID's run-to-run variation of about ±2%.)

**What to read from it.** The retrained model sets a new record of 1.36 against the previous 1.48 of cascaded diffusion. Its best churn is lower than the pretrained model's (40 versus 75), but unlike CIFAR-10, stochastic sampling still clearly beats deterministic sampling. The paper's reading is that more diverse datasets keep benefiting from stochasticity. The training improvements help both samplers.

**What this says.** The training recipe has two halves with separate jobs. The preconditioning ($c_{\text{in}}, c_{\text{skip}}, c_{\text{out}}$, $\lambda = 1/c_{\text{out}}^2$) makes every noise level an equally well-scaled regression problem; it barely changes FID alone but makes training robust. The noise distribution $p_{\text{train}}$ then decides where training effort goes, and putting it where the loss can actually be reduced is the one training change that improved FID in every column of the ablation.

::: key
*How to say it in an interview:* "EDM wraps the network as $D = c_{\text{skip}}x + c_{\text{out}}F(c_{\text{in}}x; c_{\text{noise}})$ and derives the scalings from three requirements: unit-variance input gives $c_{\text{in}} = 1/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$; a unit-variance effective target gives $c_{\text{out}}^2 = (1 - c_{\text{skip}})^2\sigma_{\text{data}}^2 + c_{\text{skip}}^2\sigma^2$; and minimizing $c_{\text{out}}$, so network errors are amplified least, gives $c_{\text{skip}} = \sigma_{\text{data}}^2/(\sigma^2 + \sigma_{\text{data}}^2)$. The loss weight $1/c_{\text{out}}^2$ makes the effective weight one everywhere, and noise levels are drawn log-normally around $\sigma \approx 0.3$ because that is where the loss can actually go down. The network smoothly switches from predicting noise at low $\sigma$ to predicting the image at high $\sigma$, and its errors are never scaled by more than $\sigma_{\text{data}}$."
:::

::: pitfall
"$c_{\text{noise}} = \frac14\ln\sigma$ comes out of the derivation." It does not; the paper says it was chosen empirically. Only $c_{\text{in}}$, $c_{\text{out}}$, $c_{\text{skip}}$ and $\lambda$ follow from the stated requirements.
:::

::: qa
**Derive EDM's $c_{\text{skip}}$, $c_{\text{out}}$ and $c_{\text{in}}$.**

Write $D = c_{\text{skip}}x + c_{\text{out}}F(c_{\text{in}}x)$ with $x = y + n$, data variance $\sigma_{\text{data}}^2$ and noise variance $\sigma^2$. Then $\lambda\lVert D - y\rVert^2 = \lambda c_{\text{out}}^2\lVert F - F_{\text{target}}\rVert^2$ with $F_{\text{target}} = (y - c_{\text{skip}}x)/c_{\text{out}}$. Unit-variance input: $c_{\text{in}}^2(\sigma^2 + \sigma_{\text{data}}^2) = 1$. Unit-variance target: $c_{\text{out}}^2 = \mathrm{Var}[(1 - c_{\text{skip}})y - c_{\text{skip}}n] = (1 - c_{\text{skip}})^2\sigma_{\text{data}}^2 + c_{\text{skip}}^2\sigma^2$. Since errors in $F$ are multiplied by $c_{\text{out}}$, minimize this convex quadratic: $-2(1 - c_{\text{skip}})\sigma_{\text{data}}^2 + 2c_{\text{skip}}\sigma^2 = 0$ gives $c_{\text{skip}} = \sigma_{\text{data}}^2/(\sigma^2 + \sigma_{\text{data}}^2)$, and substituting back gives $c_{\text{out}} = \sigma\sigma_{\text{data}}/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$. At low noise $c_{\text{skip}} \to 1$ and $F$ predicts noise; at high noise $c_{\text{skip}} \to 0$ and $F$ predicts the image; $c_{\text{out}}$ never exceeds $\sigma_{\text{data}}$.
:::

::: qa
**Why is $\epsilon$-prediction poorly conditioned at high noise, and what did EDM's preconditioning change in practice?**

With $\epsilon$-prediction, $D = x - \sigma F$, so the network must output a precise copy of the noise for the subtraction to cancel it, and any error in $F$ is multiplied by $\sigma$ (up to 80) in the clean-image estimate. At high noise the easier task is to predict the expected clean image directly, which is what EDM's $c_{\text{skip}} \to 0$ allows; its $c_{\text{out}}$ is bounded by $\sigma_{\text{data}} = 0.5$. In the ablation, switching to EDM's preconditioning (config D) left FID roughly unchanged for most settings but fixed the unstable VE models at 64×64 (FFHQ 41.62 to 3.39). The paper's point is that preconditioning mainly buys robustness, which made the following change to the loss and noise distribution (config E) work: that step improved FID in every column.
:::

::: qa
**Why does EDM set $\lambda(\sigma) = 1/c_{\text{out}}(\sigma)^2$, and what is the training loss at initialization?**

Rewriting the denoising loss in terms of the raw network gives $\lambda c_{\text{out}}^2\lVert F - F_{\text{target}}\rVert^2$, so each sample's gradient on $F$ is scaled by $w = \lambda c_{\text{out}}^2$. Setting $\lambda = 1/c_{\text{out}}^2 = (\sigma^2 + \sigma_{\text{data}}^2)/(\sigma\sigma_{\text{data}})^2$ makes this effective weight one at every noise level, so no level dominates the gradient through scale alone. With a zero-initialized output layer, $F = 0$ and $D = c_{\text{skip}}x$, and the expected per-pixel loss is $\lambda(c_{\text{skip}}^2\sigma^2 + (1 - c_{\text{skip}})^2\sigma_{\text{data}}^2) = \lambda c_{\text{out}}^2 = 1$ at every $\sigma$. The large $\lambda$ at small $\sigma$ (250,000 at 0.002) is not an emphasis on low noise; it only cancels the small $c_{\text{out}}$ there. Emphasis is set by $p_{\text{train}}$.
:::

::: qa
**Why does EDM sample training noise levels from a log-normal distribution?**

After balancing the effective weight, the paper measured the loss per noise level after training and found it drops well below its initial value only at intermediate $\sigma$. At very low noise the noise is tiny relative to image texture, hard to separate and irrelevant to the output; at very high noise the best prediction is close to the dataset mean, so the error is irreducible. Training samples at the extremes are therefore mostly wasted. A log-normal, $\ln\sigma \sim \mathcal{N}(-1.2, 1.2^2)$, puts the median at $\sigma \approx 0.3$ and 95% of samples in about $[0.027, 3.3]$, matching the range where learning happens. Together with the preconditioning, this change (config E) improved FID in every column of the ablation, by up to 1.5. Later work moves $P_{\text{mean}}$ up for latents and high resolution, for example $-0.4$ in EDM2 and $0.5$ to $1.0$ in Stable Video Diffusion's high-resolution video stages.
:::

::: qa
**How does EDM's augmentation avoid leaking into generated images?**

Each training image is geometrically transformed (flips, scaling, rotation, anisotropy, translation) before noise is added, and the transform parameters are encoded as a 9-dimensional vector that is zero when nothing is applied. That vector is an extra conditioning input to the network, so the network learns to denoise images "given that they were transformed by $a$", a family of auxiliary tasks with the real task at $a = 0$. At sampling time the vector is set to zero, so the model is asked for untransformed images, and the paper saw no transformed artifacts even with 100% augmentation probability. The auxiliary tasks act as a regularizer against memorizing individual images. It is the same idea as ADA for GANs, transferred to diffusion; only geometric augmentations helped, and it improved every CIFAR-10, FFHQ and AFHQv2 result (config F).
:::

::: qa
**When does stochastic sampling help, according to EDM?**

It helps when the network makes errors that the Langevin component can correct, and when you can afford many steps. On the three pretrained models, the tuned stochastic sampler beat the deterministic one (for example 2.64 to 1.55 on ImageNet-64), but needed hundreds of evaluations. On CIFAR-10 models trained with EDM's recipe, deterministic sampling was best and any churn hurt. On ImageNet-64 retrained with the recipe, stochastic sampling still won clearly (2.22 deterministic versus 1.36 at $S_{\text{churn}} = 40$), with less churn than the pretrained model needed; the paper reads this as more diverse datasets continuing to benefit. Too much churn degrades detail and colors, so its amount and noise range must be tuned per model, and EDM2 later used only deterministic sampling at 63 evaluations.
:::

## 6. Final results

| Dataset | Model and sampler | FID | NFE | Previous record cited by the paper |
|---|---|---|---|---|
| CIFAR-10 32×32, class-conditional | config F, deterministic (VP / VE) | 1.79 / 1.79 | 35 | 1.85 (StyleGAN-XL) |
| CIFAR-10 32×32, unconditional | config F, deterministic (VP / VE) | 1.97 / 1.98 | 35 | 2.10 (LSGM) |
| FFHQ 64×64, unconditional | config F, deterministic (VP / VE) | 2.39 / 2.53 | 79 | none given |
| AFHQv2 64×64, unconditional | config F, deterministic (VP / VE) | 1.96 / 2.16 | 79 | none given |
| ImageNet 64×64, class-conditional | pretrained ADM, EDM stochastic sampler | 1.55 | 511 | ADM's own result 2.07 |
| ImageNet 64×64, class-conditional | ADM retrained with config E, stochastic | 1.36 | 511 | 1.48 (cascaded diffusion) |

**What to read from it.** The CIFAR-10 records use a small network (about 56M parameters), deterministic sampling and only 35 evaluations; the original samplers of the pretrained CIFAR-10 models in Section 3 needed 256 (VP) to 8192 (VE). The ImageNet-64 rows split the credit: the sampler alone takes a pretrained model from 2.07 to 1.55, and retraining with the recipe takes it to 1.36. All FIDs are the minimum of three evaluations of 50,000 samples against the full training set. The paper reports that the project used about 250 MWh on an in-house V100 cluster. Its own outlook was that higher resolutions would probably need re-tuned parameters, and that the interaction between stochastic sampling and the training objective was still open; EDM2 (Section 8) is the authors' follow-up.

## 7. Connections and practice

### 7.1 Wrapping a pretrained $\epsilon$-prediction model as an EDM denoiser

Suppose you have a DDPM-style model $\epsilon_\theta(x_t, t)$, trained on $x_t = \sqrt{\bar\alpha_t}\,y + \sqrt{1 - \bar\alpha_t}\,\epsilon$ to predict $\epsilon$. To use EDM's samplers you need $D_\theta(x;\sigma)$ on the EDM variable $x = y + n$.

$$
\begin{aligned}
x &= x_t/\sqrt{\bar\alpha_t} = y + \sigma\epsilon, \quad \sigma = \sqrt{(1 - \bar\alpha_t)/\bar\alpha_t} && \text{(notation section)}\\
\sqrt{\bar\alpha_t} &= 1/\sqrt{1 + \sigma^2} && (\bar\alpha_t = 1/(1+\sigma^2))\\
x_t &= \frac{x}{\sqrt{1 + \sigma^2}} && \text{(the network's input, rewritten)}\\
\mathbb{E}[y \mid x] &= \mathbb{E}[x - \sigma\epsilon \mid x] = x - \sigma\,\mathbb{E}[\epsilon \mid x] && (y = x - \sigma\epsilon)\\
D_\theta(x;\sigma) &= x - \sigma\,\epsilon_\theta\Big(\frac{x}{\sqrt{\sigma^2 + 1}},\ t(\sigma)\Big) && (\epsilon_\theta \approx \mathbb{E}[\epsilon \mid x_t]).
\end{aligned}
$$

In Table 1 terms: $c_{\text{skip}} = 1$, $c_{\text{out}} = -\sigma$, $c_{\text{in}} = 1/\sqrt{\sigma^2 + 1}$, and $c_{\text{noise}}$ is the timestep $t(\sigma)$ whose noise level matches $\sigma$, found by inverting $\sigma_t = \sqrt{(1 - \bar\alpha_t)/\bar\alpha_t}$. This is exactly the VP and iDDPM column, and the Stable Video Diffusion paper writes Stable Diffusion 2.1 in the same form.

**Practical details**, all from how the paper handled ADM:

- **Noise range.** Use $\sigma_{\max} = \min(\sigma_{\text{hi}}, 80)$ and $\sigma_{\min} = \max(\sigma_{\text{lo}}, 0.002)$ for the model's trained range. A schedule that ends at $\sqrt{\bar\alpha_T} \approx 0.068$ (the Stable Diffusion value quoted on the diffusion page) has $\sigma_{\text{hi}} = \sqrt{1/0.068^2 - 1} \approx 14.7$, so sampling must start there, not at 80.
- **Starting noise.** Draw $x_0 = \sigma_{\max}\,z$ in EDM units. In the model's own units this is $x_T = x_0/\sqrt{1 + \sigma_{\max}^2}$.
- **Discrete timesteps.** A model trained on 1000 discrete levels can only be evaluated there: round each $\sigma_i$ of the $\rho$ grid to the nearest trained level, and in the stochastic sampler round $\hat t_i$ the same way. Heun's second evaluation lands exactly on the next grid point, so rounding the grid suffices.
- **Loss correspondence.** The model was trained with $\lambda = 1/\sigma^2$ on its own timestep distribution (Section 2.7), which you inherit; only the sampler changes.

### 7.2 $v$-prediction models are EDM with $\sigma_{\text{data}} = 1$

$v$-prediction (Salimans and Ho) trains the network to output $v = \alpha_t\epsilon - \sigma_t y$, with $\alpha_t = \sqrt{\bar\alpha_t}$ and $\sigma_t = \sqrt{1 - \bar\alpha_t}$; the clean image is $\hat y = \alpha_t x_t - \sigma_t\hat v$. Convert to EDM variables:

$$
\begin{aligned}
\alpha_t &= \frac{1}{\sqrt{1 + \sigma^2}}, \quad \sigma_t = \frac{\sigma}{\sqrt{1 + \sigma^2}}, \quad x_t = \frac{x}{\sqrt{1 + \sigma^2}} && \text{(from the notation section)}\\
D_\theta(x;\sigma) &= \frac{1}{1 + \sigma^2}\,x - \frac{\sigma}{\sqrt{1 + \sigma^2}}\,\hat v\Big(\frac{x}{\sqrt{1 + \sigma^2}}\Big) && \text{(substitute into } \alpha_t x_t - \sigma_t\hat v).
\end{aligned}
$$

So, with the network output $F = \hat v$: $c_{\text{skip}} = 1/(1 + \sigma^2)$, $c_{\text{out}} = -\sigma/\sqrt{1 + \sigma^2}$, $c_{\text{in}} = 1/\sqrt{1 + \sigma^2}$. These are EDM's formulas at $\sigma_{\text{data}} = 1$ except for the sign of $c_{\text{out}}$, which only records that a $v$-network outputs $v$ rather than $-v$. This matches Section 5.3, where EDM's effective target at $\sigma_{\text{data}} = 1$ was $-v$. Stable Video Diffusion uses exactly these functions with $c_{\text{noise}} = \frac14\ln\sigma$, noting they are EDM's with $\sigma_{\text{data}} = 1$.

### 7.3 Rectified flow in the $(s, \sigma)$ framework

Rectified flow and the flow matching used by SD3 interpolate linearly between data and noise. In SD3's convention ($t = 0$ data, $t = 1$ noise; see [flow matching](flow-matching.html)), $x_t = (1 - t)\,y + t\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$.

**Find $s(t)$ and $\sigma(t)$.**

$$
\begin{aligned}
x_t &= (1 - t)\,y + t\,\epsilon && \text{(the interpolation)}\\
&= (1 - t)\Big(y + \frac{t}{1 - t}\,\epsilon\Big) && \text{(factor out } 1 - t \text{, for } t < 1)\\
s(t) &= 1 - t, \qquad \sigma(t) = \frac{t}{1 - t} && \text{(match } s\,(y + \sigma\epsilon))\\
t &= \frac{\sigma}{1 + \sigma}, \qquad 1 - t = \frac{1}{1 + \sigma} && \text{(invert).}
\end{aligned}
$$

So rectified flow is the scaled family with $s = 1/(1 + \sigma)$. As $t$ goes from 0 to 1, $\sigma$ goes from 0 to infinity, so $t = 1$ corresponds to EDM's $\sigma = \infty$ (pure noise, zero signal).

**Its ODE is EDM's general ODE.** Plug into the result of Section 2.6:

$$
\begin{aligned}
\dot\sigma &= \frac{(1 - t) + t}{(1 - t)^2} = \frac{1}{(1 - t)^2} && \text{(quotient rule)}\\
\frac{\dot\sigma}{\sigma} &= \frac{1}{(1 - t)^2}\cdot\frac{1 - t}{t} = \frac{1}{t(1 - t)}, \qquad \frac{\dot s}{s} = \frac{-1}{1 - t} && \text{(divide)}\\
\frac{\dot\sigma}{\sigma} + \frac{\dot s}{s} &= \frac{1 - t}{t(1 - t)} = \frac1t && \text{(common denominator)}\\
\frac{\dot\sigma\,s}{\sigma} &= \frac{1}{t(1 - t)}\,(1 - t) = \frac1t && \text{(multiply by } s)\\
\frac{dx}{dt} &= \frac{x - D\big(x/(1 - t);\ t/(1 - t)\big)}{t} && \text{(substitute into Section 2.6).}
\end{aligned}
$$

Check against the velocity a flow-matching network learns, $v = \mathbb{E}[\epsilon - y \mid x_t]$. From $\epsilon = (x_t - (1 - t)y)/t$, $\epsilon - y = (x_t - y)/t$, whose posterior mean is $(x_t - D)/t$, the same field. In both formulations an Euler step to the data end returns the denoiser output: here $x_t - t\,(x_t - D)/t = D$. The difference from EDM's preferred ODE is only the scaling $s = 1/(1 + \sigma)$ and the time variable; the trajectories are the EDM trajectories $x(\sigma)$ multiplied by $1/(1 + \sigma)$, which changes their curvature but not their endpoints.

**The implied loss weight.** For a training sample, the velocity target is $\epsilon - y = (x_t - y)/t$ and the network's implied velocity is $(x_t - \hat D)/t$. Their difference is $(y - \hat D)/t$, so

$$
\lVert \hat v - (\epsilon - y)\rVert^2 = \frac{1}{t^2}\,\lVert \hat D - y\rVert^2 = \frac{(1 + \sigma)^2}{\sigma^2}\,\lVert \hat D - y\rVert^2 .
$$

EDM's weight with $\sigma_{\text{data}} = 1$ is $(1 + \sigma^2)/\sigma^2$. Divide the two weights: the ratio is $(1 + \sigma)^2/(1 + \sigma^2) = 1 + 2\sigma/(1 + \sigma^2)$. It is at least 1, and since $2\sigma \le 1 + \sigma^2$ (equivalent to $(1 - \sigma)^2 \ge 0$) it is at most 2. So the flow-matching weight is between 1 and 2 times EDM's at every noise level, approaching 1 at both extremes and reaching 2 at $\sigma = 1$. In EDM's language, plain flow matching uses a loss weight close to EDM's (for data with unit variance), so when the two are compared, the noise-level distribution, the network and the data scale are the more likely sources of a difference.

### 7.4 EDM's log-normal and SD3's logit-normal

Uniform $t$ in rectified flow corresponds to the density $p(\sigma) = dt/d\sigma = 1/(1 + \sigma)^2$, which puts half its mass below $\sigma = 1$ and has a heavy tail. SD3 instead samples $t$ from a logit-normal: draw $u \sim \mathcal{N}(m, s^2)$ and set $t$ to the logistic function of $u$, with density $\pi_{\ln}(t; m, s) = \frac{1}{s\sqrt{2\pi}}\,\frac{1}{t(1 - t)}\exp\big(-\frac{(\mathrm{logit}(t) - m)^2}{2s^2}\big)$. (Here $s$ is SD3's scale parameter, not the scaling function.) Convert it to EDM's noise level:

$$
\begin{aligned}
t &= \frac{1}{1 + e^{-u}} && \text{(logistic function)}\\
1 - t &= \frac{e^{-u}}{1 + e^{-u}} && \text{(subtract from 1)}\\
\sigma = \frac{t}{1 - t} &= \frac{1}{e^{-u}} = e^{u} && \text{(divide; Section 7.3's map)}\\
\ln\sigma &= u \sim \mathcal{N}(m, s^2) && \text{(take logs).}
\end{aligned}
$$

So SD3's logit-normal over $t$ is exactly a log-normal over the equivalent EDM noise level, with $P_{\text{mean}} = m$ and $P_{\text{std}} = s$. (The density formula is the same fact: $\mathrm{logit}(t) = \ln\sigma$, and $1/(t(1 - t))$ is the Jacobian $d\,\mathrm{logit}(t)/dt$.) SD3's preferred setting, $(m, s) = (0, 1)$, is a log-normal centered at $\sigma = 1$ instead of EDM's $0.3$, with a narrower spread. The values are not directly comparable, because SD3 works on autoencoder latents with their own scale and EDM's 0.3 was tuned for pixels with $\sigma_{\text{data}} = 0.5$; both later EDM2 and Stable Video Diffusion also moved $P_{\text{mean}}$ upward for latents and high resolution. SD3 also trained EDM itself as a baseline: with 15 settings of $(P_{\text{mean}}, P_{\text{std}})$, including EDM's $(-1.2, 1.2)$, and with EDM schedules matched to rectified flow and cosine weighting. In its ranking, rectified flow with logit-normal $(0, 1)$ sampling consistently placed near the top and beat the EDM and LDM-linear formulations.

### 7.5 Later systems that adopted EDM's formulation

Only systems whose papers state the adoption are listed.

- **Consistency models** (Song et al., 2023) adopt EDM's settings throughout: $\sigma(t) = t$, $T = 80$, $\epsilon = 0.002$, images in $[-1, 1]$, the $\rho = 7$ grid, and the Heun solver, which together with $N = 18$ worked best for consistency distillation, "in line with the recommendation" of EDM. They distill from EDM models and use EDM's preconditioning with $\sigma_{\text{data}} = 0.5$, shifted to $c_{\text{skip}}(t) = \sigma_{\text{data}}^2/((t - \epsilon)^2 + \sigma_{\text{data}}^2)$ and $c_{\text{out}}(t) = \sigma_{\text{data}}(t - \epsilon)/\sqrt{\sigma_{\text{data}}^2 + t^2}$ so that $c_{\text{skip}}(\epsilon) = 1$ and $c_{\text{out}}(\epsilon) = 0$ (details on the [flow matching](flow-matching.html) page).
- **Stable Video Diffusion** (Blattmann et al., 2023) fine-tunes Stable Diffusion 2.1 into EDM's continuous-noise framework: preconditioning with $\sigma_{\text{data}} = 1$ (Section 7.2), $\lambda(\sigma) = (1 + \sigma^2)\sigma^{-2}$, and log-normal noise with $P_{\text{mean}} = -1.2$, $P_{\text{std}} = 1$. It shifts the noise distribution toward more noise as resolution grows: $P_{\text{mean}} = 0$ for 320×576 video pretraining, then $P_{\text{mean}} = 0.5$, $P_{\text{std}} = 1.4$ for 576×1024 text-to-video fine-tuning, and up to $P_{\text{mean}} = 1.0$, $P_{\text{std}} = 1.6$ for 576×1024 image-to-video, where the conditioning frame is a strong signal. See [video generation](video-generation.html).
- **Playground v2.5** (Li et al., 2024) trains from scratch "using the EDM framework". It cites two reasons: EDM's schedule reaches near-zero SNR at the noisiest level (with EDM's pixel-space defaults $\sigma_{\max} = 80$ and $\sigma_{\text{data}} = 0.5$, the signal-to-noise ratio there is $0.25/6400 \approx 4 \times 10^{-5}$), which removes the need for offset noise and fixes muted colors; and its first-principles design of training, sampling and preconditioning. It also skews the noise schedule toward more noise at high resolution.
- **EDM2** (Section 8) keeps EDM's formulation and changes the network and training dynamics.

::: key
*How to say it in an interview:* "Any Gaussian diffusion or flow model is a choice of $s(t)$ and $\sigma(t)$ plus a parameterization. A DDPM $\epsilon$-model becomes an EDM denoiser as $D(x;\sigma) = x - \sigma\,\epsilon_\theta(x/\sqrt{1+\sigma^2}, t(\sigma))$ with $\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha}$. A $v$-model is EDM's preconditioning with $\sigma_{\text{data}} = 1$. Rectified flow is $s = 1 - t$, $\sigma = t/(1-t)$, and its ODE reduces to $dx/dt = (x - D)/t$; SD3's logit-normal timestep sampling is exactly a log-normal over that $\sigma$, since $\mathrm{logit}(t) = \ln\sigma$."
:::

::: qa
**Convert a pretrained $\epsilon$-prediction VP model into an EDM denoiser $D(x;\sigma)$.**

Use the EDM variable $x = x_t/\sqrt{\bar\alpha_t} = y + \sigma\epsilon$ with $\sigma = \sqrt{(1 - \bar\alpha_t)/\bar\alpha_t}$, so $\bar\alpha_t = 1/(1 + \sigma^2)$ and the network input is $x_t = x/\sqrt{1 + \sigma^2}$. Since $y = x - \sigma\epsilon$, the posterior mean is $D = x - \sigma\,\mathbb{E}[\epsilon \mid x]$, giving $D_\theta(x;\sigma) = x - \sigma\,\epsilon_\theta(x/\sqrt{\sigma^2 + 1}, t(\sigma))$, which is $c_{\text{skip}} = 1$, $c_{\text{out}} = -\sigma$, $c_{\text{in}} = 1/\sqrt{\sigma^2 + 1}$ and $c_{\text{noise}} = t(\sigma)$. In practice clip the sampling range to the model's trained $\sigma$ range (a schedule ending at $\sqrt{\bar\alpha_T} = 0.068$ only reaches $\sigma \approx 14.7$), start from $\mathcal{N}(0, \sigma_{\max}^2 I)$, and for discrete-time models round each grid level to the nearest trained one. The original noise MSE corresponds to EDM weight $\lambda = 1/\sigma^2$.
:::

::: qa
**Where does rectified flow sit in EDM's $(s, \sigma)$ framework?**

With $x_t = (1 - t)y + t\epsilon$, factor out $1 - t$: $x_t = (1 - t)(y + \frac{t}{1 - t}\epsilon)$, so $s(t) = 1 - t$ and $\sigma(t) = t/(1 - t)$, with $t = \sigma/(1 + \sigma)$. Plugging into EDM's ODE, $\dot\sigma/\sigma + \dot s/s = 1/t$ and $\dot\sigma s/\sigma = 1/t$, so $dx/dt = (x - D(x/(1 - t); \sigma))/t$, which equals the flow-matching velocity $\mathbb{E}[\epsilon - y \mid x_t]$. Its velocity loss is the denoising loss with weight $(1 + \sigma)^2/\sigma^2$, within a factor of 2 of EDM's weight for $\sigma_{\text{data}} = 1$. So rectified flow is EDM's canonical ODE with the scaling $s = 1/(1 + \sigma)$; differences in practice come mostly from the timestep distribution and the network.
:::

::: qa
**How does EDM's log-normal noise distribution relate to SD3's logit-normal timestep sampling?**

They are the same distribution in different coordinates. Under rectified flow, $\sigma = t/(1 - t)$, so $\mathrm{logit}(t) = \ln\sigma$. SD3 draws $u \sim \mathcal{N}(m, s^2)$ and sets $t = 1/(1 + e^{-u})$, which gives $\sigma = e^u$, a log-normal with $P_{\text{mean}} = m$ and $P_{\text{std}} = s$. Both concentrate training on intermediate noise levels, for EDM's reason that the loss can only be reduced there. SD3's default $(0, 1)$ is centered at $\sigma = 1$, higher than EDM's $e^{-1.2} \approx 0.3$, consistent with latents and high resolutions needing more noise (EDM2 uses $-0.4$, SVD up to $1.0$), though the latent scale makes the numbers only roughly comparable. In SD3's own comparison, rectified flow with this sampling beat their EDM baselines.
:::

## 8. EDM2: analyzing and improving the training dynamics

"Analyzing and Improving the Training Dynamics of Diffusion Models" (Karras, Aittala, Lehtinen, Hellsten, Aila and Laine, arXiv 2023), usually called EDM2, keeps EDM's formulation and asks a different question: does the network train in a balanced, predictable way? It finds it does not, fixes the causes one at a time in the ADM U-Net without changing its overall structure, and adds post-hoc EMA as an independent contribution.

### 8.1 The question and the setup

Diffusion training has a very noisy loss: the target depends on the noise level, the noise sample and the conditioning, and small errors early in the sampling chain can snowball. The authors argue that in such a setting the network should respond evenly and predictably to parameter updates, and that standard designs do not. They study the expected magnitudes of activations, weights and updates, and try to standardize all of them.

The testbed is class-conditional ImageNet at 512×512, run in the latent space of a pretrained autoencoder (8 times downsampling, so the network works on 64×64×4 latents), with the latents standardized to zero mean and standard deviation $\sigma_{\text{data}} = 0.5$. The exploration model has about 280M to 300M parameters ("EDM2-S"); training runs for $2^{31} \approx 2.1$ billion images at batch size 2048; sampling is EDM's deterministic sampler with 32 Heun steps (63 NFE). FID is computed without guidance on 50,000 images.

### 8.2 Config B: tuning, and a loss weight that adapts during training

Config A is EDM's setup (ADM network) moved to latents, with FID 8.00. Config B retunes it: batch size 2048 instead of 4096 with twice the learning rate ($2 \times 10^{-4}$), Adam's $\beta_2 = 0.99$ instead of 0.999, self-attention at 32×32 removed, $P_{\text{mean}} = -0.4$ and $P_{\text{std}} = 1.0$ instead of $-1.2$ and $1.2$ (in latent space the noise range where the loss can be reduced sits at higher noise than for RGB), and the power-function EMA of Section 8.6.

**Learned uncertainty weighting.** EDM's $\lambda(\sigma)$ makes the loss 1 at every noise level at initialization (Section 5.4), but as training proceeds the loss falls quickly at intermediate levels and hardly at the extremes, so the gradient contributions drift apart again. No fixed $\lambda$ can track that. EDM2 treats each noise level as a task in multi-task learning and uses the uncertainty weighting of Kendall et al. With a learned per-task uncertainty $\sigma_i$ (Kendall's symbol; here it is not a noise level), Kendall's loss is $\sum_i L_i/(2\sigma_i^2) + \ln\sigma_i$; writing $u_i = \ln\sigma_i^2$ and dropping a factor $\frac12$ gives $\sum_i L_i/e^{u_i} + u_i$. EDM2 makes this continuous over noise levels:

$$
\mathcal{L}(D_\theta, u) = \mathbb{E}_\sigma\Big[\frac{\lambda(\sigma)}{e^{u(\sigma)}}\,\mathcal{L}(D_\theta;\sigma) + u(\sigma)\Big],
$$

where $\mathcal{L}(D_\theta;\sigma)$ is the denoising loss at level $\sigma$ and $u(\sigma)$ is a learned function. What does it do?

$$
\begin{aligned}
0 &= \frac{\partial}{\partial u(\sigma)}\Big[\lambda\,e^{-u}\,\mathcal{L}(D_\theta;\sigma) + u\Big] = -\lambda\,e^{-u}\,\mathcal{L}(D_\theta;\sigma) + 1 && \text{(optimal } u \text{ at each } \sigma)\\
e^{u(\sigma)} &= \lambda(\sigma)\,\mathcal{L}(D_\theta;\sigma) && \text{(solve)}\\
\nabla_\theta\mathcal{L} &= \mathbb{E}_\sigma\Big[\frac{\lambda}{e^{u}}\,\nabla_\theta\mathcal{L}(D_\theta;\sigma)\Big] = \mathbb{E}_\sigma\Big[\frac{\nabla_\theta\mathcal{L}(D_\theta;\sigma)}{\mathcal{L}(D_\theta;\sigma)}\Big] && (u \text{ is a constant for } \theta \text{; substitute } e^u).
\end{aligned}
$$

So $u(\sigma)$ tracks the current log-loss at each noise level, and each level's gradient is divided by its current loss, which keeps the levels' contributions equal throughout training. At the optimum $\lambda$ drops out entirely; the authors found setting $\lambda = 1$ made virtually no difference, but kept EDM's $\lambda$ as a safety measure because it matters early, before $u$ has converged. In practice $u(\sigma)$ is a one-layer MLP on Fourier features of $c_{\text{noise}}(\sigma)$, trained jointly and discarded after training. In the official code the loss line is `(weight / logvar.exp()) * ((denoised - images) ** 2) + logvar`, where `logvar` is $u(\sigma)$. Config B reaches FID 7.24.

### 8.3 Config C: streamlining the architecture

To make the analysis tractable, config C simplifies the network: it removes all additive biases (and adds a constant-one input channel so the network can still offset the data), uses He's uniform initialization for all weights, replaces ADM's positional noise embedding with Fourier features, simplifies group normalization (no mean subtraction, no learned scale), and switches to cosine attention, which normalizes keys and queries before the dot product. The last change fixes attention maps that had become brittle and spiky as key and query magnitudes grew, and allows 16-bit arithmetic throughout. FID 6.96.

### 8.4 Config D: magnitude-preserving learned layers

**The problem: activations grow without bound.** In config C the activation magnitudes grow steadily throughout training, with no sign of leveling off, despite the group normalizations inside each block. The cause is the residual structure: the main path through the encoder, decoder and attention blocks has no normalization, keeps accumulating contributions from residual branches, and can amplify them through repeated convolutions. The authors hypothesize this keeps the network "in a perpetually unconverged and unoptimal state". Adding group normalization to the main path hurt quality badly, so instead every layer is made to preserve magnitudes on expectation.

**The derivation.** Assume input activations are mutually uncorrelated with equal standard deviation $\sigma_{\text{act}}$. Each output channel $i$ of a linear or convolutional layer is a dot product of a weight vector $w_i \in \mathbb{R}^m$ with $m$ input activations $a_j$:

$$
\begin{aligned}
\mathrm{Var}\Big[\sum_j w_{ij}\,a_j\Big] &= \sum_j w_{ij}^2\,\mathrm{Var}[a_j] && \text{(uncorrelated inputs)}\\
&= \lVert w_i\rVert_2^2\,\sigma_{\text{act}}^2 && \text{(equal variances)}\\
\text{output std} &= \lVert w_i\rVert_2\,\sigma_{\text{act}} && \text{(square root).}
\end{aligned}
$$

So dividing each output channel by $\lVert w_i\rVert_2$, which is the same as using $w_i/\lVert w_i\rVert_2$, preserves the activation magnitude regardless of the weights' size. With gradients flowing through the norm, this is weight normalization without its learned scale. Weights are initialized from a unit Gaussian. The direct link from weight size to activation size is cut, the activation drift disappears, and FID drops from 6.96 to 3.75, the largest single step of the ablation.

### 8.5 Config E: forced weight normalization and an explicit learning-rate decay

**The new problem: weights grow, so the effective learning rate decays.** With normalized weights, the loss is unchanged by rescaling $w$, so its gradient $g = \nabla_w\mathcal{L}$ is orthogonal to $w$; a gradient step with learning rate $\eta$ therefore always lengthens $w$:

$$
\begin{aligned}
\mathcal{L}(c\,w) &= \mathcal{L}(w) \text{ for } c > 0 && \text{(only } w/\lVert w\rVert \text{ is used)}\\
0 = \frac{d}{dc}\mathcal{L}(c\,w)\Big\vert_{c=1} &= w^\top\nabla_w\mathcal{L} && \text{(differentiate in } c)\\
\lVert w - \eta g\rVert^2 &= \lVert w\rVert^2 - 2\eta\,w^\top g + \eta^2\lVert g\rVert^2 = \lVert w\rVert^2 + \eta^2\lVert g\rVert^2 && (w^\top g = 0).
\end{aligned}
$$

Adam makes each element's step roughly $\eta$ in size whatever the gradient's scale, so a vector with $m$ elements moves by about $\eta\sqrt{m}$, and its direction $w/\lVert w\rVert$ turns by about $\eta\sqrt{m}/\lVert w\rVert$. As $\lVert w\rVert$ grows, the same $\eta$ turns the direction less: the effective learning rate decays on its own, by different amounts in different layers. Config D shows this weight growth even more strongly than config C.

**The fix.** Forced weight normalization: before each training step, rescale every stored weight vector back to a fixed norm, $\lVert w_i\rVert = \sqrt{m}$, that is, unit variance per element (in the official code, `self.weight.copy_(normalize(w))`, where `w` is the stored weight), while still normalizing on use as in config D. The two normalizations do different jobs. Normalizing on use makes the gradient orthogonal to $w_i$ (shown above), so Adam's statistics describe steps along the sphere. Resetting the stored norm stops the slow growth that the finite steps would otherwise cause. With weight norms fixed and Adam fixing the step size, each step changes the weights by a roughly fixed fraction, so the effective learning rate is set by $\eta$ alone and is the same in every layer. Because a constant learning rate would no longer converge, config E adds an inverse square-root decay, $\alpha(t) = \alpha_{\text{ref}}/\sqrt{\max(t/t_{\text{ref}}, 1)}$ for training iteration $t$ (the official code divides the reference rate by the square root of that maximum). FID 3.02.

### 8.6 Configs F and G, and post-hoc EMA

**Config F** removes all group normalization layers, now unnecessary, and adds a much weaker pixel normalization in the encoder's main path (a quarter as many layers), which still helps slightly, perhaps by counteracting correlations that violate the uncorrelated-inputs assumption. It also removes the second linear layer of the embedding network and the output nonlinearity, and moves the resampling onto the main path. FID 2.71.

**Config G** makes the fixed-function layers magnitude-preserving too: Fourier features are scaled by $\sqrt2$ to unit variance; SiLU outputs are divided by $\mathbb{E}_{x \sim \mathcal{N}(0,1)}[\mathrm{silu}(x)^2]^{1/2} \approx 0.596$; residual additions become weighted sums with a fixed residual weight of 30% in encoder and decoder blocks and 50% in the embedding, divided by the standard deviation of the weighted sum; and U-Net skip concatenations are scaled so both inputs contribute equally regardless of channel counts. Two learned, zero-initialized gains remain: one at the network output and one on the conditioning signal in each block. Dropout can now be turned off without harm in the small model. FID 2.56.

**Post-hoc EMA.** As an independent contribution, EDM2 replaces exponential averaging with a power-function profile, which puts vanishing weight on the earliest training steps and whose length is defined relative to the training time so far, stores snapshots of two such averages during training, and reconstructs any other averaging length afterwards by least squares. The optimal EMA length turns out to vary across configurations and to interact strongly with guidance strength. The mechanics are on the [deep learning training](deep-learning-training.html) page.

### 8.7 The ablation

| Config | Change | FID (ImageNet-512, no guidance) | Params (M) | Gflops |
|---|---|---|---|---|
| A | EDM baseline (ADM on latents) | 8.00 | 295.9 | 110.4 |
| B | + minor improvements (tuning, uncertainty weighting, power EMA) | 7.24 | 291.8 | 100.4 |
| C | + architectural streamlining | 6.96 | 277.8 | 100.3 |
| D | + magnitude-preserving learned layers | 3.75 | 277.8 | 101.2 |
| E | + control effective learning rate | 3.02 | 277.8 | 101.2 |
| F | + remove group normalizations | 2.71 | 280.2 | 102.1 |
| G | + magnitude-preserving fixed-function layers | 2.56 | 280.2 | 102.2 |

**What to read from it.** FID falls by a factor of about 3 at essentially constant parameter count and compute, so the gains come from training dynamics, not capacity. The two largest steps, D and E, are the two magnitude fixes: standardizing activations (6.96 to 3.75) and then weights and the effective learning rate (3.75 to 3.02). The authors also observe that in config G the optimal EMA length is sharper and the weight tensors agree on it, while in config B different tensors preferred very different lengths, which they read as the network having become more uniform.

### 8.8 Results

| ImageNet-512 | FID, no guidance | FID, with guidance | Params (M) | NFE |
|---|---|---|---|---|
| VDM++ (previous best without guidance) | 2.99 | 2.65 | 2455 | 256 |
| StyleGAN-XL (previous best with guidance) | none given | 2.41 | about 168 | 1 |
| EDM2-XS | 3.53 | 2.91 | 125 | 63 |
| EDM2-S | 2.56 | 2.23 | 280 | 63 |
| EDM2-M | 2.25 | 2.01 | 498 | 63 |
| EDM2-L | 2.06 | 1.88 | 777 | 63 |
| EDM2-XL | 1.96 | 1.85 | 1119 | 63 |
| EDM2-XXL | **1.91** | **1.81** | 1523 | 63 |

EDM2 set records of 1.91 without guidance and 1.81 with guidance (strength 1.2 for the XXL model), against 2.99 and 2.41 before, using deterministic sampling at 63 evaluations where the earlier diffusion models used stochastic sampling with hundreds. The paper's Figure 1 reports that EDM2 surpasses the previous state of the art with a model about 5 times smaller in compute per evaluation. Dropout helped the larger models (M to XXL), which showed overfitting, and hurt the smaller ones. Guidance used an XS-sized unconditional model for every size, which was sufficient even for XXL and cuts the usual doubling of guidance cost.

On ImageNet-64 in pixel space, EDM2 compares deterministic sampling: EDM's best deterministic result, 2.22, improves to 1.58 for EDM2-S at similar model size (63 evaluations) and to 1.33 for EDM2-L and XL. That is close to the best stochastic result then reported, 1.23 for RIN at 1,000 evaluations. (EDM2's table quotes EDM's ImageNet-64 results with the values from EDM's Figure 5c, Section 5.11 here, which is why the pretrained model appears as 1.57 rather than 1.55.)

The authors' next paper, autoguidance, guides these EDM2 models with a smaller, less-trained version of the same model instead of an unconditional one; see [diffusion models](diffusion-models.html).

**What this says.** EDM fixed the formulation; EDM2 fixes the optimization. Its lesson is general: in a residual network trained with normalized weights and Adam, activation and weight magnitudes drift, and the drift silently changes the effective learning rate per layer. Designing every layer to preserve magnitude, and keeping weights on a fixed sphere with an explicit learning-rate schedule, made the training dynamics controllable and was worth more than a larger model.

::: key
*How to say it in an interview:* "EDM2 keeps EDM's formulation and fixes the ADM U-Net's training dynamics. Activations grow without bound along the residual paths, so they make every layer magnitude-preserving: normalize each weight vector so a layer's output variance equals its input variance, and scale SiLU, Fourier features, sums and concatenations to unit variance. Normalized weights then grow, because the gradient is orthogonal to them, which silently decays the effective learning rate, so they force each weight vector back to a fixed norm every step and use an explicit inverse-square-root schedule. A learned per-noise-level uncertainty $u(\sigma)$ divides each level's gradient by its current loss. That took ImageNet-512 FID from 8.00 to 2.56 at the same size, and the XXL model to 1.81 with guidance at 63 deterministic evaluations."
:::

::: qa
**Why does EDM2 need magnitude-preserving layers?**

In the ADM U-Net, the main path of the residual encoder, decoder and attention blocks has no normalization, so it keeps accumulating contributions from residual branches and activation magnitudes grow throughout training without leveling off; normalization inside the blocks does not prevent it, and adding normalization to the main path hurt quality. Growing activations change how each layer responds to updates and, per the authors, keep the network from settling. Making each layer preserve magnitude on expectation removes the drift without data-dependent normalization: with uncorrelated unit-variance inputs, a unit's output standard deviation is $\lVert w_i\rVert\,\sigma_{\text{act}}$, so dividing by $\lVert w_i\rVert$ preserves it. This single change took FID from 6.96 to 3.75. Config G extends it to fixed functions: SiLU divided by 0.596, Fourier features times $\sqrt2$, weighted residual sums and balanced concatenations.
:::

::: qa
**What is forced weight normalization in EDM2, and why is it needed on top of ordinary weight normalization?**

With weight normalization the loss depends only on $w/\lVert w\rVert$, so $w^\top\nabla_w\mathcal{L} = 0$ and each gradient step increases the norm: $\lVert w - \eta g\rVert^2 = \lVert w\rVert^2 + \eta^2\lVert g\rVert^2$. Since Adam's steps have roughly fixed size, the angle the weight direction turns per step shrinks as the norm grows, so the effective learning rate decays uncontrollably and differently per layer. Forced weight normalization rescales each stored weight vector to a fixed norm ($\sqrt{\text{fan-in}}$, unit variance per element) before every step and still normalizes on use, so updates move along a sphere of fixed radius and the effective learning rate stays fixed. Because a constant rate then no longer anneals, EDM2 adds an explicit inverse square-root decay $\alpha_{\text{ref}}/\sqrt{\max(t/t_{\text{ref}}, 1)}$. Together these took FID from 3.75 to 3.02.
:::

::: qa
**Explain EDM2's learned loss weighting $u(\sigma)$.**

EDM's $\lambda(\sigma)$ equalizes the loss across noise levels only at initialization; as training progresses the loss falls at intermediate levels but not at the extremes, so gradient contributions drift apart. EDM2 treats each noise level as a task and uses Kendall et al.'s uncertainty weighting in continuous form: minimize $\mathbb{E}_\sigma[\lambda\,\mathcal{L}(\sigma)/e^{u(\sigma)} + u(\sigma)]$ jointly over the network and a small MLP $u(\sigma)$. Setting the derivative in $u$ to zero gives $e^{u} = \lambda\,\mathcal{L}(\sigma)$, so the network's gradient at level $\sigma$ becomes $\nabla\mathcal{L}(\sigma)/\mathcal{L}(\sigma)$: each level is normalized by its current loss, keeping contributions balanced over all of training. At the optimum $\lambda$ cancels; it is kept only for the early phase before $u$ converges. The MLP is discarded after training.
:::

## Rapid-fire review

::: qa
**What are EDM's default constants?**

$\sigma_{\min} = 0.002$, $\sigma_{\max} = 80$, $\rho = 7$ and $\sigma_{\text{data}} = 0.5$ for images in $[-1, 1]$; training noise $\ln\sigma \sim \mathcal{N}(-1.2, 1.2^2)$; and $N = 18$ Heun steps (35 NFE) for CIFAR-10, $N = 40$ (79 NFE) at 64×64.
:::

::: qa
**How many network evaluations do $N$ Heun steps cost in EDM?**

$2N - 1$: two per step except the last, which goes to $\sigma = 0$ with a plain Euler step that simply outputs $D_\theta(x;\sigma_{\min})$.
:::

::: qa
**What ODE does EDM integrate with $\sigma(t) = t$ and $s(t) = 1$?**

$dx/d\sigma = (x - D(x;\sigma))/\sigma$, which equals $-\sigma\nabla_x\log p(x;\sigma)$; an Euler step to $\sigma = 0$ lands exactly on $D(x;\sigma)$.
:::

::: qa
**Where does the $\rho = 7$ grid put its steps?**

Mostly at low noise: with 18 steps from 80 to 0.002, seven levels are below $\sigma = 1$ and only six above 10; step sizes scale like $\sigma^{6/7}$, nearly proportional to $\sigma$.
:::

::: qa
**What do $c_{\text{skip}}$ and $c_{\text{out}}$ do at the extremes of $\sigma$?**

As $\sigma \to 0$, $c_{\text{skip}} \to 1$ and $c_{\text{out}} \approx \sigma$, so the network predicts the noise and the input is passed through; as $\sigma \to \infty$, $c_{\text{skip}} \to 0$ and $c_{\text{out}} \to \sigma_{\text{data}}$, so the network predicts the clean image.
:::

::: qa
**What is $\sigma_{\text{data}}$ and where does it enter?**

The per-pixel standard deviation of the training data, 0.5 for images in $[-1, 1]$ (EDM2 standardizes its latents to the same value). It sets the noise level $\sigma = \sigma_{\text{data}}$ at which the preconditioning switches from noise prediction to image prediction, and it bounds $c_{\text{out}}$.
:::

::: qa
**In EDM's training ablation, what did the preconditioning (config D) and the new loss (config E) each contribute?**

D, the new preconditioning, left FID roughly unchanged except that it rescued the unstable VE models at 64×64; its value is robustness. E, the log-normal $p_{\text{train}}$ with $\lambda = 1/c_{\text{out}}^2$ on top of D, improved FID in every column and was the largest step for the VE CIFAR-10 models.
:::

::: qa
**What were EDM's headline FIDs?**

1.79 on class-conditional and 1.97 on unconditional CIFAR-10 at 35 NFE (deterministic), 1.55 for the pretrained ImageNet-64 ADM with EDM's stochastic sampler (from 2.07), and 1.36 for ADM retrained with EDM's recipe.
:::

::: qa
**Did stochastic sampling help EDM's own CIFAR-10 models?**

No. With EDM's training, deterministic sampling was best and any churn hurt; for ImageNet-64 it still helped (2.22 to 1.36), with less churn than the pretrained model needed.
:::

::: qa
**What do $S_{t\min}$ and $S_{t\max}$ guard against?**

Churn at very low and very high noise levels, where repeated noise replacement with a real denoiser drifts toward oversaturated colors; churn is applied only for $t_i \in [S_{t\min}, S_{t\max}]$.
:::

::: qa
**Why is Heun's $\alpha = 1$ convenient for discrete-time models like ADM?**

Its second slope evaluation is at exactly $t_{i+1}$, a grid point, so rounding the grid to the model's trained noise levels is enough; midpoint or Ralston variants evaluate between grid points.
:::

::: qa
**What were EDM2's ImageNet-512 results?**

FID 1.91 without guidance and 1.81 with guidance for EDM2-XXL, at 63 deterministic evaluations, against previous records of 2.99 and 2.41; EDM2-S, about 280M parameters, reaches 2.56 without guidance.
:::

::: qa
**Is EDM a new network architecture?**

No. It is a formulation plus sampling and training choices; the experiments reuse DDPM++, NCSN++ and ADM, with only a capacity redistribution for the CIFAR-10, FFHQ and AFHQv2 models. EDM2 is the paper that redesigns the network.
:::

## References

1. Karras, Aittala, Aila, Laine. "Elucidating the Design Space of Diffusion-Based Generative Models," 2022. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364); code: <https://github.com/NVlabs/edm>
2. Karras, Aittala, Lehtinen, Hellsten, Aila, Laine. "Analyzing and Improving the Training Dynamics of Diffusion Models" (EDM2), 2023. [arXiv:2312.02696](https://arxiv.org/abs/2312.02696); code: <https://github.com/NVlabs/edm2>
3. Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. "Score-Based Generative Modeling through Stochastic Differential Equations," 2020. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
4. Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models," 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
5. Song, Meng, Ermon. "Denoising Diffusion Implicit Models," 2020. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
6. Nichol, Dhariwal. "Improved Denoising Diffusion Probabilistic Models," 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
7. Dhariwal, Nichol. "Diffusion Models Beat GANs on Image Synthesis," 2021. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
8. Jolicoeur-Martineau, Li, Piché-Taillefer, Kachman, Mitliagkas. "Gotta Go Fast When Generating Data with Score-Based Models," 2021. [arXiv:2105.14080](https://arxiv.org/abs/2105.14080)
9. Karras, Aittala, Hellsten, Laine, Lehtinen, Aila. "Training Generative Adversarial Networks with Limited Data" (ADA), 2020. [arXiv:2006.06676](https://arxiv.org/abs/2006.06676)
10. Choi, Lee, Shin, Kim, Kim, Yoon. "Perception Prioritized Training of Diffusion Models," 2022. [arXiv:2204.00227](https://arxiv.org/abs/2204.00227)
11. Kendall, Gal, Cipolla. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics," 2017. [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
12. Salimans, Ho. "Progressive Distillation for Fast Sampling of Diffusion Models" ($v$-prediction), 2022. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
13. Liu, Gong, Liu. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow," 2022. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
14. Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3), 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
15. Song, Dhariwal, Chen, Sutskever. "Consistency Models," 2023. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
16. Blattmann et al. "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets," 2023. [arXiv:2311.15127](https://arxiv.org/abs/2311.15127)
17. Li et al. "Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation," 2024. [arXiv:2402.17245](https://arxiv.org/abs/2402.17245)
18. Karras, Aittala, Kynkäänniemi, Lehtinen, Aila, Laine. "Guiding a Diffusion Model with a Bad Version of Itself" (autoguidance), 2024. [arXiv:2406.02507](https://arxiv.org/abs/2406.02507)
