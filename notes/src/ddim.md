---
title: "DDIM: Denoising Diffusion Implicit Models"
subtitle: "A guided, fully derived reading of Song, Meng and Ermon (ICLR 2021): why a network trained as a DDPM also defines a family of non-Markovian samplers, including a deterministic one that skips steps, what the paper proves, and what its experiments show."
order: 1.25
updated: 2026-11-08
---

A DDPM is trained once, cheaply, but sampling from it walks a chain of $T = 1000$ steps with one network call per step. The DDIM paper cites about 20 hours to draw 50,000 images of size $32 \times 32$ from a DDPM on one RTX 2080 Ti, against under a minute for a GAN. Its key observation is that the DDPM training loss only ever looks at the noisy marginals $q(x_t \mid x_0)$, one time step at a time, never at how consecutive steps are linked. The paper builds a family of other "forward processes" with the same marginals, proves that each one's own variational objective is a reweighted version of the DDPM loss, and so reuses the trained network unchanged. The family has a free noise knob. At one end of it is a deterministic sampler that matches 1000-step quality in 20 to 100 steps, maps each noise vector to one image, and can be run backwards to encode images. Many later fast samplers (DPM-Solver, EDM's ODE samplers) and many real-image editing pipelines (DDIM inversion) build on this, which is why DDIM is among the diffusion topics most often reported in interviews.

**Roadmap.** The first two sections fix notation and list five facts that the proofs rely on. "DDPM, derived from scratch" rebuilds the parts of DDPM the paper relies on and ends with the one observation that starts DDIM: which distributions the loss touches. "The idea before the algebra" gives the whole argument in five steps. The next three sections follow the paper's Sections 3 and 4.1: the non-Markovian family and Lemma 1, the generative process and Theorem 1, and the DDIM update with its noise scale $\eta$. Two toy examples then let you check everything by hand. "Skipping steps" covers Section 4.2 and Appendix C.1, and "What the experiments show" reproduces Tables 1 and 3 with commentary. "DDIM as an ODE solver" covers Section 4.3, Proposition 1 and the link to DPM-Solver. "Consistency, interpolation and inversion" covers Sections 5.2 to 5.4. "DDIM in practice" covers timestep spacing, guidance and a sampler comparison. Interview questions follow the section they test, and a rapid-fire review closes the page. DDPM, the SDE view and DPM-Solver are treated more broadly on [diffusion models](diffusion-models.html).

## Notation

The DDIM paper uses $\alpha_t$ for what DDPM calls $\bar\alpha_t$. This page converts everything to DDPM's $\bar\alpha_t$, and writes the per-step factor as $1-\beta_t$ (never as DDPM's $\alpha_t$), matching the [diffusion models](diffusion-models.html) page. The paper also uses the letter $\sigma$ twice: $\sigma_t$ for the noise in a reverse step, and $\sigma$ for the noise level in its ODE. This page writes the first as $\varsigma_t$ (a variant of sigma, read "varsigma") and the second as $\bar\sigma$, so the two never collide.

| Symbol on this page | Meaning in words | DDIM paper | DDPM paper |
|---|---|---|---|
| $x_0$ | a clean data sample (an image), of dimension $d$ | $\boldsymbol{x}_0$ | $\mathbf{x}_0$ |
| $x_t$, $t = 1,\dots,T$ | the noisy sample at step $t$; $T = 1000$ in all experiments | $\boldsymbol{x}_t$ | $\mathbf{x}_t$ |
| $\epsilon$, $z$ | standard Gaussian noise $\mathcal{N}(0, I)$ | $\epsilon$ | $\epsilon$, $\mathbf{z}$ |
| $\beta_t$ | variance added by DDPM's forward step $t$ | $1 - \alpha_t/\alpha_{t-1}$ | $\beta_t$ |
| $1-\beta_t$ | per-step signal factor | $\alpha_t/\alpha_{t-1}$ | $\alpha_t$ |
| $\bar\alpha_t = \prod_{s \le t}(1-\beta_s)$ | fraction of signal variance left at step $t$; $\bar\alpha_0 = 1$ | $\alpha_t$ | $\bar\alpha_t$ |
| $\mathrm{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$ | signal-to-noise ratio at step $t$ | not named | not named |
| $\epsilon_\theta^{(t)}(x_t)$, short $\hat\epsilon$ | the network's estimate of the noise inside $x_t$ | $\epsilon_\theta^{(t)}(\boldsymbol{x}_t)$ | $\epsilon_\theta(\mathbf{x}_t, t)$ |
| $f_\theta^{(t)}(x_t)$, short $\hat x_0$ | the clean sample implied by $\hat\epsilon$ | $f_\theta^{(t)}$ | not named |
| $\tilde\beta_t$ | variance of DDPM's posterior $q(x_{t-1} \mid x_t, x_0)$ | written out | $\tilde\beta_t$ |
| $\varsigma_t$ | standard deviation of the fresh noise in the reverse step from $t$ to $t-1$ | $\sigma_t$ | $\sigma_t$ |
| $\eta \ge 0$ | one number that scales $\varsigma_t$: $\eta = 0$ is DDIM, $\eta = 1$ is DDPM | $\eta$ | none |
| $\hat\varsigma_t$ | the "larger variance" DDPM noise, $\sqrt{1 - \bar\alpha_t/\bar\alpha_{t-1}}$ | $\hat\sigma_t$ | $\sigma_t^2 = \beta_t$ |
| $q$ | DDPM's Markov forward process | $q$ | $q$ |
| $q_\varsigma$ | the DDIM family of inference distributions, indexed by the vector $\varsigma$ | $q_\sigma$ | none |
| $p_\theta$ | the generative (sampling) process | $p_\theta$ | $p_\theta$ |
| $\gamma = (\gamma_1, \dots, \gamma_T)$ | positive weights on the per-step noise losses | $\gamma$ | none |
| $L_\gamma$; $L_{\text{simple}}$ | weighted noise-prediction loss; $\gamma = \mathbf{1}$ gives DDPM's loss | $L_\gamma$, $L_{\mathbf{1}}$ | $L_{\text{simple}}$ |
| $J_\varsigma$ | the variational objective (negative ELBO) for $q_\varsigma$ | $J_\sigma$ | none |
| $\tau = (\tau_1 < \dots < \tau_S)$ | the increasing subsequence of steps a fast sampler visits; $\tau_0 := 0$ | $\tau$ | none |
| $S$ | number of sampling steps, which equals the number of network calls | $S$ or $\dim(\tau)$ | none |
| $\bar x = x/\sqrt{\bar\alpha}$, $\bar\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha}$ | rescaled state and its noise level, used in the ODE view | $\bar{\boldsymbol{x}}$, $\sigma$ | none |
| $\mathcal{N}(m, vI)$ | Gaussian with mean $m$ and covariance $vI$ | same | same |

::: pitfall
Reading the DDIM paper with DDPM's meaning of $\alpha_t$ gives wrong formulas. In the paper, $\alpha_t$ is the cumulative product, so its update coefficient $\sqrt{\alpha_{t-1}}$ is $\sqrt{\bar\alpha_{t-1}}$ here, and $1 - \alpha_t/\alpha_{t-1}$ is DDPM's $\beta_t$. The paper explains the choice in its Appendix C.2 (one set of hyperparameters, and it suits processes that are not diffusions) and switches back to DDPM's symbols only there.
:::

## Five tools

Every derivation below uses only these facts (four about Gaussians, one about least squares). They are stated once and cited by number.

**Tool 1 (linear-Gaussian marginalization).** If $x \sim \mathcal{N}(\mu, \Sigma)$ and $y \mid x \sim \mathcal{N}(Ax + b, \Lambda)$, then

$$
y \sim \mathcal{N}\big(A\mu + b,\ \Lambda + A\Sigma A^\top\big).
$$

Reason: write $y = Ax + b + \xi$ with $\xi \sim \mathcal{N}(0, \Lambda)$ independent of $x$. A linear function of independent Gaussians is Gaussian, means add, and covariances of independent parts add. In the isotropic case used here ($A = aI$, $\Sigma = vI$, $\Lambda = wI$) this reads: mean $a\mu + b$, covariance $(w + a^2 v)I$.

**Tool 2 (Gaussian conditioning).** If $u$ and $v$ are jointly Gaussian, then

$$
u \mid v \sim \mathcal{N}\big(m_u + C_{uv}C_{vv}^{-1}(v - m_v),\ C_{uu} - C_{uv}C_{vv}^{-1}C_{vu}\big),
$$

where $m$ are means and $C$ are covariance blocks. When every block is a scalar times $I$ (as here), $C_{uv}C_{vv}^{-1}$ is just a ratio of scalars. This is Bayes' rule for Gaussians; the "complete the square" route gives the same result, and it is done once by hand in Step 2 below.

**Tool 3 (KL between Gaussians).** In $d$ dimensions,

$$
\mathrm{KL}\big(\mathcal{N}(\mu_1, \Sigma_1)\,\big\|\,\mathcal{N}(\mu_2, \Sigma_2)\big) = \tfrac{1}{2}\Big[\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^\top\Sigma_2^{-1}(\mu_2 - \mu_1) - d + \ln\frac{\det\Sigma_2}{\det\Sigma_1}\Big].
$$

With equal covariances $\Sigma_1 = \Sigma_2 = s^2 I$, the trace is $d$ and the log-determinant ratio is $0$, so the KL is $\lVert \mu_1 - \mu_2 \rVert^2 / (2s^2)$. If only the means depend on a parameter, the trace and determinant terms are constants in that parameter.

**Tool 4 (the MSE minimizer is the conditional mean).** For random $X, Y$, $\arg\min_f \mathbb{E}\lVert Y - f(X)\rVert^2 = \mathbb{E}[Y \mid X]$. Reason: add and subtract $\mathbb{E}[Y \mid X]$; the cross term has zero mean by the tower rule, so $\mathbb{E}\lVert Y - f\rVert^2 = \mathbb{E}\lVert Y - \mathbb{E}[Y\mid X]\rVert^2 + \mathbb{E}\lVert \mathbb{E}[Y \mid X] - f(X)\rVert^2$, and only the second term depends on $f$.

**Tool 5 (the score of a Gaussian-smoothed density).** If $\bar x = x_0 + \bar\sigma\epsilon$ with $x_0 \sim p_0$, then $\nabla \log p(\bar x) = -\mathbb{E}[\epsilon \mid \bar x]/\bar\sigma$. Reason, line by line:

$$
\begin{aligned}
p(\bar x) &= \int p_0(x_0)\,\mathcal{N}(\bar x;\, x_0, \bar\sigma^2 I)\,dx_0 && \text{definition of the smoothed density}\\
\nabla p(\bar x) &= \int p_0(x_0)\,\mathcal{N}(\bar x;\, x_0, \bar\sigma^2 I)\,\Big(-\frac{\bar x - x_0}{\bar\sigma^2}\Big)dx_0 && \text{differentiate the Gaussian under the integral}\\
\nabla \log p(\bar x) &= \int p(x_0 \mid \bar x)\Big(-\frac{\bar x - x_0}{\bar\sigma^2}\Big)dx_0 = -\frac{\mathbb{E}[\bar x - x_0 \mid \bar x]}{\bar\sigma^2} && \text{divide by } p(\bar x)\text{; Bayes}\\
&= -\frac{\mathbb{E}[\epsilon \mid \bar x]}{\bar\sigma} && \bar x - x_0 = \bar\sigma\epsilon .
\end{aligned}
$$

This is Tweedie's formula in "variance exploding" form; the [diffusion models](diffusion-models.html) page has the general version.

## DDPM, derived from scratch

The DDIM paper assumes you know how DDPM's loss is derived, and its whole argument hangs on one feature of that loss. This section derives it in six steps.

### The forward chain

DDPM fixes (does not learn) a Markov chain that adds a little Gaussian noise at each step:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big), \qquad q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1}).
$$

"Markov" means $x_t$ depends on the past only through $x_{t-1}$. The factor $\sqrt{1-\beta_t}$ shrinks the signal so the variance stays bounded: if $x_{t-1}$ has unit variance, $x_t$ has variance $(1-\beta_t) + \beta_t = 1$. DDPM uses $\beta_t$ rising linearly from $10^{-4}$ to $0.02$; DDIM reuses those models and schedules.

### Step 1: the marginal $q(x_t \mid x_0)$, by induction

**Claim.** $q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)$ for every $t \ge 1$.

**Base case** $t = 1$: $q(x_1 \mid x_0) = \mathcal{N}(\sqrt{1-\beta_1}\,x_0, \beta_1 I)$, and $\bar\alpha_1 = 1 - \beta_1$, so $\sqrt{\bar\alpha_1} = \sqrt{1-\beta_1}$ and $1 - \bar\alpha_1 = \beta_1$.

**Inductive step.** Assume $x_{t-1} \mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_{t-1}}\,x_0, (1-\bar\alpha_{t-1})I)$. Apply Tool 1 with $x = x_{t-1}$, $y = x_t$, input mean $\mu = \sqrt{\bar\alpha_{t-1}}\,x_0$ and input variance $v = 1-\bar\alpha_{t-1}$ (the assumption), and with the forward step's $a = \sqrt{1-\beta_t}$, $b = 0$, $w = \beta_t$:

$$
\begin{aligned}
\text{mean} &= \sqrt{1-\beta_t}\,\sqrt{\bar\alpha_{t-1}}\,x_0 = \sqrt{\bar\alpha_t}\,x_0 && \bar\alpha_t = (1-\beta_t)\bar\alpha_{t-1}\\
\text{variance} &= \beta_t + (1-\beta_t)(1-\bar\alpha_{t-1}) && \text{Tool 1: } w + a^2 v\\
&= \beta_t + 1 - \beta_t - (1-\beta_t)\bar\alpha_{t-1} && \text{expand}\\
&= 1 - \bar\alpha_t && (1-\beta_t)\bar\alpha_{t-1} = \bar\alpha_t .
\end{aligned}
$$

Equivalently, $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$.

**What this says.** You can jump from a clean image to any noise level with one Gaussian draw; no chain needs to be simulated. With DDPM's linear schedule, $\sqrt{\bar\alpha_{1000}} \approx 0.0064$, so $x_T$ is very close to pure noise, and sampling starts from $p(x_T) = \mathcal{N}(0, I)$.

### Step 2: the posterior $q(x_{t-1} \mid x_t, x_0)$, by completing the square

By Bayes' rule and the Markov property ($x_t$ depends on $x_0$ only through $x_{t-1}$ once $x_{t-1}$ is known):

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0)\,q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} = \frac{q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} .
$$

As a function of $x_{t-1}$, the denominator is constant. Everything is isotropic, so work per coordinate and write $y$ for a coordinate of $x_{t-1}$. The log of the numerator is

$$
-\frac{\big(x_t - \sqrt{1-\beta_t}\,y\big)^2}{2\beta_t} - \frac{\big(y - \sqrt{\bar\alpha_{t-1}}\,x_0\big)^2}{2(1-\bar\alpha_{t-1})} + \text{const}.
$$

Expand both squares and collect powers of $y$:

$$
-\frac{y^2}{2}\Big[\frac{1-\beta_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}\Big] + y\Big[\frac{\sqrt{1-\beta_t}\,x_t}{\beta_t} + \frac{\sqrt{\bar\alpha_{t-1}}\,x_0}{1-\bar\alpha_{t-1}}\Big] + \text{const}.
$$

A Gaussian $\mathcal{N}(m, v)$ has log density $-y^2/(2v) + y\,m/v + \text{const}$, so match coefficients. The precision is

$$
\begin{aligned}
\frac{1}{v} &= \frac{1-\beta_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}} = \frac{(1-\beta_t)(1-\bar\alpha_{t-1}) + \beta_t}{\beta_t(1-\bar\alpha_{t-1})} && \text{common denominator}\\
&= \frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})} && \text{numerator expanded as in Step 1},
\end{aligned}
$$

so $v = \tilde\beta_t := \dfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$. The mean is $v$ times the bracket multiplying $y$:

$$
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t, \qquad q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde\mu_t, \tilde\beta_t I).
$$

**What this says.** The true reverse step $q(x_{t-1} \mid x_t)$ is intractable because it averages over all clean images that could have produced $x_t$. Once $x_0$ is also given, the step back is a Gaussian whose mean blends $x_0$ and $x_t$ and whose variance $\tilde\beta_t$ is a little smaller than $\beta_t$. It is the target that a learned reverse step will imitate. Notice where the Markov chain entered: only through the factor $q(x_t \mid x_{t-1})$.

### Step 3: the ELBO and its three kinds of terms

The model is $p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)$ with $p(x_T) = \mathcal{N}(0, I)$, and the likelihood of a data point is $p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T}$. Take any distribution $q(x_{1:T} \mid x_0)$, multiply and divide the integrand by it, so the integral becomes an expectation under $q$, and then apply Jensen's inequality ($\log$ is concave):

$$
\log p_\theta(x_0) = \log \mathbb{E}_{q(x_{1:T} \mid x_0)}\Big[\frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\Big] \ge \mathbb{E}_{q(x_{1:T} \mid x_0)}\big[\log p_\theta(x_{0:T}) - \log q(x_{1:T} \mid x_0)\big].
$$

Negate, and also average over data $x_0 \sim q(x_0)$, to get a loss $L := \mathbb{E}_{q(x_0)\,q(x_{1:T} \mid x_0)}[\log q(x_{1:T} \mid x_0) - \log p_\theta(x_{0:T})] \ge \mathbb{E}_{q(x_0)}[-\log p_\theta(x_0)]$. Below, $\mathbb{E}_q$ means this joint expectation over $x_{0:T}$. Write out both factorizations:

$$
L = \mathbb{E}_q\Big[-\log p(x_T) - \sum_{t=1}^{T}\log p_\theta(x_{t-1} \mid x_t) + \sum_{t=1}^{T}\log q(x_t \mid x_{t-1})\Big].
$$

For $t \ge 2$, rearrange the Bayes identity of Step 2: $q(x_t \mid x_{t-1}) = q(x_{t-1} \mid x_t, x_0)\,q(x_t \mid x_0)/q(x_{t-1} \mid x_0)$. Summing the logs,

$$
\sum_{t=2}^{T}\log q(x_t \mid x_{t-1}) = \sum_{t=2}^{T}\log q(x_{t-1} \mid x_t, x_0) + \underbrace{\sum_{t=2}^{T}\big[\log q(x_t \mid x_0) - \log q(x_{t-1} \mid x_0)\big]}_{\text{telescopes to } \log q(x_T \mid x_0) - \log q(x_1 \mid x_0)} .
$$

The leftover $-\log q(x_1 \mid x_0)$ cancels the $t = 1$ term $\log q(x_1 \mid x_0)$ of the original sum. Grouping each log-ratio into a KL (the expectation of $\log(q/p)$ over $q$ is a KL; under $q$, the conditional law of $x_{t-1}$ given $(x_t, x_0)$ is $q(x_{t-1} \mid x_t, x_0)$ by definition):

$$
L = \underbrace{\mathbb{E}\,\mathrm{KL}\big(q(x_T \mid x_0)\,\|\,p(x_T)\big)}_{L_T} + \sum_{t=2}^{T}\underbrace{\mathbb{E}_{q(x_0, x_t)}\,\mathrm{KL}\big(q(x_{t-1} \mid x_t, x_0)\,\|\,p_\theta(x_{t-1} \mid x_t)\big)}_{L_{t-1}} + \underbrace{\mathbb{E}_{q(x_0, x_1)}\big[-\log p_\theta(x_0 \mid x_1)\big]}_{L_0}.
$$

**What this says.** $L_T$ has no parameters. Each $L_{t-1}$ compares the learned reverse step with the Gaussian target of Step 2, averaged over pairs $(x_0, x_t)$, and $L_0$ is a reconstruction term. To evaluate $L_{t-1}$ you need samples of the pair $(x_0, x_t)$, that is $q(x_0)\,q(x_t \mid x_0)$, plus the formula for the target.

### Step 4: each $L_{t-1}$ is a squared distance between means

DDPM sets $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \varsigma_t^2 I)$ with a fixed $\varsigma_t^2$ ($\beta_t$ or $\tilde\beta_t$). By Tool 3, the trace and log-determinant terms do not involve $\theta$, so

$$
L_{t-1} = \mathbb{E}_{q(x_0, x_t)}\Big[\frac{1}{2\varsigma_t^2}\big\lVert \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\big\rVert^2\Big] + C,
$$

and when $\varsigma_t^2 = \tilde\beta_t$ the covariances are equal and $C = 0$.

### Step 5: the noise parameterization and $L_{\text{simple}}$

Substitute $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ (Step 1, solved for $x_0$) into $\tilde\mu_t$:

$$
\begin{aligned}
\tilde\mu_t &= \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\cdot\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}} + \frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t && \text{substitute}\\
&= \frac{\beta_t}{(1-\bar\alpha_t)\sqrt{1-\beta_t}}\big(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\big) + \frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t && \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} = \frac{1}{\sqrt{1-\beta_t}} .
\end{aligned}
$$

Collect the $x_t$ coefficients over the common denominator $(1-\bar\alpha_t)\sqrt{1-\beta_t}$:

$$
\frac{\beta_t + (1-\beta_t)(1-\bar\alpha_{t-1})}{(1-\bar\alpha_t)\sqrt{1-\beta_t}} = \frac{1-\bar\alpha_t}{(1-\bar\alpha_t)\sqrt{1-\beta_t}} = \frac{1}{\sqrt{1-\beta_t}} .
$$

The $\epsilon$ coefficient is $-\beta_t\sqrt{1-\bar\alpha_t}/\big((1-\bar\alpha_t)\sqrt{1-\beta_t}\big) = -\beta_t/\big(\sqrt{1-\bar\alpha_t}\sqrt{1-\beta_t}\big)$. Hence

$$
\tilde\mu_t = \frac{1}{\sqrt{1-\beta_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big).
$$

$x_t$ is an input to the network, so the only unknown is $\epsilon$. DDPM therefore uses the same form with a network $\epsilon_\theta(x_t, t)$ in place of $\epsilon$. The mean difference is $\tilde\mu_t - \mu_\theta = -\frac{\beta_t}{\sqrt{1-\beta_t}\sqrt{1-\bar\alpha_t}}(\epsilon - \epsilon_\theta)$, and Step 4 becomes

$$
L_{t-1} = \mathbb{E}_{x_0, \epsilon}\Big[\frac{\beta_t^2}{2\varsigma_t^2(1-\beta_t)(1-\bar\alpha_t)}\big\lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big)\big\rVert^2\Big] + C .
$$

Ho et al. drop the weights and sample $t$ uniformly. The DDIM paper writes the general weighted family as its eq. (5),

$$
L_\gamma(\epsilon_\theta) := \sum_{t=1}^{T}\gamma_t\,\mathbb{E}_{x_0 \sim q(x_0),\ \epsilon \sim \mathcal{N}(0, I)}\Big[\big\lVert \epsilon_\theta^{(t)}\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon\big) - \epsilon\big\rVert^2\Big],
$$

where $\epsilon_\theta^{(t)}$ is the noise predictor used at step $t$. DDPM's $L_{\text{simple}}$ is $L_{\mathbf{1}}$ (all $\gamma_t = 1$) up to the constant factor $1/T$.

### Step 6: what the loss touches

Look at what you need to compute one term of $L_\gamma$: a data sample $x_0 \sim q(x_0)$, a Gaussian $\epsilon$, and the number $\bar\alpha_t$. In distributional terms, the loss depends on the data distribution and on the $T$ marginals $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$, each one separately.

It does not depend on the forward transitions $q(x_t \mid x_{t-1})$, on the joint law of $(x_s, x_t)$ for two different times, or on whether the forward process is Markov. The Markov chain was used to derive the ELBO, but it left no trace in the objective that is actually trained.

**What this says.** $L_\gamma$, as a formula, cannot tell DDPM's forward chain apart from any other joint distribution $q'(x_{1:T} \mid x_0)$ whose single-time marginals equal $\mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$. This is only half of the argument. A different $q'$ comes with its own generative process and its own variational objective, and nothing so far says that objective reduces to $L_\gamma$. If it does, the network trained with $L_{\text{simple}}$ serves the new process too. The paper's Section 3 opens with the observation, then builds a family of such $q'$ and proves the reduction (Theorem 1 below).

::: key
**How to say it in an interview.** "DDPM's loss is a weighted noise-prediction MSE: pick $t$, noise $x_0$ to $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$, regress $\epsilon$. That only uses the per-time marginals $q(x_t \mid x_0)$; it never uses the transitions between steps or the Markov property. So other forward processes with the same marginals can lead to the same training objective. DDIM builds a non-Markovian family of them, proves that each one's variational objective is a reweighted version of this loss, and picks members whose reverse process can be deterministic and can skip steps, all with the same trained network."
:::

::: qa
**Which distributions does the DDPM training loss depend on, and why is that the starting point of DDIM?**

$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\lVert \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\rVert^2$ needs only the data distribution and the marginals $q(x_t \mid x_0)$, one $t$ at a time. The Markov transitions $q(x_t \mid x_{t-1})$ were used to derive the ELBO (they give the Gaussian posteriors and the telescoping), but after the noise reparameterization they drop out. Consequently the loss cannot distinguish inference distributions that share these marginals. That alone does not license reusing the network for another process; one must also show that the other process's own variational objective reduces to the same loss. DDIM does exactly that: it builds a non-Markovian family with DDPM's marginals, shows its variational objective is a reweighted $L_{\text{simple}}$, and reuses the DDPM network to sample from a different, faster generative process. No retraining is needed.
:::

::: qa
**Derive the DDPM posterior $q(x_{t-1} \mid x_t, x_0)$.**

Bayes and the Markov property give $q(x_{t-1} \mid x_t, x_0) \propto q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)$, a product of two Gaussians in $x_{t-1}$: $\mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t)$ and $\mathcal{N}(x_{t-1}; \sqrt{\bar\alpha_{t-1}}x_0, 1-\bar\alpha_{t-1})$. Collecting the quadratic terms gives precision $\frac{1-\beta_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}} = \frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}$, so the variance is $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$. The linear terms give the mean $\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{1-\beta_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$. Substituting $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$ turns it into $\frac{1}{\sqrt{1-\beta_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon)$, which is why predicting $\epsilon$ is enough to parameterize the reverse mean.
:::

## The idea before the algebra

The generative process of a diffusion model is built to imitate the reverse of the inference ("forward") process. So if you want a different sampler, change the inference process. Step 6 suggests what to keep fixed: every marginal $q(x_t \mid x_0)$, because those are all the loss sees. Theorem 1 will confirm that this is enough for the family DDIM builds.

A concrete picture. There are many ways to build a sequence of noisy versions of one image with the right per-time distributions. DDPM adds fresh, independent noise at every step, so the noise inside $x_t$ is a mixture of everything added so far, and consecutive states share only part of it. The opposite extreme draws a single noise vector $\epsilon$ and uses it at every level: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ for all $t$. Each $x_t$ then has exactly DDPM's marginal, but the whole path is fixed once $x_0$ and $\epsilon$ are. DDIM's family contains both of these couplings, and a continuum of others, indexed by one variance parameter per step.

The paper's argument has five steps:

1. Define a family $q_\varsigma$ of inference distributions, written backward in time, whose marginals equal DDPM's (Section 3.1, Lemma 1).
2. Define a generative process $p_\theta$ that copies the backward conditionals of $q_\varsigma$, with the unknown $x_0$ replaced by the network's estimate (Section 3.2).
3. Prove that the variational objective of each member equals $L_\gamma$ plus a constant for some positive weights, so the $L_{\text{simple}}$ network is optimal for all of them when parameters are not shared across $t$ (Theorem 1).
4. Read off the sampler. Its noise can be set to zero, which gives a deterministic "implicit" model, DDIM (Section 4.1).
5. Repeat the construction on a subsequence of time steps to sample in far fewer steps (Section 4.2, Appendix C.1).

The paper's Figure 1 shows the difference as graphical models: in DDPM's inference model each $x_t$ has one parent, $x_{t-1}$; in the new family each $x_{t-1}$ depends on $x_t$ and on $x_0$.

## Non-Markovian forward processes (Section 3.1)

### The definition

Fix a vector $\varsigma = (\varsigma_1, \dots, \varsigma_T)$ with $\varsigma_t \ge 0$. The paper's eq. (6) defines the joint distribution backward in time:

$$
q_\varsigma(x_{1:T} \mid x_0) := q_\varsigma(x_T \mid x_0)\prod_{t=2}^{T} q_\varsigma(x_{t-1} \mid x_t, x_0), \qquad q_\varsigma(x_T \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_T}\,x_0,\ (1-\bar\alpha_T)I\big),
$$

and its eq. (7) gives the backward conditionals for $t \ge 2$:

$$
q_\varsigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\Big(\sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}\cdot\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}},\ \ \varsigma_t^2 I\Big).
$$

This needs $\varsigma_t^2 \le 1 - \bar\alpha_{t-1}$ so the square root is real. It is convenient to name the coefficient

$$
k_t := \frac{\sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}}{\sqrt{1-\bar\alpha_t}}, \qquad \text{mean} = \sqrt{\bar\alpha_{t-1}}\,x_0 + k_t\big(x_t - \sqrt{\bar\alpha_t}\,x_0\big).
$$

**How to read the mean.** The fraction $(x_t - \sqrt{\bar\alpha_t}\,x_0)/\sqrt{1-\bar\alpha_t}$ is exactly the unit-variance noise $\epsilon$ that turned $x_0$ into $x_t$ (solve Step 1 for $\epsilon$). So the mean says: take the signal at the level of step $t-1$, $\sqrt{\bar\alpha_{t-1}}\,x_0$, and re-add the same noise direction with amplitude $\sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}$. Then add $\varsigma_t$ worth of fresh noise. The two noise amounts have variances $(1-\bar\alpha_{t-1}-\varsigma_t^2) + \varsigma_t^2 = 1-\bar\alpha_{t-1}$, the total noise variance step $t-1$ should have. That is Lemma 1 in one line, and it is proved properly below.

The paper keeps calling this a "forward process" (its footnote says the term is overloaded when the inference model is not a diffusion). It is defined from $x_T$ downward and is really an inference distribution over $x_{1:T}$ given $x_0$.

### Where eq. (7) comes from: design it yourself

Suppose you want a Gaussian $q(x_{t-1} \mid x_t, x_0)$ with covariance $\varsigma_t^2 I$ and a mean linear in $x_0$ and $x_t$, say $\nu\,x_0 + k\,x_t$, such that the marginals stay DDPM's. Given $x_t \mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$, Tool 1 gives

$$
x_{t-1} \mid x_0 \sim \mathcal{N}\Big(\big(\nu + k\sqrt{\bar\alpha_t}\big)x_0,\ \big(\varsigma_t^2 + k^2(1-\bar\alpha_t)\big)I\Big).
$$

Matching DDPM's marginal at $t-1$ requires two equations:

$$
\nu + k\sqrt{\bar\alpha_t} = \sqrt{\bar\alpha_{t-1}}, \qquad \varsigma_t^2 + k^2(1-\bar\alpha_t) = 1-\bar\alpha_{t-1}.
$$

There are three unknowns $(\nu, k, \varsigma_t)$ and two equations, so one degree of freedom per step remains; take it to be $\varsigma_t$. The second equation gives $k = \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}/\sqrt{1-\bar\alpha_t} = k_t$, choosing the nonnegative root. The first gives $\nu = \sqrt{\bar\alpha_{t-1}} - k_t\sqrt{\bar\alpha_t}$. The mean $\nu x_0 + k_t x_t = \sqrt{\bar\alpha_{t-1}}\,x_0 + k_t(x_t - \sqrt{\bar\alpha_t}\,x_0)$ is eq. (7).

**What this says.** Eq. (7) is not a guess: it is the general linear-Gaussian backward step that preserves DDPM's marginals. The negative root would also preserve them, but it makes consecutive states anti-correlated, which is not a gradual noising path, and it does not contain DDPM (for DDPM, $k_t > 0$, as shown later).

### Lemma 1: the marginals match

**Lemma 1.** For every $t$, $q_\varsigma(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)$.

**Proof, by induction from $t = T$ down to $t = 1$.**

*A preliminary fact.* Given $x_0$, the factorization is a chain $x_T \to x_{T-1} \to \dots \to x_1$. For any $t \ge 2$, the joint density of $(x_t, x_{t-1})$ given $x_0$ is $q_\varsigma(x_t \mid x_0)\,q_\varsigma(x_{t-1} \mid x_t, x_0)$. To see it, integrate out $x_1, \dots, x_{t-2}$ first, innermost first: each is the variable of its own conditional density, which integrates to one. Then integrate out $x_T, \dots, x_{t+1}$, which by definition leaves the marginal of $x_t$. Consequently

$$
q_\varsigma(x_{t-1} \mid x_0) = \int q_\varsigma(x_t \mid x_0)\,q_\varsigma(x_{t-1} \mid x_t, x_0)\,dx_t .
$$

*Base case* $t = T$: true by definition of $q_\varsigma(x_T \mid x_0)$.

*Inductive step.* Assume $q_\varsigma(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$. In eq. (7), $x_{t-1} \mid x_t, x_0$ is linear-Gaussian in $x_t$ with $A = k_t I$, $b = (\sqrt{\bar\alpha_{t-1}} - k_t\sqrt{\bar\alpha_t})\,x_0$ and $\Lambda = \varsigma_t^2 I$. Tool 1:

$$
\begin{aligned}
\text{mean} &= k_t\sqrt{\bar\alpha_t}\,x_0 + \big(\sqrt{\bar\alpha_{t-1}} - k_t\sqrt{\bar\alpha_t}\big)x_0 && A\mu + b\\
&= \sqrt{\bar\alpha_{t-1}}\,x_0 && \text{the } k_t \text{ terms cancel}\\
\text{variance} &= \varsigma_t^2 + k_t^2(1-\bar\alpha_t) && \Lambda + A\Sigma A^\top\\
&= \varsigma_t^2 + \frac{1-\bar\alpha_{t-1}-\varsigma_t^2}{1-\bar\alpha_t}\,(1-\bar\alpha_t) && \text{definition of } k_t\\
&= 1-\bar\alpha_{t-1}.
\end{aligned}
$$

So $q_\varsigma(x_{t-1} \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}}\,x_0, (1-\bar\alpha_{t-1})I)$, and the induction runs from $t = T$ down to $t = 2$, covering every $x_1, \dots, x_T$. $\blacksquare$

The paper's proof (Appendix B) is the same computation, citing Bishop's textbook for Tool 1.

**What this says.** Whatever $\varsigma$ you choose, each $x_t$ on its own looks exactly like a DDPM-noised image at step $t$. Only the correlations between noisy images at different times change. This is the property the loss cannot see.

::: key
**How to say it in an interview.** "DDIM defines the inference distribution backward: $x_T$ from $q(x_T \mid x_0)$, then $x_{t-1}$ given $(x_t, x_0)$ as a Gaussian with mean $\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}\,\epsilon_t$ and variance $\varsigma_t^2$, where $\epsilon_t$ is the noise in $x_t$. By induction from $T$ down, if $x_t$ has DDPM's marginal then the mean of $x_{t-1}$ is $\sqrt{\bar\alpha_{t-1}}x_0$ and its variance is $(1-\bar\alpha_{t-1}-\varsigma_t^2) + \varsigma_t^2 = 1-\bar\alpha_{t-1}$. So every member of the family has DDPM's marginals, and the loss can't tell them apart."
:::

### The implied forward step, and where non-Markovianity appears

The paper's eq. (8) recovers the forward direction by Bayes' rule, $q_\varsigma(x_t \mid x_{t-1}, x_0) = q_\varsigma(x_{t-1} \mid x_t, x_0)\,q_\varsigma(x_t \mid x_0)/q_\varsigma(x_{t-1} \mid x_0)$, and notes it is Gaussian without computing it. Here is the computation, for $t \ge 2$.

*Joint moments given $x_0$.* Reparameterize the pair: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ (Lemma 1), and from eq. (7), $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_0 + k_t\sqrt{1-\bar\alpha_t}\,\epsilon + \varsigma_t z$ with $z$ independent of $\epsilon$. Then, given $x_0$:

$$
\operatorname{Var}(x_t) = (1-\bar\alpha_t)I, \qquad \operatorname{Var}(x_{t-1}) = (1-\bar\alpha_{t-1})I, \qquad \operatorname{Cov}(x_t, x_{t-1}) = k_t(1-\bar\alpha_t)I = \sqrt{(1-\bar\alpha_t)(1-\bar\alpha_{t-1}-\varsigma_t^2)}\;I .
$$

*Condition $x_t$ on $x_{t-1}$ (Tool 2).* The regression coefficient is

$$
\rho_t := \frac{\operatorname{Cov}}{\operatorname{Var}(x_{t-1})} = \frac{\sqrt{(1-\bar\alpha_t)(1-\bar\alpha_{t-1}-\varsigma_t^2)}}{1-\bar\alpha_{t-1}},
$$

and

$$
\begin{aligned}
\text{mean} &= \sqrt{\bar\alpha_t}\,x_0 + \rho_t\big(x_{t-1} - \sqrt{\bar\alpha_{t-1}}\,x_0\big) = \rho_t\,x_{t-1} + \underbrace{\big(\sqrt{\bar\alpha_t} - \rho_t\sqrt{\bar\alpha_{t-1}}\big)}_{c_t}\,x_0 ,\\
\text{variance} &= (1-\bar\alpha_t) - \frac{(1-\bar\alpha_t)(1-\bar\alpha_{t-1}-\varsigma_t^2)}{1-\bar\alpha_{t-1}} = \frac{(1-\bar\alpha_t)\,\varsigma_t^2}{1-\bar\alpha_{t-1}} .
\end{aligned}
$$

So $q_\varsigma(x_t \mid x_{t-1}, x_0) = \mathcal{N}\big(\rho_t x_{t-1} + c_t x_0,\ \frac{(1-\bar\alpha_t)\varsigma_t^2}{1-\bar\alpha_{t-1}} I\big)$. The non-Markovianity is the term $c_t\,x_0$: the next state depends on the clean image, not only on the previous state.

*Why this one conditional decides the question.* Given $x_0$, the definition makes $x_T \to x_{T-1} \to \dots \to x_1$ a chain: each variable depends only on the one just above it. "Past and future are independent given the present" is a symmetric statement, so the same variables read upward, $x_1 \to x_2 \to \dots \to x_T$, are also a chain given $x_0$. Hence $q_\varsigma(x_t \mid x_{t-1}, \dots, x_1, x_0) = q_\varsigma(x_t \mid x_{t-1}, x_0)$: the earlier states add nothing. The forward process is Markov in DDPM's sense, with $x_t$ depending on the past only through $x_{t-1}$, exactly when this conditional does not depend on $x_0$ either, for every $t \ge 2$.

*When is it Markov?* Exactly when $c_t = 0$, that is $\rho_t = \sqrt{\bar\alpha_t/\bar\alpha_{t-1}} = \sqrt{1-\beta_t}$. Both sides are nonnegative, so squaring loses nothing. Solve for $\varsigma_t^2$:

$$
\begin{aligned}
\frac{(1-\bar\alpha_t)(1-\bar\alpha_{t-1}-\varsigma_t^2)}{(1-\bar\alpha_{t-1})^2} &= \frac{\bar\alpha_t}{\bar\alpha_{t-1}} && \text{square both sides}\\
1-\bar\alpha_{t-1}-\varsigma_t^2 &= \frac{\bar\alpha_t(1-\bar\alpha_{t-1})^2}{\bar\alpha_{t-1}(1-\bar\alpha_t)} && \text{isolate}\\
\varsigma_t^2 &= (1-\bar\alpha_{t-1})\,\frac{\bar\alpha_{t-1}(1-\bar\alpha_t) - \bar\alpha_t(1-\bar\alpha_{t-1})}{\bar\alpha_{t-1}(1-\bar\alpha_t)} && \text{common denominator}\\
&= (1-\bar\alpha_{t-1})\,\frac{\bar\alpha_{t-1} - \bar\alpha_t}{\bar\alpha_{t-1}(1-\bar\alpha_t)} = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\Big(1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}\Big) = \tilde\beta_t && \text{cancel } \bar\alpha_{t-1}\bar\alpha_t;\ 1 - \tfrac{\bar\alpha_t}{\bar\alpha_{t-1}} = \beta_t .
\end{aligned}
$$

At $\varsigma_t^2 = \tilde\beta_t$ the forward variance is $\frac{1-\bar\alpha_t}{1-\bar\alpha_{t-1}}\cdot\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t = \beta_t$ and the mean is $\sqrt{1-\beta_t}\,x_{t-1}$: exactly DDPM's forward step. If $\varsigma_t^2 \ne \tilde\beta_t$ for even one $t \ge 2$, the process is non-Markovian. In the one-parameter family used in the experiments (the noise scale $\eta$ of Section 4.1, below), this means the forward process is Markov only at $\eta = 1$.

**What this says, with the $\varsigma = 0$ case.** Set $\varsigma_t = 0$. Then the forward variance is zero and $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{\frac{1-\bar\alpha_t}{1-\bar\alpha_{t-1}}}\big(x_{t-1} - \sqrt{\bar\alpha_{t-1}}\,x_0\big)$: the step rescales the noise part of $x_{t-1}$ without adding new noise. To do that it must know which part of $x_{t-1}$ is signal and which is noise, and that requires $x_0$. This is the intuitive reason the forward process cannot be Markov. (For $\varsigma_t^2 < \tilde\beta_t$, $\rho_t > \sqrt{1-\beta_t}$, so $c_t < 0$: the step subtracts a multiple of $x_0$.)

::: key
**How to say it in an interview.** "Invert eq. (7) with Gaussian conditioning: $q_\varsigma(x_t \mid x_{t-1}, x_0)$ has mean $\rho_t x_{t-1} + c_t x_0$ with $c_t = \sqrt{\bar\alpha_t} - \rho_t\sqrt{\bar\alpha_{t-1}}$. The coefficient $c_t$ vanishes only when $\varsigma_t^2 = \tilde\beta_t$, which is DDPM. For smaller noise the forward step has to know $x_0$, because it amplifies the existing noise in $x_{t-1}$ instead of adding fresh noise, and separating noise from signal requires $x_0$. DDIM does not need the forward step to be Markov, because the loss never uses it."
:::

### How $\varsigma$ controls stochasticity

Three regimes, all with the same marginals:

- **$\varsigma_t = 0$ for all $t$.** Given $x_T$ and $x_0$, every $x_{t-1}$ is fixed. By induction, with $\epsilon := (x_T - \sqrt{\bar\alpha_T}x_0)/\sqrt{1-\bar\alpha_T}$, each step gives $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon$: one noise vector shared by the whole path. The paper says the forward process becomes deterministic given $x_{t-1}$ and $x_0$, "except for $t = 1$": $x_1$ given $x_0$ alone is still random, and that randomness is the shared $\epsilon$.
- **$\varsigma_t^2 = \tilde\beta_t$.** DDPM: fresh noise at every step, Markov forward chain.
- **$\varsigma_t^2 = 1-\bar\alpha_{t-1}$ (the maximum).** Then $k_t = 0$ and $x_{t-1}$ is independent of $x_t$ given $x_0$: each noisy image is an independent draw. For $t \ge 2$, $1-\bar\alpha_t = \beta_t + (1-\beta_t)(1-\bar\alpha_{t-1}) > \beta_t$ (because $\bar\alpha_{t-1} < 1$), so $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t < 1-\bar\alpha_{t-1}$. DDPM therefore sits strictly between the deterministic and the fully independent extremes.

::: qa
**Prove that the DDIM inference family has the same marginals as DDPM (Lemma 1).**

Induct downward from $T$, where $q_\varsigma(x_T \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_T}x_0, (1-\bar\alpha_T)I)$ by definition. Assume $x_t \mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$. Because the family is a chain from $x_T$ down, $q_\varsigma(x_{t-1} \mid x_0) = \int q_\varsigma(x_t \mid x_0)\,q_\varsigma(x_{t-1} \mid x_t, x_0)\,dx_t$, and the second factor is linear-Gaussian in $x_t$ with slope $k_t = \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}/\sqrt{1-\bar\alpha_t}$, offset $(\sqrt{\bar\alpha_{t-1}} - k_t\sqrt{\bar\alpha_t})x_0$ and variance $\varsigma_t^2$. Gaussian marginalization gives mean $k_t\sqrt{\bar\alpha_t}x_0 + \sqrt{\bar\alpha_{t-1}}x_0 - k_t\sqrt{\bar\alpha_t}x_0 = \sqrt{\bar\alpha_{t-1}}x_0$ and variance $\varsigma_t^2 + k_t^2(1-\bar\alpha_t) = 1-\bar\alpha_{t-1}$. That is DDPM's marginal at $t-1$, which completes the induction. The point to stress is that the mean of eq. (7) was designed for exactly this: re-adding the old noise direction with amplitude $\sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}$ and topping up with $\varsigma_t^2$ of fresh noise.
:::

::: qa
**Why is the DDIM forward process non-Markovian? Show it.**

Compute $q_\varsigma(x_t \mid x_{t-1}, x_0)$ by Gaussian conditioning. Given $x_0$, $x_t$ and $x_{t-1}$ have variances $1-\bar\alpha_t$ and $1-\bar\alpha_{t-1}$ and covariance $\sqrt{(1-\bar\alpha_t)(1-\bar\alpha_{t-1}-\varsigma_t^2)}$. Conditioning gives mean $\rho_t x_{t-1} + (\sqrt{\bar\alpha_t} - \rho_t\sqrt{\bar\alpha_{t-1}})x_0$ with $\rho_t = \sqrt{(1-\bar\alpha_t)(1-\bar\alpha_{t-1}-\varsigma_t^2)}/(1-\bar\alpha_{t-1})$, and variance $(1-\bar\alpha_t)\varsigma_t^2/(1-\bar\alpha_{t-1})$. The $x_0$ coefficient is zero only if $\varsigma_t^2 = \tilde\beta_t$, in which case the step is exactly $\mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$, DDPM. In the deterministic case the step rescales the noise already in $x_{t-1}$, and it needs $x_0$ to know what that noise is.
:::

::: qa
**Derive DDIM from scratch: how would you find the family of processes it uses?**

Start from the fact that $L_{\text{simple}}$ depends only on the marginals $q(x_t \mid x_0)$. Look for an inference process written backward, with Gaussian conditionals $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\nu x_0 + k x_t, \varsigma_t^2 I)$, and require that it keeps DDPM's marginals. Marginalizing $x_t \sim \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$ gives mean $(\nu + k\sqrt{\bar\alpha_t})x_0$ and variance $\varsigma_t^2 + k^2(1-\bar\alpha_t)$; setting these to $\sqrt{\bar\alpha_{t-1}}x_0$ and $1-\bar\alpha_{t-1}$ gives $k = \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}/\sqrt{1-\bar\alpha_t}$ and $\nu = \sqrt{\bar\alpha_{t-1}} - k\sqrt{\bar\alpha_t}$, leaving $\varsigma_t$ free. For sampling, replace $x_0$ by the network's $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\hat\epsilon)/\sqrt{\bar\alpha_t}$; then $(x_t - \sqrt{\bar\alpha_t}\hat x_0)/\sqrt{1-\bar\alpha_t} = \hat\epsilon$ and the step is $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}\,\hat\epsilon + \varsigma_t z$. Finally, justify reusing the DDPM network by showing that the ELBO of this process is a reweighted $L_{\text{simple}}$ (Theorem 1).
:::

## The generative process and Theorem 1 (Section 3.2)

### Predict $x_0$, then step with $q_\varsigma$

Given a noisy $x_t$, the network $\epsilon_\theta^{(t)}(x_t)$ predicts the noise. Solving $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ for $x_0$ with the predicted noise gives the paper's eq. (9), the "denoised observation":

$$
f_\theta^{(t)}(x_t) := \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}{\sqrt{\bar\alpha_t}} .
$$

The generative process (eq. 10) has prior $p_\theta(x_T) = \mathcal{N}(0, I)$ and steps

$$
p_\theta^{(t)}(x_{t-1} \mid x_t) = \begin{cases} \mathcal{N}\big(f_\theta^{(1)}(x_1),\ \varsigma_1^2 I\big) & t = 1,\\ q_\varsigma\big(x_{t-1} \mid x_t,\ f_\theta^{(t)}(x_t)\big) & t \ge 2 . \end{cases}
$$

In words: guess the clean image, then take one step of the inference process as if the guess were correct. The $t = 1$ case needs its own definition because eq. (7) only exists for $t \ge 2$ (the clean image is the conditioning variable, not a step of the chain). The paper adds Gaussian noise of variance $\varsigma_1^2$ there so that the generative process "is supported everywhere"; without it, $-\log p_\theta^{(1)}(x_0 \mid x_1)$ would be infinite for any $x_0$ other than the guess. In practice the last sampling step simply outputs $f_\theta^{(1)}(x_1)$. A footnote reports that learning a distribution over the $x_0$ prediction, instead of a point guess, brought little benefit.

### The variational objective $J_\varsigma$

For each $\varsigma$ the natural training objective is the negative ELBO with $q_\varsigma$ as the inference distribution (the Jensen step of Step 3 works for any $q$). With $q_\varsigma(x_{0:T}) = q(x_0)\,q_\varsigma(x_{1:T} \mid x_0)$:

$$
\begin{aligned}
J_\varsigma(\epsilon_\theta) &:= \mathbb{E}_{q_\varsigma(x_{0:T})}\big[\log q_\varsigma(x_{1:T} \mid x_0) - \log p_\theta(x_{0:T})\big]\\
&= \mathbb{E}_{q_\varsigma(x_{0:T})}\Big[\log q_\varsigma(x_T \mid x_0) + \sum_{t=2}^{T}\log q_\varsigma(x_{t-1} \mid x_t, x_0) - \sum_{t=1}^{T}\log p_\theta^{(t)}(x_{t-1} \mid x_t) - \log p_\theta(x_T)\Big].
\end{aligned}
$$

At first sight each $\varsigma$ would need its own model, since each defines a different objective and a different generative process. Theorem 1 says otherwise.

### Theorem 1 and its proof

**Theorem 1.** For every $\varsigma$ with all $\varsigma_t > 0$, there exist weights $\gamma \in \mathbb{R}^T_{>0}$ and a constant $C$ such that $J_\varsigma = L_\gamma + C$ as functionals of $\epsilon_\theta$.

**Proof.**

*(a) Group the terms.* Since $q_\varsigma$ is already written backward, no telescoping is needed:

$$
J_\varsigma = \underbrace{\mathbb{E}\big[\log q_\varsigma(x_T \mid x_0) - \log p_\theta(x_T)\big]}_{\text{no } \theta} + \sum_{t=2}^{T}\mathbb{E}\big[\log q_\varsigma(x_{t-1} \mid x_t, x_0) - \log p_\theta^{(t)}(x_{t-1} \mid x_t)\big] + \mathbb{E}\big[-\log p_\theta^{(1)}(x_0 \mid x_1)\big].
$$

The first term is the KL between $q(x_T \mid x_0)$ and $\mathcal{N}(0, I)$, which has no parameters. For the middle terms, the chain structure (the preliminary fact in Lemma 1's proof) says that under $q_\varsigma$, given $(x_0, x_t)$, the variable $x_{t-1}$ has law $q_\varsigma(x_{t-1} \mid x_t, x_0)$. So the inner expectation over $x_{t-1}$ is a KL, and term $t$ equals $\mathbb{E}_{q_\varsigma(x_0, x_t)}\,\mathrm{KL}\big(q_\varsigma(x_{t-1} \mid x_t, x_0)\,\|\,p_\theta^{(t)}(x_{t-1} \mid x_t)\big)$.

*(b) Where Lemma 1 enters.* The pair $(x_0, x_t)$ under $q_\varsigma$ has law $q(x_0)\,\mathcal{N}(x_t; \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$, the same as under DDPM. So we may write $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ inside every expectation. (The paper's appendix silently writes $q$ instead of $q_\varsigma$ here; this is why it may.)

*(c) KL with equal covariances.* Both distributions in the KL have covariance $\varsigma_t^2 I$, because $p_\theta^{(t)}$ is eq. (7) with $x_0$ replaced by $f_\theta^{(t)}(x_t)$. Write the mean of eq. (7) as a function of its $x_0$ argument, $m(u) = \sqrt{\bar\alpha_{t-1}}\,u + k_t(x_t - \sqrt{\bar\alpha_t}\,u)$. Tool 3 gives $\mathrm{KL} = \lVert m(x_0) - m(f_\theta^{(t)}(x_t))\rVert^2/(2\varsigma_t^2)$.

*(d) The mean difference.* $m$ is affine in $u$ with slope $\sqrt{\bar\alpha_{t-1}} - k_t\sqrt{\bar\alpha_t}$, the coefficient $\nu$ of "Where eq. (7) comes from" (the $k_t x_t$ term does not depend on $u$ and cancels), so

$$
m(x_0) - m\big(f_\theta^{(t)}(x_t)\big) = \big(\sqrt{\bar\alpha_{t-1}} - k_t\sqrt{\bar\alpha_t}\big)\big(x_0 - f_\theta^{(t)}(x_t)\big).
$$

*(e) Rewrite in noise units.* Using $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ and eq. (9),

$$
x_0 - f_\theta^{(t)}(x_t) = \frac{\big(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\big) - \big(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta^{(t)}(x_t)\big)}{\sqrt{\bar\alpha_t}} = \frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\big(\epsilon_\theta^{(t)}(x_t) - \epsilon\big).
$$

*(f) Combine and simplify the constant.* Multiply the slope from (d) by $\sqrt{1-\bar\alpha_t}/\sqrt{\bar\alpha_t}$ from (e), using $k_t\sqrt{1-\bar\alpha_t} = \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}$:

$$
\kappa_t := \big(\sqrt{\bar\alpha_{t-1}} - k_t\sqrt{\bar\alpha_t}\big)\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}} = \sqrt{\frac{\bar\alpha_{t-1}(1-\bar\alpha_t)}{\bar\alpha_t}} - \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2} .
$$

Combining (d) and (e), $m(x_0) - m(f_\theta^{(t)}(x_t)) = \kappa_t\big(\epsilon_\theta^{(t)}(x_t) - \epsilon\big)$, so by (c) the KL is $\kappa_t^2\lVert \epsilon - \epsilon_\theta^{(t)}(x_t)\rVert^2/(2\varsigma_t^2)$. For $t \ge 2$ the term is therefore $\gamma_t\,\mathbb{E}\lVert \epsilon - \epsilon_\theta^{(t)}(x_t)\rVert^2$ with $\gamma_t = \kappa_t^2/(2\varsigma_t^2)$.

*(g) The $t = 1$ term.* $-\log\mathcal{N}(x_0; f_\theta^{(1)}(x_1), \varsigma_1^2 I) = \lVert x_0 - f_\theta^{(1)}(x_1)\rVert^2/(2\varsigma_1^2) + \tfrac{d}{2}\log(2\pi\varsigma_1^2)$. By (e) with $t = 1$, this is $\gamma_1\,\lVert \epsilon - \epsilon_\theta^{(1)}(x_1)\rVert^2 + \text{const}$ with $\gamma_1 = (1-\bar\alpha_1)/(2\varsigma_1^2\bar\alpha_1)$.

*(h) Positivity.* $\gamma_1 > 0$ because $\bar\alpha_1 < 1$ and $\varsigma_1 > 0$. For $t \ge 2$ we need $\kappa_t \ne 0$. Since $\varsigma_t^2 \ge 0$, $\sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2} \le \sqrt{1-\bar\alpha_{t-1}}$, so it suffices that $\bar\alpha_{t-1}(1-\bar\alpha_t)/\bar\alpha_t > 1-\bar\alpha_{t-1}$. Multiply by $\bar\alpha_t > 0$: $\bar\alpha_{t-1} - \bar\alpha_{t-1}\bar\alpha_t > \bar\alpha_t - \bar\alpha_t\bar\alpha_{t-1}$, that is $\bar\alpha_{t-1} > \bar\alpha_t$, which holds whenever $\beta_t > 0$. So $\kappa_t > 0$, and $\varsigma_t > 0$ keeps $\gamma_t$ finite.

Adding up, $J_\varsigma = \sum_t \gamma_t\,\mathbb{E}\lVert \epsilon_\theta^{(t)}(x_t) - \epsilon\rVert^2 + C = L_\gamma + C$ with all $\gamma_t > 0$. $\blacksquare$

**A sanity check: $\eta = 1$ gives back DDPM's weight.** With $\varsigma_t^2 = \tilde\beta_t$:

$$
\begin{aligned}
1-\bar\alpha_{t-1}-\tilde\beta_t &= (1-\bar\alpha_{t-1})\Big(1 - \frac{\beta_t}{1-\bar\alpha_t}\Big) = \frac{(1-\bar\alpha_{t-1})(1-\bar\alpha_t-\beta_t)}{1-\bar\alpha_t} && \text{factor out } 1-\bar\alpha_{t-1}\\
&= \frac{(1-\beta_t)(1-\bar\alpha_{t-1})^2}{1-\bar\alpha_t} && 1-\bar\alpha_t-\beta_t = (1-\beta_t)(1-\bar\alpha_{t-1})\\
\kappa_t &= \frac{\sqrt{1-\bar\alpha_t}}{\sqrt{1-\beta_t}} - \frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{\sqrt{1-\bar\alpha_t}} && \tfrac{\bar\alpha_{t-1}}{\bar\alpha_t} = \tfrac{1}{1-\beta_t}\\
&= \frac{(1-\bar\alpha_t) - (1-\beta_t)(1-\bar\alpha_{t-1})}{\sqrt{1-\beta_t}\sqrt{1-\bar\alpha_t}} = \frac{\beta_t}{\sqrt{1-\beta_t}\sqrt{1-\bar\alpha_t}} && \text{common denominator.}
\end{aligned}
$$

So $\gamma_t = \kappa_t^2/(2\varsigma_t^2) = \beta_t^2/\big(2\varsigma_t^2(1-\beta_t)(1-\bar\alpha_t)\big)$, exactly the weight of Step 5. Theorem 1 contains DDPM's own derivation as a special case.

**A note on the paper's Appendix B.** The paper's proof ends with a different weight, $\gamma_t = 1/(2d\,\sigma_t^2\alpha_t)$ in its notation ($\sigma_t$ is $\varsigma_t$ here, $\alpha_t$ is $\bar\alpha_t$, and $d$ is the dimension of $x_0$). Two of its steps account for the difference:

- it writes each KL as $\lVert x_0 - f_\theta^{(t)}(x_t)\rVert^2/(2\sigma_t^2)$, which leaves out the squared slope $(\sqrt{\bar\alpha_{t-1}} - k_t\sqrt{\bar\alpha_t})^2$ of step (d);
- converting to noise units, it leaves out the factor $1-\bar\alpha_t$ of step (e) and divides by $d$.

Its $t = 1$ weight, $1/(2d\,\sigma_1^2\alpha_1)$, likewise lacks the factor $1-\bar\alpha_1$. The sanity check above favors the weights derived here: at $\eta = 1$ they reproduce DDPM's weight, which the paper itself writes correctly in its Appendix C.2, while $1/(2d\,\tilde\beta_t\bar\alpha_t)$ does not. The theorem only claims that some positive weights exist, so its conclusion is unaffected. The exact weights are $\gamma_t = \kappa_t^2/(2\varsigma_t^2)$ for $t \ge 2$ and $\gamma_1$ from step (g).

**What this says.** Training on $J_\varsigma$ for any $\varsigma$ is the same as training a weighted noise-prediction MSE. The weights depend on $\varsigma$, but the regression targets do not.

### Why the weights do not matter, and the "no sharing" condition

If each $\epsilon_\theta^{(t)}$ has its own parameters $\theta^{(t)}$, then $L_\gamma = \sum_t \gamma_t\,\ell_t(\theta^{(t)})$, where $\ell_t$ is the expected squared error at step $t$, is a sum of $T$ separate problems. A positive weight does not change the minimizer of its own term, so the minimizer of $L_\gamma$ is the same for every $\gamma > 0$, and equals the minimizer of $L_{\mathbf{1}}$. By Tool 4, that minimizer is $\epsilon_\theta^{(t)}(x_t) = \mathbb{E}[\epsilon \mid x_t]$.

The paper draws two consequences. First, this justifies DDPM's $L_{\mathbf{1}}$ as a surrogate for its own variational bound. Second, since $J_\varsigma = L_\gamma + C$, the $L_{\mathbf{1}}$ solution also minimizes $J_\varsigma$, for every $\varsigma$.

Real models share one network across all $t$, with a time embedding. The argument still holds at the global optimum if the network can represent all the per-$t$ minimizers at once, because then one $\theta$ minimizes every term simultaneously, whatever the weights. With limited capacity or finite training the weights do decide which errors are tolerated, so "the same optimum" is an idealization. It is a good one in practice: the paper's experiments use the DDPM checkpoints unchanged.

### What the theorem does not cover

The deterministic case $\varsigma_t = 0$ is excluded: the conditionals become point masses, the KL is undefined, and $\gamma_t \to \infty$. The paper's footnote says this case can be approximated by making $\varsigma_t$ very small. The practical message survives: $\varsigma$ is a sampling-time choice, not a training-time one.

::: key
**How to say it in an interview.** "For each choice of $\varsigma$, write the ELBO of the new process. Each term is a KL between two Gaussians with the same covariance $\varsigma_t^2 I$, one centered using the true $x_0$ and one using the predicted $\hat x_0$, so it is a squared distance between means. The means are affine in $x_0$, and $x_0 - \hat x_0$ is proportional to $\epsilon - \hat\epsilon$, so every term is a positively weighted noise-prediction MSE: $J_\varsigma = L_\gamma + C$. If parameters are not shared across $t$, positive weights don't change the minimizer, so the model trained with DDPM's $L_{\text{simple}}$ is optimal for every $\varsigma$, and $\varsigma$ can be picked at sampling time."
:::

::: qa
**Why can a network trained as a DDPM be used, unchanged, with the DDIM sampler?**

Theorem 1: for each noise choice $\varsigma > 0$, the variational objective $J_\varsigma$ of the corresponding generative process equals $L_\gamma + C$ for some positive weights $\gamma$. The proof uses Lemma 1 (the pairs $(x_0, x_t)$ have DDPM's law), then Tool 3 (equal-covariance Gaussian KL is a squared mean difference), then the fact that the mean difference is proportional to $x_0 - \hat x_0$, which is proportional to $\epsilon - \hat\epsilon$. With separate parameters per $t$, the minimizer of a positively weighted sum of separate MSEs is the per-$t$ conditional mean $\mathbb{E}[\epsilon \mid x_t]$, independent of the weights, so it coincides with the $L_{\text{simple}}$ solution. Hence one trained network is optimal for the whole family. Two caveats worth mentioning: the deterministic case is only a limit ($\varsigma \to 0$), and with a shared network the argument is exact only at a global optimum the network can reach.
:::

::: qa
**What does "parameters not shared across $t$" buy in Theorem 1's argument, and does it hold in practice?**

It makes $L_\gamma$ a sum of independent problems, one per time step, so reweighting cannot move the optimum: each term is minimized separately and its minimizer, $\mathbb{E}[\epsilon \mid x_t]$, does not care about $\gamma_t$. In practice one U-Net or transformer is shared across all $t$ via a time embedding. If the network is expressive enough to hit $\mathbb{E}[\epsilon \mid x_t]$ at every $t$, a single parameter vector still minimizes all terms at once and the conclusion holds. With finite capacity and training the weights change which time steps get fitted best, so the claim that DDPM's model is optimal for every $J_\varsigma$ is approximate. Empirically it works well enough that the paper uses DDPM's own checkpoints.
:::

::: qa
**Is the deterministic sampler ($\eta = 0$) covered by Theorem 1?**

Strictly, no. The theorem assumes every $\varsigma_t > 0$. At $\varsigma_t = 0$ both conditionals in each KL are point masses, the KL is infinite unless the means coincide exactly, and the weight $\gamma_t = \kappa_t^2/(2\varsigma_t^2)$ diverges. The paper notes this in a footnote and argues the case is approached by letting $\varsigma_t$ be very small, since the minimizer $\mathbb{E}[\epsilon \mid x_t]$ does not depend on $\varsigma$. A cleaner justification is the ODE view in the paper's own Section 4.3: with the optimal $\hat\epsilon$, deterministic DDIM is a discretization of the probability-flow ODE (Proposition 1), and that ODE has the same marginals as the diffusion (Song et al. 2021, concurrent work).
:::

### A discrete version (Appendix A)

The paper notes that the non-Markovian construction is not tied to Gaussians. For one-hot categorical data with $K$ classes, let $u = (1/K, \dots, 1/K)$ and define $q(x_t \mid x_0) = \mathrm{Cat}(\bar\alpha_t x_0 + (1-\bar\alpha_t)u)$. The backward conditional is a mixture: copy $x_t$ with probability $\varsigma_t$, copy $x_0$ with probability $\bar\alpha_{t-1} - \varsigma_t\bar\alpha_t$, otherwise draw uniformly. (Here $\varsigma_t$ is a probability, not a standard deviation.) Its consistency is a two-line check: averaging over $x_t$ replaces $x_t$ by its mean $\bar\alpha_t x_0 + (1-\bar\alpha_t)u$, giving

$$
\varsigma_t\big(\bar\alpha_t x_0 + (1-\bar\alpha_t)u\big) + (\bar\alpha_{t-1} - \varsigma_t\bar\alpha_t)x_0 + \big((1-\bar\alpha_{t-1}) - (1-\bar\alpha_t)\varsigma_t\big)u = \bar\alpha_{t-1}x_0 + (1-\bar\alpha_{t-1})u ,
$$

which is the right marginal at $t-1$. The paper leaves experiments on this case to future work.

## The DDIM update (Section 4.1)

### Deriving the three-term update

To sample $x_{t-1} \sim p_\theta^{(t)}(x_{t-1} \mid x_t) = q_\varsigma(x_{t-1} \mid x_t, \hat x_0)$ with $\hat x_0 = f_\theta^{(t)}(x_t)$, reparameterize eq. (7) with $z \sim \mathcal{N}(0, I)$:

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}\cdot\frac{x_t - \sqrt{\bar\alpha_t}\,\hat x_0}{\sqrt{1-\bar\alpha_t}} + \varsigma_t z .
$$

From eq. (9), $\sqrt{\bar\alpha_t}\,\hat x_0 = x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon$, so $x_t - \sqrt{\bar\alpha_t}\,\hat x_0 = \sqrt{1-\bar\alpha_t}\,\hat\epsilon$ and the fraction is just $\hat\epsilon$. This gives the paper's eq. (12), with the convention $\bar\alpha_0 := 1$:

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\Big(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}{\sqrt{\bar\alpha_t}}\Big)}_{\text{predicted } x_0} + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}\;\epsilon_\theta^{(t)}(x_t)}_{\text{direction pointing to } x_t} + \underbrace{\varsigma_t z}_{\text{random noise}} .
$$

The three terms, in the paper's labels:

1. **Predicted $x_0$**, scaled to the signal level of step $t-1$.
2. **Direction pointing to $x_t$.** $\hat\epsilon$ is the unit-variance noise direction that, according to the model, leads from $\hat x_0$ to $x_t$. Re-adding it keeps $x_{t-1}$ on the same side of $\hat x_0$ as $x_t$, with less noise.
3. **Random noise** that supplies the rest of the variance budget. Terms 2 and 3 have variances $(1-\bar\alpha_{t-1}-\varsigma_t^2) + \varsigma_t^2 = 1-\bar\alpha_{t-1}$, the noise level step $t-1$ should have.

Since $\bar\alpha_0 = 1$, the last step ($t = 1$ to $0$) contains $\sqrt{1-\bar\alpha_0-\varsigma_1^2} = \sqrt{-\varsigma_1^2}$, which is real only for $\varsigma_1 = 0$. So the last step of the sampler adds no noise and simply outputs $\hat x_0$. (The noise $\varsigma_1 > 0$ in the definition of $p_\theta^{(1)}$ above is there for the objective, not for sampling; the $\eta$ formula below gives $\varsigma_1 = 0$ automatically.)

A minimal deterministic sampler, with `abar[0] = 1` and `abar[t]` $= \bar\alpha_t$:

```python
def ddim_step(x, t, s, eps_model, abar):        # eta = 0, from step t to step s < t
    eps = eps_model(x, t)
    x0 = (x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
    return abar[s].sqrt() * x0 + (1 - abar[s]).sqrt() * eps

@torch.no_grad()
def ddim_sample(eps_model, abar, taus, shape):  # taus: increasing list, e.g. [1, 101, ..., 901]
    x = torch.randn(shape)
    for t, s in zip(reversed(taus), reversed([0] + taus[:-1])):
        x = ddim_step(x, t, s, eps_model, abar)
    return x                                      # the last step returns x0-hat
```

### The noise scale $\varsigma_t(\eta)$

The experiments parameterize the noise by one number $\eta \ge 0$ (the paper's eq. 16, written here for consecutive steps):

$$
\varsigma_t(\eta) = \eta\,\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\,\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}} .
$$

**$\eta = 1$ is DDPM.** Since $1 - \bar\alpha_t/\bar\alpha_{t-1} = \beta_t$, $\varsigma_t(1)^2 = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t = \tilde\beta_t$. The forward process is then Markov (previous section), and the sampling step is DDPM's ancestral step with variance $\tilde\beta_t$. To check the mean, use $\sqrt{1-\bar\alpha_{t-1}-\tilde\beta_t} = \sqrt{1-\beta_t}(1-\bar\alpha_{t-1})/\sqrt{1-\bar\alpha_t}$ (from the sanity check above) and substitute $\hat\epsilon = (x_t - \sqrt{\bar\alpha_t}\hat x_0)/\sqrt{1-\bar\alpha_t}$:

$$
\begin{aligned}
\sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \frac{\sqrt{1-\beta_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\big(x_t - \sqrt{\bar\alpha_t}\,\hat x_0\big) &= \sqrt{\bar\alpha_{t-1}}\Big[1 - \frac{(1-\beta_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\Big]\hat x_0 + \frac{\sqrt{1-\beta_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t && \sqrt{1-\beta_t}\sqrt{\bar\alpha_t} = (1-\beta_t)\sqrt{\bar\alpha_{t-1}}\\
&= \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\hat x_0 + \frac{\sqrt{1-\beta_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t && (1-\beta_t)(1-\bar\alpha_{t-1}) = 1-\beta_t-\bar\alpha_t ,
\end{aligned}
$$

where the second line puts the bracket over $1-\bar\alpha_t$: its numerator is $(1-\bar\alpha_t) - (1-\beta_t-\bar\alpha_t) = \beta_t$. The result is $\tilde\mu_t(x_t, \hat x_0)$, and by Step 5 that equals DDPM's $\mu_\theta(x_t, t)$.

**$\eta = 0$ is DDIM.** No noise is injected, so $x_0$ is a deterministic function of $x_T$. A model that generates samples by pushing a latent through a fixed procedure is an implicit probabilistic model (Mohamed and Lakshminarayanan), like a GAN generator. The paper names it the denoising diffusion implicit model, an implicit model trained with the DDPM objective "despite the forward process no longer being a diffusion".

**$0 < \eta < 1$** interpolates. Values above one are allowed while $\varsigma_t^2 \le 1-\bar\alpha_{t-1}$, but the paper reports $\eta \in \{0, 0.2, 0.5, 1\}$.

### The larger-variance variant $\hat\varsigma$

DDPM offered two fixed reverse variances, $\tilde\beta_t$ and $\beta_t$. The DDIM paper calls the second one $\hat\sigma$, here $\hat\varsigma_t = \sqrt{1 - \bar\alpha_t/\bar\alpha_{t-1}}$ ($= \sqrt{\beta_t}$ for consecutive steps), and notes that Ho et al.'s implementation used it only to produce their CIFAR-10 samples.

This choice is not a member of the family: plugging $\hat\varsigma_t^2 = \beta_t$ into eq. (12) would need $\sqrt{1-\bar\alpha_{t-1}-\beta_t}$, which is imaginary at $t = 2$ for an increasing schedule ($1 - \bar\alpha_1 - \beta_2 = \beta_1 - \beta_2 < 0$). So the paper's Appendix D.3 keeps the deterministic part of the $\eta = 1$ update and only swaps the noise:

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\varsigma_t(1)^2}\;\hat\epsilon + \hat\varsigma_t z .
$$

Even with perfect predictions ($\hat x_0 = x_0$, $\hat\epsilon = \epsilon$), the noise variance of $x_{t-1}$ is then $(1-\bar\alpha_{t-1}) + (\hat\varsigma_t^2 - \varsigma_t(1)^2)$, more than the next network call is told to expect. The excess is

$$
\hat\varsigma_t^2 - \varsigma_t(1)^2 = \hat\varsigma_t^2\Big(1 - \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\Big) = \hat\varsigma_t^2\,\frac{\bar\alpha_{t-1} - \bar\alpha_t}{1-\bar\alpha_t} .
$$

With all 1000 steps, $\hat\varsigma_t^2 = \beta_t$ and $\bar\alpha_{t-1} - \bar\alpha_t = \beta_t\bar\alpha_{t-1}$, so the excess is $\beta_t^2\bar\alpha_{t-1}/(1-\bar\alpha_t)$. Under the linear schedule its largest value is $6.5 \times 10^{-5}$ in variance, a standard deviation of $0.008$, at $t = 2$, and it is smaller at every later step (my computation). That is small in absolute terms, and with 1000 steps this sampler has the best FID in the paper's Table 1 (3.17 on CIFAR-10).

With large jumps the excess is not small. The same formula holds for a jump between any two levels, with $\bar\alpha_t$ and $\bar\alpha_{t-1}$ replaced by the two levels' values. A worked number (my computation, linear schedule, the uniform 10-step grid of the next sections): on the step from $t = 101$ to $t = 1$, the next state should carry noise of standard deviation $\sqrt{1-\bar\alpha_1} = 0.010$, but receives about $0.32$ (here $\hat\varsigma = \sqrt{1 - \bar\alpha_{101}/\bar\alpha_1} = 0.324$). The final network call is asked to remove noise of size 0.01 and leaves most of the 0.32 in the image. The paper describes the result as samples with "more noisy perturbations" and attributes the poor FID to this extra stochasticity, noting FID's sensitivity to such noise.

::: key
**How to say it in an interview.** "One DDIM step: predict $\hat\epsilon$, form $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\hat\epsilon)/\sqrt{\bar\alpha_t}$, then $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}\,\hat\epsilon + \varsigma_t z$. It re-noises the clean guess to the next level, reusing the predicted noise direction for part of the noise and fresh noise for the rest. With $\varsigma_t^2 = \tilde\beta_t$ ($\eta = 1$) this is exactly DDPM's ancestral step; with $\varsigma = 0$ it is deterministic, which is DDIM. So DDPM sampling (with the posterior variance) is the $\eta = 1$ member of the DDIM family."
:::

::: qa
**Explain the three terms of the DDIM update and where each comes from.**

The update is eq. (7), the inference process's backward step, with the true $x_0$ replaced by the network's estimate $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\hat\epsilon)/\sqrt{\bar\alpha_t}$. The first term, $\sqrt{\bar\alpha_{t-1}}\hat x_0$, is the predicted clean image scaled to the signal level of step $t-1$. The second, $\sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}\,\hat\epsilon$, appears because eq. (7) re-adds the noise direction that led from $x_0$ to $x_t$, and that direction, evaluated at $\hat x_0$, is exactly $\hat\epsilon$; the paper calls it the "direction pointing to $x_t$". The third, $\varsigma_t z$, is fresh noise. The two noise terms add up to variance $1-\bar\alpha_{t-1}$, which is what the marginal at $t-1$ requires, so the next network call sees an input at the noise level it was trained on.
:::

::: qa
**What does $\eta$ control, and what are $\eta = 0$ and $\eta = 1$?**

$\eta$ scales the fresh noise in each step: $\varsigma_t(\eta) = \eta\sqrt{(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)}\sqrt{1-\bar\alpha_t/\bar\alpha_{t-1}}$, while the predicted-noise term shrinks to compensate. At $\eta = 1$, $\varsigma_t^2 = \tilde\beta_t$, the forward process is DDPM's Markov chain and the sampler is DDPM's ancestral sampler with the posterior variance. At $\eta = 0$ the sampler is deterministic: $x_0$ is a function of $x_T$ alone, which makes it an implicit model and gives consistency, interpolation and approximate inversion. Intermediate values interpolate in quality and stochasticity. In Table 1 (CIFAR-10 and CelebA), FID gets monotonically worse as $\eta$ grows at every step count below 1000. That is not universal: on LSUN Church at 100 steps, $\eta = 1$ beats $\eta = 0$ (8.27 vs 10.58).
:::

::: {.qa asked="Tencent Youtu, Baidu, Alibaba, Xiaomi +3 (7 reports)"}
**DDPM vs DDIM: what is the difference, why can DDIM drop the Markov assumption and still be fast with good quality, and mathematically which one is the superset?**

Training is identical: DDIM reuses a DDPM-trained noise predictor. The difference is the inference process and therefore the sampler. DDPM's forward process is a Markov chain; DDIM replaces it with a family $q_\varsigma$ defined backward, whose conditionals $q_\varsigma(x_{t-1} \mid x_t, x_0)$ are Gaussian with mean $\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}-\varsigma_t^2}\,\frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}$ and variance $\varsigma_t^2$. It can drop the Markov assumption because the training loss only depends on the marginals $q(x_t \mid x_0)$, which this family preserves exactly (Lemma 1), and its ELBO is a reweighted $L_{\text{simple}}$ (Theorem 1). It is faster because the same argument works on any subsequence of time steps, and with $\varsigma = 0$ the sampler is a first-order solver of the probability-flow ODE, which tolerates large steps far better than ancestral sampling: CIFAR-10 FID 13.36 vs 41.07 at 10 steps, 4.16 vs 5.78 at 100. As families of generative processes, DDIM's $\eta$-family is the superset: $\eta = 1$ (variance $\tilde\beta_t$) is exactly DDPM, and the forward process is Markov only at that value. One nuance: DDPM's other variance choice, $\beta_t$, is not inside the family; the paper treats it as a separate variant $\hat\sigma$.
:::

## Two toy examples you can check by hand

The schedules below are made up so that $(\sqrt{\bar\alpha}, \sqrt{1-\bar\alpha})$ are easy pairs: $(0.28, 0.96)$, $(0.8, 0.6)$, $(0.96, 0.28)$ and $(1, 0)$ for the clean level. Write $a = \sqrt{\bar\alpha_t}$ and $s = \sqrt{1-\bar\alpha_t}$ for the current level and $(a', s')$ for the next one. The starting point $x_T = 0.5$ at level $(0.28, 0.96)$ is fixed, and the network is replaced by the exact optimal predictor, so every difference below comes from the sampler.

Each step jumps straight from one listed level to the next, using eq. (12) with $\bar\alpha_t$ and $\bar\alpha_{t-1}$ replaced by the two levels' values; "Skipping steps" below shows this is legitimate. With $\hat x_0$ and $\hat\epsilon$ computed at the current level, one step is

$$
x' = a'\,\hat x_0 + \sqrt{s'^2 - \varsigma^2}\;\hat\epsilon + \varsigma z, \qquad \varsigma = 0 \ \ (\eta = 0), \qquad \varsigma^2 = \frac{s'^2}{s^2}\Big(1 - \frac{a^2}{a'^2}\Big) \ \ (\eta = 1),
$$

where the $\eta = 1$ value is $\varsigma_t(1)^2$ from the previous section written with these symbols.

### Toy 1: the data is a single point

Let the data be $x_0 = c$ with probability one (one dimension). Given $x_t$, the noise is determined exactly: $\epsilon = (x_t - a c)/s$. By Tool 4 the optimal predictor is

$$
\epsilon^*(x_t) = \frac{x_t - a\,c}{s}, \qquad f^*(x_t) = \frac{x_t - s\,\epsilon^*(x_t)}{a} = c \quad\text{for every } x_t \text{ and } t .
$$

**DDIM ($\eta = 0$).** Take $c = 2$, start at level $(0.28, 0.96)$ with $x_T = 0.5$. Then $\epsilon^* = (0.5 - 0.56)/0.96 = -0.0625$ and $\hat x_0 = (0.5 + 0.96 \times 0.0625)/0.28 = 0.56/0.28 = 2$. One step straight to the clean level outputs $1 \times 2 + 0 = 2$: exact in one step. Stopping at an intermediate level $(a', s')$ instead gives $x' = 2a' + s'(-0.0625)$, whose predicted noise is again $(x' - 2a')/s' = -0.0625$. The predicted noise is constant along the deterministic path, which is the $\varsigma = 0$ forward path of the previous section run backward.

**DDPM ($\eta = 1$), two steps via $(0.8, 0.6)$.** For the first step, $s^2 = 0.9216$, $a^2 = 0.0784$, $s'^2 = 0.36$, $a'^2 = 0.64$, so $\varsigma^2 = \frac{0.36}{0.9216}\big(1 - \frac{0.0784}{0.64}\big) = 0.390625 \times 0.8775 = 0.3428$, so $\varsigma = 0.5855$, and the predicted-noise coefficient is $\sqrt{0.36 - 0.3428} = 0.1313$. The intermediate state is $x_{\text{mid}} = 0.8 \times 2 + 0.1313 \times (-0.0625) + 0.5855\,z = 1.592 + 0.5855\,z$, which is random. On the final step $\varsigma = 0$ (because $1 - \bar\alpha_0 = 0$) and the output is $f^*(x_{\text{mid}}) = 2$ again.

**What this says.** With a perfect model of a point mass, every sampler is exact; stochastic ones merely take random paths. Differences between samplers come from uncertainty about $x_0$, which the next toy adds.

### Toy 2: the data is two points

Let $x_0 = +1$ or $-1$ with probability $\tfrac12$ each. By Bayes,

$$
\frac{P(x_0 = +1 \mid x_t)}{P(x_0 = -1 \mid x_t)} = \frac{\exp\!\big(-(x_t - a)^2/(2s^2)\big)}{\exp\!\big(-(x_t + a)^2/(2s^2)\big)} = \exp\!\Big(\frac{(x_t + a)^2 - (x_t - a)^2}{2s^2}\Big) = \exp\!\Big(\frac{2a\,x_t}{s^2}\Big),
$$

so $P(+1 \mid x_t) = 1/(1 + e^{-2a x_t/s^2})$ and

$$
\hat x_0(x_t) = \mathbb{E}[x_0 \mid x_t] = P(+1 \mid x_t) - P(-1 \mid x_t) = \tanh\!\Big(\frac{a\,x_t}{s^2}\Big), \qquad \hat\epsilon(x_t) = \frac{x_t - a\,\hat x_0(x_t)}{s},
$$

using $(1 - e^{-2u})/(1 + e^{-2u}) = \tanh u$ and linearity of conditional expectation ($\epsilon = (x_t - a x_0)/s$).

**DDIM, one step from $(0.28, 0.96)$.** $\hat x_0 = \tanh(0.28 \times 0.5/0.9216) = \tanh(0.1519) = 0.151$. The output is $0.151$, the posterior mean: a blend of the two data points, not a valid sample.

**DDIM, two steps via $(0.8, 0.6)$.** Step 1: $\hat x_0 = 0.1508$, $\hat\epsilon = (0.5 - 0.28 \times 0.1508)/0.96 = 0.4769$, so $x_{\text{mid}} = 0.8 \times 0.1508 + 0.6 \times 0.4769 = 0.4067$. Step 2: $\hat x_0 = \tanh(0.8 \times 0.4067/0.36) = \tanh(0.904) = 0.718$. Output $0.718$.

**DDIM, three steps via $(0.8, 0.6)$ and $(0.96, 0.28)$.** From $x_{\text{mid}} = 0.4067$: $\hat x_0 = 0.718$, $\hat\epsilon = (0.4067 - 0.8 \times 0.718)/0.6 = -0.280$, so the next state is $0.96 \times 0.718 + 0.28 \times (-0.280) = 0.611$. Last step: $\hat x_0 = \tanh(0.96 \times 0.611/0.0784) = \tanh(7.48) = 0.999999$, and $\hat\epsilon = (0.611 - 0.96 \times 1.000)/0.28 = -1.25$. Output $1.000$.

**DDPM ($\eta = 1$), two steps via $(0.8, 0.6)$.** The coefficients are as in Toy 1: $\varsigma = 0.5855$ and $0.1313$. So $x_{\text{mid}} = 0.8 \times 0.1508 + 0.1313 \times 0.4769 + 0.5855\,z = 0.183 + 0.5855\,z$. With $z = +1$: $x_{\text{mid}} = 0.769$ and the output is $\tanh(0.8 \times 0.769/0.36) = \tanh(1.708) = 0.936$. With $z = -1$: $x_{\text{mid}} = -0.402$ and the output is $\tanh(-0.894) = -0.713$. The same $x_T$ lands on either side, depending on the injected noise.

**What this says.**

- One large step returns the posterior mean, an average of modes. Few-step error is mostly about committing to a mode, not about leftover noise; in images, the average shows up as blur.
- More deterministic steps let the posterior sharpen gradually: $0.151 \to 0.718 \to 1.000$. Note that $\hat\epsilon$ changes along this path ($0.477 \to -0.280 \to -1.25$), unlike Toy 1; the ODE section explains why that change is what costs accuracy.
- With $\eta = 0$ the endpoint is a function of $x_T$. Here each step is $x' = a'\hat x_0 + s'(x - a\hat x_0)/s = (s'/s)\,x + (a' - a s'/s)\tanh(a x/s^2)$, and both coefficients are positive when moving to lower noise ($a' > a$, $s' < s$). So each step is an odd, increasing function of $x$, every $x_T > 0$ ends on the positive side (at $+1$ with three steps), and the order of starting points is preserved (the discrete version of the fact that ODE trajectories in one dimension cannot cross). With $\eta = 1$ the endpoint is not a function of $x_T$. Consistency, interpolation and inversion all rest on this.

::: qa
**Suppose the denoiser is perfect. Why is one DDIM step still not enough?**

A perfect $\epsilon$-predictor gives $\hat x_0 = \mathbb{E}[x_0 \mid x_t]$, the posterior mean. One step from pure noise outputs exactly that, and at high noise the posterior covers many different images, so the output is their average: blurry in images, or a point between the modes for two-point data, where one step from $x_T = 0.5$ gives $0.151$ instead of $\pm 1$. Several steps help because each moves only part of the way, then re-asks the model at a lower noise level, where the posterior is narrower. In the toy, two steps reach $0.718$ and three reach $1.000$. In ODE terms, one Euler step is accurate only if $\hat\epsilon$ stays roughly constant along the trajectory, which holds for a single data point but not for multimodal data. This is one reason one-step generators are usually built by distillation or adversarial training rather than by a better ODE solver alone.
:::

## Skipping steps (Section 4.2 and Appendix C.1)

### The subsequence

Pick an increasing subsequence $\tau = (\tau_1 < \tau_2 < \dots < \tau_S)$ of $\{1, \dots, T\}$ and set $\tau_0 := 0$, with $\bar\alpha_0 = 1$. Sampling visits $x_{\tau_S}, x_{\tau_{S-1}}, \dots, x_{\tau_1}, x_0$, which the paper calls the sampling trajectory. Each step is eq. (12) with $t$ replaced by $\tau_i$ and $t-1$ by $\tau_{i-1}$:

$$
x_{\tau_{i-1}} = \sqrt{\bar\alpha_{\tau_{i-1}}}\,\hat x_0 + \sqrt{1-\bar\alpha_{\tau_{i-1}}-\varsigma_{\tau_i}^2}\;\epsilon_\theta^{(\tau_i)}(x_{\tau_i}) + \varsigma_{\tau_i} z, \qquad \varsigma_{\tau_i}(\eta) = \eta\sqrt{\frac{1-\bar\alpha_{\tau_{i-1}}}{1-\bar\alpha_{\tau_i}}}\sqrt{1 - \frac{\bar\alpha_{\tau_i}}{\bar\alpha_{\tau_{i-1}}}} .
$$

Only $S$ network calls are made. The paper's Figure 2 illustrates $\tau = [1, 3]$.

### Why the derivation still holds

Appendix C.1 defines an inference distribution over all of $x_{1:T}$ that treats the visited and unvisited steps differently. It assumes $\tau_S = T$, and its display starts the middle product at $i = 1$; since $x_0$ is not one of the latent variables, $i = 2$ is what is meant:

$$
q_{\varsigma,\tau}(x_{1:T} \mid x_0) = q_{\varsigma,\tau}(x_{\tau_S} \mid x_0)\prod_{i=2}^{S} q_{\varsigma,\tau}(x_{\tau_{i-1}} \mid x_{\tau_i}, x_0)\prod_{t \notin \tau} q_{\varsigma,\tau}(x_t \mid x_0) .
$$

- The visited variables form a chain whose backward conditionals are eq. (7) with $(t, t-1)$ replaced by $(\tau_i, \tau_{i-1})$.
- Each unvisited variable hangs directly off $x_0$ ("star graph") with DDPM's marginal $\mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$.

**Marginals.** Lemma 1's proof used only two things: the form of the backward conditional and the marginal one level up. Neither cares whether the two levels are adjacent integers. Repeating it with $\bar\alpha_{\tau_i}$ and $\bar\alpha_{\tau_{i-1}}$, the mean becomes $\sqrt{\bar\alpha_{\tau_{i-1}}}x_0$ (the $k$ terms cancel) and the variance $\varsigma_{\tau_i}^2 + (1-\bar\alpha_{\tau_{i-1}}-\varsigma_{\tau_i}^2) = 1-\bar\alpha_{\tau_{i-1}}$. Unvisited variables have the right marginals by definition. So every $q_{\varsigma,\tau}(x_t \mid x_0)$ is DDPM's.

**Generative process.** It mirrors the inference process:

$$
p_\theta(x_{0:T}) = \underbrace{p_\theta(x_T)\prod_{i=1}^{S} p_\theta^{(\tau_i)}(x_{\tau_{i-1}} \mid x_{\tau_i})}_{\text{used to produce samples}} \times \underbrace{\prod_{t \notin \tau} p_\theta^{(t)}(x_0 \mid x_t)}_{\text{only in the objective}},
$$

where the sampling factors are $q_{\varsigma,\tau}(x_{\tau_{i-1}} \mid x_{\tau_i}, \hat x_0)$ for $i > 1$, and the final step and the extra factors are $\mathcal{N}(f_\theta^{(t)}(x_t), \varsigma_t^2 I)$.

**Objective.** The chain terms are KLs between Gaussians with $\theta$-free covariances, handled exactly as in Theorem 1 steps (c) to (f) with $\tau$'s in place of $t$'s. The extra factors contribute $\mathbb{E}[-\log p_\theta^{(t)}(x_0 \mid x_t)]$, which is step (g) with $t$ in place of $1$: a weight $(1-\bar\alpha_t)/(2\varsigma_t^2\bar\alpha_t)$ times $\lVert \epsilon - \epsilon_\theta^{(t)}\rVert^2$. So the objective is again $L_\gamma + C$ with every $t \in \{1, \dots, T\}$ present and all weights positive.

The star-graph factors are a device: they put every training time step into the objective, so the same $L_{\text{simple}}$ minimizer is optimal for every subsequence at once. The paper adds that, in principle, one can train with many more steps, or continuous $t$, and sample from only a few; continuous-time models now do exactly that.

**What this says.** Nothing in the construction requires consecutive steps. A jump from $\tau_i$ to $\tau_{i-1}$ is just another backward conditional with the right marginals, and the same network serves it. The approximation is in the sampler, not in the training: with large jumps, $\hat x_0$ is a poor stand-in for $x_0$ (Toy 2), so quality drops gracefully as $S$ shrinks.

::: key
**How to say it in an interview.** "Lemma 1's induction only uses the marginal at the current level and the form of the backward conditional, never that the two levels are adjacent. So I can define the same non-Markovian chain on any subsequence $\tau$ of time steps, attach the skipped steps directly to $x_0$ so they still appear in the objective, and Theorem 1 again gives a reweighted $L_{\text{simple}}$. The DDPM network is therefore valid for any subsequence, and the update is the usual DDIM step with $\bar\alpha_{\tau_{i-1}}$ in place of $\bar\alpha_{t-1}$. Quality loss comes from large jumps making $\hat x_0$ a crude guess, not from a mismatch with training."
:::

### Choosing $\tau$: linear and quadratic

Appendix D.2 uses two rules, with a constant $c$ chosen so that the last element is close to $T$:

- **Linear:** $\tau_i = \lfloor c\,i \rfloor$.
- **Quadratic:** $\tau_i = \lfloor c\,i^2 \rfloor$.

The paper uses quadratic for CIFAR-10 and linear for CelebA and LSUN, saying each gave slightly better FID than the alternative on its dataset.

The official code (`runners/diffusion.py`) implements them as `skip_type = "uniform"`, `range(0, T, T // S)`, and `"quad"`, `(np.linspace(0, np.sqrt(0.8 * T), S) ** 2)` cast to integers. So in the code the quadratic grid ends near $0.8\,T$, not near $T$ as the appendix's wording suggests. The code counts time steps from 0, and its step $i$ uses $\bar\alpha_{i+1}$, so paper time is code index plus one. For $S = 10$ and $T = 1000$, in paper indexing:

| Rule | $\tau_1, \dots, \tau_{10}$ |
|---|---|
| linear ("uniform") | 1, 101, 201, 301, 401, 501, 601, 701, 801, 901 |
| quadratic ("quad") | 1, 10, 40, 89, 159, 247, 356, 484, 633, 801 |

Sampling runs right to left and ends with the step from $\tau_1 = 1$ to $x_0$. Three things to notice:

- **Quadratic spends its steps at low noise.** Seven of its ten steps are below $t = 400$, where fine detail is resolved; linear spends equal time everywhere.
- **Neither starts at $T$,** although Appendix C.1's construction assumes $\tau_S = T$. The first network call is at $t = 901$ (linear) or $t = 801$ (quadratic), while the input is pure noise. With the linear $\beta$ schedule, $\sqrt{\bar\alpha_{901}} = 0.016$ and $\sqrt{\bar\alpha_{801}} = 0.039$, so the model is told to expect a little signal that is not there. The effect is mild with this schedule, and it matters more for schedules that keep more signal (see "Timestep spacing" below).
- **The last step is from $t = 1$**, where $\sqrt{\bar\alpha_1} = 0.99995$ and the noise standard deviation is $0.010$: almost a no-op.

::: qa
**Why can DDIM skip steps?**

Because neither the training objective nor the sampler's derivation needs consecutive steps. The loss uses only the marginals $q(x_t \mid x_0)$. The DDIM backward conditional between any two levels $\tau_i > \tau_{i-1}$, with $\bar\alpha_{\tau_i}$ and $\bar\alpha_{\tau_{i-1}}$ in place of $\bar\alpha_t$ and $\bar\alpha_{t-1}$, preserves those marginals by the same two-line mean-and-variance check as Lemma 1. The paper's Appendix C.1 builds a full inference distribution, a chain over $\tau$ plus the other steps attached directly to $x_0$, whose ELBO is again a positively weighted $L_{\text{simple}}$, so the trained network is valid for every subsequence. Quality degrades as jumps grow only because $\hat x_0$ becomes a worse proxy for $x_0$, and the deterministic version degrades much more slowly than the stochastic one.
:::

::: qa
**Does the step-skipping construction also apply to DDPM?**

Yes. Appendix C.1 holds for every $\varsigma$, including $\eta = 1$, and the table caption in the paper says that running $\eta = 1$ with $S < T$ can be seen as simulating a DDPM trained with $S$ steps. It just works badly with few steps: CIFAR-10 FID is 41.07 at 10 steps for $\eta = 1$, against 13.36 for $\eta = 0$. Each ancestral step replaces a wide, multimodal reverse distribution with a Gaussian around the posterior mean plus fresh noise, and with few steps that noise is large and cannot be corrected later. The deterministic sampler follows a smooth trajectory instead, which a first-order method can track with large steps.
:::

## What the experiments show (Section 5 and Appendix D)

### Setup

For each dataset the paper uses one trained model with $T = 1000$ and the objective $L_{\mathbf{1}}$, and changes only the sampler: the subsequence $\tau$ and the noise $\eta$. CIFAR-10 ($32 \times 32$), LSUN Bedroom and LSUN Church ($256 \times 256$) use Ho et al.'s released DDPM checkpoints. CelebA ($64 \times 64$) uses a model the authors trained themselves with the same U-Net design (no public checkpoint existed). The noise schedule follows Ho et al., so results are directly comparable. Quality is measured by FID; lower is better.

### Table 1: CIFAR-10 and CelebA

FID on CIFAR-10 ($32 \times 32$), by number of sampling steps $S$:

| Sampler | $S = 10$ | 20 | 50 | 100 | 1000 |
|---|---|---|---|---|---|
| $\eta = 0.0$ (DDIM) | **13.36** | **6.84** | **4.67** | **4.16** | 4.04 |
| $\eta = 0.2$ | 14.04 | 7.11 | 4.77 | 4.25 | 4.09 |
| $\eta = 0.5$ | 16.66 | 8.35 | 5.25 | 4.46 | 4.29 |
| $\eta = 1.0$ (DDPM, $\tilde\beta_t$) | 41.07 | 18.36 | 8.01 | 5.78 | 4.73 |
| $\hat\varsigma$ (DDPM, $\beta_t$) | 367.43 | 133.37 | 32.72 | 9.99 | **3.17** |

FID on CelebA ($64 \times 64$):

| Sampler | $S = 10$ | 20 | 50 | 100 | 1000 |
|---|---|---|---|---|---|
| $\eta = 0.0$ (DDIM) | **17.33** | **13.73** | **9.17** | **6.53** | 3.51 |
| $\eta = 0.2$ | 17.66 | 14.11 | 9.51 | 6.79 | 3.64 |
| $\eta = 0.5$ | 19.86 | 16.06 | 11.01 | 8.09 | 4.28 |
| $\eta = 1.0$ (DDPM, $\tilde\beta_t$) | 33.12 | 26.03 | 18.48 | 13.93 | 5.98 |
| $\hat\varsigma$ (DDPM, $\beta_t$) | 299.71 | 183.83 | 71.71 | 45.20 | **3.26** |

**How to read these tables.**

- **Every row improves with $S$.** More network calls buy quality; this is the compute trade-off DDIM exposes.
- **At every $S < 1000$, FID gets worse monotonically with stochasticity:** $\eta = 0 < 0.2 < 0.5 < 1 < \hat\varsigma$. On CIFAR-10 at 10 steps that is $13.36 < 14.04 < 16.66 < 41.07 < 367.43$.
- **The gap shrinks as $S$ grows.** At 10 steps DDPM ($\eta = 1$) has about three times DDIM's FID on CIFAR-10 (41.07 vs 13.36); at 1000 steps the two are close (4.73 vs 4.04).
- **$\hat\varsigma$ is catastrophic with few steps and best with all steps.** It wins at $S = 1000$ on both datasets (3.17 and 3.26), the case the paper flags as the one where DDIM is "marginally worse", and it is the configuration behind DDPM's reported CIFAR-10 number. With 10 steps it injects far more noise than the next level expects (the worked number in the $\hat\varsigma$ subsection), and FID explodes.
- **Equal-quality comparisons.** On CelebA, 100-step DDPM ($\eta = 1$, 13.93) is about as good as 20-step DDIM (13.73), as the paper points out: five times fewer steps for the same FID. On CIFAR-10, 100-step DDIM (4.16) is close to 1000-step DDIM (4.04).

The paper's qualitative samples match the numbers: DDPM samples deteriorate quickly at 10 steps, and $\hat\varsigma$ samples show visible noise.

### Table 3: LSUN at $256 \times 256$ (appendix)

| Sampler | Bedroom $S = 10$ | 20 | 50 | 100 | Church $S = 10$ | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|
| DDIM ($\eta = 0$) | **16.95** | **8.89** | **6.75** | **6.62** | **19.45** | **12.47** | **10.84** | 10.58 |
| DDPM ($\eta = 1$) | 42.78 | 22.77 | 10.81 | 6.81 | 51.56 | 23.37 | 11.16 | **8.27** |

For reference, 1000-step DDPM reaches 6.36 on Bedroom and 7.89 on Church.

**How to read it.** The pattern of Table 1 holds at high resolution for small $S$: at 10 steps DDPM's FID is 2.5 to 2.7 times DDIM's. But on Church at 100 steps the stochastic sampler wins (8.27 vs 10.58), and 100-step DDIM stays well above 1000-step DDPM there (10.58 vs 7.89). Deterministic sampling is the better choice when steps are scarce, not uniformly. EDM later studied when injected noise helps; see [EDM](edm.html).

### Wall-clock speed

Each step is one network evaluation, so sampling time is proportional to the trajectory length; the paper's Figure 4 plots the hours needed to sample 50,000 images on one RTX 2080 Ti against the number of steps and shows this linear scaling. Its introduction gives the baseline: about 20 hours for 50,000 images of size $32 \times 32$ with a 1000-step DDPM, and nearly 1000 hours at $256 \times 256$. Since DDIM reaches quality comparable to 1000-step models within 20 to 100 steps, the paper states a 10x to 50x speed-up over the original DDPM ($1000/100$ and $1000/20$); the abstract phrases it as 10x to 50x faster in wall-clock time, and the introduction speaks of accelerating sampling by 10x to 100x with better quality than DDPM at the same step count.

::: qa
**Why is $\eta = 0$ best when there are few steps?**

With few steps each jump is large, and $\hat x_0$ is a poor, blurry estimate (the posterior mean). The deterministic step reuses the predicted noise direction, so it moves along a smooth trajectory, the probability-flow ODE, which a first-order method follows with error that grows gently with step size. Stochastic steps replace part of that direction with fresh noise whose size grows with the jump. Later steps must remove it, but each is also large and inaccurate, so the injected noise and the averaging error compound. Table 1 shows FID rising monotonically with $\eta$ at every $S < 1000$, from 13.36 ($\eta = 0$) to 41.07 ($\eta = 1$) on CIFAR-10 at 10 steps. With many steps, stochasticity can help, because fresh noise corrects accumulated model errors; on LSUN Church at 100 steps, $\eta = 1$ beats $\eta = 0$.
:::

::: qa
**What happens with the larger-variance DDPM sampler $\hat\sigma$ at 10 steps, and why is it the best at 1000?**

It collapses: CIFAR-10 FID 367.43 at 10 steps (vs 13.36 for DDIM) and CelebA 299.71. The $\hat\sigma$ update keeps the deterministic part of the $\eta = 1$ step but uses noise of standard deviation $\sqrt{1-\bar\alpha_{\tau_i}/\bar\alpha_{\tau_{i-1}}}$, larger than $\varsigma(1)$. The noise variance of the next state then exceeds $1-\bar\alpha_{\tau_{i-1}}$ by $\hat\sigma^2(\bar\alpha_{\tau_{i-1}}-\bar\alpha_{\tau_i})/(1-\bar\alpha_{\tau_i})$, which is large for big jumps. For example, stepping from $t = 101$ to $t = 1$ with the linear schedule leaves noise of standard deviation about 0.32 where the model expects 0.01, and the remaining step cannot remove it, so samples look noisy and FID, which is very sensitive to such noise, explodes. With all 1000 steps the excess is $\beta_t^2\bar\alpha_{t-1}/(1-\bar\alpha_t)$, at most about $6.5 \times 10^{-5}$ in variance under the linear schedule, and this sampler gets the best FID (3.17 on CIFAR-10, 3.26 on CelebA), slightly better than DDIM's 4.04 and 3.51.
:::

::: qa
**Is deterministic sampling always better than stochastic sampling?**

No. DDIM wins clearly at small step counts on all four datasets, but at large step counts the stochastic samplers catch up or win: at 1000 steps the $\hat\sigma$ DDPM sampler beats DDIM on CIFAR-10 (3.17 vs 4.04) and CelebA (3.26 vs 3.51), and on LSUN Church at 100 steps $\eta = 1$ beats $\eta = 0$ (8.27 vs 10.58). One explanation, studied in EDM, is that fresh noise acts like a correction that pulls the sample back toward the right marginal after model errors, while a deterministic solver carries its errors forward. So: deterministic for few steps, reproducibility, interpolation and inversion; some stochasticity when steps are plentiful and only sample quality matters.
:::

## DDIM as an ODE solver (Section 4.3 and Appendix B)

### Rewriting the update in rescaled coordinates

Take the deterministic update from level $t$ to any earlier level $s < t$ (the paper writes $s = t - \Delta t$):

$$
x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s}\,\hat\epsilon, \qquad \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}, \qquad \hat\epsilon = \epsilon_\theta^{(t)}(x_t).
$$

Divide by $\sqrt{\bar\alpha_s}$, then substitute $\hat x_0$:

$$
\begin{aligned}
\frac{x_s}{\sqrt{\bar\alpha_s}} &= \hat x_0 + \sqrt{\frac{1-\bar\alpha_s}{\bar\alpha_s}}\,\hat\epsilon && \text{divide by } \sqrt{\bar\alpha_s}\\
&= \frac{x_t}{\sqrt{\bar\alpha_t}} - \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\,\hat\epsilon + \sqrt{\frac{1-\bar\alpha_s}{\bar\alpha_s}}\,\hat\epsilon && \text{substitute } \hat x_0\\
&= \frac{x_t}{\sqrt{\bar\alpha_t}} + \Big(\sqrt{\frac{1-\bar\alpha_s}{\bar\alpha_s}} - \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\Big)\hat\epsilon && \text{collect.}
\end{aligned}
$$

This is the paper's eq. (13). Now name the rescaled state and its noise level,

$$
\bar x := \frac{x}{\sqrt{\bar\alpha}}, \qquad \bar\sigma := \sqrt{\frac{1-\bar\alpha}{\bar\alpha}} = \mathrm{SNR}^{-1/2},
$$

so the update reads

$$
\bar x_s = \bar x_t + (\bar\sigma_s - \bar\sigma_t)\,\hat\epsilon .
$$

The map back is a bijection: from $\bar\sigma^2 = (1-\bar\alpha)/\bar\alpha$ we get $\bar\alpha(1 + \bar\sigma^2) = 1$, so $\bar\alpha = 1/(1+\bar\sigma^2)$ and $x = \bar x\sqrt{\bar\alpha} = \bar x/\sqrt{\bar\sigma^2 + 1}$. The network input is therefore $x_t = \bar x_t/\sqrt{\bar\sigma_t^2 + 1}$.

Why these coordinates are natural: dividing $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ by $\sqrt{\bar\alpha_t}$ gives $\bar x_t = x_0 + \bar\sigma_t\,\epsilon$, pure additive noise of standard deviation $\bar\sigma_t$, the "variance exploding" form. With the linear schedule, $\bar\sigma$ runs from $0.010$ at $t = 1$ to about $157$ at $t = 1000$.

### The ODE

Let $\bar\sigma(t)$ be continuous and increasing with $\bar\sigma(0) = 0$. The update is a forward-Euler step, with step size $\bar\sigma_s - \bar\sigma_t$, for the paper's eq. (14):

$$
\frac{d\bar x}{d\bar\sigma} = \epsilon_\theta\Big(\frac{\bar x}{\sqrt{\bar\sigma^2 + 1}},\ \bar\sigma\Big), \qquad \text{equivalently}\quad d\bar x(t) = \epsilon_\theta^{(t)}\Big(\frac{\bar x(t)}{\sqrt{\bar\sigma(t)^2 + 1}}\Big)\,d\bar\sigma(t),
$$

started from $\bar x(T) \sim \mathcal{N}(0, \bar\sigma(T)^2 I)$ with a large $\bar\sigma(T)$ (that is, $\bar\alpha_T \approx 0$). Two consequences:

- **The error of a step depends on how much $\hat\epsilon$ changes over it.** If $\hat\epsilon$ were constant, Euler would be exact for any step size. That is Toy 1, where the trajectory is a straight line in $(\bar\sigma, \bar x)$ and one step lands on the data. In Toy 2, $\hat\epsilon$ changes along the path and a single step fails.
- **The ODE can be run in either direction.** With enough steps, integrating from $t = 0$ to $T$ maps an image to a latent $x_T$, which DDPM cannot do. Section 5.4 tests this.

### Proposition 1: the same ODE as the variance-exploding probability flow

The variance-exploding (VE) SDE of Song et al. (2021) has marginals $p_t(\bar x)$ equal to the data convolved with $\mathcal{N}(0, \bar\sigma(t)^2 I)$, diffusion coefficient $g(t) = \sqrt{d\bar\sigma^2/dt}$, and a probability-flow ODE with the same marginals:

$$
d\bar x = -\tfrac{1}{2}\,g(t)^2\,\nabla_{\bar x}\log p_t(\bar x)\,dt .
$$

**Proposition 1.** With the optimal model $\epsilon_\theta^{(t)}$, the DDIM ODE (eq. 14) is this probability-flow ODE.

**Proof.**

1. *The optimal noise predictor.* By Tool 4, $\epsilon^{(t)*}(x_t) = \mathbb{E}[\epsilon \mid x_t]$. For fixed $t$, $x_t \mapsto \bar x_t = x_t/\sqrt{\bar\alpha_t}$ is a bijection, so conditioning on $x_t$ or on $\bar x_t$ is the same: $\epsilon^{(t)*}(x_t) = \mathbb{E}[\epsilon \mid \bar x_t]$.
2. *Score in terms of noise.* $\bar x_t = x_0 + \bar\sigma_t\epsilon$, so Tool 5 gives $\nabla_{\bar x}\log p_t(\bar x) = -\mathbb{E}[\epsilon \mid \bar x]/\bar\sigma = -\epsilon^{(t)*}\big(\bar x/\sqrt{\bar\sigma^2 + 1}\big)/\bar\sigma$.
3. *Substitute.* With $g^2 = d\bar\sigma^2/dt = 2\bar\sigma\,d\bar\sigma/dt$ (chain rule):

$$
d\bar x = -\tfrac{1}{2}\cdot 2\bar\sigma\frac{d\bar\sigma}{dt}\cdot\Big(-\frac{\epsilon^{(t)*}}{\bar\sigma}\Big)dt = \epsilon^{(t)*}\Big(\frac{\bar x}{\sqrt{\bar\sigma^2 + 1}}\Big)\frac{d\bar\sigma}{dt}\,dt = \epsilon^{(t)*}\Big(\frac{\bar x}{\sqrt{\bar\sigma^2 + 1}}\Big)d\bar\sigma ,
$$

which is eq. (14). Both start from $\bar x(T) \sim \mathcal{N}(0, \bar\sigma(T)^2 I)$, so the ODEs are identical. $\blacksquare$

The paper's appendix runs the same argument through the denoising score matching objective; Tool 5 is the shortcut. Since $\bar x$ is the VP state divided by $\sqrt{\bar\alpha}$, this is also the probability-flow ODE of the VP (DDPM) SDE written in rescaled coordinates. Because that ODE has the same marginals as the reverse SDE, deterministic DDIM with an exact model and small enough steps samples (approximately) the data distribution. This covers the $\varsigma = 0$ case that Theorem 1 leaves out.

### Same ODE, different discretization

The paper stresses that equivalent ODEs do not make equivalent samplers. Euler in $t$ on the probability-flow ODE, with $g^2\,\Delta t$ replaced by the exact change in $\bar\sigma^2$ and the score replaced by $-\hat\epsilon/\bar\sigma_t$, gives the paper's eq. (15):

$$
\frac{x_s}{\sqrt{\bar\alpha_s}} = \frac{x_t}{\sqrt{\bar\alpha_t}} + \frac{1}{2}\Big(\frac{1-\bar\alpha_s}{\bar\alpha_s} - \frac{1-\bar\alpha_t}{\bar\alpha_t}\Big)\sqrt{\frac{\bar\alpha_t}{1-\bar\alpha_t}}\;\hat\epsilon, \qquad\text{that is}\qquad \bar x_s = \bar x_t + \frac{\bar\sigma_s^2 - \bar\sigma_t^2}{2\bar\sigma_t}\,\hat\epsilon .
$$

This is effectively Euler with $\bar\sigma^2$ as the integration variable. Compare its increment with DDIM's:

$$
\frac{\bar\sigma_s^2 - \bar\sigma_t^2}{2\bar\sigma_t} = (\bar\sigma_s - \bar\sigma_t)\cdot\frac{\bar\sigma_s + \bar\sigma_t}{2\bar\sigma_t} .
$$

The ratio $(\bar\sigma_s + \bar\sigma_t)/(2\bar\sigma_t)$ is close to 1 for small steps, so the two agree, as the paper says, when $\bar\alpha_t$ and $\bar\alpha_s$ are close. For a big step it is not: a single step to $\bar\sigma_s = 0$ moves DDIM by $-\bar\sigma_t\hat\epsilon$, landing exactly on $\hat x_0$, while the $\bar\sigma^2$-Euler step moves only half as far and lands at $(\bar x_t + \hat x_0)/2$, still noisy. The paper's summary: DDIM takes Euler steps with respect to $d\bar\sigma(t)$, which "depends less directly on the scaling of time", while Song et al. take them with respect to $dt$.

EDM later adopted the same choice for its deterministic sampler. It argues for the parameterization $\sigma(t) = t$ with no extra scaling, notes that this is the choice DDIM makes, and observes that one Euler step to $\sigma = 0$ then returns the denoiser output. Since that output changes only slowly with the noise level, EDM argues that trajectories are close to straight lines at high and low noise, with most of the curvature in between. It then replaces Euler with Heun's second-order method and a step spacing concentrated at low noise; it reports that its sampling changes cut the evaluations needed for high-quality results by 3.2x in its DDIM setting. See [EDM](edm.html).

### DDIM is the first-order exponential integrator (DPM-Solver-1)

In $\bar x$ coordinates the right-hand side of the ODE is just $\hat\epsilon$, with no term proportional to $\bar x$, so integrating both sides over one step gives an exact expression:

$$
\bar x_s = \bar x_t + \int_{\bar\sigma_t}^{\bar\sigma_s}\hat\epsilon\,d\bar\sigma .
$$

It is not yet a formula you can evaluate, because $\hat\epsilon$ inside the integral is evaluated along the unknown trajectory. Holding $\hat\epsilon$ constant over the step gives $\bar x_s = \bar x_t + (\bar\sigma_s - \bar\sigma_t)\hat\epsilon$: DDIM is exact integration under the single assumption that the noise prediction does not change across the step. Multiplying by $\sqrt{\bar\alpha_s}$ returns to the original state:

$$
x_s = \frac{\sqrt{\bar\alpha_s}}{\sqrt{\bar\alpha_t}}\,x_t - \sqrt{\bar\alpha_s}\,(\bar\sigma_t - \bar\sigma_s)\,\hat\epsilon .
$$

In the original (variance-preserving) coordinates, the probability-flow ODE is semi-linear: a linear term in $x$ plus a nonlinear term in $\hat\epsilon$. An exponential integrator solves the linear part exactly and approximates only the nonlinear part. The formula above is exactly that, at first order: the factor $\sqrt{\bar\alpha_s}/\sqrt{\bar\alpha_t}$ is the exact solution of the linear part. Lu et al. show that DPM-Solver-1, which writes the integral in half-log-SNR $\lambda = \log(\sqrt{\bar\alpha}/\sqrt{1-\bar\alpha}) = -\log\bar\sigma$ and holds $\hat\epsilon$ constant, coincides with DDIM step for step, and they use this to explain why DDIM beats a plain Euler step on the VP ODE. Higher-order DPM-Solvers expand $\hat\epsilon$ in $\lambda$. DPM-Solver++ does the same with the data prediction $\hat x_0$ and describes itself as the high-order generalization of DDIM ($\eta = 0$) in that parameterization; it also shows that stochastic DDIM with a particular $\eta$ equals its first-order SDE solver. Salimans and Ho had earlier shown DDIM is a first-order discretization of the diffusion ODE. The derivations are on [diffusion models](diffusion-models.html), and the equivalence with Euler on the rescaled straight-line path of rectified flow is on [flow matching](flow-matching.html). The DDIM paper itself ended by suggesting multistep ODE methods such as Adams-Bashforth for fewer steps, which is what later multistep solvers do.

::: key
**How to say it in an interview.** "Divide the deterministic DDIM update by $\sqrt{\bar\alpha}$: with $\bar x = x/\sqrt{\bar\alpha}$ and $\bar\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha}$, it is $\bar x_s = \bar x_t + (\bar\sigma_s - \bar\sigma_t)\hat\epsilon$, a forward-Euler step of $d\bar x/d\bar\sigma = \hat\epsilon$. With the optimal model that is the probability-flow ODE of the VE SDE, by Tweedie's formula. Because it steps in $\bar\sigma$, not in $t$, it is exact when $\hat\epsilon$ is constant along the path, which makes it much better than Euler in $t$ for large steps. In VP coordinates it is the first-order exponential integrator, which is exactly DPM-Solver-1."
:::

::: qa
**Is DDIM an ODE solver? Which one, and for which ODE?**

Deterministic DDIM ($\eta = 0$) is a first-order solver of the probability-flow ODE. In rescaled coordinates $\bar x = x/\sqrt{\bar\alpha}$, $\bar\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha}$, the update is $\bar x_s = \bar x_t + (\bar\sigma_s - \bar\sigma_t)\,\epsilon_\theta(x_t)$, a forward-Euler step for $d\bar x/d\bar\sigma = \epsilon_\theta(\bar x/\sqrt{\bar\sigma^2+1})$. Proposition 1 shows that with the optimal model this is the probability-flow ODE of the variance-exploding SDE: Tweedie gives score $= -\mathbb{E}[\epsilon \mid \bar x]/\bar\sigma$, and $\tfrac12 g^2 = \bar\sigma\,d\bar\sigma/dt$ cancels the $1/\bar\sigma$. In the original coordinates the same step is a first-order exponential integrator that solves the linear drift exactly, which is DPM-Solver-1, and on a rescaled linear path it is Euler for rectified flow. With $\eta > 0$ it is a discretization of a stochastic process instead.
:::

::: qa
**Why does DDIM work much better than Euler on the probability-flow ODE with few steps, if they are the same ODE?**

The step variable differs. DDIM steps in $\bar\sigma$ and freezes $\hat\epsilon$ over the step, which is exact when $\hat\epsilon$ is constant; for a point-mass data distribution the trajectory is a straight line in $(\bar\sigma, \bar x)$ and one step is exact. The probability-flow Euler step in the paper's eq. (15) moves by $(\bar\sigma_s^2 - \bar\sigma_t^2)/(2\bar\sigma_t)$ instead of $\bar\sigma_s - \bar\sigma_t$; the ratio is $(\bar\sigma_s + \bar\sigma_t)/(2\bar\sigma_t)$, which is about 1 for small steps but $1/2$ for a step to zero noise, so it undershoots badly when steps are large. Equivalently, in VP coordinates naive Euler also discretizes the linear drift term, while DDIM integrates it exactly (exponential integrator). The paper's phrasing: DDIM's choice of $d\sigma$ "depends less directly on the scaling of time".
:::

## Consistency, interpolation and inversion (Sections 5.2 to 5.4)

### Consistency across trajectory lengths

Start several DDIM runs from the same $x_T$ with different subsequences $\tau$ (10, 20, 50, 100, 1000 steps). The paper's figures on CIFAR-10, LSUN Bedroom, LSUN Church and (in the appendix) CelebA show that the outputs share their high-level features: layout, colors, the identity of the object. Often a 20-step sample already matches the 1000-step one up to small details.

The paper's reading: $x_T$ alone is an informative latent encoding of the image, while the details that affect quality are encoded in the network parameters, since longer trajectories improve quality without changing the high-level features.

Why it happens: every $\tau$ is a discretization of the same ODE from the same initial condition, so coarse and fine discretizations follow approximately the same trajectory. Coarse structure is settled at high noise, where trajectories are close to straight (the observation EDM later built on), so even large steps get it right. DDPM has no such property: its output depends on all the noise injected along the way, as the $z = \pm 1$ branches of Toy 2 show.

### Interpolation in $x_T$

Because the high-level content is a function of $x_T$, DDIM can interpolate like a GAN. The paper draws two latents $x_T^{(0)}, x_T^{(1)} \sim \mathcal{N}(0, I)$, interpolates them by spherical linear interpolation (slerp), and decodes each point with 50-step DDIM:

$$
x_T^{(\alpha)} = \frac{\sin\big((1-\alpha)\theta\big)}{\sin\theta}\,x_T^{(0)} + \frac{\sin(\alpha\theta)}{\sin\theta}\,x_T^{(1)}, \qquad \theta = \arccos\frac{\big(x_T^{(0)}\big)^\top x_T^{(1)}}{\lVert x_T^{(0)}\rVert\,\lVert x_T^{(1)}\rVert}.
$$

(Here $\alpha \in [0, 1]$ is the interpolation weight, not a noise schedule.) Grid interpolations use four latents: slerp within two pairs with one weight, then slerp across the pairs with another. The results morph smoothly between samples on CelebA, Bedroom and Church. In DDPM, by contrast, the same $x_T$ leads to very different $x_0$, so it cannot interpolate this way (a footnote notes that interpolating all $T$ noise vectors might work).

**Why slerp and not linear interpolation.** A standard Gaussian in high dimension lives near a thin shell. For $z \sim \mathcal{N}(0, I_d)$, $\lVert z\rVert^2$ has mean $d$ and variance $2d$, so $\lVert z\rVert \approx \sqrt{d}$ with a spread of about $0.71$ regardless of $d$. Two independent draws are nearly orthogonal, $z_0^\top z_1 \approx 0$. The linear midpoint then has

$$
\Big\lVert \tfrac12(z_0 + z_1)\Big\rVert^2 = \tfrac14\big(\lVert z_0\rVert^2 + \lVert z_1\rVert^2 + 2z_0^\top z_1\big) \approx \tfrac14(d + d + 0) = \tfrac{d}{2} .
$$

For CIFAR-10, $d = 3 \times 32 \times 32 = 3072$: typical norm $\sqrt{3072} \approx 55.4$, midpoint norm $\approx 39.2$, about 23 spreads inside the shell. The network would be fed a latent with far too little variance, a regime it essentially never saw in training. Slerp keeps the norm: with equal norms and $\theta = \pi/2$, $\sin((1-\alpha)\pi/2) = \cos(\alpha\pi/2)$, so the squared norm is $(\cos^2(\alpha\pi/2) + \sin^2(\alpha\pi/2))\,d = d$ for every $\alpha$.

### Encoding and reconstruction

Run the ODE forward to encode. From $x_0$, each step goes up in noise using the model's prediction at the current point: for $t' > t$,

$$
\bar x_{t'} = \bar x_t + (\bar\sigma_{t'} - \bar\sigma_t)\,\epsilon_\theta^{(t)}(x_t), \qquad\text{equivalently}\qquad x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat x_0(x_t) + \sqrt{1-\bar\alpha_{t'}}\,\epsilon_\theta^{(t)}(x_t).
$$

Then decode the resulting $x_T$ with the usual DDIM sampler. The paper encodes and decodes the CIFAR-10 test set with the CIFAR-10 model, using $S$ steps each way, and reports per-dimension mean squared error with pixel values scaled to $[0, 1]$ (Table 2, rounded to $10^{-4}$):

| $S$ | 10 | 20 | 50 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|---|---|
| Reconstruction MSE | 0.014 | 0.0065 | 0.0023 | 0.0009 | 0.0004 | 0.0001 | 0.0001 |

**How to read it.** The error falls steadily with $S$. As root-mean-square error on a 0 to 255 scale ($255\sqrt{\text{MSE}}$), that is roughly 30 levels at $S = 10$, about 8 at $S = 100$, and about 2 to 3 at $S = 1000$ (the table rounds to $10^{-4}$, so the last value is coarse). The paper compares the behavior to neural ODEs and normalizing flows; DDPM cannot do this at all because its sampler is stochastic. A footnote adds that $x_T$ has the same dimension as $x_0$, so this is about invertibility, not compression.

### Why DDIM inversion is only approximate

The exact inverse of one decoding step $x_s = F_t(x_t)$ would have to solve an implicit equation: $F_t$ uses $\epsilon_\theta(x_t)$, and $x_t$ is the unknown. Inversion instead evaluates the network at the point it already has, $\epsilon_\theta^{(s)}(x_s) \approx \epsilon_\theta^{(t)}(x_t)$. The error sources:

1. **The local approximation.** The mismatch $\epsilon_\theta^{(t)}(x_t) - \epsilon_\theta^{(s)}(x_s)$ grows with the step size and with how fast $\hat\epsilon$ changes along the trajectory, which is the curvature. An Euler step up followed by an Euler step down is not the identity; the two local errors add rather than cancel. Table 2 measures this.
2. **Discretization of the ODE itself.** Only the exact ODE flow is a bijection. Encoder and decoder are two first-order approximations to it, from opposite ends.
3. **Model error.** The ODE built from an imperfect $\epsilon_\theta$ is still invertible in continuous time, so model error does not break invertibility by itself, but the latent you get is not "the noise that generated the image".
4. **Classifier-free guidance.** Editing needs guidance, and the guided prediction $\epsilon_u + w(\epsilon_c - \epsilon_u)$ with $w > 1$ amplifies differences, including the mismatch in item 1. Mokady et al. report that for unconditional models the accumulated error is negligible, but with a large guidance scale (7.5 is Stable Diffusion's default) it is amplified, giving visual artifacts and latents that may fall outside the Gaussian distribution, which hurts editability.
5. **Latent diffusion** adds an autoencoder round trip, which is not DDIM's fault but adds to the reconstruction error.

These limits drove a line of editing work. Null-text inversion runs DDIM inversion at $w = 1$ to get a "pivot" trajectory, then optimizes a per-step unconditional (null-text) embedding so that guided sampling at $w = 7.5$ follows that pivot, which enables Prompt-to-Prompt editing of real images. EDICT obtains exact inversion with two coupled sequences. Null-text inversion and Prompt-to-Prompt are worked through on [text-to-image and multimodal](text-to-image-multimodal.html).

::: key
**How to say it in an interview.** "Inversion runs the deterministic update upward, using $\hat\epsilon$ at the current point as a stand-in for $\hat\epsilon$ at the next one. That is only exact as the step size goes to zero, and the error grows with step size, with trajectory curvature, and especially with classifier-free guidance, which amplifies the mismatch. Without guidance the error is small: CIFAR-10 round-trip MSE is $9 \times 10^{-4}$ at 100 steps. With guidance at 7.5, editing methods correct it, for example null-text inversion, which optimizes the unconditional embedding per step around a $w = 1$ inversion."
:::

::: qa
**Why does the same $x_T$ give similar images with 20 and with 1000 DDIM steps?**

With $\eta = 0$ the sampler is a discretization of one deterministic ODE, so any step schedule starting from the same $x_T$ approximates the same trajectory and ends near the same image. Coarse properties (layout, color, object identity) are determined at high noise, where the trajectories are close to straight lines in $(\bar\sigma, \bar x)$ coordinates and even large steps follow them well. Fine details are decided at low noise, where more steps help, so step count mostly changes detail and quality. The paper concludes that $x_T$ acts as an informative latent code. DDPM lacks this property because its output depends on all the noise injected during sampling.
:::

::: qa
**Why interpolate DDIM latents with slerp rather than linear interpolation?**

Latents are draws from $\mathcal{N}(0, I_d)$, which concentrate near the sphere of radius $\sqrt{d}$, with a norm spread of about 0.7 independent of $d$. Two independent draws are nearly orthogonal, so their linear midpoint has norm about $\sqrt{d/2}$: for CIFAR-10 ($d = 3072$) about 39 instead of 55, roughly 23 standard deviations from where real latents live. The model has essentially never seen such a low-variance $x_T$, so lerped latents decode to images that are not typical samples. Slerp interpolates along the great circle between the two latents and, for equal norms, keeps the norm constant (with orthogonal endpoints, $\cos^2 + \sin^2 = 1$), so every intermediate latent looks like a plausible Gaussian draw. The DDIM paper uses slerp for both line and grid interpolations.
:::

::: qa
**Why is DDIM inversion only approximate, and when does it break down?**

Decoding step $t \to s$ uses $\epsilon_\theta(x_t)$; inverting it requires $x_t$ given $x_s$, an implicit equation, so inversion substitutes $\epsilon_\theta(x_s)$, which is valid only in the small-step limit. Errors therefore grow with step size and with trajectory curvature, and they accumulate over the trajectory; both encoding and decoding are first-order discretizations of the ODE, and only the exact flow is a bijection. On unconditional CIFAR-10 the round trip is good: per-dimension MSE 0.0023 at 50 steps and 0.0009 at 100. It breaks down with classifier-free guidance at typical scales, which amplifies the mismatch: Mokady et al. found that guided inversion produces artifacts and non-Gaussian latents. The fixes are inverting at $w = 1$ and then optimizing null-text embeddings, or exact schemes such as EDICT.
:::

## DDIM in practice

### Timestep spacing and zero terminal SNR

Lin et al. (2023) name three common ways to choose $S$ sampling steps out of $T$; their table, for $T = 1000$ and $S = 10$ in 1-based indexing:

| Spacing | Rule | Steps |
|---|---|---|
| leading (DDIM, PNDM) | arange(1, T+1, floor(T/S)) | 1, 101, 201, ..., 901 |
| linspace (iDDPM) | round(linspace(1, T, S)) | 1, 112, 223, 334, 445, 556, 667, 778, 889, 1000 |
| trailing (DPM-Solver) | round(flip(arange(T, 0, -T/S))) | 100, 200, ..., 1000 |

"Leading" is exactly the DDIM repository's "uniform" rule shown earlier. Its flaw: the first step evaluates the model at $t = 901$ on pure noise, although at that step the model was trained on inputs that still contain some signal ($\sqrt{\bar\alpha_{901}} = 0.016$ under the linear schedule). Lin et al. locate the main cause in the noise schedule and say leading spacing makes it worse. Schedules such as Stable Diffusion's never remove all the signal: its last step keeps $\sqrt{\bar\alpha_T} \approx 0.068$. The model learns to read the leaked low-frequency information (such as each channel's mean), and since sampling starts from zero-mean noise, outputs stay at medium brightness.

Their fixes are: rescale the schedule to zero terminal SNR, train with $v$-prediction, start sampling at $t = T$ (trailing or linspace), and rescale classifier-free guidance. They find trailing uses steps more efficiently at small $S$, because a step at $t = 1$ barely changes anything. At $S = 5$ trailing is noticeably better than linspace; at $S = 25$ the difference is subtle. They fine-tuned Stable Diffusion 2.1-base with all fixes; sampled by DDIM with 50 trailing steps, guidance 7.5 and rescale factor 0.7, it reached FID 21.66 on 10k COCO 2014 validation captions, against 23.76 for the official model and 22.96 for a reference model fine-tuned on the same data without the fixes (all with 50 DDIM steps and guidance 7.5).

DDIM needs one change under zero terminal SNR. At $t = T$, $\bar\alpha_T = 0$, so $\hat x_0 = (x_T - \hat\epsilon)/\sqrt{\bar\alpha_T}$ divides by zero (and predicting $\epsilon$ from pure noise $x_T = \epsilon$ teaches the network nothing). Lin et al. instead train the network to predict Salimans and Ho's target

$$
v_t := \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0 .
$$

Together with $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, this gives $x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,v_t$ and $\epsilon = \sqrt{\bar\alpha_t}\,v_t + \sqrt{1-\bar\alpha_t}\,x_t$ (substitute and use $\bar\alpha_t + (1-\bar\alpha_t) = 1$). So convert the prediction $\hat v$ into $\hat x_0$ and $\hat\epsilon$ with these two formulas, then apply eq. (12) unchanged. At $t = T$ this gives $\hat\epsilon = x_T$ and $\hat x_0 = -\hat v$, the model's estimate of the (prompt-conditional) data mean, independent of the noise. (Lin et al. write $v_T = x_0$; with the definition above, which is theirs, the sign is negative, which does not affect the argument.) Lin et al. note that diversity then starts at the second step, where DDIM adds different predicted noise back onto the same $\hat x_0$. The rest of the zero-terminal-SNR story is on [diffusion models](diffusion-models.html).

### DDIM with classifier-free guidance

With guidance, compute the guided prediction $\tilde\epsilon = \epsilon_u + w(\epsilon_c - \epsilon_u)$ (library convention, $w = 1$ is unguided) and use it in place of $\hat\epsilon$ in both the $\hat x_0$ term and the direction term. Because $\hat x_0$ is affine in $\hat\epsilon$ for a fixed $x_t$, guiding the noise is the same as guiding the clean estimate:

$$
\hat x_0(\tilde\epsilon) = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\big(\epsilon_u + w(\epsilon_c - \epsilon_u)\big)}{\sqrt{\bar\alpha_t}} = \hat x_0(\epsilon_u) + w\big(\hat x_0(\epsilon_c) - \hat x_0(\epsilon_u)\big).
$$

Practical points:

- **Cost.** Each step needs two network passes, usually batched.
- **Saturation.** Large $w$ pushes $\hat x_0$ outside the data range. Remedies include clipping or dynamic thresholding of $\hat x_0$ in pixel space, the CFG rescale of Lin et al., and solvers built on $\hat x_0$ rather than $\hat\epsilon$. The DPM-Solver++ authors found earlier high-order $\epsilon$-solvers unstable under large guidance, and worse than first-order DDIM.
- **Step counts.** The same authors write that DDIM generally needs 100 to 250 steps to converge for guided sampling, while DPM-Solver++ nearly converges in 15 to 20.
- **What is being sampled.** The guided $\tilde\epsilon$ is in general not the noise prediction for the noised versions of any one fixed distribution, so DDIM and DDPM with the same guidance can produce different distributions; see the guidance section on [diffusion models](diffusion-models.html).

### Step counts in latent diffusion, as reported

Only numbers found in the papers:

- The latent diffusion paper (Rombach et al.) evaluated its text-to-image model on MS-COCO with 250 DDIM steps (FID 12.63 with guidance scale 1.5). Its table of unconditional results used 200 or 500 DDIM steps, for example 200 steps for LSUN-Churches (FID 4.02) and 500 steps for CelebA-HQ (FID 5.11). The appendix sample figures list $\eta = 0$ for some datasets (CelebA-HQ, LSUN-Churches) and $\eta = 1$ for others (FFHQ, LSUN-Bedrooms).
- Lin et al. evaluate Stable Diffusion with DDIM at $S = 50$.
- DPM-Solver++ reports 100 to 250 DDIM steps for guided sampling to converge, and 15 to 20 steps for its own solver. It also notes that the Stable Diffusion code defaulted to 50 PNDM steps, and that DPM-Solver++ reaches a similar convergence error in 15 to 20.

### Implementation details worth knowing

- **Indexing.** The official code counts time from 0, with code step $i$ using $\bar\alpha_{i+1}$; Lin et al. warn about the same off-by-one. Check that sampling and training agree on whether index $t$ means $\bar\alpha_t$ or $\bar\alpha_{t+1}$.
- **The final step.** In `functions/denoising.py`, the step after the smallest timestep goes to "$-1$", which maps to $\bar\alpha = 1$, so the noise coefficient is zero for any $\eta$ and the output is $\hat x_0$.
- **The noise coefficient in code** is `eta * sqrt((1 - at/at_next) * (1 - at_next) / (1 - at))`, which is $\varsigma_t(\eta)$.
- **Clipping.** The repository's `ddpm_steps` (used for the $\hat\sigma$ sampler) clamps $\hat x_0$ to $[-1, 1]$; its `generalized_steps` (used for every $\eta$, including $\eta = 1$) does not.

### Sampler comparison

| | DDPM ancestral | DDIM ($\eta = 0$) | DPM-Solver++ (2M) | EDM Heun |
|---|---|---|---|---|
| Deterministic | no | yes ($\eta > 0$ adds noise) | yes (an SDE variant exists) | yes (EDM also has a stochastic sampler) |
| What it discretizes | the learned reverse chain (an SDE in the limit) | probability-flow ODE, Euler in $\bar\sigma$ | probability-flow ODE, exponential integrator in $\lambda$ on $\hat x_0$, second-order multistep | probability-flow ODE in VE form, Heun (second order) |
| Network calls per step | 1 | 1 | 1 (reuses earlier evaluations) | 2 (1 on the final step) |
| Step grid | all $T$ steps, or a subsequence | linear or quadratic $\tau$ in the paper; leading, linspace or trailing in libraries | uniform in $t$ in its guided experiments | $\sigma$ grid with $\rho = 7$ |
| Reported steps | 1000 in DDPM; CIFAR-10 FID 41.07 at 10 steps | 20 to 100 for 1000-step quality (paper); 100 to 250 for guided sampling (DPM-Solver++ authors) | 15 to 20 for guided sampling | 35 evaluations for FID 1.79 on class-conditional CIFAR-10 |
| Same $x_T$ gives the same image | no | yes | yes | yes |
| Can encode by running backward | no | approximately | approximately (ODE) | approximately (ODE) |
| Retraining needed | no | no | no | no |

EDM details are on [EDM](edm.html), and DPM-Solver on [diffusion models](diffusion-models.html).

::: qa
**What is the difference between leading, linspace and trailing timestep spacing, and why does it matter for DDIM?**

They decide which $S$ of the $T$ training steps a sampler visits. Leading, $1, 1 + T/S, \dots$ (the original DDIM code's "uniform"), never visits $t = T$: with $S = 10$ the first network call is at $t = 901$. Linspace includes both $1$ and $T$; trailing counts back from $T$ in equal strides and skips $t = 1$. Starting below $T$ means the first call sees pure noise at a timestep whose training inputs contained some signal; together with schedules that never reach zero SNR (Stable Diffusion keeps $\sqrt{\bar\alpha_T} \approx 0.068$), this produced Stable Diffusion's inability to generate very dark or bright images. Lin et al. recommend trailing with a zero-terminal-SNR schedule and $v$-prediction, and they found it spends steps more efficiently at small $S$, because a step at $t = 1$ does almost nothing. The difference from linspace is clear at 5 steps and subtle at 25.
:::

::: qa
**How do you use classifier-free guidance with DDIM, and what goes wrong at high guidance?**

Run the conditional and unconditional branches (batched), form $\tilde\epsilon = \epsilon_u + w(\epsilon_c - \epsilon_u)$, and use $\tilde\epsilon$ everywhere in the DDIM step: in $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\tilde\epsilon)/\sqrt{\bar\alpha_t}$ and in the direction term. Because $\hat x_0$ is affine in $\hat\epsilon$, this is identical to guiding $\hat x_0$ with the same $w$. At high $w$ the extrapolated $\hat x_0$ leaves the data range, giving oversaturated, over-exposed images; fixes are thresholding $\hat x_0$ (pixel space), CFG rescale, or data-prediction solvers such as DPM-Solver++, whose authors also report that DDIM needs 100 to 250 steps to converge under guidance. Guidance also makes DDIM inversion much less accurate, which is why editing uses null-text inversion or similar.
:::

::: qa
**Given a pretrained diffusion model and a latency budget, which sampler would you choose?**

With roughly 10 to 25 network calls, a high-order deterministic solver: DPM-Solver++(2M) for guided text-to-image, or EDM's Heun with $\rho = 7$ spacing; DDIM at that budget is usable but noticeably worse. If I need a reproducible noise-to-image map, interpolation, or inversion for editing, a deterministic solver is required, and DDIM is the simplest and is what most inversion methods assume. With a generous budget and only sample quality in mind, some stochasticity often helps: DDPM-style sampling beat DDIM at 1000 steps on CIFAR-10 and CelebA in the DDIM paper, and EDM's stochastic sampler improved some models. In every case I would start sampling at $t = T$ (trailing or linspace spacing) and check the schedule's terminal SNR. Below about 10 calls, training-free solvers usually lose noticeable quality, and I would look at a distilled model.
:::

## Rapid-fire review

::: qa
**What does "implicit" mean in "denoising diffusion implicit model"?**

An implicit probabilistic model generates samples by pushing a latent through a fixed deterministic procedure, without a tractable density, like a GAN generator. With $\eta = 0$, DDIM's $x_0$ is a deterministic function of $x_T$.
:::

::: qa
**What does $\alpha_t$ mean in the DDIM paper?**

DDPM's $\bar\alpha_t$, the cumulative product $\prod_{s \le t}(1-\beta_s)$. DDPM's per-step $\alpha_t = 1 - \beta_t$ is $\alpha_t/\alpha_{t-1}$ in DDIM notation.
:::

::: qa
**Does DDIM require retraining a DDPM model?**

No. The DDIM family has DDPM's marginals, its ELBO is a reweighted $L_{\text{simple}}$, and the paper's experiments reuse Ho et al.'s checkpoints unchanged.
:::

::: qa
**Is DDIM's forward process a diffusion?**

Not in general. Only at $\varsigma_t^2 = \tilde\beta_t$ ($\eta = 1$) is it DDPM's Markov diffusion. Otherwise it is non-Markovian, since each step depends on $x_0$; with $\varsigma = 0$ it is deterministic given $x_0$ and $x_{t-1}$ (for $t \ge 2$). The name keeps "diffusion" because the model is trained with the DDPM objective, "despite the forward process no longer being a diffusion", in the paper's words.
:::

::: qa
**What is $\bar\alpha_0$, and why is it needed?**

$\bar\alpha_0 := 1$, the clean level. It makes the last step output $\hat x_0$: the noise coefficient $\sqrt{1-\bar\alpha_0-\varsigma_1^2}$ forces $\varsigma_1 = 0$ and drops the $\hat\epsilon$ term.
:::

::: qa
**What is the largest allowed $\varsigma_t$?**

$\varsigma_t^2 \le 1 - \bar\alpha_{t-1}$, so the square root in the update is real. At equality, $x_{t-1}$ is independent of $x_t$ given $x_0$; DDPM's $\tilde\beta_t$ lies strictly below this bound.
:::

::: qa
**Which subsequence rule did the paper use for which dataset?**

Quadratic, $\tau_i = \lfloor c i^2 \rfloor$, for CIFAR-10 and linear, $\tau_i = \lfloor c i \rfloor$, for CelebA and LSUN; each was slightly better on its dataset.
:::

::: qa
**Who used the larger-variance $\hat\sigma$ sampler originally?**

Ho et al.'s implementation, only for their CIFAR-10 samples. It gives the best 1000-step FID in the DDIM tables (3.17) and the worst at 10 steps (367.43).
:::

::: qa
**What reconstruction error does DDIM encode-decode reach on CIFAR-10?**

Per-dimension MSE on $[0, 1]$ pixels: 0.014 at 10 steps, 0.0009 at 100, and 0.0001 at 500 and 1000 steps.
:::

::: qa
**Why can't DDPM encode an image into $x_T$?**

Its sampler injects fresh noise at every step, so $x_0$ is not a function of $x_T$, and there is no deterministic map to invert.
:::

::: qa
**In which variable does DDIM take Euler steps?**

$\bar\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha} = \mathrm{SNR}^{-1/2} = e^{-\lambda}$, on the rescaled state $\bar x = x/\sqrt{\bar\alpha}$.
:::

::: qa
**Why does sampling time scale linearly with the number of steps?**

Every step is one network evaluation (two with classifier-free guidance), and the rest of the update is a few elementwise operations.
:::

::: qa
**At $\eta = 1$ and $S < T$, what is the sampler?**

DDPM's ancestral sampler on the coarser grid, with posterior variances computed between the visited steps; the paper describes it as simulating a DDPM trained with $S$ steps.
:::

## References

1. Song, Meng, Ermon, "Denoising Diffusion Implicit Models," ICLR 2021. <https://arxiv.org/abs/2010.02502>. Official code: <https://github.com/ermongroup/ddim>
2. Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models," 2020. <https://arxiv.org/abs/2006.11239>
3. Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations," 2021. <https://arxiv.org/abs/2011.13456>
4. Mohamed and Lakshminarayanan, "Learning in Implicit Generative Models," 2016. <https://arxiv.org/abs/1610.03483>
5. Nichol and Dhariwal, "Improved Denoising Diffusion Probabilistic Models," 2021. <https://arxiv.org/abs/2102.09672>
6. Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps," 2022. <https://arxiv.org/abs/2206.00927>
7. Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models," 2022. <https://arxiv.org/abs/2211.01095>
8. Salimans and Ho, "Progressive Distillation for Fast Sampling of Diffusion Models," 2022. <https://arxiv.org/abs/2202.00512>
9. Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models," 2022. <https://arxiv.org/abs/2206.00364>
10. Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed," 2023. <https://arxiv.org/abs/2305.08891>
11. Ho and Salimans, "Classifier-Free Diffusion Guidance," 2022. <https://arxiv.org/abs/2207.12598>
12. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," 2021. <https://arxiv.org/abs/2112.10752>
13. Mokady et al., "Null-text Inversion for Editing Real Images using Guided Diffusion Models," 2022. <https://arxiv.org/abs/2211.09794>
14. Hertz et al., "Prompt-to-Prompt Image Editing with Cross Attention Control," 2022. <https://arxiv.org/abs/2208.01626>
15. Wallace et al., "EDICT: Exact Diffusion Inversion via Coupled Transformations," 2022. <https://arxiv.org/abs/2211.12446>
