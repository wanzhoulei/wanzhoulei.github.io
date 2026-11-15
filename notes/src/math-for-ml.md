---
title: "Probability, Linear Algebra and Optimization for ML"
subtitle: "The probability, Gaussian algebra, estimation, information theory, Monte Carlo, linear algebra, optimization and experimental statistics that ML interviews test, with every derivation shown and every number checked."
order: 0.8
updated: 2026-11-15
---

ML research interviews test mathematical fundamentals in two ways. Some questions are direct: derive the KL divergence between two Gaussians, prove that gradient descent converges, explain why the sample variance divides by $n-1$. Others are puzzles asked to watch you reason out loud: the expected number of flips until two heads in a row, the chance you are ill after a positive test, the length of a random chord. For someone trained in applied math very little here is new. What the interview rewards is producing the clean argument quickly, attaching a number to it, and connecting it to an ML method: a diffusion posterior, a VAE loss, an optimizer's step size, an error bar in a results table. This page collects that material in one place, keeps the standard parts brisk, and spends its space on the derivations interviewers actually ask for.

**Roadmap.** Section 1 covers probability rules, the common distributions, change of variables, and the classic puzzles. Section 2 is Gaussian algebra (conditioning, products, sums, the reparameterization trick, Gaussian KL) and shows how each fact appears in diffusion models. Section 3 is estimation: maximum likelihood, MAP, bias and variance, the $n-1$, Fisher information and conjugate priors. Section 4 is information theory: entropy, KL and its two directions, mutual information, Jensen-Shannon and the ELBO. Section 5 is Monte Carlo: inverse-CDF, rejection and importance sampling, Metropolis-Hastings, Langevin dynamics and Gumbel-max. Section 6 is linear algebra: spectral theorem, SVD, norms and conditioning, the pseudo-inverse, trace and determinant identities, matrix calculus and power iteration. Section 7 is optimization: convexity, gradient descent rates with a full proof, the condition-number effect, Newton and BFGS, KKT and duality, SGD and saddle points. Section 8 is the statistics you need to report experiments honestly. Each section ends with practice questions; questions marked with a badge were reported from real interviews and are copied from the [interview question bank](interview-question-bank.html). A rapid-fire review and references close the page. Neighbouring topics live elsewhere: losses and optimizers in practice on [deep learning training](deep-learning-training.html), and the generative models themselves on [diffusion models](diffusion-models.html) and [GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html).

## Notation

| Symbol | Meaning |
|---|---|
| $P(A)$, $p(x)$ | probability of an event; a PMF or density |
| $\mathbb{E}[X]$, $\mathrm{Var}(X)$, $\mathrm{Cov}(X,Y)$ | expectation, variance, covariance |
| $\mathbb{E}_p[f]$ | expectation of $f(x)$ with $x \sim p$ |
| $\mathcal{N}(\mu, \Sigma)$, $\mathcal{N}(x; \mu, \Sigma)$ | Gaussian distribution; its density evaluated at $x$ |
| $\varphi$, $\Phi$ | standard normal density and CDF |
| $\Lambda = \Sigma^{-1}$ | precision matrix |
| $H(p)$, $H(p,q)$ | entropy, cross-entropy |
| $\mathrm{KL}(p \,\Vert\, q)$ | Kullback-Leibler divergence $\mathbb{E}_p[\log p/q]$ |
| $I(X;Y)$ | mutual information |
| $\log$ | natural log (nats) unless written $\log_2$ (bits) |
| $\theta$, $\hat\theta$ | a parameter and its estimate |
| $I(\theta)$ | Fisher information (context separates it from mutual information) |
| $\mathbf{1}\{\cdot\}$ | indicator: 1 if the condition holds, else 0 |
| $A^\top$, $A^{-1}$, $A^{+}$ | transpose, inverse, Moore-Penrose pseudo-inverse |
| $\lambda_i$, $\sigma_i$ | eigenvalues; singular values, $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ |
| $\lVert x \rVert$, $\lVert A \rVert_2$, $\lVert A \rVert_F$ | Euclidean norm; spectral norm $\sigma_1$; Frobenius norm |
| $A \succeq 0$, $A \succ 0$ | $A$ symmetric positive semidefinite, positive definite |
| $\kappa(A)$ | condition number $\sigma_{\max}/\sigma_{\min}$ |
| $\nabla f$, $\nabla^2 f$ | gradient, Hessian |
| $L$, $\mu$ | smoothness and strong-convexity constants (Section 7 only; elsewhere $\mu$ is a mean) |
| $\eta$ | step size (learning rate) |
| $\beta_t$, $\bar\alpha_t = \prod_{s \le t}(1 - \beta_s)$ | DDPM noise schedule, as on [diffusion models](diffusion-models.html) |

## 1. Probability essentials

### Bayes' rule and base rates

For events $A$ and $B$ with $P(B) > 0$,

$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}, \qquad P(B) = \sum_i P(B \mid A_i)\,P(A_i),
$$

where the second formula (the law of total probability) sums over any partition $A_1, A_2, \dots$ of the outcomes. Dividing Bayes' rule for $A$ by Bayes' rule for its complement $\bar A$ cancels $P(B)$ and gives the **odds form**, which is often the fastest way to compute:

$$
\underbrace{\frac{P(A \mid B)}{P(\bar A \mid B)}}_{\text{posterior odds}} = \underbrace{\frac{P(A)}{P(\bar A)}}_{\text{prior odds}} \times \underbrace{\frac{P(B \mid A)}{P(B \mid \bar A)}}_{\text{likelihood ratio}}.
$$

**Worked example: a disease test.** A disease has prevalence 1%. The test has sensitivity $P(+ \mid \text{ill}) = 0.99$ and specificity $P(- \mid \text{healthy}) = 0.99$. You test positive. How likely are you to be ill?

Count a population of 10,000 people. 100 are ill, and 99 of them test positive. 9,900 are healthy, and 1% of them, 99 people, also test positive. So 198 people test positive and 99 of them are ill:

$$
P(\text{ill} \mid +) = \frac{0.01 \times 0.99}{0.01 \times 0.99 + 0.99 \times 0.01} = \frac{0.0099}{0.0198} = 0.5 .
$$

In odds form: prior odds $1:99$, likelihood ratio $0.99/0.01 = 99$, posterior odds $1:1$. A "99% accurate" test gives a coin flip because the healthy group is 99 times larger, so its 1% error rate produces as many positives as the ill group's 99% hit rate. At prevalence 0.1% the same test gives $0.00099/(0.00099 + 0.00999) = 9.0\%$. A second positive from a test whose errors are independent of the first (given disease status) multiplies the odds by 99 again: $99:1$, or $P = 0.99$. The independence assumption matters: repeating the same test on the same person often repeats the same error.

**Where this appears in ML.** Precision is $P(\text{positive class} \mid \text{flagged})$, so a detector with 99% sensitivity and specificity has only 50% precision on a class with 1% prevalence; precision depends on the base rate, recall does not. A generative classifier computes $p(y \mid x) \propto p(x \mid y)\,p(y)$. In diffusion, conditional generation uses $p(x \mid y) \propto p(y \mid x)\,p(x)$; taking $\nabla_x \log$ of both sides kills the normalizer $p(y)$, giving $\nabla_x \log p(x \mid y) = \nabla_x \log p(y \mid x) + \nabla_x \log p(x)$, which is classifier guidance (see [diffusion models](diffusion-models.html)).

::: key
**How to say it in an interview.**

"Bayes' rule in odds form: posterior odds equal prior odds times the likelihood ratio. Prior odds are 1 to 99, the likelihood ratio of a positive is 99, so the posterior odds are 1 to 1. The test is good, but the healthy population is so much larger that its false positives match the true positives."
:::

### Expectation and variance rules

These are used constantly; each follows from linearity of the integral.

- $\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$, always, with no independence needed.
- $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ and $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$.
- $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)$; more generally $\mathrm{Var}(\sum_i X_i) = \sum_{i,j} \mathrm{Cov}(X_i, X_j)$.
- If $X$ and $Y$ are independent, $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$, so the covariance term vanishes. The mean of $n$ i.i.d. variables with variance $\sigma^2$ therefore has variance $\sigma^2/n$ and standard deviation $\sigma/\sqrt n$.
- **Indicator trick.** A count is a sum of indicators, so its expectation is a sum of probabilities. For example, the expected number of fixed points of a uniformly random permutation of $n$ items is $\sum_{i=1}^n P(\pi(i) = i) = n \cdot \frac1n = 1$, even though the indicators are dependent.
- **Tail sums.** For $X \ge 0$, $\mathbb{E}[X] = \int_0^\infty P(X > x)\,dx$, and for nonnegative integers $\mathbb{E}[X] = \sum_{k \ge 1} P(X \ge k)$. Write $X = \int_0^\infty \mathbf{1}\{X > x\}\,dx$ and take expectations inside the integral.

**The mean minimizes squared error.** For any constant $c$,

$$
\mathbb{E}[(X - c)^2] = \mathbb{E}[(X - \mathbb{E}X)^2] + (\mathbb{E}X - c)^2 = \mathrm{Var}(X) + (\mathbb{E}X - c)^2,
$$

because the cross term $2(\mathbb{E}X - c)\,\mathbb{E}[X - \mathbb{E}X]$ is zero. So $c = \mathbb{E}X$ is the minimizer, and the minimum is the variance. Applied conditionally on an input, this says that a network trained with MSE learns $\mathbb{E}[y \mid x]$. That is why an MSE-trained super-resolution model produces blurry output: it predicts the average of all plausible sharp images (see [restoration and super-resolution](restoration-super-resolution.html)), and it is also why a diffusion denoiser trained with MSE learns the posterior mean $\mathbb{E}[x_0 \mid x_t]$. The absolute error $\mathbb{E}|X - c|$ is minimized by a median instead, which is the L1 counterpart.

**Law of total expectation.** $\mathbb{E}[X] = \mathbb{E}\big[\mathbb{E}[X \mid Y]\big]$. In the discrete case,

$$
\mathbb{E}\big[\mathbb{E}[X \mid Y]\big] = \sum_y P(Y = y) \sum_x x\, P(X = x \mid Y = y) = \sum_x x \sum_y P(X = x, Y = y) = \sum_x x\,P(X = x),
$$

by the definition of conditional probability and then summing out $y$.

**Law of total variance.** $\mathrm{Var}(X) = \mathbb{E}\big[\mathrm{Var}(X \mid Y)\big] + \mathrm{Var}\big(\mathbb{E}[X \mid Y]\big)$. Derivation, in three lines:

$$
\begin{aligned}
\mathbb{E}\big[\mathrm{Var}(X \mid Y)\big] &= \mathbb{E}\big[\mathbb{E}[X^2 \mid Y] - (\mathbb{E}[X \mid Y])^2\big] = \mathbb{E}[X^2] - \mathbb{E}\big[(\mathbb{E}[X \mid Y])^2\big], \\
\mathrm{Var}\big(\mathbb{E}[X \mid Y]\big) &= \mathbb{E}\big[(\mathbb{E}[X \mid Y])^2\big] - \big(\mathbb{E}[\mathbb{E}[X \mid Y]]\big)^2 = \mathbb{E}\big[(\mathbb{E}[X \mid Y])^2\big] - (\mathbb{E}X)^2, \\
\text{sum} &= \mathbb{E}[X^2] - (\mathbb{E}X)^2 = \mathrm{Var}(X).
\end{aligned}
$$

The first line applies the variance formula conditionally and then the tower rule; the second is the variance formula for the random variable $\mathbb{E}[X \mid Y]$; adding them cancels the middle terms. In words: total variance is the average within-group variance plus the variance of the group means.

*Example.* Let $Z$ be a fair coin and $X \mid Z \sim \mathcal{N}(\pm 2, 1)$ (mean $-2$ for tails, $+2$ for heads). Within-group variance is 1, the group means $\pm 2$ have variance 4, so $\mathrm{Var}(X) = 5$; a simulation with $10^6$ draws gives 4.996. The same split appears in uncertainty estimation for deep ensembles: averaging each member's predicted variance gives the noise part, and the spread of the members' means gives the model-disagreement part.

### Covariance, correlation and independence

$\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}X)(Y - \mathbb{E}Y)] = \mathbb{E}[XY] - \mathbb{E}X\,\mathbb{E}Y$, and the correlation $\rho = \mathrm{Cov}(X,Y)/(\sigma_X \sigma_Y)$ lies in $[-1, 1]$ by Cauchy-Schwarz, with $|\rho| = 1$ exactly when $Y$ is an affine function of $X$ almost surely. For a random vector $x$ with mean $\mu$, the covariance matrix $\Sigma = \mathbb{E}[(x - \mu)(x - \mu)^\top]$ is positive semidefinite, because $v^\top \Sigma v = \mathrm{Var}(v^\top x) \ge 0$, and $\mathrm{Cov}(Ax + b) = A\Sigma A^\top$.

**Independence implies uncorrelated; the converse fails.** If $X$ and $Y$ are independent then $\mathbb{E}[XY] = \mathbb{E}X\,\mathbb{E}Y$, so the covariance is zero. For the converse, take $X \sim \mathcal{N}(0, 1)$ and $Y = X^2$. Then

$$
\mathrm{Cov}(X, X^2) = \mathbb{E}[X^3] - \mathbb{E}[X]\,\mathbb{E}[X^2] = 0 - 0 = 0,
$$

since odd moments of a symmetric distribution vanish, yet $Y$ is a deterministic function of $X$. A discrete version: $X$ uniform on $\{-1, 0, 1\}$ and $Y = X^2$. Correlation measures only linear dependence; mutual information (Section 4) is zero only under independence.

**The Gaussian exception, and its limit.** If $(X, Y)$ is *jointly* Gaussian and uncorrelated, then $\Sigma$ is diagonal, the density factorizes, and $X$ and $Y$ are independent. Gaussian marginals are not enough. Let $X \sim \mathcal{N}(0,1)$, let $S = \pm 1$ be a fair sign independent of $X$, and $Y = SX$. Then $Y \sim \mathcal{N}(0, 1)$ and $\mathrm{Cov}(X, Y) = \mathbb{E}[S]\,\mathbb{E}[X^2] = 0$, but $|Y| = |X|$, so they are dependent; $(X, Y)$ is not jointly Gaussian because $X + Y = 0$ with probability $1/2$.

::: pitfall
"Uncorrelated means independent" is true only for jointly Gaussian variables. Two Gaussian marginals with zero correlation can still be strongly dependent.
:::

### PMF versus PDF, and the binomial

A **PMF** gives probabilities of a discrete variable directly, $p(x) = P(X = x)$, with values in $[0, 1]$ that sum to 1. A **PDF** is a density of a continuous variable: probabilities come from integrals, $P(X \in A) = \int_A f(x)\,dx$. Consequences: every single point has probability 0; $f$ can exceed 1 (Uniform$(0, 0.5)$ has density 2); $f$ has units of one over $x$; and $f = F'$ where $F$ is the CDF. Under a change of variables $y = g(x)$, a PMF just relabels outcomes, while a density picks up a Jacobian factor (next subsection). A Gaussian "likelihood" of 3.2 is a density value, not a probability.

**Binomial mean and variance.** $X \sim \mathrm{Bin}(n, p)$ counts successes in $n$ independent trials, $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$.

*With indicators.* Write $X = \sum_{i=1}^n I_i$ with $I_i \sim \mathrm{Bernoulli}(p)$. Then $\mathbb{E}[I_i] = p$, and since $I_i^2 = I_i$, $\mathrm{Var}(I_i) = p - p^2 = p(1-p)$. Linearity gives $\mathbb{E}[X] = np$ with no independence needed; independence makes the covariances vanish, so $\mathrm{Var}(X) = np(1-p)$.

*With direct sums.* Using $k\binom{n}{k} = n\binom{n-1}{k-1}$ and then the binomial theorem,

$$
\mathbb{E}[X] = \sum_{k=1}^{n} n\binom{n-1}{k-1} p^k (1-p)^{n-k} = np \sum_{j=0}^{n-1}\binom{n-1}{j} p^j (1-p)^{n-1-j} = np.
$$

Using $k(k-1)\binom{n}{k} = n(n-1)\binom{n-2}{k-2}$ the same way gives $\mathbb{E}[X(X-1)] = n(n-1)p^2$, so

$$
\mathrm{Var}(X) = \mathbb{E}[X(X-1)] + \mathbb{E}[X] - (\mathbb{E}[X])^2 = n(n-1)p^2 + np - n^2p^2 = np(1-p).
$$

The variance is largest at $p = 1/2$. The fraction $X/n$ has standard deviation $\sqrt{p(1-p)/n}$, which is the error bar on an accuracy measured on $n$ test examples: at 90% accuracy on 10,000 examples it is 0.003, or 0.3 points (Section 8).

### Common distributions

| Distribution | Support | Mean | Variance | Where it shows up in ML |
|---|---|---|---|---|
| Bernoulli$(p)$ | $\{0, 1\}$ | $p$ | $p(1-p)$ | binary labels, logistic regression likelihood, dropout masks |
| Binomial$(n, p)$ | $\{0, \dots, n\}$ | $np$ | $np(1-p)$ | number correct on a test set, hence accuracy error bars |
| Categorical$(\pi)$ | $\{1, \dots, K\}$ | (labels) | (labels) | softmax outputs, next-token prediction, class labels |
| Geometric$(p)$, trials to first success | $\{1, 2, \dots\}$ | $1/p$ | $(1-p)/p^2$ | waiting times; proposals per accepted rejection sample |
| Poisson$(\lambda)$ | $\{0, 1, \dots\}$ | $\lambda$ | $\lambda$ | counts; photon (shot) noise in imaging |
| Uniform$(a, b)$ | $[a, b]$ | $(a+b)/2$ | $(b-a)^2/12$ | inverse-CDF sampling, dequantization noise for discrete data |
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\mathbb{R}$ | $\mu$ | $\sigma^2$ | noise models, MSE loss, VAE latents, diffusion |
| Laplace$(\mu, b)$ | $\mathbb{R}$ | $\mu$ | $2b^2$ | L1 loss; L1 (sparsity) prior |
| Exponential$(\lambda)$ | $[0, \infty)$ | $1/\lambda$ | $1/\lambda^2$ | waiting times; $-\log U$ for uniform $U$ |
| Gamma$(k, \theta)$, shape and scale | $(0, \infty)$ | $k\theta$ | $k\theta^2$ | prior on a precision; sum of $k$ exponentials |
| Beta$(\alpha, \beta)$ | $[0, 1]$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | prior on a probability; conjugate to Bernoulli |
| Dirichlet$(\alpha)$, $\alpha_0 = \sum_j \alpha_j$ | simplex | $\alpha_i/\alpha_0$ | $\frac{\alpha_i(\alpha_0 - \alpha_i)}{\alpha_0^2(\alpha_0+1)}$ | prior on a categorical; topic models; add-one smoothing |
| Student $t_\nu$ | $\mathbb{R}$ | $0$ ($\nu > 1$) | $\frac{\nu}{\nu-2}$ ($\nu > 2$) | heavy-tailed noise; the $t$-test |
| Gumbel$(0, 1)$ | $\mathbb{R}$ | $\gamma \approx 0.5772$ | $\pi^2/6$ | Gumbel-max sampling, Gumbel-softmax |
| $\mathcal{N}(\mu, \Sigma)$ on $\mathbb{R}^d$ | $\mathbb{R}^d$ | $\mu$ | $\Sigma$ | Section 2 |

Every row was checked against `scipy.stats`. Two easily confused conventions: the geometric distribution is sometimes defined as the number of failures before the first success (mean $(1-p)/p$), and the Gamma is sometimes parameterized by a rate $1/\theta$.

### Change of variables and the Jacobian

**One dimension.** Let $Y = g(X)$ with $g$ strictly increasing and differentiable. Then $F_Y(y) = P(g(X) \le y) = P(X \le g^{-1}(y)) = F_X(g^{-1}(y))$, and differentiating by the chain rule,

$$
f_Y(y) = f_X\big(g^{-1}(y)\big)\,\frac{d g^{-1}(y)}{dy}.
$$

For decreasing $g$ the inequality flips and the derivative is negative, so in general the factor is $|d g^{-1}/dy| = 1/|g'(x)|$. The density changes because $g$ stretches or compresses intervals, while the probability in an interval is conserved.

*Example.* $X \sim \mathcal{N}(0, 1)$ and $Y = e^X$ (log-normal): $x = \ln y$, $dx/dy = 1/y$, so $f_Y(y) = \varphi(\ln y)/y$. At $y = 1$ this is $\varphi(0) = 0.399$; at $y = 2$ it is $\varphi(0.693)/2 = 0.157$. *Example.* $U \sim \mathrm{Uniform}(0, 1)$ and $Y = -\ln U$: $u = e^{-y}$, $|du/dy| = e^{-y}$, so $f_Y(y) = e^{-y}$, an Exponential(1). This is inverse-CDF sampling (Section 5) seen from the other side.

**Many dimensions.** Let $y = g(x)$ with $g: \mathbb{R}^d \to \mathbb{R}^d$ invertible and differentiable, and let $J_g(x)$ with entries $\partial g_i / \partial x_j$ be its Jacobian. Near $x$, $g$ acts like the linear map $J_g(x)$, which sends a small box of volume $dV$ to a parallelepiped of volume $|\det J_g(x)|\,dV$. Equating the probability in the two regions, $p_Y(y)\,|\det J_g(x)|\,dV = p_X(x)\,dV$, so

$$
p_Y(y) = p_X(x)\,\lvert\det J_g(x)\rvert^{-1}, \qquad x = g^{-1}(y), \qquad \text{equivalently} \quad \log p_Y(y) = \log p_X(x) - \log\lvert\det J_g(x)\rvert .
$$

For an affine map $y = Ax + b$ the Jacobian is $A$ everywhere, so $\log p_Y(y) = \log p_X(A^{-1}(y - b)) - \log|\det A|$.

**Why normalizing flows need $\log|\det J|$.** A flow generates $x = f_\theta(z)$ from $z \sim \mathcal{N}(0, I)$ with $f_\theta$ invertible, and trains by maximum likelihood, which needs the exact density of $x$:

$$
\log p_\theta(x) = \log p_Z(z) - \log\lvert\det J_{f_\theta}(z)\rvert, \qquad z = f_\theta^{-1}(x).
$$

For a composition $f = f_K \circ \dots \circ f_1$, determinants multiply, so the log-determinants add over layers. A general $d \times d$ determinant costs $O(d^3)$, far too much for an image with $d = 3 \times 64 \times 64 = 12{,}288$, so flow layers are designed to have cheap determinants. An affine coupling layer (RealNVP) keeps half of the coordinates, $y_{1:k} = x_{1:k}$, and transforms the rest with scales and shifts computed from the first half, $y_{k+1:d} = x_{k+1:d} \odot \exp(s(x_{1:k})) + t(x_{1:k})$. Its Jacobian is triangular, so $\log|\det J| = \sum_j s_j(x_{1:k})$, and it is inverted by subtracting $t$ and dividing by $\exp(s)$, with no need to invert the networks $s$ and $t$. In continuous-time flows, $dx/dt = v(x, t)$, the log-density evolves by the instantaneous change of variables $\frac{d}{dt}\log p(x_t) = -\mathrm{tr}\big(\partial v/\partial x\big)$, and the trace is estimated with Hutchinson's estimator (Section 6), as in FFJORD; [flow matching](flow-matching.html) trains such a $v$ without simulating it.

**Jacobians in backpropagation.** Reverse-mode differentiation computes vector-Jacobian products $u^\top J$ layer by layer without ever forming $J$; that is why a gradient of a scalar loss costs a small constant times a forward pass.

### Classic puzzles

These come up in phone screens and first rounds. The tools are first-step analysis (condition on the first move and solve the resulting linear equations), indicators with linearity, and order statistics.

#### Expected flips until HH

Let $E_0$ be the expected number of further flips when you have no progress and $E_1$ the expected number when the last flip was H. Conditioning on the next flip,

$$
E_0 = 1 + \tfrac12 E_1 + \tfrac12 E_0, \qquad E_1 = 1 + \tfrac12 \cdot 0 + \tfrac12 E_0 .
$$

The first gives $E_0 = 2 + E_1$; substituting into the second, $E_1 = 1 + \tfrac12(2 + E_1)$, so $E_1 = 4$ and $E_0 = 6$. For HT the answer is 4: once you have an H, a further H leaves you still holding an H, whereas for HH a T after H throws away all progress. With heads probability $p$ the same two equations give $E_0 = (1+p)/p^2$, which is 6 at $p = 1/2$. This also answers the follow-up to the disease question: if each test is positive independently with probability $p$, the expected number of tests until two consecutive positives is $(1+p)/p^2$. A simulation with $10^5$ runs gives 5.99 for HH and 3.99 for HT.

#### Coupon collector

There are $n$ equally likely types; how many draws until you have seen all of them? Split the time into stages: stage $k$ starts when $k - 1$ types have been seen and ends at the next new type. Each draw in stage $k$ is new with probability $p_k = (n - k + 1)/n$, so the stage length is geometric with mean $1/p_k$. By linearity,

$$
\mathbb{E}[T] = \sum_{k=1}^n \frac{n}{n - k + 1} = n\sum_{j=1}^n \frac1j = n H_n \approx n\ln n + 0.5772\,n + \tfrac12 .
$$

For a die ($n = 6$) this is $6 \times 2.45 = 14.7$ rolls; for $n = 50$ it is 225; for $n = 1000$ classes it is about 7,485 samples before every class has appeared. The stages are independent, so $\mathrm{Var}(T) = \sum_k (1 - p_k)/p_k^2$, which is 39.0 for the die (standard deviation 6.2).

#### Sampling uniformly from a stream (reservoir sampling)

Items arrive one at a time, the length $n$ is unknown, and you may keep only $k$ items in memory. Keep the first $k$. When item $i$ arrives (counting from $i = 0$, so it is the $(i+1)$-th item), with probability $k/(i+1)$ let it replace a uniformly chosen slot.

```python
import random

def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream):        # i = 0, 1, 2, ...
        if i < k:
            reservoir.append(item)           # fill the first k slots
        else:
            j = random.randint(0, i)         # uniform on {0, ..., i}, i.e. i + 1 choices
            if j < k:                        # happens with probability k / (i + 1)
                reservoir[j] = item          # evict a uniformly chosen slot
    return reservoir

counts = [0] * 10
for _ in range(100_000):
    for x in reservoir_sample(range(10), 3):
        counts[x] += 1
print([round(c / 100_000, 3) for c in counts])   # each about 3/10
```

The output is ten values, each within 0.005 of 0.3. **Proof by induction** that after $m$ items each is in the reservoir with probability $k/m$: true for $m = k$. Suppose it holds for $m$. The new item enters with probability $k/(m+1)$. An old item that is in the reservoir is evicted only if the new item enters and picks its slot, probability $\frac{k}{m+1}\cdot\frac1k = \frac{1}{m+1}$. So it is still there with probability $\frac{k}{m}\cdot\frac{m}{m+1} = \frac{k}{m+1}$. The algorithm takes one pass, $O(n)$ time and $O(k)$ memory. For weighted sampling, give each item the key $u^{1/w}$ with $u$ uniform and keep the $k$ largest keys (Efraimidis and Spirakis, 2006).

#### Who wins the coin game?

Two players alternate tossing a coin with heads probability $p$; the first to get heads wins. The first player wins on toss 1, 3, 5, and so on, that is after an even number $2k$ of tails followed by a head:

$$
P = \sum_{k=0}^{\infty} (1-p)^{2k}\,p = \frac{p}{1 - (1-p)^2} = \frac{p}{p(2-p)} = \frac{1}{2-p} ,
$$

which is $2/3$ for a fair coin. First-step analysis gives the same answer faster: if $P$ is the probability that the player about to toss wins, then $P = p + (1-p)(1 - P)$, so $P(2 - p) = 1$. Checks: $P \to 1/2$ as $p \to 0$ (the game is long and moving first barely matters) and $P \to 1$ as $p \to 1$. The expected number of tosses is $1/p = 2$, and with three players the first wins with probability $p/(1 - (1-p)^3) = 4/7$.

#### Maxima of dice and uniforms

The trick for a maximum is that $\{\max \le x\}$ means every variable is at most $x$, so for i.i.d. variables $P(M \le x) = F(x)^n$.

*Max of $n$ uniforms.* $P(M \le x) = x^n$ on $[0, 1]$, so by the tail-sum formula

$$
\mathbb{E}[M] = \int_0^1 (1 - x^n)\,dx = 1 - \frac{1}{n+1} = \frac{n}{n+1}.
$$

More generally the $k$-th smallest of $n$ uniforms has mean $k/(n+1)$: the $n$ points cut $[0,1]$ into $n+1$ gaps of equal expected length.

*Player A throws a die $n$ times and keeps the maximum $X$; player B throws once, getting $Y$.* $P(X \le k) = (k/6)^n$, and conditioning on B's throw,

$$
P(Y > X) = \sum_{y=1}^{6} \frac16\,P(X \le y - 1) = \frac16 \sum_{k=0}^{5} \Big(\frac{k}{6}\Big)^n .
$$

For $n = 1$ this is $15/36 = 5/12 \approx 0.417$, the familiar two-dice answer; for $n = 2$ it is $55/216 \approx 0.255$, for $n = 3$ it is $25/144 \approx 0.174$, and it keeps falling. The expected maximum of two dice is $\sum_k k\,\frac{2k-1}{36} = 161/36 \approx 4.47$.

#### The length of a random chord (Bertrand's paradox)

"Random chord" is ambiguous, and different natural sampling schemes give different answers. For radius $R$, a chord at distance $r$ from the centre has length $L = 2\sqrt{R^2 - r^2}$; the schemes differ in the distribution of the chord.

1. **Two uniform endpoints on the circle.** Fix one endpoint by symmetry; the central angle $\theta$ to the other is uniform on $[0, \pi]$ after folding, and $L = 2R\sin(\theta/2)$, so $\mathbb{E}[L] = \frac1\pi\int_0^\pi 2R\sin\frac\theta2\,d\theta = \frac{2R}{\pi}\big[-2\cos\frac\theta2\big]_0^\pi = \frac{4R}{\pi} \approx 1.27R$.
2. **Uniform distance from the centre.** Pick a random radius and a point uniform along it, and take the chord perpendicular to the radius there, so $r \sim U[0, R]$: $\mathbb{E}[L] = \frac1R\int_0^R 2\sqrt{R^2 - r^2}\,dr = \frac2R\cdot\frac{\pi R^2}{4} = \frac{\pi R}{2} \approx 1.57R$, since the integral is a quarter-disk area.
3. **Midpoint uniform in the disk.** The midpoint's distance has density $2r/R^2$: $\mathbb{E}[L] = \int_0^R 2\sqrt{R^2 - r^2}\,\frac{2r}{R^2}\,dr = \frac{4}{R^2}\big[-\frac13(R^2 - r^2)^{3/2}\big]_0^R = \frac{4R}{3} \approx 1.33R$.

A Monte Carlo check with $2 \times 10^6$ chords per scheme gives 1.273, 1.571 and 1.333 for $R = 1$. The classic version asks for the probability that the chord is longer than the side $\sqrt3 R$ of the inscribed equilateral triangle: $1/3$, $1/2$ and $1/4$. State your assumption first; the point being tested is that "uniform" needs a specified measure.

#### Two non-probability warm-ups

These appear in the same rounds and use the same habits: reduce to one function, check monotonicity, test small cases.

*Which is larger, $a^b$ or $b^a$?* For $a, b > 0$, take logs and divide by $ab$: $a^b > b^a \iff \ln a / a > \ln b / b$. So compare $f(x) = \ln x/x$, whose derivative $(1 - \ln x)/x^2$ is positive below $e$ and negative above, with maximum $1/e$ at $x = e$. Hence for $e \le a < b$ the smaller base wins ($3^4 = 81 > 4^3 = 64$; $10^{11} > 11^{10}$); for $0 < a < b \le e$ the larger base wins; and $e^\pi > \pi^e$ because $f$ peaks at $e$ (23.14 versus 22.46). When the two numbers straddle $e$, evaluate $f$: $f(2) = f(4)$, so $2^4 = 4^2$, and for $b > e$, $2^b > b^2$ exactly when $b > 4$ ($2^3 = 8 < 9$ but $2^5 = 32 > 25$). If $a < 1 < b$, then $a^b < 1 < b^a$ directly. Large exponents reduce the same way: $3^{100}$ versus $100^3$ compares $100\ln 3 \approx 110$ with $3\ln 100 \approx 13.8$.

*100 lamps, student $k$ toggles every multiple of $k$.* Lamp $n$ is toggled once per divisor of $n$, so it ends on exactly when $n$ has an odd number of divisors. Divisors pair up as $(d, n/d)$, and the pair is a single divisor only when $d = \sqrt n$, so the lamps left on are the perfect squares $1, 4, 9, \dots, 100$: ten lamps, and $\lfloor\sqrt N\rfloor$ for $N$ lamps. Equivalently, $n = \prod p_i^{e_i}$ has $\prod (e_i + 1)$ divisors, odd exactly when every $e_i$ is even. Lamps toggled exactly twice are the primes, 25 of them up to 100. A simulation costs $\sum_k N/k = O(N\log N)$.

### Practice questions

::: {.qa asked="ByteDance algorithm intern, Nowcoder 2020"}
**A disease has 1% prevalence and a test is 99% accurate. If you test positive, how likely are you to be ill?**

Take sensitivity and specificity both 0.99. Then $P(+) = 0.01 \cdot 0.99 + 0.99 \cdot 0.01 = 0.0198$ and $P(\text{ill} \mid +) = 0.0099/0.0198 = 0.5$: the healthy group is 99 times larger, so its 1% false positives equal the ill group's true positives. A second independent positive multiplies the prior odds (now $1:1$) by the likelihood ratio 99, giving $0.99$. Follow-up on repeated testing: if each test is positive independently with probability $p$, first-step analysis on the states "no positive yet" and "last test positive" gives an expected $(1+p)/p^2$ tests until two consecutive positives (6 for $p = 1/2$).
:::

::: {.qa asked="Google DeepMind research intern, Glassdoor"}
**What is the difference between a PDF and a PMF? Derive the mean and variance of the binomial distribution.**

A PMF gives probabilities of a discrete variable, $p(x) = P(X = x) \in [0, 1]$, summing to 1. A PDF is a density: probabilities are integrals $\int_A f$, single points have probability 0, $f$ can exceed 1 (Uniform$(0, 0.5)$ has density 2), it has units of $1/x$, $f = F'$, and under $y = g(x)$ it picks up the Jacobian factor $|dx/dy|$ while a PMF only relabels.

For $X \sim \mathrm{Bin}(n, p)$ write $X = \sum_i I_i$ with independent Bernoulli$(p)$ indicators. $\mathbb{E}[I_i] = p$ and $\mathrm{Var}(I_i) = p - p^2$ because $I_i^2 = I_i$. Linearity gives $\mathbb{E}X = np$; independence removes covariances, so $\mathrm{Var}(X) = np(1-p)$. Directly: $k\binom nk = n\binom{n-1}{k-1}$ turns $\sum_k k\binom nk p^k(1-p)^{n-k}$ into $np$ times a binomial sum equal to 1; the same identity twice gives $\mathbb{E}[X(X-1)] = n(n-1)p^2$, and $\mathrm{Var}(X) = \mathbb{E}[X(X-1)] + \mathbb{E}X - (\mathbb{E}X)^2 = np(1-p)$. The variance is largest at $p = 1/2$, and $\sqrt{p(1-p)/n}$ is the standard error of a measured accuracy.
:::

::: {.qa asked="Google DeepMind research intern, Glassdoor"}
**What is a Jacobian, and where does it show up in generative modeling?**

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J_{ij} = \partial f_i / \partial x_j$ is the best local linear approximation of $f$, and $|\det J|$ is the factor by which $f$ scales small volumes. For an invertible $y = f(x)$, conservation of probability gives $p_Y(y) = p_X(x)\,|\det J_f(x)|^{-1}$, so $\log p_Y(y) = \log p_X(x) - \log|\det J_f(x)|$. Normalizing flows train on this exact log-likelihood, so their layers are built to make the determinant cheap: coupling layers have triangular Jacobians with $\log|\det J| = \sum_j s_j$. Continuous-time flows use $\frac{d}{dt}\log p(x_t) = -\mathrm{tr}(\partial v/\partial x)$, with the trace estimated stochastically. Backpropagation computes vector-Jacobian products without ever forming $J$.
:::

## 2. Gaussians

Almost every closed-form computation in diffusion models, VAEs, Kalman filters and Gaussian processes reduces to five facts about Gaussians: affine maps, sums, conditioning, products of densities and the KL formula. This section derives each one and then uses them to rebuild the DDPM forward process and posterior.

### The density

For $x \in \mathbb{R}^d$ with mean $\mu$ and covariance $\Sigma \succ 0$,

$$
\mathcal{N}(x; \mu, \Sigma) = (2\pi)^{-d/2}\,(\det\Sigma)^{-1/2} \exp\!\Big(-\tfrac12 (x - \mu)^\top \Sigma^{-1} (x - \mu)\Big),
$$

so the log-density is $-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu) - \tfrac12\log\det\Sigma - \tfrac d2 \log 2\pi$. The quadratic form is the squared Mahalanobis distance. Its level sets are ellipsoids whose axes are the eigenvectors of $\Sigma$, with semi-axes proportional to $\sqrt{\lambda_i}$. In the isotropic case $\Sigma = \sigma^2 I$,

$$
\log \mathcal{N}(x; \mu, \sigma^2 I) = -\frac{\lVert x - \mu\rVert^2}{2\sigma^2} - d\log\sigma - \frac d2 \log 2\pi ,
$$

so minimizing MSE is maximizing a Gaussian likelihood with fixed $\sigma$.

**Completing the square.** One identity does most of the work below. For symmetric $A \succ 0$,

$$
-\tfrac12 x^\top A x + b^\top x = -\tfrac12 (x - A^{-1}b)^\top A\, (x - A^{-1}b) + \tfrac12 b^\top A^{-1} b ,
$$

which you can check by expanding the right side. So **if a log-density is $-\tfrac12 x^\top A x + b^\top x + \text{const}$, the distribution is $\mathcal{N}(A^{-1}b, A^{-1})$**: read the precision off the quadratic term and the mean as precision-inverse times the linear term.

### Affine maps and sums

If $x \sim \mathcal{N}(\mu, \Sigma)$, then for any matrix $A$ and vector $b$,

$$
Ax + b \sim \mathcal{N}(A\mu + b,\ A\Sigma A^\top).
$$

That the result is Gaussian follows from the characteristic function, $\mathbb{E}[e^{i t^\top(Ax+b)}] = e^{i t^\top b}\,\mathbb{E}[e^{i (A^\top t)^\top x}]$, which has the Gaussian form; the mean and covariance follow from linearity and $\mathrm{Cov}(Ax) = A\Sigma A^\top$. Two uses: sampling, $x = \mu + L\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ and $\Sigma = LL^\top$ (Cholesky); and whitening, $\Sigma^{-1/2}(x - \mu) \sim \mathcal{N}(0, I)$.

**Sums of independent Gaussians.** If $x \sim \mathcal{N}(\mu_1, \Sigma_1)$ and $y \sim \mathcal{N}(\mu_2, \Sigma_2)$ are independent, stack them into one Gaussian vector with block-diagonal covariance and apply the affine map $[I\ \ I]$:

$$
x + y \sim \mathcal{N}(\mu_1 + \mu_2,\ \Sigma_1 + \Sigma_2).
$$

Variances add, standard deviations do not: $\mathcal{N}(0, 1) + \mathcal{N}(0, 4)$ is $\mathcal{N}(0, 5)$, with standard deviation $\sqrt5 \approx 2.24$, not 3.

### Marginals and conditionals

Partition $x = (x_a, x_b)$ with

$$
\mu = \begin{pmatrix} \mu_a \\ \mu_b \end{pmatrix}, \qquad \Sigma = \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}.
$$

**Marginal.** $x_a = [I\ \ 0]\,x$, so by the affine rule $x_a \sim \mathcal{N}(\mu_a, \Sigma_{aa})$: just drop the other blocks.

**Conditional.**

$$
x_a \mid x_b \sim \mathcal{N}\Big(\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b),\ \ \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\Big).
$$

*Derivation by the residual trick.* Define $r = x_a - \Sigma_{ab}\Sigma_{bb}^{-1}x_b$, the part of $x_a$ not linearly explained by $x_b$.

1. $(r, x_b)$ is an affine function of $x$, so it is jointly Gaussian.
2. $\mathrm{Cov}(r, x_b) = \Sigma_{ab} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{bb} = 0$. Jointly Gaussian and uncorrelated means independent, so conditioning on $x_b$ does not change the distribution of $r$.
3. The unconditional moments of $r$ are $\mathbb{E}[r] = \mu_a - \Sigma_{ab}\Sigma_{bb}^{-1}\mu_b$ and
$$
\mathrm{Cov}(r) = \Sigma_{aa} - 2\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba} + \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{bb}\Sigma_{bb}^{-1}\Sigma_{ba} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}.
$$
4. Given $x_b$, $x_a = r + \Sigma_{ab}\Sigma_{bb}^{-1}x_b$ is $r$ plus a constant, which gives the stated mean and covariance.

The conditional covariance $\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$ is the **Schur complement** of $\Sigma_{bb}$ in $\Sigma$, written $\Sigma/\Sigma_{bb}$. The block-inverse formula shows it is also the inverse of the $aa$ block of the precision matrix: $(\Sigma^{-1})_{aa} = (\Sigma/\Sigma_{bb})^{-1}$. So the same result follows from completing the square in $x_a$ in the joint log-density, reading the precision $\Lambda_{aa}$ off the quadratic term.

Three facts worth saying out loud: the conditional mean is linear in $x_b$ (Gaussian conditioning is linear regression, and here the linear predictor is the best predictor of any form); the conditional covariance does not depend on the observed value $x_b$; and conditioning never increases uncertainty, since $\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba} \succeq 0$.

*Example 1.* Standard bivariate Gaussian with correlation $\rho = 0.8$; observe $x_b = 1$. Then $x_a \mid x_b \sim \mathcal{N}(0.8,\ 1 - 0.64) = \mathcal{N}(0.8, 0.36)$, a standard deviation of 0.6 instead of 1. *Example 2.* $\mu = (1, 2)$, $\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 3\end{pmatrix}$, observe $x_b = 5$: mean $1 + \frac23(5 - 2) = 3$, variance $4 - \frac{4}{3} = \frac83$.

### Products of Gaussian densities

As functions of $x$,

$$
\mathcal{N}(x; a, A)\,\mathcal{N}(x; b, B) = \mathcal{N}(a; b, A + B)\;\mathcal{N}(x; c, C), \qquad C = (A^{-1} + B^{-1})^{-1}, \quad c = C\,(A^{-1}a + B^{-1}b).
$$

*Derivation.* Add the two exponents. The terms in $x$ are $-\tfrac12 x^\top(A^{-1} + B^{-1})x + (A^{-1}a + B^{-1}b)^\top x$, so by completing the square the $x$-dependence is $\mathcal{N}(x; c, C)$. The leftover factor does not depend on $x$, so it equals the integral of the left side over $x$. Since $\mathcal{N}(x; a, A) = \mathcal{N}(a; x, A)$, that integral is $\int \mathcal{N}(a; x, A)\,\mathcal{N}(x; b, B)\,dx$, the density at $a$ of $x + \xi$ with $x \sim \mathcal{N}(b, B)$ and independent $\xi \sim \mathcal{N}(0, A)$. By the sum rule this is $\mathcal{N}(a; b, A + B)$.

In words: **precisions add, and the mean is the precision-weighted average.** *Example.* $\mathcal{N}(x; 0, 1)\,\mathcal{N}(x; 2, 4)$: precision $1 + \frac14 = 1.25$, so variance 0.8, mean $0.8\,(0/1 + 2/4) = 0.4$, and normalizing constant $\mathcal{N}(0; 2, 5) = 0.1196$; numerical integration agrees with all three. The mean sits closer to 0 because that factor is four times more precise. This is the Bayesian update for a Gaussian prior and a Gaussian likelihood (sensor fusion, the Kalman filter update, Gaussian-Gaussian conjugacy in Section 3), and it is how the DDPM posterior is computed below.

### The reparameterization trick

Training a VAE requires $\nabla_\phi\,\mathbb{E}_{z \sim q_\phi}[f(z)]$, a gradient of an expectation whose distribution depends on the parameters. Sampling $z$ is not differentiable as written. The trick moves the randomness into a fixed distribution:

$$
z = \mu_\phi + \sigma_\phi \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I) \quad\Longrightarrow\quad \nabla_\phi\,\mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{\epsilon}\Big[\nabla_\phi f(\mu_\phi + \sigma_\phi\odot\epsilon)\Big],
$$

and the inner gradient flows through $f$ and the affine map by the chain rule. The alternative that works for any distribution is the **score-function** (REINFORCE) estimator,

$$
\nabla_\phi\,\mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}\big[f(z)\,\nabla_\phi \log q_\phi(z)\big],
$$

obtained from $\nabla_\phi q_\phi = q_\phi \nabla_\phi \log q_\phi$. It uses only values of $f$, not its gradient, and usually has much higher variance.

*Example.* $f(z) = z^2$, $q = \mathcal{N}(\mu, 1)$ at $\mu = 1$; the true gradient of $\mathbb{E}[z^2] = \mu^2 + 1$ is $2\mu = 2$. The reparameterized single-sample estimator is $2(\mu + \epsilon)$, with variance 4. The score-function estimator is $z^2(z - \mu) = \mu^2\epsilon + 2\mu\epsilon^2 + \epsilon^3$; using $\mathbb{E}\epsilon^4 = 3$ and $\mathbb{E}\epsilon^6 = 15$, its second moment is $\mu^4 + 18\mu^2 + 15 = 34$, so its variance is $34 - 4 = 30$. Both are unbiased (a simulation with $10^6$ samples gives means 2.001 and 2.004, variances 4.0 and 30.0), but the score-function estimator needs 7.5 times as many samples for the same accuracy here.

The trick needs $z$ continuous and $f$ differentiable in $z$. For discrete latents, use the score-function estimator with a baseline, a straight-through estimator, or the Gumbel-softmax relaxation (Section 5). The diffusion training input $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$ is the same construction.

### KL divergence between Gaussians

**One dimension.** Let $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$. Then

$$
\log\frac{p(x)}{q(x)} = \log\frac{\sigma_2}{\sigma_1} - \frac{(x - \mu_1)^2}{2\sigma_1^2} + \frac{(x - \mu_2)^2}{2\sigma_2^2}.
$$

Take $\mathbb{E}_p$ using $\mathbb{E}_p[(x - \mu_1)^2] = \sigma_1^2$ and $\mathbb{E}_p[(x - \mu_2)^2] = \sigma_1^2 + (\mu_1 - \mu_2)^2$ (variance plus squared bias, from Section 1):

$$
\mathrm{KL}(p \,\Vert\, q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12 .
$$

*Numbers.* $\mathrm{KL}(\mathcal{N}(0,1) \,\Vert\, \mathcal{N}(1, 4)) = \ln 2 + \frac{1 + 1}{8} - \frac12 = 0.443$, while $\mathrm{KL}(\mathcal{N}(1,4) \,\Vert\, \mathcal{N}(0,1)) = -\ln 2 + \frac{4 + 1}{2} - \frac12 = 1.307$; numerical integration agrees. KL is not symmetric, and it is larger when the wide distribution is measured against the narrow one. The VAE regularizer is the case $q = \mathcal{N}(0, 1)$:

$$
\mathrm{KL}\big(\mathcal{N}(\mu, \sigma^2) \,\Vert\, \mathcal{N}(0, 1)\big) = \tfrac12\big(\mu^2 + \sigma^2 - 1 - \log\sigma^2\big),
$$

which is $\tfrac12(1 + 0.25 - 1 - \ln 0.25) = 0.818$ for $\mu = 1$, $\sigma = 0.5$, summed over latent dimensions.

**Many dimensions.** For $p = \mathcal{N}(\mu_0, \Sigma_0)$ and $q = \mathcal{N}(\mu_1, \Sigma_1)$ on $\mathbb{R}^d$,

$$
\mathrm{KL}(p \,\Vert\, q) = \frac12\Big[\mathrm{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^\top\Sigma_1^{-1}(\mu_1 - \mu_0) - d + \log\frac{\det\Sigma_1}{\det\Sigma_0}\Big].
$$

*Derivation.* $\log p - \log q = \tfrac12\log\frac{\det\Sigma_1}{\det\Sigma_0} - \tfrac12 (x-\mu_0)^\top\Sigma_0^{-1}(x - \mu_0) + \tfrac12 (x - \mu_1)^\top\Sigma_1^{-1}(x-\mu_1)$. The trace trick of Section 6, $\mathbb{E}[x^\top A x] = \mathrm{tr}(A\Sigma) + m^\top A m$ for $x$ with mean $m$ and covariance $\Sigma$, gives $\mathbb{E}_p[(x - \mu_0)^\top\Sigma_0^{-1}(x - \mu_0)] = \mathrm{tr}(I) = d$ and $\mathbb{E}_p[(x - \mu_1)^\top\Sigma_1^{-1}(x - \mu_1)] = \mathrm{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_0 - \mu_1)^\top\Sigma_1^{-1}(\mu_0 - \mu_1)$. Substituting gives the formula.

*Example.* $\mu_0 = 0$, $\Sigma_0 = I_2$, $\mu_1 = (1, 0)$, $\Sigma_1 = \begin{pmatrix} 2 & 0.5 \\ 0.5 & 1\end{pmatrix}$. Then $\det\Sigma_1 = 1.75$ and $\Sigma_1^{-1} = \frac{1}{1.75}\begin{pmatrix} 1 & -0.5 \\ -0.5 & 2\end{pmatrix}$, so the trace is $3/1.75 = 1.714$, the quadratic term is $1/1.75 = 0.571$, and $\ln 1.75 = 0.560$. The KL is $\tfrac12(1.714 + 0.571 - 2 + 0.560) = 0.423$; a Monte Carlo estimate with $2 \times 10^6$ samples gives 0.423.

With equal covariances the trace and log-det terms cancel and $\mathrm{KL} = \tfrac12(\mu_1 - \mu_0)^\top\Sigma^{-1}(\mu_1 - \mu_0)$, which for $\Sigma = \sigma^2 I$ is $\lVert\mu_1 - \mu_0\rVert^2/(2\sigma^2)$. That one line is why diffusion training reduces to a mean-squared error.

### How these facts build a diffusion model

The DDPM forward process adds noise step by step, $x_t = \sqrt{1 - \beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t$ with fresh $\epsilon_t \sim \mathcal{N}(0, I)$. All the closed forms follow from the tools above; [diffusion models](diffusion-models.html) develops them in full.

**The marginal $q(x_t \mid x_0)$ by sums of Gaussians.** Suppose $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1 - \bar\alpha_{t-1}}\,\bar\epsilon$. Then

$$
x_t = \sqrt{1-\beta_t}\sqrt{\bar\alpha_{t-1}}\,x_0 + \Big[\sqrt{1-\beta_t}\sqrt{1 - \bar\alpha_{t-1}}\,\bar\epsilon + \sqrt{\beta_t}\,\epsilon_t\Big].
$$

The bracket is a sum of independent zero-mean Gaussians with variance $(1-\beta_t)(1 - \bar\alpha_{t-1}) + \beta_t = 1 - (1-\beta_t)\bar\alpha_{t-1} = 1 - \bar\alpha_t$, so by induction

$$
q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\ (1 - \bar\alpha_t) I\big), \qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon .
$$

**The posterior $q(x_{t-1} \mid x_t, x_0)$ by a product of Gaussians.** By Bayes' rule and the Markov property, $q(x_{t-1} \mid x_t, x_0) \propto q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)$. As a function of $x_{t-1}$, the first factor $\mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$ is proportional to $\mathcal{N}\big(x_{t-1}; x_t/\sqrt{1-\beta_t},\ \frac{\beta_t}{1-\beta_t} I\big)$, and the second is $\mathcal{N}(x_{t-1}; \sqrt{\bar\alpha_{t-1}}\,x_0, (1-\bar\alpha_{t-1})I)$. Precisions add:

$$
\frac{1-\beta_t}{\beta_t} + \frac{1}{1 - \bar\alpha_{t-1}} = \frac{(1-\beta_t)(1-\bar\alpha_{t-1}) + \beta_t}{\beta_t(1 - \bar\alpha_{t-1})} = \frac{1 - \bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}
\quad\Longrightarrow\quad
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t ,
$$

and the mean is the precision-weighted average,

$$
\tilde\mu_t = \tilde\beta_t\Big(\frac{\sqrt{1-\beta_t}}{\beta_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1 - \bar\alpha_{t-1}}\,x_0\Big) = \frac{\sqrt{1-\beta_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\,x_0 .
$$

*Check by simulation.* With $\bar\alpha_{t-1} = 0.5$ and $\beta_t = 0.2$ (so $\bar\alpha_t = 0.4$), the formulas give coefficients 0.745 on $x_t$ and 0.236 on $x_0$ and variance $\tilde\beta_t = 0.167$. Simulating $4 \times 10^6$ pairs $(x_{t-1}, x_t)$ from a fixed $x_0$ and regressing $x_{t-1}$ on $x_t$ gives slope 0.745 and residual variance 0.167.

**The loss by Gaussian KL.** The DDPM ELBO (Section 4) contains terms $\mathrm{KL}\big(q(x_{t-1} \mid x_t, x_0) \,\Vert\, p_\theta(x_{t-1} \mid x_t)\big)$. With the model's variance fixed to $\sigma_t^2 I$, the Gaussian KL formula gives $\lVert\tilde\mu_t - \mu_\theta(x_t, t)\rVert^2/(2\sigma_t^2)$ plus terms that do not depend on $\theta$ (the trace and log-determinant terms involve only the fixed variances), and rewriting $\tilde\mu_t$ in terms of $\epsilon$ yields the noise-prediction MSE. One more Gaussian formula appears in evaluation: FID is the Fréchet distance between Gaussians fitted to features, $\lVert\mu_1 - \mu_2\rVert^2 + \mathrm{tr}\big(\Sigma_1 + \Sigma_2 - 2(\Sigma_1\Sigma_2)^{1/2}\big)$ (see [evaluation metrics](evaluation-metrics.html)).

::: key
**How to say it in an interview.**

"For Gaussians I use three rules. Affine maps and independent sums stay Gaussian, with means and covariances adding. Conditioning gives a mean that is linear in the observation and a covariance that is the Schur complement. Multiplying two densities adds precisions and precision-weights the means. The DDPM marginal is the sum rule applied $t$ times, the posterior is the product rule, and the per-step KL with matched variances is a squared error between means."
:::

### Practice questions

::: qa
**Derive $q(x_t \mid x_0)$ for DDPM. Why do variances, not standard deviations, add?**

Induct on $t$: if $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\bar\epsilon$, then $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t$ has signal coefficient $\sqrt{\bar\alpha_t}$ and noise equal to a sum of two independent zero-mean Gaussians with variances $(1-\beta_t)(1-\bar\alpha_{t-1})$ and $\beta_t$, whose total is $1 - \bar\alpha_t$. Variances of independent variables add because the covariance term in $\mathrm{Var}(X + Y)$ is zero; standard deviations would add only for perfectly correlated noise.
:::

::: qa
**Why can't you backpropagate through sampling, and how does the reparameterization trick fix it? When does it not apply?**

The sample $z \sim q_\phi$ is drawn by a procedure whose randomness depends on $\phi$, so there is no path from $z$ back to $\phi$. Writing $z = \mu_\phi + \sigma_\phi\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ makes $z$ a differentiable function of $\phi$ given fixed noise, so $\nabla_\phi\mathbb{E}[f(z)] = \mathbb{E}_\epsilon[\nabla_\phi f(\mu_\phi + \sigma_\phi\epsilon)]$. It usually has far lower variance than the score-function estimator $\mathbb{E}[f\,\nabla_\phi\log q_\phi]$ (variance 4 against 30 for $f = z^2$ at $\mu = 1$). It does not apply to discrete $z$ or non-differentiable $f$; then use the score-function estimator with a baseline, straight-through gradients or Gumbel-softmax.
:::

::: qa
**$(x_1, x_2)$ is standard bivariate Gaussian with correlation $\rho$. What is the distribution of $x_1$ given $x_2$, and of $x_1 + x_2$?**

$x_1 \mid x_2 \sim \mathcal{N}(\rho x_2, 1 - \rho^2)$ by the conditioning formula with $\Sigma_{ab} = \rho$ and $\Sigma_{bb} = 1$. The sum is the affine map $[1\ \ 1]$, so $x_1 + x_2 \sim \mathcal{N}(0, 2 + 2\rho)$. At $\rho = 0.8$ the conditional standard deviation is 0.6, and the sum has variance 3.6.
:::

## 3. Estimation

### Maximum likelihood and MAP

Given i.i.d. data $x_1, \dots, x_n$ and a model $p(x \mid \theta)$, the **maximum likelihood estimate** is $\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_i \log p(x_i \mid \theta)$. Dividing by $n$, the objective is a Monte Carlo estimate of $\mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)] = -H(p_{\text{data}}) - \mathrm{KL}(p_{\text{data}} \,\Vert\, p_\theta)$ (Section 4), so maximum likelihood minimizes the forward KL from the data distribution to the model.

*Bernoulli.* With $k$ heads in $n$ flips, $\ell(p) = k\log p + (n-k)\log(1-p)$ and $\ell'(p) = k/p - (n-k)/(1-p) = 0$ gives $\hat p = k/n$.

*Gaussian.* $\ell(\mu, \sigma^2) = -\frac n2\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2$. Setting $\partial_\mu \ell = \frac{1}{\sigma^2}\sum_i(x_i - \mu) = 0$ gives $\hat\mu = \bar x$; setting $\partial_{\sigma^2}\ell = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_i(x_i - \mu)^2 = 0$ gives $\hat\sigma^2 = \frac1n\sum_i(x_i - \bar x)^2$, which is biased (below). The multivariate version is derived with matrix calculus in Section 6.

Standard losses are negative log-likelihoods: MSE for a Gaussian with fixed variance, L1 for a Laplace, cross-entropy for a categorical, binary cross-entropy for a Bernoulli (more on [deep learning training](deep-learning-training.html)).

The **MAP estimate** adds a prior: $\hat\theta_{\text{MAP}} = \arg\max_\theta \big[\sum_i \log p(x_i \mid \theta) + \log p(\theta)\big]$. A Gaussian prior $\mathcal{N}(0, \tau^2 I)$ contributes $-\lVert\theta\rVert^2/(2\tau^2)$, which is L2 regularization (weight decay); for linear regression with noise variance $\sigma^2$ this is ridge regression with $\lambda = \sigma^2/\tau^2$. A Laplace prior contributes $-\lVert\theta\rVert_1/b$, which is the L1 penalty and favours exact zeros. Two caveats: MAP is a point estimate and ignores posterior spread, and unlike the MLE it is not invariant to reparameterization, because the mode of a density moves under a nonlinear change of variables (the Jacobian factor of Section 1 reshapes it).

### Bias, variance and MSE of an estimator

For an estimator $\hat\theta$ of $\theta$, add and subtract $\mathbb{E}\hat\theta$:

$$
\mathrm{MSE}(\hat\theta) = \mathbb{E}\big[(\hat\theta - \theta)^2\big] = \underbrace{\mathbb{E}\big[(\hat\theta - \mathbb{E}\hat\theta)^2\big]}_{\mathrm{Var}(\hat\theta)} + \underbrace{(\mathbb{E}\hat\theta - \theta)^2}_{\mathrm{Bias}^2},
$$

since the cross term $2(\mathbb{E}\hat\theta - \theta)\,\mathbb{E}[\hat\theta - \mathbb{E}\hat\theta]$ vanishes. This is the same computation as "the mean minimizes squared error", and the model-level bias-variance decomposition on [deep learning training](deep-learning-training.html) is the same algebra applied to predictions.

Unbiased is not the same as best. For Gaussian data and estimators $c\sum_i(x_i - \bar x)^2$ of $\sigma^2$, the choice $c = 1/(n-1)$ is unbiased with MSE $2\sigma^4/(n-1)$, the MLE $c = 1/n$ has MSE $(2n-1)\sigma^4/n^2$, and $c = 1/(n+1)$ minimizes the MSE at $2\sigma^4/(n+1)$. For $n = 5$ and $\sigma = 1$ these are 0.50, 0.36 and 0.33, and a simulation with $10^6$ datasets agrees to two decimals. Accepting a little bias buys a larger drop in variance, which is the idea behind shrinkage and regularization.

### Why the sample variance divides by $n - 1$

Let $x_1, \dots, x_n$ be i.i.d. with mean $\mu$ and variance $\sigma^2$, and $\bar x$ their mean. Expand around $\mu$:

$$
\sum_{i=1}^n (x_i - \bar x)^2 = \sum_{i=1}^n (x_i - \mu)^2 - n(\bar x - \mu)^2 ,
$$

because $\sum_i (x_i - \mu)^2 = \sum_i (x_i - \bar x)^2 + 2(\bar x - \mu)\sum_i(x_i - \bar x) + n(\bar x - \mu)^2$ and the middle sum is zero. Take expectations: $\mathbb{E}[(x_i - \mu)^2] = \sigma^2$ and $\mathbb{E}[(\bar x - \mu)^2] = \mathrm{Var}(\bar x) = \sigma^2/n$, so

$$
\mathbb{E}\Big[\sum_i (x_i - \bar x)^2\Big] = n\sigma^2 - n\cdot\frac{\sigma^2}{n} = (n-1)\,\sigma^2 .
$$

Dividing by $n - 1$ gives an unbiased estimator $s^2$; dividing by $n$ underestimates by the factor $(n-1)/n$, which is 0.8 for $n = 5$ (a simulation with $10^6$ samples of size 5 gives 0.79998). The intuition: $\bar x$ is fitted to the same data, so the data sit closer to $\bar x$ than to the true $\mu$. Geometrically, the residuals $x_i - \bar x$ sum to zero, so they live in an $(n-1)$-dimensional subspace: one degree of freedom was spent estimating the mean.

::: pitfall
$s^2$ is unbiased for $\sigma^2$, but $s = \sqrt{s^2}$ is biased low for $\sigma$, because $\sqrt{\cdot}$ is concave and Jensen gives $\mathbb{E}[s] \le \sqrt{\mathbb{E}[s^2]} = \sigma$. In code, `numpy.var` defaults to dividing by $n$ (`ddof=0`) while `torch.var` defaults to $n - 1$; batch normalization uses the biased batch variance to normalize during training.
:::

### Consistency, Fisher information and the Cramér-Rao bound

An estimator is **consistent** if $\hat\theta_n \to \theta$ in probability as $n \to \infty$. Unbiasedness and consistency are different properties: $\hat\theta = x_1$ (ignore all data but the first point) is unbiased for the mean but not consistent, and the MLE $\hat\sigma^2$ is biased but consistent. Under regularity conditions the MLE is consistent and asymptotically normal: $\sqrt n(\hat\theta_n - \theta) \to \mathcal{N}(0, I(\theta)^{-1})$.

The **Fisher information** of one observation is

$$
I(\theta) = \mathbb{E}\big[(\partial_\theta \log p(x \mid \theta))^2\big] = -\mathbb{E}\big[\partial_\theta^2 \log p(x \mid \theta)\big],
$$

where the equality holds under regularity conditions. The score $\partial_\theta\log p$ has mean zero, since $\mathbb{E}[\partial_\theta\log p] = \int \partial_\theta p\,dx = \partial_\theta \int p\,dx = 0$, so $I(\theta)$ is the score's variance. For a Bernoulli, $\partial_p^2\log p(x \mid p) = -x/p^2 - (1-x)/(1-p)^2$, whose expectation is $-1/p - 1/(1-p)$, so $I(p) = 1/(p(1-p))$. For the mean of $\mathcal{N}(\mu, \sigma^2)$, $I(\mu) = 1/\sigma^2$.

**Cramér-Rao.** Any unbiased estimator from $n$ i.i.d. observations satisfies $\mathrm{Var}(\hat\theta) \ge 1/(nI(\theta))$. *Sketch.* Let $S = \sum_i \partial_\theta\log p(x_i \mid \theta)$, the total score, with mean 0 and variance $nI(\theta)$. Differentiating $\mathbb{E}[\hat\theta] = \theta$ under the integral gives $\mathrm{Cov}(\hat\theta, S) = 1$, and Cauchy-Schwarz gives $1 \le \mathrm{Var}(\hat\theta)\,nI(\theta)$. For a Bernoulli the bound is $p(1-p)/n$, which $\hat p = k/n$ attains: at $p = 0.3$ and $n = 100$ the variance is 0.0021, a standard error of 0.046.

In ML the Fisher information matrix appears as the preconditioner of the natural gradient, as the Gauss-Newton approximation to the Hessian for log-likelihood losses, and as the importance weights in elastic weight consolidation.

### Bayesian posteriors and conjugacy: Beta-Bernoulli

Bayes' rule for parameters is $p(\theta \mid D) \propto p(D \mid \theta)\,p(\theta)$. A prior is **conjugate** to a likelihood when the posterior stays in the prior's family, so updating means updating a few numbers.

**Beta-Bernoulli.** Prior $p \sim \mathrm{Beta}(\alpha, \beta)$, density $\propto p^{\alpha-1}(1-p)^{\beta-1}$. After $k$ successes in $n$ trials the likelihood is $p^k(1-p)^{n-k}$, and multiplying,

$$
p \mid D \sim \mathrm{Beta}(\alpha + k,\ \beta + n - k).
$$

The prior acts like $\alpha$ prior successes and $\beta$ prior failures (pseudo-counts).

*Worked example.* Prior $\mathrm{Beta}(2, 2)$, a mild belief that $p$ is near 1/2. Observe 7 heads in 10 flips. The posterior is $\mathrm{Beta}(9, 5)$, and:

- Posterior mean $9/14 = 0.643$. It is a weighted average of the prior mean and the MLE: with $w = (\alpha + \beta)/(\alpha + \beta + n) = 4/14$, $\frac{4}{14}(0.5) + \frac{10}{14}(0.7) = 0.643$.
- Posterior mode (the MAP estimate) $(9-1)/(9+5-2) = 8/12 = 0.667$. The MLE is $0.7$.
- A 95% equal-tailed credible interval is $[0.386, 0.861]$, from the Beta quantiles. The Wald interval $0.7 \pm 1.96\sqrt{0.21/10}$ is $[0.416, 0.984]$, which shows how poor the normal approximation is at $n = 10$.
- Posterior predictive: $P(\text{next is heads}) = \mathbb{E}[p \mid D] = 0.643$. The probability that the next two are both heads is $\mathbb{E}[p^2 \mid D] = \frac{9 \cdot 10}{14 \cdot 15} = 0.429$, not $0.643^2 = 0.413$, because both future flips share the same uncertain $p$.

Other conjugate pairs worth knowing: Gaussian likelihood with known variance and Gaussian prior on the mean (posterior precision $1/\tau_0^2 + n/\sigma^2$, mean precision-weighted: the product-of-Gaussians rule); Dirichlet-categorical (posterior mean $(\alpha_i + n_i)/(\alpha_0 + N)$, which with all $\alpha_i = 1$ is add-one smoothing); Gamma-Poisson; and Gaussian with a Gamma prior on the precision.

::: key
**How to say it in an interview.**

"MLE maximizes the likelihood, which is the same as minimizing KL from the data to the model. MAP adds a log-prior, so a Gaussian prior is L2 and a Laplace prior is L1. Full Bayes keeps the whole posterior; with a conjugate prior like Beta-Bernoulli the update is just adding counts, and the posterior mean is a weighted average of the prior mean and the MLE that moves toward the MLE as data grow."
:::

### Practice questions

::: qa
**Why does the sample variance divide by $n - 1$? Is the resulting standard deviation unbiased?**

$\sum_i(x_i - \bar x)^2 = \sum_i(x_i - \mu)^2 - n(\bar x - \mu)^2$, whose expectation is $n\sigma^2 - \sigma^2 = (n-1)\sigma^2$, so dividing by $n - 1$ is unbiased; dividing by $n$ is biased by the factor $(n-1)/n$ because $\bar x$ was fitted to the same data. The square root is biased low by Jensen's inequality ($\sqrt{\cdot}$ is concave). Unbiasedness is also not optimality: for Gaussian data, dividing by $n + 1$ gives the smallest MSE.
:::

::: qa
**Show that L2 regularization is MAP estimation with a Gaussian prior. What does L1 correspond to?**

$\log p(\theta \mid D) = \log p(D \mid \theta) + \log p(\theta) + \text{const}$. With $p(\theta) = \mathcal{N}(0, \tau^2 I)$, $\log p(\theta) = -\lVert\theta\rVert^2/(2\tau^2) + \text{const}$, so maximizing the posterior is minimizing the negative log-likelihood plus $\lVert\theta\rVert^2/(2\tau^2)$. For linear regression with Gaussian noise of variance $\sigma^2$, the negative log-likelihood is $\lVert y - X\theta\rVert^2/(2\sigma^2)$; multiplying through by $2\sigma^2$ gives ridge regression, $\lVert y - X\theta\rVert^2 + \lambda\lVert\theta\rVert^2$ with $\lambda = \sigma^2/\tau^2$. A Laplace prior $\propto e^{-\lVert\theta\rVert_1/b}$ gives the L1 penalty, whose kink at zero produces exact zeros.
:::

::: qa
**You observe 0 failures in 20 trials. Estimate the failure probability.**

The MLE is 0, which is overconfident. With a uniform $\mathrm{Beta}(1,1)$ prior the posterior is $\mathrm{Beta}(1, 21)$, with mean $1/22 = 0.045$ and 95% upper bound $1 - 0.05^{1/21} = 0.133$. The frequentist exact one-sided 95% upper bound solves $(1-p)^{20} = 0.05$, giving $p = 0.139$, and the "rule of three" approximation $3/n = 0.15$ comes from $(1-p)^n \approx e^{-np} = 0.05$ with $\ln 20 \approx 3$.
:::

## 4. Information theory

### Entropy, cross-entropy and KL

For a discrete distribution $p$, the **entropy** $H(p) = -\sum_x p(x)\log p(x)$ is the expected surprise $-\log p(x)$; in bits it is the average length of an optimal code. It is nonnegative and at most $\log K$ on $K$ outcomes, with the maximum at the uniform distribution. A fair coin has 1 bit, a coin with $P(\text{heads}) = 0.9$ has 0.469 bits, a uniform choice among 8 has 3 bits, and a fair die has $\log_2 6 = 2.585$ bits. For densities the **differential entropy** $-\int p\log p$ behaves differently: $\mathcal{N}(\mu, \sigma^2)$ has $\tfrac12\log(2\pi e\sigma^2)$, which is 1.419 nats at $\sigma = 1$ and is negative for $\sigma < 1/\sqrt{2\pi e} \approx 0.242$; it also changes under a change of variables. Among densities with a given variance, the Gaussian has the largest differential entropy.

The **cross-entropy** $H(p, q) = -\sum_x p(x)\log q(x)$ is the expected code length when data come from $p$ but the code is built for $q$, and the **KL divergence** is the excess:

$$
\mathrm{KL}(p \,\Vert\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p).
$$

*Example.* $p = (0.5, 0.5)$, $q = (0.9, 0.1)$, in nats: $H(p) = \ln 2 = 0.693$, $H(p, q) = -0.5\ln 0.9 - 0.5\ln 0.1 = 1.204$, so $\mathrm{KL}(p \,\Vert\, q) = 0.511$, while $\mathrm{KL}(q \,\Vert\, p) = 0.9\ln 1.8 + 0.1\ln 0.2 = 0.368$.

Since $H(p)$ does not depend on $q$, minimizing the cross-entropy loss over a model $q$ with labels from $p$ is minimizing $\mathrm{KL}(p \,\Vert\, q)$, which is maximum likelihood.

**Divergences.** A divergence $D(p \,\Vert\, q)$ is nonnegative and zero only when $p = q$; it need not be symmetric or satisfy the triangle inequality. Forward and reverse KL, Jensen-Shannon, total variation and the other f-divergences are divergences. The Wasserstein distance, and MMD with a characteristic kernel, are true metrics.

### KL is nonnegative (Gibbs' inequality)

Jensen's inequality says $\mathbb{E}[\phi(Y)] \le \phi(\mathbb{E}[Y])$ for concave $\phi$, with equality only if $Y$ is constant where it has mass (for strictly concave $\phi$). Apply it with $\phi = \log$ and $Y = q(x)/p(x)$ under $x \sim p$:

$$
-\mathrm{KL}(p \,\Vert\, q) = \mathbb{E}_p\Big[\log\frac{q(x)}{p(x)}\Big] \le \log \mathbb{E}_p\Big[\frac{q(x)}{p(x)}\Big] = \log\sum_{x:\,p(x) > 0} q(x) \le \log 1 = 0 .
$$

Equality in the first step needs $q/p$ constant on the support of $p$, and in the second needs $q$ to put all its mass on that support; together they force $q = p$. So $\mathrm{KL}(p \,\Vert\, q) \ge 0$ with equality iff $p = q$, and equivalently $H(p, q) \ge H(p)$: the cross-entropy loss is minimized by predicting the true distribution, which makes log loss a proper scoring rule.

### Forward versus reverse KL

Fit a simple model $q$ to a complicated target $p$. The two directions of KL behave very differently.

- **Forward KL**, $\mathrm{KL}(p \,\Vert\, q) = \mathbb{E}_p[\log p - \log q]$. Wherever $p$ has mass and $q(x) \to 0$, the term $-\log q(x)$ explodes, so $q$ must cover every region where $p$ has mass ("zero-avoiding", **mode-covering**). It needs samples from $p$ and is what maximum likelihood minimizes.
- **Reverse KL**, $\mathrm{KL}(q \,\Vert\, p) = \mathbb{E}_q[\log q - \log p]$. Wherever $q$ puts mass and $p(x) \to 0$, the term $-\log p(x)$ explodes, so $q$ must avoid regions where $p$ is small ("zero-forcing", **mode-seeking**), but it pays nothing for ignoring parts of $p$. It needs samples from $q$ and evaluations of $p$ up to a constant, which is why variational inference uses it.

**The picture, in words.** Let $p$ have two well-separated humps and let $q$ be a single Gaussian. The forward-KL fit is one wide Gaussian centred between the humps, stretched to cover both, and its peak sits in the valley where $p$ has almost no mass. The reverse-KL fit sits on one hump, matches its width, and ignores the other hump entirely.

**Numbers.** Take $p = \tfrac12\mathcal{N}(-3, 1) + \tfrac12\mathcal{N}(3, 1)$ and $q = \mathcal{N}(m, s^2)$.

*Forward KL.* Only the cross-entropy term depends on $q$: $-\mathbb{E}_p[\log q] = \tfrac12\log(2\pi s^2) + \mathbb{E}_p[(x - m)^2]/(2s^2)$. Setting the $m$-derivative to zero gives $m = \mathbb{E}_p[x] = 0$, and the $s^2$-derivative gives $s^2 = \mathrm{Var}_p(x) = 1 + 9 = 10$ (within-hump variance plus between-hump variance, the law of total variance). So the forward-KL optimum among Gaussians is moment matching, $q = \mathcal{N}(0, 10)$, with $\mathrm{KL}(p \,\Vert\, q) = 0.462$. At $x = 0$ it puts density $0.126$, where $p(0) = 0.0044$. A one-hump fit $\mathcal{N}(3, 1)$ has forward KL 8.31: forward KL heavily punishes a missed mode.

*Reverse KL.* Numerical minimization gives $q \approx \mathcal{N}(2.98, 1.02^2)$ (or its mirror image) with $\mathrm{KL}(q \,\Vert\, p) = 0.689 \approx \ln 2$. The value is about $\ln 2$ because on its hump $p \approx \tfrac12 q$, so $\log(q/p) \approx \log 2$. The moment-matched $\mathcal{N}(0, 10)$ has the larger reverse KL of 0.938: reverse KL punishes mass in the valley more than a missed mode.

**Why it matters in ML.** Maximum-likelihood models (VAEs, autoregressive models, diffusion models trained on the ELBO) minimize a forward KL, so they tend to cover all modes and may put mass between them. GANs do not minimize forward KL and are known for mode dropping. Mean-field variational inference minimizes reverse KL and tends to underestimate posterior variance. These are tendencies, not guarantees: the model family and the optimizer matter as much as the divergence (see [GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html)).

::: key
**How to say it in an interview.**

"Forward KL, $p$ to $q$, is maximum likelihood: it blows up wherever the data have mass and the model has none, so the model spreads out to cover every mode. Reverse KL, $q$ to $p$, blows up wherever the model puts mass the target does not have, so the model locks onto one mode. Fitting one Gaussian to two well-separated bumps, forward KL gives a wide Gaussian in the middle and reverse KL gives a narrow Gaussian on one bump."
:::

### Mutual information

$$
I(X; Y) = \mathrm{KL}\big(p(x, y) \,\Vert\, p(x)\,p(y)\big) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X).
$$

It is nonnegative, zero exactly when $X$ and $Y$ are independent (unlike correlation), and symmetric. For a bivariate Gaussian with correlation $\rho$, $H(X) = \tfrac12\log(2\pi e)$ and, since $X \mid Y$ has variance $1 - \rho^2$, $H(X \mid Y) = \tfrac12\log(2\pi e(1-\rho^2))$, so $I = -\tfrac12\log(1 - \rho^2)$; for $\rho = 0.8$ this is 0.511 nats, or 0.737 bits. The **data processing inequality** says that if $X \to Y \to Z$ is a Markov chain then $I(X; Z) \le I(X; Y)$: processing cannot create information about $X$. Contrastive learning maximizes a lower bound on mutual information: the InfoNCE loss with $N$ candidates satisfies $I(X; Y) \ge \log N - \mathcal{L}_{\text{InfoNCE}}$ (van den Oord et al., 2018), which also shows the bound cannot exceed $\log N$.

### Jensen-Shannon divergence and GANs

$$
\mathrm{JS}(p, q) = \tfrac12\mathrm{KL}(p \,\Vert\, m) + \tfrac12\mathrm{KL}(q \,\Vert\, m), \qquad m = \tfrac12(p + q).
$$

It is symmetric, bounded by $\log 2$, always finite, and its square root is a metric. **The link to GANs.** The GAN value function is $V(D, G) = \mathbb{E}_{p}[\log D(x)] + \mathbb{E}_{q}[\log(1 - D(x))]$ with $p$ the data and $q$ the generator's distribution. For fixed $G$, maximize the integrand pointwise: $a\log D + b\log(1 - D)$ is maximized at $D = a/(a+b)$, so $D^*(x) = p(x)/(p(x) + q(x))$. Substituting and writing $p + q = 2m$,

$$
V(D^*, G) = \mathbb{E}_p\Big[\log\frac{p}{2m}\Big] + \mathbb{E}_q\Big[\log\frac{q}{2m}\Big] = \mathrm{KL}(p \,\Vert\, m) + \mathrm{KL}(q \,\Vert\, m) - 2\log 2 = 2\,\mathrm{JS}(p, q) - \log 4 .
$$

So with an optimal discriminator the generator minimizes Jensen-Shannon divergence (Goodfellow et al., 2014). If $p$ and $q$ have disjoint supports, as low-dimensional data and generator manifolds often nearly do early in training, then $\mathrm{JS} = \log 2$ no matter how far apart they are, and its gradient carries no signal about which direction to move. That is one motivation for the Wasserstein GAN (Arjovsky et al., 2017), whose distance shrinks smoothly as the supports approach.

### The ELBO, derived two ways

Take a latent-variable model $p_\theta(x, z) = p(z)\,p_\theta(x \mid z)$ and any distribution $q(z \mid x)$ with the right support (the encoder in a VAE).

**Way 1: Jensen.** Multiply and divide by $q$ inside the integral, then use concavity of $\log$:

$$
\log p_\theta(x) = \log\int p_\theta(x, z)\,dz = \log\mathbb{E}_{q(z \mid x)}\Big[\frac{p_\theta(x, z)}{q(z \mid x)}\Big] \ge \mathbb{E}_{q(z \mid x)}\Big[\log\frac{p_\theta(x, z)}{q(z \mid x)}\Big] =: \mathrm{ELBO}.
$$

**Way 2: the KL decomposition.** $\log p_\theta(x)$ does not depend on $z$, so it equals its own expectation under $q$; then use $p_\theta(x) = p_\theta(x, z)/p_\theta(z \mid x)$ and multiply and divide by $q$:

$$
\log p_\theta(x) = \mathbb{E}_q\Big[\log\frac{p_\theta(x, z)}{p_\theta(z \mid x)}\Big] = \mathbb{E}_q\Big[\log\frac{p_\theta(x, z)}{q(z \mid x)}\Big] + \mathbb{E}_q\Big[\log\frac{q(z \mid x)}{p_\theta(z \mid x)}\Big] = \mathrm{ELBO} + \mathrm{KL}\big(q(z \mid x) \,\Vert\, p_\theta(z \mid x)\big).
$$

Way 2 says more: the gap in the bound is exactly the KL from $q$ to the true posterior. So for fixed $\theta$, maximizing the ELBO over $q$ is minimizing a reverse KL to the posterior (variational inference), and the bound is tight when $q$ equals the posterior. Writing $p_\theta(x, z) = p(z)\,p_\theta(x \mid z)$ gives the VAE form,

$$
\mathrm{ELBO} = \mathbb{E}_q\big[\log p_\theta(x \mid z)\big] - \mathrm{KL}\big(q(z \mid x) \,\Vert\, p(z)\big),
$$

reconstruction minus a KL regularizer, computed in closed form for Gaussians (Section 2). For a diffusion model, $z = x_{1:T}$ and $q$ is the fixed forward process; the ELBO splits into the per-step Gaussian KLs of Section 2.

*A check you can do by hand.* Let $z \sim \mathcal{N}(0, 1)$ and $x \mid z \sim \mathcal{N}(z, 1)$, so $p(x) = \mathcal{N}(0, 2)$ and, by Gaussian conditioning, $z \mid x \sim \mathcal{N}(x/2, 1/2)$. At $x = 1$, $\log p(1) = -\tfrac12\log(4\pi) - \tfrac14 = -1.516$. Use the prior as the "encoder", $q = \mathcal{N}(0, 1)$. Way 1: the KL to the prior is 0, and $\mathbb{E}_q[\log p(1 \mid z)] = -\tfrac12\log 2\pi - \tfrac12\mathbb{E}[(1 - z)^2] = -0.919 - 1 = -1.919$, so the ELBO is $-1.919$. Way 2: the gap should be $\mathrm{KL}(\mathcal{N}(0, 1) \,\Vert\, \mathcal{N}(0.5, 0.5)) = \log\sqrt{0.5} + \frac{1 + 0.25}{2 \times 0.5} - \frac12 = 0.403$, and indeed $-1.516 + 1.919 = 0.403$.

### Channel capacity in one paragraph

A channel that adds Gaussian noise of variance $\sigma^2$ to an input of power at most $A^2$ carries at most $C = \tfrac12\log_2(1 + A^2/\sigma^2)$ bits per use, and error-correcting codes can approach this rate with vanishing error (Shannon). Mutual information is the quantity being maximized: $C = \max I(X; Y)$ over input distributions. The noisy all_gather question below uses this as its lower bound.

### Practice questions

::: {.qa asked="OpenAI RS, MSR intern, ByteDance (Glassdoor, Nowcoder)"}
**What is a divergence? Compute a KL divergence and relate KL to cross-entropy.**

A divergence $D(p \,\Vert\, q)$ is nonnegative and zero only when $p = q$; it need not be symmetric or satisfy the triangle inequality. Examples: forward and reverse KL, Jensen-Shannon, total variation and the other f-divergences; the Wasserstein distance, and MMD with a characteristic kernel, are true metrics. $\mathrm{KL}(p \,\Vert\, q) = \mathbb{E}_p[\log p/q] \ge 0$ by Jensen, and $H(p, q) = H(p) + \mathrm{KL}(p \,\Vert\, q)$, so minimizing cross-entropy in $q$ with the data distribution $p$ fixed is minimizing KL, which is maximum likelihood. For Gaussians, $\mathrm{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \,\Vert\, \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12$, for example 0.443 nats for $\mathcal{N}(0,1)$ against $\mathcal{N}(1, 4)$ and 1.307 the other way. A discrete example: $p = (0.5, 0.5)$, $q = (0.9, 0.1)$ gives 0.511 nats. Fitting $q$ with forward KL is mode-covering; with reverse KL it is mode-seeking.
:::

::: {.qa asked="OpenAI RE (Aced, ~2025)"}
**Implement all_gather across several nodes. If the links between nodes are noisy, derive a worst-case bound on the number of rounds needed for every node to get every value within a target error, improve the bound, and then use the fact that the values are floating-point numbers to design a better algorithm.**

The report names only the stages and gives no noise model, topology or target, so state the assumptions aloud: a Gaussian channel and a ring. The main contrast is that sending a real value and averaging copies needs rounds proportional to $1/\varepsilon^2$, while sending a float's bits, decoding each exactly and forwarding it needs rounds proportional to the number of bits.

**Setup.** Node $i$ of $n$ holds $x_i \in [-R, R]$. Per round each node sends one symbol $|s| \le A$ to a neighbour, which receives $s + z$ with $z \sim \mathcal{N}(0, \sigma^2)$ independent. Ring all_gather takes $n - 1$ steps; in each, every node forwards to its successor the value it received in the previous step (its own first), for $n(n-1)$ value transmissions in all.

**Naive bound.** Send $(A/R)\,x$ $k$ times, average, forward the estimate. One hop leaves Gaussian error of variance $(R\sigma/A)^2/k$, and $n - 1$ hops add $n-1$ such errors. A Gaussian tail bound and a union bound over the $n(n-1)$ (value, receiver) pairs give

$$
2n(n-1)\exp\Big(-\frac{k\varepsilon^2}{2(n-1)(R\sigma/A)^2}\Big) \le \delta
\iff k \ge 2(n-1)\Big(\frac{R\sigma}{A\varepsilon}\Big)^2 \ln\frac{2n(n-1)}{\delta},
$$

so $(n-1)k = O\big(n^2(R\sigma/A\varepsilon)^2\log(n/\delta)\big)$ rounds.

**Improving it.** Chebyshev would give a factor $1/\delta$; the Gaussian tail gives $\log(1/\delta)$. Fewer hops per value shrink the factor $n-1$ in $k$: to 1 with direct links between all pairs, or about $\log_2 n$ with a tree. But $1/\varepsilon^2$ remains, because averaging reduces error only as $1/\sqrt k$: each extra bit of precision costs four times the rounds.

**Using floats.** A float32 is a 32-bit string. Send each bit as $\pm A$, $r$ times, and decode by the sign of the average: $P(\text{bit wrong}) = Q(A\sqrt r/\sigma) \le \tfrac12 e^{-rA^2/2\sigma^2}$. Decoded bits are forwarded clean, so noise no longer accumulates over hops. A union bound over $b$ bits in each of the $n(n-1)$ transmissions needs $r \ge (2\sigma^2/A^2)\ln\big(b\,n(n-1)/2\delta\big)$, a total of $(n-1)\,b\,r = O\big(nb(\sigma/A)^2\log(nb/\delta)\big)$ rounds, and with probability at least $1 - \delta$ every value arrives exactly. The target error only sets $b$: 32 bits for an exact float, or about $\log_2(2R/\varepsilon)$ in fixed point. With $n = 8$, $\delta = 10^{-3}$, $A = \sigma$ and $R/\varepsilon = 10^3$, averaging needs about $2.3 \times 10^7$ symbols per value even over one hop, while $b = 11$ bits with $r = 26$ repetitions need $11 \times 26 = 286$.

**Further.** Protect bits unequally (a sign or exponent error is catastrophic, low mantissa bits barely matter), and replace repetition with an error-correcting code, which costs a constant number of channel uses per bit. Nothing does much better: pinning a worst-case value to within $\varepsilon$ means distinguishing about $R/\varepsilon$ values, roughly $\log_2(R/\varepsilon)$ bits, and the channel carries at most $C = \tfrac12\log_2(1 + A^2/\sigma^2)$ bits per use, so any scheme needs about $\log_2(R/\varepsilon)/C$ rounds per hop. If the channel instead flips bits with probability $p$, majority voting over $r$ copies fails with probability at most $e^{-2r(1/2 - p)^2}$ (Hoeffding), and the rest carries over.
:::

::: qa
**Maximum likelihood minimizes which KL, and what does that predict about samples from a model trained this way?**

$\frac1n\sum_i\log p_\theta(x_i) \to \mathbb{E}_{p_{\text{data}}}[\log p_\theta] = -H(p_{\text{data}}) - \mathrm{KL}(p_{\text{data}} \,\Vert\, p_\theta)$, so MLE minimizes the forward KL. Forward KL is infinite if the model gives zero density where data exist, so a model with limited capacity tends to spread mass over all modes, including the space between them; samples can look over-dispersed or blurry. A reverse-KL fit tends to do the opposite, sharp samples from fewer modes.
:::

## 5. Sampling and Monte Carlo

### The Monte Carlo estimator

To estimate $\mu = \mathbb{E}_p[f(x)]$, draw $x_1, \dots, x_n \sim p$ i.i.d. and average: $\hat\mu_n = \frac1n\sum_i f(x_i)$. It is unbiased, its variance is $\sigma_f^2/n$ with $\sigma_f^2 = \mathrm{Var}_p(f)$, so its root-mean-square error is $\sigma_f/\sqrt n$, and by the central limit theorem $\hat\mu_n \pm 1.96\,\hat\sigma_f/\sqrt n$ is an approximate 95% confidence interval. Halving the error costs four times the samples. The rate does not depend on the dimension, which is why Monte Carlo beats grid quadrature in high dimension, although $\sigma_f$ itself can grow with dimension.

*Example: estimating $\pi$.* With $u$ uniform on $[0,1]^2$, $f = 4\cdot\mathbf{1}\{u_1^2 + u_2^2 \le 1\}$ has mean $\pi$ and standard deviation $4\sqrt{p(1-p)} = 1.642$ with $p = \pi/4$. With $n = 10^6$ the standard error is 0.0016; one run gave 3.1412. A minibatch gradient is the same kind of estimator, so its noise falls as $1/\sqrt B$ in the batch size.

### Inverse-CDF sampling

If $U \sim \mathrm{Uniform}(0, 1)$ and $F$ is a CDF, then $X = F^{-1}(U)$ has CDF $F$, because $P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$ (for non-strictly-increasing $F$ use the generalized inverse $\inf\{x : F(x) \ge u\}$). For the exponential, $F(x) = 1 - e^{-\lambda x}$ gives $X = -\ln(1 - U)/\lambda$, equivalently $-\ln U/\lambda$. For the Gumbel, $X = -\ln(-\ln U)$. The Gaussian has no closed-form $F^{-1}$, so libraries use Box-Muller or numerical inverses.

**Discrete distributions** are the most common interview version: "sample an index with probability proportional to $w_i$". Build the cumulative sums once, draw $u$ uniform on $[0, \sum_i w_i)$, and binary-search for the first index whose cumulative sum exceeds $u$: $O(n)$ setup and $O(\log n)$ per sample. For many samples from a fixed distribution, the **alias method** gives $O(1)$ per sample after $O(n)$ setup. Scale the probabilities so they average 1, then fill $n$ buckets of height 1: each bucket holds at most two outcomes, its own index with probability `prob[i]` and one "alias" otherwise. Sampling picks a bucket uniformly and flips one biased coin.

```python
import numpy as np
rng = np.random.default_rng(0)

def build_alias(p):                          # Vose's alias method, O(n) setup
    n = len(p)
    prob, alias = np.zeros(n), np.zeros(n, dtype=int)
    scaled = np.asarray(p, float) * n        # average bucket height is 1
    small = [i for i in range(n) if scaled[i] < 1]
    large = [i for i in range(n) if scaled[i] >= 1]
    while small and large:
        s, l = small.pop(), large.pop()
        prob[s], alias[s] = scaled[s], l     # bucket s: keep s w.p. scaled[s], else give l
        scaled[l] -= 1 - scaled[s]           # l donated that much mass
        (small if scaled[l] < 1 else large).append(l)
    for i in small + large:
        prob[i] = 1.0                        # leftovers are full up to rounding
    return prob, alias

def sample_alias(prob, alias, size):         # O(1) per sample: one bucket, one coin
    i = rng.integers(0, len(prob), size)
    return np.where(rng.random(size) < prob[i], i, alias[i])

p = [0.5, 0.3, 0.1, 0.1]
prob, alias = build_alias(p)
print(np.bincount(sample_alias(prob, alias, 10**6)) / 10**6)
cdf = np.cumsum(p)                           # the O(log n) alternative
print(np.bincount(np.searchsorted(cdf, rng.random(10**6) * cdf[-1], side="right"), minlength=4) / 10**6)
```

Both lines print frequencies within 0.001 of $(0.5, 0.3, 0.1, 0.1)$. When the weights change between samples, a Fenwick or segment tree over the weights supports $O(\log n)$ updates and samples.

### Rejection sampling

To sample from a density $p$ known up to a constant, $\tilde p = Zp$, find a proposal $q$ you can sample and a constant $M$ with $\tilde p(x) \le M q(x)$ everywhere. Draw $x \sim q$ and $u \sim \mathrm{Uniform}(0, 1)$, and accept $x$ if $u \le \tilde p(x)/(Mq(x))$; otherwise try again. **Why it works:** the probability of proposing a point in $dx$ and accepting it is $q(x)\,dx \cdot \tilde p(x)/(Mq(x)) = \tilde p(x)\,dx/M$, which is proportional to $p$, so accepted points have density $p$. Integrating, the acceptance rate is $Z/M$ (just $1/M$ if $p$ is normalized), and the number of proposals per accepted sample is geometric with mean $M/Z$.

*Example.* Beta(2, 2) has density $6x(1-x)$ with maximum 1.5 at $x = 1/2$. With a uniform proposal and $M = 1.5$, the acceptance rate is $1/1.5 = 2/3$; a run with $10^6$ proposals accepted 66.8%, and the accepted samples had mean 0.500 and variance 0.050, matching Beta(2, 2). In high dimension the bound $M$ usually grows exponentially: for target $\mathcal{N}(0, I_d)$ and proposal $\mathcal{N}(0, s^2 I_d)$ with $s > 1$, the best constant is $M = s^d$, already $1.1^{100} \approx 13{,}800$ for $s = 1.1$ and $d = 100$.

### Importance sampling

Rewrite the expectation under a proposal $q$ that is positive wherever $f p \ne 0$:

$$
\mathbb{E}_p[f(x)] = \mathbb{E}_q\Big[f(x)\,\frac{p(x)}{q(x)}\Big] \approx \frac1n\sum_{i=1}^n f(x_i)\,w_i, \qquad x_i \sim q,\quad w_i = \frac{p(x_i)}{q(x_i)} .
$$

This is unbiased, and its per-sample variance is

$$
\mathrm{Var}_q(f w) = \int \frac{f(x)^2 p(x)^2}{q(x)}\,dx - \mu^2 .
$$

The variance is smallest for $q^* \propto |f|\,p$ (zero when $f \ge 0$, but then $q^*$ needs the unknown $\mu$ as its normalizer): good proposals put mass where $|f|p$ is large. A bad proposal, especially one with lighter tails than $p$, can make the variance huge or infinite. When $p$ is known only up to a constant, use **self-normalized** importance sampling, $\sum_i w_i f(x_i)/\sum_i w_i$, which is biased but consistent.

*Example: a rare event.* Estimate $P(X > 4) = 3.17 \times 10^{-5}$ for $X \sim \mathcal{N}(0, 1)$. Plain Monte Carlo with $n = 10^5$ expects about 3 hits, a relative error of $\sqrt{(1-\mu)/(n\mu)} = 56\%$. With proposal $q = \mathcal{N}(4, 1)$, the second moment is $\int_4^\infty \varphi(x)^2/\varphi(x - 4)\,dx = e^{16}\,P(Z > 8)$ (complete the square: $-x^2 + (x-4)^2/2 = -(x+4)^2/2 + 16$), giving a per-sample relative standard deviation of 2.12 against 178 for plain Monte Carlo. For the same accuracy, plain Monte Carlo needs about 7,000 times as many samples; a run with $n = 10^5$ gave a relative standard error of 0.67%.

**Effective sample size.** A common diagnostic is

$$
\mathrm{ESS} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2} \in [1, n],
$$

which is $n$ when all weights are equal and 1 when one weight dominates. For target $\mathcal{N}(0, 1)$ and proposal $\mathcal{N}(\delta, 1)$, $\mathbb{E}_q[w^2] = \int \varphi(x)^2/\varphi(x - \delta)\,dx = e^{\delta^2}$, so $\mathrm{ESS}/n \to e^{-\delta^2}$: 0.37 for $\delta = 1$ and 0.018 for $\delta = 2$ (simulations give 0.368 and 0.022). In $d$ dimensions with the same shift in each coordinate the factor becomes $e^{-d\delta^2}$, so importance weights degenerate exponentially in dimension unless $q$ matches $p$ closely. Importance sampling appears in off-policy reinforcement learning, in the importance-weighted ELBO, and in estimating a model's likelihood.

### MCMC: Metropolis-Hastings from detailed balance

When neither a good proposal nor a bound $M$ is available, build a Markov chain whose stationary distribution is the target $\pi$. A transition kernel $T(x' \mid x)$ leaves $\pi$ invariant if $\int \pi(x)\,T(x' \mid x)\,dx = \pi(x')$. A sufficient condition is **detailed balance**,

$$
\pi(x)\,T(x' \mid x) = \pi(x')\,T(x \mid x'),
$$

because integrating both sides over $x$ gives $\int \pi(x)\,T(x' \mid x)\,dx = \pi(x')\int T(x \mid x')\,dx = \pi(x')$. Invariance plus irreducibility and aperiodicity gives convergence to $\pi$ from any start.

**Deriving the acceptance rule.** Propose $x' \sim q(x' \mid x)$ and accept with probability $a(x \to x')$; on rejection stay at $x$. For $x' \ne x$, $T(x' \mid x) = q(x' \mid x)\,a(x \to x')$, so detailed balance requires

$$
\frac{a(x \to x')}{a(x' \to x)} = \frac{\pi(x')\,q(x \mid x')}{\pi(x)\,q(x' \mid x)} =: r(x, x').
$$

The choice $a(x \to x') = \min\big(1, r(x, x')\big)$ satisfies this: if $r \le 1$ then $a(x \to x') = r$ and $a(x' \to x) = \min(1, 1/r) = 1$, whose ratio is $r$, and the case $r > 1$ is symmetric. It is also the largest choice, which makes the chain move as often as possible. The normalizing constant of $\pi$ cancels in $r$, which is why MCMC needs $\pi$ only up to a constant. With a symmetric proposal, such as a Gaussian random walk, the $q$ terms cancel and the rule is Metropolis's $\min(1, \pi(x')/\pi(x))$. A rejected proposal repeats the current state in the output; dropping repeats biases the chain.

```python
import numpy as np
rng = np.random.default_rng(0)

def log_p(x):                                # unnormalized log density of 0.5 N(-4,1) + 0.5 N(4,1)
    return np.logaddexp(-0.5 * (x + 4)**2, -0.5 * (x - 4)**2)

def metropolis(log_p, x0, n, step):
    x, out, acc = x0, np.empty(n), 0
    for t in range(n):
        y = x + step * rng.standard_normal()          # symmetric proposal, so q cancels
        if np.log(rng.random()) < log_p(y) - log_p(x):
            x, acc = y, acc + 1
        out[t] = x
    return out, acc / n

for step in [0.5, 3.0]:
    runs = [metropolis(log_p, 0.0, 50_000, step) for _ in range(5)]
    print(step, "acceptance", round(runs[0][1], 2),
          "chain means", [float(round(s.mean(), 1)) for s, _ in runs])   # true mean 0
```

Output: step 0.5 has acceptance 0.84 but five chains report means $0.6, 4.0, -1.6, -4.0, -0.5$; step 3.0 has acceptance 0.40 and all five means within 0.2 of 0. Small steps are almost always accepted but rarely cross the valley between the modes, so a chain can spend its whole run in one mode. **High acceptance is not good mixing.** For random-walk Metropolis on high-dimensional targets of a particular product form, Roberts, Gelman and Gilks (1997) showed the optimal acceptance rate is about 0.234, a widely used rule of thumb. Diagnose mixing by running several chains from dispersed starts and comparing them (the $\hat R$ statistic) and by computing the chain's effective sample size from its autocorrelations.

### Langevin dynamics and score-based diffusion

The Langevin SDE $dx = \nabla\log\pi(x)\,dt + \sqrt2\,dW$ has $\pi$ as its stationary distribution. *Check with the Fokker-Planck equation*, $\partial_t p = -\nabla\cdot(p\,\nabla\log\pi) + \Delta p$: at $p = \pi$, $\pi\nabla\log\pi = \nabla\pi$, so the right side is $-\nabla\cdot\nabla\pi + \Delta\pi = 0$. Discretizing with step $\epsilon/2$ in time gives the **unadjusted Langevin algorithm**,

$$
x_{k+1} = x_k + \frac{\epsilon}{2}\,\nabla_x\log\pi(x_k) + \sqrt{\epsilon}\,z_k, \qquad z_k \sim \mathcal{N}(0, I),
$$

a gradient step on $\log\pi$ plus noise. It needs only the **score** $\nabla\log\pi$, in which the normalizing constant vanishes. Discretization introduces bias. For $\pi = \mathcal{N}(0, 1)$ the score is $-x$, so $x_{k+1} = (1 - \epsilon/2)x_k + \sqrt\epsilon\,z_k$, and the stationary variance $v$ solves $v = (1 - \epsilon/2)^2 v + \epsilon$:

$$
v = \frac{\epsilon}{1 - (1 - \epsilon/2)^2} = \frac{1}{1 - \epsilon/4},
$$

which is 1.026 rather than 1 at $\epsilon = 0.1$ (a simulation gives 1.026). Using the Langevin step as a proposal inside Metropolis-Hastings removes the bias (MALA; its proposal is not symmetric, so the $q$ ratio stays in $r$). Replacing the exact gradient with a minibatch gradient gives stochastic gradient Langevin dynamics for Bayesian neural networks (Welling and Teh, 2011).

**The link to diffusion.** Score-based generative models learn $s_\theta(x, \sigma) \approx \nabla_x\log p_\sigma(x)$, the score of the data distribution blurred with Gaussian noise of level $\sigma$, by denoising score matching, and then sample with Langevin steps. Song and Ermon (2019) run Langevin at a decreasing sequence of noise levels (annealed Langevin dynamics), for two reasons the toy examples above illustrate: the learned score is inaccurate where data are rare, and Langevin, like the small-step Metropolis chain, mixes slowly between separated modes, while large noise levels connect the modes. The reverse-time SDE of a diffusion model has the same ingredients, a drift containing the score plus injected noise (see [diffusion models](diffusion-models.html)).

### The Gumbel-max trick

To sample from a categorical with probabilities $p_i \propto e^{\ell_i}$, add independent Gumbel noise $g_i = -\log(-\log U_i)$ to the logits and take the argmax: $\arg\max_i(\ell_i + g_i)$ is distributed as the categorical. *Sketch.* Normalize so $\sum_j p_j = 1$ with $\ell_i = \log p_i$ (adding a constant to all logits changes nothing). $G_i = \log p_i + g_i$ has CDF $\exp(-p_i e^{-x})$ and density $p_i e^{-x}\exp(-p_i e^{-x})$, so

$$
P(\text{index } i \text{ wins}) = \int p_i e^{-x} e^{-p_i e^{-x}}\prod_{j \ne i} e^{-p_j e^{-x}}\,dx = p_i\int e^{-x}\exp(-e^{-x})\,dx = p_i ,
$$

using $\sum_j p_j = 1$ and that the last integrand is the Gumbel density. A simulation with logits $\log(0.5, 0.3, 0.2)$ gives frequencies 0.499, 0.301, 0.200. Uses: sampling from unnormalized logits in one pass; taking the top $k$ of the perturbed logits samples $k$ items without replacement; and the **Gumbel-softmax** (Concrete) relaxation $\mathrm{softmax}((\ell + g)/\tau)$, a differentiable approximate one-hot sample that approaches the hard sample as the temperature $\tau \to 0$ (Jang et al., 2017; Maddison et al., 2017).

### Practice questions

::: {.qa asked="ByteDance CV search 2023, Meta RS full-time 2024"}
**Sample an index from a given discrete distribution. How do you make it fast?**

Build the cumulative sums once, draw $u \sim U(0, 1)$, and binary-search for the first index whose cumulative weight exceeds $u$ times the total: $O(n)$ setup and $O(\log n)$ per sample. For many samples from a fixed distribution, the alias method (Walker; Vose's $O(n)$ construction) gives $O(1)$ per sample after $O(n)$ setup: $n$ buckets of equal height, each split between its own index and one alias, so a sample is one uniform bucket choice and one biased coin. When weights change, a Fenwick or segment tree supports $O(\log n)$ updates and samples. The Gumbel-max trick, $\arg\max_i(\log p_i + g_i)$ with Gumbel noise $g_i$, is another exact method, works on unnormalized logits, and underlies the Gumbel-softmax relaxation.
:::

::: qa
**Why does importance sampling break down in high dimension?**

The weights $w = p/q$ are products of per-coordinate ratios, so their variance grows exponentially with dimension unless $q$ matches $p$ almost exactly. For $\mathcal{N}(0, I_d)$ against $\mathcal{N}(\delta\mathbf{1}, I_d)$, $\mathbb{E}_q[w^2] = e^{d\delta^2}$ and the effective sample size fraction is about $e^{-d\delta^2}$: one or a few samples carry all the weight. Rejection sampling fails for the same reason, through a bound $M$ that grows exponentially. MCMC, which moves locally, is the usual alternative.
:::

::: qa
**Your Metropolis chain accepts 95% of proposals. Is that good?**

Usually not. Very high acceptance means the steps are tiny, so successive samples are highly correlated and the chain explores slowly; in the bimodal example, step 0.5 accepted 84% of proposals yet individual chains reported means from $-4$ to $4$ for a target with mean 0. Tune the step for a moderate acceptance rate (about 0.234 is the classic high-dimensional guideline for random-walk Metropolis), and judge mixing with multiple chains, $\hat R$ and the effective sample size rather than the acceptance rate alone.
:::

## 6. Linear algebra

### Eigendecomposition and the spectral theorem

An eigenpair of a square matrix satisfies $Av = \lambda v$ with $v \ne 0$. If $A \in \mathbb{R}^{n \times n}$ has $n$ linearly independent eigenvectors, collect them as the columns of $V$ and write $A = V\Lambda V^{-1}$; then $A^k = V\Lambda^k V^{-1}$, so repeated application of $A$ grows or shrinks each eigen-direction by $|\lambda_i|^k$. That is the linear-algebra core of exploding and vanishing gradients in RNNs, where a recurrent Jacobian is applied once per time step (see [CNNs and RNNs](cnns-rnns.html)). Not every matrix is diagonalizable: $\begin{pmatrix} 0 & 1 \\ 0 & 0\end{pmatrix}$ has the eigenvalue 0 twice but only one eigen-direction. Real matrices can have complex eigenvalues: a 2-D rotation by $\theta$ has $e^{\pm i\theta}$. Two identities hold for every square matrix: $\mathrm{tr}\,A = \sum_i\lambda_i$ and $\det A = \prod_i\lambda_i$.

**Spectral theorem.** A real symmetric matrix has real eigenvalues and an orthonormal basis of eigenvectors: $A = Q\Lambda Q^\top$ with $Q^\top Q = I$. Two parts of the proof are short:

- *Real eigenvalues.* If $Av = \lambda v$ with $v$ possibly complex, then $\bar v^\top A v = \lambda\lVert v\rVert^2$; conjugating and transposing, and using $A = A^\top$ real, the same quantity equals $\bar\lambda\lVert v\rVert^2$. So $\lambda = \bar\lambda$.
- *Orthogonal eigenvectors.* If $Av_1 = \lambda_1 v_1$ and $Av_2 = \lambda_2 v_2$, then $\lambda_1 v_1^\top v_2 = (Av_1)^\top v_2 = v_1^\top A v_2 = \lambda_2 v_1^\top v_2$, so $v_1^\top v_2 = 0$ when $\lambda_1 \ne \lambda_2$.

(The remaining step, that there are $n$ of them even with repeated eigenvalues, goes by induction on the orthogonal complement of one eigenvector.) The **Rayleigh quotient** $x^\top Ax/x^\top x$ of a symmetric matrix lies between $\lambda_{\min}$ and $\lambda_{\max}$, and attains them at the corresponding eigenvectors: write $x = \sum_i c_i q_i$, so the quotient is $\sum_i\lambda_i c_i^2/\sum_i c_i^2$, a weighted average of eigenvalues.

*Example.* $A = \begin{pmatrix} 2 & 1 \\ 1 & 3\end{pmatrix}$ has trace 5 and determinant 5, so $\lambda^2 - 5\lambda + 5 = 0$ and $\lambda = (5 \pm \sqrt5)/2$, that is 3.618 and 1.382.

### Positive semidefinite matrices

For symmetric $A$, these are equivalent definitions of **positive semidefinite** ($A \succeq 0$):

1. $x^\top Ax \ge 0$ for all $x$;
2. all eigenvalues are $\ge 0$ (from $x^\top Ax = \sum_i\lambda_i (q_i^\top x)^2$);
3. $A = B^\top B$ for some $B$ (take $B = \Lambda^{1/2}Q^\top$; conversely $x^\top B^\top Bx = \lVert Bx\rVert^2$).

**Positive definite** ($A \succ 0$) replaces $\ge$ by $>$ for $x \ne 0$, and is equivalent to the existence of a Cholesky factorization $A = LL^\top$ with $L$ lower triangular with positive diagonal, which is also the cheapest numerical test. PSD matrices in ML: covariance matrices, Gram and kernel matrices $K_{ij} = k(x_i, x_j)$, $X^\top X$, the Fisher information, and the Hessian of a convex function. Sylvester's criterion (all leading principal minors positive) certifies positive definiteness, but semidefiniteness needs *all* principal minors nonnegative: $\begin{pmatrix} 0 & 0 \\ 0 & -1\end{pmatrix}$ has leading minors 0 and 0 yet is not PSD. A block matrix $\begin{pmatrix} A & B \\ B^\top & C\end{pmatrix}$ with $C \succ 0$ is PSD exactly when the Schur complement $A - BC^{-1}B^\top$ is PSD, the same object as the Gaussian conditional covariance.

### The SVD and the best low-rank approximation

Every $A \in \mathbb{R}^{m \times n}$ factors as $A = U\Sigma V^\top$ with $U$ and $V$ orthogonal and $\Sigma$ diagonal with $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$; equivalently $A = \sum_i \sigma_i u_i v_i^\top$. Geometrically, $A$ rotates (by $V^\top$), scales each axis by $\sigma_i$, and rotates again (by $U$). The link to eigendecompositions is $A^\top A = V\Sigma^2V^\top$ and $AA^\top = U\Sigma^2U^\top$: the right singular vectors are eigenvectors of $A^\top A$, and $\sigma_i^2$ are its eigenvalues. The rank is the number of nonzero $\sigma_i$, and $|\det A| = \prod_i \sigma_i$ for square $A$.

*Example by hand.* $M = \begin{pmatrix} 3 & 0 \\ 4 & 5\end{pmatrix}$. Then $M^\top M = \begin{pmatrix} 25 & 20 \\ 20 & 25\end{pmatrix}$, with eigenvalues $45$ and $5$ and eigenvectors $(1, 1)/\sqrt2$ and $(1, -1)/\sqrt2$. So $\sigma_1 = 3\sqrt5 = 6.708$ and $\sigma_2 = \sqrt5 = 2.236$, with $\sigma_1\sigma_2 = 15 = \det M$. The first left singular vector is $u_1 = Mv_1/\sigma_1 = (1, 3)/\sqrt{10}$.

**Eckart-Young theorem.** For $k < \mathrm{rank}(A)$, the truncated SVD $A_k = \sum_{i \le k}\sigma_i u_i v_i^\top$ is a best rank-$k$ approximation in both the spectral and the Frobenius norm, with errors

$$
\lVert A - A_k\rVert_2 = \sigma_{k+1}, \qquad \lVert A - A_k\rVert_F = \Big(\sum_{i > k}\sigma_i^2\Big)^{1/2}.
$$

*Sketch for the spectral norm.* Let $B$ have rank at most $k$. Its null space has dimension at least $n - k$, and $\mathrm{span}(v_1, \dots, v_{k+1})$ has dimension $k+1$, so the two subspaces share a unit vector $w$. Then $\lVert(A - B)w\rVert = \lVert Aw\rVert \ge \sigma_{k+1}$, because $w$ is a combination of the first $k+1$ right singular vectors, each stretched by at least $\sigma_{k+1}$. So every rank-$k$ $B$ has error at least $\sigma_{k+1}$, and $A_k$ achieves it since $A - A_k = \sum_{i > k}\sigma_i u_i v_i^\top$. (The Frobenius case uses Weyl's inequality for singular values; Mirsky extended the result to every unitarily invariant norm.)

*Example.* The best rank-1 approximation of $M$ is $\sigma_1 u_1 v_1^\top = \frac{3\sqrt5}{\sqrt{20}}\begin{pmatrix} 1 \\ 3\end{pmatrix}(1\ \ 1) = \begin{pmatrix} 1.5 & 1.5 \\ 4.5 & 4.5\end{pmatrix}$, with error $\sigma_2 = \sqrt5 = 2.236$ in both norms. Uses: PCA (below), compression, low-rank adaptation (LoRA constrains a weight update to rank $r$; see [LLMs](llms.html)), and denoising by truncating small singular values.

**PCA through the SVD.** For centred data $X \in \mathbb{R}^{n \times d}$ (rows are samples) with covariance $\Sigma = X^\top X/(n-1)$, the variance of the projection on a unit vector $u$ is $u^\top\Sigma u$. Maximizing it subject to $u^\top u = 1$ (a Rayleigh quotient) gives the top eigenvector, and the captured variance is the top eigenvalue. If $X = USV^\top$, then $\Sigma = V\frac{S^2}{n-1}V^\top$: the principal directions are the right singular vectors, $\lambda_i = s_i^2/(n-1)$, and the scores are $XV = US$. The practice question at the end of this section gives the full treatment.

### Matrix norms and the condition number

The **Frobenius norm** $\lVert A\rVert_F = (\sum_{ij}a_{ij}^2)^{1/2} = (\mathrm{tr}\,A^\top A)^{1/2} = (\sum_i\sigma_i^2)^{1/2}$ treats the matrix as a vector. The **spectral norm** $\lVert A\rVert_2 = \max_{x \ne 0}\lVert Ax\rVert/\lVert x\rVert = \sigma_1$ is the largest stretch. The **nuclear norm** $\sum_i\sigma_i$ is the convex surrogate for rank used in matrix completion. They satisfy $\lVert A\rVert_2 \le \lVert A\rVert_F \le \sqrt{r}\,\lVert A\rVert_2$ for rank $r$, and operator norms are submultiplicative, $\lVert AB\rVert \le \lVert A\rVert\,\lVert B\rVert$. In ML, the Lipschitz constant of a network with 1-Lipschitz activations is at most the product of its layers' spectral norms, which is what spectral normalization controls by dividing each weight by an estimate of $\sigma_1$ (Miyato et al., 2018).

The **condition number** of an invertible matrix is $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ (for symmetric positive definite $A$, $\lambda_{\max}/\lambda_{\min}$). It bounds how much relative error in the data can be amplified in the solution of $Ax = b$. If $b$ is perturbed by $\delta b$, then $\delta x = A^{-1}\delta b$, so $\lVert\delta x\rVert \le \lVert\delta b\rVert/\sigma_{\min}$, while $\lVert b\rVert = \lVert Ax\rVert \le \sigma_{\max}\lVert x\rVert$; multiplying the two,

$$
\frac{\lVert\delta x\rVert}{\lVert x\rVert} \le \kappa(A)\,\frac{\lVert\delta b\rVert}{\lVert b\rVert}.
$$

A rule of thumb is that you lose about $\log_{10}\kappa$ significant digits. The $10 \times 10$ Hilbert matrix $H_{ij} = 1/(i + j - 1)$ has $\kappa \approx 1.6 \times 10^{13}$, leaving about 3 of double precision's 16 digits. Forming $A^\top A$ squares the condition number: $\kappa(M) = 3$ but $\kappa(M^\top M) = 45/5 = 9$. **Why it matters for optimization:** for a quadratic $f(x) = \tfrac12 x^\top Ax - b^\top x$, $\kappa(A) = L/\mu$ is exactly the ratio that sets how many gradient-descent steps are needed (Section 7). Preconditioning, feature standardization and normalization layers are all ways to reduce the effective condition number.

### Rank, null space and the pseudo-inverse

The **column space** of $A \in \mathbb{R}^{m \times n}$ is $\{Ax\}$, of dimension $\mathrm{rank}(A)$; the **null space** is $\{x : Ax = 0\}$; and rank-nullity says $\mathrm{rank}(A) + \dim\mathrm{null}(A) = n$. Vectors are linearly independent exactly when the matrix with them as columns has full column rank, which in floating point you test by the ratio of the smallest to the largest singular value (or a rank-revealing pivoted QR), not by an exact zero test.

**Least squares.** Minimizing $\lVert Ax - b\rVert^2$ gives the normal equations $A^\top Ax = A^\top b$ (the gradient $2A^\top(Ax - b)$ from the table below set to zero). If $A$ has full column rank the solution is unique, $x = (A^\top A)^{-1}A^\top b$. Otherwise there are infinitely many solutions, differing by null-space vectors.

The **Moore-Penrose pseudo-inverse** is $A^+ = V\Sigma^+U^\top$, where $\Sigma^+$ inverts the nonzero singular values and leaves the zeros. The vector $x^+ = A^+b$ is the **minimum-norm least-squares solution**: among all minimizers of $\lVert Ax - b\rVert$ it has the smallest $\lVert x\rVert$. With full column rank, $A^+ = (A^\top A)^{-1}A^\top$; in general $A^+ = \lim_{\lambda \to 0^+}(A^\top A + \lambda I)^{-1}A^\top$, the ridge solution with vanishing regularization.

*Examples.* For one equation $x_1 + x_2 = 2$, $A = (1\ \ 1)$, $A^+ = (\tfrac12, \tfrac12)^\top$, and $x^+ = (1, 1)$: of all points on the line, the one closest to the origin. For the rank-1 matrix $B = \begin{pmatrix} 1 & 2 \\ 2 & 4\end{pmatrix} = 5\,uu^\top$ with $u = (1, 2)/\sqrt5$, $B^+ = \frac15 uu^\top = B/25$. A useful ML fact: gradient descent on an underdetermined least-squares problem started from $x = 0$ stays in the row space of $A$ (every gradient $A^\top(Ax - b)$ lies there) and so converges to $x^+$, an example of the implicit bias of an optimizer.

### Solving linear systems in practice

Inverting an $n \times n$ matrix by Gaussian elimination or LU costs $O(n^3)$, and you rarely need the inverse. Solve $Ax = b$ with an LU factorization (with pivoting); use Cholesky when $A$ is symmetric positive definite, at about half the cost and numerically stable; use QR or the SVD for least squares, which avoids squaring the condition number as the normal equations do; and use conjugate gradients for large sparse SPD systems, which needs only matrix-vector products and needs a number of iterations proportional to $\sqrt\kappa$ per digit of accuracy. The **Woodbury identity**

$$
(A + UCV)^{-1} = A^{-1} - A^{-1}U\big(C^{-1} + VA^{-1}U\big)^{-1}VA^{-1}
$$

turns a low-rank update of a matrix with a known inverse into a small $k \times k$ inverse; it appears in Gaussian processes, Kalman filters and second-order optimizers.

### Trace and determinant facts

**Trace.**

- $\mathrm{tr}(AB) = \mathrm{tr}(BA)$, and more generally the trace is invariant under *cyclic* permutations, $\mathrm{tr}(ABC) = \mathrm{tr}(CAB) = \mathrm{tr}(BCA)$, but not arbitrary ones.
- $x^\top Ax = \mathrm{tr}(Axx^\top)$, a scalar is its own trace.
- For a random $x$ with mean $m$ and covariance $\Sigma$: $\mathbb{E}[x^\top Ax] = \mathrm{tr}(A\,\mathbb{E}[xx^\top]) = \mathrm{tr}(A\Sigma) + m^\top Am$. This gave the Gaussian KL in Section 2.
- $\mathrm{tr}\,A = \sum_i\lambda_i$, and $\lVert A\rVert_F^2 = \mathrm{tr}(A^\top A)$.
- **Hutchinson's estimator**: if $\mathbb{E}[\epsilon\epsilon^\top] = I$ (Rademacher or Gaussian entries), then $\mathbb{E}[\epsilon^\top A\epsilon] = \mathrm{tr}(A)$. It needs only products $A\epsilon$, so it estimates the trace of a Jacobian or Hessian you never form; FFJORD uses it for the divergence term of continuous normalizing flows.

**Determinant.**

- $\det(AB) = \det A\det B$, $\det A^\top = \det A$, $\det(A^{-1}) = 1/\det A$, $\det(cA) = c^n\det A$ for $n \times n$ $A$.
- $\det A = \prod_i\lambda_i$, and $|\det A|$ is the factor by which $A$ scales volume.
- A triangular matrix has determinant equal to the product of its diagonal, the reason coupling layers have cheap log-determinants.
- **Matrix determinant lemma**: $\det(A + uv^\top) = (1 + v^\top A^{-1}u)\det A$, used for the log-determinant of planar flows (Rezende and Mohamed, 2015).
- Compute $\log\det$ of an SPD matrix as $2\sum_i\log L_{ii}$ from its Cholesky factor; the determinant itself overflows or underflows easily.
- $\log\det$ is concave on positive definite matrices.

### Matrix calculus

With the convention that the gradient of a scalar with respect to $x$ (or $X$) has the same shape as $x$ (or $X$):

| Function | Gradient | Note |
|---|---|---|
| $a^\top x$ | $a$ | linear |
| $x^\top Ax$ | $(A + A^\top)x$ | $2Ax$ if $A$ is symmetric |
| $\lVert Ax - b\rVert^2$ | $2A^\top(Ax - b)$ | least squares; zero gives the normal equations |
| $\lVert x\rVert_2$ | $x/\lVert x\rVert_2$ | for $x \ne 0$ |
| $\mathrm{tr}(AX)$ | $A^\top$ | linear in $X$ |
| $\mathrm{tr}(X^\top AX)$ | $(A + A^\top)X$ | |
| $\lVert X\rVert_F^2$ | $2X$ | weight decay |
| $\log\det X$ | $X^{-\top}$ | $X^{-1}$ if $X$ is symmetric positive definite |
| $\log\sum_j e^{z_j}$ | $\mathrm{softmax}(z)$ | log-sum-exp |
| $-\sum_j y_j\log\mathrm{softmax}(z)_j$ | $\mathrm{softmax}(z) - y$ | cross-entropy in the logits, for one-hot $y$ |

*Derivations of the less obvious rows.*

- $x^\top Ax$: perturb, $(x + h)^\top A(x + h) - x^\top Ax = h^\top Ax + x^\top Ah + O(\lVert h\rVert^2) = h^\top(A + A^\top)x + O(\lVert h\rVert^2)$.
- $\mathrm{tr}(AX) = \sum_{k,l}A_{kl}X_{lk}$, so $\partial/\partial X_{ij} = A_{ji}$, which is $A^\top$.
- $\log\det X$: $\det(X + H) = \det X\,\det(I + X^{-1}H) = \det X\,\big(1 + \mathrm{tr}(X^{-1}H) + O(\lVert H\rVert^2)\big)$, so $d\log\det X = \mathrm{tr}(X^{-1}H) = \langle X^{-\top}, H\rangle$.

**Application: the Gaussian MLE for a covariance.** With precision $\Lambda = \Sigma^{-1}$ and scatter matrix $S = \sum_i(x_i - \mu)(x_i - \mu)^\top$, the log-likelihood is $\frac n2\log\det\Lambda - \frac12\sum_i(x_i - \mu)^\top\Lambda(x_i - \mu) + \text{const} = \frac n2\log\det\Lambda - \frac12\mathrm{tr}(\Lambda S) + \text{const}$, using the trace trick. Its gradient in $\Lambda$ is $\frac n2\Lambda^{-1} - \frac12 S$ (from the table, with $S$ symmetric), so $\hat\Sigma = \Lambda^{-1} = S/n$: the multivariate version of the biased variance MLE.

### Power iteration

To find the dominant eigenvector of $A$, repeatedly multiply and normalize: $v_{k+1} = Av_k/\lVert Av_k\rVert$. Expand the start vector in the eigenbasis, $v_0 = \sum_i c_iq_i$. Then $A^kv_0 = \lambda_1^k\big(c_1q_1 + \sum_{i \ge 2}c_i(\lambda_i/\lambda_1)^kq_i\big)$, so if $|\lambda_1| > |\lambda_2|$ and $c_1 \ne 0$ (true with probability 1 for a random start), the direction converges to $q_1$ with error shrinking like $|\lambda_2/\lambda_1|^k$. For symmetric $A$, the Rayleigh quotient $v_k^\top Av_k$ converges twice as fast in the exponent, like $|\lambda_2/\lambda_1|^{2k}$.

```python
import numpy as np

def power_iteration(A, n_iter=100, tol=1e-10, seed=0):
    v = np.random.default_rng(seed).standard_normal(A.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(n_iter):
        w = A @ v                            # one matrix-vector product per step
        v_new = w / np.linalg.norm(w)
        lam_new = v_new @ A @ v_new          # Rayleigh quotient estimate of the eigenvalue
        if abs(lam_new - lam) < tol:
            break
        v, lam = v_new, lam_new
    return lam_new, v_new

A = np.array([[2.0, 1.0], [1.0, 3.0]])
lam, v = power_iteration(A)
print(lam, v, np.linalg.eigvalsh(A))         # 3.618..., eigenvalues 1.382 and 3.618
```

It prints $3.6180339887$ and the eigenvector $\pm(0.526, 0.851)$. Here $\lambda_2/\lambda_1 = 1.382/3.618 = 0.382$: starting from $(1, 0)$, the vector error falls by about 0.38 per step and the eigenvalue error by about $0.38^2 = 0.146$ per step, reaching $1.2 \times 10^{-6}$ after 8 steps. Convergence is slow when $\lambda_2 \approx \lambda_1$; Lanczos or randomized SVD are faster, and orthogonal (subspace) iteration finds the top $k$. In ML, spectral normalization runs one power-iteration step per training step and carries the vector over; the top Hessian eigenvalue, which sets the stable step size, is estimated by power iteration on Hessian-vector products; and PageRank is power iteration on a stochastic matrix.

### Practice questions

::: {.qa asked="Google DeepMind research intern, Glassdoor"}
**What does it cost to invert a matrix, how should you solve a linear system in practice, and how do you test linear independence?**

Inversion by Gaussian elimination or LU costs $O(n^3)$. You rarely need the inverse: solve $Ax = b$ with an LU factorization; with Cholesky when $A$ is symmetric positive definite (about half the cost, and stable); with QR for least squares, which avoids squaring the condition number as the normal equations do; or with conjugate gradients for large sparse SPD systems. The Woodbury identity handles low-rank updates. Vectors are linearly independent when the matrix they form has full column rank; numerically, check the smallest singular value relative to the largest (via the SVD or pivoted QR) rather than testing for an exact zero, because the relative error in a solve is amplified by up to $\kappa = \sigma_{\max}/\sigma_{\min}$.
:::

::: {.qa asked="NetEase Games image algorithm, Nowcoder 2021"}
**What do the eigenvectors of a covariance matrix mean, and how does the SVD relate to PCA?**

The eigenvectors are orthogonal directions along which the projected coordinates are uncorrelated, ordered by the variance they carry; the eigenvalues are those variances. PCA keeps the top ones, and the SVD of the centred data matrix computes the same thing more stably.

*Derivation.* For centred $X \in \mathbb{R}^{n \times d}$ and $\Sigma = X^\top X/(n-1)$, the variance of the projection on unit $u$ is $u^\top\Sigma u$. The Lagrangian $u^\top\Sigma u - \lambda(u^\top u - 1)$ gives $\Sigma u = \lambda u$, where the objective equals $\lambda$, so the best direction is the top eigenvector; requiring orthogonality to earlier components gives the next ones in order. $\Sigma$ is symmetric PSD, so its eigenvectors are orthogonal and its eigenvalues nonnegative; $\sum_i\lambda_i = \mathrm{tr}\,\Sigma$ is the total variance, and $\lambda_i/\sum_j\lambda_j$ is the explained-variance ratio. In the eigenbasis the covariance is diagonal; for Gaussian data the eigenvectors are the axes of the density's ellipsoids, with semi-axes proportional to $\sqrt{\lambda_i}$.

*SVD.* $X = USV^\top$ gives $\Sigma = V\frac{S^2}{n-1}V^\top$: right singular vectors are the principal directions, $\lambda_i = s_i^2/(n-1)$, and the scores are $XV = US$. Forming $X^\top X$ squares the condition number, so small-variance directions lose about twice as many digits; the SVD works on $X$ directly. For the top $k$ components of large data use a truncated or randomized SVD; when $d \gg n$, the $n \times n$ Gram matrix $XX^\top$ has the same nonzero eigenvalues and is cheaper.

*Whitening.* $z = \Lambda^{-1/2}V^\top x$ has identity covariance (PCA whitening); $V\Lambda^{-1/2}V^\top x$ (ZCA) also whitens but stays closest to the original coordinates, so whitened images still look like images. Add a small $\epsilon$ to each $\lambda_i$ so near-zero, mostly noise, directions are not blown up.

*Pitfalls.* Forgetting to centre; mixing units, since PCA is not scale-invariant (standardize first); reading meaning into eigenvector signs, which are arbitrary; and expecting PCA to find nonlinear structure, since it uses only second moments.
:::

::: qa
**Show that a covariance matrix is positive semidefinite. When is it singular?**

For any $v$, $v^\top\Sigma v = v^\top\mathbb{E}[(x - \mu)(x - \mu)^\top]v = \mathbb{E}[(v^\top(x - \mu))^2] = \mathrm{Var}(v^\top x) \ge 0$. It is singular exactly when some nonzero $v$ has $\mathrm{Var}(v^\top x) = 0$, that is, when the data lie in an affine subspace (a linear constraint holds almost surely). A sample covariance from $n \le d$ points is always singular, since $n$ centred points span at most $n - 1$ dimensions; add a ridge $\epsilon I$ or use shrinkage before inverting it.
:::

## 7. Optimization

In this section $\mu$ and $L$ are the strong-convexity and smoothness constants, $x^*$ a minimizer, $f^* = f(x^*)$, and $g_k = \nabla f(x_k)$.

### Convexity

A set $C$ is convex if it contains the segment between any two of its points. A function $f$ on a convex set is **convex** if

$$
f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y) \qquad \text{for all } x, y \text{ and } \lambda \in [0, 1],
$$

that is, chords lie above the graph. Equivalently, $t \mapsto f(x + tv)$ is convex for every $x$ and direction $v$, which reduces any check to one dimension.

**First-order condition.** A differentiable $f$ is convex if and only if

$$
f(y) \ge f(x) + \nabla f(x)^\top(y - x) \qquad \text{for all } x, y :
$$

tangent planes lie below the graph. *Proof of "only if".* Convexity gives $f(x + \lambda(y - x)) \le f(x) + \lambda(f(y) - f(x))$; subtract $f(x)$, divide by $\lambda$, and let $\lambda \to 0$ to get $\nabla f(x)^\top(y - x) \le f(y) - f(x)$. *"If".* Apply the inequality at $z = \lambda x + (1-\lambda)y$ toward $x$ and toward $y$, multiply by $\lambda$ and $1 - \lambda$, and add; the gradient terms cancel and leave $f(z) \le \lambda f(x) + (1-\lambda)f(y)$. The key consequence: if $\nabla f(x^*) = 0$, then $f(y) \ge f(x^*)$ for all $y$, so **every stationary point of a convex function is a global minimum**.

**Second-order condition.** A twice-differentiable $f$ is convex if and only if $\nabla^2 f(x) \succeq 0$ everywhere. In one dimension, $g(t) = f(x + tv)$ has $g''(t) = v^\top\nabla^2 f\,v$, and $g'' \ge 0$ means $g'$ is nondecreasing, which is convexity of $g$. You check it with eigenvalues or a Cholesky factorization; numerically, one point with a negative Hessian eigenvalue disproves convexity, while sampling points can never prove it.

**Strong convexity and smoothness.** $f$ is $\mu$-**strongly convex** if $f(y) \ge f(x) + \nabla f(x)^\top(y - x) + \frac\mu2\lVert y - x\rVert^2$ (for $C^2$ functions, $\nabla^2 f \succeq \mu I$): it curves up at least like a quadratic. $f$ is $L$-**smooth** if $\nabla f$ is $L$-Lipschitz (for $C^2$ functions, $-LI \preceq \nabla^2 f \preceq LI$): it curves at most like a quadratic. For $f(x) = \tfrac12 x^\top Ax$ with $A \succ 0$, $\mu = \lambda_{\min}(A)$, $L = \lambda_{\max}(A)$, and $\kappa = L/\mu$ is the condition number.

**Building convex functions.** Nonnegative weighted sums, composition with an affine map, pointwise maxima, and $h(g(x))$ with $h$ convex and nondecreasing and $g$ convex all preserve convexity. That covers norms, $\lVert Ax - b\rVert^2$, the hinge and logistic losses, log-sum-exp, softmax cross-entropy as a function of the logits, and $-\log\det X$ on positive definite matrices. A neural network's loss is not convex in its weights: permuting the hidden units of a layer gives a different weight vector with the same loss, so if $\theta^*$ is a minimizer so is its permutation $\pi(\theta^*)$, and convexity would force every point on the segment between them to be a minimizer too, which generally fails.

**Worked example: the centroid.** The point minimizing the sum of squared distances to $p_1, \dots, p_n$ is their mean. $f(x) = \sum_i\lVert x - p_i\rVert^2$ has $\nabla f = 2\sum_i(x - p_i) = 0$ at $x = \bar p$, and $\nabla^2 f = 2nI \succ 0$, so it is the unique global minimum. Equivalently, expanding around the mean, $f(x) = n\lVert x - \bar p\rVert^2 + \sum_i\lVert p_i - \bar p\rVert^2$, because the cross term $2(x - \bar p)^\top\sum_i(\bar p - p_i)$ vanishes. This is why the k-means update uses the mean; with unsquared distances the minimizer is the geometric median, which has no closed form.

### Gradient descent: convergence rates and the proof

Gradient descent is $x_{k+1} = x_k - \eta\,g_k$. The three standard results, all with step $\eta = 1/L$:

| Assumption on $f$ | Guarantee after $K$ steps | Steps for accuracy $\epsilon$ |
|---|---|---|
| $L$-smooth, bounded below | $\min_{k < K}\lVert g_k\rVert^2 \le 2L(f(x_0) - f^*)/K$ | $O(1/\epsilon^2)$ for $\lVert g\rVert \le \epsilon$ |
| $L$-smooth and convex | $f(x_K) - f^* \le L\lVert x_0 - x^*\rVert^2/(2K)$ | $O(1/\epsilon)$ |
| $L$-smooth and $\mu$-strongly convex | $f(x_K) - f^* \le (1 - \mu/L)^K(f(x_0) - f^*)$ | $O(\kappa\log(1/\epsilon))$ |

**Step 1: the descent lemma.** For $L$-smooth $f$, by the fundamental theorem of calculus along the segment and then Cauchy-Schwarz with the Lipschitz bound,

$$
f(y) - f(x) - \nabla f(x)^\top(y - x) = \int_0^1\big(\nabla f(x + t(y - x)) - \nabla f(x)\big)^\top(y - x)\,dt \le \int_0^1 Lt\,\lVert y - x\rVert^2\,dt = \frac L2\lVert y - x\rVert^2 .
$$

So $f$ lies below a quadratic upper bound around every point. Put $y = x - \eta\nabla f(x)$:

$$
f(x - \eta\nabla f(x)) \le f(x) - \eta\Big(1 - \frac{L\eta}{2}\Big)\lVert\nabla f(x)\rVert^2 .
$$

Any $\eta < 2/L$ decreases $f$, and $\eta = 1/L$ maximizes the guaranteed decrease:

$$
f(x_{k+1}) \le f(x_k) - \frac{1}{2L}\lVert g_k\rVert^2 . \qquad (\star)
$$

**Nonconvex case.** Sum $(\star)$ over $k = 0, \dots, K-1$; the left side telescopes: $\frac{1}{2L}\sum_k\lVert g_k\rVert^2 \le f(x_0) - f(x_K) \le f(x_0) - f^*$. The minimum is at most the average, which is the first row of the table. The point found is stationary, which may be a saddle.

**Convex case, in full.**

1. *Convexity bounds the gap by the gradient.* The first-order condition at $x_k$ toward $x^*$ gives $f^* \ge f(x_k) + g_k^\top(x^* - x_k)$, that is, $f(x_k) - f^* \le g_k^\top(x_k - x^*)$.
2. *Combine with $(\star)$.* $f(x_{k+1}) - f^* \le g_k^\top(x_k - x^*) - \frac{1}{2L}\lVert g_k\rVert^2$.
3. *Recognize a difference of distances.* Since $x_{k+1} = x_k - g_k/L$,
$$
\lVert x_{k+1} - x^*\rVert^2 = \lVert x_k - x^*\rVert^2 - \frac2L g_k^\top(x_k - x^*) + \frac{1}{L^2}\lVert g_k\rVert^2 ,
$$
so the right side of step 2 equals $\frac L2\big(\lVert x_k - x^*\rVert^2 - \lVert x_{k+1} - x^*\rVert^2\big)$. Therefore
$$
f(x_{k+1}) - f^* \le \frac L2\Big(\lVert x_k - x^*\rVert^2 - \lVert x_{k+1} - x^*\rVert^2\Big).
$$
As a by-product, the distance to $x^*$ never increases.
4. *Telescope.* Summing over $k = 0, \dots, K-1$:
$$
\sum_{k=1}^{K}\big(f(x_k) - f^*\big) \le \frac L2\Big(\lVert x_0 - x^*\rVert^2 - \lVert x_K - x^*\rVert^2\Big) \le \frac L2\lVert x_0 - x^*\rVert^2 .
$$
5. *Use monotonicity.* By $(\star)$, $f(x_k)$ is nonincreasing, so the last term is the smallest: $K\big(f(x_K) - f^*\big) \le \sum_{k=1}^K\big(f(x_k) - f^*\big)$. Dividing by $K$ gives $f(x_K) - f^* \le L\lVert x_0 - x^*\rVert^2/(2K)$. $\blacksquare$

**Strongly convex case.** Minimize both sides of the strong-convexity inequality over $y$: the right side is minimized at $y = x - \nabla f(x)/\mu$, giving $f^* \ge f(x) - \lVert\nabla f(x)\rVert^2/(2\mu)$, that is,

$$
\lVert\nabla f(x)\rVert^2 \ge 2\mu\big(f(x) - f^*\big),
$$

the **Polyak-Łojasiewicz (PL) inequality**. Substituting into $(\star)$: $f(x_{k+1}) - f^* \le f(x_k) - f^* - \frac{\mu}{L}(f(x_k) - f^*) = (1 - \mu/L)(f(x_k) - f^*)$. That is linear convergence, and since $(1 - 1/\kappa)^K \le e^{-K/\kappa}$, reaching accuracy $\epsilon$ takes $O(\kappa\log(1/\epsilon))$ steps. The proof used only the PL inequality, so it also covers nonconvex functions that satisfy it, such as some overparameterized least-squares problems.

**For context.** Nesterov's accelerated gradient improves the rates to $O(1/K^2)$ and to a factor $1 - 1/\sqrt\kappa$ per step, and these match lower bounds for first-order methods (Nesterov's lectures). SGD on convex problems is slower, $O(1/\sqrt K)$, because its gradient noise does not vanish at the optimum (below).

::: key
**How to say it in an interview.**

"Smoothness gives the descent lemma: with step $1/L$ each step decreases $f$ by at least $\lVert g\rVert^2/2L$. Without convexity, telescoping gives a small gradient at rate $1/K$ in squared norm. With convexity, the decrease telescopes against the distance to the optimum and gives $L\lVert x_0 - x^*\rVert^2/2K$. With strong convexity, the gradient is at least proportional to the suboptimality, so each step removes a fixed fraction $\mu/L$ of the gap: linear convergence in $\kappa\log(1/\epsilon)$ steps."
:::

### The condition-number effect

Take $f(x) = \tfrac12(x_1^2 + \kappa x_2^2)$, so $\mu = 1$ and $L = \kappa$. Gradient descent acts on each coordinate separately: $x_1 \leftarrow (1 - \eta)x_1$ and $x_2 \leftarrow (1 - \eta\kappa)x_2$. Stability in the steep direction requires $|1 - \eta\kappa| < 1$, that is $\eta < 2/\kappa$, and then the shallow direction shrinks by at least $1 - 2/\kappa$ per step: progress along the flat valley is throttled by the curvature across it. In pictures, the iterate zig-zags across a long narrow valley while creeping along its floor.

```python
import numpy as np

def iters(kappa, step, beta=0.0, tol=1e-6):
    h = np.array([1.0, kappa])            # f(x) = (x1^2 + kappa * x2^2) / 2, so mu = 1, L = kappa
    x = x_prev = np.array([1.0, 1.0])
    k = 0
    while np.linalg.norm(x) > tol * np.sqrt(2):
        x, x_prev = x - step * h * x + beta * (x - x_prev), x   # beta > 0 adds heavy-ball momentum
        k += 1
    return k

for kappa in [1, 10, 100, 1000]:
    r = np.sqrt(kappa)
    print(kappa,
          iters(kappa, 1 / kappa),                                 # step 1/L
          iters(kappa, 2 / (1 + kappa)),                           # best fixed step 2/(mu+L)
          iters(kappa, 4 / (1 + r)**2, ((r - 1) / (r + 1))**2))    # heavy ball, tuned
```

Steps to shrink the distance to the optimum by $10^6$:

| $\kappa$ | GD, $\eta = 1/L$ | GD, $\eta = 2/(\mu + L)$ | Heavy ball, tuned |
|---|---|---|---|
| 1 | 1 | 1 | 1 |
| 10 | 128 | 69 | 27 |
| 100 | 1,341 | 691 | 93 |
| 1000 | 13,463 | 6,908 | 315 |

With $\eta = 1/L$ the shallow coordinate contracts by $1 - 1/\kappa$, so the count is about $\kappa\ln 10^6 \approx 13.8\kappa$. The best fixed step $2/(\mu + L)$ balances the two directions to the factor $(\kappa - 1)/(\kappa + 1) \approx 1 - 2/\kappa$ and halves the count. Polyak's heavy-ball momentum, tuned for the quadratic, contracts by $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ and scales with $\sqrt\kappa$ instead of $\kappa$. Changing coordinates to make $\kappa = 1$ (preconditioning) would converge in one step, which is what Newton's method does.

### Newton and quasi-Newton methods

**Newton's method** minimizes the local quadratic model $f(x) + g^\top p + \tfrac12 p^\top Hp$; its gradient $g + Hp$ vanishes at $p = -H^{-1}g$, so $x_{k+1} = x_k - H_k^{-1}g_k$. On a quadratic with $H \succ 0$ this lands on the minimizer in one step, whatever $\kappa$ is, because Newton's method is invariant to affine changes of coordinates. **Local quadratic convergence:** with $e_k = x_k - x^*$ and $\nabla f(x^*) = 0$, $g_k = \int_0^1 H(x^* + te_k)\,e_k\,dt$. If $H_k \succeq \mu I$ and the Hessian is $M$-Lipschitz,

$$
\lVert e_{k+1}\rVert = \Big\lVert H_k^{-1}\int_0^1\big(H_k - H(x^* + te_k)\big)e_k\,dt\Big\rVert \le \frac1\mu\int_0^1 M(1-t)\lVert e_k\rVert^2\,dt = \frac{M}{2\mu}\lVert e_k\rVert^2 ,
$$

so the number of correct digits roughly doubles per step near the solution. The costs are $O(d^2)$ memory and an $O(d^3)$ solve per step. Far from the solution it can fail: with an indefinite $H$ the step may go uphill or toward a saddle, and it can diverge even on a convex function. For $f(x) = \sqrt{1 + x^2}$, $f' = x/\sqrt{1 + x^2}$ and $f'' = (1 + x^2)^{-3/2}$, so the Newton step is $x \mapsto x - x(1 + x^2) = -x^3$, which diverges for $|x| > 1$. Fixes: a line search (damped Newton), trust regions, or Levenberg-Marquardt damping $(H + \lambda I)^{-1}$.

**Quasi-Newton methods** build curvature from gradients alone. With $s_k = x_{k+1} - x_k$ and $y_k = g_{k+1} - g_k$, they impose the **secant condition** $B_{k+1}s_k = y_k$, which a true Hessian satisfies approximately. **BFGS** updates an inverse-Hessian approximation $C_k$ by a rank-two correction,

$$
C_{k+1} = (I - \rho_k s_k y_k^\top)\,C_k\,(I - \rho_k y_k s_k^\top) + \rho_k s_k s_k^\top, \qquad \rho_k = \frac{1}{y_k^\top s_k},
$$

which stays positive definite when $y_k^\top s_k > 0$ (a Wolfe line search guarantees this), costs $O(d^2)$ per step, and converges superlinearly. **L-BFGS** stores only the last $m$ pairs (typically 3 to 20) and applies $C_k$ to a gradient with a two-loop recursion in $O(md)$. Minibatch noise corrupts the $(s, y)$ pairs, so in deep learning L-BFGS is used mainly for full-batch problems (for example, style transfer or fitting small models), while curvature enters large-scale training through Hessian-vector products, diagonal or Kronecker-factored approximations, and Adam's per-coordinate scaling (see [deep learning training](deep-learning-training.html)).

**One dimension: bisection versus Newton.** Computing $\sqrt x$ is the classic small case. Newton on $y^2 - x = 0$ gives $y \leftarrow (y + x/y)/2$: from $y = 1$ and $x = 2$ the iterates are 1.5, 1.41667, 1.4142157, 1.41421356237469, and the fifth equals $\sqrt2$ to double precision. Bisection gains one bit per step but needs only comparisons and never leaves its bracket. The practice question below gives the careful version.

### Lagrange multipliers and the KKT conditions

**Equality constraints.** To minimize $f(x)$ subject to $h(x) = 0$, note that at a constrained minimum the gradient of $f$ can have no component along the constraint surface, or moving along the surface would decrease $f$. So $\nabla f$ is parallel to $\nabla h$: $\nabla f(x^*) + \nu\nabla h(x^*) = 0$ for some multiplier $\nu$, which is stationarity of the Lagrangian $f + \nu h$. Two ML examples:

- *PCA.* Maximizing $u^\top\Sigma u$ subject to $u^\top u = 1$ gives $\Sigma u = \lambda u$ (Section 6).
- *Maximum entropy gives softmax.* Maximize $-\sum_i p_i\log p_i$ subject to $\sum_i p_i = 1$ and $\sum_i p_iE_i = \bar E$. Setting the derivative of $-\sum p\log p + \alpha(\sum p - 1) - \beta(\sum pE - \bar E)$ to zero gives $-\log p_i - 1 + \alpha - \beta E_i = 0$, so $p_i \propto e^{-\beta E_i}$: the Boltzmann distribution, which is a softmax of the negative energies with inverse temperature $\beta$.

**Inequality constraints: KKT.** For

$$
\min_x f(x) \quad \text{s.t.} \quad g_i(x) \le 0,\ \ h_j(x) = 0,
$$

form the Lagrangian $\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_i\lambda_ig_i(x) + \sum_j\nu_jh_j(x)$. The **KKT conditions** are:

1. stationarity: $\nabla f(x) + \sum_i\lambda_i\nabla g_i(x) + \sum_j\nu_j\nabla h_j(x) = 0$;
2. primal feasibility: $g_i(x) \le 0$ and $h_j(x) = 0$;
3. dual feasibility: $\lambda_i \ge 0$;
4. complementary slackness: $\lambda_ig_i(x) = 0$, so a multiplier can be nonzero only on an active constraint.

For a convex problem that has a strictly feasible point (Slater's condition), the KKT conditions are necessary and sufficient for optimality. Without convexity they are necessary only under a constraint qualification, and then not sufficient.

**Worked example.** Minimize $f(x, y) = (x - 2)^2 + (y - 1)^2$ subject to $x + y \le 1$, the projection of the point $(2, 1)$ onto a half-plane.

- *Try the constraint inactive* ($\lambda = 0$): stationarity gives $(x, y) = (2, 1)$, but $2 + 1 = 3 > 1$ violates feasibility.
- *So it is active:* $x + y = 1$. Stationarity reads $2(x - 2) + \lambda = 0$ and $2(y - 1) + \lambda = 0$, so $x - 2 = y - 1$, i.e. $x = y + 1$. With $x + y = 1$: $y = 0$, $x = 1$, and $\lambda = 2(2 - x) = 2 \ge 0$, so dual feasibility holds.
- The optimum is $(1, 0)$ with $f^* = 1 + 1 = 2$; a numerical solver agrees.

The multiplier is a sensitivity: for the constraint $x + y \le b$, the squared distance from $(2, 1)$ to the half-plane is $(3 - b)^2/2$, whose derivative at $b = 1$ is $-2 = -\lambda$. Loosening the constraint by $\delta$ lowers the optimal cost by about $\lambda\delta$.

### Convex duality, briefly

The **dual function** $q(\lambda, \nu) = \inf_x\mathcal{L}(x, \lambda, \nu)$ is concave (an infimum of affine functions of $(\lambda, \nu)$), and for $\lambda \ge 0$ it is a lower bound on the optimal value $p^*$: at any feasible $x$, $\mathcal{L}(x, \lambda, \nu) \le f(x)$ because the added terms are $\le 0$. The **dual problem** maximizes $q$ over $\lambda \ge 0$; its value $d^*$ always satisfies $d^* \le p^*$ (weak duality), and for convex problems satisfying Slater's condition $d^* = p^*$ (strong duality), with the KKT conditions linking the two solutions.

*In the example,* $\mathcal{L} = (x - 2)^2 + (y - 1)^2 + \lambda(x + y - 1)$ is minimized at $x = 2 - \lambda/2$, $y = 1 - \lambda/2$, giving $q(\lambda) = \frac{\lambda^2}{4} + \frac{\lambda^2}{4} + \lambda(2 - \lambda) = 2\lambda - \frac{\lambda^2}{2}$. It is maximized at $\lambda = 2$ with $d^* = 2 = p^*$. Where duality shows up in ML: the SVM dual, which depends on the data only through inner products and so admits kernels, with support vectors as the points with $\lambda_i > 0$; and the Kantorovich-Rubinstein dual of the Wasserstein-1 distance, which is why a WGAN critic is constrained to be 1-Lipschitz.

### SGD convergence and step-size decay

SGD replaces $\nabla f$ by an unbiased estimate $\hat g$ from a minibatch. Assume $f$ is $L$-smooth and $\mu$-strongly convex, $\mathbb{E}[\hat g \mid x] = \nabla f(x)$, and $\mathbb{E}\lVert\hat g\rVert^2 \le \lVert\nabla f(x)\rVert^2 + \sigma^2$. The descent lemma in expectation, with $\eta \le 1/L$, gives

$$
\mathbb{E}f(x_{k+1}) \le f(x_k) - \eta\lVert\nabla f\rVert^2 + \frac{L\eta^2}{2}\big(\lVert\nabla f\rVert^2 + \sigma^2\big) \le f(x_k) - \frac\eta2\lVert\nabla f\rVert^2 + \frac{L\eta^2\sigma^2}{2},
$$

and the PL inequality turns this into

$$
\mathbb{E}\big[f(x_{k+1}) - f^*\big] \le (1 - \eta\mu)\,\mathbb{E}\big[f(x_k) - f^*\big] + \frac{L\eta^2\sigma^2}{2}.
$$

Unrolling, the gap shrinks linearly to a **noise floor** at the fixed point of this recursion, $\eta L\sigma^2/(2\mu)$. For $f(x) = x^2/2$ with unit gradient noise the exact stationary value of $\mathbb{E}f$ is $\eta/(2(2 - \eta))$, about $\eta/4$ and below the bound $\eta/2$; simulations with $\eta = 0.1$ and $0.01$ give 0.0264 and 0.0025. A constant step therefore converges quickly to a neighbourhood whose size is proportional to $\eta$, and only a decaying step reaches the optimum. The Robbins-Monro conditions

$$
\sum_k\eta_k = \infty, \qquad \sum_k\eta_k^2 < \infty
$$

(for example $\eta_k \propto 1/k$) ensure the steps travel arbitrarily far while the accumulated noise stays finite. The standard rates, stated: for convex problems with bounded stochastic gradients, steps $\propto 1/\sqrt k$ with iterate averaging give $\mathbb{E}f(\bar x_K) - f^* = O(1/\sqrt K)$; for strongly convex problems, steps $\propto 1/(\mu k)$ with suitable averaging give $O(1/K)$; for smooth nonconvex problems, $\min_k\mathbb{E}\lVert\nabla f(x_k)\rVert^2 = O(1/\sqrt K)$ with steps $\propto 1/\sqrt K$ (Ghadimi and Lan, 2013). This is the theory behind the observation that loss drops when the learning rate is decayed at the end of training (schedules are on [deep learning training](deep-learning-training.html)).

### Saddle points in high dimension

A saddle point has zero gradient and a Hessian with both positive and negative eigenvalues; $f(x, y) = x^2 - y^2$ at the origin is the model. Gradient descent with step $\eta$ multiplies $y$ by $1 + 2\eta$ per step, so from $y_0 = 10^{-8}$ with $\eta = 0.1$ it takes about $\ln(10^8)/\ln 1.2 \approx 101$ steps to escape: escape is exponential but can start slowly, which shows up as a loss plateau. Starting exactly on the stable manifold ($y_0 = 0$) converges to the saddle, but that set has measure zero: gradient descent with a small fixed step and random initialization converges to a strict saddle with probability zero (Lee et al., 2016).

**Why saddles, not bad local minima, are the usual worry.** A heuristic: if the Hessian at a critical point looked like a random symmetric matrix, all its eigenvalues would be positive with a probability that falls fast with dimension. For matrices with i.i.d. Gaussian entries, symmetrized, a simulation gives 0.50 for $d = 1$, 0.15 for $d = 2$, 0.025 for $d = 3$ and $1.5 \times 10^{-4}$ for $d = 5$. Dauphin et al. (2014), building on results for random Gaussian fields and experiments with neural networks, argued that in high dimension critical points with loss well above the minimum are overwhelmingly saddles, that local minima tend to have loss close to the global minimum, and that saddles and the plateaus around them are the main obstacle. Noise helps escape: noisy SGD escapes strict saddles in polynomial time (Ge et al., 2015), and perturbed gradient descent finds approximate second-order stationary points with only a polylogarithmic dependence on dimension (Jin et al., 2017), whereas plain gradient descent can take exponential time on functions constructed for the purpose, even with random initialization (Du et al., 2017). Newton's method is attracted to saddles, since it solves $\nabla f = 0$ regardless of curvature sign, which is one reason second-order methods for deep learning damp or modify the Hessian.

### Practice questions

::: {.qa asked="Alibaba, Nowcoder 2022"}
**Prove that gradient descent converges, and state the rate.**

Assume $f$ is $L$-smooth and take $\eta = 1/L$. The descent lemma (integrate the gradient along the segment and use the Lipschitz bound) gives $f(x_{k+1}) \le f(x_k) - \frac{1}{2L}\lVert g_k\rVert^2$. Then:

- *Nonconvex:* telescoping gives $\min_{k<K}\lVert g_k\rVert^2 \le 2L(f(x_0) - f^*)/K$, an $\epsilon$-stationary point (possibly a saddle) in $O(1/\epsilon^2)$ steps.
- *Convex:* combining $f(x_k) - f^* \le g_k^\top(x_k - x^*)$ with the descent step and the expansion of $\lVert x_{k+1} - x^*\rVert^2$ gives $f(x_{k+1}) - f^* \le \frac L2(\lVert x_k - x^*\rVert^2 - \lVert x_{k+1} - x^*\rVert^2)$; summing and using monotonicity, $f(x_K) - f^* \le L\lVert x_0 - x^*\rVert^2/(2K)$.
- *Strongly convex:* the PL inequality $\lVert g\rVert^2 \ge 2\mu(f - f^*)$ turns the descent step into $f(x_{k+1}) - f^* \le (1 - \mu/L)(f(x_k) - f^*)$, so $O(\kappa\log(1/\epsilon))$ steps; this holds for any function satisfying PL.

The full five-step proof of the convex case is in the section above. Nesterov acceleration improves the rates to $O(1/K^2)$ and $1 - 1/\sqrt\kappa$ per step; SGD on convex problems is slower, $O(1/\sqrt K)$, because gradient noise does not vanish at the optimum.
:::

::: {.qa asked="NetEase Games image algorithm, Nowcoder 2021"}
**How do you test whether a function is convex, and how do Newton and quasi-Newton methods use the Hessian?**

*Tests:* the definition, or equivalently convexity of every one-dimensional restriction $t \mapsto f(x + tv)$; the first-order condition $f(y) \ge f(x) + \nabla f(x)^\top(y - x)$, which makes any stationary point a global minimum; the second-order condition $\nabla^2 f \succeq 0$ everywhere, checked with eigenvalues or Cholesky (leading principal minors certify definiteness, but semidefiniteness needs all principal minors nonnegative); and composition rules (nonnegative sums, affine precomposition, pointwise maxima, convex nondecreasing of convex), which cover norms, log-sum-exp and softmax cross-entropy in the logits. One point with a negative Hessian eigenvalue disproves convexity; sampling cannot prove it.

*Newton* minimizes the local quadratic model, $x_{k+1} = x_k - H_k^{-1}g_k$: quadratic convergence near the minimizer ($\lVert e_{k+1}\rVert \le \frac{M}{2\mu}\lVert e_k\rVert^2$), insensitive to conditioning, but $O(d^2)$ memory and $O(d^3)$ per solve, and unreliable far away (uphill or saddle-seeking steps with indefinite $H$, divergence even on $\sqrt{1 + x^2}$), fixed by line search, trust regions or Levenberg-Marquardt damping. *Quasi-Newton* methods impose the secant condition $B_{k+1}s_k = y_k$; BFGS keeps a positive definite inverse approximation with a rank-two update in $O(d^2)$ and converges superlinearly, and L-BFGS stores $m$ pairs and costs $O(md)$. Minibatch noise corrupts the $(s, y)$ pairs, so deep learning uses L-BFGS mainly for full-batch problems and otherwise gets curvature from Hessian-vector products or structured approximations.
:::

::: {.qa asked="Alibaba Tongyi Lab intern, Nowcoder 2024"}
**Compute $\sqrt{x}$ by binary search instead of Newton's method. How do you set the bounds for very large or very small inputs?**

Bisect on $y$ for the root of $y^2 = x$, keeping $\mathit{lo} \le \sqrt x \le \mathit{hi}$. Bounds that always work are $[\min(x, 1), \max(x, 1)]$: for $x \ge 1$, $1 \le \sqrt x \le x$, and for $0 < x < 1$, $x < \sqrt x < 1$. The common bug is $\mathit{hi} = x$, which excludes the answer when $x < 1$.

*Stopping.* Use a relative tolerance, $\mathit{hi} - \mathit{lo} \le \mathrm{tol}\cdot\mathit{lo}$. An absolute tolerance is meaningless for tiny inputs (with tolerance $10^{-9}$, any value below $10^{-9}$ passes for $x = 10^{-20}$, whose root is $10^{-10}$) and unreachable for huge ones (adjacent doubles near $10^{150}$ are about $10^{134}$ apart). Each step halves the interval, so it takes about $\log_2\big((\max(x,1) - \min(x,1))/(\mathrm{tol}\sqrt x)\big)$ iterations: 40 for $x = 2$ and 539 for $x = 10^{\pm 300}$ at $\mathrm{tol} = 10^{-12}$. An iteration cap guards against a tolerance below machine precision.

*Shrinking the range.* To make the count independent of magnitude, write $x = m\cdot 2^e$ with $e$ even and $m \in [0.5, 2)$ (`math.frexp`, moving one factor of 2 into $m$ when $e$ is odd), bisect for $\sqrt m$ in $[0.7, 1.5]$, and return $\sqrt m\cdot 2^{e/2}$ (`math.ldexp`). About 40 iterations then reach $10^{-12}$ for any input.

*Overflow.* Compare `mid > x / mid` rather than `mid * mid > x`; in the integer version this matters in fixed-width languages.

```python
import math

def sqrt_bisect(x, rel_tol=1e-12, max_iter=2000):
    if x < 0: raise ValueError("negative input")
    if x == 0 or math.isinf(x): return x
    lo, hi = min(x, 1.0), max(x, 1.0)      # sqrt(x) lies between x and 1
    for _ in range(max_iter):
        mid = lo + (hi - lo) / 2
        if mid > x / mid: hi = mid         # mid*mid > x, without overflow
        else: lo = mid
        if hi - lo <= rel_tol * lo: break  # relative tolerance
    return lo + (hi - lo) / 2

def isqrt(n):                              # largest r with r*r <= n
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2           # upper middle, so lo = mid makes progress
        if mid <= n // mid: lo = mid       # mid*mid <= n, without overflow
        else: hi = mid - 1
    return lo
```

Time is $O(\log(\mathrm{range}/\mathrm{tol}))$ iterations for the float version and $O(\log n)$ for the integer one, with $O(1)$ memory. Newton's update $y \leftarrow (y + x/y)/2$ doubles the number of correct digits per step once close, so after the exponent reduction about 5 steps from $y = 1$ reach double precision; bisection gains one bit per step but needs only comparisons and always stays inside its bracket.
:::

## 8. Statistics for experiments

Research interviews increasingly ask how you would know that an improvement is real. The tools are standard; the ML-specific part is knowing which source of randomness each one accounts for.

### Confidence intervals

A 95% **confidence interval** is a procedure that, over repeated experiments, covers the true parameter 95% of the time. It is not a 95% probability that the parameter lies in this particular interval; that is the Bayesian credible interval of Section 3. Most intervals in practice come from the central limit theorem: estimate $\pm$ 1.96 standard errors.

*Accuracy on a test set.* With $n$ independent test examples and accuracy $\hat p$, the standard error is $\sqrt{\hat p(1 - \hat p)/n}$ (the binomial of Section 1). At $\hat p = 0.9$: $n = 10{,}000$ gives SE 0.003 and a 95% interval of $\pm 0.59$ points; $n = 1{,}000$ gives $\pm 1.86$ points. So a 1-point gain on a 1,000-example test set is within noise from test-set sampling alone. For small $n$ or accuracies near 0 or 1, use the Wilson interval instead of this Wald interval.

*Mean over seeds.* With $n$ runs, mean $\bar x$ and sample standard deviation $s$, the interval is $\bar x \pm t_{0.975, n-1}\,s/\sqrt n$, using the Student $t$ quantile because $s$ is itself estimated: $t_{0.975, 4} = 2.78$ for 5 seeds and $t_{0.975, 2} = 4.30$ for 3 seeds, against 1.96 for large $n$.

### Hypothesis tests and p-values

A test fixes a **null hypothesis** $H_0$ (for example, "the two models have the same expected accuracy") and a test statistic. The **p-value** is the probability, computed assuming $H_0$, of a statistic at least as extreme as the one observed. Rejecting when $p \le \alpha$ makes the false-positive (type I) rate $\alpha$; the probability of detecting a real effect of a given size is the **power** $1 - \beta$.

Common misreadings, each of which interviewers like to probe:

- $p$ is not the probability that $H_0$ is true, and $1 - p$ is not the probability that the effect is real.
- $p$ is not "the probability the result is due to chance".
- A small $p$ does not mean a large or important effect; with enough data, tiny effects become significant. Report the effect size with an interval.
- $p > 0.05$ is not evidence of no effect; it may mean too little data.
- A p-value is valid only for a test planned in advance. Trying many tests, metrics, subsets or stopping times and reporting the best (p-hacking) invalidates it.

The American Statistical Association's statement on p-values (Wasserstein and Lazar, 2016) makes most of these points. The first misreading connects back to the disease test: how often a "significant" result is a true effect depends on the base rate of true effects. If 10% of tested ideas work, tests have power 0.8 and $\alpha = 0.05$, then significant results are true effects with probability $0.08/(0.08 + 0.045) = 0.64$, so about a third of significant findings are false positives (the argument of Ioannidis, 2005).

### The t-test for comparing two models

Suppose model A and model B were each trained with 5 seeds. The numbers below are made up for illustration.

| Seed | 1 | 2 | 3 | 4 | 5 | Mean | Std |
|---|---|---|---|---|---|---|---|
| A | 76.2 | 75.8 | 76.5 | 75.9 | 76.1 | 76.10 | 0.27 |
| B | 76.6 | 76.0 | 77.0 | 76.3 | 76.4 | 76.46 | 0.37 |
| B − A | 0.4 | 0.2 | 0.5 | 0.4 | 0.3 | 0.36 | 0.11 |

**Unpaired (Welch's) test.** Treat the runs as two independent samples:

$$
t = \frac{\bar x_B - \bar x_A}{\sqrt{s_A^2/n_A + s_B^2/n_B}} = \frac{0.36}{0.206} = 1.74,
$$

with the Welch-Satterthwaite degrees of freedom $\nu = (s_A^2/n_A + s_B^2/n_B)^2 / \big(\frac{(s_A^2/n_A)^2}{n_A - 1} + \frac{(s_B^2/n_B)^2}{n_B - 1}\big) \approx 7.4$, giving $p = 0.12$. Welch's version does not assume equal variances and is the safer default.

**Paired test.** If seed $i$ of A and seed $i$ of B share something that moves both results (the same data split, data order, or evaluation set), analyze the differences $d_i = B_i - A_i$:

$$
t = \frac{\bar d}{s_d/\sqrt n} = \frac{0.36}{0.114/\sqrt5} = \frac{0.36}{0.051} = 7.06, \qquad 4 \text{ degrees of freedom}, \quad p = 0.002,
$$

and a 95% interval for the gain of $0.36 \pm 2.78 \times 0.051 = [0.22, 0.50]$. Same data, very different conclusion: the seed-to-seed swings that both models share cancel in the differences. Pair only when the pairing is real; if the seeds control unrelated randomness in the two models, pairing by seed number is meaningless and Welch's test is right. Both tests assume roughly normal per-run results, which 5 runs cannot check, so a permutation test or the Wilcoxon signed-rank test is a sensible cross-check. For a single run of each model on the same test set, compare per-example correctness with McNemar's test on the examples where they disagree (Dietterich, 1998) or with the paired bootstrap below. Note that this covers test-set sampling but not training randomness: these are different sources of variance, and a careful comparison accounts for both.

### Bootstrap confidence intervals

The bootstrap estimates the sampling distribution of any statistic by resampling the data with replacement: draw $n$ items from the $n$ observed, recompute the statistic, repeat $B$ times, and take the 2.5% and 97.5% quantiles (the percentile interval). It needs no formula for the standard error, which makes it the default for metrics like F1, mAP or a difference of accuracies. For comparing two models on the same test set, resample *examples* and evaluate both models on the same resampled rows, so the comparison is paired.

```python
import numpy as np
rng = np.random.default_rng(0)

def paired_bootstrap_ci(a, b, n_boot=5000, alpha=0.05):
    """a, b: per-example scores (e.g. 1 if correct) of two models on the same test set."""
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))   # resample test examples with replacement
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)      # same rows for both models: paired
    return b.mean() - a.mean(), np.quantile(diffs, [alpha / 2, 1 - alpha / 2])

# Synthetic test set: B copies A's answer on 90% of examples and is right w.p. 0.95 on the rest.
n = 2000
a = (rng.random(n) < 0.82).astype(float)
b = np.where(rng.random(n) < 0.9, a, (rng.random(n) < 0.95).astype(float))
diff, (lo, hi) = paired_bootstrap_ci(a, b)
print(f"A {a.mean():.3f}  B {b.mean():.3f}  diff {diff:+.4f}  paired 95% CI [{lo:+.4f}, {hi:+.4f}]")

ia, ib = rng.integers(0, n, size=(2, 5000, n))             # unpaired: resample each model separately
d = b[ib].mean(axis=1) - a[ia].mean(axis=1)
print("unpaired 95% CI", np.round(np.quantile(d, [0.025, 0.975]), 4))
```

Output: accuracies 0.815 and 0.825, a difference of $+0.0100$, paired interval $[+0.0050, +0.0155]$, unpaired interval $[-0.0145, +0.0335]$. The paired interval excludes zero because the two models agree on most examples, so their per-example errors are strongly correlated; resampling them independently throws that information away. Two cautions: resample at the level where observations are independent (images rather than patches, prompts rather than the several samples drawn per prompt), and do not trust percentile intervals for very small $n$ or for statistics like a maximum.

### Multiple comparisons

Run $m$ independent tests at $\alpha = 0.05$ when every null is true, and the chance of at least one false positive is $1 - 0.95^m$: 0.40 for $m = 10$ and 0.64 for $m = 20$. Corrections:

- **Bonferroni:** test each at $\alpha/m$. It controls the family-wise error rate (the chance of any false positive) under any dependence, and is conservative.
- **Holm:** sort the p-values $p_{(1)} \le \dots \le p_{(m)}$, compare $p_{(i)}$ with $\alpha/(m - i + 1)$, and reject until the first failure. It also controls the family-wise error rate and never rejects less than Bonferroni.
- **Benjamini-Hochberg:** find the largest $i$ with $p_{(i)} \le i\alpha/m$ and reject hypotheses $1, \dots, i$. It controls the false discovery rate (the expected fraction of rejections that are false) under independence or positive dependence, and is much less conservative.

*Example.* Ten sorted p-values $0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216$ at $\alpha = 0.05$. Unadjusted, 5 are significant. Bonferroni's threshold 0.005 keeps 1. Holm keeps 1, because $0.008 > 0.05/9 = 0.0056$. Benjamini-Hochberg keeps 2: $0.008 \le 2 \times 0.005$, but $0.039 > 3 \times 0.005$ and no later p-value passes its threshold.

In ML the same problem appears as selection: evaluating many configurations, checkpoints or prompts on the test set and reporting the best overfits the test set. Select on a validation set, evaluate the chosen configuration once on the test set, and report how many configurations were tried.

### How to report results over seeds

- Run several seeds (at least 3, more when the gap is small relative to the seed spread) and use the same seeds and data splits for every method, so comparisons can be paired.
- Report the mean with either the standard deviation (the spread of single runs) or the standard error or a confidence interval (the uncertainty of the mean), and say which; the standard error is $s/\sqrt n$, smaller than the standard deviation.
- With few seeds, show every run or the minimum and maximum. Never report the best seed alone, and never compare your best seed with a baseline's average.
- Give baselines the same tuning budget as your method.
- Report effect sizes with intervals rather than only p-values, and use paired tests or the paired bootstrap when runs share randomness.
- Randomize every source of variation you can (data sampling, data order, initialization, augmentation), not just the initialization seed. Bouthillier et al. (2021) found that data sampling, initialization and hyperparameter choice all change results markedly, and that randomizing more sources of variation across runs gives more reliable comparisons for the same compute.
- Henderson et al. (2018) showed that in deep reinforcement learning two sets of five seeds of the same algorithm could produce learning curves that look significantly different, and Agarwal et al. (2021) recommend interquartile means with stratified bootstrap intervals when only a few runs are affordable.

::: key
**How to say it in an interview.**

"I would first ask what varies: the test set, the training seed, or both. For seeds, I run the same seeds for both models and test the paired differences, which removes shared seed variance; for the test set, I use a paired bootstrap over examples. I report the gain with a confidence interval rather than just a p-value, and if I compared many variants I correct for multiple comparisons or confirm on held-out data."
:::

### Practice questions

::: qa
**Model B beats model A by 0.3 points, averaged over 3 seeds each. Is the improvement real?**

Not knowable from the means alone. Compare the gap with the seed-to-seed spread: if each model's standard deviation is about 0.3, the Welch standard error of the difference is $\sqrt{0.09/3 + 0.09/3} \approx 0.24$, so $t \approx 1.2$, far from significant with so few degrees of freedom. If the runs share seeds and data splits, test the paired differences, which can be much tighter. Also check test-set noise (on 1,000 examples at 90% accuracy, the 95% interval is about $\pm 1.9$ points from test sampling alone) and whether the baseline got the same tuning budget. The honest answer is an interval for the gain, with more seeds if it straddles zero.
:::

::: qa
**What does $p = 0.03$ mean, and what does it not mean?**

If the null hypothesis were true, a result at least this extreme would occur 3% of the time. It is not a 3% chance that the null is true, not a 97% chance the effect is real, and not a measure of the effect's size or importance. Whether the finding is likely real also depends on the prior plausibility of the effect and on how many tests were run, as the 64% calculation above shows.
:::

::: qa
**You evaluated 20 hyperparameter configurations on the test set and report the best. What is wrong, and how do you fix it?**

The maximum of 20 noisy estimates is biased upward: even if all configurations were equal, the best would look better than average, and with 20 tests at $\alpha = 0.05$ the chance of at least one spurious "significant" win is 64%. Select on a validation set (or by cross-validation), evaluate only the chosen configuration on the test set, report the number of configurations tried, and if several comparisons are reported, apply Holm or Benjamini-Hochberg.
:::

## Rapid-fire review

::: {.qa asked="ByteDance, Nowcoder 2022"}
**Two players take turns tossing a fair coin, and the first to get heads wins. What is the probability that the first player wins?**

$2/3$; with heads probability $p$ it is $1/(2-p)$. First-step analysis: $P = p + (1 - p)(1 - P)$. Equivalently the series $\sum_k(1-p)^{2k}p = p/(1 - (1-p)^2)$. A simulation of $10^6$ games gives about 0.667; with three players the first wins with probability $4/7$.
:::

::: {.qa asked="ByteDance algorithm intern, Nowcoder 2020"}
**Player A throws a die $n$ times and keeps the maximum; player B throws once. What is the probability that B's throw is larger?**

$P(X \le k) = (k/6)^n$ for A's maximum, so $P(Y > X) = \frac16\sum_{k=0}^{5}(k/6)^n$: $5/12$ for $n = 1$, $55/216 \approx 0.255$ for $n = 2$, $25/144 \approx 0.174$ for $n = 3$, falling quickly.
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**What is the expected maximum of $n$ i.i.d. Uniform(0, 1) variables?**

$P(M \le x) = x^n$, so $\mathbb{E}[M] = \int_0^1(1 - x^n)\,dx = n/(n+1)$. The $k$-th smallest has mean $k/(n+1)$: the points split the interval into $n + 1$ gaps of equal expected length.
:::

::: {.qa asked="ByteDance algorithm intern, Nowcoder 2020"}
**Prove that the point minimizing the sum of squared distances to a set of points is their centroid.**

$f(x) = \sum_i\lVert x - p_i\rVert^2$ has gradient $2\sum_i(x - p_i)$, zero at $x = \bar p$, and Hessian $2nI \succ 0$, so $\bar p$ is the unique global minimum. Equivalently $f(x) = n\lVert x - \bar p\rVert^2 + \sum_i\lVert p_i - \bar p\rVert^2$ because the cross term vanishes. This is the k-means update; with unsquared distances the minimizer is the geometric median.
:::

::: {.qa asked="bigo image/CV, Nowcoder 2019"}
**What is the expected length of a random chord of a circle?**

It depends on how the chord is sampled (Bertrand's paradox). For radius $R$: two uniform endpoints give $4R/\pi \approx 1.27R$; a uniform distance from the centre gives $\pi R/2 \approx 1.57R$; a midpoint uniform in the disk gives $4R/3 \approx 1.33R$ (derivations in Section 1; Monte Carlo agrees). State the sampling assumption first.
:::

::: {.qa asked="SenseTime, Nowcoder 2022"}
**Without a calculator, how do you decide whether $a^b$ or $b^a$ is larger?**

Compare $\ln a/a$ with $\ln b/b$. $f(x) = \ln x/x$ increases up to $e$ and decreases after, so for $e \le a < b$, $a^b > b^a$ ($3^4 > 4^3$), for $a < b \le e$ the reverse, $e^\pi > \pi^e$, and $2^b > b^2$ exactly when $b > 4$ for $b > e$, with $2^4 = 4^2$. If $a < 1 < b$ then $a^b < 1 < b^a$.
:::

::: {.qa asked="SenseTime, Nowcoder 2022"}
**100 lamps start off; student $k$ toggles every lamp whose number is a multiple of $k$. Which lamps are on at the end?**

The perfect squares $1, 4, \dots, 100$, ten lamps. Lamp $n$ is toggled once per divisor, divisors pair as $(d, n/d)$, and only a square has an unpaired divisor $\sqrt n$, so only squares are toggled an odd number of times. Lamps toggled exactly twice are the 25 primes up to 100.
:::

::: qa
**What is the expected number of flips until HH? Until HT?**

6 and 4. First-step analysis for HH: $E_0 = 1 + \frac12E_1 + \frac12E_0$ and $E_1 = 1 + \frac12E_0$ give $E_0 = 6$. With heads probability $p$, HH takes $(1 + p)/p^2$.
:::

::: qa
**How many rolls of a die until every face has appeared?**

$6H_6 = 6(1 + \frac12 + \dots + \frac16) = 14.7$ on average (coupon collector), with standard deviation about 6.2. In general $nH_n \approx n\ln n + 0.577n$.
:::

::: qa
**How many people for a better-than-even chance of a shared birthday?**

23: $1 - \prod_{k=0}^{22}(1 - k/365) = 0.507$. The number of pairs, $\binom{23}{2} = 253$, is what grows quickly.
:::

::: qa
**What is the expected number of fixed points of a random permutation?**

1, for every $n$: by linearity, $\sum_i P(\pi(i) = i) = n\cdot\frac1n$, even though the indicators are dependent.
:::

::: qa
**Does zero correlation imply independence?**

No: $X \sim \mathcal{N}(0,1)$ and $Y = X^2$ are uncorrelated but dependent. It does for jointly Gaussian variables, but not for variables that are merely Gaussian marginally.
:::

::: qa
**State the law of total variance and say what each term means.**

$\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X \mid Y)] + \mathrm{Var}(\mathbb{E}[X \mid Y])$: average within-group variance plus variance of the group means. For a 50/50 mixture of $\mathcal{N}(\pm 2, 1)$ it is $1 + 4 = 5$.
:::

::: qa
**What is $\mathrm{KL}(\mathcal{N}(\mu, \sigma^2) \,\Vert\, \mathcal{N}(0, 1))$?**

$\frac12(\mu^2 + \sigma^2 - 1 - \log\sigma^2)$, the VAE regularizer per latent dimension; 0.818 for $\mu = 1$, $\sigma = 0.5$.
:::

::: qa
**What is the gap between $\log p(x)$ and the ELBO?**

$\mathrm{KL}(q(z \mid x) \,\Vert\, p(z \mid x))$, the reverse KL from the approximate posterior to the true one. The bound is tight exactly when $q$ is the true posterior.
:::

::: qa
**How does Monte Carlo error scale with the number of samples?**

As $\sigma_f/\sqrt n$, independent of dimension: four times the samples for half the error.
:::

::: qa
**Why does the Metropolis-Hastings acceptance ratio not need the normalizing constant?**

It enters as $\pi(x')/\pi(x)$, where the constant cancels; detailed balance only constrains ratios of the target density.
:::

::: qa
**What is the condition number of an orthogonal matrix, and of $X^\top X$ relative to $X$?**

1, since all singular values equal 1. $\kappa(X^\top X) = \kappa(X)^2$, which is why least squares is solved with QR or the SVD rather than the normal equations.
:::

::: qa
**What are the eigenvalues of an orthogonal projection matrix, and its trace?**

0 and 1 (since $P^2 = P$ forces $\lambda^2 = \lambda$); the trace equals the rank, the dimension of the subspace projected onto.
:::

::: qa
**What is the gradient of $\log\det\Sigma$ with respect to $\Sigma$?**

$\Sigma^{-1}$ for symmetric positive definite $\Sigma$ (in general $X^{-\top}$), from $\det(X + H) \approx \det X\,(1 + \mathrm{tr}(X^{-1}H))$.
:::

::: qa
**When does Newton's method converge in one step?**

On a strictly convex quadratic, since the quadratic model is exact; the condition number does not matter.
:::

::: qa
**Gradient descent with step $1/L$ on a $\mu$-strongly convex, $L$-smooth function: how many steps to accuracy $\epsilon$?**

$O(\kappa\log(1/\epsilon))$ with $\kappa = L/\mu$, from the per-step factor $1 - \mu/L$; Nesterov acceleration improves this to $O(\sqrt\kappa\log(1/\epsilon))$, and tuned heavy-ball momentum achieves it on quadratics.
:::

::: qa
**What do the KKT multipliers tell you?**

They are sensitivities: $\lambda_i$ is the rate at which the optimal value drops as constraint $i$ is loosened, and complementary slackness makes it zero for inactive constraints.
:::

::: qa
**What is a 95% interval for a 90% accuracy measured on 1,000 test examples?**

$0.9 \pm 1.96\sqrt{0.09/1000} = 0.9 \pm 0.019$, about $\pm 1.9$ points.
:::

## References

1. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006.
2. Murphy. *Probabilistic Machine Learning: An Introduction*. MIT Press, 2022; and *Probabilistic Machine Learning: Advanced Topics*, 2023.
3. Boyd, Vandenberghe. *Convex Optimization*. Cambridge University Press, 2004.
4. Cover, Thomas. *Elements of Information Theory*, 2nd ed. Wiley, 2006.
5. Nesterov. *Introductory Lectures on Convex Optimization: A Basic Course*. Kluwer, 2004 (expanded as *Lectures on Convex Optimization*, Springer, 2018).
6. Robert, Casella. *Monte Carlo Statistical Methods*, 2nd ed. Springer, 2004.
7. Wasserman. *All of Statistics*. Springer, 2004.
8. Casella, Berger. *Statistical Inference*, 2nd ed. Duxbury, 2002.
9. Golub, Van Loan. *Matrix Computations*, 4th ed. Johns Hopkins University Press, 2013.
10. Trefethen, Bau. *Numerical Linear Algebra*. SIAM, 1997.
11. Nocedal, Wright. *Numerical Optimization*, 2nd ed. Springer, 2006.
12. Owen. *Monte Carlo Theory, Methods and Examples*. 2013.
13. Efron, Tibshirani. *An Introduction to the Bootstrap*. Chapman and Hall, 1993.
14. Eckart, Young. The approximation of one matrix by another of lower rank. *Psychometrika*, 1936.
15. Metropolis, Rosenbluth, Rosenbluth, Teller, Teller. Equation of State Calculations by Fast Computing Machines. *Journal of Chemical Physics*, 1953.
16. Hastings. Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 1970.
17. Roberts, Gelman, Gilks. Weak convergence and optimal scaling of random walk Metropolis algorithms. *Annals of Applied Probability*, 1997.
18. Welling, Teh. Bayesian Learning via Stochastic Gradient Langevin Dynamics. ICML 2011.
19. Song, Ermon. Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS 2019. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
20. Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
21. Kingma, Welling. Auto-Encoding Variational Bayes. ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
22. Rezende, Mohamed, Wierstra. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. ICML 2014. [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)
23. Rezende, Mohamed. Variational Inference with Normalizing Flows. ICML 2015. [arXiv:1505.05770](https://arxiv.org/abs/1505.05770)
24. Dinh, Sohl-Dickstein, Bengio. Density Estimation using Real NVP. ICLR 2017. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
25. Chen, Rubanova, Bettencourt, Duvenaud. Neural Ordinary Differential Equations. NeurIPS 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
26. Grathwohl, Chen, Bettencourt, Sutskever, Duvenaud. FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. ICLR 2019. [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)
27. Goodfellow et al. Generative Adversarial Nets. NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
28. Arjovsky, Chintala, Bottou. Wasserstein GAN. ICML 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
29. Jang, Gu, Poole. Categorical Reparameterization with Gumbel-Softmax. ICLR 2017. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
30. Maddison, Mnih, Teh. The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. ICLR 2017. [arXiv:1611.00712](https://arxiv.org/abs/1611.00712)
31. Blei, Kucukelbir, McAuliffe. Variational Inference: A Review for Statisticians. *JASA*, 2017. [arXiv:1601.00670](https://arxiv.org/abs/1601.00670)
32. van den Oord, Li, Vinyals. Representation Learning with Contrastive Predictive Coding. 2018. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
33. Vitter. Random Sampling with a Reservoir. *ACM Transactions on Mathematical Software*, 1985.
34. Efraimidis, Spirakis. Weighted random sampling with a reservoir. *Information Processing Letters*, 2006.
35. Vose. A Linear Algorithm for Generating Random Numbers with a Given Distribution. *IEEE Transactions on Software Engineering*, 1991.
36. Miyato, Kataoka, Koyama, Yoshida. Spectral Normalization for Generative Adversarial Networks. ICLR 2018. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
37. Robbins, Monro. A Stochastic Approximation Method. *Annals of Mathematical Statistics*, 1951.
38. Polyak. Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 1964.
39. Bottou, Curtis, Nocedal. Optimization Methods for Large-Scale Machine Learning. *SIAM Review*, 2018. [arXiv:1606.04838](https://arxiv.org/abs/1606.04838)
40. Ghadimi, Lan. Stochastic First- and Zeroth-order Methods for Nonconvex Stochastic Programming. *SIAM Journal on Optimization*, 2013. [arXiv:1309.5549](https://arxiv.org/abs/1309.5549)
41. Dauphin, Pascanu, Gulcehre, Cho, Ganguli, Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. NeurIPS 2014. [arXiv:1406.2572](https://arxiv.org/abs/1406.2572)
42. Lee, Simchowitz, Jordan, Recht. Gradient Descent Only Converges to Minimizers (arXiv title: Gradient Descent Converges to Minimizers). COLT 2016. [arXiv:1602.04915](https://arxiv.org/abs/1602.04915)
43. Ge, Huang, Jin, Yuan. Escaping From Saddle Points: Online Stochastic Gradient for Tensor Decomposition. COLT 2015. [arXiv:1503.02101](https://arxiv.org/abs/1503.02101)
44. Jin, Ge, Netrapalli, Kakade, Jordan. How to Escape Saddle Points Efficiently. ICML 2017. [arXiv:1703.00887](https://arxiv.org/abs/1703.00887)
45. Du, Jin, Lee, Jordan, Poczos, Singh. Gradient Descent Can Take Exponential Time to Escape Saddle Points. NeurIPS 2017. [arXiv:1705.10412](https://arxiv.org/abs/1705.10412)
46. Wasserstein, Lazar. The ASA Statement on p-Values: Context, Process, and Purpose. *The American Statistician*, 2016.
47. Ioannidis. Why Most Published Research Findings Are False. *PLoS Medicine*, 2005.
48. Holm. A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 1979.
49. Benjamini, Hochberg. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *JRSS B*, 1995.
50. Dietterich. Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 1998.
51. Bouthillier et al. Accounting for Variance in Machine Learning Benchmarks. MLSys 2021. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
52. Henderson, Islam, Bachman, Pineau, Precup, Meger. Deep Reinforcement Learning that Matters. AAAI 2018. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
53. Agarwal, Schwarzer, Castro, Courville, Bellemare. Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
