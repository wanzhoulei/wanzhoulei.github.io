---
title: "Deep Learning Training Essentials"
subtitle: "Losses as likelihoods, gradient flow, initialization, optimizers, schedules, numerics, EMA, regularization, scale and debugging, with the derivations interviewers ask for."
order: 11
updated: 2026-11-08
---

These training fundamentals show up in nearly every ML research interview: why cross-entropy is the loss it is, why deep nets have gradient trouble, what Adam computes, why warmup and decay help, what bf16 buys, how memory and parallelism work at scale, and how to debug a failing run. The questions are short ("Adam vs AdamW?"); the follow-ups test whether you can derive the answer. Notation: $\theta$ parameters, $g_t$ stochastic gradient, $\eta$ learning rate, $B$ batch size, $\Psi$ parameter count. Normalization layers and attention are in [vision backbones](vision-backbones.html).

## 1. Losses as likelihoods

### MLE, MAP, and where regularizers come from

Maximum likelihood picks $\hat\theta_{\text{MLE}}=\arg\max_\theta\sum_i\log p_\theta(y_i\mid x_i)$. MAP adds a prior and returns the posterior mode:

$$
\hat\theta_{\text{MAP}} = \arg\max_\theta \Big[\sum_{i} \log p_\theta(y_i\mid x_i) + \log p(\theta)\Big].
$$

A prior $\mathcal{N}(0,\tau^2 I)$ contributes $-\|\theta\|^2/(2\tau^2)$, an L2 penalty; a Laplace prior gives L1. On the likelihood side, Gaussian noise with fixed variance gives $-\log p=(y-f_\theta(x))^2/(2\sigma^2)+\text{const}$, so MSE is MLE under Gaussian noise, and a categorical likelihood gives cross-entropy.

::: qa
**What is the difference between MLE and MAP, and how does L2 regularization fit in?**

MLE maximizes the likelihood alone; MAP maximizes likelihood times prior and returns the posterior mode. In log space a zero-mean Gaussian prior adds $-\|\theta\|^2/(2\tau^2)$, which is exactly an L2 penalty; a Laplace prior gives L1. As data grows the likelihood dominates and MAP approaches MLE. MAP is still a point estimate: it does not average over the posterior, and the mode moves if you reparameterize $\theta$.
:::

### Cross-entropy, KL, and forward vs reverse KL

$$
H(p,q) = -\sum_y p(y)\log q(y) = H(p) + \mathrm{KL}(p\,\|\,q).
$$

With $\hat p$ the empirical data distribution, the average training NLL equals $H(\hat p,q_\theta)$, and $H(\hat p)$ does not depend on $\theta$. So minimizing cross-entropy, maximizing likelihood and minimizing the forward KL $\mathrm{KL}(\hat p\,\|\,q_\theta)$ are the same problem. Forward KL blows up where $p>0$ but $q\approx0$, so it is mass-covering: a single Gaussian fit to a bimodal $p$ straddles both modes. Reverse KL $\mathrm{KL}(q\,\|\,p)$ blows up where $q>0$ but $p\approx0$, so it is mode-seeking: the Gaussian locks onto one mode, which is why variational inference tends to underestimate posterior variance. Likelihood-based generative models (autoregressive, VAEs, diffusion through its ELBO) use forward KL (see [GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html)).

::: qa
**Why is minimizing cross-entropy the same as minimizing KL divergence? Which KL?**

$H(p,q)=H(p)+\mathrm{KL}(p\|q)$, and with $p$ the data distribution $H(p)$ is constant in the model parameters, so both have the same minimizers and gradients. It is the forward KL, data first, so MLE is forward-KL minimization. The constant offset is why cross-entropy bottoms out at the data entropy, not at zero, when labels are noisy.
:::

### Softmax and the logit gradient

With logits $z$ and temperature $T$, $q_k=e^{z_k/T}/\sum_j e^{z_j/T}$. For label $y$, $\mathcal{L}=-z_y/T+\log\sum_j e^{z_j/T}$, so

$$
\frac{\partial \mathcal{L}}{\partial z_k} = \frac{1}{T}\big(q_k - \mathbb{1}[k=y]\big).
$$

At $T=1$ this is $q-y$, bounded and zero only at a perfect prediction. Softmax is shift-invariant, so implementations subtract $\max_j z_j$ first (log-sum-exp). $T\to0$ approaches argmax; $T\to\infty$ approaches uniform.

::: qa
**Derive the softmax cross-entropy gradient with respect to the logits. Why fuse log and softmax?**

Write $\mathcal{L}=-z_y+\log\sum_j e^{z_j}$; differentiating gives $-\mathbb{1}[k=y]+e^{z_k}/\sum_j e^{z_j}=q_k-y_k$. Fusing lets you compute $\log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m}$ with $m=\max_j z_j$, which avoids overflow in $e^z$ and $\log 0$ when a probability underflows. A separate log of a softmax can give $-\infty$ or NaN in 16-bit.
:::

### Logistic loss

With $p=\sigma(z)$, $z=w^\top x$, the Bernoulli NLL is $\ell=-y\log p-(1-y)\log(1-p)$, and since $\sigma'=\sigma(1-\sigma)$, $\partial\ell/\partial z=p-y$. Its Hessian in $w$ is $\sum_i p_i(1-p_i)x_ix_i^\top\succeq0$, so it is convex.

::: {.qa asked="TikTok DS (Glassdoor 2026); Tencent intern (Nowcoder 2019)"}
**What is the loss function for logistic regression, and why use log loss instead of mean squared error?**

The loss is the negative Bernoulli log-likelihood (binary cross-entropy), the MLE objective for a Bernoulli model. Its logit gradient $\sigma(z)-y$ is large when the model is confidently wrong. With MSE on the sigmoid, $\partial_z(\sigma-y)^2=2(\sigma-y)\sigma(1-\sigma)$, which vanishes when the sigmoid saturates, including on the wrong side, so learning stalls exactly when it should be fastest. MSE through a sigmoid is also non-convex in $w$; log loss is convex.
:::

### Bias–variance decomposition

Let $y=f(x)+\epsilon$ ($\mathbb{E}\epsilon=0$, $\operatorname{Var}\epsilon=\sigma^2$), $\hat f_{\mathcal{D}}$ trained on random data, $\bar f=\mathbb{E}_{\mathcal{D}}\hat f_{\mathcal{D}}$. Write $y-\hat f=\epsilon+(f-\bar f)+(\bar f-\hat f)$ and square. The cross terms vanish because $\epsilon$ is independent with mean zero and $\mathbb{E}_{\mathcal{D}}[\bar f-\hat f]=0$:

$$
\mathbb{E}\big[(y-\hat f_{\mathcal{D}}(x))^2\big] = \sigma^2 + \big(f(x)-\bar f(x)\big)^2 + \mathbb{E}_{\mathcal{D}}\big[(\hat f_{\mathcal{D}}(x)-\bar f(x))^2\big].
$$

::: {.qa asked="JD algorithm engineer, Nowcoder 2019"}
**Derive the bias–variance decomposition. Does "bigger model, more variance" hold for deep nets?**

Add and subtract $\bar f$; the cross terms vanish in expectation, leaving noise, squared bias and variance. The decomposition is exact for squared error, but the classical U-curve is not guaranteed. Past the interpolation threshold, test error often falls again as models grow (double descent, Section 7), because the training algorithm picks a low-complexity interpolant among many.
:::

## 2. Gradient flow and initialization

### Vanishing and exploding gradients

For $h_\ell=\phi(W_\ell h_{\ell-1})$, backprop multiplies one Jacobian per layer:

$$
\frac{\partial\mathcal{L}}{\partial h_0} = J_1^\top \cdots J_L^\top \frac{\partial\mathcal{L}}{\partial h_L},\qquad J_\ell = \operatorname{diag}\big(\phi'(a_\ell)\big)\,W_\ell .
$$

If typical singular values of $J_\ell$ are below 1 the gradient shrinks geometrically with depth; above 1 it grows. Causes: saturating activations ($\sigma'\le1/4$), badly scaled initialization, depth, and in RNNs the same $W_{hh}$ multiplied at every step.

::: {.qa asked="ByteDance algorithm engineer, Nowcoder 2022"}
**What causes vanishing and exploding gradients, and how do you handle them?**

Backprop multiplies per-layer Jacobians $\operatorname{diag}(\phi'(a_\ell))W_\ell$, so gradient norms change exponentially with depth. Vanishing comes from saturating activations (sigmoid's derivative is at most $1/4$), weights that are too small, or recurrent weights with spectral radius below 1; exploding from the opposite. Fixes: ReLU-family activations, variance-preserving initialization, residual connections, normalization layers, gating in RNNs, and gradient-norm clipping for explosions. Clipping treats only the exploding case.
:::

### Residual connections

With $h_{\ell+1}=h_\ell+F_\ell(h_\ell)$,

$$
\frac{\partial h_L}{\partial h_\ell} = \prod_{k=\ell}^{L-1}\Big(I+\frac{\partial F_k}{\partial h_k}\Big) = I + \sum_k \frac{\partial F_k}{\partial h_k} + \cdots ,
$$

so the gradient at any layer contains $\partial\mathcal{L}/\partial h_L$ unattenuated. The cost is in the forward pass: with independent branches, $\operatorname{Var}(h_L)\approx\operatorname{Var}(h_0)+\sum_k\operatorname{Var}(F_k)$ grows with depth. That is why skips are paired with normalization and with zero or small initialization of each branch's last layer (Goyal et al. zero-initialize the last BN scale in each residual block).

::: {.qa asked="Ping An Technology CV algorithm, Nowcoder 2019"}
**Why do residual connections make very deep networks trainable?**

Each block's Jacobian is $I+\partial F/\partial h$, so the product over blocks expands into the identity plus branch terms, and the top-layer gradient reaches every layer directly, whatever the branches do. At initialization a block whose branch outputs near zero is close to the identity, so a deep residual net starts near a shallow function and uses depth as branches learn. The caveat is that activation variance accumulates across blocks, which normalization and downscaled branch initialization control.
:::

::: key
Most training fixes are about keeping signal scale constant across depth and across time: initialization keeps it at step 0, residuals and normalization keep it across layers, and warmup, clipping and adaptive optimizers keep step sizes sane over time.
:::

### Initialization: the variance argument

For $y_i=\sum_{j=1}^{n_{\text{in}}}W_{ij}x_j$ with i.i.d. zero-mean weights of variance $v$, independent of $x$: $\operatorname{Var}(y_i)=n_{\text{in}}\,v\,\mathbb{E}[x_j^2]$.

- Forward, zero-mean inputs: $v=1/n_{\text{in}}$. Backward ($\partial_x=W^\top\partial_y$): $v=1/n_{\text{out}}$.
- **Xavier/Glorot** compromises: $v=2/(n_{\text{in}}+n_{\text{out}})$, e.g. $U[-a,a]$ with $a=\sqrt{6/(n_{\text{in}}+n_{\text{out}})}$, since $\operatorname{Var}(U[-a,a])=a^2/3$.
- **He/Kaiming** for ReLU: if the previous pre-activation $y'$ is symmetric with mean zero, $x=\max(0,y')$ has $\mathbb{E}[x^2]=\tfrac12\operatorname{Var}(y')$, so $\operatorname{Var}(y)=\tfrac12 n_{\text{in}}v\operatorname{Var}(y')$ and $v=2/n_{\text{in}}$. For conv, $n_{\text{in}}=k^2c_{\text{in}}$.

::: qa
**Derive He initialization. Why $2/n_{\text{in}}$ instead of Xavier's value?**

$\operatorname{Var}(y_i)=n_{\text{in}}v\,\mathbb{E}[x^2]$, and the key step is using $\mathbb{E}[x^2]$, not $\operatorname{Var}(x)$, because ReLU outputs are not zero-mean. A symmetric zero-mean pre-activation keeps half its second moment through ReLU, so preserving variance needs $v=2/n_{\text{in}}$; the 2 compensates for the zeroed half. Xavier assumes activations roughly linear around zero, and with ReLU the signal shrinks by about $\sqrt2$ per layer, which matters in deep plain nets.
:::

::: pitfall
"He init gives unit-variance outputs" holds only at initialization and only under the independence assumptions. It does not account for residual sums (variance accumulates), attention, unnormalized inputs, or anything after training starts.
:::

## 3. Optimizers

### SGD, momentum, Nesterov

Heavy ball (PyTorch convention): $v_t=\mu v_{t-1}+g_t$, $\theta_t=\theta_{t-1}-\eta v_t$. Unrolled, $v_t=\sum_k\mu^kg_{t-k}$; under a constant gradient $v\to g/(1-\mu)$, so the effective step is $\eta/(1-\mu)$. Components that flip sign across steps cancel; consistent ones accumulate. On a quadratic with condition number $\kappa$, heavy ball with step size and momentum both tuned to the extreme Hessian eigenvalues contracts the error asymptotically by $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ per step, versus $(\kappa-1)/(\kappa+1)$ for gradient descent with its best fixed step (Polyak's classical result); the speedup need not carry over beyond quadratics. Nesterov evaluates the gradient at the look-ahead point; in the form of Sutskever et al. (2013), where $v$ absorbs the step size, $v_{t+1}=\mu v_t-\eta\nabla f(\theta_t+\mu v_t)$, $\theta_{t+1}=\theta_t+v_{t+1}$.

::: {.qa asked="ByteDance algorithm engineer, Nowcoder 2022"}
**Explain SGD, momentum and Adam. How do they differ?**

SGD steps along the negative minibatch gradient with one global learning rate. Momentum keeps an exponential sum of past gradients, damping oscillation across steep directions, accelerating along consistent ones, and raising the effective step to $\eta/(1-\mu)$. Adam adds an EMA of squared gradients and divides by its square root per coordinate, giving each parameter its own step size and making the update roughly invariant to gradient scale. SGD with momentum stays competitive for CNN classification; Adam-family optimizers are the default for transformers and diffusion models.
:::

### Adam from first principles

Adam (defaults $\alpha=10^{-3},\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, & v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\\
\hat m_t &= m_t/(1-\beta_1^t), & \hat v_t &= v_t/(1-\beta_2^t),\\
\theta_t &= \theta_{t-1} - \alpha\,\hat m_t/(\sqrt{\hat v_t}+\epsilon). &&
\end{aligned}
$$

**Bias correction.** From $m_0=0$, $m_t=(1-\beta_1)\sum_{i=1}^t\beta_1^{t-i}g_i$. If $\mathbb{E}[g_i]=\mathbb{E}[g]$,

$$
\mathbb{E}[m_t] = \mathbb{E}[g]\,(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i} = \mathbb{E}[g]\,(1-\beta_1^t),
$$

and likewise for $v_t$. Without correction the update is the corrected one times $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$: about 3.2 at $t=1$, 6.5 at $t=10$, still 3.2 at $t=100$ with default betas, because $v$ stays biased toward zero much longer than $m$.

**Interpretation.** Ignoring $\epsilon$, the update is invariant to loss rescaling and has per-coordinate size about $\alpha$, a trust-region radius; with $\beta_1=\beta_2=0$ it is sign descent. $\epsilon$ is the gradient scale below which normalization stops: if gradient RMS nears $\epsilon$, updates shrink (Wortsman et al., 2023), and if $\epsilon$ is tiny, a coordinate with near-zero $v$ can take a huge step.

::: {.qa asked="ByteDance ML algorithm engineer, Nowcoder 2019"}
**Write Adam's update equations and derive the bias correction. What goes wrong without it?**

Adam keeps EMAs $m_t$ of $g_t$ and $v_t$ of $g_t^2$, bias-corrects both, and steps by $\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. Unrolling $m_t$ from zero gives weights that sum to $1-\beta_1^t$, not 1, so under stationary gradients $\mathbb{E}[m_t]=(1-\beta_1^t)\mathbb{E}[g]$; dividing by $1-\beta_1^t$ removes the bias, and the same holds for $v_t$ with $\beta_2$. Without it both moments are shrunk toward zero, but $v$ much more, because $\beta_2=0.999$ averages over about 1000 steps. The ratio $m_t/\sqrt{v_t}$ is then several times too large over the first hundred or so steps, which is when the second-moment estimate is least reliable.
:::

### AdamW: decoupled weight decay

With L2 in Adam, the penalty gradient is preconditioned: coordinate $i$ gets roughly $-\alpha(\nabla_i\mathcal{L}+\lambda\theta_i)/(\sqrt{\hat v_i}+\epsilon)$, so weights with large gradient history are decayed less. Loshchilov and Hutter decouple the decay:

$$
\theta_t = \theta_{t-1} - \eta_t\Big(\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Big),
$$

with $\eta_t$ the schedule multiplier. For plain SGD, L2 with coefficient $\lambda/\alpha$ is equivalent; with a non-scalar preconditioner no L2 coefficient is. Convention warning: PyTorch's `AdamW` multiplies the decay by the full learning rate ($\theta\leftarrow\theta-\gamma\lambda\theta$), so its $\lambda$ is not the paper's.

::: qa
**What is the difference between Adam with L2 regularization and AdamW?**

With L2, $\lambda\theta$ is added to the gradient, enters $m$ and $v$, and is divided by $\sqrt{\hat v}$. Parameters with large gradients are regularized less and those with small gradients more, an accidental coupling to gradient statistics. AdamW applies $\theta\leftarrow\theta-\eta\lambda\theta$ outside the adaptive step, so all weights shrink at the same relative rate. Under plain SGD the two coincide after rescaling $\lambda$ by the learning rate; under Adam they do not. The paper reports better generalization and much less coupling between the best learning rate and the best $\lambda$.
:::

### Why transformers favor Adam, and newer optimizers

The evidence on Adam's advantage is in Zhang et al. (2019), Kunstner et al. (2023, 2024) and Zhang et al. (2024), summarized in the question below. Three newer alternatives:

- **Lion** stores only momentum: $\theta_t=\theta_{t-1}-\eta_t(\operatorname{sign}(\beta_1m_{t-1}+(1-\beta_1)g_t)+\lambda\theta_{t-1})$, then $m_t=\beta_2m_{t-1}+(1-\beta_2)g_t$ (defaults 0.9, 0.99). The authors suggest a learning rate 3 to 10 times smaller than AdamW's and a weight decay 3 to 10 times larger.
- **Adafactor** stores only row and column sums of the second-moment EMA for each matrix, so state is $O(n+m)$ instead of $O(nm)$, and can drop momentum.
- **Muon** replaces the momentum update $G=USV^\top$ of each 2D hidden weight by an approximation of $UV^\top$ (a few Newton–Schulz iterations); other parameters use AdamW. With added weight decay and update rescaling, Liu et al. (2025) report about $2\times$ compute efficiency over AdamW in LLM scaling runs.

::: qa
**Why do transformers train much better with Adam than with SGD?**

Different parts of a transformer (embeddings, attention, MLP, norms) have very different gradient and curvature scales. Adam's per-coordinate normalization adapts to each, while SGD's single step size must suit the sharpest block and is then too slow elsewhere. Evidence: block-heterogeneous Hessian spectra, heavy-tailed gradient noise in attention, and a gap that survives full-batch training, which points to Adam's sign-like update more than to noise handling. For language, rare tokens are another factor. No single explanation covers every setting.
:::

## 4. Learning-rate schedules and batch size

### Warmup

Linear warmup ramps $\eta_t=\eta_{\max}t/T_w$. RAdam shows Adam's adaptive learning rate has huge variance with few samples (at $t=1$, $1/|g_1|$ for Gaussian $g_1$ has divergent variance) and replaces warmup with a rectification term. Kalra and Barkeshli (2024) find warmup mainly lets the network tolerate a larger target rate by first reaching better-conditioned regions. Goyal et al. need a 5-epoch gradual warmup for large batches, and Xiong et al. (2020) show Post-LN transformers have large output-side gradients at initialization, unlike Pre-LN.

::: qa
**Why does learning-rate warmup help, especially with Adam and large batches?**

Early on, Adam's $v$ averages only a few gradients, so per-coordinate learning rates are very noisy, and the loss surface near initialization tends to be sharp, so a full-size step can push the model somewhere it never recovers from. Small early steps let the statistics settle and the network reach flatter regions, after which a larger peak rate is stable. For large-batch SGD the same issue appears as linear scaling failing while weights change fast; in Post-LN transformers, as large output-side gradients at initialization.
:::

### Cosine and warmup-stable-decay

Cosine: $\eta_t=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})(1+\cos(\pi t/T))$. It needs $T$ in advance, and the Chinchilla authors found that matching the cycle length to the actual token budget matters. Warmup-stable-decay (WSD, introduced with MiniCPM) holds $\eta_{\max}$ for most of training and then decays quickly; a decay can branch from any stable-phase checkpoint. Hägele et al. (2024) find constant rate plus cooldown scales predictably, similar to cosine.

::: {.qa asked="Didi LLM algorithm, Nowcoder 2024"}
**How does the learning rate usually evolve in large-model training, why decay at the end, and why might you prefer WSD over cosine?**

Typically a short linear warmup, then cosine decay (Chinchilla decays about 10× over the run) or WSD's long constant phase plus a short anneal. At fixed $\eta$ the iterates hover in a band around the valley floor whose size grows with $\eta$ and with the gradient noise; decay lets them settle, which is why loss drops visibly during cooldown. Cosine bakes the total budget in, so stopping early or extending training gives a suboptimal curve. WSD adds a short decay whenever you need a final model, so one run yields good checkpoints at several budgets and can be resumed from the stable phase.
:::

### Batch size and learning-rate scaling

**Linear rule** (Goyal et al., 2017, for SGD with momentum). Compare $k$ SGD steps on minibatches $\mathcal{B}_j$ of size $n$ with one step on their union:

$$
\theta_{t+k}=\theta_t-\eta\frac1n\sum_{j<k}\sum_{x\in\mathcal{B}_j}\nabla\ell(x,\theta_{t+j}),\qquad
\hat\theta_{t+1}=\theta_t-\hat\eta\frac{1}{kn}\sum_{j<k}\sum_{x\in\mathcal{B}_j}\nabla\ell(x,\theta_t).
$$

If $\nabla\ell(x,\theta_{t+j})\approx\nabla\ell(x,\theta_t)$ they match for $\hat\eta=k\eta$. The approximation fails early in training (hence warmup) and when $\hat\eta$ exceeds the curvature limit. In SDE terms, $\eta/B$ sets SGD's noise scale.

**Square-root rule** (a heuristic). Gradient variance scales as $1/B$, so keeping the per-step update variance $\eta^2\sigma^2/B$ fixed gives $\eta\propto\sqrt B$. Krizhevsky (2014) states this argument but found linear scaling worked better for his SGD runs; Hoffer et al. (2017) used $\sqrt B$ scaling for SGD, motivated by matching the random-walk statistics of the weights, together with longer training. For Adam, Malladi et al. (2022) derive it from an SDE approximation: scaling the batch by $\kappa$ calls for $\eta\sqrt\kappa$, betas $1-\kappa(1-\beta)$ and $\epsilon/\sqrt\kappa$, valid only while the SDE approximation holds. Beyond a critical batch size (estimated by the gradient noise scale $\operatorname{tr}(\Sigma)/\|G\|^2$, McCandlish et al., 2018), larger batches give diminishing returns.

::: qa
**If you increase the batch size 8×, how should you change the learning rate?**

For SGD with momentum, start from the linear rule: multiply $\eta$ by 8 and add warmup, since one step on an 8× batch approximates 8 small steps when the gradient barely changes over them. It breaks down early in training, past the critical batch size, and when the scaled rate exceeds the curvature-based stability limit. For Adam, the square-root heuristic from SDE analysis (Malladi et al., 2022) suggests $\times\sqrt8\approx2.8$ with adjusted betas and $\epsilon$. Either way, sweep around the scaled value.
:::

### Gradient clipping and accumulation

Global-norm clipping, $g\leftarrow g\cdot\min(1,c/\|g\|_2)$, preserves direction, unlike per-element value clipping; it was popularized for RNN exploding gradients (Pascanu et al., 2012), and with fp16 loss scaling you unscale first. Gradient accumulation sums $k$ micro-batch gradients per step: effective batch $=$ micro-batch $\times k\times$ workers, at one micro-batch's memory.

::: qa
**Is gradient accumulation exactly equivalent to a larger batch? What are the gotchas?**

For a mean loss over independent examples, summing $k$ micro-batch gradients of (loss$/k$) matches the big-batch gradient up to floating-point order. Gotchas: BatchNorm statistics are per micro-batch; with variable-length sequences, averaging per-micro-batch token means is not the global token mean, so normalize by total tokens; under DDP, skip gradient sync on intermediate micro-steps (`no_sync`) or you all-reduce $k$ times; and step the scheduler per optimizer step, not per micro-batch.
:::

## 5. Numerics and mixed precision

| Format | Exponent bits | Mantissa bits | Max | Smallest normal | Spacing at 1 |
|---|---|---|---|---|---|
| fp32 | 8 | 23 | $\approx 3.4\times10^{38}$ | $\approx 1.2\times10^{-38}$ | $2^{-23}\approx1.2\times10^{-7}$ |
| fp16 | 5 | 10 | 65504 | $2^{-14}\approx6.1\times10^{-5}$ | $2^{-10}\approx9.8\times10^{-4}$ |
| bf16 | 8 | 7 | $\approx 3.4\times10^{38}$ | $\approx 1.2\times10^{-38}$ | $2^{-7}\approx7.8\times10^{-3}$ |

fp16 subnormals reach $2^{-24}\approx6\times10^{-8}$; anything smaller becomes zero. The mixed-precision recipe (Micikevicius et al., 2017) is: 16-bit weights, activations and gradients for compute; an **fp32 master copy** of the weights for the update; **loss scaling** (multiply the loss by $S$, so every gradient is scaled by $S$ and small values escape underflow; unscale before the update); and fp32 accumulation in matmuls and reductions. Dynamic scaling skips steps with inf/NaN gradients and halves $S$, then grows it after a run of clean steps.

Master weights are needed because an update below $2^{-24}$ is zero in fp16, and a representable update is still lost in $w+\Delta w$ once $|w|/|\Delta w|$ exceeds about $2^{11}=2048$ (10 mantissa bits). The same argument gives about $2^8$ for bf16, so bf16 training also keeps fp32 master weights or uses stochastic rounding.

::: {.qa asked="Alibaba Tongyi LLM, secondhand (Nowcoder 2025)"}
**Why is bf16 usually preferred over fp16 for training large models?**

bf16 has fp32's 8-bit exponent, so it has the same dynamic range: activations and gradients essentially never overflow or underflow, loss scaling is usually unnecessary, and spikes such as large attention logits do not hit fp16's 65504 ceiling. The cost is 7 mantissa bits instead of 10. That is usually tolerable because gradient noise dwarfs rounding error, provided accumulations, reductions and the weight update stay in fp32. For large models the practical win is fewer overflow-driven skipped steps and NaNs.
:::

::: {.qa asked="iFlytek AI research algorithm engineer, Nowcoder 2024"}
**In mixed precision, why keep fp32 master weights, and why does loss scaling work?**

Two ways 16-bit updates fail: tiny updates underflow to zero, and when the weight is about 2048 times larger than its update (fp16) the addition rounds the update away. Many small updates that should accumulate then do nothing, so the master copy accumulates them in fp32. Loss scaling targets a different problem, small gradients underflowing during backprop. Backprop is linear in the upstream gradient, so multiplying the loss by $S$ multiplies every gradient by $S$, and dividing by $S$ in fp32 before the step restores them. The only risk is overflow, which dynamic scaling detects.
:::

## 6. EMA of weights

Keep a shadow copy, used only for evaluation and sampling: $\bar\theta_t=\beta\bar\theta_{t-1}+(1-\beta)\theta_t$. Unrolled,

$$
\bar\theta_t = (1-\beta)\sum_{k=0}^{t-1}\beta^k\theta_{t-k} + \beta^t\bar\theta_0 ,
$$

a geometric average over roughly the last $1/(1-\beta)$ steps, with half-life $\approx\ln2/(1-\beta)$. Late-phase SGD iterates fluctuate around a valley at a scale set by $\eta$ and gradient noise; averaging reduces that variance (Polyak–Ruppert), like a learning-rate decay that does not slow the optimizer. DDPM used decay 0.9999, and NCSNv2 recommended EMA to remove global color shifts and FID swings between checkpoints.

Karras et al. (EDM2) use a **power-function** profile $\bar\theta_\gamma(t)\propto\int_0^t\tau^\gamma\theta(\tau)\,d\tau$, computed incrementally with $\beta_\gamma(t)=(1-1/t)^{\gamma+1}$. It gives zero weight to initialization, stretches with training length, and is parameterized by its relative width $\sigma_{\text{rel}}$. **Post-hoc EMA**: store periodic snapshots of two such averages ($\sigma_{\text{rel}}=0.05$ and $0.10$), then after training reconstruct any other profile as the least-squares best linear combination of the snapshots. They find the optimal length varies across configurations, shrinks as the learning rate grows, and interacts with guidance strength (see [diffusion models](diffusion-models.html)).

::: key
EMA is a variance-reduction trick on the parameter trajectory. It matters most when training runs at a non-decayed learning rate on a noisy objective, which is the normal regime for diffusion models.
:::

::: qa
**What does an EMA of weights do, and why do diffusion models depend on it?**

It averages the parameter trajectory over roughly $1/(1-\beta)$ steps, cancelling the jitter SGD leaves in the weights; the optimizer never sees it. Diffusion models train on a very high-variance objective (random timesteps and noise), often long at a near-constant learning rate, so raw weights jitter and their samples show global color shifts and FID swings between neighboring checkpoints. The EMA model behaves like a well-annealed one and usually samples better.
:::

::: qa
**How would you choose the EMA decay, and what is post-hoc EMA?**

$\beta=1-1/N$ averages about the last $N$ steps. Pick $N$ as a modest fraction of the run, avoid windows reaching back into early training (or ramp $\beta$ up), and expect longer runs and lower learning rates to want longer windows. EDM2's power-function EMA defines width relative to training time and never weights the initialization. Post-hoc EMA stores snapshots of two such averages during training and later solves a small least-squares problem to synthesize weights for any other profile, so EMA length can be swept against FID without retraining.
:::

::: pitfall
Keep the EMA buffer in fp32: with $\beta=0.9999$ the per-step increment $(1-\beta)(\theta_t-\bar\theta_{t-1})$ is far below bf16 resolution and rounds away. If the model has BatchNorm, the EMA weights need their own running statistics (average the buffers or recompute them).
:::

## 7. Regularization and generalization

**Weight decay** under SGD equals L2/Gaussian-prior MAP; with normalization, many weights are scale-invariant and decay mainly changes their effective learning rate. **Dropout** (Srivastava et al., 2014) zeroes units at random in training. Inverted dropout with drop probability $p$ uses $\tilde h=m\odot h/(1-p)$, $m_i\sim\text{Bernoulli}(1-p)$, so $\mathbb{E}[\tilde h]=h$ and evaluation is the identity; the original paper instead scaled weights by the retain probability at test time. It can be viewed as training $2^n$ weight-sharing subnetworks, with weight scaling an approximate average (exact for a single linear layer, approximate once nonlinearities are stacked). **Label smoothing** uses target $(1-\varepsilon)y+\varepsilon/K$; the loss becomes $(1-\varepsilon)H(y,q)+\varepsilon H(u,q)$ with $H(u,q)=\mathrm{KL}(u\|q)+\log K$, a penalty on overconfidence that keeps logit gaps finite. **Early stopping** limits how far the weights travel. For a quadratic loss (linear least squares, or a quadratic approximation around a minimum), gradient descent from $\theta=0$ with step $\eta$ stopped after $t$ steps approximately matches an L2 penalty of strength $\alpha\approx1/(\eta t)$; the match uses first-order expansions valid when $\eta\lambda_i\ll1$ and $\lambda_i\ll\alpha$ for the Hessian eigenvalues $\lambda_i$ (Goodfellow et al., 2016, Sec. 7.8). For deep nets it is an analogy, not an equivalence.

::: {.qa asked="ByteDance algorithm engineer, Nowcoder 2022"}
**What is the difference between L1 and L2 regularization?**

L2 adds $\lambda\|\theta\|_2^2$ (Gaussian prior); its gradient $2\lambda\theta$ shrinks weights in proportion to size, so they get small but rarely exactly zero. L1 adds $\lambda\|\theta\|_1$ (Laplace prior); its subgradient $\lambda\operatorname{sign}(\theta)$ pulls equally regardless of size, driving small weights exactly to zero, i.e. sparsity. Geometrically, loss contours tend to touch the L1 ball at its corners on the axes. L2 spreads weight across correlated features; L1 picks among them, which can be unstable. In deep nets, decoupled weight decay is the default.
:::

::: {.qa asked="Meizhi Tech CV algorithm, Nowcoder 2019"}
**How does dropout differ between training and inference, and why the $1/(1-p)$ factor?**

In training each unit is zeroed with probability $p$; at inference nothing is dropped. Without correction the next layer would see inputs about $1/(1-p)$ times larger at test time than in training, so inverted dropout divides kept units by $1-p$ during training and leaves inference untouched. Expectations match but variances do not, which is why dropout before BatchNorm can shift the statistics BN collected (Li et al., 2018). Forgetting `model.eval()` leaves dropout on at test time.
:::

::: qa
**What does label smoothing do, and when can it hurt?**

It adds a term proportional to $\mathrm{KL}(u\|q)$, so the model cannot lower the loss by pushing the correct logit to infinity; predictions become less overconfident and usually better calibrated. Müller et al. (2019) show it also squeezes each class's penultimate representations into tight, equidistant clusters, erasing information about which wrong classes are similar. That makes label-smoothed teachers worse for distillation.
:::

### Double descent and implicit regularization

Nakkiran et al. (2019) show test error versus model size following the classical U, peaking near the **interpolation threshold** (the smallest model that fits the training set), then descending again. Similar curves appear versus epochs, in some regimes more data hurts, and label noise sharpens the peak. Tuned ridge regularization can remove the peak in linear models.

Which interpolant training picks is implicit regularization. Gradient descent from zero on underdetermined least squares finds the minimum-norm solution; on separable data, logistic regression converges in direction to max-margin (Soudry et al., 2017). Gradient descent with step $h$ approximately follows $\mathcal{L}+\tfrac h4\|\nabla\mathcal{L}\|^2$, favoring flat regions (Barrett and Dherin, 2020), and SGD adds a minibatch-gradient-norm penalty scaling with $\eta/B$ (Smith et al., 2021). This helps explain the large-batch generalization gap (Keskar et al., 2016), much of which closes with adjusted learning rates and longer training (Hoffer et al., 2017).

::: qa
**What is double descent, and how does it square with the bias–variance trade-off?**

Test error versus capacity (or epochs) follows the classical U up to the interpolation threshold, peaks where the model can barely fit the training data, then falls again as capacity grows. At the threshold there is essentially one interpolant, which must contort to fit noise, so variance spikes; with more capacity, gradient-based training selects a smooth, small-norm interpolant. The decomposition still holds; monotone variance in parameter count does not. An epoch-wise version can mislead early stopping.
:::

::: {.qa asked="Tuniu algorithm engineer, Nowcoder 2018"}
**How do you diagnose overfitting versus underfitting, and what do you try first?**

Plot training and validation loss together. Overfitting: training loss keeps falling while validation loss rises or plateaus. Underfitting: both high and close. First rule out leakage, train/val shift and preprocessing mismatches, which mimic the same gap. For real overfitting try more or augmented data, stronger regularization, early stopping, then a smaller model. For underfitting train longer, add capacity, reduce regularization and check the optimization.
:::

## 8. Training at scale

### Memory math

For mixed-precision Adam, ZeRO counts 2 bytes fp16 weights, 2 bytes fp16 gradients and $K=12$ bytes of optimizer state (fp32 master weights, $m$, $v$): **$16\Psi$ bytes**, so 120 GB for 7.5B parameters before activations. Per transformer layer with 16-bit activations, Korthikanti et al. count about $sbh(34+5as/h)$ bytes ($s$ sequence, $b$ micro-batch, $h$ hidden, $a$ heads); the $5as^2b$ part is the attention matrix, which fused attention kernels do not store.

::: {.qa asked="XVERSE LLM algorithm, Nowcoder 2024"}
**Estimate the GPU memory needed to train a 7B-parameter model with Adam in mixed precision.**

Model states take 16 bytes per parameter (fp16 weights 2, grads 2, fp32 master 4, $m$ 4, $v$ 4): about 112 GB, more than one 80 GB GPU before activations. Activations scale with micro-batch $\times$ sequence length $\times$ hidden size $\times$ layers and shrink with checkpointing. To fit, shard states with ZeRO/FSDP ($112/N$ GB per GPU at full sharding), offload optimizer state to CPU, or shrink trainable state (LoRA, 8-bit optimizers).
:::

### Data parallelism, DDP, ZeRO and FSDP

Data parallelism averages gradients with all-reduce (a reduce-scatter plus an all-gather, about $2\Psi$ elements per worker). ZeRO removes the redundant copies of model state. With $N_d$ workers:

| Stage | Sharded | Per-GPU model states | Communication |
|---|---|---|---|
| DP | nothing | $16\Psi$ | $2\Psi$ |
| ZeRO-1 | optimizer states | $4\Psi+12\Psi/N_d$ | $2\Psi$ |
| ZeRO-2 | + gradients | $2\Psi+14\Psi/N_d$ | $2\Psi$ |
| ZeRO-3 | + parameters | $16\Psi/N_d$ | $3\Psi$ |

For 7.5B parameters and $N_d=64$: 120, 31.4, 16.6 and 1.9 GB. PyTorch FSDP implements the fully sharded scheme, with options matching the lighter stages.

::: {.qa asked="ByteDance algorithm engineer, Nowcoder 2022"}
**What is the difference between DataParallel and DistributedDataParallel in PyTorch?**

`DataParallel` is one process driving several GPUs with threads: each iteration it scatters the batch, replicates the model from GPU 0, and gathers outputs and gradients there, which brings GIL contention, unbalanced memory and poor scaling, and it runs on a single machine only. `DistributedDataParallel` runs one process per GPU with a persistent replica and all-reduces gradients in buckets during backward, overlapping communication with compute. It works across nodes and is recommended even on one machine.
:::

::: {.qa asked="iFlytek AI research algorithm engineer, Nowcoder 2024"}
**Explain ZeRO stages 1, 2 and 3 (or FSDP). What does each save and cost?**

ZeRO-1 shards the $12\Psi$ of optimizer state: gradients are reduce-scattered so each GPU gets the averaged gradient for its shard, updates that shard, and all-gathers the new weights, $\Psi+\Psi$, the same $2\Psi$ as an all-reduce. ZeRO-2 also shards gradient memory, keeping only its shard's reduced gradients, still $2\Psi$. ZeRO-3 (FSDP full sharding) also shards parameters, all-gathering each layer just in time in both forward and backward and freeing it afterwards. Per-GPU state becomes $16\Psi/N_d$ at about $3\Psi$ communication, which prefetching usually hides on fast interconnects.
:::

### Model parallelism and activation checkpointing

**Tensor parallelism** (Megatron-LM) splits matmuls inside a layer: in an MLP the first weight is split by columns and the second by rows, needing one all-reduce per block in each direction; this per-layer traffic usually keeps it within a node. **Pipeline parallelism** (GPipe) assigns contiguous layers to $K$ stages and streams $M$ micro-batches; the idle bubble is $(K-1)/(M+K-1)$, negligible for $M\ge4K$ in GPipe's experiments. **Activation checkpointing** stores a subset of activations and recomputes the rest in backward: $O(\sqrt n)$ memory for an $n$-layer chain at about one extra forward pass (Chen et al., 2016), roughly 33% more compute.

### Scaling laws

Training compute is $C\approx6ND$: about $2N$ FLOPs per token forward (a multiply and an add per weight) and $4N$ backward. Kaplan et al. (2020) fit $L(N)=(N_c/N)^{\alpha_N}$ with $\alpha_N\approx0.076$ (non-embedding parameters), $\alpha_D\approx0.095$ for data, and concluded $N_{\text{opt}}\propto C^{0.73}$. Hoffmann et al. (2022) found $N_{\text{opt}}\propto C^a$, $D_{\text{opt}}\propto C^b$ with $a\approx b\approx0.5$ (0.46 to 0.50 across three methods), about 20 tokens per parameter; their 70B Chinchilla on 1.4T tokens beat the 280B Gopher at equal compute. From their fit $L=E+A/N^\alpha+B/D^\beta$ ($\alpha=0.34$, $\beta=0.28$), substituting $D=C/(6N)$ and setting $\partial L/\partial N=0$:

$$
\alpha A N^{-\alpha} = \beta B\,(C/6)^{-\beta}N^{\beta}\;\Rightarrow\; N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}\approx C^{0.45},\quad D_{\text{opt}}\propto C^{\alpha/(\alpha+\beta)}\approx C^{0.55}.
$$

With unrounded exponents the paper reports 0.46 and 0.54 for this approach.

::: qa
**How do the Kaplan and Chinchilla scaling conclusions differ, and why?**

Kaplan et al. put extra compute mostly into model size ($N\propto C^{0.73}$); Chinchilla grows parameters and tokens equally ($\approx C^{0.5}$ each), so many large models of the time were undertrained. Chinchilla's main explanation: Kaplan used one schedule and token budget for all runs, so losses read partway through a long cosine schedule overestimate properly scheduled shorter runs, biasing the fit toward bigger models; Kaplan's runs were also mostly small. Neither accounts for inference cost, so deployed models often see far more tokens per parameter.
:::

## 9. Debugging training

Beyond the usual suspects (learning rate, fp16 overflow, unsafe ops, bad batches), transformers show growing attention logits and diverging output logits. Wortsman et al. (2023) reproduce these at small scale and mitigate them with qk-layernorm and z-loss. PaLM still saw about 20 spikes in its largest run despite clipping, and handled them by restarting about 100 steps earlier and skipping roughly 200 to 500 batches. A learning-rate range test (Smith, 2015) ramps the rate over a short run and picks a value below where loss starts to diverge.

::: qa
**Your loss goes to NaN after 20k steps. How do you debug it?**

Localize first: from logged per-layer gradient norms, activation statistics and loss scale, find the first non-finite step, rewind to a checkpoint just before it, replay that batch, and use anomaly detection or hooks to find the first op producing inf/NaN. Then check the usual causes: fp16 overflow (use bf16; keep softmax, losses and norms in fp32), a learning rate too high (a gradient-norm spike first; lower it or clip), an Adam coordinate with near-zero $v$ hit by a large gradient, unsafe ops ($\log 0$, tiny denominators), a corrupt sample, and transformer logit growth (qk-norm, z-loss). Skipping bad steps buys time; fix the systematic cause.
:::

::: qa
**What sanity checks do you run before launching a long training job?**

Initial loss near $\log K$. Overfit one small batch to near-zero loss, which tests model, loss and optimizer together. Inspect exactly what enters the network after preprocessing, including label alignment. Check that every parameter gets a nonzero gradient and that example $i$'s loss has zero gradient with respect to input $j\ne i$ (catches batch-mixing bugs). Run a short learning-rate sweep, verify checkpoint resume reproduces the loss curve, and confirm evaluation uses `eval()` mode and the intended EMA or raw weights.
:::

::: qa
**How do you make a run reproducible, and what remains non-deterministic?**

Seed Python, NumPy, PyTorch (CPU and CUDA) and the data loader's workers and sampler; enable deterministic kernels (`torch.use_deterministic_algorithms(True)`, deterministic cuDNN without benchmarking, the cuBLAS workspace setting); pin library versions. Bitwise equality is still not guaranteed across GPU types, library versions or worker counts: floating-point addition is not associative, so different reduction orders in all-reduce or atomic kernels change the last bits, and the differences grow over a run. Report variance across seeds.
:::

## Rapid-fire review

::: qa
**What should the initial loss of a $K$-class softmax classifier be?**

About $\log K$, from near-uniform predictions. Much higher means badly scaled initial logits; much lower suggests leakage.
:::

::: {.qa asked="ByteDance ads algorithm intern, Nowcoder 2020"}
**Why not initialize all weights to zero or to the same value?**

Symmetry: identical units get identical gradients and stay identical, so a layer acts like one unit. Random initialization breaks the tie; biases can start at zero.
:::

::: qa
**What is the effective learning rate of SGD with momentum $\mu$?**

About $\eta/(1-\mu)$ along directions where gradients are consistent, so $10\eta$ for $\mu=0.9$.
:::

::: qa
**What are Adam's defaults, and what do published LLM recipes often change?**

$\alpha=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. Several LLM papers lower $\beta_2$, e.g. GPT-3 and LLaMA use 0.95 (with weight decay 0.1), which shortens $v$'s window to about 20 steps so it tracks sudden changes in gradient scale.
:::

::: qa
**Why multiply the distillation soft loss by $T^2$?**

Soft-target gradients scale roughly as $1/T^2$, so multiplying by $T^2$ keeps their weight relative to the hard-label loss stable as $T$ changes (Hinton et al., 2015).
:::

::: qa
**Bytes per parameter for mixed-precision Adam training?**

16: weights 2, gradients 2, fp32 master copy 4, $m$ 4, $v$ 4. Activations come on top.
:::

::: qa
**Training FLOPs for $N$ parameters and $D$ tokens?**

About $6ND$: $2N$ per token forward, $4N$ backward.
:::

::: qa
**Clip gradients by norm or by value?**

Usually global norm: it rescales the whole gradient and keeps its direction. Value clipping changes the direction coordinate by coordinate.
:::

::: qa
**An EMA with decay 0.9999 averages roughly how many steps?**

About $10^4$, with a half-life near 6900 steps. In a short run that window still includes early, poor weights.
:::

::: {.qa asked="Qutoutiao algorithm engineer, Nowcoder 2018"}
**When does a ReLU "die"?**

A unit whose pre-activation is negative for every input, so it outputs zero and receives zero gradient forever, often after a large update. Use a lower learning rate, He init, normalization, or Leaky ReLU/GELU.
:::

::: qa
**What does activation checkpointing trade?**

Compute for memory: about one extra forward pass (roughly 33% more compute) for activation memory that can drop to $O(\sqrt n)$ in depth.
:::

## References

1. Kingma, Ba. Adam: A Method for Stochastic Optimization. 2014. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
2. Loshchilov, Hutter. Decoupled Weight Decay Regularization. 2017. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
3. Glorot, Bengio. Understanding the difficulty of training deep feedforward neural networks. AISTATS 2010. [PMLR](https://proceedings.mlr.press/v9/glorot10a.html)
4. He, Zhang, Ren, Sun. Delving Deep into Rectifiers. 2015. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
5. He, Zhang, Ren, Sun. Identity Mappings in Deep Residual Networks. 2016. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
6. Sutskever, Martens, Dahl, Hinton. On the importance of initialization and momentum in deep learning. ICML 2013. [PMLR](https://proceedings.mlr.press/v28/sutskever13.html)
7. Goyal et al. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. 2017. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
8. Krizhevsky. One weird trick for parallelizing convolutional neural networks. 2014. [arXiv:1404.5997](https://arxiv.org/abs/1404.5997)
9. Malladi, Lyu, Panigrahi, Arora. On the SDEs and Scaling Rules for Adaptive Gradient Algorithms. 2022. [arXiv:2205.10287](https://arxiv.org/abs/2205.10287)
10. McCandlish, Kaplan, Amodei et al. An Empirical Model of Large-Batch Training. 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
11. Liu et al. On the Variance of the Adaptive Learning Rate and Beyond (RAdam). 2019. [arXiv:1908.03265](https://arxiv.org/abs/1908.03265)
12. Kalra, Barkeshli. Why Warmup the Learning Rate? Underlying Mechanisms and Improvements. 2024. [arXiv:2406.09405](https://arxiv.org/abs/2406.09405)
13. Xiong et al. On Layer Normalization in the Transformer Architecture. 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
14. Loshchilov, Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. 2016. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
15. Hu et al. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. 2024. [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)
16. Hägele et al. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. 2024. [arXiv:2405.18392](https://arxiv.org/abs/2405.18392)
17. Pascanu, Mikolov, Bengio. On the difficulty of training Recurrent Neural Networks. 2012. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
18. Zhang et al. Why are Adaptive Methods Good for Attention Models? 2019. [arXiv:1912.03194](https://arxiv.org/abs/1912.03194)
19. Kunstner et al. Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be. 2023. [arXiv:2304.13960](https://arxiv.org/abs/2304.13960)
20. Zhang et al. Why Transformers Need Adam: A Hessian Perspective. 2024. [arXiv:2402.16788](https://arxiv.org/abs/2402.16788)
21. Kunstner et al. Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models. 2024. [arXiv:2402.19449](https://arxiv.org/abs/2402.19449)
22. Chen et al. Symbolic Discovery of Optimization Algorithms (Lion). 2023. [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)
23. Shazeer, Stern. Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. 2018. [arXiv:1804.04235](https://arxiv.org/abs/1804.04235)
24. Jordan et al. Muon: An optimizer for hidden layers in neural networks. Blog, 2024. [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
25. Liu et al. Muon is Scalable for LLM Training. 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
26. Micikevicius et al. Mixed Precision Training. 2017. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
27. Kalamkar et al. A Study of BFLOAT16 for Deep Learning Training. 2019. [arXiv:1905.12322](https://arxiv.org/abs/1905.12322)
28. Wortsman et al. Small-scale proxies for large-scale Transformer training instabilities. 2023. [arXiv:2309.14322](https://arxiv.org/abs/2309.14322)
29. Chowdhery et al. PaLM: Scaling Language Modeling with Pathways. 2022. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
30. Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
31. Song, Ermon. Improved Techniques for Training Score-Based Generative Models. 2020. [arXiv:2006.09011](https://arxiv.org/abs/2006.09011)
32. Karras et al. Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2). 2023. [arXiv:2312.02696](https://arxiv.org/abs/2312.02696)
33. Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 2014. [JMLR](https://www.jmlr.org/papers/v15/srivastava14a.html)
34. Li, Chen, Hu, Yang. Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift. 2018. [arXiv:1801.05134](https://arxiv.org/abs/1801.05134)
35. Müller, Kornblith, Hinton. When Does Label Smoothing Help? 2019. [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)
36. Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
37. Nakkiran et al. Deep Double Descent: Where Bigger Models and More Data Hurt. 2019. [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)
38. Nakkiran, Venkat, Kakade, Ma. Optimal Regularization Can Mitigate Double Descent. 2020. [arXiv:2003.01897](https://arxiv.org/abs/2003.01897)
39. Soudry et al. The Implicit Bias of Gradient Descent on Separable Data. 2017. [arXiv:1710.10345](https://arxiv.org/abs/1710.10345)
40. Barrett, Dherin. Implicit Gradient Regularization. 2020. [arXiv:2009.11162](https://arxiv.org/abs/2009.11162)
41. Smith, Dherin, Barrett, De. On the Origin of Implicit Regularization in Stochastic Gradient Descent. 2021. [arXiv:2101.12176](https://arxiv.org/abs/2101.12176)
42. Keskar et al. On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. 2016. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836)
43. Hoffer, Hubara, Soudry. Train longer, generalize better. 2017. [arXiv:1705.08741](https://arxiv.org/abs/1705.08741)
44. Rajbhandari, Rasley, Ruwase, He. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. 2019. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
45. Zhao et al. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. 2023. [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)
46. Shoeybi et al. Megatron-LM. 2019. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
47. Huang et al. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. 2018. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
48. Chen, Xu, Zhang, Guestrin. Training Deep Nets with Sublinear Memory Cost. 2016. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
49. Korthikanti et al. Reducing Activation Recomputation in Large Transformer Models. 2022. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
50. Kaplan et al. Scaling Laws for Neural Language Models. 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
51. Hoffmann et al. Training Compute-Optimal Large Language Models. 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
52. Smith. Cyclical Learning Rates for Training Neural Networks. 2015. [arXiv:1506.01186](https://arxiv.org/abs/1506.01186)
53. Karpathy. A Recipe for Training Neural Networks. Blog, 2019. [karpathy.github.io](https://karpathy.github.io/2019/04/25/recipe/)
54. Stanford CS231n notes, Neural Networks Parts 2 and 3. [cs231n.github.io/neural-networks-2](https://cs231n.github.io/neural-networks-2/), [neural-networks-3](https://cs231n.github.io/neural-networks-3/)
55. Dive into Deep Learning: Numerical Stability and Initialization; Adam. [d2l.ai init](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html), [d2l.ai Adam](https://d2l.ai/chapter_optimization/adam.html)
56. Goodfellow, Bengio, Courville. Deep Learning, Ch. 7 Regularization (Sec. 7.8 Early Stopping). MIT Press, 2016. [deeplearningbook.org](https://www.deeplearningbook.org/contents/regularization.html)
57. Brown et al. Language Models are Few-Shot Learners (GPT-3). 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
58. Touvron et al. LLaMA: Open and Efficient Foundation Language Models. 2023. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
