---
title: "Deep Learning Fundamentals and Training"
subtitle: "A guided, fully derived reading of the training fundamentals interviewers probe, from backpropagation, losses, initialization and normalization to optimizers, mixed precision, scaling and debugging, with numbers you can check by hand."
order: 0.6
updated: 2026-11-15
---

ML interviews keep returning to fundamentals that everyone uses but few re-derive: what backpropagation costs, why classification uses cross-entropy, where the 2 in He initialization comes from, what BatchNorm does at test time, what Adam's bias correction corrects, why bf16 replaced fp16, how much memory a 7B model needs. The opening question is short; the follow-ups ("derive it", "what if the batch size is 2") test whether you can rebuild the answer out loud. This page rebuilds each topic with every derivation step and small numeric examples you can check by hand. Every number was computed in Python (float64 unless a format is named), and every code block was run with PyTorch 2.11.

**Roadmap.** Sections 1 to 3 cover one training step (backpropagation, losses, activations); Sections 4 to 6 keep signals at a sensible scale through depth (initialization, vanishing gradients, normalization); then come regularization (7), optimizers (8), schedules and batch size (9), mixed precision (10), weight EMA (11), training at scale (12), debugging (13) and from-scratch code (14). Interview questions follow the section they test, and a rapid-fire review closes the page. Attention, positional encodings, the transformer block, the KV cache, FlashAttention and mixture-of-experts are on [transformers](transformers.html); convolution and recurrent layers in depth on [CNNs and RNNs](cnns-rnns.html); probability, KL divergence and optimization theory on [math for ML](math-for-ml.html).

## Notation

| Symbol | Meaning |
|---|---|
| $x$, $y$, $\hat y$ | an input, its target, the model's prediction |
| $X \in \mathbb{R}^{B \times n_{\text{in}}}$ | a batch of layer inputs, one example per row |
| $W$, $b$ | a layer's weight matrix and bias, as in $Y = XW + b$ |
| $n_{\text{in}}$, $n_{\text{out}}$ | fan-in and fan-out of a layer; for a $k\times k$ convolution, $n_{\text{in}} = k^2 C_{\text{in}}$ |
| $a_\ell$, $h_\ell = \phi(a_\ell)$ | pre-activation and activation of layer $\ell$; $\phi$ is the activation function |
| $L$ | number of layers or residual blocks |
| $\mathcal{L}$, $\ell_i$ | the scalar training loss; the loss of example $i$ |
| $\delta_v = \partial\mathcal{L}/\partial v$ | gradient of the loss with respect to $v$, a tensor with the shape of $v$ |
| $\theta$, $g_t$ | all parameters; the minibatch gradient at step $t$ |
| $\eta$ | learning rate (the Adam paper calls it $\alpha$) |
| $B$ | batch size |
| $\mu$ | momentum coefficient in PyTorch's SGD |
| $\beta$, $\beta_1$, $\beta_2$ | decay factors of exponential moving averages (EMAs) |
| $m_t$, $v_t$; $\hat m_t$, $\hat v_t$ | Adam's first and second moment estimates; their bias-corrected versions |
| $\lambda$ | weight decay or L2 coefficient |
| $\epsilon$ | small constant added to a denominator |
| $z \in \mathbb{R}^K$, $p = \mathrm{softmax}(z)$ | logits over $K$ classes; predicted class probabilities |
| $\sigma(z) = 1/(1+e^{-z})$ | the sigmoid; $\sigma^2$ always denotes a variance |
| $\gamma$, $\beta$ (Section 6) | learned scale and shift of a normalization layer |
| $m$ (Section 6) | number of values one normalization statistic averages over |
| $\Psi$ | parameter count in memory formulas (ZeRO's notation) |
| $N$, $D$, $C$ | parameters, training tokens and training FLOPs in scaling laws |
| $\odot$, $\mathbb{1}[\cdot]$ | elementwise product; indicator, 1 if the condition holds and 0 otherwise |

$C$ also counts channels ($C_{\text{in}}$, $C_{\text{out}}$), and $\beta$ is both an EMA decay and a normalization shift; context decides.

## 1. Backpropagation

### Computation graphs and the chain rule

A network plus its loss is a composition of simple operations (matrix products, additions, elementwise nonlinearities, reductions), which we draw as a directed acyclic graph of intermediate tensors. For any node $u$ write $\delta_u = \partial\mathcal{L}/\partial u$. If $u$ is read by the nodes $v_1, \dots, v_r$, it affects $\mathcal{L}$ only through them, so the multivariable chain rule gives

$$
\delta_u = \sum_{j=1}^{r} \Big(\frac{\partial v_j}{\partial u}\Big)^{\!\top} \delta_{v_j}.
$$

Processing nodes in reverse topological order (every child before its parents), starting from $\delta_{\mathcal{L}} = 1$, makes each $\delta_{v_j}$ complete when it is needed. No Jacobian is ever formed: each operation only needs a rule that maps the gradient of its output to the gradients of its inputs, a vector–Jacobian product (VJP). For ReLU, $h = \max(a, 0)$, the Jacobian is diagonal with entries $\mathbb{1}[a_i > 0]$, so the rule is $\delta_a = \delta_h \odot \mathbb{1}[a > 0]$. Backpropagation (Rumelhart, Hinton and Williams, 1986) is this procedure, reverse-mode automatic differentiation, applied to a loss. The sum over children is why a value used several times (a convolution weight at every position, a tensor read by both a residual branch and its skip path, tied embeddings) receives the sum of the gradients from all its uses.

### Why reverse mode, and what it costs

For $f: \mathbb{R}^n \to \mathbb{R}^m$ with Jacobian $J$, forward-mode differentiation carries a tangent along with the values and returns one Jacobian-vector product $Ju$ per pass; reverse mode carries adjoints backward and returns one vector-Jacobian product $w^\top J$ per pass. The full Jacobian costs about $n$ forward-mode passes or $m$ reverse-mode passes. Training has $m = 1$ and $n$ in the millions or billions, so one reverse pass gives every partial derivative, where forward mode would need one pass per parameter.

That reverse pass costs a small constant multiple of the forward pass whatever $n$ is (the "cheap gradient principle"; Baydin et al., 2018, survey the theory). For a linear layer $Y = XW$ with $X \in \mathbb{R}^{B \times n_{\text{in}}}$ and $W \in \mathbb{R}^{n_{\text{in}} \times n_{\text{out}}}$, the forward pass is one product of $B\, n_{\text{in}} n_{\text{out}}$ multiply-adds, and the backward pass needs $\delta_X = \delta_Y W^\top$ and $\delta_W = X^\top \delta_Y$, two products of the same size. So for networks dominated by matrix products, backward costs about two forward passes and a training step about three, which is where the $6N$ training FLOPs per token of Section 12 come from ($2N$ forward, $4N$ backward). The price is memory, discussed after the example.

### A fully worked example

Two inputs, two hidden ReLU units, one linear output, squared error with a factor $\tfrac12$:

$$
a = W_1 x + b_1, \qquad h = \max(a, 0), \qquad \hat y = w_2^\top h + b_2, \qquad \mathcal{L} = \tfrac12(\hat y - y)^2 ,
$$

with $x = (1, 2)^\top$, $W_1 = \begin{pmatrix} 1 & -1 \\ 0.5 & 0.25 \end{pmatrix}$, $b_1 = (0, 0)^\top$, $w_2 = (2, 3)^\top$, $b_2 = 0.5$ and target $y = 1$. Row $j$ of $W_1$ holds the incoming weights of hidden unit $j$.

**Forward pass**, keeping every intermediate value:

$$
\begin{aligned}
a &= \begin{pmatrix} 1\cdot 1 + (-1)\cdot 2 \\ 0.5\cdot 1 + 0.25\cdot 2 \end{pmatrix} = \begin{pmatrix} -1 \\ 1 \end{pmatrix}, \qquad h = \begin{pmatrix} 0 \\ 1 \end{pmatrix},\\
\hat y &= 2\cdot 0 + 3\cdot 1 + 0.5 = 3.5, \qquad \mathcal{L} = \tfrac12(3.5 - 1)^2 = 3.125 .
\end{aligned}
$$

**Backward pass**, in reverse order, with the local rule on the right:

$$
\begin{aligned}
\delta_{\hat y} &= \hat y - y = 2.5 && \text{derivative of } \tfrac12(\hat y - y)^2\\
\delta_{b_2} &= \delta_{\hat y} = 2.5 && \partial\hat y/\partial b_2 = 1\\
\delta_{w_2} &= \delta_{\hat y}\, h = (0,\ 2.5)^\top && \partial\hat y/\partial w_2 = h\\
\delta_{h} &= \delta_{\hat y}\, w_2 = (5,\ 7.5)^\top && \partial\hat y/\partial h = w_2\\
\delta_{a} &= \delta_h \odot \mathbb{1}[a > 0] = (0,\ 7.5)^\top && \text{ReLU passes gradient only where } a > 0\\
\delta_{W_1} &= \delta_a\, x^\top = \begin{pmatrix} 0 & 0 \\ 7.5 & 15 \end{pmatrix} && \partial a_j/\partial (W_1)_{jk} = x_k\\
\delta_{b_1} &= \delta_a = (0,\ 7.5)^\top && \partial a/\partial b_1 = I\\
\delta_{x} &= W_1^\top \delta_a = (3.75,\ 1.875)^\top && \partial a/\partial x = W_1
\end{aligned}
$$

Autograd returns the same numbers. Hidden unit 1 is off ($a_1 = -1$), so its weights and bias get zero gradient even though the gradient reaching it, $\delta_{h_1} = 5$, is not zero: the ReLU blocks it. The output weight $w_{2,1}$ gets zero gradient for a different reason: the value it multiplies, $h_1$, is zero. And $\delta_{W_1}$ is an outer product, output gradient times input, the pattern the matrix version below generalizes.

Now take one gradient step. With $\eta = 0.01$, unit 2's weights become $(0.425, 0.1)$ with bias $-0.075$, so $a_2 = 0.55$, $\hat y = 2.111$, and the loss falls from 3.125 to 0.617. With $\eta = 0.1$ the step overshoots: unit 2's weights become $(-0.25, -1.25)$ with bias $-0.75$, so $a_2 = -3.5$ and both hidden units are off. The loss is lower (0.281), but this example can no longer send any gradient to $W_1$. One large step has switched a unit off; if that happens for every input, the unit is "dead" (Section 3).

### The linear layer in matrix form

Let $Y = XW + b$ with $X \in \mathbb{R}^{B \times n_{\text{in}}}$, $W \in \mathbb{R}^{n_{\text{in}} \times n_{\text{out}}}$, and $b$ added to every row, and suppose $\delta_Y \in \mathbb{R}^{B\times n_{\text{out}}}$ is given. In indices $Y_{ij} = \sum_k X_{ik}W_{kj} + b_j$.

**Weight.** $\partial Y_{ij}/\partial W_{kl} = X_{ik}\,\mathbb{1}[j = l]$, since $W_{kl}$ appears only in column $l$ of $Y$, multiplied by $X_{ik}$:

$$
(\delta_W)_{kl} = \sum_{i,j} (\delta_Y)_{ij}\, X_{ik}\,\mathbb{1}[j=l] = \sum_i X_{ik}\, (\delta_Y)_{il} = (X^\top \delta_Y)_{kl}.
$$

**Input.** $\partial Y_{ij}/\partial X_{pk} = \mathbb{1}[i=p]\,W_{kj}$, since row $p$ of $X$ affects only row $p$ of $Y$:

$$
(\delta_X)_{pk} = \sum_{j} (\delta_Y)_{pj}\, W_{kj} = (\delta_Y W^\top)_{pk}.
$$

**Bias.** $\partial Y_{ij}/\partial b_l = \mathbb{1}[j = l]$ for every row, since $b$ is copied into each row, so $(\delta_b)_l = \sum_i (\delta_Y)_{il}$: the column sums of $\delta_Y$. The backward pass of a copy (a broadcast) is a sum.

**Shape check**, the fastest way to rebuild these under pressure: $\delta_W$ must be $n_{\text{in}} \times n_{\text{out}}$, and the only product of $X$ ($B\times n_{\text{in}}$) and $\delta_Y$ ($B\times n_{\text{out}}$) with that shape is $X^\top\delta_Y$; $\delta_X$ must be $B\times n_{\text{in}}$, and the only option is $\delta_Y W^\top$. With $B = 4$, $n_{\text{in}} = 3$, $n_{\text{out}} = 2$: $(3\times4)(4\times2) = 3\times2$ and $(4\times2)(2\times3) = 4\times3$. Note that $X^\top\delta_Y = \sum_i x_i\, \delta_{y_i}^\top$ is a sum of $B$ outer products, so a batch gradient of a linear layer has rank at most $B$. PyTorch's `nn.Linear` stores its weight as $(n_{\text{out}}, n_{\text{in}})$ and computes $XW^\top + b$, so there $\delta_W = \delta_Y^\top X$.

**Convolutions** apply the same small linear map at every output position, so their backward pass is the linear-layer rule applied at every position and summed over positions (the weights are shared); counts are in the question below, and the full derivation is on [CNNs and RNNs](cnns-rnns.html).

### Why the forward activations must be stored

Every rule above uses a forward value: $\delta_W = X^\top\delta_Y$ needs the layer's input, ReLU its mask, softmax $p$, BatchNorm its normalized values and standard deviation. The backward pass visits the graph in reverse, so all of it must be kept until used, and memory peaks when the backward pass begins. Activation memory grows with batch size, depth, width and resolution; parameter memory does not depend on the batch. One bf16 feature map of shape $(16, 128, 256, 256)$, a single stage of a super-resolution U-Net at $256^2$ with batch 16, is $16\cdot128\cdot256^2\cdot2$ bytes, exactly 256 MiB, and a U-Net keeps dozens of such tensors. Inside `torch.no_grad()` nothing is kept, which is why inference fits much larger batches. The remedies are smaller batches or crops, lower precision, and activation checkpointing (Section 12).

### What PyTorch autograd does

Each operation on a tensor with `requires_grad=True` records a node (`grad_fn`) that saves what its backward rule needs; the graph is rebuilt every forward pass. `loss.backward()` walks it in reverse, calling each VJP, and adds results into `.grad` of the leaves. Because `.grad` accumulates (correct for shared parameters, and free gradient accumulation), it must be cleared each step with `optimizer.zero_grad()`. `torch.no_grad()` stops recording (evaluation, EMA updates); `t.detach()` cuts a tensor from the graph so gradients stop there (fixed targets, logging, truncated backpropagation through time).

```python
import torch

w = torch.tensor(1.0, requires_grad=True)
for _ in range(3):
    (2 * w).backward()        # d(2w)/dw = 2, added into w.grad each time
print(w.grad)                 # tensor(6.): accumulated, not 2
w.grad = None                 # what optimizer.zero_grad() does by default

with torch.no_grad():
    y = 3 * w                 # no graph recorded
print(y.requires_grad, (3 * w).detach().requires_grad)   # False False
```

::: pitfall
`model.eval()` and `torch.no_grad()` do different things. `model.eval()` switches dropout and BatchNorm to inference behavior but still records the graph; `torch.no_grad()` stops recording but leaves dropout and BatchNorm in whatever mode they are in. Evaluation normally needs both.
:::

### Checking a gradient by finite differences

Expand $f(\theta \pm \varepsilon e_i) = f(\theta) \pm \varepsilon\,\partial_i f + \tfrac{\varepsilon^2}{2}\,\partial_i^2 f \pm \tfrac{\varepsilon^3}{6}\,\partial_i^3 f + O(\varepsilon^4)$. Subtracting the two expansions cancels the even-order terms, so

$$
\frac{f(\theta + \varepsilon e_i) - f(\theta - \varepsilon e_i)}{2\varepsilon} = \partial_i f + O(\varepsilon^2),
$$

while the one-sided difference keeps the $\tfrac{\varepsilon}{2}\partial_i^2 f$ term and has $O(\varepsilon)$ error. In the worked example, perturbing unit 2's weight on $x_1$ by $\varepsilon = 10^{-4}$ gives a central difference of $7.500000000005$ against the backpropagated 7.5, and a one-sided difference of $7.50045$; that error, $4.5\times10^{-4}$, is exactly $\tfrac{\varepsilon}{2}\partial^2\mathcal{L}$ with $\partial^2\mathcal{L} = (w_{2,2}x_1)^2 = 9$. Compare with the relative error $\lvert g_{\text{num}} - g_{\text{bp}}\rvert / (\lvert g_{\text{num}}\rvert + \lvert g_{\text{bp}}\rvert)$, in float64 (rounding contributes an error of order $u/\varepsilon$, with $u$ the unit roundoff), with dropout off, on a few random coordinates per tensor, and expect false alarms when a perturbation pushes a ReLU input across zero. In float64 a relative error near $10^{-7}$ or below indicates a correct gradient.

```python
import torch

def grad_check(f, params, n_checks=5, eps=1e-6):
    """f() returns the scalar loss; params are float64 tensors whose .grad is filled."""
    worst = 0.0
    for p in params:
        for _ in range(n_checks):
            i = tuple(torch.randint(s, ()).item() for s in p.shape)
            old = p.data[i].item()
            p.data[i] = old + eps; lp = f().item()
            p.data[i] = old - eps; lm = f().item()
            p.data[i] = old
            num, ana = (lp - lm) / (2 * eps), p.grad[i].item()
            worst = max(worst, abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
    return worst

torch.manual_seed(0)
X, y = torch.randn(8, 5, dtype=torch.float64), torch.randint(0, 3, (8,))
W1 = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
W2 = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
f = lambda: torch.nn.functional.cross_entropy(torch.tanh(X @ W1) @ W2, y)
f().backward()
print(grad_check(f, [W1, W2]))      # 2.6e-08 in this run: the gradients agree
```

::: key
**How to say it in an interview.** "Backprop is reverse-mode autodiff: every operation has a local rule that maps the gradient of its output to the gradients of its inputs, applied from the loss backward and summed over every use of a value. For a scalar loss one backward pass gives all parameter gradients for about twice the cost of the forward pass. The catch is memory: the backward rules need the forward activations, so we store them."
:::

::: qa
**Why is backpropagation done in reverse mode rather than forward mode, and what does the backward pass cost?**

Forward mode gives one Jacobian-vector product per pass, so $n$ parameters need about $n$ passes; reverse mode gives one vector-Jacobian product per pass, so a scalar loss needs one. Each operation's backward rule costs a small multiple of its forward cost: $Y = XW$ needs two products of the same size backward ($\delta_Y W^\top$ and $X^\top\delta_Y$), so backward is about twice forward and a training step about three forward passes. Reverse mode pays in memory, because it needs the stored forward activations.
:::

::: {.qa asked="Meta MLE 2026 (PracHub); AIGC interview repo"}
**Implement the forward and backward pass of a two-layer MLP in NumPy.**

Forward, caching intermediates: $z_1 = XW_1 + b_1$, $h = \max(z_1, 0)$, $z_2 = hW_2 + b_2$, $p = \mathrm{softmax}(z_2)$ row-wise (subtract each row's maximum first), $\mathcal{L} = -\frac1B\sum_i \log p_{i,y_i}$. Backward, using that every gradient has its variable's shape: $\delta_{z_2} = (P - Y_{\text{onehot}})/B$; $\delta_{W_2} = h^\top\delta_{z_2}$ and $\delta_{b_2}$ = column sums; $\delta_{z_1} = (\delta_{z_2}W_2^\top) \odot \mathbb{1}[z_1 > 0]$; $\delta_{W_1} = X^\top\delta_{z_1}$ and $\delta_{b_1}$ = column sums. The code in [Section 14](#pytorch-from-scratch) uses only operations with direct NumPy equivalents and matches autograd to about $10^{-17}$ in float64. Verify with a central-difference check in float64. Each pass costs $O(B(DH + HC))$ multiply-adds and the cache holds $O(B(H + C))$ numbers.
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**How does PyTorch autograd work? How do you take second derivatives, why call a module instead of its forward method, and what are ReLU's backward caveats?**

Autograd records a graph during the forward pass: each differentiable operation creates a node (`grad_fn`) that links to its inputs and saves what its backward rule needs. `loss.backward()` traverses it in reverse topological order, computes vector-Jacobian products, and accumulates into `.grad` of leaf tensors, hence `zero_grad()`. Non-leaf tensors keep no `.grad` unless `retain_grad()` is called, the graph is freed after backward unless `retain_graph=True`, and modifying a saved tensor in place raises an error (version counters). **Second derivatives:** `torch.autograd.grad(loss, w, create_graph=True)` records the backward computation too, so the gradient is differentiable; a Hessian-vector product is then `torch.autograd.grad(g @ v, w)`, used for gradient penalties (R1, WGAN-GP) and meta-learning. **Module call:** `module(x)` runs `__call__`, which runs forward pre-hooks and hooks; `module.forward(x)` silently skips them, breaking feature-extraction hooks and hook-based reparameterizations such as the older `spectral_norm`. **ReLU:** PyTorch uses derivative 0 at exactly 0, which matters only for exact zeros; in-place ReLU is safe after a linear or convolution layer but not after an operation that saved its output for backward (sigmoid, tanh, exp); and units negative on every input get no gradient and stay dead.
:::

::: {.qa asked="Google RI, ByteDance, Huawei, SenseTime (4+ reports)"}
**How many parameters and FLOPs does a convolutional layer have? What do 1x1 and depthwise-separable convolutions save?**

A $k\times k$ convolution from $C_{\text{in}}$ to $C_{\text{out}}$ channels has $k^2C_{\text{in}}C_{\text{out}}$ weights plus $C_{\text{out}}$ biases and costs $k^2C_{\text{in}}C_{\text{out}}H_{\text{out}}W_{\text{out}}$ multiply-adds (double for FLOPs). For $3\times3$, 64 to 128 channels at $56\times56$: $9\cdot64\cdot128 = 73{,}728$ weights plus 128 biases, 73,856 parameters (what `nn.Conv2d(64, 128, 3)` reports), and $73{,}728\cdot56^2 \approx 2.31\times10^8$ multiply-adds. A $1\times1$ convolution is a per-pixel linear layer over channels, the cheap way to change width. A depthwise-separable convolution (a per-channel $k\times k$ filter, then a $1\times1$) has $k^2C_{\text{in}} + C_{\text{in}}C_{\text{out}}$ weights, a ratio of $1/C_{\text{out}} + 1/k^2$ to the full convolution: $8{,}768$ weights and 0.119 in the example, so 8 to 9 times cheaper for $3\times3$. On GPUs the wall-clock saving is smaller, because depthwise convolutions are limited by memory bandwidth.
:::

## 2. Losses as likelihoods

### The recipe

Most standard losses come from one recipe: choose conditional distributions $p_\theta(y \mid x)$ whose parameters are the network's outputs, and minimize the average negative log-likelihood (NLL), $\mathcal{L}(\theta) = -\frac1n\sum_i \log p_\theta(y_i \mid x_i)$. Logs turn the product of probabilities, which would underflow, into a sum. The choice of distribution fixes the loss and also what the trained network estimates.

### Gaussian noise gives mean squared error

If $y = f_\theta(x) + \xi$ with $\xi \sim \mathcal{N}(0, \sigma^2 I)$ in $d$ dimensions, then

$$
-\log p_\theta(y\mid x) = \frac{\lVert y - f_\theta(x)\rVert^2}{2\sigma^2} + \frac d2\log(2\pi\sigma^2),
$$

so with $\sigma$ fixed, minimizing MSE is maximum likelihood. Minimizing $\mathbb{E}[(y - c)^2 \mid x]$ over a constant $c$ (derivative $-2\,\mathbb{E}[y - c \mid x] = 0$) gives $c = \mathbb{E}[y \mid x]$: an MSE-trained network predicts the conditional mean. That is why MSE-trained super-resolution networks are blurry: when several sharp images fit one low-resolution input, their average is blurry. **Linear regression** is the case $f(x) = Xw$: setting the gradient $-2X^\top(y - Xw)$ to zero gives the normal equations $X^\top X w = X^\top y$.

### Laplace noise gives L1, and Huber sits between

With Laplace noise, $p(y\mid x) \propto \exp(-\lvert y - f_\theta(x)\rvert/s)$, the NLL is the absolute error. The derivative of $\mathbb{E}[\lvert y - c\rvert \mid x]$ is $P(y < c\mid x) - P(y > c \mid x)$, zero at the conditional median. The gradient has constant magnitude, so a large residual pushes no harder than a small one: for residuals $(0.1, 0.1, 10)$, squared error gives $(0.01, 0.01, 100)$ and absolute error $(0.1, 0.1, 10)$, and the mean of $\{1, 2, 100\}$ is 34.3 while the median is 2. The Huber loss is $\tfrac12 r^2$ for $\lvert r\rvert \le \delta$ and $\delta(\lvert r\rvert - \tfrac12\delta)$ beyond, so its gradient is the residual clipped to $[-\delta, \delta]$. Restoration networks often train with L1: Zhao et al. (2017) found $\ell_1$-trained networks beat $\ell_2$-trained ones even on PSNR, and attributed it to $\ell_2$ training reaching poorer local minima.

### Categorical outputs give cross-entropy

For $K$ classes with logits $z$ and $p_k = e^{z_k}/\sum_j e^{z_j}$, the NLL of the true class $y$ is

$$
\mathcal{L} = -\log p_y = -z_y + \log\sum_{j} e^{z_j},
$$

the cross-entropy $-\sum_k y_k\log p_k$ with $y$ one-hot. Since $H(q, p) = H(q) + \mathrm{KL}(q\,\|\,p)$ and the data entropy $H(q)$ does not depend on $\theta$, minimizing cross-entropy, maximizing likelihood and minimizing the forward KL $\mathrm{KL}(q\,\|\,p_\theta)$ are the same problem. Forward KL is mass-covering; reverse KL is mode-seeking (derivations on [math for ML](math-for-ml.html)).

### The softmax cross-entropy gradient, step by step

Differentiate $\mathcal{L} = -z_y + \log\sum_j e^{z_j}$ with respect to $z_k$:

$$
\begin{aligned}
\frac{\partial}{\partial z_k}\big(-z_y\big) &= -\mathbb{1}[k = y] && \text{only the true-class logit appears}\\
\frac{\partial}{\partial z_k}\log\sum_j e^{z_j} &= \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k && (\log s)' = 1/s,\ \partial s/\partial z_k = e^{z_k}\\
\frac{\partial\mathcal{L}}{\partial z_k} &= p_k - y_k && \text{add the two lines.}
\end{aligned}
$$

Via the softmax Jacobian $\partial p_i/\partial z_k = p_i(\mathbb{1}[i=k] - p_k)$ the same result appears: $-\sum_i \frac{y_i}{p_i}p_i(\mathbb{1}[i=k] - p_k) = -y_k + p_k\sum_i y_i = p_k - y_k$. The gradient lies in $[-1, 1]$, sums to zero over $k$, and vanishes only when the prediction equals the target; with temperature $T$ it is $(p - y)/T$. Numbers: $z = (2, 1, 0)$ with the first class true gives $p = (0.665, 0.245, 0.090)$, $\mathcal{L} = 0.408$, and gradient $(-0.335, 0.245, 0.090)$.

### Why cross-entropy and not MSE for classification

In the binary case, $p = \sigma(z)$, label $y\in\{0,1\}$, and $\sigma' = \sigma(1-\sigma)$:

$$
\frac{\partial}{\partial z}\big[-y\log\sigma - (1-y)\log(1-\sigma)\big] = \sigma - y,
\qquad
\frac{\partial}{\partial z}(\sigma - y)^2 = 2(\sigma - y)\,\sigma(1 - \sigma).
$$

The MSE gradient carries the factor $\sigma(1-\sigma) \le \tfrac14$, which vanishes whenever the sigmoid saturates, including on the wrong side. At $z = -5$ with $y = 1$ (confidently wrong, $\sigma = 0.0067$), cross-entropy's gradient is $-0.993$ and MSE's is $-0.0132$, 75 times smaller, exactly when the model most needs to move. With $K$ classes, MSE on softmax outputs passes through $\mathrm{diag}(p) - pp^\top$, which vanishes as $p$ approaches any one-hot vector. Cross-entropy is also the maximum-likelihood loss, and for logistic regression it is convex in the weights (Hessian $\sum_i p_i(1-p_i)x_ix_i^\top \succeq 0$), while squared error through a sigmoid is not.

**Logistic loss.** Using $1 - \sigma(z) = \sigma(-z)$ and $-\log\sigma(z) = \log(1 + e^{-z})$, binary cross-entropy is $\log(1 + e^{-sz})$ with labels $s\in\{-1,+1\}$. `BCEWithLogitsLoss` computes it as $\max(z, 0) - zy + \log(1 + e^{-\lvert z\rvert})$, which never exponentiates a large positive number; a naive $\log\sigma(-200)$ in float32 is $-\infty$, while `logsigmoid` returns $-200$. Multi-label problems use an independent sigmoid per label.

### The log-sum-exp trick

For any $c$, $\log\sum_j e^{z_j} = c + \log\sum_j e^{z_j - c}$, because $e^c$ factors out of the sum. With $c = \max_j z_j$ the largest term is $e^0 = 1$, so the sum lies in $[1, K]$: no overflow, and the log never sees zero. The same identity gives $\mathrm{softmax}(z) = \mathrm{softmax}(z - c)$. Numbers: $\log(e^{1000} + e^{999}) = 1000 + \log(1 + e^{-1}) = 1000.313$, while the naive float32 softmax is $\infty/\infty$ = NaN. $e^z$ overflows above $z \approx 88.7$ in float32 and above $z \approx 11.09$ in fp16. This is why `cross_entropy` takes logits and why `log(softmax(z))` should be `log_softmax(z)`.

### Label smoothing and focal loss

**Label smoothing** (Szegedy et al., 2016, with $\varepsilon = 0.1$) trains against $(1 - \varepsilon)\,y + \varepsilon/K$. Cross-entropy is linear in the target, so the loss is $(1-\varepsilon)H(y, p) + \varepsilon H(u, p)$ with $u$ uniform and $H(u, p) = \mathrm{KL}(u\,\|\,p) + \log K$: a penalty on predictions far from uniform. With one-hot targets the loss keeps falling as the true logit runs to infinity; with smoothing the optimum is $p$ equal to the smoothed target. For $K = 10$, $\varepsilon = 0.1$, the targets are 0.91 and 0.01, the optimal gap between the true logit and each other logit is $\log 91 = 4.51$, and the smallest achievable loss is the target's entropy, 0.500. It usually improves calibration; Müller et al. (2019) show it also pulls each class's penultimate features into tight, equally spaced clusters, erasing which wrong classes are similar, so label-smoothed teachers distill worse.

**Focal loss** (Lin et al., 2017) is $-\alpha_t(1 - p_t)^\gamma\log p_t$ with $p_t$ the true-class probability. With $\gamma = 2$, an easy example at $p_t = 0.9$ is down-weighted by $0.01$ (loss 0.105 becomes 0.00105), one at $p_t = 0.5$ by 0.25, and a hard one at $p_t = 0.1$ only by 0.81. It was introduced for dense detection, where tens of thousands of easy background anchors per image dominate the summed loss; the paper's best setting was $\gamma = 2$, $\alpha = 0.25$.

### MLE, MAP and L2 as a Gaussian prior

MAP adds a log-prior: $\hat\theta_{\text{MAP}} = \arg\max_\theta\big[\sum_i \log p_\theta(y_i\mid x_i) + \log p(\theta)\big]$. A prior $\theta \sim \mathcal{N}(0, \tau^2 I)$ contributes $-\lVert\theta\rVert^2/(2\tau^2)$, an L2 penalty. For linear regression with noise variance $\sigma^2$, the negative log-posterior is $\lVert y - Xw\rVert^2/(2\sigma^2) + \lVert w\rVert^2/(2\tau^2)$; multiplying by $2\sigma^2$ gives ridge regression with $\lambda = \sigma^2/\tau^2$ and solution $(X^\top X + \lambda I)^{-1}X^\top y$. A Laplace prior gives L1. The likelihood term grows with $n$ while the prior is fixed, so MAP approaches MLE with more data.

**Class imbalance.** A cross-entropy model fits the training class frequencies $\pi_y$: it estimates $p_{\text{train}}(y\mid x) \propto p(x\mid y)\,\pi_y$. If deployment priors are $\pi'_y$ and $p(x\mid y)$ is unchanged, Bayes' rule gives $p'(y\mid x) \propto p_{\text{train}}(y\mid x)\,\pi'_y/\pi_y$, so add $\log\pi'_y - \log\pi_y$ to each logit (post-hoc logit adjustment, Menon et al., 2021; threshold moving in the binary case). The other remedies are in the question below.

::: key
**How to say it in an interview.** "A loss is a negative log-likelihood: Gaussian noise gives MSE and the conditional mean, Laplace gives L1 and the median, a categorical model gives cross-entropy. The softmax cross-entropy gradient with respect to the logits is $p - y$. MSE through a sigmoid or softmax multiplies that by a saturating factor, so a confidently wrong model barely learns. Adding a prior gives MAP, and a Gaussian prior is L2."
:::

::: qa
**What is the difference between MLE and MAP, and how does L2 regularization fit in?**

MLE maximizes the likelihood alone; MAP maximizes likelihood times prior and returns the posterior mode. A zero-mean Gaussian prior with variance $\tau^2$ adds $-\lVert\theta\rVert^2/(2\tau^2)$ in log space, exactly an L2 penalty ($\lambda = \sigma^2/\tau^2$ for linear regression, giving ridge); a Laplace prior gives L1. As data grows the likelihood dominates and MAP approaches MLE. MAP is still a point estimate: it does not average over the posterior, and the mode moves under reparameterization.
:::

::: qa
**Why is minimizing cross-entropy the same as minimizing KL divergence? Which KL?**

$H(q,p) = H(q) + \mathrm{KL}(q\,\|\,p)$, and with $q$ the data distribution $H(q)$ is constant in the parameters, so both have the same minimizers and gradients. It is the forward KL, data first, so maximum likelihood is forward-KL minimization, which is mass-covering. The constant offset is why cross-entropy bottoms out at the data entropy, not zero, when labels are noisy or soft.
:::

::: qa
**Derive the softmax cross-entropy gradient with respect to the logits. Why fuse log and softmax?**

$\mathcal{L} = -z_y + \log\sum_j e^{z_j}$; the first term contributes $-\mathbb{1}[k=y]$ and the second $e^{z_k}/\sum_j e^{z_j} = p_k$, so $\partial\mathcal{L}/\partial z_k = p_k - y_k$. Fusing lets the implementation compute $\log\sum_j e^{z_j}$ as $c + \log\sum_j e^{z_j - c}$ with $c = \max_j z_j$, avoiding overflow in $e^z$ and $\log 0$ when a probability underflows; a separate log of a softmax can give $-\infty$ or NaN, especially in fp16, where $e^{12}$ already overflows. The fused backward also returns $p - y$ directly.
:::

::: {.qa asked="ByteDance, iQIYI, aibee (5 interviews, 4 posts)"}
**Write softmax cross-entropy and its gradient. Why cross-entropy rather than MSE for classification, and what is focal loss?**

$\mathcal{L} = -\sum_k y_k\log p_k$ with $p = \mathrm{softmax}(z)$, and $\partial\mathcal{L}/\partial z = p - y$. MSE on softmax outputs passes through the softmax Jacobian $\mathrm{diag}(p) - pp^\top$, which vanishes as $p$ approaches a one-hot vector, even the wrong one; in the binary case MSE's gradient at $z = -5$, $y = 1$ is 75 times smaller than cross-entropy's. Cross-entropy is also the categorical maximum-likelihood loss and convex for a linear model. Multi-label problems use per-label sigmoids with binary cross-entropy. Focal loss, $-\alpha_t(1 - p_t)^\gamma\log p_t$, down-weights easy examples (with $\gamma = 2$, one at $p_t = 0.9$ counts 100 times less) so dense detectors focus on hard ones; $\gamma = 0$ recovers ($\alpha$-weighted) cross-entropy.
:::

::: qa
**What is the loss function for logistic regression, and why use log loss instead of mean squared error?**

The negative Bernoulli log-likelihood, $-y\log\sigma(z) - (1 - y)\log(1-\sigma(z))$ with $z = w^\top x$. Its logit gradient $\sigma(z) - y$ is large when the model is confidently wrong, whereas MSE's, $2(\sigma - y)\sigma(1-\sigma)$, vanishes when the sigmoid saturates on either side, so learning stalls when it should be fastest. Log loss is also convex in $w$ (Hessian $\sum_i p_i(1-p_i)x_ix_i^\top \succeq 0$); MSE through a sigmoid is not.
:::

::: {.qa asked="Microsoft AS, Baidu, Alibaba, ByteDance (4 reports)"}
**Compare L1 and L2, as losses and as regularizers. Where does Huber fit?**

As losses, L2 is the Gaussian likelihood, estimates the conditional mean, punishes outliers quadratically, and its gradient shrinks near zero; L1 is the Laplace likelihood, estimates the conditional median, is robust to outliers and has a constant-magnitude gradient. Restoration networks often train with L1, which tends to give sharper results and, in Zhao et al. (2017), even better PSNR. Huber is quadratic within $\delta$ and linear outside, so its gradient is the residual clipped to $[-\delta, \delta]$. As regularizers, L2 (a Gaussian prior, weight decay) shrinks weights in proportion to their size, while L1 (a Laplace prior) sets many exactly to zero, because its subgradient $\lambda\,\mathrm{sign}(\theta)$ does not shrink with the weight; geometrically, the L1 ball has corners on the axes.
:::

::: {.qa asked="Apple ML intern 2024, Meta MLE 2026"}
**Derive linear regression, and explain the statistical basis of the MSE loss.**

Minimize $\lVert y - Xw\rVert^2$: the gradient $-2X^\top(y - Xw) = 0$ gives $X^\top Xw = X^\top y$, so $w = (X^\top X)^{-1}X^\top y$ when $X$ has full column rank. In practice solve with QR or Cholesky, or use SGD at scale; ridge adds $\lambda I$ to $X^\top X$, which also fixes rank deficiency. If $y = Xw + \xi$ with $\xi \sim \mathcal{N}(0, \sigma^2 I)$, the log-likelihood is $-\lVert y - Xw\rVert^2/(2\sigma^2)$ plus a constant, so least squares is maximum likelihood under Gaussian noise and estimates $\mathbb{E}[y\mid x]$; a Gaussian prior on $w$ makes it ridge (MAP).
:::

::: qa
**What does label smoothing do, and when can it hurt?**

It replaces the one-hot target with $(1-\varepsilon)y + \varepsilon/K$, adding a term proportional to $\mathrm{KL}(u\,\|\,p)$, so the loss can no longer be lowered by pushing the correct logit to infinity (for $K = 10$, $\varepsilon = 0.1$ the optimal logit gap is $\log 91 \approx 4.5$). Predictions become less overconfident and usually better calibrated. It hurts where the relative probabilities of wrong classes matter: it squeezes each class's features into tight, equidistant clusters (Müller et al., 2019), so label-smoothed teachers are worse for distillation.
:::

::: {.qa asked="ByteDance, Huawei, SenseTime (Nowcoder 2019–22)"}
**How do you handle class imbalance?**

First decide what you optimize: cross-entropy targets accuracy under the training frequencies, so imbalance hurts when the metric or deployment weights rare classes more, or when rare classes have too few examples.

1. **Metrics**: with 1% positives, always predicting "negative" scores 99% accuracy, so use per-class precision and recall, balanced accuracy, macro-F1 and PR-AUC.
2. **Resampling**: oversample rare classes (with augmentation), undersample frequent ones, or build class-balanced batches; only on the training split, after splitting.
3. **Loss weighting**: weight class $c$ by about $1/n_c$, or by the inverse effective number $(1-\beta^{n_c})/(1-\beta)$ (Cui et al., 2019); focal loss down-weights easy, mostly majority, examples.
4. **Prior correction**: add $\log\pi'_y - \log\pi_y$ to the logits; for balanced error, post-hoc logit adjustment $\arg\max_y z_y - \tau\log\pi_y$ is consistent (Menon et al., 2021); in binary problems, move the threshold on validation data.
5. **Decoupled training**: learn features with ordinary sampling, then retrain only the classifier with class-balanced sampling (Kang et al., 2020).

Reweighting and resampling distort predicted probabilities, so recalibrate on data from the true distribution, and report per-class results.
:::

## 3. Activation functions

### Why a nonlinearity is needed at all

Two linear layers with nothing between them collapse into one:

$$
W_2(W_1x + b_1) + b_2 = (W_2W_1)\,x + (W_2b_1 + b_2),
$$

and by induction so does any depth of affine layers (with at most a rank limit from the narrowest layer). An elementwise nonlinearity between layers breaks the collapse; with it, one wide enough hidden layer can approximate any continuous function on a compact set (the universal approximation theorem, which says nothing about how many units are needed or how to find the weights).

### The standard activations

| Name | $\phi(x)$ | $\phi'(x)$ | Range | Notes |
|---|---|---|---|---|
| Sigmoid | $\sigma(x) = 1/(1+e^{-x})$ | $\sigma(x)(1 - \sigma(x))$ | $(0, 1)$ | derivative at most $1/4$, at $x = 0$ |
| Tanh | $\tanh x = 2\sigma(2x) - 1$ | $1 - \tanh^2 x$ | $(-1, 1)$ | zero-centred; derivative at most 1 |
| ReLU | $\max(x, 0)$ | $\mathbb{1}[x > 0]$ | $[0, \infty)$ | PyTorch uses derivative 0 at $x = 0$ |
| Leaky ReLU | $\max(x, ax)$, $0 < a < 1$ | $1$ if $x > 0$, else $a$ | $\mathbb{R}$ | PyTorch default $a = 0.01$; PReLU learns $a$ |
| GELU | $x\,\Phi(x)$, $\Phi$ the normal CDF | $\Phi(x) + x\,e^{-x^2/2}/\sqrt{2\pi}$ | $[-0.170, \infty)$ | minimum at $x \approx -0.752$ |
| SiLU (Swish) | $x\,\sigma(x)$ | $\sigma(x)\big(1 + x(1 - \sigma(x))\big)$ | $[-0.278, \infty)$ | minimum at $x \approx -1.278$ |

The GELU and SiLU derivatives follow from the product rule and were checked against autograd. GELU (Hendrycks and Gimpel, 2016) is often computed with the tanh approximation $\tfrac12 x\big(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715x^3))\big)$, within $4.7\times10^{-4}$ of the exact form; SiLU was proposed by Elfwing et al. (2017) and found by automated search as Swish (Ramachandran et al., 2017).

**Sigmoid and tanh** saturate: for large $\lvert x\rvert$ the derivative is near zero, and backpropagation multiplies by it at every layer; ten sigmoid layers scale a gradient by at most $0.25^{10} \approx 9.5\times10^{-7}$ before counting the weights. Sigmoid outputs are also always positive, so for a unit with positive inputs every entry of $\delta_w = \delta_a\,x$ has the sign of $\delta_a$, and all its weights move up or down together on each example, a zig-zag that tanh's zero-centred output avoids. Both survive as gates, where a bounded output is the point. **ReLU** (Nair and Hinton, 2010) has derivative exactly 1 when active, is cheap, and gives sparse activations; its weakness is the zero gradient for negative inputs. **GELU and SiLU** are smooth versions of ReLU with a shallow dip below zero; GELU is standard in BERT, GPT-2 and ViT, and SiLU in diffusion U-Nets (DDPM's code calls it swish; Stable Diffusion uses it). Their gains over ReLU are empirical.

### Dying ReLU

A unit with pre-activation $a = w^\top x + b$ is dead if $a < 0$ for every input in the data. Its output is then 0 and, by $\delta_a = \delta_h\odot\mathbb{1}[a > 0]$, the gradient to $w$ and $b$ is zero on every example, so gradient descent can never revive it. The worked example of Section 1 showed the mechanism: a step of $\eta = 0.1$ sent unit 2's pre-activation from $+1$ to $-3.5$. Causes are a learning rate that is too high, a large negative bias, and badly scaled initialization. Remedies: a lower learning rate with warmup; He initialization (Section 4); normalization before the activation, which re-centres pre-activations so roughly half are positive; or an activation with nonzero gradient for negative inputs (Leaky ReLU, PReLU, GELU, SiLU). To detect it, log the fraction of units that output zero on a whole batch.

### Gated linear units

A gated linear unit multiplies two projections, one through a gate (Dauphin et al., 2017):

$$
\mathrm{GLU}(x) = (xW + b)\odot\sigma(xV + c), \qquad \mathrm{SwiGLU}(x) = \mathrm{SiLU}(xW)\odot(xV).
$$

The feed-forward layer $\big(\mathrm{SiLU}(xW_1)\odot xW_3\big)W_2$ gave lower perplexity than ReLU or GELU layers at equal parameter count (Shazeer, 2020) and is used in PaLM and LLaMA. It has three matrices instead of two, so its hidden width is cut to $\tfrac23\cdot 4d$ to keep the parameter count: LLaMA-7B, with $d = 4096$, gets $10{,}923$ rounded up to a multiple of 256, $11{,}008$. The product of two functions of the input is a multiplicative interaction that a plain activation cannot express with the same number of units.

::: key
**How to say it in an interview.** "Without a nonlinearity, any stack of linear layers is one linear map. Sigmoid and tanh saturate, so their derivatives shrink gradients layer after layer; ReLU passes gradients unchanged when active but can die if a unit's input is negative for every example, usually after a big step. GELU and SiLU are smooth ReLU variants used in transformers and diffusion U-Nets, and gated units like SwiGLU multiply two projections."
:::

::: qa
**Why do neural networks need nonlinear activation functions?**

Because composing affine maps gives an affine map, $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$, so without nonlinearities a network of any depth computes only linear functions. An elementwise nonlinearity between layers lets it represent nonlinear functions; one wide enough hidden layer can approximate any continuous function on a compact set, and depth can represent some functions far more compactly.
:::

::: {.qa asked="Qualcomm (Glassdoor), Horizon (Nowcoder 2022)"}
**List common activation functions with their pros and cons. Why do many ReLU units die during training?**

Sigmoid and tanh are smooth and bounded but saturate, causing vanishing gradients (sigmoid's derivative is at most 1/4), and sigmoid is not zero-centred. ReLU is cheap and does not saturate for positive inputs, but gives zero gradient for negative ones. Leaky ReLU and PReLU keep a small negative slope. GELU and SiLU are smooth, slightly non-monotonic near zero, and standard in transformers and diffusion U-Nets; gated units (GLU, SwiGLU) are standard in LLM feed-forward layers. A unit dies when a large update or a strongly negative bias pushes its pre-activation below zero for all inputs; then $\delta_a = \delta_h\odot\mathbb{1}[a > 0]$ is zero on every example and it cannot recover. Lower learning rates with warmup, He initialization, normalization before the activation, or leaky and smooth variants prevent it.
:::

::: {.qa asked="Tencent multimodal, Nowcoder 2026"}
**What gating mechanisms appear in neural networks, and what does a gate buy you?**

A gate is a learned factor, usually a sigmoid in $(0, 1)$, sometimes a softmax or a scalar like $\tanh(\alpha)$, that multiplies another signal and decides how much passes. Examples: LSTM and GRU gates (the cell path $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$ has Jacobian $\mathrm{diag}(f_t)$, so gradients survive many steps when $f_t\approx1$); highway networks, $y = H(x)T(x) + x(1 - T(x))$, which ResNets simplify to a fixed identity; GLU and SwiGLU feed-forward layers; Squeeze-and-Excitation channel gates; mixture-of-experts routers (see [transformers](transformers.html)); and conditioning gates such as DiT's zero-initialized adaLN-Zero gate and Flamingo's $y + \tanh(\alpha)\,\mathrm{xattn}(y, \text{visual})$ with $\alpha = 0$ at the start. A gate buys input-dependent multiplicative routing, near-identity gradient paths when open, safe insertion of new modules into a pretrained network, and conditional computation. Caveats: a saturated sigmoid passes almost no gradient; never zero both a gate and its branch so that neither gets gradient (Flamingo's $\alpha$ gets $\tanh'(0) = 1$ times the nonzero branch output); MoE routers can collapse without a balancing loss.
:::

## 4. Initialization

If every weight in a layer starts equal, every hidden unit computes the same function, gets the same gradient and takes the same update, so the units stay identical and the layer acts like one unit. Random initialization breaks this symmetry (biases can start at zero). The scale of the random weights comes from tracking variances.

### The forward variance recursion

Take $y_i = \sum_{j=1}^{n_{\text{in}}} W_{ij}x_j$ with $W_{ij}$ i.i.d., mean 0, variance $v$, independent of the identically distributed inputs $x_j$:

$$
\begin{aligned}
\mathbb{E}[y_i] &= \textstyle\sum_j \mathbb{E}[W_{ij}]\,\mathbb{E}[x_j] = 0 && \text{independence, and } \mathbb{E}[W_{ij}] = 0\\
\mathrm{Var}(y_i) &= \textstyle\sum_j \mathrm{Var}(W_{ij}x_j) && \text{cross terms vanish: } \mathbb{E}[W_{ij}W_{ik}] = 0 \text{ for } j\neq k\\
&= \textstyle\sum_j \big(\mathbb{E}[W_{ij}^2]\,\mathbb{E}[x_j^2] - (\mathbb{E}[W_{ij}]\,\mathbb{E}[x_j])^2\big) && \text{variance of a product of independent factors}\\
&= n_{\text{in}}\, v\, \mathbb{E}[x^2] && \mathbb{E}[W^2] = v,\ \mathbb{E}[W] = 0 .
\end{aligned}
$$

The key detail is $\mathbb{E}[x^2]$, not $\mathrm{Var}(x)$; they differ after a ReLU, whose outputs are not zero-mean. **Backward**, the gradient flows through $W^\top$, $\delta_{x_j} = \sum_{i=1}^{n_{\text{out}}} W_{ij}\,\delta_{y_i}$, so under a similar independence approximation $\mathrm{Var}(\delta_{x_j}) = n_{\text{out}}\, v\,\mathrm{Var}(\delta_y)$. Constant forward scale asks for $n_{\text{in}}v = 1$ (with zero-mean inputs) and constant gradient scale for $n_{\text{out}}v = 1$.

### Xavier and He initialization

**Xavier (Glorot and Bengio, 2010)** assumes activations roughly linear near zero, like tanh, so $\mathbb{E}[x^2] \approx \mathrm{Var}(y_{\text{prev}})$, and compromises between the two conditions:

$$
v = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad\text{or uniform } U[-r, r] \text{ with } r = \sqrt{6/(n_{\text{in}} + n_{\text{out}})},
$$

since $U[-r, r]$ has variance $r^2/3$. For a $512\times512$ layer: standard deviation $\sqrt{1/512} = 0.0442$, or $r = 0.0765$.

**He (He et al., 2015)** redoes the analysis for ReLU. Let the previous pre-activation $y_{\text{prev}}$ be symmetric about zero (true at initialization with symmetric weights and zero biases) and $x = \max(y_{\text{prev}}, 0)$. Then

$$
\mathbb{E}[x^2] = \mathbb{E}\big[y_{\text{prev}}^2\,\mathbb{1}[y_{\text{prev}} > 0]\big] = \tfrac12\,\mathbb{E}[y_{\text{prev}}^2] = \tfrac12\,\mathrm{Var}(y_{\text{prev}}),
$$

by symmetry (the two halves contribute equally to $\mathbb{E}[y^2]$) and mean zero. Substituting,

$$
\mathrm{Var}(y) = \tfrac12\, n_{\text{in}}\,v\,\mathrm{Var}(y_{\text{prev}}), \qquad \mathrm{Var}(y_L) = \mathrm{Var}(y_1)\prod_{\ell=2}^{L}\tfrac12\,n_{\text{in},\ell}\,v_\ell ,
$$

and setting each factor to 1 gives $v = 2/n_{\text{in}}$, standard deviation $\sqrt{2/n_{\text{in}}}$ ($n_{\text{in}} = k^2C_{\text{in}}$ for a convolution; 0.0625 for a 512-wide layer). The 2 compensates for the half of the signal ReLU zeroes. He et al. note that either the forward condition or the backward one ($v = 2/n_{\text{out}}$) suffices, because with one satisfied the other's product over layers telescopes to a ratio of the first and last widths, which does not grow with depth. For Leaky ReLU with slope $a$ the condition is $\tfrac12(1 + a^2)\,n_{\text{in}}v = 1$. With Xavier's $v = 1/n$ in a ReLU network each factor is $\tfrac12$, so the standard deviation shrinks by $\sqrt 2$ per layer; He et al. report a 30-layer ReLU network that converged with their initialization and stalled completely with Xavier's (at 22 layers both converged).

### A 50-layer numerical demo

```python
import math, torch

torch.manual_seed(0)
n, depth = 512, 50
x0 = torch.randn(1000, n)                         # 1000 inputs with unit variance
inits = {"N(0, 0.01^2)": 0.01,
         "Xavier": math.sqrt(2 / (n + n)),        # 0.0442
         "He": math.sqrt(2 / n),                  # 0.0625
         "N(0, 0.1^2)": 0.1}
for name, std in inits.items():
    torch.manual_seed(1)
    x, out = x0, []
    for layer in range(1, depth + 1):
        a = x @ (torch.randn(n, n) * std).T      # pre-activation
        x = torch.relu(a)
        if layer in (1, 2, 5, 10, 25, 50):
            out.append(f"{a.std().item():.3g}")
    print(f"{name:13s}", out)
```

Standard deviation of the pre-activations through 50 ReLU layers of width 512:

| Weight distribution | Layer 1 | Layer 2 | Layer 5 | Layer 10 | Layer 25 | Layer 50 |
|---|---|---|---|---|---|---|
| $\mathcal{N}(0, 0.01^2)$ | 0.227 | 0.0366 | $1.37\times10^{-4}$ | $1.66\times10^{-8}$ | $9.65\times10^{-21}$ | $1.77\times10^{-40}$ |
| Xavier, std 0.0442 | 1.00 | 0.715 | 0.231 | 0.0471 | $1.31\times10^{-4}$ | $3.28\times10^{-8}$ |
| He, std 0.0625 | 1.42 | 1.43 | 1.31 | 1.51 | 0.761 | 1.10 |
| $\mathcal{N}(0, 0.1^2)$ | 2.27 | 3.66 | 13.7 | 166 | $9.65\times10^{4}$ | $1.77\times10^{10}$ |

The recursion predicts each row: the per-layer factor on the standard deviation is $\sqrt{\tfrac12\cdot512\cdot v}$, which is 0.16 for std 0.01, $\sqrt{1/2}$ for Xavier (predicting $0.707^{49} \approx 4.2\times10^{-8}$ at layer 50 against $3.3\times10^{-8}$ measured), exactly 1 for He (a constant $\sqrt2 = 1.41$, since the raw input has $\mathbb{E}[x^2] = 1$), and 1.6 for std 0.1. He's row drifts between 0.76 and 1.51 because each finite-width layer multiplies the scale by a random factor near 1 and the fluctuations compound; over ten seeds the layer-50 value ranged from 0.79 to 1.55 with median 1.40. The $1.77\times10^{-40}$ is below float32's smallest normal number and would be exactly zero in fp16 or bf16.

::: pitfall
PyTorch's default initialization for `nn.Linear` and `nn.Conv2d` is not He: it draws from $U(-1/\sqrt{n_{\text{in}}}, 1/\sqrt{n_{\text{in}}})$, variance $1/(3n_{\text{in}})$, one sixth of He's (measured std 0.0255 for $n_{\text{in}} = 512$). Normalization layers usually remove the scale, but a deep plain ReLU network of default layers shrinks its signal at every layer; call `nn.init.kaiming_normal_` explicitly there.
:::

### Initializing residual networks

For $h_{\ell+1} = h_\ell + F_\ell(h_\ell)$ with a variance-preserving branch whose output is uncorrelated with $h_\ell$,

$$
\mathrm{Var}(h_{\ell+1}) = \mathrm{Var}(h_\ell) + \mathrm{Var}(F_\ell(h_\ell)) = 2\,\mathrm{Var}(h_\ell),
$$

so without normalization the standard deviation grows by $\sqrt2$ per block, $(\sqrt2)^{50} \approx 3.4\times10^7$ after 50 blocks. Scaling each branch's last layer by $c$ makes each block multiply the variance by $1 + c^2$; with $c = 1/\sqrt L$ the total is $(1 + 1/L)^L \le e$, and with $c = 0$ each block is the identity at initialization. A check with 50 blocks of width 512:

```python
def residual_stack(branch_scale, depth=50):
    torch.manual_seed(2)
    h = x0
    for _ in range(depth):
        W1 = torch.randn(n, n) * math.sqrt(2 / n)                  # He, before the ReLU
        W2 = torch.randn(n, n) * math.sqrt(1 / n) * branch_scale   # last layer of the branch
        h = h + torch.relu(h @ W1.T) @ W2.T
    return h.std().item()

print([f"{residual_stack(s):.3g}" for s in (1.0, 1 / math.sqrt(50), 0.0)])
# ['3.97e+07', '1.64', '1']
```

The values match the predictions ($(\sqrt2)^{50} = 3.4\times10^7$, and $(1 + 1/50)^{25} = 1.64$ for the standard deviation). Practice follows the same idea: Goyal et al. (2017) zero-initialize the scale $\gamma$ of the last BatchNorm in each ResNet block; Fixup (Zhang, Dauphin and Ma, 2019) rescales branch weights by $L^{-1/(2m-2)}$ ($m$ layers per branch), zeroes the last layer and trains deep ResNets without normalization; GPT-2 scales the layers that write into the residual stream by $1/\sqrt N$ for $N$ residual layers; and DiT's adaLN-Zero, ControlNet's zero convolutions and LoRA's $B = 0$ start a new path at exactly zero, so the model starts as the identity or as the pretrained network.

::: pitfall
Zero-initialize only the last layer of a branch. With $W_2 = 0$ and $W_1$ random, $\delta_{W_2} = h^\top\delta$ is nonzero because $h \ne 0$, so $W_2$ moves at once and $W_1$ gets gradient right after. If $W_1$ is zero too, $h = \mathrm{ReLU}(0) = 0$, every gradient in the branch is zero, and it never trains.
:::

::: key
**How to say it in an interview.** "Initialize so each layer preserves the variance of activations and gradients. For $y = Wx$ the variance is $n_{\text{in}}\mathrm{Var}(w)\mathbb{E}[x^2]$. Xavier sets $\mathrm{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$ for tanh-like units; ReLU zeroes half the second moment, so He uses $2/n_{\text{in}}$. In a 50-layer ReLU net Xavier shrinks the signal to $10^{-8}$ while He keeps it near 1. In residual nets you also scale down or zero each branch's last layer, or variance doubles every block."
:::

::: {.qa asked="Tencent multimodal 2026, Horizon 2022 (Nowcoder)"}
**How should weights be initialized? Derive Kaiming initialization.**

Keep activation and gradient variance roughly constant across layers at the start. For $y = Wx$ with fan-in $n$ and zero-mean i.i.d. weights independent of $x$, $\mathrm{Var}(y_i) = n\,\mathrm{Var}(w)\,\mathbb{E}[x^2]$. If $x$ is the ReLU of a zero-mean symmetric $u$, $\mathbb{E}[x^2] = \tfrac12\mathrm{Var}(u)$, so $\mathrm{Var}(w) = 2/n$ keeps the variance constant (He; $n = k^2C_{\text{in}}$ for convolutions). Xavier uses $2/(n_{\text{in}} + n_{\text{out}})$ for linear or tanh units. Through 50 ReLU layers of width 512, He keeps the standard deviation near 1.4 while Xavier lets it fall to $3\times10^{-8}$. All-zero initialization fails by symmetry, but zeroing selected layers is deliberate: the last layer of a residual branch, LoRA's $B$ and ControlNet's zero convolutions make a new path start as a no-op.
:::

::: qa
**Derive He initialization. Why $2/n_{\text{in}}$ instead of Xavier's value?**

$\mathrm{Var}(y_i) = n_{\text{in}}v\,\mathbb{E}[x^2]$, and the key step is $\mathbb{E}[x^2]$ rather than $\mathrm{Var}(x)$, because ReLU outputs are not zero-mean. A symmetric zero-mean pre-activation keeps half its second moment through ReLU, so preserving variance needs $\tfrac12 n_{\text{in}}v = 1$; the 2 compensates for the zeroed half. Xavier assumes units roughly linear near zero; with ReLU the signal then shrinks by $\sqrt2$ per layer, $2^{-24.5}\approx 4\times10^{-8}$ after 50 layers. Both hold only at initialization under independence assumptions.
:::

::: qa
**Why zero-initialize the last layer of a residual branch, and what goes wrong if you zero the whole branch?**

Without normalization, a variance-preserving branch adds its own variance at every block, so activations grow by $\sqrt2$ per block, about $4\times10^7$ after 50 blocks in simulation. Zeroing each branch's last layer (or last BatchNorm scale) makes every block the identity at initialization, so the network starts shallow, gradients reach every layer through the skips, and depth is used as the branches learn. It still trains because the last layer's gradient, $h^\top\delta$, is nonzero. Zero the whole branch and its hidden activations are zero, so every gradient inside it is zero and it never moves.
:::

## 5. Vanishing and exploding gradients

### A product of Jacobians

For $h_\ell = \phi(a_\ell)$ with $a_\ell = W_\ell h_{\ell-1}$, one layer's Jacobian is $J_\ell = \mathrm{diag}\big(\phi'(a_\ell)\big)W_\ell$ (the elementwise $\phi$, then the linear map), and backpropagation multiplies their transposes:

$$
\delta_{h_0} = J_1^\top J_2^\top\cdots J_L^\top\,\delta_{h_L}, \qquad \prod_\ell s_{\min}(J_\ell)\,\lVert\delta_{h_L}\rVert \;\le\; \lVert\delta_{h_0}\rVert \;\le\; \prod_\ell s_{\max}(J_\ell)\,\lVert\delta_{h_L}\rVert,
$$

with $s_{\min}, s_{\max}$ the extreme singular values. A typical per-layer factor $\rho$ compounds to about $\rho^L$: $0.9^{50} = 0.0052$ and $1.1^{50} = 117$. The causes are saturating activations ($\sigma' \le \tfrac14$), weight scales that do not preserve variance, and depth, or sequence length in a recurrent network.

**Recurrent networks** are the classic case: $h_t = \tanh(W h_{t-1} + Ux_t)$ gives $\partial h_T/\partial h_k = \prod_{t=k+1}^{T}\mathrm{diag}(\tanh'(a_t))\,W$, roughly $W^{T-k}$, so components along directions $W$ stretches explode and those it shrinks vanish. Pascanu et al. (2013) show that if $\lvert\tanh'\rvert\le\gamma$ and the largest singular value of $W$ is below $1/\gamma$, long-range gradient terms vanish, and exceeding $1/\gamma$ is necessary for explosion. The LSTM's additive cell $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$ has direct-path Jacobian $\mathrm{diag}(f_t)$, so with the forget gate near 1 the gradient passes many steps nearly unchanged (initializing the forget-gate bias near 1 helps; Jozefowicz et al., 2015). Backpropagation through time is derived on [CNNs and RNNs](cnns-rnns.html).

### Why residual connections help

For $h_{\ell+1} = h_\ell + F_\ell(h_\ell)$, the identity contributes $I$ and the branch its Jacobian, so $\partial h_{\ell+1}/\partial h_\ell = I + \partial F_\ell/\partial h_\ell$. Chaining to the top and expanding the product,

$$
\frac{\partial h_L}{\partial h_\ell} = \prod_{k=\ell}^{L-1}\Big(I + \frac{\partial F_k}{\partial h_k}\Big) = I + \sum_{k}\frac{\partial F_k}{\partial h_k} + \sum_{k < k'}\frac{\partial F_{k'}}{\partial h_{k'}}\frac{\partial F_k}{\partial h_k} + \cdots,
$$

so $\delta_{h_\ell} = \delta_{h_L} + (\text{terms through the branches})$: the top gradient reaches every layer unattenuated, with no product that must stay near 1. This is clean only when the skip path is a pure identity, which is why pre-activation ResNets (He et al., 2016) and pre-norm transformers put normalization inside the branch. It does not rule out explosion, since the branch terms can add up (the variance growth of Section 4), so residuals are paired with normalization or down-scaled branches. He et al. (2016) also note their deep plain networks with BatchNorm did not have vanishing gradients yet trained worse than shallower ones (the degradation problem); the identity path makes "do nothing" the easy default.

**The fixes, and why each works:** ReLU-family activations (derivative 1 when active); variance-preserving initialization (He, Xavier, orthogonal for RNNs); normalization layers, which rescale activations at every layer so weight-scale drift does not compound; residual connections (the identity term); gating in LSTMs and GRUs; gradient-norm clipping, which bounds a step during a spike and keeps its direction (Section 9); and smaller learning rates with warmup, truncated backpropagation through time, and bf16 instead of fp16. Clipping treats only explosions. To diagnose, log per-layer gradient norms: vanishing shows as tiny norms in early layers and a loss that plateaus from the start; exploding as gradient-norm spikes, then loss spikes, then NaNs.

::: {.qa asked="ByteDance algorithm intern, Nowcoder 2020"}
**What causes vanishing and exploding gradients, and how do you handle them?**

Backpropagation multiplies one Jacobian per layer, or per time step in an RNN: $\delta_{h_0} = J_1^\top\cdots J_L^\top\delta_{h_L}$ with $J_\ell = \mathrm{diag}(\phi'(a_\ell))W_\ell$, so a typical per-layer factor $\rho$ compounds to about $\rho^L$ ($0.9^{50}\approx0.005$, $1.1^{50}\approx117$). Vanishing comes from saturating activations (sigmoid's derivative is at most 1/4), weights that are too small, or recurrent weights with small singular values; exploding from the opposite. Fixes: ReLU-family activations; He, Xavier or orthogonal initialization; normalization; residual connections, whose Jacobian $I + \partial F/\partial h$ gives an identity path; LSTM/GRU gating; global-norm clipping for explosions; smaller learning rates with warmup, truncated BPTT, bf16 instead of fp16. Clipping treats only explosions. Diagnose with per-layer gradient norms.
:::

::: qa
**Why do residual connections make very deep networks trainable?**

Each block computes $h + F(h)$, so its Jacobian is $I + \partial F/\partial h$, and the product over blocks expands into the identity plus branch terms: the top-layer gradient reaches every layer directly, whatever the branches do. At initialization, a block whose branch outputs nearly zero is close to the identity, so a deep residual network starts near a shallow function and uses depth as the branches learn. The caveat is that activation variance accumulates across blocks (it doubles per block for variance-preserving branches), which normalization and down-scaled or zero-initialized last branch layers control.
:::

::: {.qa asked="Ctrip, bigo, NVIDIA (Nowcoder, PracHub)"}
**Why do residual connections work? Compare ResNet with VGG and DenseNet.**

A residual block computes $y = x + F(x)$, so the identity is easy to represent and extra depth need not hurt optimization, and the Jacobian $I + \partial F/\partial x$ gives gradients a direct path through depth. ResNet fixed the degradation problem, in which deeper plain networks had higher training error even with BatchNorm. VGG stacks plain $3\times3$ convolutions, puts most of its parameters in fully connected layers, and becomes hard to train beyond roughly 20 layers, while ResNets train with 100 or more. DenseNet concatenates all earlier feature maps within a block, which encourages feature reuse with fewer parameters but costs memory for the growing concatenations. Architectures in detail: [CNNs and RNNs](cnns-rnns.html).
:::

## 6. Normalization layers

Every normalization layer standardizes a group of values with their own mean and variance, then applies a learned per-feature scale and shift; the layers differ only in which values form a group. [Vision backbones](vision-backbones.html#normalization) summarizes them and covers conditioning variants (AdaIN, adaLN); this section derives the details.

### BatchNorm: training mode, running statistics, evaluation mode

For one channel, let $x_1, \dots, x_m$ be all its values in the batch: $m = B$ for a $(B, C)$ input and $m = BHW$ for a $(B, C, H, W)$ map, since every position of a channel comes from the same filter. Ioffe and Szegedy (2015) compute

$$
\mu = \frac1m\sum_{i=1}^m x_i,\qquad \sigma^2 = \frac1m\sum_{i=1}^m (x_i - \mu)^2,\qquad \hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}},\qquad y_i = \gamma\,\hat x_i + \beta ,
$$

with the biased variance and $\epsilon = 10^{-5}$ in PyTorch. Gradients flow through $\mu$ and $\sigma^2$, so each example's output depends on its batch-mates. Example: $x = (1, 2, 3, 6)$ gives $\mu = 3$, $\sigma^2 = 3.5$, $\hat x = (-1.069, -0.535, 0, 1.604)$.

**Running statistics** are EMAs kept for test time. PyTorch, with `momentum` 0.1, updates $r_\mu \leftarrow 0.9\,r_\mu + 0.1\,\mu$ and $r_{\sigma^2} \leftarrow 0.9\,r_{\sigma^2} + 0.1\cdot\frac{m}{m-1}\sigma^2$ (storing the unbiased variance). From $r_\mu = 0$, $r_{\sigma^2} = 1$, the batch above gives $r_\mu = 0.3$ and $r_{\sigma^2} = 0.9 + 0.1\cdot(14/3) = 1.367$, as `nn.BatchNorm1d` does. PyTorch's "momentum" weights the new value, the opposite of Keras, whose 0.99 weights the old one. **Evaluation mode** uses the running averages, $y = \gamma\,(x - r_\mu)/\sqrt{r_{\sigma^2} + \epsilon} + \beta$: deterministic, independent of the batch, and a fixed per-channel affine map that can be folded into the preceding convolution.

**Why $\gamma$ and $\beta$.** Forcing zero mean and unit variance would restrict what the network can represent (a sigmoid after BatchNorm would be stuck near its linear region); with $\gamma = \sqrt{\sigma^2 + \epsilon}$ and $\beta = \mu$ the layer undoes the normalization exactly, so nothing is lost and each channel's scale and offset become learnable. A layer over $C$ channels has $2C$ learned parameters and $2C$ buffers (plus a scalar counter in PyTorch); `BatchNorm2d(64)` has 128 parameters. A bias in the convolution before it is redundant (the mean subtraction removes it and $\beta$ replaces it), hence `bias=False`.

**Small or unusual batches.** With few values per channel the batch statistics are noisy and differ from the running averages used at test time; with one value per channel PyTorch raises "Expected more than 1 value per channel when training". With batch size 2 per GPU, GroupNorm's ResNet-50 ImageNet error is 10.6% lower than BatchNorm's (Wu and He, 2018). BatchNorm also misbehaves with non-i.i.d. batches, train/test shift, variable-length sequences, contrastive learning (statistics can leak which samples are positives), and across GPUs unless `SyncBatchNorm` is used. Gradient accumulation does not help, since statistics are per micro-batch.

### The BatchNorm backward pass

Given $g_i = \partial\mathcal{L}/\partial y_i$ for one channel and $s = \sqrt{\sigma^2 + \epsilon}$, the parameters are immediate from $y_i = \gamma\hat x_i + \beta$: $\partial\mathcal{L}/\partial\beta = \sum_i g_i$, $\partial\mathcal{L}/\partial\gamma = \sum_i g_i\hat x_i$, and $\partial\mathcal{L}/\partial\hat x_i = \gamma g_i$. The input $x_i$ reaches $\mathcal{L}$ through three paths: directly through $\hat x_i$, and through the shared $\mu$ and $\sigma^2$.

**Through $\sigma^2$.** $\partial\hat x_j/\partial\sigma^2 = -\tfrac12(x_j - \mu)s^{-3}$, so $\dfrac{\partial\mathcal{L}}{\partial\sigma^2} = -\dfrac{\gamma}{2s^3}\sum_j g_j(x_j - \mu)$.

**Through $\mu$.** $\partial\hat x_j/\partial\mu = -1/s$, and the path through $\sigma^2$ vanishes because $\partial\sigma^2/\partial\mu = -\tfrac2m\sum_j(x_j - \mu) = 0$, so $\dfrac{\partial\mathcal{L}}{\partial\mu} = -\dfrac{\gamma}{s}\sum_j g_j$.

**Assemble** with $\partial\hat x_i/\partial x_i = 1/s$, $\partial\sigma^2/\partial x_i = 2(x_i - \mu)/m$ and $\partial\mu/\partial x_i = 1/m$:

$$
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial x_i} &= \frac{\gamma g_i}{s} + \frac{\partial\mathcal{L}}{\partial\sigma^2}\cdot\frac{2(x_i - \mu)}{m} + \frac{\partial\mathcal{L}}{\partial\mu}\cdot\frac1m\\
&= \frac{\gamma g_i}{s} - \frac{\gamma\,(x_i - \mu)}{m\,s^3}\sum_j g_j(x_j - \mu) - \frac{\gamma}{m\,s}\sum_j g_j && \text{substitute}\\
&= \frac{\gamma}{m\,s}\Big(m\,g_i - \sum_j g_j - \hat x_i\sum_j g_j\hat x_j\Big) && x_j - \mu = s\,\hat x_j .
\end{aligned}
$$

The last line is a projection: remove the upstream gradient's mean and its component along $\hat x$, then scale by $\gamma/s$. Shifting or scaling the whole batch does not change the output, so the gradient has no component in those directions; $\sum_i \partial\mathcal{L}/\partial x_i = 0$ exactly. The formula matched autograd to $2\times10^{-16}$ in float64. In evaluation mode the statistics are constants and the backward pass is just $\gamma g_i/s$.

### LayerNorm, InstanceNorm, GroupNorm and RMSNorm

None of these uses the batch, so they behave identically in training and evaluation and work with a batch of one. For a $D$-dimensional token,

$$
\mathrm{LN}(x) = \gamma\odot\frac{x - \mu(x)}{\sqrt{\sigma^2(x) + \epsilon}} + \beta,\qquad
\mathrm{RMSNorm}(x) = \gamma\odot\frac{x}{\sqrt{\tfrac1D\sum_j x_j^2 + \epsilon}},
$$

with statistics over that token's $D$ features (Ba et al., 2016; Zhang and Sennrich, 2019). RMSNorm drops the mean subtraction and the shift, keeping invariance to rescaling the input; its paper reports 7% to 64% lower running time than LayerNorm.

| Method | Statistics over, for $(B, C, H, W)$ | For $(B, N, D)$ | Learned affine (PyTorch default) | Uses the batch | Typical use |
|---|---|---|---|---|---|
| BatchNorm | $(B, H, W)$, per channel | $(B, N)$, per feature (rare) | $\gamma, \beta \in \mathbb{R}^C$ | yes | CNN classifiers, large batches |
| LayerNorm | $(C, H, W)$ per sample, or $C$ per pixel (ConvNeXt) | $D$, per token | $\gamma, \beta$ with the normalized shape | no | transformers, ConvNeXt |
| InstanceNorm | $(H, W)$, per sample and channel | rarely used | none by default | no | style transfer |
| GroupNorm | $(C/G, H, W)$, per sample and group | rarely used | $\gamma, \beta \in \mathbb{R}^C$ | no | small-batch detection, diffusion U-Nets |
| RMSNorm | rarely used | $D$, per token, no mean | $\gamma \in \mathbb{R}^D$ only | no | LLaMA, many recent LLMs and DiTs |

GroupNorm splits the channels into $G$ groups (32 in the paper and most diffusion U-Nets) and normalizes each over its channels and all positions; $G = 1$ normalizes over $(C, H, W)$ like LayerNorm and $G = C$ is InstanceNorm (both checked numerically without the affine part). InstanceNorm removes each image's per-channel statistics, which carry contrast and style, hence its use in style transfer. PyTorch parameter counts: `LayerNorm(768)` 1536, `RMSNorm(768)` 768, `GroupNorm(32, 64)` 128, `InstanceNorm2d(64)` 0, and `LayerNorm([64, 32, 32])` 131,072, because its affine parameters have the full normalized shape.

::: pitfall
"LayerNorm" means two things in vision code. `nn.LayerNorm([C, H, W])` normalizes each sample over all channels and positions and learns a $\gamma$ and $\beta$ for each of the $CHW$ positions, tying it to one resolution. ConvNeXt and ViTs normalize over the $C$ channels at each position separately, with per-channel parameters (`nn.LayerNorm(C)` on a channels-last tensor). `GroupNorm(1, C)` normalizes like the first but has per-channel parameters.
:::

### Why normalization helps

BatchNorm was motivated as reducing "internal covariate shift", the drift of each layer's input distribution as earlier layers update, and matched its Inception baseline in 14 times fewer training steps. That explanation has not held up well. Santurkar et al. (2018) injected random noise with a changing mean and variance after each BatchNorm layer, deliberately creating distribution shift, and the network still trained about as well; by their measure BatchNorm does not necessarily reduce the shift. What it does, in their experiments and in analysis of simplified settings, is make the loss and its gradients smoother along the training path, so larger learning rates are stable. The full mechanism is still debated; scale invariance is the cleanest established effect.

**Scale invariance and the effective learning rate.** If weights $w$ feed a normalization, scaling $w$ by $c > 0$ is undone, so $f(cw) = f(w)$. Differentiating:

$$
\begin{aligned}
\frac{d}{dc}f(cw)\Big|_{c=1} = w^\top\nabla f(w) &= 0 && \text{the gradient is orthogonal to } w\\
c\,\nabla f(cw) = \nabla f(w)\ \Rightarrow\ \nabla f(cw) &= \nabla f(w)/c && \text{differentiate } f(cw) = f(w) \text{ in } w .
\end{aligned}
$$

A gradient step is orthogonal to $w$, so $\lVert w - \eta\nabla f\rVert^2 = \lVert w\rVert^2 + \eta^2\lVert\nabla f\rVert^2$: without weight decay the norm only grows. Only the direction $w/\lVert w\rVert$ matters, and the angle moved is about $\eta\lVert\nabla f(w)\rVert/\lVert w\rVert = \eta\lVert\nabla f(\hat w)\rVert/\lVert w\rVert^2$, using the second identity with $c = \lVert w\rVert$. So the effective learning rate is $\eta/\lVert w\rVert^2$: doubling the weight norm divides it by 4. Large steps grow $\lVert w\rVert$ and shrink later steps, which stabilizes training, and weight decay on such weights cannot change the function but keeps $\lVert w\rVert$ small and the effective learning rate up (van Laarhoven, 2017; Zhang et al., 2019). Normalization also makes curvature more uniform across layers, reducing sensitivity to initialization, and BatchNorm's batch noise is a mild regularizer.

::: key
**How to say it in an interview.** "BatchNorm standardizes each channel with the batch mean and variance in training, keeps running averages for test time, and applies a learned scale and shift so it can represent the identity. LayerNorm, GroupNorm, InstanceNorm and RMSNorm differ only in the axes they average over and never use the batch. The covariate-shift story is weak: Santurkar et al. showed BatchNorm still works with injected shift and argued it smooths the loss. A cleaner fact is scale invariance: normalized weights have effective learning rate $\eta/\lVert w\rVert^2$, which is how weight decay interacts with normalization."
:::

::: {.qa asked="ByteDance, Alibaba, Ctrip, bigo (5 interviews, 2019–22)"}
**How does BatchNorm differ between training and inference? Why the learned scale and shift, and how many parameters does it have?**

In training, each channel is normalized with the mean and biased variance over $(N, H, W)$ of the current batch, gradients flow through those statistics, and running averages are updated (momentum 0.1 in PyTorch, storing the unbiased variance). At inference the running averages are used, so outputs are deterministic and batch-independent, and the layer can be folded into the preceding convolution. $\gamma$ and $\beta$ let the network undo or reshape the normalization ($\gamma = \sigma$, $\beta = \mu$ is the identity). Parameters: $2C$ learned plus $2C$ running-statistic buffers that are not trained; `BatchNorm2d(64)` has 128. Statistics are per channel because channels carry different features while positions share one filter. Batch noise regularizes mildly, but BatchNorm degrades with small or non-i.i.d. batches and train-test shift; the classic bugs are forgetting `model.eval()` and unconverged running statistics.
:::

::: {.qa asked="Google intern, Glassdoor (role unclear)"}
**Compare BatchNorm, LayerNorm, InstanceNorm and GroupNorm.**

They differ only in the axes averaged. For $(N, C, H, W)$: BatchNorm over $(N, H, W)$ per channel; LayerNorm over $(C, H, W)$ per sample (over features per token in transformers); InstanceNorm over $(H, W)$ per sample and channel; GroupNorm over $(H, W)$ and a group of $C/G$ channels per sample, with $G = 1$ LayerNorm-like and $G = C$ InstanceNorm. BatchNorm couples samples, needs large i.i.d. batches and behaves differently at test time; the others work with batch size one and behave the same in training and inference. InstanceNorm removes per-image contrast and style, hence style transfer (AdaIN injects a style through the affine). Diffusion U-Nets use GroupNorm because per-GPU batches are small and mix very different noise levels; transformers and DiTs use LayerNorm or RMSNorm, with timestep-dependent affine parameters in DiT's adaLN.
:::

::: qa
**Derive the gradient of BatchNorm with respect to its input.**

With $s = \sqrt{\sigma^2 + \epsilon}$ and $g_i = \partial\mathcal{L}/\partial y_i$: $\partial\mathcal{L}/\partial\hat x_i = \gamma g_i$; $\partial\mathcal{L}/\partial\sigma^2 = -\tfrac{\gamma}{2s^3}\sum_j g_j(x_j - \mu)$; $\partial\mathcal{L}/\partial\mu = -\tfrac{\gamma}{s}\sum_j g_j$, the path through $\sigma^2$ vanishing because deviations sum to zero. With $\partial\mu/\partial x_i = 1/m$ and $\partial\sigma^2/\partial x_i = 2(x_i - \mu)/m$,

$$
\frac{\partial\mathcal{L}}{\partial x_i} = \frac{\gamma}{m s}\Big(m g_i - \sum_j g_j - \hat x_i\sum_j g_j\hat x_j\Big),
$$

which removes the mean of the upstream gradient and its component along $\hat x$, because shifting or scaling the batch leaves the output unchanged. Also $\partial\mathcal{L}/\partial\gamma = \sum_i g_i\hat x_i$ and $\partial\mathcal{L}/\partial\beta = \sum_i g_i$.
:::

::: qa
**Why does normalization help optimization? Is reducing internal covariate shift the explanation?**

The original paper argued so, but Santurkar et al. (2018) injected time-varying noise after each BatchNorm layer, deliberately creating shift, and training was about as good; they argue BatchNorm instead makes the loss and gradients smoother, allowing larger learning rates. Two further effects are well established: scale invariance, which makes the effective learning rate of normalized weights $\eta/\lVert w\rVert^2$ (self-stabilizing, and the reason weight decay acts as a learning-rate control there), and better conditioning, which reduces sensitivity to initialization. I would present smoothness and scale invariance as the supported explanations and say the full story is debated.
:::

## 7. Regularization and generalization

Regularization means anything that lowers test error without aiming to lower training error: penalties, noise injected in training, augmentation, limits on training time, and averaging.

### Weight decay

Under plain SGD an L2 penalty $\tfrac\lambda2\lVert\theta\rVert^2$ adds $\lambda\theta$ to the gradient, so $\theta \leftarrow (1 - \eta\lambda)\,\theta - \eta\nabla\mathcal{L}$: each step shrinks the weights by $1 - \eta\lambda$ first, hence "weight decay". It is MAP with a Gaussian prior (Section 2). Typical values are about $10^{-4}$ for SGD on ResNets and 0.01 to 0.1 for AdamW on transformers, and many codebases exclude biases and normalization parameters. With adaptive optimizers L2 and decay differ (Section 8), and on weights followed by normalization, decay mostly raises the effective learning rate (Section 6).

### Dropout

Dropout (Srivastava et al., 2014) zeroes units at random in training. In the "inverted" form used by modern libraries, with drop probability $p$, $\tilde h_i = m_i\,h_i/(1-p)$ with $m_i \sim \mathrm{Bernoulli}(1-p)$.

- **Expectation**: $\mathbb{E}[\tilde h_i] = (1-p)h_i/(1-p) = h_i$, so **evaluation is the identity**, with no mask and no rescaling. With $p = 0.5$, kept units are doubled in training. The original paper left training unscaled and multiplied weights by the keep probability at test time; the two agree in expectation.
- **Variance**: $\mathrm{Var}(\tilde h_i) = h_i^2\,p/(1-p)$, equal to $h_i^2$ at $p = 0.5$, against 0 at test time. Means match but variances do not, which is why dropout before BatchNorm shifts the statistics BN collected (Li et al., 2019).
- **Backward**: for a fixed mask the layer is linear, so $\delta_h = m\odot\delta_{\tilde h}/(1-p)$ with the same stored mask; dropped units get no gradient that step.
- **Why it regularizes**: each step trains a random weight-sharing subnetwork, so units cannot rely on specific partners. Test-time scaling approximates averaging the $2^n$ subnetworks, exactly for a single linear layer by linearity of expectation and approximately once nonlinearities are stacked. Typical rates: 0.5 in AlexNet-era fully connected layers, 0.1 in transformers, often 0 in modern CNNs; DDPM's CIFAR-10 model used 0.1.

::: pitfall
The original paper's $p$ is the probability of keeping a unit (0.5 for hidden units, 0.8 for inputs); PyTorch's `p` is the probability of dropping one. Leaving the model in `train()` mode at test time keeps dropout on, unless you intend Monte Carlo dropout, which averages several stochastic passes to estimate uncertainty (Gal and Ghahramani, 2016).
:::

### Data augmentation and early stopping

Augmentation trains on transformed copies that keep the label, teaching invariances and enlarging the data: crops, flips and color jitter, policies such as RandAugment, and label mixing (mixup trains on $\lambda x_i + (1-\lambda)x_j$ with target $\lambda y_i + (1-\lambda)y_j$; CutMix pastes patches with area-proportional labels). Super-resolution uses aligned random crops of the low- and high-resolution pair, the eight flips and 90-degree rotations, and synthetic blur, noise and compression pipelines (see [restoration and super-resolution](restoration-super-resolution.html)). An augmentation must not change the target: a flipped "6" can become a "9", and a color change to a super-resolution input must also be applied to its target.

**Early stopping** keeps the checkpoint with the best validation loss. Why it regularizes is clearest for a quadratic loss $\tfrac12(\theta - \theta^*)^\top H(\theta - \theta^*)$. In the eigenbasis of $H$ (eigenvalues $\lambda_i$), gradient descent from $\theta_0 = 0$ obeys $\theta_{t+1,i} - \theta^*_i = (1 - \eta\lambda_i)(\theta_{t,i} - \theta^*_i)$, so

$$
\theta_{t,i} = \big(1 - (1 - \eta\lambda_i)^t\big)\,\theta^*_i ,
$$

while an L2 penalty $\tfrac\alpha2\lVert\theta\rVert^2$ has minimizer $\theta_i = \frac{\lambda_i}{\lambda_i + \alpha}\theta^*_i$ (set $\lambda_i(\theta_i - \theta^*_i) + \alpha\theta_i = 0$). For $\eta\lambda_i \ll 1$, $(1 - \eta\lambda_i)^t \approx e^{-\eta\lambda_i t}$, and if also $\lambda_i \ll \alpha$ both factors are about $\lambda_i/\alpha$ with $\alpha = 1/(\eta t)$. With $\eta = 0.01$, $t = 100$ ($\alpha = 1$): for $\lambda_i = 0.01$ the factors are 0.00995 and 0.00990, for $\lambda_i = 0.1$ 0.0952 and 0.0909, for $\lambda_i = 10$ 0.99997 and 0.909. High-curvature directions, which the data pins down, are fit early; flat directions, where noise lives, late. So stopping early acts like L2 of strength about $1/(\eta t)$ (Goodfellow et al., 2016, Sec. 7.8); for deep networks this is an analogy.

### The bias–variance decomposition

Let $y = f(x) + \xi$ with $\mathbb{E}[\xi] = 0$, $\mathrm{Var}(\xi) = \sigma^2$, let $\hat f_{\mathcal{D}}$ be trained on a random dataset independent of the test noise, and let $\bar f(x) = \mathbb{E}_{\mathcal{D}}[\hat f_{\mathcal{D}}(x)]$. At a fixed $x$, write $y - \hat f_{\mathcal{D}} = \xi + (f - \bar f) + (\bar f - \hat f_{\mathcal{D}})$, square, and take expectations. The cross terms vanish: those with $\xi$ because it has mean zero and is independent of $\mathcal{D}$, and $2(f - \bar f)\,\mathbb{E}_{\mathcal{D}}[\bar f - \hat f_{\mathcal{D}}] = 0$ by the definition of $\bar f$. So

$$
\mathbb{E}\big[(y - \hat f_{\mathcal{D}}(x))^2\big] = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{\big(f(x) - \bar f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\big[(\hat f_{\mathcal{D}}(x) - \bar f(x))^2\big]}_{\text{variance}} .
$$

A biased estimator can win. Estimate a mean $\mu = 1$ from $n = 10$ samples of variance 1 with $c\,\bar x$: bias $(c-1)\mu$, variance $c^2/n$. With $c = 1$ the expected squared error is 0.1; with $c = 0.9$ it is $0.01 + 0.081 = 0.091$; the best, $c = 1/1.1$, gives 0.0909. Shrinking trades a little bias for a larger cut in variance, which is what regularizers do.

### Diagnosing from learning curves

| What the curves show | Likely meaning | What to try first |
|---|---|---|
| Training and validation both high and close | underfitting, or an optimization problem | train longer, add capacity, less regularization, check the learning rate |
| Training falls, validation rises or plateaus | overfitting | more data or augmentation, more regularization, early stopping, a smaller model |
| Validation below training early | normal when dropout or augmentation is active only in training | nothing |
| Training loss rising or erratic | optimization | lower the learning rate, clip, inspect the data |
| Large gap from the very start | train/validation mismatch or leakage | check splits, preprocessing and `eval()` behavior |

Leakage, shifted splits and mismatched preprocessing all mimic overfitting, so rule them out first. In generative models, overfitting shows up as memorization, checked by nearest-neighbor search of samples against the training set in a feature space.

### Double descent and implicit regularization

Belkin et al. (2019) and, for deep networks, Nakkiran et al. (2020) show test error peaking near the **interpolation threshold** (the smallest model that fits the training set) and falling again as capacity grows; the same shape appears against epochs, label noise sharpens the peak, and tuned ridge regularization can remove it in linear models. At the threshold essentially one model interpolates and it must contort to fit the noise; with more capacity the training algorithm picks a smooth interpolant. That choice is implicit regularization: gradient descent from zero on underdetermined least squares finds the minimum-norm solution, logistic regression on separable data converges in direction to the max-margin classifier (Soudry et al., 2018), and finite-step gradient descent and SGD approximately add penalties on the gradient norm that favor flat regions, the SGD one growing with $\eta/B$ (Barrett and Dherin, 2021; Smith et al., 2021). This helps explain the large-batch generalization gap (Keskar et al., 2017), much of which closes with retuned learning rates and longer training (Hoffer et al., 2017).

::: key
**How to say it in an interview.** "For squared error, expected test error is noise plus squared bias plus variance; regularization trades a little bias for less variance. I diagnose with training and validation curves, but first rule out leakage and train/eval mismatches. For large networks capacity is not a monotone knob because of double descent, and the optimizer itself regularizes by choosing among interpolating solutions."
:::

::: qa
**What is the difference between L1 and L2 regularization?**

L2 adds $\lambda\lVert\theta\rVert_2^2$ (a Gaussian prior); its gradient $2\lambda\theta$ shrinks weights in proportion to size, so they get small but rarely zero. L1 adds $\lambda\lVert\theta\rVert_1$ (a Laplace prior); its subgradient $\lambda\,\mathrm{sign}(\theta)$ pulls equally regardless of size, driving small weights exactly to zero. Loss contours tend to touch the L1 ball at its corners, on the axes. L2 spreads weight across correlated features; L1 picks among them, which can be unstable. Deep networks default to decoupled weight decay.
:::

::: qa
**How does dropout differ between training and inference, and why the $1/(1-p)$ factor?**

In training each unit is zeroed with probability $p$; at inference nothing is dropped. Without correction the next layer would see inputs $1/(1-p)$ times larger at test time than on average in training, so inverted dropout divides kept units by $1-p$ in training ($\mathbb{E}[\tilde h] = h$) and inference is the identity. Expectations match but variances do not, which can shift the statistics a later BatchNorm collected. Forgetting `model.eval()` leaves dropout on at test time.
:::

::: {.qa asked="ByteDance CV 2023, Horizon 2022 (Nowcoder)"}
**How do dropout and pooling layers behave in the forward and backward passes, and how does dropout change between training and inference?**

None has parameters; their backward passes only route the upstream gradient. **Inverted dropout**: forward $y = m\odot x/(1-p)$, $m_i\sim\mathrm{Bernoulli}(1-p)$; backward $\delta_x = m\odot\delta_y/(1-p)$ with the same stored mask; inference is the identity. **Max pooling**: forward $y_j = \max_{i\in R_j}x_i$, storing each window's argmax $i^*(j)$; backward, the winner receives the whole $\delta_{y_j}$ and the rest of the window zero, $\delta_{x_i} = \sum_{j:\,i^*(j)=i}\delta_{y_j}$. **Average pooling**: forward $y_j = \frac{1}{\lvert R_j\rvert}\sum_{i\in R_j}x_i$; backward, each input in the window gets $\delta_{y_j}/\lvert R_j\rvert$ (global average pooling gives each position $1/(HW)$). With overlapping windows an input's gradients from all its windows add up. Pooling is the same in training and inference.
:::

::: qa
**Derive the bias–variance decomposition. Does "bigger model, more variance" hold for deep nets?**

Write $y - \hat f = \xi + (f - \bar f) + (\bar f - \hat f)$ with $\bar f = \mathbb{E}_{\mathcal{D}}\hat f$, square and take expectations; the cross terms vanish because the noise has mean zero and is independent of the training set, and $\mathbb{E}_{\mathcal{D}}[\bar f - \hat f] = 0$. That leaves noise, squared bias and variance. The decomposition is exact for squared error, but the classical U-curve is not guaranteed: past the interpolation threshold test error often falls again as models grow (double descent), because training selects a low-complexity interpolant. So variance is not monotone in parameter count.
:::

::: {.qa asked="Amazon AS, Qualcomm, NVIDIA (3 reports)"}
**Explain the bias-variance trade-off and how you diagnose and fix overfitting.**

Expected squared error is $(\mathbb{E}\hat f - f)^2 + \mathrm{Var}(\hat f) + \sigma^2$: bias from a model too simple or constrained, variance from sensitivity to the training sample, and irreducible noise. Regularizers accept some bias for a larger cut in variance (shrinking a sample mean by 0.9 lowers its error from 0.1 to 0.091 in the example above). Overfitting shows as falling training loss with rising validation loss. Fixes: more data or augmentation, weight decay, dropout, early stopping, a smaller model, ensembling or a weight EMA. Double descent means model size is not monotone. In generative models, check memorization by nearest-neighbor search of samples against the training set.
:::

::: qa
**How do you diagnose overfitting versus underfitting, and what do you try first?**

Plot training and validation loss together. Overfitting: training keeps falling while validation rises or plateaus. Underfitting: both high and close. First rule out leakage, split shift and preprocessing mismatches, which mimic a gap. For overfitting try more or augmented data, stronger regularization and early stopping, then a smaller model; for underfitting train longer, add capacity, reduce regularization, and check that the model can overfit a single batch.
:::

::: qa
**What is double descent, and how does it square with the bias–variance trade-off?**

Test error against capacity (or epochs) follows the classical U up to the interpolation threshold, peaks where the model can barely fit the training data, then falls as capacity grows. At the threshold essentially one solution interpolates and must contort to fit noise, so variance spikes; with more capacity, training selects a smooth, small-norm interpolant. The decomposition still holds; monotone variance in parameter count does not. Label noise sharpens the peak, tuned regularization can remove it, and the epoch-wise version means the first validation minimum is not always the best stopping point.
:::

::: {.qa asked="Alibaba, Nowcoder 2022"}
**What is the difference between empirical and structural risk minimization, why does early stopping work, and what does information theory say about generalization?**

**ERM** minimizes training loss $\hat R_n(h)$ over a fixed class $\mathcal{H}$, which works when $\hat R_n$ is uniformly close to the true risk over $\mathcal{H}$, so $\mathcal{H}$ must be small relative to $n$. **SRM** (Vapnik) takes nested classes $\mathcal{H}_1\subset\mathcal{H}_2\subset\cdots$ and minimizes training loss plus a capacity penalty, such as the VC bound $R(h) \le \hat R_n(h) + O\big(\sqrt{(d\log(n/d) + \log(1/\delta))/n}\big)$ for VC dimension $d$; regularized ERM and tuning model size on validation data are SRM in practice. For deep networks such bounds are vacuous, since the networks fit random labels (Zhang et al., 2017). **Early stopping**: weights reachable in $t$ steps form a growing nested family, and for a quadratic loss stopping at $t$ approximates L2 of strength $1/(\eta t)$, fitting high-curvature directions first. **Information theory**: Xu and Raginsky (2017) bound the expected generalization gap by $\sqrt{2\sigma^2 I(S;W)/n}$ for $\sigma$-sub-Gaussian losses, with $I(S;W)$ the mutual information between training sample and learned weights: an algorithm generalizes when its output reveals little about its sample (the bound suits noisy algorithms, since $I$ is typically infinite for deterministic ones). The information-bottleneck account (Shwartz-Ziv and Tishby, 2017) is disputed (Saxe et al., 2018).
:::

## 8. Optimizers

### SGD and minibatch noise

With $\mathcal{L} = \frac1n\sum_i\ell_i$, minibatch SGD steps $\theta \leftarrow \theta - \eta g$ with

$$
g = \frac1B\sum_{i\in\mathcal{B}}\nabla\ell_i(\theta), \qquad \mathbb{E}[g] = \nabla\mathcal{L}(\theta), \qquad \mathrm{Cov}(g) = \frac{\Sigma}{B},
$$

where $\Sigma$ is the covariance of one example's gradient: each sampled term has expectation $\nabla\mathcal{L}$, and a mean of $B$ i.i.d. terms divides the covariance by $B$ (sampling with replacement). So the gradient noise has standard deviation $\propto 1/\sqrt B$. The noise helps SGD leave sharp regions and contributes to implicit regularization (Section 7), and it is why the learning rate must be decayed to settle (Section 9).

### Momentum as an exponential moving average

PyTorch's SGD with momentum $\mu$ uses $v_t = \mu\,v_{t-1} + g_t$, $\theta_t = \theta_{t-1} - \eta\,v_t$, so $v_t = \sum_{k=0}^{t-1}\mu^k g_{t-k}$. The EMA form used by Adam, $m_t = \beta m_{t-1} + (1-\beta)g_t$, has weights $(1-\beta)\beta^k$ summing to 1; PyTorch's velocity is that EMA divided by $1 - \mu$, a rescaling of the learning rate.

**Effective step.** For a constant gradient, $v_t \to g/(1-\mu)$, so the step is $\eta/(1-\mu)$, ten times $\eta$ at $\mu = 0.9$. For an alternating gradient $g_t = (-1)^t$, the steady state $v_t = (-1)^tA$ satisfies $A = -\mu A + 1$, so $A = 1/(1+\mu) = 0.53$. Consistent components are amplified and oscillating ones damped, a ratio $(1+\mu)/(1-\mu) = 19$ at $\mu = 0.9$: across a narrow valley the gradient flips sign and cancels, along it the gradient accumulates.

**Effective horizon.** Terms older than $K$ steps carry total weight $\beta^K$, and for $K = 1/(1-\beta)$, $\beta^{1/(1-\beta)} = \exp\big(\ln\beta/(1-\beta)\big) \approx e^{-1}$ because $\ln\beta \approx -(1-\beta)$ near 1. So an EMA remembers about $1/(1-\beta)$ steps (10 for 0.9, 1000 for 0.999), with half-life $\ln 2/(-\ln\beta) \approx \ln 2/(1-\beta)$: 6.6, 693 and 6931 steps for 0.9, 0.999 and 0.9999.

**On a quadratic** with condition number $\kappa$, gradient descent with its best fixed step contracts the error by $(\kappa-1)/(\kappa+1)$ per step, and heavy-ball momentum tuned to the extreme eigenvalues asymptotically by $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$ (Polyak, 1964): for $\kappa = 100$, 0.980 versus 0.818, so about 345 versus 34 steps to cut the error a thousandfold. The speedup is a result about quadratics. **Nesterov momentum** evaluates the gradient at the look-ahead point, $v_{t+1} = \mu v_t - \eta\nabla f(\theta_t + \mu v_t)$, $\theta_{t+1} = \theta_t + v_{t+1}$ (Sutskever et al., 2013; PyTorch implements a rearrangement that steps along $g_t + \mu v_t$). On smooth convex functions it attains error $O(1/t^2)$ against $O(1/t)$ for gradient descent; in deep learning the gain over plain momentum is usually small.

### AdaGrad and RMSProp

**AdaGrad** (Duchi, Hazan and Singer, 2011) divides each coordinate by the root of its accumulated squared gradients, $G_t = G_{t-1} + g_t^2$, $\theta_t = \theta_{t-1} - \eta\,g_t/(\sqrt{G_t} + \epsilon)$, elementwise. Rarely active coordinates keep large steps, which suited sparse features, but $G_t$ only grows, so steps decay like $1/\sqrt t$ and long training stalls. **RMSProp** (Tieleman and Hinton, 2012) uses an EMA instead, $s_t = \rho\,s_{t-1} + (1-\rho)\,g_t^2$, $\theta_t = \theta_{t-1} - \eta\,g_t/(\sqrt{s_t} + \epsilon)$ ($\rho = 0.9$ in the lecture, 0.99 in PyTorch). For a constant gradient $c$, $s_t \to c^2$ and the step tends to $\eta\,\mathrm{sign}(c)$: about $\eta$ whatever the gradient's scale.

### Adam, with the bias correction derived

Adam (Kingma and Ba, 2015), with defaults $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g_t, & v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2,\\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, & \hat v_t &= \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.
\end{aligned}
$$

**Bias correction.** Unrolling from $m_0 = 0$ (by induction: $m_1 = (1-\beta_1)g_1$, $m_2 = \beta_1(1-\beta_1)g_1 + (1-\beta_1)g_2$, ...) gives $m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$. If the gradient distribution is stationary, $\mathbb{E}[g_i] = \mathbb{E}[g]$, then by the geometric sum

$$
\mathbb{E}[m_t] = (1-\beta_1)\,\mathbb{E}[g]\sum_{k=0}^{t-1}\beta_1^{k} = (1-\beta_1)\,\mathbb{E}[g]\,\frac{1-\beta_1^t}{1-\beta_1} = (1-\beta_1^t)\,\mathbb{E}[g].
$$

So $m_t$ is biased toward zero by $1 - \beta_1^t$, and dividing by it removes the bias; likewise $\mathbb{E}[v_t] = (1 - \beta_2^t)\,\mathbb{E}[g^2]$. Without correction the update would be the corrected one times $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$: 3.16 at $t = 1$, 6.53 at $t = 10$, 3.24 at $t = 100$ and 1.26 at $t = 1000$ with default betas. $v$ stays biased far longer because $\beta_2$ averages about 1000 steps, so early steps would be several times too large exactly when the estimates are least reliable.

**A step by hand.** Minimize $f(\theta) = \theta^2$ from $\theta_0 = 1$ with $\eta = 0.1$ and default betas:

$$
\begin{aligned}
t = 1:&\quad g = 2,\quad m = 0.1\cdot 2 = 0.2,\quad v = 0.001\cdot 4 = 0.004,\quad \hat m = 2,\quad \hat v = 4,\quad \theta_1 = 1 - 0.1\cdot\tfrac{2}{2} = 0.9 .\\
t = 2:&\quad g = 1.8,\quad m = 0.9\cdot0.2 + 0.1\cdot1.8 = 0.36,\quad v = 0.999\cdot0.004 + 0.001\cdot3.24 = 0.007236,\\
&\quad \hat m = 0.36/0.19 = 1.8947,\quad \hat v = 0.007236/0.001999 = 3.6198,\quad \theta_2 = 0.9 - 0.1\cdot\tfrac{1.8947}{1.9026} = 0.8004 .
\end{aligned}
$$

`torch.optim.Adam` gives $\theta_2 = 0.8004122$. With bias correction the first step is exactly $\eta\,\mathrm{sign}(g_1)$ in every coordinate, and the step barely depends on the gradient's scale: on $f = 1000\theta^2$ the first step is still 0.1, whereas gradient descent with $\eta = 0.1$ would multiply $\theta$ by $1 - 200 = -199$ and diverge. Ignoring $\epsilon$, the update is invariant to rescaling the loss, and Kingma and Ba show each coordinate moves at most about $\eta$ (up to $\eta(1-\beta_1)/\sqrt{1-\beta_2} \approx 3.2\eta$ under extreme sparsity), so $\eta$ acts like a per-coordinate trust region; with $\beta_1 = \beta_2 = 0$ Adam is sign descent. $\epsilon$ is the gradient scale below which normalization stops: steps shrink if a coordinate's gradient RMS approaches it, and a tiny $\epsilon$ lets a coordinate with near-zero $v$ take a huge step when a large gradient arrives.

### Why L2 inside Adam is not weight decay, and AdamW

With an L2 penalty, Adam is fed $g + \lambda\theta$, which enters $m$ and $v$. With $\beta_1 = 0$ for clarity, coordinate $i$ updates as

$$
\theta_i \leftarrow \theta_i - \eta\,\frac{g_i}{\sqrt{\hat v_i}+\epsilon} - \underbrace{\frac{\eta\lambda}{\sqrt{\hat v_i}+\epsilon}}_{\text{decay rate of coordinate } i}\theta_i .
$$

Weights with large gradients are barely regularized and weights with small gradients strongly: with $\eta = 10^{-3}$ and $\lambda = 0.01$, a coordinate with $\sqrt{\hat v} = 10$ decays by $10^{-6}$ per step and one with $\sqrt{\hat v} = 0.1$ by $10^{-4}$, while decoupled decay would shrink both by $10^{-5}$. Under plain SGD the two coincide (L2 coefficient $\lambda'$ is decay at rate $\eta\lambda'$). Under a preconditioner $P = \mathrm{diag}(1/(\sqrt{\hat v}+\epsilon))$, matching $\theta - \eta P(g + \lambda'\theta)$ to $\theta - \eta Pg - \eta\lambda\theta$ for every $\theta$ requires $\lambda' P = \lambda I$, impossible unless $P$ is a multiple of the identity. Loshchilov and Hutter (2019) decouple the decay:

$$
\theta_t = \theta_{t-1} - \eta_t\Big(\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Big),
$$

with $\eta_t$ a schedule multiplier; they report better generalization and much less coupling between the best learning rate and the best $\lambda$. PyTorch's `AdamW` multiplies the decay by the full learning rate, $\theta \leftarrow \theta - \gamma\lambda\theta$ with default $\lambda = 0.01$, so its $\lambda$ is not the paper's. A sanity number: at learning rate $3\times10^{-4}$ with $\lambda = 0.1$, a weight that gets no gradient halves every $\ln 2/(3\times10^{-5}) \approx 23{,}000$ steps.

### Memory, and which optimizer to use

| Optimizer | State per parameter | Bytes with fp32 state |
|---|---|---|
| SGD | none | 0 |
| SGD with momentum, Nesterov | velocity | 4 |
| AdaGrad, RMSProp | squared-gradient accumulator | 4 |
| Adam, AdamW | $m$ and $v$ | 8 |
| Lion | momentum | 4 |
| Adafactor | row and column statistics per matrix | $O(n + m)$ per $n\times m$ matrix |

Mixed precision adds a 4-byte fp32 master copy (Section 10), giving Adam's 16 bytes per parameter (Section 12). SGD with momentum remains strong for convolutional classifiers, with half of Adam's state, and generalized better than adaptive methods on several benchmarks (Wilson et al., 2017). AdamW is the default for transformers, diffusion models and super-resolution networks (EDSR, ESRGAN and SwinIR all train with Adam); why transformers need it is the last question below.

**Newer optimizers, briefly.** **Lion** (Chen et al., 2023) stores only momentum and steps by $\eta\big(\mathrm{sign}(\beta_1m + (1-\beta_1)g) + \lambda\theta\big)$, then updates $m \leftarrow \beta_2 m + (1-\beta_2)g$ (defaults 0.9, 0.99); since every coordinate moves by exactly $\eta$, the authors suggest a learning rate 3 to 10 times smaller than AdamW's and weight decay 3 to 10 times larger. **Shampoo** (Gupta, Koren and Singer, 2018) preconditions each weight matrix's gradient $G$ from both sides with $L = \sum GG^\top$ and $R = \sum G^\top G$, stepping along $L^{-1/4}GR^{-1/4}$. **Muon** (Jordan et al., 2024) replaces the momentum matrix $M = USV^\top$ of each 2D hidden weight by an approximation of $UV^\top$ from a few Newton–Schulz iterations, giving every singular direction the same step; other parameters use AdamW, and Liu et al. (2025) report about twice AdamW's compute efficiency in LLM runs.

::: key
**How to say it in an interview.** "Momentum is an EMA of gradients: consistent directions accumulate to an effective step of $\eta/(1-\mu)$ and oscillating ones cancel. RMSProp divides by an EMA of squared gradients, so each coordinate moves about $\eta$. Adam does both and divides each average by $1-\beta^t$, because averages started at zero are biased low by exactly that factor. AdamW applies weight decay outside the adaptive step, because L2 added to the gradient gets divided by $\sqrt v$ and decays each weight at a different rate."
:::

::: qa
**Explain SGD, momentum and Adam. How do they differ?**

SGD steps along the negative minibatch gradient (unbiased, covariance $\Sigma/B$) with one global learning rate. Momentum keeps an exponentially weighted sum of past gradients, damping oscillation across steep directions, accelerating along consistent ones, and raising the effective step to $\eta/(1-\mu)$. Adam adds an EMA of squared gradients and divides by its root per coordinate, giving each parameter its own step size and making the update roughly invariant to gradient scale, with bias correction for the zero-initialized averages. SGD with momentum stays competitive for CNN classification with half the optimizer memory; Adam-family optimizers are the default for transformers and diffusion models.
:::

::: {.qa asked="MSR intern, ByteDance 2020, Meituan 2022"}
**Explain SGD with momentum, RMSProp and Adam. When would you choose each, and how do you control how fast Adam converges?**

Momentum averages gradient directions ($v_t = \mu v_{t-1} + g_t$, effective step $\eta/(1-\mu)$ along consistent directions); RMSProp divides each coordinate by an EMA of its squared gradient, so steps are about $\eta$ whatever the gradient scale; Adam does both and divides each average by $1 - \beta^t$ to remove the bias from starting at zero. **Choosing**: SGD with momentum for CNN classification, where it often generalizes as well or better with half the state; AdamW for transformers, diffusion and restoration models, whose blocks have very different gradient scales; RMSProp is rarely used now (Adam with $\beta_1 = 0$ is RMSProp plus bias correction). **Controlling Adam's speed**: each coordinate moves about $\eta$ per step, so the learning rate and schedule are the main lever, with warmup while $v$ is poorly estimated; $\beta_1$ sets inertia; $\beta_2$ sets the memory of the gradient scale (0.999 can lag a sudden rise and cause spikes, hence 0.95 in many LLM recipes); a larger $\epsilon$ makes small-gradient coordinates take SGD-like steps; and clipping caps steps during spikes.
:::

::: qa
**Write Adam's update equations and derive the bias correction. What goes wrong without it?**

Adam keeps EMAs $m_t$ of $g_t$ and $v_t$ of $g_t^2$ and steps by $\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. Unrolling from zero, $m_t = (1-\beta_1)\sum_{i=1}^t\beta_1^{t-i}g_i$, whose weights sum to $1 - \beta_1^t$, so under stationary gradients $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]$; dividing by $1 - \beta_1^t$ removes the bias, and likewise for $v_t$. Without correction both are shrunk toward zero, $v$ much more because $\beta_2 = 0.999$ averages about 1000 steps, so $m_t/\sqrt{v_t}$ is too large by $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$: 3.2 at step 1, 6.5 at step 10, still 3.2 at step 100, when the estimates are least reliable. With correction the first step is exactly $\eta\,\mathrm{sign}(g_1)$.
:::

::: qa
**What is the difference between Adam with L2 regularization and AdamW?**

With L2, $\lambda\theta$ is added to the gradient, enters $m$ and $v$, and is divided by $\sqrt{\hat v}$, so coordinate $i$ decays at rate $\eta\lambda/\sqrt{\hat v_i}$: weights with large gradients are regularized less, an accidental coupling to gradient statistics (a factor of 100 between $\sqrt{\hat v} = 10$ and 0.1). AdamW applies $\theta\leftarrow\theta - \eta\lambda\theta$ outside the adaptive step, so all weights shrink at the same relative rate. Under plain SGD they coincide after rescaling $\lambda$ by the learning rate; under Adam no L2 coefficient matches, because the preconditioner is not a multiple of the identity. AdamW generalizes better and decouples the best learning rate from the best $\lambda$.
:::

::: qa
**Why do transformers train much better with Adam than with SGD?**

Different parts of a transformer (embeddings, attention, MLPs, normalization) have very different gradient and curvature scales. Adam's per-coordinate normalization adapts to each, while SGD's single step size must suit the sharpest block and is then too slow elsewhere. Evidence: block-heterogeneous Hessian spectra (Zhang et al., 2024), heavy-tailed gradient noise in attention (Zhang et al., 2020), a gap that survives full-batch training, pointing to Adam's sign-like update (Kunstner et al., 2023), and for language, heavy-tailed class imbalance from rare tokens (Kunstner et al., 2024). No single explanation covers every setting.
:::

## 9. Learning-rate schedules and batch size

### Why the learning rate has to come down

A one-dimensional model explains most schedule folklore. Take the loss $\tfrac12h\theta^2$ ($h > 0$) and a gradient estimate $h\theta + \xi$ with noise variance $s^2/B$, so one SGD step is $\theta_{t+1} = (1 - \eta h)\,\theta_t - \eta\,\xi_t$. Taking variances (the noise is independent of $\theta_t$), $V_{t+1} = (1-\eta h)^2V_t + \eta^2s^2/B$, whose fixed point is

$$
V_\infty = \frac{\eta^2 s^2/B}{1 - (1-\eta h)^2} = \frac{\eta\,s^2}{B\,h\,(2 - \eta h)} \approx \frac{\eta}{B}\cdot\frac{s^2}{2h}\quad(\eta h \ll 1).
$$

(Simulating with $\eta = 0.1$, $h = 2$, $s^2 = 1$, $B = 4$ gives 0.006943 against the formula's 0.006944.) At a fixed learning rate SGD never converges: it hovers in a band whose size grows with $\eta$ and the noise, and decaying $\eta$ shrinks the band, which is why loss drops visibly when the rate is decayed. The band depends on $\eta$ and $B$ only through $\eta/B$, the origin of the linear scaling rule below. And the iteration contracts only if $\lvert 1 - \eta h\rvert < 1$, i.e. $\eta < 2/h$: the sharpest direction caps the learning rate.

### Warmup

Linear warmup ramps the rate from near zero to its peak over the first few hundred to few thousand steps. Reasons, depending on the setup:

- **Adam's second moment is poor at first.** At $t = 1$, $\hat v_1 = g_1^2$, so every coordinate moves by exactly $\eta$, including pure-noise ones; RAdam (Liu et al., 2020) shows the adaptive step's variance is very large early and proposes a rectification term instead.
- **The loss surface near initialization tends to be sharp**, so full-size steps can throw the network somewhere it does not recover from; Kalra and Barkeshli (2024) find warmup mainly lets the network reach better-conditioned regions where a larger target rate is stable.
- **Post-LN transformers** have large gradients near the output at initialization (Xiong et al., 2020), while Pre-LN can train without warmup (see [transformers](transformers.html)).
- **Large-batch SGD** needs a 5-epoch gradual warmup for the linear scaling rule to work while weights change fast (Goyal et al., 2017).

### Step, cosine and warmup-stable-decay

**Step decay** multiplies the rate by 0.1 at fixed epochs (30, 60 and 80 of 90 for ResNet-50 in Goyal et al.). **Cosine decay** (from SGDR, Loshchilov and Hutter, 2017), $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})\big(1 + \cos(\pi t/T)\big)$, falls slowly at both ends: its multiplier is 0.854, 0.5 and 0.146 at $T/4$, $T/2$ and $3T/4$. It needs $T$ in advance, and the Chinchilla authors found matching the cycle to the actual token budget matters. **Warmup-stable-decay** (WSD, introduced with MiniCPM, Hu et al., 2024) holds the peak rate for most of training and then decays quickly; Hägele et al. (2024) find a constant rate plus a short cooldown matches cosine. Since a decay can branch off any stable-phase checkpoint, one run yields good models at several budgets. Diffusion models are often trained at a constant rate after warmup and evaluated with EMA weights (Section 11); EDM2 uses an inverse-square-root decay.

### Batch size and the linear scaling rule

Goyal et al. (2017) compare $k$ SGD steps on minibatches $\mathcal{B}_0, \dots, \mathcal{B}_{k-1}$ of size $n$ with one step on their union:

$$
\theta_{t+k} = \theta_t - \eta\,\frac1n\sum_{j<k}\sum_{x\in\mathcal{B}_j}\nabla\ell(x, \theta_{t+j}),\qquad
\hat\theta_{t+1} = \theta_t - \hat\eta\,\frac{1}{kn}\sum_{j<k}\sum_{x\in\mathcal{B}_j}\nabla\ell(x, \theta_t).
$$

If gradients barely change over $k$ steps, $\nabla\ell(x, \theta_{t+j}) \approx \nabla\ell(x, \theta_t)$, the updates match when $\hat\eta = k\eta$: **multiply the learning rate by the factor the batch grew**. The noise analysis agrees, since fixed $\eta/B$ keeps the band fixed. Their rule $\eta = 0.1\cdot B/256$ gives $\eta = 3.2$ at $B = 8192$, which with warmup matched small-batch accuracy while training ResNet-50 on ImageNet in one hour on 256 GPUs. It fails early in training (hence warmup), when $k\eta$ passes the curvature limit $2/h$, and past the critical batch size.

**Square-root scaling for Adam.** Matching the variance of one step, $\eta^2\Sigma/B$, suggests $\eta\propto\sqrt B$; Krizhevsky (2014) stated this but found linear scaling better for his SGD runs. For adaptive methods, Malladi et al. (2022) derive from a stochastic differential equation approximation that scaling the batch by $\kappa$ calls for $\eta\sqrt\kappa$, $1 - \beta \to \kappa(1-\beta)$ for both betas, and $\epsilon/\sqrt\kappa$, while that approximation holds. Either rule is a starting point for a sweep.

**The critical batch size.** McCandlish et al. (2018) model the trade-off between optimization steps $S$ and examples $E$ as $(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$, with critical batch $B_{\text{crit}} = E_{\min}/S_{\min}$, related to the gradient noise scale $\mathrm{tr}(\Sigma)/\lVert G\rVert^2$. Well below it, doubling the batch nearly halves the steps; well above it, larger batches mostly waste examples. As the loss falls $\lVert G\rVert$ shrinks and the noise scale grows, one reason to ramp the batch size up during training, as GPT-3 did.

### Gradient clipping and accumulation

**Global-norm clipping** rescales the whole gradient when its norm exceeds $c$, $g \leftarrow g\cdot\min(1, c/\lVert g\rVert_2)$. For $g = (3, 4)$ and $c = 1$, $\lVert g\rVert = 5$ and $g$ becomes $(0.6, 0.8)$, same direction; clipping each value to $[-1, 1]$ gives $(1, 1)$, turning the direction from $53.1°$ to $45°$. It was popularized for RNNs (Pascanu et al., 2013); GPT-3 and LLaMA clip at 1.0. Clip after accumulation and the data-parallel all-reduce, unscale first under fp16 loss scaling, and log the pre-clip norm, one of the earliest signs of instability.

**Gradient accumulation** runs $k$ micro-batches, calls backward on each with the loss divided by $k$, and steps once. For a mean loss over independent examples this is the large-batch gradient exactly, up to floating-point order, since $\frac1k\sum_j\frac1n\sum_{x\in\mathcal{B}_j}\nabla\ell(x) = \frac{1}{kn}\sum_{x}\nabla\ell(x)$ and `.grad` accumulation performs the outer sum; the exceptions are in the question below.

::: qa
**Why does learning-rate warmup help, especially with Adam and large batches?**

Early on, Adam's $v$ averages only a few gradients, so per-coordinate step sizes are noisy (at step 1 every coordinate moves exactly $\eta$), and the loss surface near initialization tends to be sharp, so a full-size step can push the model somewhere it never recovers from. Small early steps let the statistics settle and the network reach flatter regions, after which a larger peak rate is stable. For large-batch SGD the same issue appears as the linear scaling rule failing while weights change fast; in Post-LN transformers, as large gradients near the output at initialization.
:::

::: {.qa asked="Meta research round, Glassdoor"}
**When you increase the batch size, how should you change the learning rate and other hyperparameters?**

For SGD, scale the learning rate linearly with the batch, with warmup: $k$ small steps match one $k$-times larger step when the gradient changes little, and the noise level depends on $\eta/B$. This holds up to a critical batch size, related to the gradient noise scale $\mathrm{tr}(\Sigma)/\lVert G\rVert^2$, beyond which larger batches save few steps. For Adam, square-root scaling is a common heuristic, with SDE-based adjustments to the betas and $\epsilon$. With a fixed budget of samples or tokens a larger batch means fewer steps, so define warmup and decay in samples or tokens, and revisit weight decay and EMA decay, which act per step. Check that loss curves against samples seen roughly overlap, and sweep around the scaled value.
:::

::: qa
**If you increase the batch size 8×, how should you change the learning rate?**

For SGD with momentum, start from the linear rule ($\times 8$) with warmup; it breaks down early in training, past the critical batch size, and when the scaled rate exceeds the curvature limit $2/h$. For Adam, the SDE-based square-root rule (Malladi et al., 2022) suggests $\times\sqrt8 \approx 2.8$, with $1 - \beta$ multiplied by 8 and $\epsilon$ divided by $\sqrt8$. Either way, sweep around the scaled value.
:::

::: qa
**How does the learning rate usually evolve in large-model training, why decay at the end, and why might you prefer WSD over cosine?**

A short linear warmup, then cosine decay to about 10% of the peak, or WSD's long constant phase plus a short anneal. At a fixed rate the iterates hover in a band around the valley floor whose size grows with $\eta/B$ and the gradient noise; decay lets them settle, which is why loss drops visibly during cooldown. Cosine bakes the total budget in, so stopping early or extending training gives a suboptimal curve. WSD adds a short decay whenever a model is needed, so one run yields good checkpoints at several budgets and can be continued from the stable phase; Hägele et al. (2024) find it matches cosine.
:::

::: qa
**Is gradient accumulation exactly equivalent to a larger batch? What are the gotchas?**

For a mean loss over independent examples, summing $k$ micro-batch gradients of (loss$/k$) equals the big-batch gradient up to floating-point order. Gotchas: BatchNorm statistics are per micro-batch; with variable-length sequences, averaging per-micro-batch token means is not the global token mean (10 tokens with summed loss 20 and 90 with summed loss 90 give 1.5 averaged per micro-batch but 1.1 per token), so normalize by the total token count; under DDP, skip gradient sync on intermediate micro-steps (`no_sync`) or you all-reduce $k$ times; and step the scheduler per optimizer step.
:::

## 10. Numerics and mixed precision

### Floating-point formats

A normal floating-point number stores a sign $s$, an exponent field $e$ and a mantissa $f$, representing $(-1)^s\,2^{e - \text{bias}}\,(1.f)_2$: exponent bits set the range, mantissa bits the precision. Every entry below was checked with `torch.finfo` and by casting powers of two:

| Format | Sign | Exponent | Mantissa | Largest value | Smallest normal | Smallest subnormal | Machine epsilon |
|---|---|---|---|---|---|---|---|
| fp32 | 1 | 8 | 23 | $3.40\times10^{38}$ | $2^{-126} \approx 1.18\times10^{-38}$ | $2^{-149} \approx 1.4\times10^{-45}$ | $2^{-23} \approx 1.19\times10^{-7}$ |
| fp16 | 1 | 5 | 10 | 65504 | $2^{-14} \approx 6.10\times10^{-5}$ | $2^{-24} \approx 5.96\times10^{-8}$ | $2^{-10} \approx 9.77\times10^{-4}$ |
| bf16 | 1 | 8 | 7 | $3.39\times10^{38}$ | $2^{-126} \approx 1.18\times10^{-38}$ | $2^{-133} \approx 9.2\times10^{-41}$ | $2^{-7} = 7.81\times10^{-3}$ |

fp16 has exponent bias 15 and largest finite exponent field 30, so its largest value is $(2 - 2^{-10})\cdot2^{15} = 65504$. bf16 is the top 16 bits of fp32: same 8-bit exponent, mantissa cut to 7 bits, so fp32's range, $(2 - 2^{-7})\cdot2^{127} \approx 3.39\times10^{38}$, with far less precision. Machine epsilon is the gap between 1 and the next number; the worst relative rounding error is half of it. In fp16, $1 + 0.0004$ rounds to 1 (below half the spacing, 0.000488) while $1 + 0.0005$ gives 1.000977; in bf16, $1 + 0.003$ rounds to 1 and $1 + 0.004$ gives 1.0078.

### Loss scaling, and why bf16 usually does not need it

Many gradients, especially activation gradients deep in a network, are tiny. fp16 cannot represent anything below $2^{-24}$, and values below half that round to zero: $10^{-8}$ becomes exactly 0. **Loss scaling** multiplies the loss by $S$ before the backward pass; since the backward pass is linear in the upstream gradient, every gradient is multiplied by $S$. With $S = 1024$ the $10^{-8}$ gradient becomes $1.03\times10^{-5}$ (a subnormal), with $S = 65536$ it becomes $6.55\times10^{-4}$ (normal). Gradients are divided by $S$ in fp32 before clipping and the update. The risk is overflow above 65504, so **dynamic loss scaling** adapts $S$: PyTorch's `GradScaler` starts at $2^{16}$, skips any step with an inf or NaN gradient and halves $S$, and doubles $S$ after 2000 clean steps. bf16 has fp32's exponent range, so loss scaling is usually unnecessary; Kalamkar et al. (2019) found bf16 training matched fp32 across many models without hyperparameter changes. Its weakness is precision (relative rounding error up to 0.4%), so hardware accumulates matrix products in fp32 and autocast runs reductions (softmax, normalization statistics, losses) in fp32. That is the recipe of Micikevicius et al. (2018): 16-bit compute, an fp32 master copy of the weights, loss scaling for fp16, and fp32 accumulation.

### Why keep fp32 master weights

An update $\theta \leftarrow \theta - \Delta$ is lost if $\lvert\Delta\rvert$ is below half the spacing near $\theta$: about $\lvert\theta\rvert/2048$ in fp16 and $\lvert\theta\rvert/256$ in bf16 (the exact factor depends on where $\theta$ sits between powers of two). A weight of 1.0 updated by $10^{-4}$ stays 1.0 in both 16-bit formats, while fp32 gives 0.9999. With learning rates of $10^{-4}$ to $10^{-5}$, thousands of small updates that should add up would each round away, so the optimizer updates an fp32 copy and casts to 16 bits for the next forward pass (alternatives: stochastic rounding, compensated summation). Optimizer state has the same problem: Adam's default $\epsilon = 10^{-8}$ rounds to exactly 0 in fp16.

### Overflow in softmax and exp

$e^z$ overflows above $z \approx 88.7$ in fp32 and bf16 and above $z \approx 11.09$ in fp16 ($e^{12}$ is `inf` in fp16), so a softmax over logits of 12 gives $\infty/\infty$, NaN, in fp16. The fix is max-subtraction: $\mathrm{softmax}(z) = \mathrm{softmax}(z - \max_j z_j)$, because $e^{-\max_j z_j}$ cancels between numerator and denominator and every exponent becomes at most 0. Related habits: use fused `log_softmax`, `logsigmoid` and `BCEWithLogitsLoss` instead of logs of probabilities that may underflow; compute variances by a two-pass or Welford algorithm rather than $\mathbb{E}[x^2] - \mathbb{E}[x]^2$; keep $\epsilon$ inside square roots of variances. Growing attention logits are a known overflow source in transformers (see [transformers](transformers.html)).

::: key
**How to say it in an interview.** "fp16 has 5 exponent bits, so it tops out at 65504 and small gradients underflow below $6\times10^{-8}$; loss scaling multiplies the loss so every gradient moves into range, then divides back in fp32. bf16 keeps fp32's 8 exponent bits, so it rarely needs scaling, but has only 7 mantissa bits, so we accumulate in fp32 and keep fp32 master weights: an update smaller than about 1/256 of the weight vanishes in bf16."
:::

::: qa
**Why is bf16 usually preferred over fp16 for training large models?**

bf16 has fp32's 8-bit exponent and so the same dynamic range: activations and gradients essentially never overflow or underflow, loss scaling is usually unnecessary, and spikes such as large attention logits do not hit fp16's 65504 ceiling (in fp16, $e^{12}$ already overflows). The cost is 7 mantissa bits instead of 10, about 0.4% relative precision instead of 0.05%, which is tolerable because gradient noise dwarfs rounding error, provided accumulations, reductions and the weight update stay in fp32. The practical win is fewer overflow-driven skipped steps and NaNs.
:::

::: qa
**In mixed precision, why keep fp32 master weights, and why does loss scaling work?**

Two ways 16-bit updates fail: tiny updates underflow to zero, and when the weight is about 2048 times larger than its update (fp16; about 256 for bf16) the addition rounds the update away, as $1 - 10^{-4}$ rounds to 1 in both formats. Many small updates that should accumulate then do nothing, so the master copy accumulates them in fp32. Loss scaling addresses a different problem, small gradients underflowing during backprop (fp16's smallest positive value is $2^{-24}$). Backprop is linear in the upstream gradient, so multiplying the loss by $S$ multiplies every gradient by $S$, and dividing by $S$ in fp32 before the step restores them. Overflow is the only risk, which dynamic scaling handles by skipping the step and halving $S$.
:::

::: {.qa asked="ByteDance (2 reports), Tencent Hunyuan (Nowcoder)"}
**How does mixed-precision training work, and why loss scaling? What do gradient checkpointing and ZeRO save?**

Forward and backward run in fp16 or bf16, with matrix products accumulated in fp32 and reductions in fp32, while an fp32 master copy of the weights and the optimizer states is kept for the update. fp16's narrow exponent range lets small gradients underflow, so the loss is multiplied by $S$ (PyTorch starts at $2^{16}$) and gradients unscaled before the update, with $S$ halved and the step skipped on inf or NaN. bf16 has fp32's range and usually needs no scaling. Mixed precision does not shrink model-state memory (16 bytes per parameter with Adam); it cuts activation memory and speeds up compute. Gradient checkpointing stores only some activations and recomputes the rest during backward, down to $O(\sqrt L)$ layers' worth for about one extra forward pass. ZeRO partitions optimizer states (stage 1), gradients (stage 2) and parameters (stage 3) across data-parallel ranks, from $16\Psi$ toward $16\Psi/N$ bytes per GPU, for more communication.
:::

## 11. EMA of weights

Keep a second copy of the weights, updated after every optimizer step and used only for evaluation and sampling:

$$
\bar\theta_t = \beta\,\bar\theta_{t-1} + (1-\beta)\,\theta_t \quad\Longrightarrow\quad \bar\theta_t = (1-\beta)\sum_{k=0}^{t-1}\beta^k\,\theta_{t-k} + \beta^t\,\bar\theta_0 ,
$$

a geometric average over roughly the last $1/(1-\beta)$ steps, with half-life about 693 steps for $\beta = 0.999$ and 6931 for 0.9999. The optimizer never sees it.

**Why it helps.** At a non-decayed learning rate the iterates jitter around the valley floor with variance $\propto\eta/B$ (Section 9). Averaging $K$ roughly independent iterates divides that variance by about $K$, and for a locally convex loss the loss at the average is no larger than the average loss (Jensen's inequality): Polyak–Ruppert averaging (Polyak and Juditsky, 1992), a learning-rate decay that does not slow the optimizer. Diffusion models depend on it because their objective is very noisy, they often train at a near-constant rate, and samples are sensitive to small weight changes (color shifts, FID swings between checkpoints; Song and Ermon, 2020); DDPM used $\beta = 0.9999$. The same EMA is the teacher in MoCo ($\beta = 0.999$) and BYOL.

**The initialization problem.** Started as a copy of $\theta_0$, the EMA keeps weight $\beta^t$ on the initial weights: with $\beta = 0.9999$, 0.905 after 1000 steps and still $e^{-1}\approx 0.368$ after 10,000. Remedies: bias correction as in Adam if the EMA starts at zero ($\bar\theta_t/(1 - \beta^t)$); a warmed-up decay such as TensorFlow's $\beta_t = \min\big(\beta, \tfrac{1+t}{10+t}\big)$, which is 0.1 at step 0, 0.918 at step 100, 0.991 at step 1000, and reaches 0.9999 only after about 90,000 steps; or a profile defined relative to training length. EDM2 (Karras et al., 2024) uses a power-function EMA, $\bar\theta(t)\propto\int_0^t\tau^\gamma\theta(\tau)\,d\tau$, implemented with $\beta_t = (1 - 1/t)^{\gamma+1}$, which gives the initialization zero weight and is parameterized by a relative width $\sigma_{\text{rel}}$. Its **post-hoc EMA** stores snapshots of two such averages ($\sigma_{\text{rel}} = 0.05$ and 0.10) and afterwards reconstructs any other profile as their least-squares best linear combination, so EMA length can be tuned against FID without retraining; the best length varies across configurations and interacts with guidance strength (see [diffusion models](diffusion-models.html)).

```python
import copy, torch

@torch.no_grad()
def ema_update(ema_model, model, step, beta=0.9999):
    b = min(beta, (1 + step) / (10 + step))          # warmed-up decay
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.lerp_(p, 1 - b)                            # pe = b*pe + (1-b)*p
    for be, bm in zip(ema_model.buffers(), model.buffers()):
        be.copy_(bm)                                  # e.g. BatchNorm running statistics

model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
ema_model = copy.deepcopy(model).eval()               # evaluation and sampling use this copy
ema_update(ema_model, model, step=0)
```

::: pitfall
Keep the EMA copy in fp32: with $\beta = 0.9999$ each step moves it by $10^{-4}$ of the gap to the current weights, far below bf16's resolution, so a bf16 EMA mostly stops moving. Update it under `torch.no_grad()` after the optimizer step, and give it BatchNorm running statistics if the model has BatchNorm.
:::

::: qa
**What does an EMA of weights do, and why do diffusion models depend on it?**

It averages the parameter trajectory over roughly $1/(1-\beta)$ steps, cancelling the jitter SGD leaves in the weights at a non-decayed learning rate; the optimizer never sees it. For a locally convex loss the average is at least as good as the typical iterate, with much smaller variance. Diffusion models train on a very high-variance objective, often long at a near-constant rate, and their samples are sensitive to small weight changes, so raw weights show color shifts and FID swings between neighboring checkpoints. The EMA model behaves like a well-annealed one; DDPM used $\beta = 0.9999$.
:::

::: qa
**How would you choose the EMA decay, and what is post-hoc EMA?**

$\beta = 1 - 1/N$ averages about the last $N$ steps (half-life $0.69N$). Pick $N$ as a modest fraction of the run, avoid windows reaching back into early training (or warm up $\beta$, since an EMA started at $\theta_0$ with $\beta = 0.9999$ is still 90% initialization after 1000 steps), and expect longer runs and lower learning rates to want longer windows. EDM2's power-function EMA defines width relative to training time and never weights the initialization. Post-hoc EMA stores snapshots of two such averages during training and later solves a small least-squares problem to synthesize weights for any other profile, so EMA length can be swept against FID without retraining.
:::

## 12. Training at scale

### Memory for model states: 16 bytes per parameter

Mixed-precision Adam keeps, per parameter, a 16-bit weight for compute, its 16-bit gradient, an fp32 master weight, and Adam's $m$ and $v$ in fp32: **16 bytes**, the ZeRO paper's accounting, before any activations.

| Item | Bytes per parameter | 1B parameters |
|---|---|---|
| bf16 weights | 2 | 2 GB |
| bf16 gradients | 2 | 2 GB |
| fp32 master weights | 4 | 4 GB |
| Adam $m$ (fp32) | 4 | 4 GB |
| Adam $v$ (fp32) | 4 | 4 GB |
| **Total** | **16** | **16 GB** (14.9 GiB) |

A 7B model needs 112 GB, more than one 80 GB GPU before a single activation. Pure fp32 Adam also costs 16 bytes (4 weights, 4 gradients, 8 moments), so mixed precision does not reduce model-state memory; it halves activation memory and speeds up matrix products. Variations: fp32 gradients (18 bytes), 8-bit optimizer states, and LoRA-style fine-tuning, which keeps optimizer state only for a few trainable parameters.

### Activation memory and checkpointing

For transformers, Korthikanti et al. (2022) count about $sbh(34 + 5as/h)$ bytes of 16-bit activations per layer, for sequence length $s$, micro-batch $b$, hidden size $h$ and $a$ heads; the $5as/h$ term is the attention-score matrix, which fused attention kernels do not store. For $h = 2048$, 24 layers, 16 heads (about $12\cdot24\cdot2048^2 \approx 1.21$B parameters in the blocks, so 19 GB of model states), $s = 2048$ and $b = 8$: storing everything takes 3.83 GB per layer, 91.8 GB in total; without the attention matrix, $34\,sbh \approx 1.14$ GB per layer, 27.4 GB; with full checkpointing, keeping only each layer's input ($2sbh$ bytes), 67 MB per layer, 1.6 GB, plus one layer's activations during recomputation.

**Activation checkpointing** (Chen et al., 2016) stores activations only at checkpoints and recomputes the rest during backward. Splitting $n$ layers into $k$ segments keeps $k$ segment inputs plus the $n/k$ layers of the segment being recomputed; $k + n/k$ is smallest at $k = \sqrt n$, giving $2\sqrt n$ layers' worth instead of $n$ (20 instead of 100 for $n = 100$). The cost is one extra forward pass over the recomputed layers, roughly 33% more compute since the forward pass is about a third of a step. Selective recomputation recomputes only tensors that are large but cheap, such as attention scores. In PyTorch: `torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)`.

### Data parallelism, DDP, ZeRO and FSDP

In data parallelism each of $N$ GPUs holds a full replica, processes its slice of the batch, and averages gradients with an all-reduce before each step. A ring all-reduce is a reduce-scatter (each GPU passes $1/N$ of the gradient to its neighbor $N-1$ times, ending with the full sum of its slice) followed by an all-gather of the summed slices, so each GPU sends $2(N-1)\Psi/N \approx 2\Psi$ values per step: 3.5 GB for 1B bf16 gradients on 8 GPUs. PyTorch's DDP (Li et al., 2020) runs one process per GPU and all-reduces gradients in 25 MB buckets as they become ready, overlapping communication with backward. ZeRO (Rajbhandari et al., 2020) removes the redundant copies of model state across the $N$ ranks:

| Stage | Partitioned | Per-GPU model states | Communication | 7.5B, $N = 64$ | 1B, $N = 8$ |
|---|---|---|---|---|---|
| Data parallel | nothing | $16\Psi$ | $2\Psi$ | 120 GB | 16 GB |
| ZeRO-1 | optimizer states | $4\Psi + 12\Psi/N$ | $2\Psi$ | 31.4 GB | 5.5 GB |
| ZeRO-2 | + gradients | $2\Psi + 14\Psi/N$ | $2\Psi$ | 16.6 GB | 3.75 GB |
| ZeRO-3 | + parameters | $16\Psi/N$ | $3\Psi$ | 1.9 GB | 2.0 GB |

The 7.5B column reproduces the ZeRO paper's Figure 1; the mechanics are in the question below. PyTorch's FSDP (Zhao et al., 2023) implements full sharding with options matching the lighter stages, and ZeRO-Offload moves optimizer state to CPU. No stage shards activations.

### Tensor and pipeline parallelism

**Tensor parallelism** (Megatron-LM, Shoeybi et al., 2019) splits matrices inside a layer: for $Y = \mathrm{GeLU}(XA)B$, split $A$ by columns so each GPU computes $\mathrm{GeLU}(XA_i)$ alone (the nonlinearity is elementwise), and $B$ by rows so $Y = \sum_i\mathrm{GeLU}(XA_i)B_i$ needs one all-reduce; attention is split by heads. Its all-reduces are frequent and blocking, so it stays within a node. **Pipeline parallelism** (GPipe, Huang et al., 2019) puts contiguous layers on $K$ stages and streams $M$ micro-batches; the idle bubble is $(K-1)/(M+K-1)$: 43% for $K = M = 4$, 16% for $M = 16$, 8.6% for $M = 32$. Only boundary activations cross GPUs, so pipelines span nodes. Large runs combine all three.

### Scaling laws and compute

A forward pass costs about $2N$ FLOPs per token (a multiply and an add per weight) and the backward pass about twice that (Section 1), so training on $D$ tokens costs

$$
C \approx 6ND \ \text{FLOPs},
$$

ignoring attention-score FLOPs, which matter at long context. Kaplan et al. (2020) fit power laws such as $L(N) = (N_c/N)^{\alpha_N}$ with $\alpha_N \approx 0.076$ (non-embedding parameters) and $\alpha_D \approx 0.095$, and concluded extra compute should go mostly into model size, $N_{\text{opt}}\propto C^{0.73}$. Hoffmann et al. (2022) found parameters and tokens should grow about equally ($N_{\text{opt}}\propto C^a$, $D_{\text{opt}}\propto C^b$, $a \approx b \approx 0.5$), about **20 tokens per parameter**; their 70B Chinchilla on 1.4T tokens ($6ND \approx 5.9\times10^{23}$ FLOPs) beat the 280B Gopher on 300B tokens ($5.0\times10^{23}$). From their fit $L(N, D) = E + A/N^\alpha + B/D^\beta$ with $\alpha = 0.34$, $\beta = 0.28$, substitute $D = C/(6N)$ and set $\partial L/\partial N = 0$ at fixed $C$:

$$
-\alpha A N^{-\alpha-1} + \beta B\,(C/6)^{-\beta}N^{\beta-1} = 0 \;\Rightarrow\; N^{\alpha+\beta} = \frac{\alpha A}{\beta B}\Big(\frac C6\Big)^{\beta} \;\Rightarrow\; N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)} = C^{0.45},
$$

and $D_{\text{opt}} = C/(6N_{\text{opt}})\propto C^{0.55}$ (0.46 and 0.54 with unrounded exponents). Chinchilla's main explanation of the disagreement: Kaplan et al. used one schedule length for all runs, so losses read partway through a long cosine schedule overstate properly scheduled shorter runs, biasing the fit toward bigger models. Models that will serve many queries are deliberately trained on more tokens per parameter (LLaMA-7B saw 1T tokens), trading training compute for cheaper inference.

::: key
**How to say it in an interview.** "Mixed-precision Adam costs 16 bytes per parameter (2 weights, 2 gradients, 4 master copy, 4 and 4 for Adam's moments), so 1B parameters is 16 GB before activations. Activations scale with batch, length and depth, and checkpointing trades about a third more compute for most of that memory. DDP all-reduces about $2\Psi$ per step; ZeRO stages 1 to 3 shard optimizer state, gradients and parameters, down to $16\Psi/N$. Training compute is about $6ND$, and Chinchilla says about 20 tokens per parameter is compute-optimal."
:::

::: qa
**Estimate the GPU memory needed to train a 7B-parameter model with Adam in mixed precision.**

Model states take 16 bytes per parameter (16-bit weights 2, gradients 2, fp32 master 4, $m$ 4, $v$ 4): 112 GB, more than one 80 GB GPU before activations, which scale with micro-batch, sequence length, width and depth. To fit: shard with ZeRO or FSDP (14 GB per GPU on 8 GPUs at full sharding), checkpoint activations, offload optimizer state to CPU, or shrink the trainable state (LoRA, 8-bit optimizer states).
:::

::: {.qa asked="Runway (Glassdoor 2025–26)"}
**Estimate the compute and memory needed to train a given network.**

Memory: about 16 bytes per parameter for mixed-precision Adam, plus activations, which scale with batch, sequence length or resolution, width and depth, often dominate, and shrink with checkpointing. Compute: a transformer costs about $6\times$ parameters $\times$ tokens FLOPs; for a CNN, sum per-layer multiply-adds ($k^2C_{\text{in}}C_{\text{out}}H_{\text{out}}W_{\text{out}}$ per convolution), double for FLOPs, and triple for forward plus backward. Divide by achievable throughput, typically 30 to 50% of peak. Example: 1B parameters on 100B tokens is $6\times10^{20}$ FLOPs; at an effective $4\times10^{14}$ FLOP/s per GPU (about 40% of an H100's dense bf16 peak) that is $1.5\times10^6$ GPU-seconds, about 420 GPU-hours. Compute-optimal for 1B parameters would be about 20B tokens, $1.2\times10^{20}$ FLOPs.
:::

::: qa
**What is the difference between DataParallel and DistributedDataParallel in PyTorch?**

`DataParallel` is one process driving several GPUs with threads: every iteration it scatters the batch, replicates the model from GPU 0, and gathers outputs and gradients back there, giving GIL contention, unbalanced memory and poor scaling, on one machine only. `DistributedDataParallel` runs one process per GPU with a persistent replica, all-reduces gradients in buckets during backward so communication overlaps computation, and works across nodes. PyTorch recommends DDP even on one machine.
:::

::: qa
**Explain ZeRO stages 1, 2 and 3 (or FSDP). What does each save and cost?**

ZeRO-1 shards the $12\Psi$ bytes of fp32 master weights and Adam states: gradients are reduce-scattered so each GPU gets the averaged gradient for its shard, updates it, and all-gathers the new weights, the same $2\Psi$ as an all-reduce. ZeRO-2 also shards the gradients, still $2\Psi$. ZeRO-3 (FSDP full sharding) also shards parameters, all-gathering each layer just in time in forward and backward, about $3\Psi$, usually hidden by prefetching. For 7.5B parameters on 64 GPUs that is 31.4, 16.6 and 1.9 GB per GPU instead of 120 GB.
:::

::: {.qa asked="NVIDIA SWE 2025, ByteDance MM 2025, Qualcomm"}
**Compare data, tensor and pipeline parallelism. How do DeepSpeed and Megatron-LM differ?**

**Data parallelism** replicates the model and splits the batch; gradients are all-reduced (about $2\Psi$ per GPU per step), and memory per GPU shrinks only if ZeRO or FSDP shard the states. **Tensor parallelism** (Megatron-LM) splits each layer's matrices, dividing weights, compute and in-layer activations at the cost of two blocking all-reduces forward and two backward per transformer layer, so it stays within a node. **Pipeline parallelism** splits layers into $K$ stages and the batch into $M$ micro-batches, with bubble $(K-1)/(M+K-1)$; 1F1B holds at most $K$ micro-batches' activations, and interleaving $v$ chunks per device cuts the bubble by $v$ (Narayanan et al., 2021). Only boundary activations cross GPUs, so pipelines span nodes. **DeepSpeed** is a library built around ZeRO (sharding, CPU/NVMe offload, few model changes); **Megatron-LM** is a transformer codebase with hand-written tensor-parallel layers, pipeline schedules and fused kernels, faster but requiring its model style. They are often combined: tensor parallelism within a node, pipeline across nodes, data parallelism over the rest.
:::

::: qa
**How do the Kaplan and Chinchilla scaling conclusions differ, and why?**

Kaplan et al. put extra compute mostly into model size ($N\propto C^{0.73}$); Chinchilla grows parameters and tokens about equally ($\approx C^{0.5}$ each, about 20 tokens per parameter), so many large models of the time were undertrained, and Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) at similar compute. The main explanation: Kaplan used one schedule length for all runs, so losses read partway through a long cosine schedule overstate properly scheduled shorter runs, biasing the fit toward bigger models. Neither accounts for inference cost.
:::

## 13. Debugging a training run

### A checklist before the long run

1. **Look at the data after preprocessing**, exactly as the model receives it: shapes, ranges, normalization, augmentations, label alignment.
2. **Check the initial loss.** A $K$-class classifier with small initial logits predicts nearly uniformly, so the loss should be near $\ln K$: 2.303 for 10 classes, 6.908 for 1000 (the loop below starts at 0.763 against $\ln 2 = 0.693$). Much higher means badly scaled logits; much lower suggests leakage.
3. **Overfit one small batch** to near-zero loss, which tests model, loss and optimizer together; if it fails, the bug is in the code.
4. **Run a learning-rate range test** (Smith, 2017): raise the rate exponentially over a few hundred steps and pick a peak somewhat below where the loss starts to rise.
5. **Check gradients reach every parameter**, with sensible per-layer norms, and that example $i$'s loss has zero gradient with respect to input $j \ne i$ (catches batch-mixing bugs).
6. **Check evaluation and resumption**: `eval()` mode, `no_grad`, the intended EMA or raw weights, and a resumed run that continues the same curve (save optimizer, scheduler, scaler and random states, not only weights).

| Symptom | Common causes | First checks |
|---|---|---|
| NaN or inf loss | fp16 overflow; learning rate too high; $\log 0$ or tiny denominators; a corrupt sample; an Adam coordinate with near-zero $v$ hit by a large gradient | find the first non-finite operation (anomaly detection, hooks); bf16; fp32 softmax, losses and norms; lower the rate; clip |
| Loss spikes | learning rate too high; $\beta_2$ so close to 1 that $v$ lags a jump in gradient scale; a bad batch; growing attention or output logits | gradient-norm log; lower the rate or $\beta_2$ (0.95 is common for LLMs); clip; skip the batch; QK-normalization or z-loss (Wortsman et al., 2023) |
| Flat loss from the start | rate far too small or large; dead ReLUs; misaligned labels; missing `optimizer.step()`; `detach` or `no_grad` cutting the graph; frozen parameters; softmax before a loss that expects logits | overfit one batch; per-layer gradient norms; `requires_grad` and the optimizer's parameter list |
| Training falls, validation does not | overfitting; train/eval mismatch; leakage; split shift | evaluate on training data in eval mode; compare preprocessing paths |

Large runs spike despite these checks: PaLM's largest run had about 20 spikes despite clipping, handled by restarting about 100 steps earlier and skipping roughly 200 to 500 batches (Chowdhery et al., 2022).

### The training loop, in the right order

```python
import math, torch

torch.manual_seed(0)
X = torch.randn(512, 20); y = (X[:, :5].sum(1) > 0).long()
loader = [(X[i:i + 64], y[i:i + 64]) for i in range(0, 512, 64)]
model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.GELU(), torch.nn.Linear(64, 2))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
total, warmup = 200, 20
def lr_factor(s):                                    # linear warmup, then cosine to zero
    if s < warmup:
        return (s + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

step = 0
model.train()
while step < total:
    for xb, yb in loader:
        opt.zero_grad(set_to_none=True)              # 1. clear old gradients
        loss = torch.nn.functional.cross_entropy(model(xb), yb)   # 2. forward
        loss.backward()                              # 3. backward
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 4. clip; log gnorm
        opt.step()                                   # 5. update
        sched.step()                                 # 6. advance the schedule, once per step
        if step == 0:
            print(f"initial loss {loss.item():.3f}, ln 2 = {math.log(2):.3f}")
        step += 1
        if step == total:
            break
model.eval()
with torch.no_grad():
    print("train accuracy", (model(X).argmax(1) == y).float().mean().item())
```

It prints an initial loss of 0.763 and a final training accuracy of 0.996. With fp16 autocast the middle becomes `scaler.scale(loss).backward()`, then `scaler.unscale_(opt)` before clipping (so the threshold applies to true gradients), then `scaler.step(opt)` and `scaler.update()`; bf16 needs no scaler.

::: {.qa asked="Meta GenAI, OpenAI RE, Luma RE (3 reports)"}
**Debug this PyTorch training script: the loss does not go down, or becomes NaN.**

Work one hypothesis at a time, out loud. First try to overfit a single small batch: if that fails, the bug is in the model or the loss, not the data or the optimizer; and check the initial loss against $\ln K$. Then the usual suspects: misaligned or unnormalized data and labels; broadcasting errors (a $(B, 1)$ tensor minus a $(B,)$ tensor silently becomes $(B, B)$); a softmax before a loss that expects logits, or over the wrong dimension; a missing `zero_grad` or `optimizer.step()`, or `zero_grad` between `backward` and `step`; the wrong `train()`/`eval()` mode; a stray `detach` or `no_grad`; parameters missing from the optimizer; an inverted attention mask; in-place operations that break autograd; learning-rate or warmup problems, including a scheduler stepped per micro-batch; and fp16 overflow. For NaNs, find the first non-finite operation with anomaly detection. Confirm each fix before moving on.
:::

::: qa
**Your loss goes to NaN after 20k steps. How do you debug it?**

Localize first: from logged per-layer gradient norms, activation statistics and the loss scale, find the first non-finite step, rewind to a checkpoint just before it, replay that batch, and use anomaly detection or forward hooks to find the first operation producing inf or NaN. Then check the usual causes: fp16 overflow (use bf16; keep softmax, losses and norms in fp32), a learning rate too high (a gradient-norm spike comes first), an Adam coordinate with near-zero $v$ hit by a large gradient, unsafe operations ($\log 0$, tiny denominators, an $\epsilon$ that rounds to zero in fp16), a corrupt sample, and transformer logit growth (QK-normalization, z-loss). Skipping bad steps buys time; fix the systematic cause.
:::

::: qa
**What sanity checks do you run before launching a long training job?**

Initial loss near $\ln K$. Overfit one small batch to near-zero loss. Inspect exactly what enters the network after preprocessing, including label alignment. Check that every parameter gets a nonzero gradient and that example $i$'s loss has zero gradient with respect to input $j \ne i$. Run a short learning-rate sweep or range test, verify that resuming from a checkpoint reproduces the loss curve, and confirm evaluation uses `eval()` mode and the intended EMA or raw weights.
:::

::: qa
**How do you make a run reproducible, and what remains non-deterministic?**

Seed Python, NumPy and PyTorch (CPU and CUDA) and the data loader's workers and sampler; enable deterministic kernels (`torch.use_deterministic_algorithms(True)`, deterministic cuDNN without benchmarking, the cuBLAS workspace setting); pin library versions; checkpoint optimizer, scheduler and random states. Bitwise equality is still not guaranteed across GPU types, library versions or worker counts: floating-point addition is not associative, so different reduction orders in all-reduces or atomic kernels change the last bits, and the differences grow over a run. Report variance across seeds.
:::

## 14. PyTorch from scratch

Three implementations interviewers ask for, each checked against PyTorch and run as shown.

### A manual backward pass for a two-layer MLP

This implements the derivation of Section 1 and the $p - y$ gradient of Section 2, then compares with autograd. Every line has a direct NumPy equivalent.

```python
import torch
torch.manual_seed(0)
B, D, H, C = 16, 5, 8, 3
X = torch.randn(B, D, dtype=torch.float64)
y = torch.randint(0, C, (B,))
W1 = torch.randn(D, H, dtype=torch.float64, requires_grad=True)
b1 = torch.zeros(H, dtype=torch.float64, requires_grad=True)
W2 = torch.randn(H, C, dtype=torch.float64, requires_grad=True)
b2 = torch.zeros(C, dtype=torch.float64, requires_grad=True)

# forward, keeping what backward needs
z1 = X @ W1 + b1                      # (B, H)
h = z1.clamp(min=0)                   # ReLU
z2 = h @ W2 + b2                      # (B, C) logits
loss = torch.nn.functional.cross_entropy(z2, y)   # mean over the batch

# manual backward (no autograd): every gradient has its variable's shape
with torch.no_grad():
    d2 = torch.softmax(z2, dim=1)
    d2[torch.arange(B), y] -= 1
    d2 /= B                           # dL/dz2 = (p - onehot) / B
    gW2, gb2 = h.T @ d2, d2.sum(0)    # (H, C), (C,)
    dz1 = (d2 @ W2.T) * (z1 > 0)      # back through W2, then the ReLU mask
    gW1, gb1 = X.T @ dz1, dz1.sum(0)  # (D, H), (H,)

loss.backward()                       # autograd, for comparison
for name, mine, ref in [("W1", gW1, W1.grad), ("b1", gb1, b1.grad),
                        ("W2", gW2, W2.grad), ("b2", gb2, b2.grad)]:
    print(name, (mine - ref).abs().max().item())
```

The largest differences are between $1.4\times10^{-17}$ and $5.6\times10^{-17}$, float64 rounding. Say aloud while writing it: the $1/B$ from the mean loss, bias gradients as column sums, the ReLU mask from $z_1$, and $W_2^\top$ for the input gradient.

### BatchNorm forward, in training and evaluation mode

```python
import torch
torch.manual_seed(0)

def bn_forward(x, gamma, beta, rmean, rvar, training, momentum=0.1, eps=1e-5):
    # x: (B, C); statistics per feature, over the batch axis
    if training:
        mu = x.mean(0)
        var = x.var(0, unbiased=False)            # biased variance to normalize
        B = x.shape[0]
        rmean.mul_(1 - momentum).add_(momentum * mu.detach())
        rvar.mul_(1 - momentum).add_(momentum * var.detach() * B / (B - 1))  # unbiased in the buffer
    else:
        mu, var = rmean, rvar
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta

C = 4
ref = torch.nn.BatchNorm1d(C)
with torch.no_grad():
    ref.weight.uniform_(0.5, 1.5); ref.bias.uniform_(-0.5, 0.5)
gamma, beta = ref.weight.detach().clone(), ref.bias.detach().clone()
rmean, rvar = torch.zeros(C), torch.ones(C)

for _ in range(3):                                # a few training steps
    x = 3 * torch.randn(32, C) + 1
    ref.train()
    print("train", (bn_forward(x, gamma, beta, rmean, rvar, True) - ref(x)).abs().max().item())
print("buffers", (rmean - ref.running_mean).abs().max().item(), (rvar - ref.running_var).abs().max().item())
x = 3 * torch.randn(5, C) + 1
ref.eval()
print("eval", (bn_forward(x, gamma, beta, rmean, rvar, False) - ref(x)).abs().max().item())
```

Outputs, buffers and evaluation-mode outputs agree with `nn.BatchNorm1d` to within $5\times10^{-7}$, float32 rounding. The details that matter: reduce over the batch axis (over $(B, H, W)$ for images, with statistics reshaped to $(1, C, 1, 1)$); normalize with the biased variance but store the unbiased one; update buffers without gradient; use the buffers in evaluation mode.

### One Adam step

```python
import torch
torch.manual_seed(0)

def adam_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m.mul_(b1).add_((1 - b1) * g)                 # first-moment EMA
    v.mul_(b2).add_((1 - b2) * g * g)             # second-moment EMA
    m_hat = m / (1 - b1 ** t)                     # bias corrections
    v_hat = v / (1 - b2 ** t)
    p.sub_(lr * m_hat / (v_hat.sqrt() + eps))

w_ref = torch.randn(10, requires_grad=True)
w = w_ref.detach().clone()
m, v = torch.zeros(10), torch.zeros(10)
opt = torch.optim.Adam([w_ref], lr=1e-2)
target = torch.randn(10)
for t in range(1, 6):
    opt.zero_grad()
    loss = ((w_ref - target) ** 4).sum()
    loss.backward()
    g = 4 * (w - target) ** 3                     # same gradient, by hand
    adam_step(w, g, m, v, t, lr=1e-2)
    opt.step()
    print(t, (w - w_ref.detach()).abs().max().item())
```

The parameters match `torch.optim.Adam` exactly for four steps and to $3\times10^{-8}$ at the fifth, a float32 rounding difference from a different order of operations. Note that $t$ starts at 1 (at $t = 0$ the bias correction divides by zero), and that AdamW would add `p.mul_(1 - lr * weight_decay)` before the update instead of adding `weight_decay * p` to `g`.

## Rapid-fire review

::: qa
**For $Y = XW + b$, what are $\partial\mathcal{L}/\partial W$, $\partial\mathcal{L}/\partial X$ and $\partial\mathcal{L}/\partial b$?**

$X^\top\delta_Y$, $\delta_Y W^\top$, and the column sums of $\delta_Y$. Check shapes: each gradient has the shape of its variable.
:::

::: qa
**How much does the backward pass cost relative to the forward pass?**

About twice, for networks dominated by matrix products: one product for the input gradient and one for the weight gradient. A training step is about three forward passes, hence $6N$ FLOPs per token.
:::

::: qa
**What should the initial loss of a $K$-class softmax classifier be?**

About $\ln K$ (2.303 for 10 classes, 6.908 for 1000), from near-uniform predictions. Much higher means badly scaled initial logits; much lower suggests leakage.
:::

::: qa
**What is the largest derivative of the sigmoid, and what does it imply?**

$1/4$, at 0. Ten sigmoid layers scale a gradient by at most $0.25^{10}\approx 10^{-6}$, before counting the weights.
:::

::: qa
**When does a ReLU "die"?**

When its pre-activation is negative for every input, so it outputs zero and receives zero gradient forever, often after a large update. Use a lower learning rate, He initialization, normalization, or Leaky ReLU, GELU or SiLU.
:::

::: qa
**Why not initialize all weights to zero or to the same value?**

Symmetry: identical units get identical gradients and stay identical, so a layer acts like one unit. Random initialization breaks the tie; biases can start at zero.
:::

::: qa
**He initialization's standard deviation for fan-in 512?**

$\sqrt{2/512} = 0.0625$. Xavier for a $512\times512$ layer gives $\sqrt{1/512} = 0.0442$.
:::

::: qa
**How many parameters does `BatchNorm2d(64)` have?**

128 learned ($\gamma$ and $\beta$, 64 each) plus 128 buffer values (running mean and variance), which are not trained.
:::

::: qa
**Why do convolutions followed by BatchNorm usually have `bias=False`?**

BatchNorm subtracts the per-channel mean, which removes any constant bias, and its $\beta$ provides the shift.
:::

::: qa
**What is the effective learning rate of SGD with momentum $\mu$?**

About $\eta/(1-\mu)$ along directions where gradients are consistent, so $10\eta$ for $\mu = 0.9$; oscillating components are damped to $1/(1+\mu)$.
:::

::: qa
**How large is Adam's first step?**

Exactly $\eta$ per coordinate, in the direction of $-\mathrm{sign}(g_1)$: bias correction makes $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$.
:::

::: qa
**What are Adam's defaults, and what do published LLM recipes often change?**

$\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Several LLM papers lower $\beta_2$; GPT-3 and LLaMA use 0.95 (with weight decay 0.1), which shortens $v$'s memory to about 20 steps so it tracks sudden changes in gradient scale.
:::

::: qa
**An EMA with decay 0.9999 averages roughly how many steps?**

About $10^4$, with a half-life near 6900 steps (0.999: about 1000 steps, half-life 693). Started from the initial weights, it is still 90% initialization after 1000 steps.
:::

::: qa
**Why multiply the distillation soft loss by $T^2$?**

Soft-target gradients scale roughly as $1/T^2$, so multiplying by $T^2$ keeps their weight relative to the hard-label loss stable as $T$ changes (Hinton et al., 2015).
:::

::: qa
**What are fp16's largest value and smallest positive value, and bf16's machine epsilon?**

65504 and $2^{-24}\approx 6\times10^{-8}$ (a subnormal); bf16's spacing at 1 is $2^{-7} = 0.0078$.
:::

::: qa
**Bytes per parameter for mixed-precision Adam training?**

16: weights 2, gradients 2, fp32 master copy 4, $m$ 4, $v$ 4. Activations come on top. A 1B model needs 16 GB, a 7B model 112 GB.
:::

::: qa
**Training FLOPs for $N$ parameters and $D$ tokens?**

About $6ND$: $2N$ per token forward, $4N$ backward.
:::

::: qa
**Compute-optimal tokens per parameter according to Chinchilla?**

About 20; Chinchilla itself was 70B parameters on 1.4T tokens.
:::

::: qa
**Linear scaling rule: base learning rate 0.1 at batch 256; what at batch 8192?**

$0.1\times 8192/256 = 3.2$, with a gradual warmup (Goyal et al., 2017).
:::

::: qa
**Clip gradients by norm or by value?**

Usually by global norm: it rescales the whole gradient and keeps its direction. Value clipping changes the direction coordinate by coordinate ($(3, 4)$ becomes $(1, 1)$).
:::

::: qa
**What does activation checkpointing trade?**

Compute for memory: about one extra forward pass (roughly 33% more compute) for activation memory that can drop to $O(\sqrt n)$ for $n$ layers.
:::

## References

1. Rumelhart, Hinton, Williams. Learning representations by back-propagating errors. Nature 323, 1986. [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0)
2. Baydin, Pearlmutter, Radul, Siskind. Automatic Differentiation in Machine Learning: a Survey. JMLR 2018. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
3. Glorot, Bengio. Understanding the difficulty of training deep feedforward neural networks. AISTATS 2010. [PMLR](https://proceedings.mlr.press/v9/glorot10a.html)
4. He, Zhang, Ren, Sun. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. ICCV 2015. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
5. He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
6. He, Zhang, Ren, Sun. Identity Mappings in Deep Residual Networks. ECCV 2016. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
7. Zhang, Dauphin, Ma. Fixup Initialization: Residual Learning Without Normalization. ICLR 2019. [arXiv:1901.09321](https://arxiv.org/abs/1901.09321)
8. Radford et al. Language Models are Unsupervised Multitask Learners (GPT-2). OpenAI, 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
9. Ioffe, Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
10. Ba, Kiros, Hinton. Layer Normalization. 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
11. Ulyanov, Vedaldi, Lempitsky. Instance Normalization: The Missing Ingredient for Fast Stylization. 2016. [arXiv:1607.08022](https://arxiv.org/abs/1607.08022)
12. Wu, He. Group Normalization. ECCV 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
13. Zhang, Sennrich. Root Mean Square Layer Normalization. NeurIPS 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
14. Santurkar, Tsipras, Ilyas, Madry. How Does Batch Normalization Help Optimization? NeurIPS 2018. [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
15. van Laarhoven. L2 Regularization versus Batch and Weight Normalization. 2017. [arXiv:1706.05350](https://arxiv.org/abs/1706.05350)
16. Zhang, Wang, Xu, Grosse. Three Mechanisms of Weight Decay Regularization. ICLR 2019. [arXiv:1810.12281](https://arxiv.org/abs/1810.12281)
17. Nair, Hinton. Rectified Linear Units Improve Restricted Boltzmann Machines. ICML 2010.
18. Hendrycks, Gimpel. Gaussian Error Linear Units (GELUs). 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
19. Elfwing, Uchibe, Doya. Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning. 2017. [arXiv:1702.03118](https://arxiv.org/abs/1702.03118)
20. Ramachandran, Zoph, Le. Searching for Activation Functions. 2017. [arXiv:1710.05941](https://arxiv.org/abs/1710.05941)
21. Dauphin, Fan, Auli, Grangier. Language Modeling with Gated Convolutional Networks. ICML 2017. [arXiv:1612.08083](https://arxiv.org/abs/1612.08083)
22. Shazeer. GLU Variants Improve Transformer. 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
23. Srivastava, Greff, Schmidhuber. Highway Networks. 2015. [arXiv:1505.00387](https://arxiv.org/abs/1505.00387)
24. Hu, Shen, Sun. Squeeze-and-Excitation Networks. CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
25. Hochreiter, Schmidhuber. Long Short-Term Memory. Neural Computation 9(8), 1997.
26. Pascanu, Mikolov, Bengio. On the difficulty of training Recurrent Neural Networks. ICML 2013. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
27. Jozefowicz, Zaremba, Sutskever. An Empirical Exploration of Recurrent Network Architectures. ICML 2015. [PMLR](https://proceedings.mlr.press/v37/jozefowicz15.html)
28. Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 2014. [JMLR](https://www.jmlr.org/papers/v15/srivastava14a.html)
29. Li, Chen, Hu, Yang. Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift. CVPR 2019. [arXiv:1801.05134](https://arxiv.org/abs/1801.05134)
30. Gal, Ghahramani. Dropout as a Bayesian Approximation. ICML 2016. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)
31. Szegedy, Vanhoucke, Ioffe, Shlens, Wojna. Rethinking the Inception Architecture for Computer Vision (label smoothing). CVPR 2016. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
32. Müller, Kornblith, Hinton. When Does Label Smoothing Help? NeurIPS 2019. [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)
33. Lin, Goyal, Girshick, He, Dollár. Focal Loss for Dense Object Detection. ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
34. Zhao, Gallo, Frosio, Kautz. Loss Functions for Image Restoration with Neural Networks. IEEE TCI 2017. [arXiv:1511.08861](https://arxiv.org/abs/1511.08861)
35. Cui, Jia, Lin, Song, Belongie. Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019. [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)
36. Menon et al. Long-tail learning via logit adjustment. ICLR 2021. [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
37. Kang et al. Decoupling Representation and Classifier for Long-Tailed Recognition. ICLR 2020. [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)
38. Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
39. Zhang, Cisse, Dauphin, Lopez-Paz. mixup: Beyond Empirical Risk Minimization. ICLR 2018. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
40. Yun et al. CutMix. ICCV 2019. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
41. Cubuk, Zoph, Shlens, Le. RandAugment. 2019. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
42. Lim, Son, Kim, Nah, Lee. Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR). CVPRW 2017. [arXiv:1707.02921](https://arxiv.org/abs/1707.02921)
43. Wang et al. ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. ECCVW 2018. [arXiv:1809.00219](https://arxiv.org/abs/1809.00219)
44. Liang et al. SwinIR: Image Restoration Using Swin Transformer. ICCVW 2021. [arXiv:2108.10257](https://arxiv.org/abs/2108.10257)
45. Goodfellow, Bengio, Courville. Deep Learning, Ch. 7 Regularization (Sec. 7.8 Early Stopping). MIT Press, 2016. [deeplearningbook.org](https://www.deeplearningbook.org/contents/regularization.html)
46. Belkin, Hsu, Ma, Mandal. Reconciling modern machine learning practice and the bias-variance trade-off. PNAS 2019. [arXiv:1812.11118](https://arxiv.org/abs/1812.11118)
47. Nakkiran et al. Deep Double Descent: Where Bigger Models and More Data Hurt. ICLR 2020. [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)
48. Nakkiran, Venkat, Kakade, Ma. Optimal Regularization Can Mitigate Double Descent. ICLR 2021. [arXiv:2003.01897](https://arxiv.org/abs/2003.01897)
49. Soudry et al. The Implicit Bias of Gradient Descent on Separable Data. JMLR 2018. [arXiv:1710.10345](https://arxiv.org/abs/1710.10345)
50. Barrett, Dherin. Implicit Gradient Regularization. ICLR 2021. [arXiv:2009.11162](https://arxiv.org/abs/2009.11162)
51. Smith, Dherin, Barrett, De. On the Origin of Implicit Regularization in Stochastic Gradient Descent. ICLR 2021. [arXiv:2101.12176](https://arxiv.org/abs/2101.12176)
52. Keskar et al. On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. ICLR 2017. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836)
53. Hoffer, Hubara, Soudry. Train longer, generalize better. NeurIPS 2017. [arXiv:1705.08741](https://arxiv.org/abs/1705.08741)
54. Zhang, Bengio, Hardt, Recht, Vinyals. Understanding deep learning requires rethinking generalization. ICLR 2017. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
55. Xu, Raginsky. Information-theoretic analysis of generalization capability of learning algorithms. NeurIPS 2017. [arXiv:1705.07809](https://arxiv.org/abs/1705.07809)
56. Shwartz-Ziv, Tishby. Opening the Black Box of Deep Neural Networks via Information. 2017. [arXiv:1703.00810](https://arxiv.org/abs/1703.00810)
57. Saxe et al. On the Information Bottleneck Theory of Deep Learning. ICLR 2018.
58. Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 1964.
59. Nesterov. A method for solving the convex programming problem with convergence rate $O(1/k^2)$. Soviet Mathematics Doklady, 1983.
60. Sutskever, Martens, Dahl, Hinton. On the importance of initialization and momentum in deep learning. ICML 2013. [PMLR](https://proceedings.mlr.press/v28/sutskever13.html)
61. Duchi, Hazan, Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. JMLR 2011. [JMLR](https://jmlr.org/papers/v12/duchi11a.html)
62. Tieleman, Hinton. Lecture 6.5, RMSProp. COURSERA: Neural Networks for Machine Learning, 2012.
63. Kingma, Ba. Adam: A Method for Stochastic Optimization. ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
64. Loshchilov, Hutter. Decoupled Weight Decay Regularization. ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
65. Wilson, Roelofs, Stern, Srebro, Recht. The Marginal Value of Adaptive Gradient Methods in Machine Learning. NeurIPS 2017. [arXiv:1705.08292](https://arxiv.org/abs/1705.08292)
66. Zhang et al. Why are Adaptive Methods Good for Attention Models? NeurIPS 2020. [arXiv:1912.03194](https://arxiv.org/abs/1912.03194)
67. Kunstner et al. Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be. ICLR 2023. [arXiv:2304.13960](https://arxiv.org/abs/2304.13960)
68. Kunstner et al. Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models. 2024. [arXiv:2402.19449](https://arxiv.org/abs/2402.19449)
69. Zhang et al. Why Transformers Need Adam: A Hessian Perspective. 2024. [arXiv:2402.16788](https://arxiv.org/abs/2402.16788)
70. Chen et al. Symbolic Discovery of Optimization Algorithms (Lion). 2023. [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)
71. Gupta, Koren, Singer. Shampoo: Preconditioned Stochastic Tensor Optimization. ICML 2018. [arXiv:1802.09568](https://arxiv.org/abs/1802.09568)
72. Shazeer, Stern. Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. 2018. [arXiv:1804.04235](https://arxiv.org/abs/1804.04235)
73. Jordan et al. Muon: An optimizer for hidden layers in neural networks. Blog, 2024. [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
74. Liu et al. Muon is Scalable for LLM Training. 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
75. Wortsman et al. Small-scale proxies for large-scale Transformer training instabilities. 2023. [arXiv:2309.14322](https://arxiv.org/abs/2309.14322)
76. Goyal et al. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. 2017. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
77. Krizhevsky. One weird trick for parallelizing convolutional neural networks. 2014. [arXiv:1404.5997](https://arxiv.org/abs/1404.5997)
78. Malladi, Lyu, Panigrahi, Arora. On the SDEs and Scaling Rules for Adaptive Gradient Algorithms. NeurIPS 2022. [arXiv:2205.10287](https://arxiv.org/abs/2205.10287)
79. McCandlish, Kaplan, Amodei et al. An Empirical Model of Large-Batch Training. 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
80. Liu et al. On the Variance of the Adaptive Learning Rate and Beyond (RAdam). ICLR 2020. [arXiv:1908.03265](https://arxiv.org/abs/1908.03265)
81. Kalra, Barkeshli. Why Warmup the Learning Rate? Underlying Mechanisms and Improvements. 2024. [arXiv:2406.09405](https://arxiv.org/abs/2406.09405)
82. Xiong et al. On Layer Normalization in the Transformer Architecture. ICML 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
83. Loshchilov, Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
84. Hu et al. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. 2024. [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)
85. Hägele et al. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. 2024. [arXiv:2405.18392](https://arxiv.org/abs/2405.18392)
86. Smith. Cyclical Learning Rates for Training Neural Networks. WACV 2017. [arXiv:1506.01186](https://arxiv.org/abs/1506.01186)
87. Micikevicius et al. Mixed Precision Training. ICLR 2018. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
88. Kalamkar et al. A Study of BFLOAT16 for Deep Learning Training. 2019. [arXiv:1905.12322](https://arxiv.org/abs/1905.12322)
89. Polyak, Juditsky. Acceleration of Stochastic Approximation by Averaging. SIAM Journal on Control and Optimization, 1992.
90. Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
91. Song, Ermon. Improved Techniques for Training Score-Based Generative Models. NeurIPS 2020. [arXiv:2006.09011](https://arxiv.org/abs/2006.09011)
92. Karras et al. Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2). CVPR 2024. [arXiv:2312.02696](https://arxiv.org/abs/2312.02696)
93. He, Fan, Wu, Xie, Girshick. Momentum Contrast for Unsupervised Visual Representation Learning (MoCo). CVPR 2020. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
94. Grill et al. Bootstrap Your Own Latent (BYOL). NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
95. Peebles, Xie. Scalable Diffusion Models with Transformers (DiT). ICCV 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
96. Zhang, Rao, Agrawala. Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet). ICCV 2023. [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
97. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
98. Alayrac et al. Flamingo: a Visual Language Model for Few-Shot Learning. NeurIPS 2022. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
99. Rajbhandari, Rasley, Ruwase, He. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. SC 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
100. Li et al. PyTorch Distributed: Experiences on Accelerating Data Parallel Training. VLDB 2020. [arXiv:2006.15704](https://arxiv.org/abs/2006.15704)
101. Zhao et al. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. VLDB 2023. [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)
102. Shoeybi et al. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. 2019. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
103. Huang et al. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. NeurIPS 2019. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
104. Narayanan et al. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. SC 2021. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
105. Chen, Xu, Zhang, Guestrin. Training Deep Nets with Sublinear Memory Cost. 2016. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
106. Korthikanti et al. Reducing Activation Recomputation in Large Transformer Models. 2022. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
107. Kaplan et al. Scaling Laws for Neural Language Models. 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
108. Hoffmann et al. Training Compute-Optimal Large Language Models (Chinchilla). NeurIPS 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
109. Brown et al. Language Models are Few-Shot Learners (GPT-3). NeurIPS 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
110. Touvron et al. LLaMA: Open and Efficient Foundation Language Models. 2023. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
111. Chowdhery et al. PaLM: Scaling Language Modeling with Pathways. 2022. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
