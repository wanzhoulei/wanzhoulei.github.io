---
title: "Convolutional and Recurrent Networks"
subtitle: "A guided reading of convolution (its forward pass, backward pass and cost), the classic CNNs from LeNet to U-Net, and recurrent networks from the vanilla RNN through LSTM, GRU and sequence-to-sequence models with attention."
order: 0.62
updated: 2026-11-15
---

Convolutional and recurrent networks are the two architectures that build a symmetry of the data into the layer itself. A convolution reuses one small filter at every position of an image, because a pattern means the same thing wherever it appears. A recurrent network reuses one update rule at every time step, because the rule for reading the next word does not depend on where in the sentence you are. Transformers have replaced both in many large models. Interviews still test them, though, for three reasons. First, they are the cleanest place to ask counting questions: output sizes, parameters, multiply-adds and receptive fields, done by hand. Second, they carry the standard derivations: the backward pass of a convolution, backpropagation through time, and why an LSTM's cell state keeps gradients alive. Third, they are still in use. The U-Net that denoises in most diffusion models is a CNN, super-resolution networks upsample with sub-pixel convolutions, perceptual losses and LPIPS read features from VGG and AlexNet, and state-space models such as Mamba are recurrent networks again.

**Roadmap.** "Why convolution" starts from a dense layer and shows what locality and weight sharing each save, proves translation equivariance, and settles the convolution-versus-correlation question. "The convolution layer in detail" covers the forward formula, output sizes, parameters and compute, receptive fields, $1\times1$, grouped, depthwise and dilated convolutions, upsampling layers and pooling. "Backward pass of convolution" derives both gradients, works a tiny example by hand, and explains im2col, FFT and Winograd convolution. "Classic CNN architectures" goes through LeNet, AlexNet, VGG, GoogLeNet, ResNet, DenseNet, MobileNet, EfficientNet, ConvNeXt and U-Net, each with its main idea and checked numbers, including a layer-by-layer parameter count of ResNet-50. "Recurrent networks" derives backpropagation through time and the vanishing and exploding gradient argument. "LSTM and GRU" gives the gate equations, the Jacobian of the cell state and parameter counts. "Sequence-to-sequence" covers encoder–decoder models, teacher forcing, beam search and Bahdanau attention, and ends where [transformers](transformers.html) begins, with a short note on state-space models. Two short from-scratch implementations, a rapid-fire review and references close the page. Backpropagation in general, initialization and normalization live in [deep learning training](deep-learning-training.html); vision transformers and self-supervised features in [vision backbones](vision-backbones.html).

## Notation

All tensors use PyTorch's layout: channels first, batch first.

| Symbol | Meaning |
|---|---|
| $x$, $y$ | input and output of a layer; $x[n]$ in 1-D, $x[c, i, j]$ for channel $c$ at row $i$, column $j$ |
| $w$, $b$ | convolution kernel and bias; $w[o, c, u, v]$ has shape $(C_\text{out}, C_\text{in}, k, k)$ |
| $C_\text{in}$, $C_\text{out}$ | number of input and output channels |
| $H \times W$, $H_\text{out} \times W_\text{out}$ | spatial size of the input and output (in conv contexts, $W$ is a width) |
| $n$, $n_\text{out}$ | length of one spatial side, in and out |
| $k$ | kernel size (square $k \times k$ unless stated) |
| $p$, $s$, $d$ | padding per side, stride, dilation |
| $G$ | number of groups in a grouped convolution |
| $r_\ell$, $j_\ell$ | receptive-field size and jump (input-pixel spacing of neighbouring units) after layer $\ell$ |
| $\star$, $*$ | cross-correlation (what deep-learning "convolution" layers compute) and true convolution |
| $\mathcal{L}$ | scalar loss |
| $\delta = \partial \mathcal{L}/\partial y$ | upstream gradient arriving at a layer's output |
| MAC | one multiply-accumulate; 1 MAC $\approx$ 2 FLOPs |
| $x_t \in \mathbb{R}^X$, $h_t \in \mathbb{R}^H$ | RNN input and hidden state at time $t$; $X$ and $H$ are their sizes |
| $W_{xh}$, $W_{hh}$, $b_h$ | input-to-hidden ($H \times X$) and hidden-to-hidden ($H \times H$) weights, hidden bias |
| $a_t$ | RNN pre-activation, $h_t = \tanh(a_t)$ |
| $T$ | sequence length |
| $i_t, f_t, o_t$ | LSTM input, forget and output gates, each in $(0,1)^H$ |
| $g_t$, $c_t$ | LSTM candidate values in $(-1,1)^H$ and cell state |
| $z_t$, $r_t$ | GRU update and reset gates |
| $\sigma$, $\odot$ | logistic sigmoid; elementwise product |
| $\lVert A \rVert_2$, $\rho(A)$ | largest singular value; spectral radius (largest eigenvalue magnitude) |

## Why convolution

### What locality and weight sharing each save

Take a $28 \times 28$ grayscale image and ask for 8 output feature maps of the same size. Three ways to build that layer:

| Layer | Each output sees | Weights | MACs |
|---|---|---|---|
| Dense (fully connected) | all 784 input pixels | $784 \times 6{,}272 = 4{,}917{,}248$ | 4,917,248 |
| Locally connected, $3 \times 3$ | its $3 \times 3$ neighbourhood, own weights | $6{,}272 \times 9 = 56{,}448$ | 56,448 |
| Convolution, $3 \times 3$ | its $3 \times 3$ neighbourhood, shared weights | $8 \times 9 = 72$ | 56,448 |

Here $6{,}272 = 28 \cdot 28 \cdot 8$ is the number of output values, and biases are left out (6,272 for the first two layers, 8 for the convolution). Reading the table row by row:

- **Locality** (each output looks only at nearby inputs) cuts the weights by $784/9 \approx 87$. It also cuts the compute by the same factor, because a dense layer does one MAC per weight.
- **Weight sharing** (every position uses the same 9 numbers per output channel) cuts the weights by another factor of 784, the number of positions, but leaves the compute unchanged: the convolution still does 9 MACs per output value, $784 \times 72 = 56{,}448$ in total.

So sharing saves parameters, and with them the data needed to fit them, while locality saves compute. Both are assumptions about images. Locality says that nearby pixels are strongly related and distant ones weakly, so a small window is enough at each layer; stacking layers widens the view (the receptive field, below). Sharing says the statistics of images are roughly the same everywhere (stationarity): an edge detector that works in the top-left corner should also work in the bottom-right.

### Translation equivariance, proved in 1-D

Let $(S_\tau x)[n] = x[n - \tau]$ shift a signal right by $\tau$ samples. A 1-D convolution layer (as deep-learning libraries define it) computes

$$
y[n] = (w \star x)[n] = \sum_{u=0}^{k-1} w[u]\, x[n+u].
$$

Assume the signal is infinite, or that indices wrap around (circular padding), so that no window falls off an edge. Apply the layer to the shifted input:

$$
\begin{aligned}
(w \star S_\tau x)[n] &= \sum_{u=0}^{k-1} w[u]\, (S_\tau x)[n+u] && \text{definition of the layer} \\
&= \sum_{u=0}^{k-1} w[u]\, x[n+u-\tau] && \text{definition of } S_\tau \\
&= y[n-\tau] && \text{the same sum, evaluated at } n-\tau \\
&= (S_\tau y)[n] && \text{definition of } S_\tau .
\end{aligned}
$$

So $f(S_\tau x) = S_\tau f(x)$: shifting the input shifts the output by the same amount. This property is called **equivariance**. A pointwise nonlinearity also commutes with shifts, since $\phi(x[n-\tau])$ is the shifted $\phi(x[n])$, and a composition of shift-commuting maps commutes with shifts. So a stack of stride-1 convolutions and activations is equivariant.

**Invariance** is different: $f(S_\tau x) = f(x)$. A classifier wants invariance, and gets it from a global pooling at the end, because $\sum_n (S_\tau y)[n] = \sum_n y[n]$ on a circular signal.

**The converse is also true, and it explains why convolution is the natural layer.** Suppose $L$ is any linear map that commutes with all shifts. Write the input as a sum of shifted impulses, $x = \sum_m x[m]\, S_m e_0$, where $e_0$ is 1 at index 0 and 0 elsewhere. Then

$$
L x = \sum_m x[m]\, L S_m e_0 = \sum_m x[m]\, S_m (L e_0),
$$

using linearity and then shift-commutation. With $h = L e_0$ (the "impulse response"), the output at $n$ is $\sum_m x[m]\, h[n-m]$, a convolution. So the linear layers that respect translation are exactly the convolutions. This is the linear time-invariant system theorem from signal processing.

**Where real CNNs break equivariance.**

- *Borders.* With zero padding, windows near an edge see zeros, so the network can tell where the edge is. CNNs do encode absolute position this way (Islam et al., 2020).
- *Stride and pooling.* Subsampling by $s$ keeps only every $s$-th output, so the network is equivariant only to shifts by multiples of $s$, and by multiples of the total stride for the whole network (32 for ResNet-50). A one-pixel example: subsampling $x = [1, 0, 0, 0]$ at even indices gives $[1, 0]$; shifting first gives $[0, 1, 0, 0]$, which subsamples to $[0, 0]$, which is not a shifted copy of $[1, 0]$. Azulay and Weiss (2019) showed that shifting an image by one pixel can change a modern CNN's prediction, and Zhang (2019) restored much of the lost stability by blurring before every subsampling step, the anti-aliasing a signal-processing engineer would add.
- *Other transformations.* A CNN is not equivariant to rotation or scale. Data augmentation teaches approximate invariance; group-equivariant CNNs (Cohen and Welling, 2016) build some rotations in.

### Cross-correlation versus convolution

In mathematics, convolution flips the kernel:

$$
(w * x)[n] = \sum_{u} w[u]\, x[n-u], \qquad (w \star x)[n] = \sum_u w[u]\, x[n+u].
$$

The second is cross-correlation, and it is what `nn.Conv2d`, `F.conv2d` and every major framework compute. The two are related by a flip: $w \star x = \tilde w * x$ with $\tilde w[u] = w[-u]$. Since the kernel is learned, a network trained with correlation simply learns the flipped kernel of the network trained with convolution, and nothing about its accuracy changes. Small example: with $x = [1, 2, 3]$ and $w = [1, -1]$, correlation gives $x[n] - x[n+1]$, that is $[-1, -1]$, while convolution gives $x[n] - x[n-1]$ on the valid range, that is $[1, 1]$.

The difference matters in three places:

- copying a hand-designed filter (a Sobel kernel from a textbook, or `scipy.signal.convolve`) into a conv layer, which needs a flip, or else the output changes sign or direction;
- algebra: true convolution is commutative and associative, $a * (b * c) = (a * b) * c$, so two stacked linear convolutions equal one convolution with $a * b$, while correlation is not commutative;
- the backward pass below, where the input gradient is naturally a true convolution, or equivalently a correlation with the flipped kernel.

::: key
**How to say it in an interview.**

A convolution layer is a dense layer with two constraints: each output looks only at a small window (locality), and every position uses the same weights (sharing). Locality saves compute; sharing saves parameters. Together they make the layer translation equivariant, and conversely every linear shift-equivariant layer is a convolution. Stride, pooling and zero padding break exact equivariance, and global pooling at the end turns equivariance into invariance.
:::

::: qa
**Prove that a convolution layer is translation equivariant. When does a real CNN stop being equivariant?**

With $y[n] = \sum_u w[u] x[n+u]$ and $(S_\tau x)[n] = x[n-\tau]$, the layer applied to $S_\tau x$ gives $\sum_u w[u] x[n+u-\tau] = y[n-\tau]$, the shifted output. Pointwise activations commute with shifts too, so the whole stride-1 stack is equivariant, on infinite or circular signals. It breaks with stride and pooling (only shifts by multiples of the total stride are respected, and aliasing makes outputs jump under one-pixel shifts), with zero padding at borders (which leaks absolute position), and for rotations and scalings, which convolution does not handle at all.
:::

::: qa
**A dense layer maps a $28\times28$ image to eight $28\times28$ maps. How many weights does it have, and how many would a $3\times3$ convolution have? Does the convolution also save compute by the same factor?**

Dense: $784 \times 6{,}272 = 4{,}917{,}248$ weights. Convolution: $8 \times 9 = 72$. The compute does not fall by the same factor: the convolution does 9 MACs per output value, 56,448 in total, against 4,917,248 for the dense layer. That is an 87-fold saving from locality, whereas weight sharing alone saves parameters (a factor of 784) but no compute.
:::

::: {.qa asked="Google intern, Ant, Alibaba Amap (3 reports)"}
**What are the drawbacks of ViTs compared with CNNs, and vice versa?**

CNNs build in locality and translation equivariance, so they learn from less data, cost time linear in the number of pixels, and form multi-scale feature hierarchies naturally through pooling and stride. ViTs lack these priors: they need more data, strong augmentation or distillation (DeiT), or large-scale pretraining; global attention is quadratic in the number of tokens, which limits resolution; large patches give a single coarse feature map, a handicap for dense prediction; and position embeddings must be interpolated for new resolutions. In return, ViTs have a global receptive field from the first layer, keep improving as data and compute grow, and use the same token interface as text, which simplifies multimodal models. Hybrids sit in between: Swin uses windowed attention in a hierarchy, and ConvNeXt shows a CNN modernized with the transformer's training recipe and design choices can match Swin. Details are on [vision backbones](vision-backbones.html).
:::

## The convolution layer in detail

### The multi-channel forward pass

An input with $C_\text{in}$ channels (3 for RGB, 64 or more inside a network) is filtered by a kernel that spans all of them. With padding $p$, stride $s$ and dilation $d$, output channel $o$ at position $(i, j)$ is

$$
y[o, i, j] = b[o] + \sum_{c=0}^{C_\text{in}-1} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} w[o, c, u, v]\; x_\text{pad}[c,\ s i + d u,\ s j + d v],
$$

where $x_\text{pad}$ is the input with $p$ zeros added on each side, so $x_\text{pad}[c, a, b] = x[c, a - p, b - p]$ inside the image and 0 outside. Three readings of this formula are useful.

1. *Per output channel:* $y[o] = b[o] + \sum_c w[o, c] \star x[c]$. Each output map is a sum over input channels of single-channel 2-D correlations, each with its own $k \times k$ filter. People often forget the sum over $c$, and so undercount parameters by a factor of $C_\text{in}$.
2. *Per output position:* $y[\cdot, i, j]$ is a dense layer of shape $C_\text{out} \times C_\text{in}k^2$ applied to the $C_\text{in}k^2$ numbers in the window at $(i, j)$. This is the im2col view used in the backward-pass section.
3. *Taps:* $s$ moves the window $s$ pixels per output step; $d$ spaces the kernel's taps $d$ pixels apart, so a kernel of $k$ taps spans $d(k-1)+1$ pixels.

With a batch, add an index $b$ to $x$ and $y$; the weights do not depend on it.

### Output size

**Derivation (one spatial axis).** The padded input has $n + 2p$ positions, indexed $0, \dots, n + 2p - 1$. A window whose first tap is at position $a$ has its last tap at $a + d(k-1)$, so it fits if

$$
a + d(k-1) \le n + 2p - 1 .
$$

Windows start at $a = s \cdot i$ for $i = 0, 1, 2, \dots$, so the largest valid $i$ is $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor$, and counting $i = 0$ as well,

$$
n_\text{out} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1 .
$$

With $d = 1$ this is the familiar $\lfloor (n + 2p - k)/s \rfloor + 1$. Worked cases, all checked against `F.conv2d`:

| $n$ | $k$ | $p$ | $s$ | $d$ | $n_\text{out}$ | Comment |
|---|---|---|---|---|---|---|
| 32 | 3 | 1 | 1 | 1 | 32 | "same" padding, $p = (k-1)/2$ |
| 32 | 5 | 2 | 1 | 1 | 32 | same rule for $k = 5$ |
| 32 | 3 | 1 | 2 | 1 | 16 | the usual halving layer |
| 31 | 3 | 1 | 2 | 1 | 16 | odd sizes round up: $\lceil 31/2 \rceil$ |
| 6 | 3 | 0 | 2 | 1 | 2 | windows start at 0 and 2; pixel 5 is never read |
| 224 | 7 | 3 | 2 | 1 | 112 | ResNet stem |
| 227 | 11 | 0 | 4 | 1 | 55 | AlexNet's first layer |
| 5 | 3 | 0 | 1 | 2 | 1 | dilation 2 makes a $3$-tap kernel span 5 |

For stride 1, the output keeps the input size exactly when $2p = d(k-1)$, which needs an odd kernel when $d = 1$. That is one reason kernels are odd: an odd kernel has a center tap, so "same" padding is symmetric and outputs stay aligned with inputs.

::: pitfall
The floor hides mistakes. AlexNet's paper says its inputs are $224 \times 224$ and its first layer has $11 \times 11$ kernels with stride 4 and 55 outputs per side, but $(224 - 11)/4 + 1 = 54.25$ is not an integer; the arithmetic works for 227 (or 224 with padding). Frameworks silently floor the 54.25 to 54. When a size does not divide evenly, some border pixels are never read, so check the division, not just the final number. Also, TensorFlow's `padding="same"` with stride $s$ gives $\lceil n/s \rceil$ and may pad unequally on the two sides, which does not match PyTorch's symmetric padding.
:::

### Parameters and compute

Each output value is one dot product of length $C_\text{in}k^2/G$ (with $G$ groups, below; $G = 1$ normally), so

$$
\#\text{params} = C_\text{out}\cdot\frac{C_\text{in}k^2}{G} + C_\text{out}, \qquad
\text{MACs} = H_\text{out}W_\text{out}\, C_\text{out}\cdot\frac{C_\text{in}k^2}{G}.
$$

Ignoring the bias, MACs $=$ weights $\times\, H_\text{out}W_\text{out}$: every weight is used once per output position. Parameters do not depend on image size; compute grows linearly with the number of output pixels.

**Worked: the first two layers of VGG-16** at $224 \times 224$, both $3 \times 3$ with padding 1:

| Layer | Channels | Weights + biases | MACs |
|---|---|---|---|
| conv1_1 | $3 \to 64$ | $27 \cdot 64 + 64 = 1{,}792$ | $224^2 \cdot 64 \cdot 27 = 86.7$M |
| conv1_2 | $64 \to 64$ | $576 \cdot 64 + 64 = 36{,}928$ | $224^2 \cdot 64 \cdot 576 = 1.85$G |

The second layer alone costs about as much as a whole ResNet-18 forward pass (1.8G MACs). Its output is $224 \cdot 224 \cdot 64 = 3.2$M numbers, 12.8 MB in fp32 for a single image, which must be stored for the backward pass. The pattern is general: early, high-resolution layers have few parameters but most of the compute and activation memory, while late layers and fully connected heads hold most of the parameters but little compute. In VGG-16 the three fully connected layers hold 123.6M of the 138.4M parameters (89%), but only 0.12G of the roughly 15.5G MACs (0.8%).

::: pitfall
Papers mean different things by "FLOPs". The ResNet paper's "3.8 billion FLOPs" for ResNet-50 are multiply-adds, while hardware FLOP counts treat a multiply-add as 2 operations. State which one you count. Also, a convolution directly followed by BatchNorm needs no bias: BN subtracts the per-channel mean, which removes any constant, and its own shift $\beta$ takes the bias's role.
:::

::: {.qa asked="Google RI, ByteDance, Huawei, SenseTime (4+ reports)"}
**How many parameters and FLOPs does a convolutional layer have? What do 1x1 and depthwise-separable convolutions save?**

A $k \times k$ convolution from $C_\text{in}$ to $C_\text{out}$ channels has $k^2 C_\text{in} C_\text{out}$ weights plus $C_\text{out}$ biases, independent of image size, and costs $k^2 C_\text{in} C_\text{out} H_\text{out} W_\text{out}$ multiply-adds (about twice that in FLOPs), because each output value is a dot product over a $k \times k \times C_\text{in}$ window. Example: $3\times3$, 64 to 128 channels, $56 \times 56$ output: 73,728 weights and 231M MACs. A $1\times1$ convolution is a per-pixel linear map across channels with $C_\text{in}C_\text{out}$ weights, the cheap way to change width; inserting a $1\times1$ reduction before an expensive $3\times3$ or $5\times5$ convolution (a bottleneck) cuts cost by roughly the reduction factor. A depthwise-separable convolution replaces the $k \times k$ convolution by a per-channel $k\times k$ filter ($k^2 C_\text{in}$ weights) plus a $1\times1$ mix ($C_\text{in}C_\text{out}$), a cost ratio of $1/C_\text{out} + 1/k^2$, so 8 to 9 times cheaper for $3\times3$ kernels. On GPUs the speed-up is smaller than the FLOP count suggests, because depthwise convolutions do few operations per byte of memory traffic.
:::

### Receptive field

The receptive field (RF) of a unit is the set of input pixels that can influence it. Two quantities are tracked layer by layer: the RF size $r_\ell$, and the jump $j_\ell$, the distance in input pixels between two neighbouring units of layer $\ell$. Start with $r_0 = j_0 = 1$ (an input pixel sees itself, and neighbours are 1 apart).

**Derivation.** A unit of layer $\ell$ reads $k_\ell$ units of layer $\ell - 1$ whose positions are $d_\ell$ apart, so the first and last of them are $d_\ell(k_\ell - 1)$ units apart, which is $d_\ell(k_\ell-1)j_{\ell-1}$ input pixels. Each of them sees $r_{\ell-1}$ input pixels. The union runs from the start of the first one's field to the end of the last one's:

$$
r_\ell = r_{\ell-1} + d_\ell (k_\ell - 1)\, j_{\ell-1}, \qquad j_\ell = j_{\ell-1}\, s_\ell .
$$

The jump multiplies by each stride because moving one unit in layer $\ell$ moves $s_\ell$ units in layer $\ell - 1$. Unrolling gives the closed form

$$
r_L = 1 + \sum_{\ell=1}^{L} d_\ell(k_\ell - 1)\prod_{m<\ell} s_m ,
$$

so every stride multiplies what all later layers add. Pooling layers count as layers with their own $k$ and $s$.

**Worked example.** A small 1-D stack, all without dilation:

| Layer | $k$ | $s$ | $r_\ell = r_{\ell-1} + (k-1)j_{\ell-1}$ | $j_\ell$ |
|---|---|---|---|---|
| input | | | 1 | 1 |
| conv | 3 | 1 | $1 + 2 \cdot 1 = 3$ | 1 |
| conv | 3 | 2 | $3 + 2 \cdot 1 = 5$ | 2 |
| conv | 3 | 1 | $5 + 2 \cdot 2 = 9$ | 2 |
| pool | 2 | 2 | $9 + 1 \cdot 2 = 11$ | 4 |
| conv | 3 | 1 | $11 + 2 \cdot 4 = 19$ | 4 |

The last $3 \times 3$ conv adds 8 pixels, four times what the first one added, because it sits after a total stride of 4. To check this, build the stack with all-ones kernels, backpropagate from one central output unit, and count the input positions with nonzero gradient: there are 19. Other useful values from the same recursion: two stacked $3\times3$ convolutions see 5 pixels, three see 7, and three $3\times3$ convolutions with dilations 1, 2, 4 see $1 + 2(1 + 2 + 4) = 15$.

**Theoretical versus effective RF.** The recursion gives the region that *can* matter. How much each pixel actually matters is measured by the gradient of the unit with respect to the input. Luo et al. (2016) found this effective receptive field is roughly Gaussian and grows like the square root of depth, so it covers only a fraction of the theoretical RF. Practical consequence: for tasks that need global context (segmentation of large objects, denoising at high noise levels), networks add downsampling, dilation or attention rather than relying on depth alone. More on this, with the ResNet stem's RF, is on [vision backbones](vision-backbones.html).

::: qa
**Compute the receptive field of conv3–conv3(stride 2)–conv3–maxpool2(stride 2)–conv3.**

Use $r_\ell = r_{\ell-1} + (k_\ell - 1)j_{\ell-1}$ and $j_\ell = j_{\ell-1}s_\ell$ from $r = j = 1$: $r = 3, 5, 9, 11, 19$ with jumps $1, 2, 2, 4, 4$. The final unit sees a 19-pixel window (a $19 \times 19$ patch in 2-D). The lesson is that layers after a stride add more: the last conv adds $2 \times 4 = 8$ pixels. Mention that the effective RF, measured by gradients, is Gaussian-shaped and much smaller.
:::

### 1x1, grouped and depthwise convolutions

**$1 \times 1$ convolution.** With $k = 1$ the forward formula becomes $y[o, i, j] = b[o] + \sum_c w[o, c]\, x[c, i, j]$: the same $C_\text{out} \times C_\text{in}$ matrix applied at every pixel, i.e. a dense layer over channels, shared across positions. It changes the channel count cheaply ($C_\text{in}C_\text{out}$ weights), mixes channels after a depthwise convolution, implements projection shortcuts in ResNets, and adds a nonlinearity per pixel when followed by an activation (the "Network in Network" idea of Lin et al., 2013).

**Grouped convolution.** Split the $C_\text{in}$ input channels into $G$ groups and the $C_\text{out}$ output channels into $G$ groups; output group $g$ sees only input group $g$. Each filter now spans $C_\text{in}/G$ channels, so parameters and MACs both fall by a factor of $G$. Example: a $3\times3$ convolution from 256 to 256 channels has $589{,}824$ weights; with $G = 32$ it has $18{,}432$. AlexNet used $G = 2$ because the model was split over two GPUs; ResNeXt (Xie et al., 2017) made $G = 32$ a design choice ("cardinality").

**Depthwise convolution** is the extreme $G = C_\text{in} = C_\text{out}$: each channel is filtered by its own $k \times k$ kernel, $k^2 C$ weights, with no mixing across channels. **Depthwise-separable convolution** follows it with a $1\times1$ convolution that does all the channel mixing. Cost ratio against a standard convolution, per output pixel:

$$
\frac{\overbrace{k^2 C_\text{in}}^{\text{depthwise}} + \overbrace{C_\text{in} C_\text{out}}^{1\times1}}{k^2 C_\text{in} C_\text{out}}
= \frac{1}{C_\text{out}} + \frac{1}{k^2}.
$$

The same ratio holds for parameters (without biases). With $k = 3$ and $C_\text{out} = 256$ it is $0.0039 + 0.1111 = 0.115$, an 8.7-fold saving; as $C_\text{out}$ grows the saving approaches $k^2 = 9$. The MobileNet paper works one layer in its Table 3, which you can reproduce: input $14 \times 14 \times 512$, $3\times3$ kernels, 512 outputs.

| Version | MACs | Parameters |
|---|---|---|
| Standard convolution | $14^2 \cdot 512 \cdot 512 \cdot 9 = 462$M | $9 \cdot 512^2 = 2.36$M |
| Depthwise separable | $14^2 \cdot 512 \cdot 9 + 14^2 \cdot 512^2 = 52.3$M | $9 \cdot 512 + 512^2 = 0.27$M |
| Width multiplier 0.75 (384 channels) | 29.6M | 0.15M |
| and resolution $10\times10$ | 15.1M | 0.15M |

The ratio $462/52.3 = 8.8$ matches the formula. Note that after the factorization almost all the cost sits in the $1\times1$ part ($512^2$ against $9 \cdot 512$ per pixel).

**Spatial factorizations** work the same way. A $k \times k$ kernel can be split into $k \times 1$ followed by $1 \times k$, $2k$ instead of $k^2$ weights per channel pair ($14$ against $49$ for $7\times7$), as in Inception-v3 (Szegedy et al., 2016); this is exact only for rank-1 kernels, but stacked with a nonlinearity in between it is a different, cheaper layer that works well. Video models factor a $k\times k\times k$ 3-D kernel into a $k \times k$ spatial convolution and a $k$-tap temporal one, the "(2+1)D" convolution: $k^2C^2 + kC^2$ against $k^3C^2$ weights, $12$ against $27$ (times $C^2$) for $k = 3$. See [video generation](video-generation.html) for causal 3-D convolutions in video VAEs.

::: pitfall
FLOPs are not latency. A depthwise convolution performs only $k^2$ MACs per input value it reads, so on GPUs it is limited by memory bandwidth rather than arithmetic, and a depthwise-separable network with 8 times fewer FLOPs can run much less than 8 times faster. Grouped convolutions with many small groups have the same problem.
:::

### Dilated convolution

A dilated (atrous) convolution spaces its taps $d$ apart: $y[n] = \sum_u w[u]\, x[n + d u]$. It keeps $k$ weights but spans $d(k-1)+1$ pixels, so it enlarges the receptive field without downsampling and without more parameters. Stacking $3$-tap layers with dilations $1, 2, 4, \dots, 2^{L-1}$ gives

$$
r = 1 + 2\sum_{\ell=0}^{L-1} 2^\ell = 2^{L+1} - 1,
$$

exponential growth in depth at full resolution (Yu and Koltun, 2016; DeepLab uses the same idea under the name atrous convolution). WaveNet (van den Oord et al., 2016) stacks dilated *causal* convolutions over raw audio: padding only on the past side by $d(k-1)$ makes output $n$ depend only on inputs up to $n$, which is what an autoregressive model needs.

**Gridding.** Dilation makes a unit read a sparse lattice. With dilations $2, 2, 2$ and 3-tap kernels, the input offsets a unit can reach are sums of three numbers from $\{-2, 0, 2\}$: the even numbers from $-6$ to $6$, 7 of the 13 pixels in the span. Odd pixels never contribute, which shows up as grid-shaped artifacts in dense prediction. Mixing dilation rates so that their gaps are covered, or starting with $d = 1$, avoids it.

### Transposed convolution and checkerboard artifacts

A strided convolution is a linear map $y = A x$ from $n$ inputs to $n_\text{out} \approx n/s$ outputs. A **transposed convolution** computes $A^\top \delta$, a map back from $n_\text{out}$ to $n$ positions. It is called "transposed" for this reason, and "deconvolution" is a misnomer (it is not an inverse). In practice it is used as a learnable upsampling layer: each input value $x[i]$ adds the scaled kernel $x[i]\, w[u]$ onto output position $s i + u - p$, so copies of the kernel are pasted $s$ apart and summed where they overlap. Output size, from the last input's paste position $(n-1)s$ plus the kernel span $d(k-1) + 1$, minus the $2p$ cropped positions, plus an optional extra row $a$ ("output padding") to resolve the ambiguity of the floor in the forward size:

$$
n_\text{out} = (n-1)s - 2p + d(k-1) + a + 1 .
$$

With $k = 4$, $s = 2$, $p = 1$: $31 \cdot 2 - 2 + 3 + 1 = 64$ from $n = 32$, the standard exact $2\times$ upsampler.

**Where checkerboards come from (Odena, Dumoulin and Olah, 2016).** Feed a constant input of ones and a kernel of ones; the output counts how many pasted copies overlap at each position. In 1-D, over 8 inputs:

| $k$, $s$ | Overlap counts | Interior |
|---|---|---|
| 3, 2 | 1 1 2 1 2 1 2 … 2 1 1 | alternates 1, 2 |
| 4, 2 | 1 1 2 2 2 … 2 2 1 1 | constant 2 |
| 2, 2 | 1 1 1 … 1 | constant 1 |

When $s$ does not divide $k$, neighbouring outputs receive different numbers of contributions, so even a perfectly flat input gives a periodic output. In 2-D the pattern is the product of two such rows; for $k = 3$, $s = 2$ the interior repeats

$$
\begin{matrix} 1 & 2 & 1 & 2 \\ 2 & 4 & 2 & 4 \\ 1 & 2 & 1 & 2 \\ 2 & 4 & 2 & 4 \end{matrix}
$$

which is the checkerboard. Choosing $k$ divisible by $s$ makes the overlap even, but learned weights can still differ between the output phases, so artifacts can remain. The robust fix is to **resize, then convolve**: nearest-neighbour (or bilinear) upsampling followed by an ordinary $3 \times 3$ convolution. Every pixel then gets the same kind of computation. This is what common diffusion U-Nets and the Stable Diffusion VAE decoder do.

### Sub-pixel convolution (pixel shuffle)

Shi et al. (2016) proposed computing the upsampled image entirely at low resolution: a convolution produces $C r^2$ channels at size $H \times W$, and a fixed rearrangement, the periodic shuffle, turns them into $C$ channels at $rH \times rW$:

$$
y[c,\ r i + a,\ r j + b] = z[c\, r^2 + a\, r + b,\ i,\ j], \qquad 0 \le a, b < r,
$$

which is PyTorch's `nn.PixelShuffle(r)` ordering. Each of the $r^2$ sub-pixel positions $(a, b)$ has its own output channel, so its own filter.

Two facts make it easy to explain:

- *Cost.* All convolutions run on the $H \times W$ grid, $r^2$ times fewer positions than convolving after upsampling. This is why ESPCN could run super-resolution in real time on video.
- *Equivalence.* A sub-pixel convolution with a $k \times k$ kernel equals a transposed convolution with stride $r$ and kernel size $rk$, with the weights rearranged (Shi et al. made this point in a follow-up note). You can check it numerically in 1-D: the two outputs match exactly after a fixed crop. Since $rk$ is divisible by $r$, the overlap is even, which is why pixel shuffle avoids the uneven-overlap checkerboard. The $r^2$ phases are still independent filters, though, so a randomly initialized pixel shuffle can produce a fixed-pattern artifact. ICNR initialization (Aitken et al., 2017) sets all $r^2$ phase filters equal at the start, so the layer begins as nearest-neighbour upsampling followed by a convolution.

::: {.qa asked="bigo image/CV, Nowcoder 2019"}
**Which upsampling layers do super-resolution networks use? Explain sub-pixel convolution.**

Pre-upsampling networks (SRCNN, VDSR) interpolate to full size first and then refine, which is simple but runs every convolution at high resolution. Post-upsampling networks keep the work at low resolution and upsample at the end, with one of three layers: a transposed convolution (learnable, but checkerboard-prone when the stride does not divide the kernel size), resize-then-convolve (nearest or bilinear followed by a $3\times3$ conv, no checkerboard), or sub-pixel convolution. Sub-pixel convolution (pixel shuffle, ESPCN) predicts $C r^2$ channels at low resolution and rearranges each group of $r^2$ channels into an $r \times r$ block, giving $C$ channels at $r$ times the size. It is cheap because the convolution never sees the large grid, it is equivalent to a transposed convolution with stride $r$ and kernel $rk$ (even overlap), and ICNR initialization removes its remaining start-up artifacts. Progressive designs (LapSRN) upsample in $2\times$ stages, and arbitrary-scale methods (Meta-SR, LIIF) replace the layer with a function of continuous coordinates; see [restoration and super-resolution](restoration-super-resolution.html).
:::

::: {.qa asked="Megvii CV algorithm, Nowcoder 2022"}
**Implement nearest-neighbour upsampling. How does it differ from bilinear?**

For each output pixel, copy the input pixel whose area contains the output pixel's center. With the pixel-center convention, output index $j$ maps to input coordinate $(j + 0.5)\,W_\text{in}/W_\text{out} - 0.5$, and rounding it equals $\lfloor (j + 0.5)\,W_\text{in}/W_\text{out}\rfloor$:

```python
import numpy as np
def nearest_resize(img, out_h, out_w):
    H, W = img.shape[:2]
    ys = np.minimum(((np.arange(out_h) + 0.5) * H / out_h).astype(int), H - 1)
    xs = np.minimum(((np.arange(out_w) + 0.5) * W / out_w).astype(int), W - 1)
    return img[ys[:, None], xs[None, :]]   # works for (H, W) and (H, W, C)
```

This matches `F.interpolate(mode="nearest-exact")` for up- and downsampling at integer and non-integer scales; for an integer factor it is `img.repeat(s, 0).repeat(s, 1)`. Bilinear instead blends the four surrounding pixels with weights linear in the distances. Nearest creates no new values, which is right for label maps, masks and index images, but gives blocky staircase edges on natural images; bilinear is smooth but blurrier. In networks, nearest upsampling followed by a $3\times3$ convolution is the standard checkerboard-free upsampler, and its gradient sums the upstream gradient over each copied block.
:::

### Pooling and global average pooling

A pooling layer summarizes each window of each channel separately, with no parameters. Max pooling outputs $y_m = \max_{i \in R_m} x_i$ over window $R_m$; average pooling outputs $\frac{1}{|R_m|}\sum_{i \in R_m} x_i$. The common settings are $2 \times 2$ with stride 2 (VGG) and overlapping $3 \times 3$ with stride 2 (AlexNet, ResNet's stem). Pooling halves the resolution, which quarters the compute of later layers and enlarges their receptive fields, and max pooling gives a little invariance to small shifts inside a window.

**Backward.** Max pooling routes the whole upstream gradient to the input that won the window and gives the rest zero (the max is differentiable almost everywhere, with a subgradient at ties); the forward pass stores the argmax for this. Average pooling spreads $\delta_m/|R_m|$ evenly. Worked: window $\begin{bmatrix}1 & 3\\ 2 & 0\end{bmatrix}$, upstream gradient 5. Max pooling outputs 3 and sends back $\begin{bmatrix}0 & 5\\ 0 & 0\end{bmatrix}$; average pooling outputs 1.5 and sends back $1.25$ to each entry. With overlapping windows an input can win or belong to several windows, and its gradients add.

Many modern networks replace pooling with strided convolutions, which learn how to downsample (Springenberg et al., 2015). Either way, subsampling without a low-pass filter aliases, which is the shift-instability discussed earlier; Zhang's (2019) "BlurPool" evaluates the max densely (stride 1), blurs, then subsamples.

**Global average pooling (GAP)** averages each channel over all positions: $y[c] = \frac{1}{HW}\sum_{i,j} x[c, i, j]$. Introduced in Network in Network and used in GoogLeNet and ResNet, it replaces the large fully connected head. VGG-16's first fully connected layer maps $7 \cdot 7 \cdot 512 = 25{,}088$ inputs to 4,096 outputs, $102.8$M parameters; ResNet-50's head is GAP followed by a $2048 \to 1000$ layer, 2.05M parameters. GAP also lets the network accept any input size, and it makes the final layer's weights readable as per-class maps over positions (class activation maps, Zhou et al., 2016). In the backward pass each of the $HW$ positions receives $1/(HW)$ of the gradient.

::: {.qa asked="ByteDance CV 2023, Horizon 2022 (Nowcoder)"}
**How do dropout and pooling layers behave in the forward and backward passes, and how does dropout change between training and inference?**

None of these layers has parameters; their backward passes only route the upstream gradient. Max pooling forwards the window maximum and stores its position; backward sends the full gradient to that position and zero to the others. Average pooling forwards the mean and sends back $\delta/|R|$ to every entry; for global average pooling that is $\delta/(HW)$. Overlapping windows add their contributions. Pooling behaves identically in training and inference. Inverted dropout in training samples a mask $m_i \sim \mathrm{Bernoulli}(1-p)$ and outputs $m \odot x/(1-p)$, so the expectation is unchanged; backward multiplies the upstream gradient by the same mask and the same $1/(1-p)$, so dropped units get zero gradient in that step. At inference dropout is the identity, with no rescaling because it was applied during training. More on dropout is in [deep learning training](deep-learning-training.html).
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**What changes when you use interpolation to downsample instead of upsample?**

Upsampling only has to fill in values between samples; downsampling must also remove frequencies above the new Nyquist limit, or they fold back (alias) into moiré and jagged edges. Bilinear and bicubic kernels have a fixed support of 2 or 4 input pixels, which is enough when upsampling but filters far too little when shrinking by a large factor, because each output then reads only a few of the many input pixels it represents. Correct downsampling widens the kernel in proportion to the scale factor (antialiased resizing in PIL, MATLAB's `imresize`, or `antialias=True` in PyTorch), or blurs first. The same issue appears inside CNNs, where stride and pooling subsample without a low-pass filter. For super-resolution it matters because training pairs are made by downsampling, and a mismatch with how real low-resolution images were formed is a common reason models fail on real photos.
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**What is deformable convolution, and how would you implement it?**

A standard $k\times k$ convolution reads its $K = k^2$ taps at fixed offsets $p_n$ around each output position $p_0$: $y(p_0) = \sum_n w_n\, x(p_0 + p_n)$. A deformable convolution (Dai et al., 2017) adds learned, input-dependent fractional offsets: $y(p_0) = \sum_n w_n\, x(p_0 + p_n + \Delta p_n)\,\Delta m_n$, so the sampling grid can follow object shape. The offsets come from an ordinary convolution on the same input with $2K$ output channels (a $(\Delta y, \Delta x)$ pair per tap and position), initialized to zero so training starts from a regular convolution. Fractional positions are read by bilinear interpolation, which is differentiable in both the input and the offsets. DCNv2 (Zhu et al., 2019) adds the modulation $\Delta m_n \in [0, 1]$ from $K$ more channels through a sigmoid (DCNv1 has $\Delta m_n = 1$). Implementation: build the $K$ sampling grids from the offsets, gather $K$ bilinearly shifted copies of the input with `grid_sample`, multiply by the modulation, stack them as an im2col matrix and contract with the weights in one matrix multiply; `torchvision.ops.deform_conv2d` provides a fused kernel.
:::

## Backward pass of convolution

Automatic differentiation computes these gradients for you, but interviewers ask for them because the answer shows whether you understand the layer: both gradients are themselves convolutions, which is why frameworks can reuse the forward kernels, and why the transposed convolution exists.

### Derivation in 1-D

Take one channel, stride 1, no padding. The input has length $N$, the kernel length $k$, and the output length $M = N - k + 1$:

$$
y[n] = \sum_{u=0}^{k-1} w[u]\, x[n+u], \qquad n = 0, \dots, M-1 .
$$

Let $\delta[n] = \partial\mathcal{L}/\partial y[n]$ be the upstream gradient. Every gradient follows from the chain rule summed over the outputs that depend on the variable.

**Weight gradient.** $w[u]$ appears in every output, multiplied by $x[n+u]$, so $\partial y[n]/\partial w[u] = x[n+u]$ and

$$
\frac{\partial\mathcal{L}}{\partial w[u]} = \sum_{n=0}^{M-1} \delta[n]\, x[n+u] = (\delta \star x)[u], \qquad u = 0, \dots, k-1 .
$$

This is a cross-correlation of the input with the upstream gradient, with $\delta$ acting as the "kernel" (length $M$) and $x$ as the signal; the valid output length is $N - M + 1 = k$, one entry per weight, as it should be. The bias gradient is $\partial\mathcal{L}/\partial b = \sum_n \delta[n]$.

**Input gradient.** $x[m]$ appears in output $n$ at tap $u = m - n$, whenever $0 \le m - n \le k - 1$, so $\partial y[n]/\partial x[m] = w[m-n]$ (taken as 0 outside the kernel) and

$$
\frac{\partial\mathcal{L}}{\partial x[m]} = \sum_{n} \delta[n]\, w[m-n] = (\delta * w)[m], \qquad m = 0, \dots, N-1 .
$$

This is a *true* convolution of $\delta$ with $w$, evaluated over its full length $M + k - 1 = N$ (the "full" mode, where the kernel slides partly off both ends). Writing it as a correlation instead: pad $\delta$ with $k - 1$ zeros on each side and correlate with the flipped kernel $\tilde w[u] = w[k-1-u]$. That is the textbook phrase "a full convolution with the flipped kernel".

Why the flip appears: in the forward pass, input $x[m]$ meets the *first* tap $w[0]$ in the output to its own position and the *last* tap $w[k-1]$ in the output $k-1$ places to its left. Going backward, it collects from those outputs in the opposite order.

### A tiny example by hand

Let $x = [1, 2, -1, 3]$ and $w = [1, 2]$. Forward:

$$
y = [1 + 4,\ 2 - 2,\ -1 + 6] = [5, 0, 5].
$$

Take the loss $\mathcal{L} = y[0] - y[2]$, so $\delta = [1, 0, -1]$.

- Weight gradient, $\sum_n \delta[n] x[n+u]$: for $u = 0$, $1\cdot 1 + 0 \cdot 2 + (-1)(-1) = 2$; for $u = 1$, $1 \cdot 2 + 0 \cdot (-1) + (-1)\cdot 3 = -1$. So $\partial\mathcal{L}/\partial w = [2, -1]$.
- Input gradient, the full convolution $[1, 0, -1] * [1, 2]$: $[1,\ 2 + 0,\ 0 - 1,\ -2] = [1, 2, -1, -2]$.
- Same thing as a correlation: pad $\delta$ to $[0, 1, 0, -1, 0]$ and slide the flipped kernel $[2, 1]$ over it: $[0 + 1,\ 2 + 0,\ 0 - 1,\ -2 + 0] = [1, 2, -1, -2]$.

Check directly. Expanding, $\mathcal{L} = (x_0 + 2x_1) - (x_2 + 2x_3)$, whose gradient in $x$ is $[1, 2, -1, -2]$. Grouping by weights, $\mathcal{L} = w_0(x_0 - x_2) + w_1(x_1 - x_3) = 2w_0 - w_1$, whose gradient in $w$ is $[2, -1]$. Both match.

**With stride.** For $y[n] = \sum_u w[u]\, x[s n + u]$ the same steps give

$$
\frac{\partial\mathcal{L}}{\partial w[u]} = \sum_n \delta[n]\, x[sn + u], \qquad
\frac{\partial\mathcal{L}}{\partial x[m]} = \sum_n \delta[n]\, w[m - sn].
$$

Both are the stride-1 formulas applied to $\delta$ *dilated* by $s$, that is with $s - 1$ zeros inserted between its entries. Worked: $x = [1, 2, -1, 3, 0.5]$, $w = [1, -1, 2]$, $s = 2$ give $y = [-3, -3]$. With $\delta = [1, -2]$, the dilated gradient is $[1, 0, -2]$. Correlating $x$ with it gives $\partial\mathcal{L}/\partial w = [3, -4, -2]$, and the full convolution $[1, 0, -2] * [1, -1, 2] = [1, -1, 0, 2, -4]$ gives $\partial\mathcal{L}/\partial x$. Autograd agrees. The input gradient of a strided convolution is therefore exactly a transposed convolution of $\delta$, which is where that layer gets its name.

### Multi-channel form

For the 2-D layer with batch index $b$:

$$
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial w[o,c,u,v]} &= \sum_{b,i,j} \delta[b,o,i,j]\; x_\text{pad}[b,\ c,\ s i + u,\ s j + v], \\
\frac{\partial\mathcal{L}}{\partial x} &= \operatorname{ConvTranspose}(\delta,\ w,\ \text{stride } s,\ \text{padding } p), \\
\frac{\partial\mathcal{L}}{\partial b[o]} &= \sum_{b,i,j}\delta[b,o,i,j].
\end{aligned}
$$

Two details. The weight gradient sums over the batch, which plays the role the input channel plays in the forward pass (it is the index contracted away), so it can be computed by an ordinary convolution after swapping the batch and channel axes. The input gradient sums over *output* channels, so it uses the kernel with its two channel axes swapped, $w[o, c]$ read as a map from $o$ to $c$, as well as flipped in space. Forgetting the channel swap is the common mistake when writing this by hand.

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[[1., 2., -1., 3.]]], requires_grad=True)   # (batch, channel, length)
w = torch.tensor([[[1., 2.]]], requires_grad=True)
delta = torch.tensor([[[1., 0., -1.]]])                         # upstream gradient dL/dy
y = F.conv1d(x, w)                                              # [5, 0, 5]
y.backward(delta)
print(w.grad, F.conv1d(x.detach(), delta))             # dL/dw = correlate x with delta: [2, -1]
print(x.grad, F.conv_transpose1d(delta, w.detach()))   # dL/dx = full convolution:     [1, 2, -1, -2]

# 2-D, multi-channel, stride 2, padding 1: both formulas against autograd
x = torch.randn(2, 3, 7, 7, requires_grad=True)
w = torch.randn(4, 3, 3, 3, requires_grad=True)
y = F.conv2d(x, w, stride=2, padding=1)
delta = torch.randn_like(y)
y.backward(delta)
dw = torch.nn.grad.conv2d_weight(x.detach(), w.shape, delta, stride=2, padding=1)
dx = F.conv_transpose2d(delta, w.detach(), stride=2, padding=1)
print(torch.allclose(dw, w.grad, atol=1e-5), torch.allclose(dx, x.grad, atol=1e-5))  # True True
```

::: key
**How to say it in an interview.**

The weight gradient is the correlation of the layer's input with the upstream gradient, summed over the batch. The input gradient is the upstream gradient convolved with the kernel in full mode, which is the same as correlating the zero-padded gradient with the spatially flipped kernel, with input and output channels swapped. With stride $s$, dilate the upstream gradient by $s$ first; the input gradient is then exactly a transposed convolution. All three are convolutions, so the backward pass costs about twice the forward pass.
:::

::: qa
**Derive the gradient of a 1-D convolution with respect to its input, and explain the flipped kernel.**

From $y[n] = \sum_u w[u]x[n+u]$, the input $x[m]$ enters output $n$ through tap $m - n$, so $\partial\mathcal{L}/\partial x[m] = \sum_n \delta[n]w[m-n]$, the full convolution $\delta * w$. As a correlation it uses the kernel reversed: $x[m]$ meets $w[0]$ at output $m$ and $w[k-1]$ at output $m - k + 1$, so going backward it collects the taps in reverse order. Example: $\delta = [1, 0, -1]$, $w = [1, 2]$ give $[1, 2, -1, -2]$.
:::

::: qa
**Why is a transposed convolution called the gradient of a convolution?**

A convolution with stride $s$ is a linear map $y = Ax$ with a sparse, banded $A$. Backpropagation through it multiplies the upstream gradient by $A^\top$, and $A^\top\delta$ is what a transposed convolution computes: each entry of $\delta$ scatters a copy of the kernel, $s$ apart. Equivalently, dilate $\delta$ by $s$ and run a full convolution with $w$. cuDNN-backed frameworks typically run `ConvTranspose2d`'s forward pass with the same kernel as `Conv2d`'s input-gradient pass.
:::

### Convolution as a matrix multiply: im2col

The im2col ("image to columns") trick turns a convolution into one large matrix multiplication, the most heavily optimized operation on every processor. Copy each $C_\text{in} \times k \times k$ input window into a column of a matrix $X_\text{col}$ of shape $(C_\text{in}k^2,\ H_\text{out}W_\text{out})$, reshape the weights into $W_\text{mat}$ of shape $(C_\text{out},\ C_\text{in}k^2)$, and compute

$$
Y = W_\text{mat}\, X_\text{col} + b\,\mathbf{1}^\top, \qquad Y \in \mathbb{R}^{C_\text{out} \times H_\text{out}W_\text{out}} .
$$

This is the per-position reading of the forward formula. The cost is memory: each input value is copied into up to $k^2$ columns. For a $64 \times 56 \times 56$ input and a $3 \times 3$ kernel with stride 1, $X_\text{col}$ has $576 \times 3{,}136 = 1.81$M entries against 0.20M in the input, exactly 9 times more. cuDNN (Chetlur et al., 2014) avoids materializing the full matrix by forming tiles of it on the fly inside the multiply ("implicit GEMM").

The backward pass is equally simple in this view. From $Y = W_\text{mat}X_\text{col}$ with upstream gradient $\Delta$ (same shape as $Y$):

$$
\frac{\partial\mathcal{L}}{\partial W_\text{mat}} = \Delta\, X_\text{col}^\top, \qquad \frac{\partial\mathcal{L}}{\partial X_\text{col}} = W_\text{mat}^\top \Delta ,
$$

and the column gradient is folded back into image shape by adding each column into the window it came from ("col2im", `F.fold`). The adding is necessary because overlapping windows share input pixels. This is the same transposed convolution as before, now written as a matrix product.

In 1-D the matrix is explicit. For $x$ of length 4 and $w = [1, 2]$,

$$
A = \begin{bmatrix} 1 & 2 & 0 & 0 \\ 0 & 1 & 2 & 0 \\ 0 & 0 & 1 & 2 \end{bmatrix}, \qquad
A x = [5, 0, 5], \qquad A^\top [1, 0, -1] = [1, 2, -1, -2],
$$

reproducing both the forward pass and the input gradient of the worked example. $A$ is a Toeplitz matrix (constant along diagonals), which is weight sharing written as matrix structure.

### FFT and Winograd convolution

**FFT.** The convolution theorem says the discrete Fourier transform turns circular convolution into pointwise multiplication: $\mathcal{F}(x \circledast w) = \mathcal{F}x \cdot \mathcal{F}w$. For linear (non-circular) convolution, zero-pad both to at least $N + k - 1$ per axis so the wrap-around falls in the padding. For an $N \times N$ image and a $k \times k$ kernel, direct convolution costs $O(N^2k^2)$ per channel pair while the FFT route costs $O(N^2 \log N)$, independent of $k$. In a layer with many channels the transforms are shared: transform each of the $C_\text{in}$ inputs once, each of the $C_\text{in}C_\text{out}$ kernels once (precomputable at inference), multiply and sum over input channels in the frequency domain at $C_\text{in}C_\text{out}N^2$ complex multiply-adds, and run $C_\text{out}$ inverse transforms. The product step has no $k^2$ factor, so the FFT wins when kernels are large (Mathieu et al., 2014). For the $3 \times 3$ kernels of most CNNs it usually loses: kernels must be padded to the full image size, complex arithmetic costs about 4 real multiplies, and memory traffic grows. Where very long kernels appear, FFT convolution is the standard tool: large image filters, and the long convolutions of state-space models such as S4, computed in $O(L\log L)$ for sequence length $L$.

**Winograd.** For small kernels, Lavin and Gray (2016) used Winograd's minimal filtering algorithms, which trade multiplications for additions. Computing a $2 \times 2$ output tile of a $3\times3$ convolution, written $F(2\times2, 3\times3)$, takes $(2 + 3 - 1)^2 = 16$ multiplications instead of $2^2 \cdot 3^2 = 36$, a 2.25-fold reduction; $F(4\times4, 3\times3)$ takes 36 instead of 144, four-fold, at some cost in numerical precision. cuDNN chooses among direct, implicit-GEMM, FFT and Winograd algorithms per layer shape (`torch.backends.cudnn.benchmark = True` makes PyTorch time them and pick).

::: {.qa asked="OPPO 2021, NetEase 2021 (Nowcoder)"}
**How do you convolve an image with a very large kernel quickly?**

Direct convolution costs $O(HWk^2)$ per channel, so exploit structure in the kernel or change domains. (1) Separable kernels: if the kernel is an outer product $uv^\top$ (Gaussian, box), filter rows with $v$ and columns with $u$, $O(2HWk)$; a general kernel can be approximated by its top $r$ singular components at $O(2rHWk)$. (2) FFT: multiply the transforms and invert, $O(N\log N)$ with $N = (H + k - 1)(W + k - 1)$ after zero-padding (to avoid circular wrap-around), independent of $k$; for very large images, process blocks with overlap-add or overlap-save. (3) Box filters in $O(1)$ per pixel with an integral image (summed-area table), and Gaussians approximated by three repeated box filters (the central limit theorem) or by recursive IIR filters, whose cost per pixel does not depend on $\sigma$. (4) Image pyramids: downsample, filter with a smaller kernel, upsample. For small kernels, direct, implicit-GEMM or Winograd convolution is faster; the crossover depends on hardware, so benchmark. Inside networks, large receptive fields come from dilation, stacks of small kernels, or large depthwise kernels (ConvNeXt's $7 \times 7$), which cost $O(HWk^2C)$ instead of $O(HWk^2C^2)$.
:::

::: qa
**What does im2col cost, and why do libraries use it anyway?**

It copies every input window into a column, so the buffer is $C_\text{in}k^2 \times H_\text{out}W_\text{out}$, about $k^2$ times the input for stride 1 (9 times for $3 \times 3$). In exchange the convolution becomes one dense matrix multiply, which runs near peak hardware throughput, and the backward pass becomes two more matrix multiplies plus a fold. Implicit-GEMM kernels get the speed without the memory by building the tiles on the fly.
:::

## Classic CNN architectures: a guided history

Each architecture below is remembered for one or two ideas that later networks kept. The table gives the headline numbers as reported in each paper; ImageNet numbers are on the 1000-class ILSVRC benchmark, top-5 error for the early competition entries and top-1 accuracy for the efficiency-focused later models.

| Model | Year | Key idea | Parameters | Reported result |
|---|---|---|---|---|
| LeNet-5 | 1998 | convolution + subsampling, trained end to end | 60K | 0.95% MNIST test error |
| AlexNet | 2012 | ReLU, dropout, GPU training, augmentation | 60M | 15.3% top-5 (ILSVRC-2012 test) |
| VGG-16 / 19 | 2014 | only $3\times3$ convolutions, more depth | 138M / 144M | 7.3% top-5 (7-model ensemble) |
| GoogLeNet | 2014 | Inception modules with $1\times1$ reductions | 12× fewer than AlexNet | 6.67% top-5 |
| ResNet-50 / 152 | 2015 | residual learning, bottleneck blocks | 25.6M / 60.2M | 3.57% top-5 (ensemble) |
| DenseNet-121 | 2016 | concatenate all earlier feature maps | 8.0M | |
| MobileNet v1 | 2017 | depthwise-separable convolutions | 4.2M | 70.6% top-1 |
| MobileNet v2 | 2018 | inverted residuals, linear bottlenecks | 3.4M | 72.0% top-1 |
| EfficientNet-B0 / B7 | 2019 | compound scaling of depth, width, resolution | 5.3M / 66M | 77.1% / 84.3% top-1 |
| ConvNeXt-T | 2022 | a ResNet modernized with transformer-era choices | 28M | 82.1% top-1 |
| U-Net | 2015 | encoder–decoder with skip connections | | biomedical segmentation |

### LeNet-5 (LeCun et al., 1998)

LeNet-5 read $32 \times 32$ images of handwritten digits and already had every ingredient of a modern CNN: local receptive fields, shared weights, spatial subsampling, and training of the whole stack by gradient descent. (LeCun et al., 1989, had trained a convolutional network on zip codes with backpropagation; Fukushima's Neocognitron, 1980, had the local, shared, hierarchical structure without gradient training.) The layers and their parameters, each derivable from the earlier formulas:

| Layer | Output | Trainable parameters | How |
|---|---|---|---|
| C1: conv $5\times5$ | $6 @ 28 \times 28$ | 156 | $6 \cdot (25 + 1)$ |
| S2: subsample $2\times2$ | $6 @ 14 \times 14$ | 12 | one coefficient and one bias per map |
| C3: conv $5\times5$ | $16 @ 10\times10$ | 1,516 | partial connection table, below |
| S4: subsample $2\times2$ | $16 @ 5 \times 5$ | 32 | $2 \cdot 16$ |
| C5: conv $5\times5$ | $120 @ 1 \times 1$ | 48,120 | $120 \cdot (16 \cdot 25 + 1)$ |
| F6: fully connected | 84 | 10,164 | $84 \cdot (120 + 1)$ |
| Output | 10 | | Euclidean radial-basis units |

The sum is exactly 60,000, the paper's figure. Details worth knowing:

- *Subsampling layers* add the four inputs of each $2 \times 2$ window, multiply by a trainable coefficient, add a trainable bias and apply a sigmoid. This is average pooling with a learned scale.
- *C3 is not fully connected to S2.* Six of its maps read 3 of S2's maps, nine read 4, and one reads all 6: $6 \cdot 76 + 9 \cdot 101 + 151 = 1{,}516$ parameters, instead of $16 \cdot (6 \cdot 25 + 1) = 2{,}416$ for full connection. The paper's reasons were to keep the number of connections bounded and to break symmetry so that different maps learn different features. Modern networks connect all channels; grouped convolutions are the controlled version of this idea.
- *C5* is a convolution whose $5\times5$ kernel exactly covers its $5 \times 5$ input, so it acts as a fully connected layer for $32 \times 32$ inputs but would slide over larger ones.
- The activation is a scaled $\tanh$, $f(a) = A\tanh(Sa)$ with $A = 1.7159$.
- Test error on MNIST was 0.95%, and 0.8% when training on artificially distorted copies of the images.

### AlexNet (Krizhevsky, Sutskever and Hinton, 2012)

AlexNet won ILSVRC-2012 with 15.3% top-5 test error against 26.2% for the second-best entry, and that gap started the deep-learning era in vision. The network has five convolutional layers and three fully connected layers, 60M parameters and 650,000 neurons; the first layer has 96 kernels of size $11\times11\times3$ with stride 4. Its ideas:

- **ReLU.** $\max(0, a)$ does not saturate for positive inputs, so its gradient there is exactly 1, while $\tanh'$ approaches 0 for large $\lvert a\rvert$. In the paper, a four-layer CNN with ReLUs reached 25% training error on CIFAR-10 six times faster than the same network with $\tanh$.
- **GPUs.** It was trained on two GTX 580 GPUs with 3 GB of memory each, for five to six days. The network was split across the two GPUs, which communicated only at certain layers; this is where grouped convolution with $G = 2$ came from.
- **Dropout** with probability 0.5 in the first two fully connected layers, which held most of the parameters. The paper notes it roughly doubles the number of iterations needed to converge. The training and inference behaviour of dropout is covered in [deep learning training](deep-learning-training.html).
- **Data augmentation**: random $224 \times 224$ crops of $256 \times 256$ images with horizontal flips, and a PCA-based color jitter.
- **Overlapping max pooling** ($3 \times 3$ windows, stride 2), which lowered top-1 and top-5 error by 0.4% and 0.3% compared with non-overlapping pooling, and local response normalization, which later networks dropped in favour of BatchNorm.

Where the parameters are: in torchvision's single-GPU version (61.1M parameters), the fully connected layers hold 58.6M, and the first of them alone, $6 \cdot 6 \cdot 256 \cdot 4096 + 4096 = 37.8$M. The five convolutional layers hold only 2.5M but do most of the computation.

### VGG (Simonyan and Zisserman, 2015)

VGG asked one question: what happens if you keep the design fixed and only add depth? Every convolution is $3 \times 3$ with stride 1 and padding 1, every downsampling is a $2\times2$ max pool with stride 2, the widths are 64, 128, 256, 512, 512 across five stages, and three fully connected layers (4096, 4096, 1000) finish the network. VGG-16 and VGG-19 have 16 and 19 weight layers. VGG was second in the ILSVRC-2014 classification task with 7.3% top-5 test error (an ensemble of 7 models), and 6.8% after the competition with 2 models.

**Why stacks of $3 \times 3$.** Two stacked $3\times3$ convolutions see $1 + 2 + 2 = 5$ pixels, the same receptive field as one $5 \times 5$, and three see 7, the same as one $7\times7$ (receptive-field recursion with stride 1). Compare parameters, with $C$ channels in and out and no biases:

$$
\begin{aligned}
\text{two } 3\times3: &\quad 2 \cdot 9C^2 = 18C^2 \quad\text{vs.}\quad \text{one } 5\times5: 25C^2 \qquad (28\% \text{ fewer}), \\
\text{three } 3\times3: &\quad 3 \cdot 9C^2 = 27C^2 \quad\text{vs.}\quad \text{one } 7\times7: 49C^2 .
\end{aligned}
$$

The paper puts the second comparison as "81% more" parameters for the $7 \times 7$ ($49/27 = 1.81$). The stack also inserts two extra ReLUs, so the same receptive field computes a more nonlinear function, and the authors describe the stack as imposing a kind of regularization on the $7\times7$ filters, forcing them to decompose through $3\times3$ ones. The MAC counts scale the same way as the parameters. A small subtlety: the stack can represent only some $5 \times 5$ functions, because a composition of two linear $3\times3$ filters has a constrained form; with the ReLU in between it represents different functions altogether, so "equivalent" means equal receptive field, not equal function class.

**Costs.** VGG-16 has 138.4M parameters, of which 123.6M (89%) sit in the fully connected head (102.8M in the first layer, as computed earlier), and about 15.5G MACs at $224 \times 224$; VGG-19 has 143.7M parameters and 19.6G MACs. To train the deeper configurations the authors initialized some of their layers from a trained shallower configuration, and noted after submission that Glorot initialization works without this step. Accuracy saturated at 19 layers.

VGG is still used, but as a feature extractor: perceptual losses (Johnson et al., 2016) and LPIPS (Zhang et al., 2018) compare images in the feature space of a pretrained VGG or AlexNet, which matters for super-resolution and image generation (see [evaluation metrics](evaluation-metrics.html)).

### GoogLeNet and the Inception module (Szegedy et al., 2015)

GoogLeNet won ILSVRC-2014 classification with 6.67% top-5 error. It is 22 layers deep (27 counting pooling), was designed for a budget of 1.5 billion multiply-adds at inference, and uses 12 times fewer parameters than AlexNet.

**The Inception module** runs $1\times1$, $3\times3$ and $5\times5$ convolutions and a $3\times3$ max pool in parallel on the same input and concatenates their outputs along channels, so each layer can pick up features at several scales. Done naively this is expensive in two ways: the $5 \times 5$ branch on many channels costs a lot, and the pooling branch keeps all its input channels, so the output width grows from module to module. The fix is **$1\times1$ reductions**: a cheap $1\times1$ convolution to few channels before each $3 \times 3$ and $5\times5$ convolution, and after the pooling ("pool projection").

**Worked: module 3a** from the paper's Table 1. Input $28 \times 28 \times 192$. Branches: 64 channels of $1 \times 1$; a reduction to 96 then 128 of $3 \times 3$; a reduction to 16 then 32 of $5 \times 5$; a pool followed by 32 of $1\times1$. Output channels $64 + 128 + 32 + 32 = 256$. Weights:

$$
\underbrace{192 \cdot 64}_{1\times1} + \underbrace{192 \cdot 96 + 9 \cdot 96 \cdot 128}_{3\times3 \text{ branch}} + \underbrace{192 \cdot 16 + 25 \cdot 16 \cdot 32}_{5\times5 \text{ branch}} + \underbrace{192 \cdot 32}_{\text{pool proj}} = 163{,}328,
$$

and MACs $= 28 \cdot 28 \cdot 163{,}328 = 128.0$M, the "128M ops" of Table 1 (the table's "159K" parameters is $163{,}328/1024$). The $5\times5$ branch alone would need $25 \cdot 192 \cdot 32 = 153{,}600$ weights without its reduction and needs $15{,}872$ with it, 9.7 times fewer. Without any reductions or projections the module, with the same branch widths, would have 387,072 weights, cost 303M MACs, and output $64 + 128 + 32 + 192 = 416$ channels, which would make the next module more expensive still.

Other parts of GoogLeNet that stuck: global average pooling instead of a large fully connected head, and two auxiliary classifiers on intermediate layers, whose losses are added with weight 0.3 to give the middle of the network a direct training signal. Later versions added BatchNorm, factorized $n \times n$ convolutions into $1 \times n$ and $n \times 1$ (Inception-v3), and residual connections (Inception-ResNet).

### ResNet (He et al., 2016)

**The problem: degradation.** Stacking more layers onto a well-trained network made the *training* error worse, not just the test error, so overfitting was not the cause. On ImageNet, a plain 34-layer network had higher error than a plain 18-layer one. The networks used BatchNorm, and the authors checked that forward activations and backward gradients had healthy norms, so they argued it was an optimization difficulty rather than plain vanishing gradients. Yet a deeper network can always copy the shallower one and set its extra layers to the identity, so a solution at least as good exists; the optimizer just does not find it.

**Residual learning.** Instead of asking a few layers to fit a desired mapping $\mathcal{H}(x)$, let them fit the residual $F(x) = \mathcal{H}(x) - x$ and add the input back:

$$
y = F(x) + x .
$$

If the identity is close to optimal, pushing $F$ toward zero is easy (shrink the weights), whereas making a stack of nonlinear layers reproduce the identity is not. The backward pass gets a direct path: $\partial\mathcal{L}/\partial x = \partial\mathcal{L}/\partial y\,(I + \partial F/\partial x)$, so the gradient reaches earlier layers through the identity term even when $\partial F/\partial x$ is small. With residual connections the 34-layer network beat the 18-layer one (10-crop top-1 validation error: plain-34 28.54% against plain-18 27.94%, but ResNet-34 25.03% against ResNet-18 27.88%), and a 152-layer ensemble won ILSVRC-2015 with 3.57% top-5 error. The pre-activation variant, the "1" in the gradient that reaches any earlier layer, and the link to transformers' residual stream are covered on [vision backbones](vision-backbones.html) and [deep learning training](deep-learning-training.html).

**Blocks.** ResNet-18 and ResNet-34 use the *basic block*: two $3 \times 3$ convolutions, each followed by BatchNorm, with a ReLU between them and another after the addition. ResNet-50, 101 and 152 use the *bottleneck block*: a $1 \times 1$ convolution reduces the width from $4w$ to $w$, a $3\times3$ convolution works at width $w$, and a $1 \times 1$ convolution expands back to $4w$. At $4w = 256$ a bottleneck has $256 \cdot 64 + 9 \cdot 64^2 + 64 \cdot 256 = 69{,}632$ weights, against $2 \cdot 9 \cdot 256^2 = 1{,}179{,}648$ for two $3 \times 3$ convolutions at full width, 17 times fewer. When the shape changes at the start of a stage (stride 2, more channels), the shortcut becomes a $1\times1$ convolution with stride 2 followed by BatchNorm (a "projection shortcut").

**Deriving ResNet-50's 25.6M parameters.** Count convolution weights (no biases, since BN follows every convolution), BatchNorm parameters ($\gamma$ and $\beta$, two per channel), and the final classifier. A bottleneck block of width $w$ whose input has $c_\text{in}$ channels has

$$
\underbrace{c_\text{in}\, w}_{1\times1 \text{ reduce}} + \underbrace{9w^2}_{3\times3} + \underbrace{4w^2}_{1\times1 \text{ expand}} \;\big(+\ \underbrace{4w\, c_\text{in}}_{\text{projection}}\big)
$$

convolution weights, and BatchNorm parameters $2(w + w + 4w) = 12w$, plus $8w$ with a projection. Only the first block of each stage has a projection; in the other blocks $c_\text{in} = 4w$, and the conv weights simplify to $4w^2 + 9w^2 + 4w^2 = 17w^2$. ResNet-50 has 3, 4, 6 and 3 blocks in its four stages:

| Stage | $w$ | Blocks | First-block input $c_\text{in}$ | Conv weights | BN parameters | Total |
|---|---|---|---|---|---|---|
| stem: $7\times7$ conv, $3 \to 64$ | | | 3 | 9,408 | 128 | 9,536 |
| conv2_x | 64 | 3 | 64 | 212,992 | 2,816 | 215,808 |
| conv3_x | 128 | 4 | 256 | 1,212,416 | 7,168 | 1,219,584 |
| conv4_x | 256 | 6 | 512 | 7,077,888 | 20,480 | 7,098,368 |
| conv5_x | 512 | 3 | 1024 | 14,942,208 | 22,528 | 14,964,736 |
| fc: $2048 \to 1000$ | | | | | | 2,049,000 |
| **Total** | | | | | | **25,557,032** |

Check one row by hand. In conv3_x the first block has $256 \cdot 128 + 13 \cdot 128^2 + 4 \cdot 128 \cdot 256 = 32{,}768 + 212{,}992 + 131{,}072 = 376{,}832$ conv weights, and each of the other three has $17 \cdot 128^2 = 278{,}528$, for $376{,}832 + 3 \cdot 278{,}528 = 1{,}212{,}416$. The total, 25,557,032, matches `torchvision.models.resnet50()` exactly. Two things stand out. The last stage holds 58.6% of all parameters, because weights grow with $w^2$ and $w$ doubles each stage while the resolution halves. And the parameters do not tell you the compute: conv5_x runs on a $7 \times 7$ grid, so its 15M weights cost only $49 \times 15$M $\approx 0.73$G MACs.

**Compute.** The paper reports $3.8 \times 10^9$ multiply-adds for ResNet-50 at $224 \times 224$ (3.6 for ResNet-34, 11.3 for ResNet-152, 19.6 for VGG-19). Counting with forward hooks reproduces 3.86G for the original design, which puts the stride-2 of each downsampling block on the first $1 \times 1$ convolution. torchvision's ResNet-50 moves that stride to the $3 \times 3$ convolution (often called "ResNet v1.5"), which is slightly more accurate and costs 4.09G MACs with the same parameter count.

::: {.qa asked="ByteDance/OPPO/Momo image roles, Nowcoder 2021"}
**Implement a ResNet basic block as a reusable module.**

The block computes $\mathrm{ReLU}(F(x) + S(x))$ with $F$ = conv3x3, BN, ReLU, conv3x3, BN, and a shortcut $S$ that is the identity when shapes match and a strided $1\times1$ convolution plus BN otherwise.

```python
import torch.nn as nn, torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)  # BN follows: no bias
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        nn.init.zeros_(self.bn2.weight)                  # branch starts at zero: block = identity
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:                 # projection when the shape changes
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                          nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))            # ReLU after the addition
```

Points to say: convolutions have no bias because BN follows; only the first block of a stage has stride 2; zero-initializing the last BN's $\gamma$ (Goyal et al., 2017; torchvision's `zero_init_residual`) makes each block start as the identity, which helps early training of deep networks. Check: `nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))` has 525,568 parameters, the same as torchvision's `resnet18().layer2` ($73{,}728 + 147{,}456 + 8{,}192$ conv weights and 768 BN parameters in the first block, $2 \cdot 147{,}456$ and 512 in the second).
:::

### DenseNet (Huang et al., 2017)

DenseNet takes the shortcut idea further. Inside a *dense block*, layer $\ell$ receives the concatenation of the block input and every earlier layer's output,

$$
x_\ell = H_\ell\big([x_0, x_1, \dots, x_{\ell-1}]\big),
$$

where $H_\ell$ is BN, ReLU, conv. Each $H_\ell$ adds only $k$ new channels, the *growth rate*, so layer $\ell$ sees $k_0 + k(\ell - 1)$ input channels, and a network of $L$ such layers has $L(L+1)/2$ direct connections. DenseNet-121 uses $k = 32$ and blocks of 6, 12, 24 and 16 layers. Each $H_\ell$ is a bottleneck (BN-ReLU-$1\times1$ conv to $4k$ channels, then BN-ReLU-$3\times3$ conv to $k$), and *transition layers* between blocks halve the channels with a $1\times1$ convolution (compression $\theta = 0.5$) and halve the resolution with $2 \times 2$ average pooling.

**Worked channel count for DenseNet-121.** Block 1 starts with 64 channels and ends with $64 + 6 \cdot 32 = 256$; the transition halves this to 128. Block 2: $128 + 12 \cdot 32 = 512 \to 256$. Block 3: $256 + 24 \cdot 32 = 1024 \to 512$. Block 4: $512 + 16 \cdot 32 = 1024$, which is the width of the final feature vector. The whole network has 8.0M parameters, under a third of ResNet-50's 25.6M.

**Add versus concatenate.** A ResNet block computes $x + F(x)$: the width stays fixed and features are summed, so later layers cannot separate what was there from what was added. DenseNet keeps every earlier feature map intact, so layers can reuse them directly and need only a few new channels each; every layer also has a short path to the loss. The price is memory: the concatenated inputs grow linearly within a block, and a naive implementation stores a new concatenated copy at every layer, so memory grows quadratically with block depth. Pleiss et al. (2017) fixed this with shared buffers and recomputation.

::: {.qa asked="Ctrip, bigo, NVIDIA (Nowcoder, PracHub)"}
**Why do residual connections work? Compare ResNet with VGG and DenseNet.**

A residual block computes $y = x + F(x)$. The identity is the default (set $F = 0$), so extra depth cannot make a good solution harder to reach, which fixed the degradation problem where deeper plain networks had higher *training* error. In the backward pass the Jacobian is $I + \partial F/\partial x$, so gradients reach early layers through the identity path even when the branch's Jacobian is small. VGG stacks plain $3\times3$ convolutions with no shortcuts: it showed depth helps, put 89% of its 138M parameters in the fully connected head, and stopped improving at 19 layers. ResNet adds shortcuts, uses bottleneck blocks and global average pooling, and trains at 152 layers with 25.6M parameters for ResNet-50. DenseNet concatenates all earlier feature maps in a block instead of adding, which encourages feature reuse and is parameter-efficient (8M for DenseNet-121) but costs more activation memory for the growing concatenations.
:::

### MobileNet v1 and v2

**MobileNet v1 (Howard et al., 2017)** builds the whole network, after a standard first layer, from depthwise-separable blocks: $3\times3$ depthwise conv, BN, ReLU, $1\times1$ conv, BN, ReLU. Per block this saves the factor $1/C_\text{out} + 1/k^2$ derived earlier, "between 8 to 9 times less computation". Two global knobs trade accuracy for cost: a width multiplier $\alpha$ scales every channel count, which scales the dominant $1\times1$ cost $C_\text{in}C_\text{out}$ by $\alpha^2$; a resolution multiplier $\rho$ scales the input size, which scales cost by $\rho^2$. The full-size model, "1.0 MobileNet-224", has 4.2M parameters, costs 569M MACs, and reaches 70.6% ImageNet top-1. Summing the layer table gives 569M MACs and 4.21M parameters, with 95% of the MACs and 75% of the parameters in the $1\times1$ convolutions, which is why later designs worked on the $1\times1$ part.

**MobileNet v2 (Sandler et al., 2018)** introduced the **inverted residual with linear bottleneck**. Each block takes a thin input with $C$ channels and

1. expands it with a $1\times1$ convolution to $tC$ channels ($t = 6$ in the paper), followed by ReLU6, $\min(\max(0, a), 6)$;
2. filters with a $3\times3$ depthwise convolution at the expanded width, followed by ReLU6;
3. projects back down with a $1\times1$ convolution to $C'$ channels, with **no** activation (the linear bottleneck);
4. adds the input when the stride is 1 and $C = C'$.

The MACs of one block at stride 1 on an $H \times W$ map are $HW \cdot tC\,(C + k^2 + C')$: the expansion costs $HW \cdot C \cdot tC$, the depthwise filter $HW \cdot tC \cdot k^2$, and the projection $HW \cdot tC \cdot C'$.

Why "inverted": a ResNet bottleneck goes wide, narrow, wide and connects the *wide* ends with the shortcut; MobileNet v2 goes narrow, wide, narrow and connects the *narrow* ends, so the tensors that must be kept for the shortcut are small. Why linear: the authors argue that the information of interest lies on a low-dimensional manifold inside the activations, and a ReLU applied in a low-dimensional space destroys information (everything on the negative side of each unit is zeroed), while in a high-dimensional space the same information survives in other channels. So nonlinearities are applied only at the expanded width. MobileNet v2 reaches 72.0% top-1 with 3.4M parameters and 300M MACs.

### EfficientNet: compound scaling (Tan and Le, 2019)

Networks are usually scaled up along one axis: deeper (ResNet-18 to 152), wider (Wide ResNets), or with larger inputs. Tan and Le observed that each axis alone saturates, and that the axes interact: higher-resolution inputs need more depth for a larger receptive field and more width to capture the finer patterns. Their **compound scaling** grows all three together with one coefficient $\phi$:

$$
\text{depth } d = \alpha^\phi, \qquad \text{width } w = \beta^\phi, \qquad \text{resolution } r = \gamma^\phi, \qquad \text{subject to } \alpha\beta^2\gamma^2 \approx 2,\ \ \alpha, \beta, \gamma \ge 1 .
$$

**Why $\beta^2$ and $\gamma^2$.** Convolution MACs are linear in the number of layers, quadratic in width (the $C_\text{in}C_\text{out}$ product, when all widths scale by $w$) and quadratic in resolution (the $H_\text{out}W_\text{out}$ product). So the total cost scales as $d\,w^2r^2 = (\alpha\beta^2\gamma^2)^\phi \approx 2^\phi$: each unit of $\phi$ roughly doubles the FLOPs. A small grid search at $\phi = 1$ found $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$, for which $\alpha\beta^2\gamma^2 = 1.2 \cdot 1.21 \cdot 1.3225 = 1.92$. Worked: at $\phi = 2$ depth grows by $1.44$, width by $1.21$, resolution by $1.32$, and FLOPs by $1.92^2 \approx 3.7$.

The baseline, EfficientNet-B0, came from a neural architecture search over MobileNet-v2-style inverted residual blocks with squeeze-and-excitation; it has 5.3M parameters, 0.39B FLOPs and 77.1% top-1. Scaling it up gives B1 to B7; B7 reaches 84.3% top-1 with 66M parameters and 37B FLOPs, which the paper reports as 8.4 times smaller and 6.1 times faster at inference than the best earlier ConvNet (GPipe). (The first arXiv version reported 76.3% for B0 and 84.4% for B7; the numbers here are from the final version.)

### ConvNeXt, briefly (Liu et al., 2022)

ConvNeXt asks how much of the vision transformer's advantage comes from attention and how much from everything around it. It starts from ResNet-50 and changes one design choice at a time toward Swin Transformer. The modern training recipe alone (AdamW, longer training, heavy augmentation) raised ResNet-50 from 76.1% to 78.8% top-1. The architectural steps included (3, 3, 9, 3) blocks per stage, a $4\times4$ stride-4 "patchify" stem, an inverted bottleneck with $7\times7$ depthwise convolutions, GELU, fewer activations and normalizations, and LayerNorm instead of BatchNorm. ConvNeXt-T reaches 82.1% top-1 at 4.5 GFLOPs against Swin-T's 81.3%: a pure CNN can match a hierarchical vision transformer at this scale, so attention is not the only reason transformers did well. The ViT side of the comparison is on [vision backbones](vision-backbones.html).

### U-Net (Ronneberger, Fischer and Brox, 2015)

U-Net was designed for biomedical image segmentation with very few training images. It has a **contracting path** (encoder) and an **expanding path** (decoder) of matching resolutions:

- Encoder: at each of four levels, two unpadded $3 \times 3$ convolutions with ReLU, then $2 \times 2$ max pooling with stride 2; the channel count doubles at each level, from 64 to 1024 at the bottom.
- Decoder: at each level, a $2 \times 2$ "up-convolution" (a transposed convolution with stride 2) that halves the channels, concatenation with the encoder's feature map from the same level, then two $3\times3$ convolutions with ReLU.
- A final $1\times1$ convolution maps to class scores.

That is $4 \cdot 2 + 2 + 4 \cdot 3 + 1 = 23$ convolutional layers, as the paper says. Because the convolutions are unpadded, each one trims 2 pixels, and the output is smaller than the input: following $572 \to 568 \to 284 \to 280 \to 140 \to 136 \to 68 \to 64 \to 32 \to 28$ at the bottom, then $56 \to 52 \to 104 \to 100 \to 200 \to 196 \to 392 \to 388$, a $572 \times 572$ input tile gives a $388 \times 388$ output. The encoder maps are larger than the decoder maps they join (64 against 56 at the deepest skip), so they are center-cropped before concatenation. Large images are processed in overlapping tiles, with missing context at the image border filled by mirroring. The paper also used heavy elastic-deformation augmentation and a loss weighting that emphasizes the thin borders between touching cells, and won two categories of the ISBI cell tracking challenge 2015.

**Why the skip connections.** The encoder throws away spatial detail to gain context: by the bottom, each unit sees a large region but on a coarse $28 \times 28$ grid. The decoder must produce a per-pixel output, and it cannot recover exact boundaries from the coarse map alone. Concatenating the encoder's high-resolution maps gives it those details directly, while the bottleneck supplies the context. These skips are long and concatenate; ResNet's are short, add, and exist for optimization.

**Why U-Nets became the workhorse of diffusion models.** A diffusion denoiser maps a noisy image to a prediction of the same shape (the noise, the clean image or a velocity), at every noise level. The U-Net fits this well:

- *Same-shape, pixel-aligned output.* Encoder–decoder with skips naturally produces a full-resolution, spatially aligned prediction.
- *Multi-scale work for multi-scale noise.* At high noise the model must decide global layout, which needs the large receptive field of the low-resolution levels (DDPM adds self-attention at $16 \times 16$); at low noise it must fix fine texture, which the high-resolution skips carry.
- *The right prior and cost.* Convolutions are translation equivariant, share weights across positions and cost time linear in the number of pixels.
- *Easy conditioning.* The timestep embedding is added inside every residual block, and text enters through cross-attention layers at the lower resolutions (Stable Diffusion).

The DDPM U-Net (Ho et al., 2020) followed the PixelCNN++ backbone, a U-Net of Wide-ResNet blocks, with group normalization; later models (ADM, Stable Diffusion) added more attention and adaptive normalization. Since 2023 many large text-to-image models have replaced the U-Net by a transformer on latent patches (DiT, the MMDiT of Stable Diffusion 3); see [diffusion models](diffusion-models.html) and [text-to-image and multimodal](text-to-image-multimodal.html). U-Nets remain common in pixel-space restoration and super-resolution.

::: {.qa asked="ByteDance 2020, Baidu 2024 (Nowcoder)"}
**Why does a U-Net downsample and upsample, and what do its skip connections do?**

Downsampling grows the receptive field and gathers context cheaply, since each level halves the resolution and quarters the cost per layer; upsampling returns to full resolution because the task needs a per-pixel output. Skip connections concatenate the encoder's feature maps at each resolution into the decoder, restoring the spatial detail the bottleneck lost, and shorten gradient paths. In diffusion U-Nets this multi-scale structure suits denoising at every noise level: the low resolutions (with self-attention) handle global layout, the skips carry fine texture, and the timestep embedding enters every residual block. U-Net skips concatenate and carry information; ResNet skips add and ease optimization; diffusion U-Nets use both.
:::

::: qa
**Why does VGG use stacks of $3\times3$ convolutions instead of larger kernels?**

Two stacked $3\times3$ convolutions have the receptive field of one $5\times5$ ($1 + 2 + 2 = 5$) and three have that of a $7\times7$, with fewer parameters: $18C^2$ against $25C^2$, and $27C^2$ against $49C^2$ (the single $7\times7$ has 81% more). The stack also adds a ReLU between layers, so it computes a more nonlinear function over the same window, and compute falls in the same ratio as parameters. The equivalence is in receptive field, not in the set of functions.
:::

::: qa
**What does a $1\times1$ reduction save in an Inception module? Use module 3a.**

The $5\times5$ branch maps $28 \times 28 \times 192$ to 32 channels. Directly: $25 \cdot 192 \cdot 32 = 153{,}600$ weights. With a $1\times1$ reduction to 16 channels first: $192 \cdot 16 + 25 \cdot 16 \cdot 32 = 15{,}872$, 9.7 times fewer, and MACs fall by the same factor because both run on the same $28 \times 28$ grid. The pool projection also stops the output width from growing (256 channels instead of 416).
:::

## Recurrent networks

### The vanilla RNN

A recurrent network reads a sequence $x_1, \dots, x_T$ one element at a time and keeps a hidden state $h_t$ that summarizes what it has seen. The Elman RNN (Elman, 1990) updates

$$
a_t = W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h, \qquad h_t = \tanh(a_t), \qquad y_t = W_{hy}\, h_t + b_y ,
$$

starting from $h_0 = 0$ (or a learned vector). The same weights are used at every step, which is weight sharing in time, the analogue of a convolution's sharing in space, and it lets one network handle sequences of any length. How $y_t$ is used depends on the task: predicting $x_{t+1}$ (a language model), labelling each step (tagging), or reading only the final $h_T$ (classifying the whole sequence).

**Parameters.** $W_{xh}$ is $H \times X$, $W_{hh}$ is $H \times H$ and $b_h$ has $H$ entries, so the recurrent part has $H(H + X) + H$ parameters. With $X = 128$ and $H = 256$: $256 \cdot 384 + 256 = 98{,}560$. PyTorch's `nn.RNN` has two bias vectors (`bias_ih` and `bias_hh`, which add up to one effective bias), so it reports $98{,}816$.

**Unrolling.** Written out over $T$ steps, an RNN is a $T$-layer feed-forward network in which every layer has the same weights. Backpropagation through this unrolled graph is called backpropagation through time (BPTT). Unlike a convolution, whose receptive field is fixed by the architecture, the state can in principle carry information from any earlier step. Whether training can make use of that is the vanishing gradient question.

### Backpropagation through time, derived

Let the total loss be a sum of per-step losses, $\mathcal{L} = \sum_{t=1}^{T} \ell_t(y_t)$. Define the total gradient with respect to each hidden state, $\bar h_t = \partial\mathcal{L}/\partial h_t$, counting every way $h_t$ affects the loss, and the gradient with respect to the pre-activation, $\bar a_t = \partial\mathcal{L}/\partial a_t$.

**Step 1: the backward recursion.** $h_t$ affects the loss in two ways: through its own output $y_t$, and through the next state, since $a_{t+1} = W_{hh}h_t + \dots$. By the chain rule,

$$
\bar h_t = W_{hy}^\top \frac{\partial \ell_t}{\partial y_t} + W_{hh}^\top\, \bar a_{t+1}, \qquad
\bar a_t = \big(1 - h_t^2\big) \odot \bar h_t ,
$$

with $\bar a_{T+1} = 0$, because nothing follows the last step. The factor $1 - h_t^2$ is $\tanh'(a_t)$ written in terms of the output. The recursion runs from $t = T$ down to $t = 1$.

**Step 2: parameter gradients.** Each weight is used at every step, so its gradient is the sum of the per-step contributions:

$$
\frac{\partial\mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \bar a_t\, h_{t-1}^\top, \qquad
\frac{\partial\mathcal{L}}{\partial W_{xh}} = \sum_{t=1}^{T} \bar a_t\, x_t^\top, \qquad
\frac{\partial\mathcal{L}}{\partial b_h} = \sum_{t=1}^{T} \bar a_t, \qquad
\frac{\partial\mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial y_t}\, h_t^\top .
$$

Each term has the shape of a dense layer's weight gradient: upstream gradient times the layer's input, transposed.

**A two-step example by hand.** Take a scalar RNN with no output layer: $W_{hh} = 0.5$, $W_{xh} = 1$, no bias, $h_0 = 0$, inputs $x_1 = 1$, $x_2 = 0$, and loss $\mathcal{L} = h_2$.

- Forward: $a_1 = 1$, $h_1 = \tanh 1 = 0.7616$; $a_2 = 0.5 \cdot 0.7616 = 0.3808$, $h_2 = \tanh 0.3808 = 0.3634$.
- Backward: $\bar h_2 = 1$, so $\bar a_2 = 1 - 0.3634^2 = 0.8679$. Then $\bar h_1 = W_{hh}\bar a_2 = 0.4340$ (there is no $\ell_1$), and $\bar a_1 = (1 - 0.7616^2) \cdot 0.4340 = 0.4200 \cdot 0.4340 = 0.1823$.
- Gradients: $\partial\mathcal{L}/\partial W_{hh} = \bar a_2 h_1 + \bar a_1 h_0 = 0.8679 \cdot 0.7616 = 0.6610$, and $\partial\mathcal{L}/\partial W_{xh} = \bar a_2 x_2 + \bar a_1 x_1 = 0.1823$.

PyTorch's autograd gives the same two numbers. Note how the input at step 1 reaches the loss only after passing through a factor $W_{hh} = 0.5$ and two $\tanh'$ factors: $0.1823 = 0.4200 \cdot 0.5 \cdot 0.8679$. Longer sequences multiply more such factors.

### Why gradients vanish or explode

Unroll the recursion to see how the loss at step $t$ reaches an earlier state $h_k$. Since $h_m = \tanh(W_{hh}h_{m-1} + \dots)$, the one-step Jacobian is

$$
J_m = \frac{\partial h_m}{\partial h_{m-1}} = \operatorname{diag}\big(\tanh'(a_m)\big)\, W_{hh},
$$

and by the chain rule

$$
\frac{\partial \ell_t}{\partial h_k} = \left(\frac{\partial h_t}{\partial h_k}\right)^{\!\top} \frac{\partial \ell_t}{\partial h_t}, \qquad
\frac{\partial h_t}{\partial h_k} = J_t\, J_{t-1} \cdots J_{k+1} = \prod_{m=k+1}^{t} \operatorname{diag}\big(\tanh'(a_m)\big)\, W_{hh} .
$$

The gradient of $W_{hh}$ is a sum over all pairs $k \le t$ of terms containing this product, so it is a sum over time lags $t - k$. The question is how the product behaves for large lags.

**Upper bound (vanishing).** The operator norm is submultiplicative, and $\lvert\tanh'\rvert \le 1$, so

$$
\left\lVert \frac{\partial h_t}{\partial h_k} \right\rVert_2 \le \prod_{m=k+1}^{t} \big\lVert \operatorname{diag}(\tanh'(a_m))\big\rVert_2 \, \lVert W_{hh}\rVert_2 \le \big(\gamma\, \lVert W_{hh}\rVert_2\big)^{t-k},
$$

where $\gamma = \max\lvert \tanh' \rvert \le 1$ over the steps involved. If $\lVert W_{hh}\rVert_2 < 1/\gamma$, the long-lag terms shrink geometrically. Pascanu, Mikolov and Bengio (2013) state it this way: a largest singular value below $1/\gamma$ is *sufficient* for vanishing, and so above $1/\gamma$ is *necessary* for exploding. Saturation makes it worse: a unit at $h = 0.9$ has $\tanh' = 1 - 0.81 = 0.19$, so five such steps contribute $0.19^5 \approx 0.00025$ even with a well-scaled $W_{hh}$.

**The linear case (exploding).** With the nonlinearity removed, $\partial h_t/\partial h_k = W_{hh}^{\,t-k}$. If $W_{hh} = Q\Lambda Q^{-1}$ is diagonalizable, $W_{hh}^n = Q\Lambda^n Q^{-1}$, so the component of the gradient along eigenvector $i$ is scaled by $\lambda_i^n$. The spectral radius $\rho(W_{hh})$ decides: below 1 every component vanishes; above 1 the components along the large eigenvalues explode. With $W_{hh} = 0.9I$ over 50 steps the factor is $0.9^{50} \approx 0.005$, and with $1.1I$ it is $1.1^{50} \approx 117$. A 20% change in one number changes the 50-step gradient by a factor of over 20,000.

**Consequences.** When gradients vanish, the total gradient is dominated by short lags, so the network learns short-range dependencies and cannot learn long ones, even though the state could carry them (Bengio, Simard and Frasconi, 1994, analysed this; Hochreiter's 1991 thesis had identified it). When they explode, the loss surface has steep "cliffs" (Pascanu et al.): a step that lands on a cliff produces a huge gradient that throws the parameters far away.

### Fixes: clipping, truncation and initialization

**Gradient clipping** handles explosions. Global-norm clipping rescales the whole gradient vector $g$ when its norm exceeds a threshold $c$:

$$
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g\rVert}\right),
$$

which keeps the direction and bounds the step. Pascanu et al. (2013) proposed it for RNNs; Mikolov had clipped each component separately, which changes the direction. A threshold near the typical gradient norm seen in normal training is a common heuristic. Clipping does nothing for vanishing gradients: it can only shrink a gradient, never restore information that the product of Jacobians has already erased. Clipping in general is covered in [deep learning training](deep-learning-training.html).

**Truncated BPTT** handles length. Full BPTT stores all $T$ hidden states (memory $O(TH)$ per sequence) and does a backward pass over the whole sequence before every update. Truncated BPTT splits a long stream into chunks of $K$ steps, carries the hidden state forward from chunk to chunk, but stops gradients at chunk boundaries (in PyTorch, `h = h.detach()` between chunks). Memory becomes $O(KH)$ and updates come every $K$ steps. The price is that gradients never connect events more than $K$ steps apart, so dependencies longer than $K$ can only be learned by accident. Forgetting the `detach` makes the graph, and the memory, grow without bound across chunks.

**Initialization and activation.** An orthogonal $W_{hh}$ has all singular values equal to 1, so the linear part of each Jacobian preserves norms at the start of training (Saxe et al., 2014). Le, Jaitly and Hinton (2015) initialized $W_{hh} = I$ with ReLU units (the IRNN). These help at initialization; the lasting fix was to change the architecture so that the state is updated additively, which is what the LSTM does.

::: {.qa asked="ByteDance algorithm intern, Nowcoder 2020"}
**What causes vanishing and exploding gradients, and how do you handle them?**

Backpropagation multiplies one Jacobian per layer, or per time step in an RNN, so the gradient changes geometrically with depth. In an RNN, $\partial h_t/\partial h_k = \prod_{m=k+1}^t \operatorname{diag}(\tanh'(a_m))W_{hh}$, whose norm is at most $(\gamma\lVert W_{hh}\rVert_2)^{t-k}$ with $\gamma = \max\lvert\tanh'\rvert \le 1$: a largest singular value below $1/\gamma$ is sufficient for vanishing, and above $1/\gamma$ is necessary for exploding (Pascanu et al., 2013); for a linear RNN the spectral radius decides, $0.9^{50} \approx 0.005$ and $1.1^{50} \approx 117$. Saturating activations ($\sigma' \le 1/4$, $\tanh' \to 0$) and badly scaled weights are the usual causes in deep feed-forward networks. Diagnose by logging per-layer gradient norms: vanishing shows as tiny norms in early layers and a loss that plateaus from the start; exploding as spikes, then NaNs. Fixes: ReLU-family activations, variance-preserving (or orthogonal) initialization, normalization layers, residual connections, gating (LSTM and GRU, whose additive cell update gives a direct path with Jacobian $\operatorname{diag}(f_t)$), and for explosions gradient-norm clipping, smaller learning rates and truncated BPTT. Clipping cannot fix vanishing gradients.
:::

::: qa
**Write the BPTT recursion for a vanilla RNN and the gradient of $W_{hh}$.**

With $a_t = W_{hh}h_{t-1} + W_{xh}x_t + b$ and $h_t = \tanh(a_t)$: $\bar h_t = W_{hy}^\top\,\partial\ell_t/\partial y_t + W_{hh}^\top \bar a_{t+1}$, $\bar a_t = (1 - h_t^2)\odot\bar h_t$, run from $t = T$ down with $\bar a_{T+1} = 0$. Then $\partial\mathcal{L}/\partial W_{hh} = \sum_t \bar a_t h_{t-1}^\top$, summed because the weight is shared across steps. Unrolling the recursion shows the product of Jacobians $\operatorname{diag}(\tanh'(a_m))W_{hh}$ responsible for vanishing and exploding gradients.
:::

::: qa
**Why does gradient clipping fix exploding gradients but not vanishing ones?**

Clipping rescales a gradient whose norm exceeds a threshold, keeping its direction, so a cliff in the loss surface produces a bounded step instead of a jump. Vanishing gradients are too small, not too large, and the information about long-range dependencies has already been multiplied away inside the product of Jacobians; no rescaling of the total gradient can recover the long-lag terms relative to the short-lag ones. That needs an architectural change (gating, residual paths) or a better initialization.
:::

## LSTM and GRU

### The LSTM equations

The long short-term memory (LSTM) network of Hochreiter and Schmidhuber (1997) adds a second state, the **cell** $c_t$, updated additively and controlled by gates. In the form used today (with the forget gate added by Gers, Schmidhuber and Cummins, 2000), with $[h_{t-1}; x_t]$ feeding four affine maps:

$$
\begin{aligned}
i_t &= \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i) && \text{input gate: how much new content to write} \\
f_t &= \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f) && \text{forget gate: how much old memory to keep} \\
g_t &= \tanh(W_{xg}x_t + W_{hg}h_{t-1} + b_g) && \text{candidate content} \\
o_t &= \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o) && \text{output gate: how much memory to expose} \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t && \text{additive cell update} \\
h_t &= o_t \odot \tanh(c_t) && \text{hidden state, the layer's output.}
\end{aligned}
$$

The gates are sigmoids, so each entry lies in $(0, 1)$ and acts as a soft switch per unit; the candidate is a $\tanh$ in $(-1, 1)$. The cell is the long-term memory, and $h_t$ is what the rest of the network sees. The 1997 LSTM had no forget gate, so its cell was $c_t = c_{t-1} + i_t \odot g_t$: the "constant error carousel", whose Jacobian along the cell is exactly the identity. The forget gate lets the network reset its memory, which the original could not do on long continuous streams.

### Why the additive cell path eases vanishing gradients

Differentiate $c_t$ with respect to $c_{t-1}$. The cell depends on $c_{t-1}$ in two ways: directly, through $f_t \odot c_{t-1}$, and indirectly, through $h_{t-1} = o_{t-1} \odot \tanh(c_{t-1})$, which feeds all the gates and the candidate at step $t$.

**Direct path.** Holding $h_{t-1}$ fixed, the gates do not change, and

$$
\left.\frac{\partial c_t}{\partial c_{t-1}}\right|_\text{direct} = \operatorname{diag}(f_t) .
$$

**Full Jacobian.** By the product rule on each term of $c_t = f_t \odot c_{t-1} + i_t \odot g_t$, with the gates' dependence on $h_{t-1}$ included:

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) + \Big[\operatorname{diag}(c_{t-1})\frac{\partial f_t}{\partial h_{t-1}} + \operatorname{diag}(g_t)\frac{\partial i_t}{\partial h_{t-1}} + \operatorname{diag}(i_t)\frac{\partial g_t}{\partial h_{t-1}}\Big]\frac{\partial h_{t-1}}{\partial c_{t-1}},
$$

where $\partial f_t/\partial h_{t-1} = \operatorname{diag}(f_t(1 - f_t))W_{hf}$, $\partial i_t/\partial h_{t-1} = \operatorname{diag}(i_t(1-i_t))W_{hi}$, $\partial g_t/\partial h_{t-1} = \operatorname{diag}(1 - g_t^2)W_{hg}$, and $\partial h_{t-1}/\partial c_{t-1} = \operatorname{diag}\big(o_{t-1}\odot(1 - \tanh^2 c_{t-1})\big)$.

The bracketed terms look like a vanilla RNN's Jacobian: weight matrices times derivatives of squashing functions, which can shrink or grow geometrically. The first term is different. Over many steps, the direct path contributes

$$
\left.\frac{\partial c_t}{\partial c_k}\right|_\text{direct} = \prod_{m=k+1}^{t} \operatorname{diag}(f_m) = \operatorname{diag}\Big(\prod_{m=k+1}^{t} f_m\Big),
$$

an elementwise product of forget-gate values. Three things are different from the RNN's $\prod_m \operatorname{diag}(\tanh'(a_m))W_{hh}$:

- there is no repeated weight matrix, so no spectral radius compounds;
- there is no $\tanh'$ factor on this path, because the cell update has no squashing;
- the factors are gates, computed from the data, so the network can learn to hold $f \approx 1$ on units that must remember and let other units forget.

With $f = 0.99$ the 100-step factor is $0.99^{100} = 0.37$; with $f = 0.9$ it is $0.9^{100} = 2.7 \times 10^{-5}$. This is also why the forget-gate bias is often initialized to 1 (Gers et al., 2000; Jozefowicz, Zaremba and Sutskever, 2015, recommend it for every LSTM): with zero pre-activation, $\sigma(0) = 0.5$ gives $0.5^{20} \approx 10^{-6}$ over 20 steps, while $\sigma(1) = 0.731$ gives $0.0019$, about 2,000 times more gradient at the start of training. PyTorch initializes all biases uniformly, so you set this yourself.

The idea is the same as a residual connection: $c_t = c_{t-1} + (\text{update})$ when $f = 1$ is exactly a residual update in time, and highway networks (Srivastava et al., 2015), the precursor of ResNets, were explicitly modelled on LSTM gating. The LSTM *eases* the vanishing gradient problem rather than solving it: gradients still pass through the multiplicative indirect paths, forget gates below 1 still decay, and exploding gradients remain possible, so clipping is still used.

### Parameter count

Each of the four gate blocks ($i$, $f$, $g$, $o$) has an $H \times X$ input matrix, an $H \times H$ recurrent matrix and an $H$-vector bias:

$$
\#\text{params}_\text{LSTM} = 4\big(H(H + X) + H\big).
$$

Worked: $X = 128$, $H = 256$ gives $4(256 \cdot 384 + 256) = 4 \cdot 98{,}560 = 394{,}240$. PyTorch's `nn.LSTM(128, 256)` reports $395{,}264 = 4(H(H+X) + 2H)$, because it keeps two bias vectors per gate. Each time step costs $4H(H+X) = 393{,}216$ MACs per sequence in the batch, four times a vanilla RNN of the same size. PyTorch stores the four blocks stacked in the order input, forget, cell (candidate), output, which you need to know when initializing the forget bias or porting weights.

### GRU

The gated recurrent unit (Cho et al., 2014) merges the cell and hidden state and uses two gates. In the paper's convention:

$$
\begin{aligned}
r_t &= \sigma(W_{xr}x_t + W_{hr}h_{t-1} + b_r) && \text{reset gate} \\
z_t &= \sigma(W_{xz}x_t + W_{hz}h_{t-1} + b_z) && \text{update gate} \\
\tilde h_t &= \tanh\big(W_{xh}x_t + W_{hh}(r_t \odot h_{t-1}) + b_h\big) && \text{candidate state} \\
h_t &= z_t \odot h_{t-1} + (1 - z_t) \odot \tilde h_t && \text{interpolate old and new.}
\end{aligned}
$$

The update gate does the work of the LSTM's forget and input gates together, coupled so that keeping old content ($z \to 1$) and writing new content ($1 - z$) trade off exactly. The reset gate lets the candidate ignore the previous state, for example at the start of a new phrase. The direct-path Jacobian is $\partial h_t/\partial h_{t-1}|_\text{direct} = \operatorname{diag}(z_t)$, the same additive mechanism as the LSTM's $\operatorname{diag}(f_t)$.

Two convention traps. Some texts swap the roles, writing $h_t = (1 - z_t)\odot h_{t-1} + z_t \odot \tilde h_t$; this is the same model with the gate relabelled $z \leftrightarrow 1 - z$. PyTorch's `nn.GRU` matches Cho et al.'s interpolation but applies the reset gate *after* the recurrent matrix, $\tanh(W_{xh}x_t + b_{xh} + r_t \odot (W_{hh}h_{t-1} + b_{hh}))$, a variant chosen for efficiency (cuDNN uses it), so its outputs differ from the paper's formula with the same weights.

**Parameters.** Three blocks instead of four: $3(H(H+X) + H) = 295{,}680$ for $X = 128$, $H = 256$, or $296{,}448$ in PyTorch with its two biases.

**LSTM or GRU?** The GRU has three quarters of the parameters and compute, one state instead of two, and no output gate, so its whole memory is always exposed. The LSTM's separate cell lets it keep information that it does not currently output. Empirically neither wins consistently: Chung et al. (2014) found the two comparable and both better than a tanh RNN; Jozefowicz et al. (2015) found that an LSTM with forget-gate bias 1 matched the GRU on most of their tasks; Greff et al. (2017) compared eight LSTM variants and found none clearly better than the standard one, with the forget gate and the output activation the most important parts. A reasonable default is the LSTM, with the GRU when parameters or speed are tight.

### Bidirectional and stacked RNNs

A **bidirectional** RNN (Schuster and Paliwal, 1997) runs one RNN forward over $x_1, \dots, x_T$ and another backward over $x_T, \dots, x_1$, and represents step $t$ by the concatenation $[\overrightarrow{h}_t; \overleftarrow{h}_t]$ of size $2H$. Each step then sees both its past and its future, which helps tagging, speech recognition and encoders in translation. It needs the whole sequence in advance, so it cannot generate text left to right.

A **stacked** (deep) RNN feeds the output sequence of layer $\ell - 1$ as the input sequence of layer $\ell$. Layers above the first have input size $H$, or $2H$ if the layer below is bidirectional. Dropout is usually applied between layers but not on the recurrent connections (Zaremba et al., 2014).

**Worked parameter count:** `nn.LSTM(128, 256, num_layers=2, bidirectional=True)`. Layer 1, per direction: $4(256 \cdot 384 + 2 \cdot 256) = 395{,}264$, so $790{,}528$ for both directions. Layer 2 reads the $2 \cdot 256 = 512$-dimensional concatenation, so per direction $4(256 \cdot (512 + 256) + 512) = 788{,}480$, and $1{,}576{,}960$ for both. Total $2{,}367{,}488$, which matches PyTorch.

::: key
**How to say it in an interview.**

In a vanilla RNN the gradient across $n$ steps is a product of $n$ Jacobians $\operatorname{diag}(\tanh')W_{hh}$, so it scales roughly like the $n$-th power of $W_{hh}$'s largest singular value or spectral radius, and vanishes or explodes. The LSTM adds a cell updated additively, $c_t = f_t \odot c_{t-1} + i_t\odot g_t$. Along the cell the Jacobian is just $\operatorname{diag}(f_t)$: no weight matrix and no squashing derivative, and the forget gate is learned, so the network can keep it near 1 where it needs memory. That is the same idea as a residual connection, in time. It eases vanishing gradients but does not remove them, and clipping is still needed for explosions.
:::

::: qa
**Derive $\partial c_t/\partial c_{t-1}$ for an LSTM and explain why it helps.**

$c_t = f_t\odot c_{t-1} + i_t \odot g_t$. Holding $h_{t-1}$ fixed, the gates are constant and the direct Jacobian is $\operatorname{diag}(f_t)$. The full Jacobian adds terms through $h_{t-1} = o_{t-1}\odot\tanh(c_{t-1})$, which feeds $f_t$, $i_t$ and $g_t$; those terms involve weight matrices and squashing derivatives like a vanilla RNN. Over many steps the direct path gives $\operatorname{diag}(\prod_m f_m)$, a product of learned, data-dependent numbers near 1 when the network wants to remember, instead of the repeated $\operatorname{diag}(\tanh')W_{hh}$ whose spectrum compounds. Example: $0.99^{100} = 0.37$, while $0.9^{100} = 2.7\times10^{-5}$.
:::

::: qa
**How many parameters does an LSTM layer with input size 128 and hidden size 256 have? And a GRU?**

Four gate blocks, each with $H \times X$ input weights, $H \times H$ recurrent weights and an $H$ bias: $4(H(H+X) + H) = 4(256\cdot384 + 256) = 394{,}240$. PyTorch reports $395{,}264$ because it has two bias vectors per gate. The GRU has three blocks: $295{,}680$, or $296{,}448$ in PyTorch. Stacking a second bidirectional layer changes its input size to $2H$.
:::

::: qa
**Compare LSTM and GRU.**

The LSTM has a separate cell and three gates (input, forget, output), four weight blocks; the GRU merges cell and hidden state and has two gates (update, reset), three blocks, so 25% fewer parameters and less compute. The GRU's update gate couples forgetting and writing ($z$ and $1 - z$); it has no output gate, so all of its memory is exposed. Both get their long-range ability from an additive state update with a gated direct Jacobian ($\operatorname{diag}(f_t)$ or $\operatorname{diag}(z_t)$). Benchmarks (Chung et al., 2014; Jozefowicz et al., 2015; Greff et al., 2017) show no consistent winner, especially once the LSTM's forget bias is initialized to 1.
:::

## Sequence-to-sequence models

### Encoder–decoder

Translation maps a sequence to a sequence of a different length. The RNN encoder–decoder (Cho et al., 2014; Sutskever, Vinyals and Le, 2014) uses two RNNs. The encoder reads the source $x_1, \dots, x_S$ and its final state becomes a fixed-size summary $v$. The decoder is a conditional language model started from $v$:

$$
p(y_1, \dots, y_T \mid x) = \prod_{t=1}^{T} p(y_t \mid y_1, \dots, y_{t-1}, v),
$$

where each factor is a softmax over the vocabulary computed from the decoder state, and generation stops at an end-of-sequence token. Sutskever et al. used 4-layer LSTMs with 1000 cells per layer (384M parameters per model). Their most useful trick was to feed the source sentence *reversed*: the first source words then sit close to the first target words they align with, which shortens the dependencies the gradients must cover; it lowered test perplexity from 5.8 to 4.7 and raised BLEU from 25.9 to 30.6. An ensemble of 5 reversed LSTMs with beam size 12 scored 34.81 BLEU on WMT'14 English to French.

The weakness is the bottleneck: the whole source must fit into one vector $v$, and translation quality of such models fell as sentences got longer. Attention, below, removed it.

### Teacher forcing and exposure bias

**Teacher forcing** trains the decoder with the ground-truth previous token as input, not its own prediction: the loss is $-\sum_t \log p(y_t^\star \mid y_{<t}^\star, x)$ with $y^\star$ the reference. This is exactly maximum likelihood for the factorization above, it is stable, and every step gets a correct context. Autoregressive transformers are trained the same way; for them it also means all positions can be computed in parallel.

**Exposure bias** (the term is from Ranzato et al., 2016) is the resulting mismatch: in training the decoder only ever sees correct prefixes, while at test time it conditions on its own outputs, so one early mistake puts it in states it never saw, and errors can compound. Remedies include scheduled sampling (Bengio et al., 2015), which feeds the model's own samples with a probability that increases during training, and sequence-level training that optimizes a test metric of whole sampled outputs (Ranzato et al., 2016).

### Beam search

At test time we want $\arg\max_y p(y \mid x)$, which is intractable: the number of sequences grows like $V^T$. **Greedy decoding** picks the most likely token at each step, which can miss the best sequence. A two-step example with tokens "red", "blue" and an end token:

| Prefix | Next-token probabilities |
|---|---|
| (start) | red: 0.6, blue: 0.4 |
| red | red: 0.3, blue: 0.3, end: 0.4 |
| blue | red: 0.05, blue: 0.05, end: 0.9 |

Greedy picks red (0.6), then end (0.4): probability $0.24$. The sequence "blue end" has $0.4 \cdot 0.9 = 0.36$. **Beam search** with beam width $B$ keeps the $B$ highest-scoring partial sequences at each step, scored by summed log-probabilities. With $B = 2$: after step 1 the beam is {red: 0.6, blue: 0.4}; the six extensions score red red 0.18, red blue 0.18, red end 0.24, blue red 0.02, blue blue 0.02, blue end 0.36; the top two are "blue end" and "red end", both finished, so beam search returns "blue end", the true optimum here.

Practical details:

- Finished hypotheses (those that emitted the end token) are set aside, and the search stops when $B$ of them are finished or a maximum length is reached. Cost: $B$ decoder steps per position plus a top-$B$ selection over $B \times V$ candidates.
- Log-probabilities are negative and add up, so the raw score favours short outputs. Length normalization divides by a function of the length, such as $\lvert y\rvert^\alpha$, or GNMT's $\big((5 + \lvert y\rvert)/6\big)^\alpha$ (Wu et al., 2016), with $\alpha$ tuned.
- Beam search is still approximate, and larger beams do not always help: translation quality often drops with very large beams (Koehn and Knowles, 2017). For open-ended text generation, beam search tends to produce repetitive, bland text, and sampling methods are preferred (see [LLMs](llms.html)).

### Bahdanau attention: the bridge to transformers

Bahdanau, Cho and Bengio (2015) let the decoder look back at all encoder states instead of one summary. The encoder is a bidirectional RNN, giving one annotation per source position, $h_j = [\overrightarrow{h}_j; \overleftarrow{h}_j]$. At decoder step $t$, with previous decoder state $s_{t-1}$:

$$
\begin{aligned}
e_{tj} &= v_a^\top \tanh\big(W_a s_{t-1} + U_a h_j\big) && \text{alignment score of position } j \\
\alpha_{tj} &= \frac{\exp(e_{tj})}{\sum_{j'=1}^{S}\exp(e_{tj'})} && \text{softmax over source positions} \\
c_t &= \sum_{j=1}^{S} \alpha_{tj}\, h_j && \text{context vector for this step} \\
s_t &= f(s_{t-1}, y_{t-1}, c_t) && \text{decoder state update} \\
p(y_t \mid y_{<t}, x) &= g(y_{t-1}, s_t, c_t) && \text{output distribution.}
\end{aligned}
$$

The score is a small one-hidden-layer network ("additive" attention), $W_a$, $U_a$ and $v_a$ are learned, and $f$ is the decoder RNN (the gated unit of Cho et al. in the paper). Worked: scores $e = [2, 0, -1]$ over three source positions give weights $e^2, e^0, e^{-1}$ normalized by $7.389 + 1 + 0.368 = 8.757$, that is $\alpha = [0.844, 0.114, 0.042]$; with scalar annotations $h = [1, 2, 3]$ the context is $0.844 + 0.228 + 0.126 = 1.198$.

What changed: the context is recomputed for every output word, so nothing has to be squeezed into one vector; the weights $\alpha_{tj}$ form a soft alignment that can be plotted and inspected; and every source position is one step from the output in the backward pass, so gradients no longer cross the whole sequence. The paper's attention model kept its quality on long sentences where the plain encoder–decoder fell off. Luong, Pham and Manning (2015) simplified the score to a dot product $s_t^\top h_j$ or a bilinear form $s_t^\top W h_j$. The transformer (Vaswani et al., 2017) keeps exactly this read operation, a softmax over scores followed by a weighted sum of values, uses scaled dot-product scores, removes the recurrence, and applies attention within each sequence as well as between them (self-attention).

### Why transformers replaced RNNs

An RNN must compute $h_t$ before $h_{t+1}$, so training takes $O(T)$ sequential steps per layer even with teacher forcing, which uses GPUs poorly; a transformer computes all positions of a training sequence in parallel. Information between positions $i$ and $j$ passes through $\lvert i - j\rvert$ state updates in an RNN, with the vanishing gradients that implies, but through one attention step in a transformer. And an RNN must compress the whole past into a fixed-size state, while a transformer keeps every past token's keys and values. The costs go the other way: attention is quadratic in sequence length, and the key–value cache grows with the context, while an RNN needs constant memory and compute per generated token. Details are on [transformers](transformers.html).

**State-space models.** That last point brought recurrence back. S4 (Gu, Goel and Ré, 2022) uses a linear state-space layer, $h_t = \bar A h_{t-1} + \bar B x_t$, $y_t = C h_t$, obtained by discretizing a continuous-time system with a structured matrix $A$ chosen for long memory. Because it is linear and time-invariant, unrolling gives $y = \bar K * x$ with kernel $\bar K = (C\bar B, C\bar A\bar B, C\bar A^2\bar B, \dots)$, so it trains as one long convolution (computed with FFTs) and generates as a recurrence with constant memory per step. Mamba (Gu and Dao, 2023) makes $B$, $C$ and the step size $\Delta$ functions of the input, so the model can choose what to store and what to ignore, much like an LSTM's gates. That breaks the convolution form, so Mamba trains with a hardware-aware parallel scan, still linear in sequence length. The paper reports 5 times higher inference throughput than transformers, and a 3B-parameter Mamba matching transformers twice its size in language modelling. Hybrid models now interleave attention and SSM layers.

::: qa
**What is exposure bias, and how does teacher forcing cause it?**

Teacher forcing trains each step on the ground-truth prefix, which is maximum likelihood and is stable. At inference the model conditions on its own outputs, so after one mistake it is in a state distribution it never trained on, and errors can compound. Remedies: scheduled sampling (gradually feed the model's own samples during training), sequence-level objectives on sampled outputs, or simply strong models and decoding choices that keep early mistakes rare.
:::

::: qa
**Show with an example that greedy decoding can miss the most likely sequence, and how beam search helps.**

If $p(\text{red}) = 0.6$, $p(\text{blue}) = 0.4$, $p(\text{end}\mid\text{red}) = 0.4$ and $p(\text{end}\mid\text{blue}) = 0.9$, greedy decoding returns "red end" with probability 0.24, while "blue end" has 0.36. Beam search with width 2 keeps both first tokens, extends both, and keeps the best two of the six extensions, which finds "blue end". It is still a heuristic: it needs length normalization because summed log-probabilities favour short outputs, and very wide beams can hurt.
:::

::: qa
**Write Bahdanau attention and explain how it leads to the transformer.**

With encoder annotations $h_j$ and previous decoder state $s_{t-1}$: scores $e_{tj} = v_a^\top\tanh(W_a s_{t-1} + U_a h_j)$, weights $\alpha_{tj} = \operatorname{softmax}_j(e_{tj})$, context $c_t = \sum_j \alpha_{tj}h_j$, fed into the decoder update. It removes the fixed-vector bottleneck, gives a soft alignment, and puts every source position one step from the output. The transformer keeps the softmax-weighted read with dot-product scores ($q^\top k/\sqrt{d}$), drops the recurrence entirely, and uses attention within a sequence too, which makes training parallel over positions.
:::

## From-scratch code

Both snippets were run with Python 3 and PyTorch 2.11; the printed results are in the comments.

**2-D convolution, two ways.** The first uses `F.unfold` (im2col) and one matrix multiply, and supports stride, padding and dilation. The second uses explicit loops over output positions, the version to write on a whiteboard.

```python
import torch
import torch.nn.functional as F

def conv2d_unfold(x, w, b=None, stride=1, padding=0, dilation=1):
    # x: (B, C_in, H, W); w: (C_out, C_in, kh, kw). Cross-correlation, like F.conv2d.
    B, C_in, H, W = x.shape
    C_out, _, kh, kw = w.shape
    H_out = (H + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    cols = F.unfold(x, (kh, kw), dilation=dilation, padding=padding, stride=stride)
    # cols: (B, C_in*kh*kw, H_out*W_out), one column per output position
    out = w.reshape(C_out, -1) @ cols                 # (B, C_out, H_out*W_out)
    if b is not None:
        out = out + b[:, None]
    return out.reshape(B, C_out, H_out, W_out)

def conv2d_loops(x, w, b, stride=1, padding=0):
    x = F.pad(x, (padding,) * 4)
    B, C_in, H, W = x.shape
    C_out, _, kh, kw = w.shape
    H_out, W_out = (H - kh) // stride + 1, (W - kw) // stride + 1
    out = torch.empty(B, C_out, H_out, W_out)
    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]  # (B, C_in, kh, kw)
            out[:, :, i, j] = torch.einsum('bcuv,ocuv->bo', patch, w) + b
    return out

torch.manual_seed(0)
x, w, b = torch.randn(2, 3, 11, 9), torch.randn(5, 3, 3, 4), torch.randn(5)
for s, p, d in [(1, 0, 1), (2, 1, 1), (3, 2, 2)]:
    ref = F.conv2d(x, w, b, stride=s, padding=p, dilation=d)
    print(ref.shape, torch.allclose(conv2d_unfold(x, w, b, s, p, d), ref, atol=1e-5))
# torch.Size([2, 5, 9, 6]) True
# torch.Size([2, 5, 6, 4]) True
# torch.Size([2, 5, 4, 3]) True
print(torch.allclose(conv2d_loops(x, w, b, 2, 1), F.conv2d(x, w, b, 2, 1), atol=1e-5))  # True
```

The test uses a non-square kernel ($3 \times 4$) and a non-square image so that swapped height and width would be caught. Groups are a natural follow-up: reshape the weights per group and multiply each group's slice of `cols` separately.

::: {.qa asked="NVIDIA, Meta, Baidu, Xiaomi, Megvii (5 reports)"}
**Implement 2D convolution without calling a convolution layer.**

Use either version above. The unfold version builds the im2col matrix $(B, C_\text{in}k_hk_w, H_\text{out}W_\text{out})$ and multiplies by the weights reshaped to $(C_\text{out}, C_\text{in}k_hk_w)$; the loop version takes the dot product of each $(C_\text{in}, k_h, k_w)$ window with every filter. State $H_\text{out} = \lfloor (H + 2p - d(k_h - 1) - 1)/s\rfloor + 1$, that the operation is cross-correlation, and the cost in MACs, and check against `F.conv2d` with stride and padding using `torch.allclose`. If asked for the backward pass: the weight gradient is `F.unfold`'s columns times the upstream gradient, and the input gradient is a transposed convolution (`F.fold` of $W_\text{mat}^\top\Delta$).
:::

**An LSTM cell**, checked against `nn.LSTMCell` using its own weights:

```python
import torch
import torch.nn as nn

def lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh):
    # PyTorch layout: the 4H rows of W_ih, W_hh are stacked as [input, forget, cell, output]
    a = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh        # (B, 4H) pre-activations
    i, f, g, o = a.chunk(4, dim=1)
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
    g = torch.tanh(g)
    c_new = f * c + i * g                             # additive cell update
    h_new = o * torch.tanh(c_new)
    return h_new, c_new

torch.manual_seed(0)
B, X, H = 4, 10, 16
cell = nn.LSTMCell(X, H)
x, h, c = torch.randn(B, X), torch.randn(B, H), torch.randn(B, H)
h_ref, c_ref = cell(x, (h, c))
h_new, c_new = lstm_cell(x, h, c, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
print(torch.allclose(h_new, h_ref, atol=1e-6), torch.allclose(c_new, c_ref, atol=1e-6))  # True True
print(sum(p.numel() for p in cell.parameters()), 4 * (H * (H + X) + 2 * H))            # 1792 1792
```

One matrix multiply computes all four gates at once, which is how real implementations do it. A sequence model is a loop that calls the cell at each step and collects the $h_t$; for a forget-gate bias of 1, set `cell.bias_ih.data[H:2*H] = 1` and `cell.bias_hh.data[H:2*H] = 0`.

## Rapid-fire review

1. **Output size of a convolution?** $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$.
2. **Parameters of a $3\times3$ conv from 64 to 128 channels?** $9 \cdot 64 \cdot 128 = 73{,}728$ weights plus 128 biases.
3. **What does locality save, and what does weight sharing save?** Locality saves compute and parameters; sharing saves only parameters.
4. **Equivariant or invariant?** Convolutions are shift equivariant; global pooling makes the output invariant; stride and pooling break exact equivariance.
5. **Convolution versus correlation?** Correlation does not flip the kernel; frameworks compute correlation; with learned kernels it makes no difference.
6. **Receptive-field recursion?** $r_\ell = r_{\ell-1} + d_\ell(k_\ell - 1)j_{\ell-1}$, $j_\ell = j_{\ell-1}s_\ell$.
7. **RF of three stacked $3\times3$ convs?** 7. With dilations 1, 2, 4: 15.
8. **Depthwise-separable cost ratio?** $1/C_\text{out} + 1/k^2$, about $1/9$ for $3\times3$.
9. **What does a $1\times1$ convolution do?** A per-pixel linear map across channels: change width, mix channels, bottlenecks, projection shortcuts.
10. **Why checkerboard artifacts?** Transposed convolutions with $k$ not divisible by $s$ give uneven overlap; fix with resize-then-convolve.
11. **Pixel shuffle?** Predict $Cr^2$ channels at low resolution and rearrange them into $r \times r$ blocks; equivalent to a transposed convolution with kernel $rk$.
12. **Max-pool backward?** All the gradient goes to the argmax of each window.
13. **Weight gradient of a convolution?** Correlation of the input with the upstream gradient, summed over the batch.
14. **Input gradient?** Full convolution of the upstream gradient with the kernel (flipped kernel, swapped channels), a transposed convolution when strided.
15. **When does FFT convolution pay off?** Large kernels; for $3\times3$, Winograd ($16$ vs $36$ multiplies for $F(2\times2, 3\times3)$) or GEMM.
16. **AlexNet's ingredients?** ReLU, dropout, two GPUs, augmentation, overlapping pooling; 15.3% top-5.
17. **VGG's argument?** Three $3\times3$ = one $7\times7$ in receptive field, $27C^2$ vs $49C^2$ weights, two more ReLUs.
18. **Inception's trick?** $1\times1$ reductions before expensive branches; module 3a costs 128M MACs.
19. **What did ResNet fix?** Degradation: deeper plain nets had higher training error.
20. **ResNet-50's parameters?** 25.6M; 58.6% in the last stage; 3.8G multiply-adds.
21. **DenseNet versus ResNet?** Concatenate versus add; growth rate $k$; more memory, fewer parameters.
22. **MobileNet v2 block?** Expand $1\times1$, depthwise $3\times3$, linear $1\times1$ projection, residual between thin ends.
23. **EfficientNet's constraint?** $\alpha\beta^2\gamma^2 \approx 2$, so FLOPs roughly double per unit of $\phi$.
24. **Why U-Nets for diffusion?** Same-shape pixel-aligned output, multi-scale context plus detail through skips, convolutional priors, easy timestep conditioning.
25. **RNN Jacobian across $n$ steps?** Product of $\operatorname{diag}(\tanh'(a_m))W_{hh}$; vanishes if $\lVert W_{hh}\rVert_2 < 1/\gamma$.
26. **LSTM direct cell Jacobian?** $\operatorname{diag}(f_t)$.
27. **LSTM parameters?** $4(H(H+X) + H)$; PyTorch adds a second bias.
28. **Why forget-bias 1?** Start with $f \approx 0.73$ instead of $0.5$, so early gradients survive many steps.
29. **Exposure bias?** Train on true prefixes, test on generated ones.
30. **Bahdanau attention?** Additive scores, softmax over source positions, context vector per output step.
31. **Why transformers replaced RNNs?** Parallel training over positions, one-step paths between tokens, no fixed-size state bottleneck.

## References

1. Aitken, Ledig, Theis, Caballero, Wang, Shi. Checkerboard artifact free sub-pixel convolution. 2017.
2. Azulay, Weiss. Why do deep convolutional networks generalize so poorly to small image transformations? JMLR, 2019.
3. Bahdanau, Cho, Bengio. Neural machine translation by jointly learning to align and translate. ICLR, 2015.
4. Bengio, Simard, Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 1994.
5. Bengio, Vinyals, Jaitly, Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. NeurIPS, 2015.
6. Chetlur et al. cuDNN: efficient primitives for deep learning. 2014.
7. Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk, Bengio. Learning phrase representations using RNN encoder–decoder for statistical machine translation. EMNLP, 2014.
8. Chung, Gulcehre, Cho, Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. 2014.
9. Cohen, Welling. Group equivariant convolutional networks. ICML, 2016.
10. Dai et al. Deformable convolutional networks. ICCV, 2017.
11. Elman. Finding structure in time. Cognitive Science, 1990.
12. Gers, Schmidhuber, Cummins. Learning to forget: continual prediction with LSTM. Neural Computation, 2000.
13. Goyal et al. Accurate, large minibatch SGD: training ImageNet in 1 hour. 2017.
14. Greff, Srivastava, Koutník, Steunebrink, Schmidhuber. LSTM: a search space odyssey. IEEE TNNLS, 2017.
15. Gu, Dao. Mamba: linear-time sequence modeling with selective state spaces. 2023.
16. Gu, Goel, Ré. Efficiently modeling long sequences with structured state spaces. ICLR, 2022.
17. He, Zhang, Ren, Sun. Deep residual learning for image recognition. CVPR, 2016.
18. Ho, Jain, Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020.
19. Hochreiter, Schmidhuber. Long short-term memory. Neural Computation, 1997.
20. Howard et al. MobileNets: efficient convolutional neural networks for mobile vision applications. 2017.
21. Huang, Liu, van der Maaten, Weinberger. Densely connected convolutional networks. CVPR, 2017.
22. Islam, Jia, Bruce. How much position information do convolutional neural networks encode? ICLR, 2020.
23. Johnson, Alahi, Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. ECCV, 2016.
24. Jozefowicz, Zaremba, Sutskever. An empirical exploration of recurrent network architectures. ICML, 2015.
25. Koehn, Knowles. Six challenges for neural machine translation. 2017.
26. Krizhevsky, Sutskever, Hinton. ImageNet classification with deep convolutional neural networks. NeurIPS, 2012.
27. Lavin, Gray. Fast algorithms for convolutional neural networks. CVPR, 2016.
28. Le, Jaitly, Hinton. A simple way to initialize recurrent networks of rectified linear units. 2015.
29. LeCun et al. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1989.
30. LeCun, Bottou, Bengio, Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 1998.
31. Lin, Chen, Yan. Network in network. 2013.
32. Liu, Mao, Wu, Feichtenhofer, Darrell, Xie. A ConvNet for the 2020s. CVPR, 2022.
33. Luo, Li, Urtasun, Zemel. Understanding the effective receptive field in deep convolutional neural networks. NeurIPS, 2016.
34. Luong, Pham, Manning. Effective approaches to attention-based neural machine translation. EMNLP, 2015.
35. Mathieu, Henaff, LeCun. Fast training of convolutional networks through FFTs. ICLR, 2014.
36. Odena, Dumoulin, Olah. Deconvolution and checkerboard artifacts. Distill, 2016.
37. Pascanu, Mikolov, Bengio. On the difficulty of training recurrent neural networks. ICML, 2013.
38. Pleiss, Chen, Huang, Li, van der Maaten, Weinberger. Memory-efficient implementation of DenseNets. 2017.
39. Ranzato, Chopra, Auli, Zaremba. Sequence level training with recurrent neural networks. ICLR, 2016.
40. Ronneberger, Fischer, Brox. U-Net: convolutional networks for biomedical image segmentation. MICCAI, 2015.
41. Sandler, Howard, Zhu, Zhmoginov, Chen. MobileNetV2: inverted residuals and linear bottlenecks. CVPR, 2018.
42. Saxe, McClelland, Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR, 2014.
43. Schuster, Paliwal. Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 1997.
44. Shi, Caballero, Huszár, Totz, Aitken, Bishop, Rueckert, Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. CVPR, 2016.
45. Shi, Caballero, Theis, Huszár, Aitken, Ledig, Wang. Is the deconvolution layer the same as a convolutional layer? 2016.
46. Simonyan, Zisserman. Very deep convolutional networks for large-scale image recognition. ICLR, 2015.
47. Springenberg, Dosovitskiy, Brox, Riedmiller. Striving for simplicity: the all convolutional net. ICLR workshop, 2015.
48. Srivastava, Greff, Schmidhuber. Training very deep networks (highway networks). NeurIPS, 2015.
49. Sutskever, Vinyals, Le. Sequence to sequence learning with neural networks. NeurIPS, 2014.
50. Szegedy et al. Going deeper with convolutions. CVPR, 2015.
51. Szegedy, Vanhoucke, Ioffe, Shlens, Wojna. Rethinking the Inception architecture for computer vision. CVPR, 2016.
52. Tan, Le. EfficientNet: rethinking model scaling for convolutional neural networks. ICML, 2019.
53. van den Oord et al. WaveNet: a generative model for raw audio. 2016.
54. Vaswani et al. Attention is all you need. NeurIPS, 2017.
55. Wu et al. Google's neural machine translation system: bridging the gap between human and machine translation. 2016.
56. Xie, Girshick, Dollár, Tu, He. Aggregated residual transformations for deep neural networks (ResNeXt). CVPR, 2017.
57. Yu, Koltun. Multi-scale context aggregation by dilated convolutions. ICLR, 2016.
58. Zaremba, Sutskever, Vinyals. Recurrent neural network regularization. 2014.
59. Zhang. Making convolutional networks shift-invariant again. ICML, 2019.
60. Zhang, Isola, Efros, Shechtman, Wang. The unreasonable effectiveness of deep features as a perceptual metric (LPIPS). CVPR, 2018.
61. Zhou, Khosla, Lapedriza, Oliva, Torralba. Learning deep features for discriminative localization. CVPR, 2016.
62. Zhu, Hu, Lin, Dai. Deformable ConvNets v2: more deformable, better results. CVPR, 2019.
