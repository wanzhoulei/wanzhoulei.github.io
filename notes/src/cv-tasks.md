---
title: "Core Computer Vision Tasks"
subtitle: "Classification, detection, segmentation and other dense tasks, plus the sampling, interpolation and color conventions that image restoration depends on and a short camera model leading to NeRF, each derived step by step with numbers you can check by hand."
order: 0.64
updated: 2026-11-15
---

Generative-vision interviews still include questions from core computer vision: how a detector turns a feature map into boxes, how mAP is computed, why RoIAlign replaced RoI pooling, how to write bilinear interpolation, why you blur before downsampling. These questions have exact answers that can be checked on a whiteboard, and the same ideas sit inside generative models. Diffusion U-Nets come from segmentation networks. Resizing and color conventions decide how super-resolution training pairs are made and how they are scored. Hungarian matching and set losses from DETR reappear in many query-based models. The pinhole camera is the starting point of NeRF and 3D Gaussian splatting. This page rebuilds each topic from its definitions. Every number in a worked example was checked in Python, and the from-scratch code near the end was run against `torchvision` and PyTorch.

**What is elsewhere.** Convolution layers, receptive fields and classic CNN architectures (VGG, ResNet, MobileNet) are on [CNNs and RNNs](cnns-rnns.html). ViTs and self-supervised features (DINO, CLIP, MAE) are on [vision backbones](vision-backbones.html). Sampling theory and degradation models for super-resolution are treated at greater depth on [restoration and super-resolution](restoration-super-resolution.html). Losses, optimizers and regularization in general are on [deep learning training](deep-learning-training.html), and PSNR, SSIM and FID on [evaluation metrics](evaluation-metrics.html).

**Roadmap.** "Image classification" covers the pipeline, augmentation (with the mixup and CutMix formulas), transfer learning, label noise and top-$k$ accuracy. "Object detection" is the longest section: box targets, IoU, NMS, anchors, the R-CNN family and RoIAlign, YOLO and SSD, focal loss, feature pyramids, anchor-free detectors, DETR, and mAP computed by hand. "Segmentation" covers the three task variants, FCN, U-Net, DeepLab, Mask R-CNN, the Dice and IoU losses with their gradients, mIoU, panoptic quality and SAM. "Other dense prediction tasks" briefly treats pose, monocular depth and optical flow. "Image processing for restoration and generation" keeps only what deep restoration and generation rely on: color spaces, Gaussian blur as a degradation kernel, the Fourier transform and aliasing, and interpolation. A short camera model then leads into NeRF and Gaussian splatting. Then come the from-scratch code, a rapid-fire review and references. Practice questions follow each section; the ones marked with a badge were reported in real interviews.

## Notation

A few letters are used in two places; the table says where.

| Symbol | Meaning |
|---|---|
| $H, W$ | image or feature-map height and width, in pixels or cells |
| $n_{\text{cls}}$ | number of classes |
| $z$, $\hat p = \mathrm{softmax}(z)$ | logits and predicted class probabilities |
| $y$, $\mathbf{y}$ | a class label, and its one-hot (or soft) target vector |
| $(x_1, y_1, x_2, y_2)$ | a box in corner format: top-left and bottom-right corners |
| $(x, y, w, h)$ | a box in center format: center, width, height |
| $(x_a, y_a, w_a, h_a)$ | the reference box (anchor or proposal) a prediction is made relative to |
| $t = (t_x, t_y, t_w, t_h)$ | box-regression targets relative to the reference box |
| $\mathrm{IoU}(A, B)$ | intersection over union of two boxes or masks |
| $s$ | stride of a feature map, or a downsampling factor |
| $p_t$, $\gamma$, $\alpha$ | focal loss: probability given to the true class, focusing exponent, class weight |
| $\lambda$ | mixing weight in mixup and CutMix |
| $p_i$, $g_i$ | segmentation: predicted foreground probability and 0/1 label of pixel $i$ |
| $I(x, y)$; $I_x, I_y, I_t$ | image intensity; its partial derivatives in $x$, $y$ and time |
| $(u, v)$ | optical-flow section: the flow vector. Camera section: pixel coordinates |
| $\mathbf{X} = (X, Y, Z)$ | a 3D point; $Z$ is depth along the optical axis |
| $K$; $f_x, f_y, c_x, c_y$ | camera intrinsic matrix; focal lengths and principal point, in pixels |
| $R$, $\mathbf{t}$ | rotation and translation from world to camera coordinates |
| $k$ | a blur kernel (in detection, also the number of anchors per location) |

## Image classification

### The pipeline

Classification maps an image to one of $n_{\text{cls}}$ labels. A modern pipeline has six parts.

1. **Input.** Decode the image, apply random augmentation during training, resize and crop to a fixed size such as $224 \times 224$, convert to float, and normalize each channel by the dataset mean and standard deviation (for ImageNet, means $(0.485, 0.456, 0.406)$ and standard deviations $(0.229, 0.224, 0.225)$ on a $[0, 1]$ scale). The usual ImageNet evaluation transform resizes the shorter side to 256 and takes the central $224 \times 224$ crop.
2. **Backbone.** A CNN or ViT turns the image into a feature map or a set of tokens (see [CNNs and RNNs](cnns-rnns.html) and [vision backbones](vision-backbones.html)).
3. **Pooling.** Global average pooling turns the final feature map into one vector $h \in \mathbb{R}^D$ ($D = 2048$ for ResNet-50); a ViT uses its class token or the mean of its tokens.
4. **Head.** A linear layer gives logits $z = Wh + b$ and a softmax gives $\hat p$.
5. **Loss.** Cross-entropy $L = -\sum_k \mathbf{y}_k \log \hat p_k$, whose gradient with respect to the logits is $\hat p - \mathbf{y}$. Label smoothing replaces the one-hot target by $(1 - \epsilon)\mathbf{y} + \epsilon/n_{\text{cls}}$ (Inception-v3 used $\epsilon = 0.1$), so the network is never pushed toward infinite logits.
6. **Evaluation.** Top-1 and top-5 accuracy on a held-out set, sometimes averaged over flipped or multiple crops (test-time augmentation).

### Data augmentation

Augmentation applies random transformations that should not change the label. Each one states an invariance you want the network to have, and it multiplies the number of distinct training inputs, which reduces overfitting.

**Random resized crops.** Torchvision's `RandomResizedCrop`, the standard ImageNet recipe, crops a region whose area is a uniform fraction in $[0.08, 1]$ of the image, with aspect ratio log-uniform in $[3/4, 4/3]$, and resizes it to $224 \times 224$. It teaches invariance to position and scale. An 8% crop can miss the object and make that sample's label wrong, so small or object-centric datasets often use a larger minimum area.

**Flips.** A horizontal flip with probability 0.5 is almost always safe for natural images. Vertical flips are not, because gravity gives scenes an orientation. Flips are wrong when left and right carry meaning: text, some digits, and medical images where laterality matters. In detection and pose estimation the labels must be flipped too: box $x$-coordinates become $W - x$, and keypoint names swap (left eye becomes right eye).

**Color jitter.** Random changes to brightness, contrast, saturation and hue, and random grayscale conversion, teach invariance to lighting and camera color response; AlexNet's "PCA lighting" added random multiples of the principal components of the RGB values. Avoid it when color defines the class (ripe versus unripe fruit, some medical stains).

**Mixup** (Zhang et al., 2018) trains on convex combinations of pairs of examples and of their one-hot labels:

$$
\tilde x = \lambda x_i + (1 - \lambda) x_j, \qquad \tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1 - \lambda) \mathbf{y}_j, \qquad \lambda \sim \mathrm{Beta}(\alpha, \alpha).
$$

Because cross-entropy is linear in the target, the loss on the mixed target splits into two ordinary losses:

$$
\begin{aligned}
L(\tilde{\mathbf{y}}, \hat p) &= -\sum_k \big(\lambda \mathbf{y}_{i,k} + (1 - \lambda)\mathbf{y}_{j,k}\big)\log \hat p_k && \text{definition of cross-entropy} \\
&= \lambda\Big(-\sum_k \mathbf{y}_{i,k} \log \hat p_k\Big) + (1 - \lambda)\Big(-\sum_k \mathbf{y}_{j,k} \log \hat p_k\Big) && \text{the sum is linear in the target} \\
&= \lambda\, L(\mathbf{y}_i, \hat p) + (1 - \lambda)\, L(\mathbf{y}_j, \hat p).
\end{aligned}
$$

This is how mixup is implemented: shuffle the batch to get partners $j$, mix the inputs, and mix two cross-entropies with the same $\lambda$. Example: with $\lambda = 0.7$, a cat image mixed with a dog image has target $(0.7, 0.3)$ on (cat, dog). If the network predicts $(0.6, 0.4)$, the loss is $-(0.7 \log 0.6 + 0.3 \log 0.4) = 0.632$. With small $\alpha$ the Beta distribution puts most mass near 0 and 1, so most mixed images are close to one of the two originals; the paper found $\alpha \in [0.1, 0.4]$ helped on ImageNet and that large $\alpha$ caused underfitting. Mixup asks the network to behave linearly between training examples, which smooths the decision boundary, makes it harder to memorize individual labels, and tends to give less overconfident, better calibrated probabilities (Thulasidasan et al., 2019).

**CutMix** (Yun et al., 2019) pastes a rectangle from image $B$ into image $A$ instead of blending whole images. With a binary mask $M$ that is 0 inside the rectangle and 1 elsewhere,

$$
\tilde x = M \odot x_A + (\mathbf{1} - M) \odot x_B, \qquad \tilde{\mathbf{y}} = \lambda\, \mathbf{y}_A + (1 - \lambda)\, \mathbf{y}_B .
$$

The paper draws $\lambda \sim \mathrm{Beta}(1, 1)$, which is uniform on $[0, 1]$. It picks the rectangle center $(r_x, r_y)$ uniformly in the image and sets its size to $r_w = W\sqrt{1 - \lambda}$, $r_h = H\sqrt{1 - \lambda}$, so the pasted area is $r_w r_h = (1 - \lambda)WH$: a fraction $1 - \lambda$ of the image comes from $B$, matching the label weight. When the rectangle is clipped at the image border, $\lambda$ is recomputed from the actual pasted area. Example on a $224 \times 224$ image with $\lambda = 0.75$: the patch is $112 \times 112$. Centered at $(200, 100)$ it would span $x \in [144, 256]$, which clips to $[144, 224]$, so the pasted area is $80 \times 112 = 8960$ pixels, a fraction $0.179$ of the image, and the label weight becomes $\lambda = 0.821$. Compared with mixup, every pixel stays a real pixel, so local statistics look natural; compared with Cutout (DeVries and Taylor, 2017), which fills the rectangle with zeros, no pixels are wasted. Both force the network to use more of the object than its single most discriminative part.

**Policies.** RandAugment (Cubuk et al., 2020) applies a chosen number of random operations at a chosen strength, replacing AutoAugment's expensive search. ViTs trained on ImageNet-1k need strong augmentation because their built-in priors are weaker than a CNN's ([vision backbones](vision-backbones.html)).

::: pitfall
Geometric augmentations must be applied identically to the labels: boxes, keypoints and masks move with the image. Masks and other label maps must be resized with nearest-neighbor interpolation, because bilinear interpolation invents in-between label values that do not exist. And validation data get only the deterministic resize and crop, never random augmentation.
:::

### Transfer learning and fine-tuning

Transfer learning starts from a backbone pretrained on a large dataset (ImageNet, or web-scale image–text pairs) and adapts it to a new task with fewer labels. Early layers learn generic features such as edges, colors and textures, and later layers become more specific to the pretraining task (Yosinski et al., 2014), so how much of the network to retrain depends on how much data you have and how different your domain is.

**Linear probe.** Freeze the backbone, compute features $h$ once, and train only a linear classifier on them. For ResNet-50 and 10 classes that is $2048 \times 10 + 10 = 20{,}490$ parameters, against about 25.6 million in the backbone. The problem is convex (multinomial logistic regression), features can be cached so each epoch is cheap, and there is little to overfit. A linear probe is also the standard way to measure how good a representation is, which is why self-supervised papers report it.

**Full fine-tuning.** Initialize from the pretrained weights and update everything, typically with a learning rate several times smaller than for training from scratch, a short warmup, and fewer epochs. It usually gives the best in-distribution accuracy when you have enough labels or the domain differs from pretraining (medical scans, satellite images).

**In between.** Freeze the early stages and fine-tune the later ones, use layer-wise learning rates that grow toward the head, or train a few added parameters (adapters, LoRA; see [LLMs](llms.html)).

**LP-FT.** Fine-tuning with a randomly initialized head sends large, essentially random gradients into the pretrained features during the first steps, which can damage them. Kumar et al. (2022) found that, on their benchmarks, full fine-tuning gave on average 2% higher accuracy in distribution but 7% lower accuracy out of distribution than linear probing, and that training a linear probe first and then fine-tuning everything (LP-FT) was 1% better in distribution and 10% better out of distribution than full fine-tuning.

| | Small target dataset | Large target dataset |
|---|---|---|
| Similar to pretraining data | linear probe, or fine-tune only the top | full fine-tuning |
| Different from pretraining data | fine-tune upper layers with strong regularization; LP-FT | full fine-tuning (pretrained start still usually helps) |

Practical details: match the pretraining preprocessing (normalization constants, resolution); with small batches keep BatchNorm statistics frozen; interpolate a ViT's position embeddings when changing resolution. Kornblith et al. (2019) found that ImageNet accuracy predicts transfer accuracy well under fine-tuning, but that regularizers such as label smoothing and dropout, which raise ImageNet accuracy, make the frozen features transfer worse.

### Label noise

Real labels are wrong more often than people assume. Northcutt et al. (2021) estimated that at least 6% of the ImageNet validation labels are wrong, and an average of at least 3.3% across ten common test sets. Sources include annotator mistakes, genuinely ambiguous images, images with several objects but one label, and labels scraped from web text.

**Noise models.** *Symmetric* noise replaces the true label, with probability $\eta$, by one of the other $n_{\text{cls}} - 1$ classes chosen uniformly. *Class-conditional* noise has a transition matrix $T_{ij} = P(\tilde y = j \mid y = i)$, which captures systematic confusions such as husky versus wolf. *Instance-dependent* noise depends on the image itself, for example blurry images are mislabeled more often; it is the most realistic and the hardest.

**Why mild symmetric noise does not change the best classifier.** The noisy label distribution at an input $x$ is $\tilde p(j \mid x) = \sum_i p(i \mid x)\, T_{ij}$. With symmetric noise, $T_{jj} = 1 - \eta$ and $T_{ij} = \eta/(n_{\text{cls}} - 1)$ for $i \ne j$, so

$$
\begin{aligned}
\tilde p(j \mid x) &= (1 - \eta)\, p(j \mid x) + \frac{\eta}{n_{\text{cls}} - 1}\big(1 - p(j \mid x)\big) && \text{the other classes have total probability } 1 - p(j \mid x) \\
&= \frac{\eta}{n_{\text{cls}} - 1} + \Big(1 - \frac{\eta\, n_{\text{cls}}}{n_{\text{cls}} - 1}\Big)\, p(j \mid x) && \text{collect the terms in } p(j \mid x).
\end{aligned}
$$

This is an increasing function of $p(j \mid x)$ whenever $\eta < (n_{\text{cls}} - 1)/n_{\text{cls}}$, so the class with the highest clean probability still has the highest noisy probability. With infinite data, minimizing cross-entropy on noisy labels therefore still gives the clean Bayes classifier. Example: with 10 classes and $\eta = 0.4$, a clean label of class 3 becomes "class 3 with probability 0.6, each other class with probability 0.044". Systematic noise is different: if cats are labeled "dog" with probability above 0.5, the noisy argmax flips.

**Why it hurts anyway.** Training data are finite and networks are large enough to memorize them: Zhang et al. (2017) showed that standard networks fit ImageNet-sized datasets with completely random labels. Networks tend to fit the clean, consistent patterns first and memorize the wrong labels later, so with noisy labels validation accuracy often rises, peaks and then falls.

**What to do.**

1. *Measure.* Relabel a random sample to estimate the noise rate and the common confusions.
2. *Clean.* Flag examples whose out-of-sample (cross-validated) probability for their given label is low, the idea behind confident learning (Northcutt et al., 2021), and relabel or drop them.
3. *Robust losses.* The gradient of cross-entropy is $\nabla_\theta(-\log p_y) = -\frac{1}{p_y}\nabla_\theta p_y$, so an implausible example (small $p_y$, as a mislabeled one often is) gets a large weight $1/p_y$. The mean absolute error $1 - p_y$ has gradient $-\nabla_\theta p_y$ and weights all examples equally: more robust, but slow to train. Generalized cross-entropy $L_q = (1 - p_y^q)/q$ (Zhang and Sabuncu, 2018) moves between the two: cross-entropy as $q \to 0$, MAE at $q = 1$.
4. *Sample selection.* Co-teaching (Han et al., 2018) trains two networks, each updating on the small-loss examples picked by the other, since small-loss examples are more likely clean.
5. *Regularize, stop early, correct.* Label smoothing, mixup and early stopping on a clean validation set reduce memorization. With an estimated transition matrix $T$, train so that $T^\top \hat p$ fits the noisy labels (forward correction, Patrini et al., 2017).

### Top-1 and top-5 accuracy

Top-$k$ accuracy counts a prediction as correct if the true label is among the $k$ classes with the highest scores:

$$
\mathrm{acc}@k = \frac{1}{N}\sum_{n=1}^{N} \mathbf{1}\big[\mathrm{rank}_n(y_n) \le k\big],
$$

where $\mathrm{rank}_n(y_n)$ is the position of the true class when image $n$'s scores are sorted in decreasing order. Example: four images whose true class is ranked 1st, 3rd, 1st and 7th give top-1 accuracy $2/4 = 50\%$ and top-5 accuracy $3/4 = 75\%$. Top-5 was the official ImageNet challenge (ILSVRC) metric because many of its 1000 classes are fine-grained (over a hundred dog breeds) and many images contain several objects but carry one label. AlexNet won ILSVRC-2012 with a top-5 test error of 15.3%, against 26.2% for the second entry. Top-1 is the usual headline today; on imbalanced data also report per-class (balanced) accuracy, since overall accuracy can hide failing rare classes.

::: qa
**Why does mixup help, and how do you implement its loss?**

Mixup trains on $\tilde x = \lambda x_i + (1-\lambda)x_j$ with the labels mixed the same way, $\lambda \sim \mathrm{Beta}(\alpha,\alpha)$. Cross-entropy is linear in the target, so the loss equals $\lambda L(\mathbf{y}_i, \hat p) + (1-\lambda)L(\mathbf{y}_j, \hat p)$: shuffle the batch to get partners, mix the inputs, and mix two cross-entropies. It asks the network to behave linearly between examples, which smooths decision boundaries, makes memorizing individual labels harder and tends to improve calibration. Small $\alpha$ keeps samples near real images; large $\alpha$ underfits.
:::

::: qa
**You have a pretrained backbone and 2,000 labeled images from a new domain. Linear probe or full fine-tuning?**

Start with a linear probe: cheap, convex, hard to overfit, and a baseline in minutes; if the domain is close to pretraining it may be near the best you can do. If the domain is far (medical, satellite) or the probe plateaus, fine-tune with a small learning rate, preferably LP-FT (probe first, then unfreeze everything), so a random head does not distort the features early on. Use strong augmentation, frozen BatchNorm statistics with small batches, and early stopping, and keep the probe as the baseline: Kumar et al. (2022) found full fine-tuning can lose out-of-distribution accuracy relative to probing.
:::

::: qa
**You suspect 20% of your training labels are wrong. What do you do?**

Measure first: relabel a random sample to get the noise rate and the common confusions. Clean: flag examples whose cross-validated probability for their own label is low, and relabel or drop them. Train robustly: a loss that does not weight implausible examples by $1/p_y$ (generalized cross-entropy), small-loss selection such as co-teaching, label smoothing or mixup, and early stopping on a clean validation set, because networks fit clean patterns before memorizing wrong labels. Keep the test set clean or you cannot tell whether anything worked. Uniform noise this mild leaves the best classifier unchanged in the infinite-data limit, so the damage comes from memorization; systematic confusions can change the decision itself.
:::

## Object detection

A detector outputs a variable-size set of (box, class, score) triples: the number of objects is unknown, and each one must be both recognized and localized. The designs below differ in where candidate boxes come from and how duplicates are avoided. Two-stage detectors propose regions and then classify them; one-stage detectors predict densely at every feature-map location and remove duplicates with NMS; DETR predicts a fixed-size set trained to contain no duplicates.

### Boxes and the regression targets

A box is stored either by its corners $(x_1, y_1, x_2, y_2)$ or by its center and size $(x, y, w, h)$, with $x = (x_1 + x_2)/2$ and $w = x_2 - x_1$. Detectors rarely regress the raw coordinates. Each output is attached to a reference box $(x_a, y_a, w_a, h_a)$, an anchor or a proposal, and the network predicts a correction to it.

We want targets that (1) do not change if the image and both boxes are translated, (2) do not change if the image is rescaled, and (3) are unconstrained real numbers of order one, so a linear output layer can produce them. Derive them one coordinate at a time.

- *Center.* The offset $x - x_a$ is unchanged by translation. Rescaling the image by $c$ multiplies it by $c$, and also multiplies $w_a$ by $c$, so dividing by the anchor width removes the scale: $t_x = (x - x_a)/w_a$, and likewise $t_y = (y - y_a)/h_a$.
- *Size.* The ratio $w/w_a$ is unchanged by both translation and rescaling, but it is positive and asymmetric: doubling gives 2 and halving gives 0.5. Its logarithm is unconstrained and symmetric ($\log 2 = -\log 0.5$), and an error in log space is a relative error, so a 10% mistake costs the same for small and large objects: $t_w = \log(w/w_a)$, $t_h = \log(h/h_a)$.

$$
t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}, \quad t_w = \log\frac{w}{w_a}, \quad t_h = \log\frac{h}{h_a};
\qquad
x = x_a + t_x w_a, \quad w = w_a\, e^{t_w}.
$$

The right-hand pair decodes a prediction back into a box. Example: anchor $(100, 100, 64, 64)$ and ground truth $(110, 96, 96, 48)$ give $t = (10/64, -4/64, \log 1.5, \log 0.75) = (0.156, -0.0625, 0.405, -0.288)$, and decoding returns exactly $(110, 96, 96, 48)$. Implementations rescale the targets to a spread near one (Detectron's second stage multiplies them by $(10, 10, 5, 5)$) and clip $t_w, t_h$ before exponentiating (torchvision at $\log(1000/16) \approx 4.14$) so early predictions cannot overflow.

The loss on $t$ in Fast and Faster R-CNN is the smooth L1 (Huber-like) loss, applied to each coordinate:

$$
\mathrm{smooth}_{L_1}(r) = \begin{cases} 0.5\, r^2 & \text{if } |r| < 1, \\ |r| - 0.5 & \text{otherwise.} \end{cases}
$$

Near zero it is quadratic, so the gradient shrinks smoothly to zero; far away it is linear, so a badly wrong box (often a mislabeled or ambiguous one) cannot produce a huge gradient, unlike L2. Later detectors often add or substitute losses based on IoU itself (GIoU, DIoU, CIoU), which measure overlap directly and are scale-invariant.

### Intersection over union

For two boxes $A$ and $B$,

$$
\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.
$$

The intersection of two axis-aligned boxes is itself a box: its top-left corner is the elementwise maximum of the two top-left corners, and its bottom-right corner the elementwise minimum of the two bottom-right corners. If the boxes do not overlap, that "box" has negative width or height, so clip each side length at zero.

**Worked example.** $A = (1, 1, 5, 5)$ and $B = (3, 2, 7, 6)$ both have area 16. The intersection runs from $(\max(1,3), \max(1,2)) = (3, 2)$ to $(\min(5,7), \min(5,6)) = (5, 5)$, so it is $2 \times 3 = 6$. The union is $16 + 16 - 6 = 26$ and $\mathrm{IoU} = 6/26 = 0.231$.

**Properties.** IoU lies in $[0, 1]$, is symmetric, and does not change if both boxes are rescaled together; $1 - \mathrm{IoU}$ (the Jaccard distance) is a metric. Scale invariance hides one fact: a fixed pixel error hurts small boxes much more. Shifting a $4 \times 4$ box by one pixel gives IoU $12/20 = 0.6$, while shifting a $100 \times 100$ box by one pixel gives $9900/10100 = 0.98$. That is why small objects score poorly under strict IoU thresholds. And IoU is exactly zero for every pair of disjoint boxes, however far apart, so as a loss it gives no gradient until the boxes overlap. **Generalized IoU** (Rezatofighi et al., 2019) fixes this by subtracting the fraction of the smallest enclosing box $C$ not covered by the union: $\mathrm{GIoU} = \mathrm{IoU} - |C \setminus (A \cup B)|/|C|$. In the example, $C = (1, 1, 7, 6)$ has area 30, so $\mathrm{GIoU} = 0.231 - 4/30 = 0.097$.

### Non-maximum suppression

A dense detector fires at many neighboring anchors or locations for the same object, all with high scores and heavily overlapping boxes. Non-maximum suppression (NMS) keeps the best of each cluster.

1. Sort the boxes of one class by score.
2. Take the highest-scoring remaining box and keep it.
3. Remove every remaining box whose IoU with it is above a threshold (often 0.5 for final detections; Faster R-CNN uses 0.7 on its proposals).
4. Repeat from step 2 until no boxes remain.

It runs per class, or across all classes at once after offsetting each box by a large constant times its class index, so boxes of different classes never overlap. A score threshold and a top-$k$ cut before NMS bound the cost.

**Worked example.** Boxes $b_1 = (10, 10, 50, 50)$, $b_2 = (12, 12, 52, 52)$, $b_3 = (60, 60, 100, 100)$, $b_4 = (30, 30, 70, 70)$ with scores 0.9, 0.8, 0.7, 0.6 and threshold 0.5. Keep $b_1$. Its IoU with $b_2$ is $38^2/(3200 - 38^2) = 1444/1756 = 0.82$, so $b_2$ is removed; with $b_3$ it is 0; with $b_4$ it is $400/2800 = 0.14$, so both stay. Keep $b_3$; its IoU with $b_4$ is $100/3100 = 0.03$. Keep $b_4$. Result: $b_1, b_3, b_4$.

**Complexity.** Sorting costs $O(N \log N)$. Keeping $K$ boxes costs $O(NK)$ IoU computations, $O(N^2)$ in the worst case, when little overlaps and almost every box is kept. GPU versions compute the full $N \times N$ IoU matrix in parallel and then do a cheap sequential pass.

**Failure modes.** In crowds, two real objects can overlap above the threshold and one is deleted, losing recall; a higher threshold keeps them but also keeps more duplicates, which become false positives. NMS also keeps the box with the highest classification score, not necessarily the best-localized one.

**Soft-NMS** (Bodla et al., 2017) lowers scores instead of deleting boxes. When box $M$ is kept, every remaining box $b_i$ gets

$$
s_i \leftarrow s_i\,\big(1 - \mathrm{IoU}(M, b_i)\big) \ \ \text{if } \mathrm{IoU}(M, b_i) \ge N_t \quad \text{(linear)}, \qquad s_i \leftarrow s_i\, e^{-\mathrm{IoU}(M, b_i)^2/\sigma} \quad \text{(Gaussian)},
$$

and the loop continues on the rescored set. In the example, $b_2$'s score drops from 0.8 to $0.8 \times 0.18 = 0.14$ (linear) or $0.8\, e^{-0.82^2/0.5} = 0.21$ (Gaussian, $\sigma = 0.5$) instead of to zero. If $b_2$ was really a second, overlapping object, it can still be counted at a lower rank, which raises recall and AP; if it was a duplicate, its low score puts it at the end of the ranking, where it costs little (see the mAP section). Soft-NMS needs no retraining and has the same complexity.

### Anchors

Anchors are fixed reference boxes placed at every location of a feature map, several per location with different sizes and aspect ratios. Each output is a classification ("is there an object matching this anchor?") and a small regression $t$ relative to that anchor. This turns detection into many independent, local, well-conditioned problems, and lets one location predict several objects of different shapes.

Faster R-CNN uses $k = 9$ anchors per location: areas $128^2, 256^2, 512^2$ pixels times aspect ratios 1:2, 1:1, 2:1. For area $A$ and aspect ratio $r = h/w$, the anchor has $w = \sqrt{A/r}$ and $h = \sqrt{A r}$, so the $128^2$ anchors are $181 \times 91$, $128 \times 128$ and $91 \times 181$. On a stride-16 feature map of an $800 \times 800$ image there are $50 \times 50 \times 9 = 22{,}500$ anchors; the paper counts about 20,000 for a typical $1000 \times 600$ image.

**Assigning labels to anchors.** Training needs a target for every anchor. Faster R-CNN's region proposal network calls an anchor positive if its IoU with some ground-truth box exceeds 0.7, or if it is the anchor with the highest IoU for some ground-truth box (so every object gets at least one); negative if its IoU is below 0.3 with every ground-truth box; and ignores the rest. Each image contributes a random sample of 256 anchors with up to half positives. RetinaNet uses 0.5 and 0.4 as its two thresholds.

**Costs.** Scales, ratios and IoU thresholds are dataset-dependent hyperparameters (YOLOv2 instead chose its anchor shapes by $k$-means on the training boxes with distance $1 - \mathrm{IoU}$), and almost all anchors are negatives, the imbalance focal loss addresses. Both costs motivated anchor-free detectors.

### Two-stage detectors: the R-CNN family

**R-CNN** (Girshick et al., 2014) warped each of about 2,000 selective-search proposals per image to $227 \times 227$, ran a CNN on each, classified the features with one linear SVM per class and refined the boxes with linear regression. It raised PASCAL VOC 2012 mAP to 53.3%, more than 30% relative to the previous best, but ran the CNN 2,000 times per image and trained its stages separately.

**Fast R-CNN** (Girshick, 2015) runs the CNN once on the whole image and crops each proposal from the shared feature map with **RoI pooling**: the proposal is projected onto the feature map by dividing its coordinates by the stride, its window is divided into a fixed $H \times W$ grid of bins ($7 \times 7$ for VGG16), and each bin is max-pooled. The resulting fixed-size feature goes through fully connected layers into two sibling outputs: a softmax over $n_{\text{cls}} + 1$ classes (including background) and one box regression per class. The multi-task loss is

$$
L = L_{\text{cls}}(\hat p, u) + \lambda\, [u \ge 1]\, \sum_{i \in \{x, y, w, h\}} \mathrm{smooth}_{L_1}\big(t^u_i - t^*_i\big),
$$

where $u$ is the true class, $u = 0$ is background (which has no box, hence the indicator), and $t^u$ is the regression output for class $u$. Everything except the proposals is trained end to end. With VGG16 it trained 9 times and tested 213 times faster than R-CNN, but selective search still took about 2 seconds per image on a CPU.

**Faster R-CNN** (Ren et al., 2015) replaced selective search with a **region proposal network** (RPN) on the shared features: a $3 \times 3$ convolution (512 channels for VGG16) followed by two sibling $1 \times 1$ convolutions outputs, for each of $k$ anchors per location, an objectness score and four box deltas. The decoded proposals are clipped to the image, reduced by NMS at IoU 0.7, and the top 300 or so go to a Fast R-CNN head. The system ran at about 5 frames per second with VGG16, and proposals became nearly free.

**The quantization problem.** RoI pooling rounds twice. At stride 16, a proposal edge at $x_1 = 37$ pixels is at $37/16 = 2.31$ cells, rounded to 2, so the crop starts 5 image pixels too far left; then the $7 \times 7$ bin edges are rounded again, giving bins of unequal size. Each rounding moves a boundary by up to half a cell, 8 pixels at stride 16. Classification hardly cares, because the class does not change when the crop moves a few pixels; pixel-accurate masks and tight boxes at high IoU thresholds do.

**RoIAlign** (He et al., 2017, in Mask R-CNN) removes every rounding. It keeps the proposal's coordinates as real numbers, divides the window into exactly equal bins, computes the feature value at four regularly spaced points in each bin by bilinear interpolation from the four surrounding cells, and takes their max or average. Moving the box by a fraction of a cell now changes the pooled features smoothly instead of not at all or by a jump. The paper reports that RoIAlign improves mask accuracy by 10% to 50% relative, with larger gains under stricter localization metrics.

::: key
**How to say it in an interview.** "R-CNN ran a CNN on each of about 2,000 warped proposals. Fast R-CNN ran the CNN once and pooled each proposal from the shared feature map with RoI pooling, training classification and box regression jointly. Faster R-CNN generated the proposals with a small network on the same features, using anchors. RoI pooling rounds the proposal and its bins to the feature grid, which misaligns features by up to half a cell; RoIAlign samples with bilinear interpolation at exact positions, which matters most for masks."
:::

### One-stage detectors: YOLO and SSD

One-stage detectors skip proposals: one network pass predicts classes and boxes densely, and NMS cleans up. They were historically faster and less accurate than two-stage detectors; focal loss and feature pyramids closed most of the accuracy gap.

**YOLO** (Redmon et al., 2016) resizes the image to $448 \times 448$ and divides it into an $S \times S$ grid with $S = 7$. Each cell predicts $B = 2$ boxes, each as $(x, y, w, h, \text{confidence})$, plus one set of $C = 20$ class probabilities shared by the cell, so the output is a $7 \times 7 \times 30$ tensor, 1,470 numbers. The box center $(x, y)$ is relative to the cell, and $w, h$ are relative to the image. The confidence target is $\Pr(\text{object}) \times \mathrm{IoU}(\text{pred}, \text{truth})$. The cell containing an object's center is responsible for it, and of that cell's two boxes, the one with higher IoU with the truth is trained on it. The loss is a sum of squared errors over coordinates, confidences and class probabilities, with $\lambda_{\text{coord}} = 5$ to up-weight localization and $\lambda_{\text{noobj}} = 0.5$ to down-weight the many cells without objects. It regresses $\sqrt w$ and $\sqrt h$ instead of $w$ and $h$ so that a fixed pixel error counts more for small boxes: a 5-pixel error changes $\sqrt w$ by $\sqrt{15} - \sqrt{10} = 0.71$ for a 10-pixel box but only by $\sqrt{105} - \sqrt{100} = 0.25$ for a 100-pixel box. At test time the class-specific score is $\Pr(\text{class} \mid \text{object}) \times \text{confidence}$, followed by NMS. YOLO ran at 45 frames per second (a smaller "Fast YOLO" at 155) with 63.4% mAP on VOC 2007. Its limits come from the coarse grid: at most 49 objects, one class per cell, and trouble with small objects in groups.

Later versions changed most of this. YOLOv2 added batch normalization, anchors chosen by $k$-means, and a center kept inside its cell, $b_x = \sigma(t_x) + c_x$ and $b_w = p_w e^{t_w}$, with $(c_x, c_y)$ the cell's corner and $p_w$ the anchor width. YOLOv3 predicted at three scales. YOLOX and YOLOv8 are anchor-free with separate classification and regression branches and IoU-based box losses, and YOLOv10 is trained with one-to-one assignment so it can drop NMS.

**SSD** (Liu et al., 2016) predicts from six feature maps of decreasing resolution, $38^2, 19^2, 10^2, 5^2, 3^2$ and $1^2$ cells for a $300 \times 300$ input. Each cell has 4 or 6 default boxes (SSD's name for anchors) with aspect ratios from $\{1, 2, 3, 1/2, 1/3\}$ and scales growing linearly from 0.2 to 0.9 of the image side across the maps, $38^2 \cdot 4 + 19^2 \cdot 6 + 10^2 \cdot 6 + 5^2 \cdot 6 + 3^2 \cdot 4 + 1 \cdot 4 = 8{,}732$ boxes in all, and a $3 \times 3$ convolution predicts class scores and four offsets for each. Each ground-truth box is matched to its best default box and to every default box with IoU above 0.5, and only the hardest negatives are kept, at most three per positive (hard negative mining). SSD300 reached 74.3% mAP on VOC 2007 at 59 frames per second. It was weak on small objects because its high-resolution map comes from an early layer with little semantic information, which is what feature pyramids fix.

### Focal loss and RetinaNet

A one-stage detector scores about 100,000 anchors per image, of which perhaps a hundred are positive. Most negatives are easy (sky, road) and the network quickly gives them probability near 1 of being background. Each then has a tiny loss, but there are so many that they dominate the total. Two-stage detectors avoid the problem: the RPN discards most easy background, and the second stage samples a fixed ratio of foreground to background.

**Numeric example.** Take 100,000 negatives each classified with $p_t = 0.99$ and 100 positives each with $p_t = 0.5$. Under cross-entropy $-\log p_t$, the negatives contribute $100{,}000 \times 0.01005 = 1005$ and the positives $100 \times 0.693 = 69.3$. Easy negatives make up 94% of the loss and of the gradient, even though the network already handles them.

**Focal loss** (Lin et al., 2017) multiplies cross-entropy by a factor that vanishes for well-classified examples. With $p$ the predicted probability of the foreground class, define $p_t = p$ if the anchor is a positive and $p_t = 1 - p$ otherwise. Then

$$
\mathrm{FL}(p_t) = -\alpha_t\, (1 - p_t)^\gamma \log p_t ,
$$

where $\alpha_t = \alpha$ for positives and $1 - \alpha$ for negatives. With $\gamma = 0$ it is weighted cross-entropy. With $\gamma = 2$, the reduction factor $(1 - p_t)^{-2}$ is:

| $p_t$ | cross-entropy | focal loss ($\gamma = 2$, no $\alpha$) | reduction |
|---|---|---|---|
| 0.99 | 0.01005 | 0.000001 | $10^4 \times$ |
| 0.9 | 0.105 | 0.00105 | $100 \times$ |
| 0.5 | 0.693 | 0.173 | $4 \times$ |
| 0.1 | 2.303 | 1.865 | $1.2 \times$ |

In the example, the negatives now contribute $100{,}000 \times 10^{-6} = 0.10$ and the positives $100 \times 0.173 = 17.3$: easy negatives fall from 94% of the loss to under 1%, and training focuses on hard examples, which are mostly the positives and the confusing negatives. The best setting in the paper was $\gamma = 2$ with $\alpha = 0.25$. That $\alpha$ gives positives the *smaller* weight, which surprises people; the explanation is that $\gamma$ has already removed most of the negatives' loss, so the remaining balance tips the other way. The loss is summed over all anchors and divided by the number of positive anchors.

**Initialization.** The bias of the final classification layer is set to $b = -\log\big((1 - \pi)/\pi\big) = -4.60$ with $\pi = 0.01$, so every anchor starts with foreground probability 0.01. With a zero bias every anchor would start at $p = 0.5$, and the 100,000 background anchors would each contribute $\log 2$, a huge and unstable initial loss.

**RetinaNet** is a ResNet with a feature pyramid (levels P3 to P7), 9 anchors per location, and two small convolutional heads shared across levels, one for classification (a sigmoid per class and anchor) and one for box regression. With ResNet-101 it reached 39.1 AP on COCO test-dev, ahead of the two-stage detectors it was compared with, while staying one-stage.

::: qa
**Why does focal loss use $(1 - p_t)^\gamma$, and why is the best $\alpha$ for positives below 0.5?**

The factor vanishes as an example becomes well classified, so the huge number of easy background anchors, each with a tiny but non-zero cross-entropy, stops dominating the sum: with $\gamma = 2$ an anchor at $p_t = 0.9$ loses 100 times less than under cross-entropy and one at $p_t = 0.99$ 10,000 times less, while hard examples barely change. Once the easy negatives are suppressed, the positives carry most of the remaining loss, so the class weight that works best gives them less: $\alpha = 0.25$ in the paper. It also needs a prior bias of about $-4.6$ so initial predictions are 0.01 foreground, not 0.5.
:::

### Feature pyramid networks

Objects in one image can differ in size by a factor of 50 or more. Image pyramids multiply the cost, the last feature map (stride 32) loses small objects, and the early high-resolution maps SSD used have weak features.

A **feature pyramid network** (FPN, Lin et al., 2017) adds a top-down path with lateral connections. Let $C_2, \dots, C_5$ be the backbone's last feature maps at strides 4, 8, 16 and 32. Then

$$
P_5 = \mathrm{conv}_{1\times1}(C_5), \qquad P_\ell = \mathrm{conv}_{1\times1}(C_\ell) + \mathrm{Upsample}_{2\times}(P_{\ell+1}) \quad (\ell = 4, 3, 2),
$$

with nearest-neighbor upsampling, followed by a $3 \times 3$ convolution on each merged map to reduce the aliasing of the upsampling, all with 256 channels. The $1 \times 1$ lateral convolution brings each backbone map to 256 channels so the addition is defined. The top-down path carries semantic information from the deep layers down to the high-resolution ones, and the lateral path carries precise locations up. Every level ends up with strong features at its own resolution, and one shared detection head can run on all levels.

In an FPN version of Faster R-CNN, the RPN uses one anchor size per level ($32^2$ on $P_2$ up to $512^2$ on an extra level $P_6$), each with three aspect ratios. Each proposal of width $w$ and height $h$ is pooled from level

$$
k = \Big\lfloor k_0 + \log_2\big(\sqrt{wh}/224\big) \Big\rfloor, \qquad k_0 = 4,
$$

clamped to the available levels. The idea: a $224 \times 224$ proposal (ImageNet's training size) goes to $P_4$, the level a single-scale detector would use, and each halving of the size moves one level finer. Examples: a $112 \times 112$ proposal goes to $P_3$, a $448 \times 448$ one to $P_5$, and a $50 \times 50$ one gives $\lfloor 1.84 \rfloor = 1$, clamped to $P_2$.

### Anchor-free detectors

Anchor-free detectors drop the reference boxes. Each feature-map location predicts a box directly, either by regressing distances to the box sides or by detecting the object's center as a keypoint.

**FCOS** (Tian et al., 2019) treats every feature-map location $(x, y)$ (mapped back to image coordinates) that falls inside a ground-truth box $(x_1, y_1, x_2, y_2)$ as a positive for that box, and regresses the four distances from the location to the box's sides:

$$
l^* = x - x_1, \quad t^* = y - y_1, \quad r^* = x_2 - x, \quad b^* = y_2 - y.
$$

The network predicts them through an exponential so they stay positive. Each pyramid level handles one range of sizes: on $P_3$ through $P_7$, a location is a positive only if $\max(l^*, t^*, r^*, b^*)$ falls in $[0, 64]$, $[64, 128]$, $[128, 256]$, $[256, 512]$ or $[512, \infty)$ respectively. If a location is still inside two boxes, it takes the one with the smaller area. Locations far from an object's center produce poor boxes, so FCOS adds a **center-ness** branch trained to predict

$$
\text{centerness} = \sqrt{\frac{\min(l^*, r^*)}{\max(l^*, r^*)} \cdot \frac{\min(t^*, b^*)}{\max(t^*, b^*)}},
$$

which is 1 at the box center and falls toward 0 at the edges. Example: $(l^*, r^*, t^*, b^*) = (10, 30, 20, 20)$ gives $\sqrt{(10/30)(20/20)} = 0.58$. At test time the class score is multiplied by the predicted center-ness, so off-center boxes rank lower and are removed by NMS. FCOS still uses NMS.

**CenterNet** ("Objects as Points", Zhou et al., 2019) predicts one heatmap per class at stride 4 whose peaks are object centers, trained against Gaussian bumps at the true centers with a focal-loss variant, and regresses box size and a sub-pixel offset at each center. Detections are locations at least as large as their 8 neighbors, found with one $3 \times 3$ max-pooling, and the top 100 are kept, with no IoU-based NMS. CornerNet (Law and Deng, 2018) earlier detected boxes as pairs of corner keypoints. The heatmap machinery is the same as for human pose (below).

::: pitfall
"Anchor-free" does not mean "NMS-free". FCOS and most YOLO versions without anchors still produce several boxes per object and need NMS. Dropping NMS requires one-to-one assignment during training: each object is matched to exactly one prediction, as in DETR, CenterNet's peak extraction, or YOLOv10's one-to-one head.
:::

### DETR: detection as set prediction

DETR (Carion et al., 2020) removes anchors, proposals and NMS. A CNN backbone produces a feature map, which is flattened into tokens with positional encodings and processed by a transformer encoder. A transformer decoder takes $N = 100$ learned **object queries**, which attend to the image tokens and to each other, and a shared feed-forward network turns each query into a class distribution, including a "no object" class $\varnothing$, and a normalized box (center, width, height). The output is always a set of 100 predictions, most of them $\varnothing$.

**Hungarian matching.** To compute a loss, the ground-truth objects are padded with $\varnothing$ to $N$ entries, and the predictions are matched one-to-one to the ground truth by the permutation $\hat\sigma$ with the lowest total cost:

$$
\hat\sigma = \arg\min_{\sigma} \sum_{i=1}^{N} \mathcal{C}\big(y_i, \hat y_{\sigma(i)}\big), \qquad \mathcal{C} = -\hat p_{\sigma(i)}(c_i) + \lambda_{L_1}\lVert b_i - \hat b_{\sigma(i)}\rVert_1 + \lambda_{\text{iou}}\,\mathcal{L}_{\text{GIoU}}(b_i, \hat b_{\sigma(i)})
$$

for real objects (weights $\lambda_{L_1} = 5$, $\lambda_{\text{iou}} = 2$), and zero cost for matching a prediction to $\varnothing$. The Hungarian algorithm solves this assignment problem exactly in $O(N^3)$ time. The loss is then cross-entropy on every prediction against its matched class (with the $\varnothing$ term down-weighted by a factor of 10) plus the box losses on the matched real objects, applied after every decoder layer.

**Why matching rather than greedy assignment.** Take two ground-truth objects and three queries with costs $(-0.8, -0.7)$ for query 1, $(-0.75, 0.2)$ for query 2 and $(0.5, 0.6)$ for query 3. Greedy matching takes the cheapest pair first, query 1 with object 1 ($-0.8$), and then must give object 2 to query 2 ($0.2$): total $-0.6$. The optimal matching gives object 2 to query 1 and object 1 to query 2: total $-0.7 - 0.75 = -1.45$. Query 3 is matched to $\varnothing$.

**Why no NMS.** Each object is matched to exactly one query, so if a second query also predicts that object, the loss trains it toward $\varnothing$. Duplicates are penalized during training instead of removed afterwards, and decoder self-attention lets queries see each other and coordinate which one claims each object. The paper tried adding NMS: it helped after the first decoder layer and slightly lowered AP after the last.

**Slow convergence.** DETR needed 500 training epochs on COCO to reach 42.0 AP, roughly matching a Faster R-CNN baseline of similar size that trains far shorter, and it was weaker on small objects. The usual explanations: attention over all image positions starts almost uniform and takes long to focus; early matching is unstable, so a query's target changes between epochs; and attention cost grows with the square of the number of tokens, which forces a single low-resolution feature map. Deformable DETR (Zhu et al., 2021) attends only to a few learned sampling points around a reference point on multi-scale features and reports better accuracy, especially on small objects, with 10 times fewer epochs. DN-DETR and DINO add noised ground-truth boxes as extra queries to reconstruct, which gives stable targets and speeds training further.

::: {.qa asked="CV interview advice thread (Reddit 2025), not a report"}
**How does DETR detect objects without anchors or non-maximum suppression?**

It treats detection as set prediction. A CNN backbone and a transformer encoder produce image tokens, and a decoder turns $N = 100$ learned object queries into $N$ predictions of a class (possibly "no object") and a box. Training finds the minimum-cost one-to-one matching between predictions and the ground truth padded with "no object", using the Hungarian algorithm with a cost combining class probability, L1 box distance and GIoU, then applies cross-entropy to all predictions and box losses to the matched ones, after every decoder layer. A second query predicting an already matched object is trained toward "no object", and self-attention among queries lets them coordinate, so training removes duplicates and NMS is unnecessary; learned queries replace anchors. Weaknesses: slow convergence (500 epochs for 42.0 AP on COCO) and weak small objects, from near-uniform initial attention, unstable early matching and one low-resolution feature map; Deformable DETR and query denoising (DN-DETR, DINO) fix most of this.
:::

::: {.qa asked="Kuaishou multimodal round 2 (Zhihu snippet)"}
**How would you combine diffusion with DETR-style detection?**

Make the box coordinates the variable that is diffused. In training, add Gaussian noise to the ground-truth boxes (padded with random boxes to a fixed count) and train a DETR-like decoder, conditioned on image features, to recover the clean boxes and their classes; at inference, start from random boxes and refine them over a few steps. This is DiffusionDet (Chen et al., 2023); the number of boxes and of refinement steps can change at test time without retraining. The related non-generative idea is DN-DETR's query denoising, which adds noised ground-truth boxes as extra queries only to stabilize training. Compare at equal inference compute, since each step is another decoder pass.
:::

### Mean average precision, step by step

Mean average precision (mAP) is the standard detection metric. It rewards finding every object (recall), not producing false alarms (precision), and ranking confident correct detections above wrong ones. It is computed per class and then averaged.

1. **Collect and sort.** For one class, gather every detection from every image in the test set and sort them by score, highest first.
2. **Match.** Go down the list. A detection is a true positive (TP) if its IoU with a ground-truth box of that class in the same image is at least the threshold and that ground-truth box has not already been matched; it takes the unmatched ground-truth box with the highest IoU. Otherwise it is a false positive (FP). A second detection of an already matched object is an FP. Ground truth marked "difficult" (VOC) or "crowd" (COCO) is ignored: matching it is neither rewarded nor penalized.
3. **Precision and recall at each rank.** After the top $k$ detections, $\text{precision}_k = \mathrm{TP}_k / k$ and $\text{recall}_k = \mathrm{TP}_k / N_{\text{gt}}$, where $N_{\text{gt}}$ counts all ground-truth boxes of the class.
4. **Interpolate.** Replace precision at recall $r$ by the best precision at any recall at least $r$: $p_{\text{interp}}(r) = \max_{r' \ge r} p(r')$. This makes the curve non-increasing and reflects a sensible user, who would never pick a threshold that gives lower recall and lower precision than another.
5. **Integrate.** PASCAL VOC 2007 averaged $p_{\text{interp}}$ at the 11 recalls $0, 0.1, \dots, 1$. VOC from 2010 on uses the exact area under the interpolated curve. COCO averages it at the 101 recalls $0, 0.01, \dots, 1$. Recall levels the detector never reaches count as precision 0.
6. **Average.** The mean over classes is mAP. COCO additionally averages over the ten IoU thresholds $0.50, 0.55, \dots, 0.95$ and calls the result AP (often written AP@[.5:.95]); AP$_{50}$ and AP$_{75}$ use a single threshold. COCO also reports AP$_S$, AP$_M$ and AP$_L$ for objects with area below $32^2$, between $32^2$ and $96^2$, and above $96^2$ pixels, and keeps at most 100 detections per image.

**Worked example.** One class, two images, four ground-truth objects: $g_1, g_2$ in image 1 and $g_3, g_4$ in image 2. Six detections, sorted by score, with their IoU to the best-overlapping ground truth:

| rank | score | image | best overlap | match at IoU 0.5 | TP so far | precision | recall |
|---|---|---|---|---|---|---|---|
| 1 | 0.95 | 1 | $g_1$, 0.82 | TP | 1 | 1.00 | 0.25 |
| 2 | 0.90 | 1 | $g_1$, 0.71 | FP ($g_1$ taken) | 1 | 0.50 | 0.25 |
| 3 | 0.80 | 2 | $g_3$, 0.66 | TP | 2 | 0.67 | 0.50 |
| 4 | 0.70 | 1 | $g_2$, 0.58 | TP | 3 | 0.75 | 0.75 |
| 5 | 0.60 | 2 | $g_4$, 0.30 | FP (IoU too low) | 3 | 0.60 | 0.75 |
| 6 | 0.50 | 2 | none | FP | 3 | 0.50 | 0.75 |

The interpolated precision is $\max(1, 0.5, 0.67, 0.75, 0.6, 0.5) = 1$ for recall up to 0.25, and 0.75 for recall in $(0.25, 0.75]$ (the best precision at recall 0.5 or more is the 0.75 at rank 4). Recall above 0.75 is never reached.

- *Area under the curve (VOC 2010+):* $0.25 \times 1 + 0.5 \times 0.75 = 0.625$.
- *11-point (VOC 2007):* recalls 0, 0.1, 0.2 give 1; recalls 0.3 to 0.7 (five points) give 0.75; recalls 0.8 to 1.0 give 0. AP $= (3 + 3.75)/11 = 0.614$.
- *101-point (COCO):* 26 recall levels (0 to 0.25) give 1, 50 levels (0.26 to 0.75) give 0.75, 25 give 0. AP$_{50} = (26 + 37.5)/101 = 0.629$.

Raising the IoU threshold turns poorly localized detections into false positives. At 0.6 and 0.65, rank 4 (IoU 0.58) fails and AP drops to 0.422 (101-point). From 0.7 to 0.8 only rank 1 remains a TP, so AP is 0.257. From 0.85 up nothing matches and AP is 0. The COCO average over the ten thresholds is $0.287$, less than half of AP$_{50}$: COCO AP rewards tight localization, not just finding the object.

Two consequences. Appending detections with scores below all existing ones never lowers AP: the interpolated precision at earlier recalls can only rise (it is a maximum over a larger set), and new true positives add recall. That is why detectors output many low-confidence boxes and why COCO caps them at 100 per image. And AP depends only on the ranking within each class, so scores need not be calibrated, but they must be comparable across images.

::: {.qa asked="Amazon AS intern, Glassdoor 2023"}
**Define IoU and mAP for object detection.**

IoU is the area of the intersection of two boxes divided by the area of their union. For AP, take one class, sort all its detections over the dataset by score, and mark each a true positive if its IoU with a not-yet-matched ground-truth box in the same image reaches the threshold (0.5 for PASCAL VOC), otherwise a false positive, so duplicates are false positives. This gives precision and recall at every rank. Make precision non-increasing by taking, at each recall, the best precision at that recall or higher, and average it over recall (11 points in VOC 2007, the exact area from VOC 2010, 101 points in COCO). mAP averages AP over classes; COCO's AP also averages over IoU thresholds from 0.50 to 0.95 in steps of 0.05, which rewards tight boxes, and reports AP for small, medium and large objects. Instance segmentation uses mask IoU for mask AP, and semantic segmentation uses mIoU.
:::

::: {.qa asked="Qualcomm 2026, Glassdoor"}
**How does YOLO work?**

YOLO is a one-stage detector: one forward pass predicts, at each cell of one or more feature grids, box parameters, an objectness or confidence score and class probabilities, and NMS removes duplicates. In the original, a $448 \times 448$ image maps to a $7 \times 7$ grid; each cell predicts two boxes (center relative to the cell, $\sqrt w$ and $\sqrt h$ relative to the image, and a confidence trained toward $\Pr(\text{object}) \times \mathrm{IoU}$) and 20 class probabilities. The cell containing an object's center is responsible for it, and the loss is weighted sum-squared error. It ran at 45 frames per second with 63.4% mAP on VOC 2007 but struggled with small, crowded objects. YOLOv2 added $k$-means anchors and batch normalization, YOLOv3 multi-scale heads; later versions such as YOLOX and YOLOv8 are anchor-free with FPN- or PAN-style necks, decoupled heads and IoU-based box losses, and YOLOv10 trains one-to-one assignments to drop NMS. It is faster than two-stage detectors such as Faster R-CNN and was historically weaker on small objects.
:::

::: qa
**Walk through Faster R-CNN at inference, from image to final boxes.**

The backbone (often with an FPN) runs once. The RPN predicts objectness and deltas for $k$ anchors per location; the decoded, clipped boxes go through NMS at IoU 0.7 and the top few hundred become proposals. RoIAlign pools a fixed-size feature for each proposal, and the head outputs probabilities over $n_{\text{cls}} + 1$ classes and class-specific deltas that refine each box again. Per class, low scores are dropped, NMS at about IoU 0.5 removes duplicates, and the top 100 detections are returned.
:::

::: qa
**Why does COCO average AP over IoU thresholds from 0.5 to 0.95 instead of using 0.5?**

At IoU 0.5 a loose box still counts, so AP$_{50}$ cannot tell rough from precise localization. Averaging over thresholds up to 0.95 gives credit in proportion to how tight the boxes are: in the worked example AP$_{50}$ is 0.63 but the average is 0.29, because the matches have IoU between 0.58 and 0.82. COCO AP therefore responds to localization improvements, such as RoIAlign or better box losses, that barely move AP$_{50}$.
:::

## Segmentation

### Semantic, instance and panoptic

Segmentation labels pixels. The three variants differ in whether they separate objects of the same class.

| Task | Output | Separates objects of one class? | Main metric |
|---|---|---|---|
| Semantic | one class per pixel | no: two adjacent cars are one "car" region | mIoU |
| Instance | one mask, class and score per object, for countable "thing" classes | yes; masks may overlap; background is unlabeled | mask AP |
| Panoptic | a class and an instance id for every pixel, no overlaps | yes for things; amorphous "stuff" (sky, road, grass) gets one region per class | PQ |

### FCN

The fully convolutional network (FCN, Long et al., 2015) turned an ImageNet classifier into a segmenter. A fully connected layer applied to a $7 \times 7 \times 512$ feature map is the same computation as a $7 \times 7$ convolution with as many output channels, so rewriting the classifier's fully connected layers as convolutions lets it accept any input size and output a coarse map of class scores (stride 32) instead of one vector. That map is upsampled to the input resolution by a transposed convolution initialized to bilinear interpolation, and trained with per-pixel cross-entropy. Upsampling by 32 in one step gives blobby boundaries (FCN-32s), so FCN adds score maps predicted from the earlier pool4 (stride 16) and pool3 (stride 8) features before upsampling: FCN-16s and FCN-8s, each sharper than the last. This combination of coarse, semantic and fine, local predictions is the idea every later segmentation architecture develops.

### U-Net

U-Net (Ronneberger et al., 2015) makes the decoder as large as the encoder. The contracting path repeats two $3 \times 3$ convolutions with ReLU and a $2 \times 2$ max-pooling, doubling the channels from 64 to 1024. The expanding path repeats a $2 \times 2$ up-convolution that halves the channels, concatenation with the encoder features at the same scale, and two $3 \times 3$ convolutions; a $1 \times 1$ convolution gives class scores. The original convolutions are unpadded, so a $572 \times 572$ input yields a $388 \times 388$ output and large images are processed in overlapping tiles. Built for biomedical images with few labels, it relied on elastic-deformation augmentation and a loss weighting that emphasized the thin gaps between touching cells.

The skip connections are the key part. Pooling discards exact positions, and the decoder cannot recover a boundary's position from coarse features alone, so it receives the encoder's high-resolution features directly. U-Net's skips concatenate, ResNet's add. The same design became the standard denoiser in [diffusion models](diffusion-models.html), with attention at low resolutions and the noise level injected into every block.

::: {.qa asked="ByteDance 2020, Baidu 2024 (Nowcoder)"}
**Why does a U-Net downsample and upsample, and what do its skip connections do?**

Downsampling grows the receptive field cheaply, so deep features see context from a large area, and saves computation; upsampling returns to full resolution because the output is a per-pixel map. Pooling discards exact positions, so the skip connections concatenate the encoder's features at each scale into the decoder at the same scale, supplying the detail needed for sharp boundaries and shortening gradient paths. U-Net skips concatenate while ResNet skips add. Diffusion U-Nets reuse the structure: coarse levels handle global layout, fine levels and skips handle detail, attention sits at low resolutions, and the timestep embedding enters every residual block.
:::

### DeepLab and atrous convolution

Pooling by 32 loses detail, but simply removing the strides shrinks every later layer's receptive field and breaks the pretrained weights. **Atrous (dilated) convolution** spaces the kernel taps $r$ pixels apart:

$$
y[i] = \sum_{k} x[i + r\,k]\; w[k].
$$

A $3 \times 3$ kernel with rate $r$ spans $3 + 2(r - 1) = 2r + 1$ pixels per side but still has 9 weights. DeepLab removes the strides of the last one or two backbone stages and dilates every later convolution by the factor the removed strides would have contributed (rate 2 after one removed stride, 4 after two). The network then keeps the pretrained weights and receptive fields while producing features at output stride 16 or 8 instead of 32. The price is computation: going from output stride 16 to 8 makes the last stage process 4 times as many positions.

**Atrous spatial pyramid pooling** (ASPP) runs several atrous convolutions with different rates in parallel and concatenates them, so each position sees context at several scales. DeepLabv3 at output stride 16 uses a $1 \times 1$ convolution, three $3 \times 3$ convolutions at rates 6, 12 and 18 (spans of 13, 25 and 37 cells; 37 cells cover $37 \times 16 = 592$ input pixels), and an image-level branch that global-average-pools the features. That branch is needed because at rates close to the feature-map size most taps fall into padding and the kernel degenerates to its center weight. DeepLabv1 and v2 refined boundaries with a fully connected CRF (v2 introduced ASPP); v3 dropped the CRF; v3+ (Chen et al., 2018) added a light decoder that merges low-level features, and depthwise-separable atrous convolutions.

### Mask R-CNN

Mask R-CNN (He et al., 2017) is Faster R-CNN with a third head. For each proposal, a small FCN on the RoIAligned features predicts one $m \times m$ mask per class ($28 \times 28$ in the FPN version), each with a per-pixel sigmoid. The mask loss is the average binary cross-entropy of the mask for the proposal's ground-truth class only; the other classes' masks receive no gradient. This decouples mask and class prediction: the classification head decides what the object is, and the mask head only has to decide which pixels belong to it, without classes competing per pixel as they would under a softmax. The total loss is $L = L_{\text{cls}} + L_{\text{box}} + L_{\text{mask}}$. At inference, the mask of the predicted class is resized to the box and thresholded at 0.5. RoIAlign matters most here, because a mask that is shifted by a few pixels has a visibly lower IoU. The same design predicts human keypoints by treating each keypoint as a one-hot $m \times m$ mask.

Later universal architectures such as Mask2Former (Cheng et al., 2022) handle semantic, instance and panoptic segmentation with one DETR-style model: each query predicts a class and a mask, trained with Hungarian matching.

### Losses: cross-entropy, Dice and IoU

**Per-pixel cross-entropy** averages $-\log \hat p_i(y_i)$ over pixels. Unlabeled pixels are excluded with an ignore index (255 in many datasets), and rare classes can be up-weighted. Its gradient with respect to each pixel's logits is $\hat p_i - \mathbf{y}_i$, a purely local signal: every pixel pushes on its own, so the class with the most pixels supplies most of the gradient.

**Dice loss.** For a binary mask with predicted probabilities $p_i$ and labels $g_i \in \{0, 1\}$, the soft Dice coefficient and loss are

$$
D = \frac{2\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i} = \frac{2I}{S}, \qquad L_{\text{Dice}} = 1 - D,
$$

where $I = \sum_i p_i g_i$ is the soft intersection and $S = \sum_i p_i + \sum_i g_i$. With hard 0/1 predictions this is the familiar $2|P \cap G|/(|P| + |G|)$. (V-Net, Milletari et al., 2016, used $\sum p_i^2 + \sum g_i^2$ in the denominator; the behavior is similar.) Its gradient follows from the quotient rule, with $\partial I/\partial p_j = g_j$ and $\partial S/\partial p_j = 1$:

$$
\frac{\partial D}{\partial p_j} = \frac{2 g_j S - 2I}{S^2} = \begin{cases} \dfrac{2(S - I)}{S^2} > 0 & \text{if } g_j = 1 \ \ \text{(raise foreground)}, \\[2ex] -\dfrac{2I}{S^2} < 0 & \text{if } g_j = 0 \ \ \text{(lower background)}. \end{cases}
$$

Every foreground pixel gets the same push up and every background pixel the same push down, and both are divided by $S^2$, a global quantity of the size of the object. **Worked example.** An image has 10,000 pixels and a 100-pixel object, and the network outputs $p = 0.1$ everywhere. Then $I = 100 \times 0.1 = 10$, $S = 10{,}000 \times 0.1 + 100 = 1100$, and $D = 0.018$. Each foreground pixel's gradient is $2 \times 1090/1100^2 = 0.0018$ and each background pixel's is $-20/1100^2 = -0.0000165$, 109 times smaller. Summed over pixels, the foreground total is $0.180$ and the background total $-0.164$: the two classes pull with nearly equal total strength although background has 99 times more pixels. Under per-pixel cross-entropy, the summed logit gradients are $100 \times (0.1 - 1) = -90$ for foreground and $9900 \times 0.1 = 990$ for background, so background dominates elevenfold. That is the intuition for Dice: it measures overlap relative to the object's size, so it is insensitive to class imbalance and suits small structures such as lesions.

Its costs: when the object is tiny or absent, $S$ is small and the gradient is large and noisy, so a smoothing constant is added to numerator and denominator; it does not produce calibrated probabilities; and it can be unstable early in training. A common recipe is cross-entropy plus Dice.

**IoU (Jaccard) loss.** The soft Jaccard index is $J = I/(S - I)$, and $L_{\text{IoU}} = 1 - J$. Dice and Jaccard are monotonically related on a single mask: from $J = I/(S - I)$ we get $S = I(1 + J)/J$, and substituting into $D = 2I/S$ gives

$$
D = \frac{2J}{1 + J},
$$

so they rank predictions of one image identically (in the example, $J = 10/1090 = 0.0092$ and $2J/(1+J) = 0.018 = D$), but their averages over a dataset weight errors differently. The Lovász-softmax loss (Berman et al., 2018) is a tighter convex surrogate for the Jaccard loss, designed to optimize mIoU directly.

### Metrics: mIoU and panoptic quality

**mIoU.** Accumulate one confusion matrix $M$ over the whole test set, where $M_{ab}$ counts pixels of true class $a$ predicted as class $b$. For class $c$, $\mathrm{TP}_c = M_{cc}$, $\mathrm{FN}_c$ is the rest of row $c$ and $\mathrm{FP}_c$ the rest of column $c$, so

$$
\mathrm{IoU}_c = \frac{M_{cc}}{\sum_b M_{cb} + \sum_a M_{ac} - M_{cc}}, \qquad \mathrm{mIoU} = \frac{1}{n_{\text{cls}}}\sum_c \mathrm{IoU}_c .
$$

**Worked example.** Three classes with confusion matrix (rows true, columns predicted) $\begin{bmatrix} 50 & 3 & 2 \\ 4 & 20 & 1 \\ 1 & 0 & 9 \end{bmatrix}$. Class 0: $50/(55 + 55 - 50) = 0.833$. Class 1: $20/(25 + 23 - 20) = 0.714$. Class 2: $9/(10 + 12 - 9) = 0.692$. The mIoU is $0.747$. Pixel accuracy is $79/90 = 0.878$, higher because it is dominated by the large class 0. mIoU gives each class equal weight, so small classes matter.

::: pitfall
Averaging per-image IoUs is not the same as the dataset-level mIoU, and it is ill-defined for classes absent from an image (0/0). Standard benchmarks accumulate intersections and unions over the whole dataset first, then divide. When computing over batches, accumulate the confusion matrix, not per-batch IoUs.
:::

**Panoptic quality** (Kirillov et al., 2019) first matches predicted and true segments of the same class, counting a pair as a match when their IoU exceeds 0.5. Because panoptic segments do not overlap, such a match is unique: if two predicted segments $p_1, p_2$ both had IoU above 0.5 with ground truth $g$, then $|p_k \cap g| > 0.5\,|p_k \cup g| \ge 0.5\,|g|$ for each, so $|p_1 \cap g| + |p_2 \cap g| > |g|$, impossible for disjoint $p_1, p_2$. So no matching algorithm is needed. Then, per class,

$$
\mathrm{PQ} = \frac{\sum_{(p, g) \in \mathrm{TP}} \mathrm{IoU}(p, g)}{|\mathrm{TP}| + \tfrac12|\mathrm{FP}| + \tfrac12|\mathrm{FN}|}
= \underbrace{\frac{\sum_{\mathrm{TP}} \mathrm{IoU}}{|\mathrm{TP}|}}_{\text{segmentation quality (SQ)}} \times \underbrace{\frac{|\mathrm{TP}|}{|\mathrm{TP}| + \tfrac12|\mathrm{FP}| + \tfrac12|\mathrm{FN}|}}_{\text{recognition quality (RQ)}},
$$

averaged over classes, things and stuff alike. Example: three true objects and three predicted segments; two matches with IoU 0.9 and 0.6, one unmatched prediction (FP) and one unmatched object (FN). Then $\mathrm{PQ} = 1.5/(2 + 0.5 + 0.5) = 0.5$, with $\mathrm{SQ} = 0.75$ and $\mathrm{RQ} = 2/3$. RQ is an F1 score for detection and SQ is the average mask quality of what was detected.

### Segment Anything

SAM (Kirillov et al., 2023) defines *promptable segmentation*: given an image and a prompt (foreground or background points, a box, or a rough mask), return a valid mask for what the prompt indicates. A large MAE-pretrained ViT encodes the image once; a light prompt encoder and a small two-way transformer decoder then produce masks fast enough for interactive use. A single point is ambiguous (the shirt or the whole person?), so SAM returns three candidate masks with a predicted IoU for each. Its generality comes largely from data: a model-in-the-loop data engine (assisted manual, then semi-automatic, then fully automatic annotation) produced SA-1B, 11 million images with 1.1 billion masks. The masks carry no class labels, so SAM is often prompted with a detector's boxes. SAM 2 (Ravi et al., 2024) extends it to video with a memory of past frames, so a prompt on one frame segments and tracks the object through the clip.

::: {.qa asked="Horizon Robotics, Nowcoder 2022"}
**How would you combine super-resolution with semantic segmentation?**

It works in both directions. Segmentation can guide SR: condition the SR network on semantic probability maps so that generated texture matches each region's class, as SFTGAN does with spatial feature transform layers that scale and shift features according to the segmentation. SR can help segmentation: an SR branch sharing the encoder keeps fine detail in the shared features (dual super-resolution learning), or small, distant regions are super-resolved before segmenting. Test it with mIoU overall and on small objects and thin structures, against a strong segmenter run at higher input resolution with the same compute.
:::

::: qa
**When would you use Dice loss instead of cross-entropy for segmentation, and what goes wrong with it?**

When the foreground is a small fraction of the image, as with medical lesions; usually Dice is added to cross-entropy. Its gradient is normalized by the total predicted plus true foreground, so foreground and background pull with comparable total strength, while cross-entropy's per-pixel sum is dominated by the majority class (11 to 1 in the worked example). Problems: tiny or empty masks make it noisy (add smoothing and decide how empty ground truth is scored), its outputs are not calibrated probabilities, and per-image overlap is not dataset-level mIoU.
:::

## Other dense prediction tasks

### Keypoints and human pose

Pose estimation locates $J$ body keypoints (COCO uses 17). The dominant approach predicts one **heatmap** per keypoint instead of regressing coordinates. The training target for keypoint $j$ at location $\mu_j$ is a Gaussian bump $H_j(\mathbf{q}) = \exp\big(-\lVert \mathbf{q} - \mu_j\rVert^2 / (2\sigma^2)\big)$ on a grid at stride 4 (a $64 \times 48$ map for a $256 \times 192$ person crop), trained with mean squared error. Heatmaps keep the problem fully convolutional (a shifted person gives a shifted heatmap), allow several candidate locations, and train more easily than direct regression of coordinates from a pooled vector. Stacked hourglass networks (Newell et al., 2016) are an early, influential example. The prediction is the heatmap's argmax, but at stride 4 the argmax alone is off by up to 2 pixels in each direction, so implementations shift it a quarter cell toward the higher of its two neighbors, or fit a local quadratic.

*Top-down* methods detect each person and run a single-person network on each crop (SimpleBaselines, Xiao et al., 2018; HRNet, Sun et al., 2019, which keeps a high-resolution branch throughout); they are accurate, but cost grows with the number of people. *Bottom-up* methods find all keypoints at once and group them into people; OpenPose (Cao et al., 2017) predicts part affinity fields, vector fields along each limb that score candidate connections. COCO evaluates keypoints with object keypoint similarity, $\mathrm{OKS} = \sum_j \exp\big(-d_j^2/(2 s^2 k_j^2)\big)\,[v_j > 0] / \sum_j [v_j > 0]$, with distance error $d_j$, object area $s^2$, per-keypoint tolerance $k_j$ and visibility flag $v_j$; OKS plays the role IoU plays for boxes.

### Monocular depth

Predicting depth from one image is ill-posed: a scene and a copy of it twice as large and twice as far away produce the same image, so absolute scale can come only from learned priors about object sizes. Supervised methods train on depth from LiDAR or RGB-D sensors. Eigen et al. (2014) used a scale-invariant loss on log depths: with $d_i = \log \hat z_i - \log z_i$ over $n$ pixels,

$$
L = \frac{1}{n}\sum_i d_i^2 - \frac{\lambda}{n^2}\Big(\sum_i d_i\Big)^2 .
$$

With $\lambda = 1$ this is the variance of $d_i$, which does not change when every prediction is multiplied by the same factor (that adds a constant to every $d_i$): a prediction twice the true depth everywhere has loss 0. They trained with $\lambda = 0.5$. MiDaS (Ranftl et al., 2020) used a loss invariant to both scale and shift of inverse depth, which allowed mixing datasets with incompatible depth scales and gave strong zero-shot *relative* depth; Depth Anything (Yang et al., 2024) scaled this with large amounts of pseudo-labeled unlabeled images. Self-supervised methods learn from stereo pairs (Monodepth, Godard et al., 2017, reconstructs one view from the other through the predicted disparity) or from video with a predicted camera motion. Common metrics are the absolute relative error $\frac{1}{n}\sum |\hat z_i - z_i|/z_i$ and the fraction of pixels with $\max(\hat z_i/z_i, z_i/\hat z_i) < 1.25$.

### Optical flow

Optical flow assigns each pixel a 2D displacement $(u, v)$ between two frames. The classical starting point is **brightness constancy**: a point keeps its intensity as it moves, $I(x + u, y + v, t + 1) = I(x, y, t)$. For small motion, a first-order Taylor expansion of the left side gives $I + I_x u + I_y v + I_t = I$, so

$$
I_x u + I_y v + I_t = 0 .
$$

This is one equation in two unknowns per pixel. Writing it as $\nabla I \cdot (u, v) = -I_t$ shows that only the component of motion along the gradient, $-I_t/\lVert \nabla I\rVert$, is determined; motion along an edge is invisible. This is the **aperture problem**. Example: at a vertical edge with $\nabla I = (2, 0)$ and $I_t = -4$, the equation reads $2u = 4$, so $u = 2$ and $v$ can be anything: a vertical line moving diagonally looks, through a small window, exactly like one moving horizontally.

Two classical fixes add assumptions. **Lucas–Kanade** (1981) assumes the flow is constant in a small window and solves the stacked equations by least squares,

$$
\Big(\sum_{\text{window}} \nabla I\, \nabla I^\top\Big) \begin{bmatrix} u \\ v \end{bmatrix} = -\sum_{\text{window}} \nabla I\; I_t ,
$$

which is well-posed only where the $2 \times 2$ structure tensor on the left has two large eigenvalues, that is, at corners and textured patches. **Horn–Schunck** (1981) instead adds a global smoothness term, minimizing $\int (I_x u + I_y v + I_t)^2 + \alpha^2(\lVert\nabla u\rVert^2 + \lVert\nabla v\rVert^2)$, which fills in flow where the data say nothing. Both need coarse-to-fine image pyramids for motions larger than a pixel or two, because the Taylor expansion fails.

**RAFT** (Teed and Deng, 2020) is the reference learned method. It correlates features of every pixel pair between the two frames at 1/8 resolution into a 4D volume, pools it into a pyramid, and refines a flow field from zero with a recurrent unit that repeatedly looks up correlations around the current estimate and outputs a correction; keeping all pairs avoids losing small, fast objects. Details are on [vision backbones](vision-backbones.html#optical-flow). Flow is scored by end-point error (mean distance between predicted and true vectors) and used for stabilization, frame interpolation, alignment in video super-resolution and filtering static clips from video training data. It measures 2D image motion, not 3D scene motion, and is undefined at occlusions.

::: qa
**Why can't optical flow be computed from a single pixel, and how do Lucas–Kanade and Horn–Schunck get around it?**

Linearized brightness constancy gives one equation, $I_x u + I_y v + I_t = 0$, for two unknowns, so only the flow component along the image gradient is determined (the aperture problem). Lucas–Kanade assumes constant flow in a small window and solves the overdetermined system by least squares, which works where the structure tensor $\sum \nabla I \nabla I^\top$ is well-conditioned (corners, texture) and fails on edges and flat regions. Horn–Schunck adds a global smoothness penalty that fills in uninformative regions but blurs motion boundaries. Both linearize, so large motions need coarse-to-fine pyramids; RAFT replaces linearization with feature correlation and iterative refinement.
:::

## Image processing for restoration and generation

Restoration and generation models are trained and scored on images produced by resizing, blurring and color conversion, and the conventions behind those steps silently change benchmark numbers. This section covers only those pieces; [restoration and super-resolution](restoration-super-resolution.html) builds full degradation models on top of them.

### Images as arrays and color spaces

An image is an $H \times W \times C$ array: rows are $y$, columns are $x$. Files usually hold 8-bit unsigned integers in $[0, 255]$; networks want floats. NumPy and OpenCV use height–width–channel order, PyTorch uses channel–height–width, and OpenCV loads color images in BGR order, not RGB. Arithmetic on `uint8` wraps around silently: $200 + 100$ gives $44$ ($300 \bmod 256$), so convert to float (or a wider integer type) before filtering or blending.

**sRGB and linear light.** Stored sRGB values are gamma-encoded, roughly the 1/2.2 power of linear light intensity, which spends more code values on dark tones where vision is more sensitive. Physically meaningful operations (averaging, blurring, resizing, blending) belong in linear light, although most pipelines skip the conversion.

**YCbCr** separates brightness (luma $Y$) from two color-difference signals. With BT.601 weights,

$$
Y = 0.299R + 0.587G + 0.114B, \qquad C_b = \frac{B - Y}{2(1 - 0.114)}, \qquad C_r = \frac{R - Y}{2(1 - 0.299)},
$$

where the denominators scale each difference into $[-0.5, 0.5]$ (JPEG then adds 128 on an 8-bit scale, so $C_b = 128 + 0.564(B - Y)$). The weights reflect that green contributes most to perceived brightness and blue least. MATLAB's `rgb2ycbcr`, used by most super-resolution benchmark code, uses the studio range instead: $Y = 16 + 65.481R + 128.553G + 24.966B$ for $R, G, B \in [0, 1]$, which maps white to 235 and black to 16.

**Why super-resolution works on luma.** Human vision resolves fine detail much better in brightness than in color, which is why JPEG and video codecs commonly store chroma at half resolution in each direction (4:2:0 subsampling); most edges and texture therefore live in $Y$. Early SR networks such as SRCNN super-resolved only $Y$ and upsampled $C_b, C_r$ bicubically, and SR papers by convention report PSNR and SSIM on $Y$ with a few border pixels cropped. PSNR on $Y$ and on RGB differ for the same images, so state the convention ([evaluation metrics](evaluation-metrics.html)).

Other spaces: HSV and HSL (hue, saturation, value or lightness) for intuitive color edits; CIE XYZ, the device-independent reference; and CIELAB, designed so that Euclidean distance roughly matches perceived color difference.

::: {.qa asked="ByteDance image intern, Nowcoder 2020"}
**How do RAW and RGB images differ, and which colour spaces do you know?**

RAW data are linear sensor readings, 10 to 14 bits deep, with one color per pixel behind a Bayer color filter array, before white balance or gamma. An sRGB image is the output of the camera's image signal processor: black-level subtraction, demosaicing, white balance, a color-correction matrix, denoising and sharpening, tone mapping and gamma, then 8-bit quantization and usually compression. RAW keeps linear noise statistics and more dynamic range, which is why denoising and low-light enhancement work better on it. Color spaces: sRGB (gamma-encoded) and linear RGB; CIE XYZ; YCbCr or YUV, which separate luma from chroma for compression and are why SR papers evaluate on the Y channel; HSV and HSL for hue and saturation edits; and CIELAB, approximately perceptually uniform. Higher bit depth reduces banding in smooth gradients.
:::

### Gaussian blur as a degradation kernel

Super-resolution and deblurring models are usually trained on pairs synthesized from clean images with the degradation model

$$
y = (k * x)\downarrow_s + n ,
$$

blur the high-resolution image $x$ with a kernel $k$, keep every $s$-th pixel, and add noise $n$. The kernel stands for the camera's optical blur and pixel integration, and it doubles as the anti-aliasing filter for the downsampling (next subsection). It is most often a Gaussian, isotropic or anisotropic with a random covariance, as in BSRGAN's randomized degradation pipeline (Zhang et al., 2021). A model trained on one kernel tends to fail on images blurred by another, which is the starting point of blind super-resolution ([restoration and super-resolution](restoration-super-resolution.html)).

**Convolution.** Blurring is a linear shift-invariant filter: every output pixel is the same weighted sum of its neighbors, $(x * k)[i, j] = \sum_{u, v} x[i - u, j - v]\, k[u, v]$. Convolution flips the kernel and correlation does not; deep-learning "convolution" layers compute correlation, which makes no difference for learned kernels or for symmetric ones such as the Gaussian. At the borders, reflection padding avoids the dark frame that zero padding produces in a blurred image.

**The Gaussian kernel** is $G_\sigma(u, v) = \frac{1}{2\pi\sigma^2} e^{-(u^2 + v^2)/(2\sigma^2)}$, truncated at radius $r = \lceil 3\sigma \rceil$ (so $k = 2r + 1$ taps per side) and renormalized to sum to 1. An anisotropic Gaussian replaces $\sigma^2 I$ by a covariance $\Sigma$, usually built from two standard deviations and a rotation angle.

**Separability, derived.** The isotropic Gaussian factors as $G(u, v) = g(u)\, g(v)$, with $g$ the 1D Gaussian, because $e^{-(u^2 + v^2)/2\sigma^2} = e^{-u^2/2\sigma^2} e^{-v^2/2\sigma^2}$. Then

$$
\begin{aligned}
(x * G)[i, j] &= \sum_u \sum_v x[i - u, j - v]\; g(u)\, g(v) && \text{substitute the factorization} \\
&= \sum_u g(u) \underbrace{\Big(\sum_v x[i - u, j - v]\; g(v)\Big)}_{x_1[i - u,\ j]} && g(u) \text{ does not depend on } v \\
&= \sum_u g(u)\, x_1[i - u, j] ,
\end{aligned}
$$

where $x_1$ is the image filtered along each row with the 1D kernel; the outer sum then filters along the columns. So a 2D Gaussian blur is two 1D passes, and the cost per pixel drops from $k^2$ to $2k$ multiply-adds. For $\sigma = 3$, $k = 19$: 361 versus 38, a factor of 9.5, or 2.25 versus 0.24 billion multiply-adds on a $1920 \times 1080$ RGB image. A kernel is separable exactly when its matrix has rank 1, and the SVD gives the two factors. An anisotropic Gaussian is separable along $x$ and $y$ only when its covariance is diagonal: otherwise the exponent contains a cross term proportional to $uv$, which does not split into a function of $u$ times a function of $v$, so a rotated kernel needs a full 2D convolution. In PyTorch a fixed blur is a depthwise `conv2d` (`groups` equal to the number of channels), or two of them with $k \times 1$ and $1 \times k$ kernels.

::: qa
**Why is a Gaussian the usual blur kernel when synthesizing SR training data, and how do you apply it cheaply?**

It is a reasonable model of optical blur and sensor integration, it is a smooth low-pass filter that suppresses aliasing before downsampling (its frequency response is again a Gaussian, with no ripples), and a few parameters (two widths and an angle) span a useful family of isotropic and anisotropic kernels to randomize over. An isotropic or axis-aligned Gaussian is separable, so apply it as two 1D depthwise convolutions, $2k$ instead of $k^2$ multiply-adds per pixel; a rotated anisotropic kernel needs one 2D depthwise convolution. Use reflection padding, and remember that real camera blur also includes motion, defocus and lens aberrations, which is why practical degradation pipelines randomize the kernel and its order relative to noise and compression.
:::

### The Fourier transform, the convolution theorem and aliasing

The 2D discrete Fourier transform writes an $M \times N$ image as a sum of 2D sinusoids:

$$
X[k, l] = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} x[m, n]\; e^{-2\pi i\,(km/M + ln/N)} .
$$

Low frequencies describe smooth variation, high frequencies describe edges, texture and noise. Natural images have most of their energy at low frequencies, with amplitude falling roughly in inverse proportion to frequency.

**Convolution theorem.** For circular convolution $y[n] = \sum_m x[m]\, h[(n - m) \bmod N]$ (1D for brevity),

$$
\begin{aligned}
Y[k] &= \sum_n \sum_m x[m]\, h[(n - m) \bmod N]\, e^{-2\pi i k n/N} && \text{definition of the DFT} \\
&= \sum_m x[m]\, e^{-2\pi i k m/N} \sum_{n'} h[n']\, e^{-2\pi i k n'/N} && n' = (n - m) \bmod N, \text{ and } e^{-2\pi i k n/N} \text{ has period } N \\
&= X[k]\, H[k].
\end{aligned}
$$

Convolution in space is multiplication in frequency. So a filter is described by its frequency response $H$, and convolution can be computed with FFTs in $O(N \log N)$ instead of $O(Nk)$ for a kernel of length $k$, after zero-padding both signals to length at least $N + k - 1$ so the circular wrap does not corrupt the result; this pays off for large kernels. The Fourier transform of a Gaussian is a Gaussian: a blur of standard deviation $\sigma$ pixels multiplies frequency $f$ (in cycles per pixel) by $e^{-2\pi^2\sigma^2 f^2}$, so a wider blur keeps fewer frequencies.

**Sampling and aliasing.** Sampling with one sample per unit makes frequencies $f$ and $f + 1$ indistinguishable, and also $f$ and $1 - f$ for real signals. Only frequencies below the **Nyquist frequency**, half the sampling rate (0.5 cycles per sample), are represented uniquely; a frequency above it appears as a lower one, which is aliasing. Example: $\cos(2\pi \cdot 0.7 n) = \cos(2\pi n - 2\pi \cdot 0.3 n) = \cos(2\pi \cdot 0.3 n)$ at every integer $n$, so a 0.7 cycles-per-sample pattern is recorded as 0.3. If a band-limited signal is sampled above twice its highest frequency, it can be reconstructed exactly (the sampling theorem).

**Why blur before downsampling.** Keeping every second pixel halves the sampling rate, so the new Nyquist limit is 0.25 cycles per original pixel. A stripe pattern at 0.4 cycles per pixel becomes 0.8 cycles per new pixel after decimation and aliases to $1 - 0.8 = 0.2$: a false, coarser stripe pattern (moiré) that was never in the scene and cannot be separated from real content afterwards. So low-pass filter first. A Gaussian with $\sigma = 1$ pixel keeps 73% of the amplitude at 0.125 cycles per pixel, 29% at the new Nyquist limit 0.25, and only 4% at 0.4, trading a little blur for almost no aliasing. That is why image pyramids blur before each halving, why antialiased resizers widen their kernel when shrinking, and why strided convolutions and pooling in CNNs are not shift-invariant unless a blur precedes the subsampling (Zhang, 2019). [Restoration and super-resolution](restoration-super-resolution.html#sampling-and-aliasing) derives the decimation spectrum in full.

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**What changes when you use interpolation to downsample instead of upsample?**

Upsampling only fills in values between samples; downsampling must also remove frequencies above the new Nyquist limit, or they alias into moiré and jagged edges. Bilinear and bicubic kernels span only 2 or 4 input pixels, a reasonable low-pass filter when upsampling but far too narrow when shrinking by a large factor, when most input pixels are simply skipped. Correct downsampling widens the kernel by the scale factor: antialiased resizing, as in PIL, MATLAB's `imresize` by default, PyTorch's `F.interpolate(..., antialias=True)`, or OpenCV's `INTER_AREA`. This matters for SR because low-resolution training inputs are made by downsampling: a model trained on one resizer's output can fail on images from another resizer or a real camera.
:::

### Interpolation: nearest, bilinear, bicubic

Resizing evaluates a continuous reconstruction of the image at new positions. All three standard methods are separable: interpolate along $x$, then along $y$.

**Coordinate mapping.** First map output index $j$ to a continuous input coordinate. If pixel $j$ covers the interval $[j, j + 1)$, its center is at $j + 0.5$; scaling the centers by $W_{\text{in}}/W_{\text{out}}$ and converting back to index units gives

$$
x_{\text{in}} = (j + 0.5)\,\frac{W_{\text{in}}}{W_{\text{out}}} - 0.5 \quad (\texttt{align\_corners=False}), \qquad x_{\text{in}} = j\,\frac{W_{\text{in}} - 1}{W_{\text{out}} - 1} \quad (\texttt{align\_corners=True}).
$$

The first aligns the images' outer edges (pixel areas), the second the centers of the corner pixels. Library defaults differ, and the resulting sub-pixel shifts lower PSNR in super-resolution comparisons.

**Nearest neighbor** takes the input pixel whose area contains the output pixel's center, $\lfloor (j + 0.5)\, W_{\text{in}}/W_{\text{out}} \rfloor$. Its kernel is a box: blocky output, no new values, which is exactly right for label maps and masks.

**Bilinear** interpolation at a point $(x, y)$ with $x_0 = \lfloor x \rfloor$, $x_1 = x_0 + 1$, $\delta_x = x - x_0$ (likewise for $y$) interpolates linearly along $x$ on the two neighboring rows and then along $y$:

$$
\begin{aligned}
\text{top} &= (1 - \delta_x)\, I[y_0, x_0] + \delta_x\, I[y_0, x_1], \qquad \text{bottom} = (1 - \delta_x)\, I[y_1, x_0] + \delta_x\, I[y_1, x_1], \\
I(x, y) &= (1 - \delta_y)\,\text{top} + \delta_y\,\text{bottom} \\
&= (1 - \delta_x)(1 - \delta_y)\, I_{00} + \delta_x (1 - \delta_y)\, I_{01} + (1 - \delta_x)\delta_y\, I_{10} + \delta_x \delta_y\, I_{11}.
\end{aligned}
$$

Each corner's weight is the area of the rectangle opposite it, and the weights sum to $(1 - \delta_x + \delta_x)(1 - \delta_y + \delta_y) = 1$. **Worked example.** For the $2 \times 2$ image $\begin{bmatrix} 10 & 20 \\ 30 & 40 \end{bmatrix}$ at $(x, y) = (0.25, 0.5)$: $0.375 \cdot 10 + 0.125 \cdot 20 + 0.375 \cdot 30 + 0.125 \cdot 40 = 3.75 + 2.5 + 11.25 + 5 = 22.5$. Upsampling the same image by 2 with `align_corners=False` maps output columns to $x = -0.25, 0.25, 0.75, 1.25$, clamped to $0, 0.25, 0.75, 1$, giving rows $(10, 12.5, 17.5, 20)$, $(15, 17.5, 22.5, 25)$, $(25, 27.5, 32.5, 35)$, $(30, 32.5, 37.5, 40)$. With `align_corners=True` the columns map to $0, 1/3, 2/3, 1$ and the first row is $(10, 13.33, 16.67, 20)$. Both match `F.interpolate`. The bilinear kernel is a triangle of width 2 (frequency response $\mathrm{sinc}^2$), smoother but blurrier than nearest.

**Bicubic** interpolation uses 4 taps per axis (16 in 2D) with Keys' cubic convolution kernel,

$$
w(s) = \begin{cases} (a + 2)|s|^3 - (a + 3)|s|^2 + 1 & |s| \le 1, \\ a|s|^3 - 5a|s|^2 + 8a|s| - 4a & 1 < |s| < 2, \\ 0 & \text{otherwise.} \end{cases}
$$

Halfway between pixels, the four weights are $(-0.0625, 0.5625, 0.5625, -0.0625)$ for $a = -0.5$ (MATLAB) and $(-0.094, 0.594, 0.594, -0.094)$ for $a = -0.75$ (PyTorch and OpenCV). The negative outer lobes sharpen the result, a better approximation of ideal sinc interpolation, but overshoot at edges (slight ringing) and can produce values outside $[0, 255]$, so outputs are clamped.

::: {.qa asked="Megvii PhD SR (2021, 2022), Adobe RI (3 reports)"}
**Implement 2x bilinear interpolation. What are the pitfalls?**

Map output pixel $j$ to $x = (j + 0.5)/2 - 0.5$ for the pixel-center convention (PyTorch's `align_corners=False`) or $x = j(W_{\text{in}} - 1)/(W_{\text{out}} - 1)$ for `align_corners=True`, clamp to the image, take $x_0 = \lfloor x \rfloor$, $x_1 = \min(x_0 + 1, W - 1)$, $\delta_x = x - x_0$, likewise for $y$, and blend the four neighbors with weights $(1-\delta_x)(1-\delta_y)$, $\delta_x(1-\delta_y)$, $(1-\delta_x)\delta_y$ and $\delta_x\delta_y$. Vectorize with index and weight arrays per axis and gather instead of looping; the operation is separable. Pitfalls: the half-pixel convention differs across libraries and settings and shifts the result by sub-pixel amounts that cost PSNR; indices must be clamped at the border; integer images must be converted to float; and plain bilinear aliases when downsampling. The tested implementation is in the code section.
:::

::: {.qa asked="Megvii CV algorithm, Nowcoder 2022"}
**Implement nearest-neighbour upsampling. How does it differ from bilinear?**

For each output pixel, copy the input pixel whose area contains the output pixel's center: index $\lfloor (j + 0.5)\, W_{\text{in}}/W_{\text{out}} \rfloor$, clamped to $W_{\text{in}} - 1$, along each axis, then gather `img[ys[:, None], xs[None, :]]`. This matches PyTorch's `mode="nearest-exact"`; the older `mode="nearest"` uses $\lfloor j\, W_{\text{in}}/W_{\text{out}} \rfloor$, which picks different pixels at non-integer scales. For an integer factor $s$ both equal `img.repeat(s, 0).repeat(s, 1)`. Compared with bilinear: a box kernel instead of a triangle, so blocky output instead of blur; no new values, which makes it correct for masks and label maps; even worse aliasing when downsampling; and a backward pass that sums the gradients of each copied block. Nearest upsampling followed by a convolution avoids the checkerboard artifacts of transposed convolutions.
:::

::: {.qa asked="Ant digital human (Nowcoder roundup 2025)"}
**Write pseudocode for quadrilinear interpolation in PyTorch or CUDA.**

It is linear interpolation along each of four axes. Find the grid cell that contains the point, take the fractional offset $f_a$ along each axis, and sum the cell's $2^4 = 16$ corners, each weighted by the product of four 1D weights:

$$
v(\mathbf{x}) = \sum_{\mathbf{c} \in \{0,1\}^4} \Big(\prod_{a=1}^4 f_a^{c_a}(1 - f_a)^{1 - c_a}\Big)\, G[i_1 + c_1, \dots, i_4 + c_4].
$$

The weights sum to $\prod_a (f_a + 1 - f_a) = 1$. Equivalently, do 15 linear interpolations: 8 along the first axis, then 4, 2 and 1.

```python
import itertools, torch

def multilinear(grid, pts):
    """grid: (C, D1, ..., Dn); pts: (N, n) continuous index coordinates. n = 4 is quadrilinear."""
    C, *D = grid.shape
    hi = torch.tensor(D) - 1
    pts = torch.minimum(pts.clamp(min=0), hi)          # clamp to the grid
    i0 = torch.minimum(pts.floor().long(), hi - 1)     # lower corner, so i0 + 1 is valid
    f = pts - i0                                       # fractional offsets in [0, 1]
    out = 0
    for c in itertools.product((0, 1), repeat=len(D)): # the 2^n cell corners
        c = torch.tensor(c)
        w = torch.where(c.bool(), f, 1 - f).prod(1)    # product of 1D weights, (N,)
        out = out + grid[(slice(None), *(i0 + c).unbind(1))] * w   # gather (C, N)
    return out.T                                       # (N, C)
```

Tested: it reproduces the multilinear function $x_1x_2x_3x_4 + x_1$ exactly and, with three axes, matches `F.grid_sample` (trilinear, `align_corners=True`); autograd gives gradients for both the grid and the coordinates. In CUDA, launch one thread per point, loop over the 16 corners (bit $a$ of the corner index picks $i_a$ or $i_a + 1$), and in the backward pass scatter $w\,\partial L/\partial v$ into the corners with `atomicAdd`, since many points share a cell. Texture hardware filters at most three dimensions, so a 4D lookup does the 16 reads by hand or combines two trilinear fetches with a final linear step. The cost is $O(16NC)$, dominated by memory reads.
:::

## Camera model, NeRF and Gaussian splatting

**Pinhole camera.** A camera moves a world point into its own coordinates with the extrinsics, $\mathbf{X}_c = R\mathbf{X}_w + \mathbf{t}$, and projects it with the intrinsics $K$: by similar triangles a point at depth $Z$ lands at $u = f_x X/Z + c_x$, $v = f_y Y/Z + c_y$, or in matrix form $Z\,(u, v, 1)^\top = K\mathbf{X}_c$ with $K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$, where $f_x, f_y$ are focal lengths in pixels and $(c_x, c_y)$ is the principal point (Hartley and Zisserman, 2004). Example: with $f_x = f_y = 800$ and $(c_x, c_y) = (320, 240)$, the camera-frame point $(0.5, -0.25, 4)$ lands at $u = 800 \cdot 0.125 + 320 = 420$, $v = 800 \cdot (-0.0625) + 240 = 190$. The depth divides out, so every point on the ray from the camera center $\mathbf{o} = -R^\top\mathbf{t}$ in direction $\mathbf{d} \propto R^\top K^{-1}(u, v, 1)^\top$ lands on the same pixel. NeRF and Gaussian splatting start from exactly these rays, with $K$, $R$ and $\mathbf{t}$ for every training photo estimated by structure from motion, usually with COLMAP (Schönberger and Frahm, 2016).

**NeRF** (Mildenhall et al., 2020) represents one scene as an MLP that maps a 3D position and a viewing direction to a volume density $\sigma$ and a color $\mathbf{c}$ (density depends only on position, so geometry is consistent across views; color also depends on direction, which models reflections). A pixel's color is rendered by sampling points $t_i$ along its camera ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ and compositing them,

$$
\hat C = \sum_i T_i\,\big(1 - e^{-\sigma_i\delta_i}\big)\,\mathbf{c}_i, \qquad T_i = \exp\Big(-\sum_{j < i}\sigma_j\delta_j\Big),
$$

where $\delta_i$ is the spacing between samples, $1 - e^{-\sigma_i\delta_i}$ is the opacity of segment $i$, and $T_i$ is the transmittance, the probability that light reaches it unblocked. Rendering is differentiable, so the MLP is fitted to posed photographs (poses from structure from motion) by minimizing pixel error. Two details made it work: a positional encoding that maps each coordinate $p$ to $\big(\sin(2^k\pi p), \cos(2^k\pi p)\big)$ for $k = 0, \dots, L - 1$ ($L = 10$ for position, 4 for direction), without which the MLP produces only blurry, low-frequency results; and hierarchical sampling, where a coarse network's weights along each ray decide where 128 extra samples go for a fine network (64 coarse samples per ray). Its costs are per-scene optimization (around 1–2 days per scene on one V100 in the paper) and slow rendering, since each pixel needs hundreds of network evaluations.

**3D Gaussian splatting** (Kerbl et al., 2023) replaces the network with a large set of explicit 3D Gaussians (often millions), initialized from the structure-from-motion points. Each has a position, a covariance stored as a rotation and per-axis scales ($\Sigma = R S S^\top R^\top$, which stays positive semi-definite during optimization), an opacity, and view-dependent color as spherical-harmonic coefficients. To render, each Gaussian is projected to a 2D Gaussian on the image (covariance $J W \Sigma W^\top J^\top$, with $W$ the viewing transform and $J$ the Jacobian of the projection), the splats are sorted by depth per screen tile, and each pixel alpha-blends them front to back with NeRF's compositing formula. All parameters are optimized on photometric loss, with periodic densification (cloning small and splitting large Gaussians where the gradient is large) and pruning of transparent ones. Rasterization makes rendering real-time (30 frames per second or more at 1080p in the paper) at quality comparable to the best NeRF variants, at the cost of much more memory.

::: {.qa asked="Ant digital human (Nowcoder roundup 2025)"}
**How do multi-view stereo and NeRF differ in where they sample 3D points?**

Multi-view stereo (MVS) tests a fixed set of depth hypotheses for each pixel of a reference view and picks one by matching; NeRF samples points along each pixel's ray, at positions that change every iteration, and integrates a learned density and color. Plane-sweep MVS chooses $D$ depths between near and far bounds (MVSNet used 256), warps each source image or feature map into the reference view as if the scene were a plane at each depth, scores photometric consistency (NCC, or the variance of warped features across views) into a cost volume over pixels and depths, regularizes it (semi-global matching or a 3D CNN), and takes a (soft) argmin per pixel before fusing depth maps. NeRF draws one uniform sample per bin along the ray (stratified sampling), evaluates a coarse network, turns the weights $w_i = T_i(1 - e^{-\sigma_i\delta_i})$ into a piecewise-constant distribution, and draws more samples from it by inverse-transform sampling for the fine network (64 plus 128 in the paper). Trade-offs: learned MVS is feed-forward and outputs depth directly but needs texture and roughly Lambertian surfaces and handles occlusion heuristically; NeRF models occlusion through transmittance and view-dependent appearance but needs per-scene optimization and wastes samples in empty space. They combine well: MVS depth can concentrate NeRF's samples near surfaces.
:::

## From-scratch code

Each snippet below was run with Python 3, NumPy and PyTorch 2.11, and the printed results are shown in comments.

**IoU and NMS**, checked against `torchvision.ops.nms`. The IoU function broadcasts to the full $N \times M$ matrix; NMS keeps a box and drops everything that overlaps it by more than the threshold, as in the worked example.

```python
import numpy as np, torch, torchvision

def box_iou(a, b):
    """a: (N, 4), b: (M, 4) boxes as (x1, y1, x2, y2). Returns the (N, M) IoU matrix."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])          # top-left of each intersection
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])          # bottom-right
    inter = np.clip(br - tl, 0, None).prod(-1)               # zero if the boxes do not overlap
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)

def nms(boxes, scores, thr=0.5):
    order = np.argsort(-scores, kind="stable")    # highest score first
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)                            # the best remaining box survives
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= thr]            # drop boxes that overlap it too much
    return np.array(keep)

print(box_iou(np.array([[1., 1, 5, 5]]), np.array([[3., 2, 7, 6], [5, 5, 9, 9]])))  # [[0.2308 0.]]

rng = np.random.default_rng(0)
xy = rng.uniform(0, 100, (500, 2)); wh = rng.uniform(10, 40, (500, 2))
boxes = np.hstack([xy, xy + wh]); scores = rng.random(500)
ref = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()
mine = nms(boxes, scores, 0.5)
print(len(mine), np.array_equal(mine, ref))   # 274 True (both sorted by decreasing score)
```

**Bilinear resize**, checked against `F.interpolate` for both values of `align_corners`, for upsampling and downsampling and non-integer factors. Without `antialias=True`, PyTorch's bilinear downsampling uses the same 2-tap kernel, so it matches; with antialiasing it would not.

```python
import numpy as np, torch, torch.nn.functional as F

def bilinear_resize(img, out_h, out_w, align_corners=False):
    """img: (H, W) or (H, W, C) float array."""
    H, W = img.shape[:2]
    def coords(n_out, n_in):
        j = np.arange(n_out)
        if align_corners:                          # corner pixel centers map onto each other
            x = j * (n_in - 1) / max(n_out - 1, 1)
        else:                                      # pixel areas map onto each other
            x = np.clip((j + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        x0 = np.minimum(np.floor(x).astype(int), n_in - 1)
        x1 = np.minimum(x0 + 1, n_in - 1)          # clamp at the border
        return x0, x1, x - x0
    y0, y1, dy = coords(out_h, H)
    x0, x1, dx = coords(out_w, W)
    extra = (None,) * (img.ndim - 2)               # broadcast over channels if present
    dy, dx = dy[(slice(None), None) + extra], dx[(None, slice(None)) + extra]
    top = img[y0][:, x0] * (1 - dx) + img[y0][:, x1] * dx   # along x on row y0
    bot = img[y1][:, x0] * (1 - dx) + img[y1][:, x1] * dx   # along x on row y1
    return top * (1 - dy) + bot * dy                         # then along y

img = np.random.default_rng(0).random((7, 9, 3))
t = torch.tensor(img).permute(2, 0, 1)[None]               # (1, C, H, W)
for ac in (False, True):
    for size in [(14, 18), (5, 4), (10, 13)]:
        ref = F.interpolate(t, size=size, mode="bilinear", align_corners=ac)[0].permute(1, 2, 0).numpy()
        print(ac, size, np.abs(bilinear_resize(img, *size, ac) - ref).max() < 1e-12)   # all True
```

**Average precision** for the toy example of the mAP section: greedy matching in score order, then COCO-style 101-point interpolation, at one IoU threshold and averaged over ten.

```python
import numpy as np

def average_precision(tp, n_gt, n_points=101):
    """tp: 0/1 flags of detections sorted by descending score."""
    tp = np.asarray(tp, float)
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]   # best precision at recall >= r
    idx = np.searchsorted(recall, np.linspace(0, 1, n_points), side="left")
    return np.mean([envelope[i] if i < len(tp) else 0.0 for i in idx])   # unreached recall -> 0

def match(dets, iou_thr):
    """dets: list of (score, {gt_id: iou}). Returns TP flags in score order."""
    taken, flags = set(), []
    for score, ious in sorted(dets, key=lambda d: -d[0]):
        cands = [(v, g) for g, v in ious.items() if g not in taken and v >= iou_thr]
        if cands:
            taken.add(max(cands)[1]); flags.append(1)    # best free ground truth
        else:
            flags.append(0)                              # duplicate or poor overlap
    return flags

dets = [(0.95, {"g1": 0.82}), (0.90, {"g1": 0.71}), (0.80, {"g3": 0.66}),
        (0.70, {"g2": 0.58}), (0.60, {"g4": 0.30}), (0.50, {})]
ap50 = average_precision(match(dets, 0.5), n_gt=4)
ap = np.mean([average_precision(match(dets, t), n_gt=4) for t in np.linspace(0.5, 0.95, 10)])
print(round(ap50, 4), round(ap, 4))                               # 0.6287 0.2875
print(round(average_precision(match(dets, 0.5), 4, n_points=11), 4))   # 0.6136
```

::: {.qa asked="Xiaomi, Baidu, Ctrip, Horizon (4 reports)"}
**Implement IoU and non-maximum suppression.**

Use the code above. For IoU, the intersection's top-left corner is the elementwise maximum of the top-left corners and its bottom-right corner the minimum of the bottom-right ones; clip its width and height at zero and divide by the sum of the areas minus the intersection; broadcasting gives the $N \times M$ matrix at once. For NMS, sort by score, keep the top box, drop the remaining boxes whose IoU with it exceeds the threshold, and repeat; worst case $O(N^2)$. Run it per class, or offset boxes by class index so classes never overlap. Mention Soft-NMS for crowded scenes, polygon intersection (Sutherland–Hodgman clipping) for rotated boxes, and accumulating intersections and unions before dividing when computing mIoU over a dataset.
:::

## Rapid-fire review

::: qa
**Why regress $\log(w/w_a)$ rather than $w - w_a$?**

The log ratio is unchanged by translating or rescaling the image, unconstrained in sign, symmetric between doubling and halving, and its errors are relative errors, so small and large objects are treated alike.
:::

::: qa
**IoU of two $10 \times 10$ boxes offset by one pixel in $x$ and in $y$?**

Intersection $9 \times 9 = 81$, union $200 - 81 = 119$, IoU $= 0.68$.
:::

::: qa
**What is the worst-case cost of greedy NMS, and when does it happen?**

$O(N^2)$ IoU computations, when few boxes overlap so almost every box is kept and compared with all remaining ones. Pre-filtering to the top few thousand by score bounds it.
:::

::: qa
**Why does RoIAlign matter more for masks than for classification?**

The class does not change when the crop shifts by a few pixels, but a mask shifted by a few pixels loses IoU; RoI pooling's rounding causes shifts of up to half a feature cell.
:::

::: qa
**What is the focal-loss factor for an example with $p_t = 0.9$ and $\gamma = 2$?**

$(1 - 0.9)^2 = 0.01$: its loss is 100 times smaller than under cross-entropy.
:::

::: qa
**Which FPN level does a $112 \times 112$ RoI use?**

$\lfloor 4 + \log_2(112/224) \rfloor = 3$, so $P_3$.
:::

::: qa
**What does FCOS's center-ness do?**

It predicts how central a location is within its box, $\sqrt{\tfrac{\min(l,r)}{\max(l,r)}\tfrac{\min(t,b)}{\max(t,b)}}$, and multiplies the class score at test time so off-center, poorly localized boxes rank lower before NMS.
:::

::: qa
**Why is DETR's matching done with the Hungarian algorithm rather than greedily?**

Greedy assignment can be far from the minimum total cost (in the example, $-0.6$ against $-1.45$); the Hungarian algorithm finds the optimal one-to-one matching in $O(N^3)$.
:::

::: qa
**What does a dilated $3 \times 3$ convolution with rate 6 cover?**

13 pixels per side with 9 weights: span $2r + 1$.
:::

::: qa
**Why does Mask R-CNN use per-class sigmoid masks rather than a softmax over classes?**

It decouples masks from classification: the box head decides the class, and the mask head only separates object from background, without classes competing per pixel.
:::

::: qa
**Write the relation between Dice and IoU.**

$D = 2J/(1 + J)$, from $J = I/(S - I)$ and $D = 2I/S$; they rank predictions of one image identically.
:::

::: qa
**What does PQ factor into?**

Segmentation quality (mean IoU of matched segments) times recognition quality ($|\mathrm{TP}|/(|\mathrm{TP}| + \tfrac12|\mathrm{FP}| + \tfrac12|\mathrm{FN}|)$, an F1 score); matches need IoU above 0.5, which makes them unique.
:::

::: qa
**Why are pose keypoints predicted as heatmaps?**

Heatmaps keep the task fully convolutional and translation-equivariant, allow several candidates, and train more easily than regressing coordinates from a pooled vector; sub-pixel refinement fixes the stride quantization.
:::

::: qa
**What is the aperture problem?**

Linearized brightness constancy gives one equation for two unknowns per pixel, so only the flow component along the image gradient is observable.
:::

::: qa
**How much does separability save for a $19 \times 19$ Gaussian?**

$361$ versus $38$ multiply-adds per pixel, a factor of 9.5.
:::

::: qa
**What degradation model are super-resolution training pairs usually synthesized with?**

$y = (k * x)\downarrow_s + n$: blur with a kernel (often a random isotropic or anisotropic Gaussian), keep every $s$-th pixel, add noise; the blur doubles as the anti-aliasing filter.
:::

::: qa
**What does a 0.4 cycles-per-pixel pattern become after keeping every second pixel without blurring?**

0.8 cycles per new pixel, above the new Nyquist limit of 0.5, so it aliases to 0.2: a false, coarser pattern.
:::

::: qa
**What are the bicubic weights halfway between pixels with $a = -0.5$?**

$(-0.0625, 0.5625, 0.5625, -0.0625)$; the negative lobes sharpen and can overshoot.
:::

::: qa
**What makes 3D Gaussian splatting faster to render than NeRF?**

It rasterizes explicit Gaussians, projected and alpha-blended per screen tile, instead of evaluating a neural network at hundreds of points along every ray.
:::

## References

1. Krizhevsky, Sutskever and Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," 2012.
2. Russakovsky et al., "ImageNet Large Scale Visual Recognition Challenge," 2015. [arXiv:1409.0575](https://arxiv.org/abs/1409.0575)
3. Szegedy et al., "Rethinking the Inception Architecture for Computer Vision," 2016. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
4. Zhang, Cisse, Dauphin and Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," 2018. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
5. Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features," 2019. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
6. DeVries and Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout," 2017. [arXiv:1708.04552](https://arxiv.org/abs/1708.04552)
7. Cubuk et al., "RandAugment: Practical Automated Data Augmentation with a Reduced Search Space," 2020. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
8. Thulasidasan et al., "On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks," 2019. [arXiv:1905.11001](https://arxiv.org/abs/1905.11001)
9. Yosinski et al., "How Transferable Are Features in Deep Neural Networks?," 2014. [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
10. Kornblith, Shlens and Le, "Do Better ImageNet Models Transfer Better?," 2019. [arXiv:1805.08974](https://arxiv.org/abs/1805.08974)
11. Kumar et al., "Fine-Tuning Can Distort Pretrained Features and Underperform Out-of-Distribution," 2022. [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)
12. Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization," 2017. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
13. Northcutt, Athalye and Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks," 2021. [arXiv:2103.14749](https://arxiv.org/abs/2103.14749)
14. Northcutt, Jiang and Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels," 2021.
15. Zhang and Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels," 2018. [arXiv:1805.07836](https://arxiv.org/abs/1805.07836)
16. Han et al., "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels," 2018. [arXiv:1804.06872](https://arxiv.org/abs/1804.06872)
17. Patrini et al., "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach," 2017.
18. Girshick, Donahue, Darrell and Malik, "Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation," 2014. [arXiv:1311.2524](https://arxiv.org/abs/1311.2524)
19. Girshick, "Fast R-CNN," 2015. [arXiv:1504.08083](https://arxiv.org/abs/1504.08083)
20. Ren, He, Girshick and Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," 2015. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
21. He, Gkioxari, Dollár and Girshick, "Mask R-CNN," 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
22. Redmon, Divvala, Girshick and Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," 2016. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
23. Redmon and Farhadi, "YOLO9000: Better, Faster, Stronger," 2017. [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)
24. Redmon and Farhadi, "YOLOv3: An Incremental Improvement," 2018. [arXiv:1804.02767](https://arxiv.org/abs/1804.02767)
25. Wang et al., "YOLOv10: Real-Time End-to-End Object Detection," 2024.
26. Liu et al., "SSD: Single Shot MultiBox Detector," 2016. [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)
27. Lin, Dollár, Girshick, He, Hariharan and Belongie, "Feature Pyramid Networks for Object Detection," 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
28. Lin, Goyal, Girshick, He and Dollár, "Focal Loss for Dense Object Detection," 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
29. Bodla, Singh, Chellappa and Davis, "Soft-NMS: Improving Object Detection with One Line of Code," 2017. [arXiv:1704.04503](https://arxiv.org/abs/1704.04503)
30. Rezatofighi et al., "Generalized Intersection over Union: A Metric and a Loss for Bounding Box Regression," 2019. [arXiv:1902.09630](https://arxiv.org/abs/1902.09630)
31. Tian, Shen, Chen and He, "FCOS: Fully Convolutional One-Stage Object Detection," 2019. [arXiv:1904.01355](https://arxiv.org/abs/1904.01355)
32. Zhou, Wang and Krähenbühl, "Objects as Points," 2019. [arXiv:1904.07850](https://arxiv.org/abs/1904.07850)
33. Law and Deng, "CornerNet: Detecting Objects as Paired Keypoints," 2018. [arXiv:1808.01244](https://arxiv.org/abs/1808.01244)
34. Carion et al., "End-to-End Object Detection with Transformers," 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
35. Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection," 2021. [arXiv:2010.04159](https://arxiv.org/abs/2010.04159)
36. Chen, Sun, Song and Luo, "DiffusionDet: Diffusion Model for Object Detection," 2023. [arXiv:2211.09788](https://arxiv.org/abs/2211.09788)
37. Everingham et al., "The PASCAL Visual Object Classes (VOC) Challenge," 2010.
38. Lin et al., "Microsoft COCO: Common Objects in Context," 2014. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
39. Long, Shelhamer and Darrell, "Fully Convolutional Networks for Semantic Segmentation," 2015. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
40. Ronneberger, Fischer and Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
41. Chen et al., "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs," 2017. [arXiv:1606.00915](https://arxiv.org/abs/1606.00915)
42. Chen, Papandreou, Schroff and Adam, "Rethinking Atrous Convolution for Semantic Image Segmentation," 2017. [arXiv:1706.05587](https://arxiv.org/abs/1706.05587)
43. Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation," 2018. [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)
44. Milletari, Navab and Ahmadi, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation," 2016. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
45. Berman, Triki and Blaschko, "The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-over-Union Measure in Neural Networks," 2018. [arXiv:1705.08790](https://arxiv.org/abs/1705.08790)
46. Kirillov, He, Girshick, Rother and Dollár, "Panoptic Segmentation," 2019. [arXiv:1801.00868](https://arxiv.org/abs/1801.00868)
47. Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation," 2022. [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)
48. Kirillov et al., "Segment Anything," 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
49. Ravi et al., "SAM 2: Segment Anything in Images and Videos," 2024. [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
50. Newell, Yang and Deng, "Stacked Hourglass Networks for Human Pose Estimation," 2016. [arXiv:1603.06937](https://arxiv.org/abs/1603.06937)
51. Cao, Simon, Wei and Sheikh, "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields," 2017. [arXiv:1611.08050](https://arxiv.org/abs/1611.08050)
52. Xiao, Wu and Wei, "Simple Baselines for Human Pose Estimation and Tracking," 2018. [arXiv:1804.06208](https://arxiv.org/abs/1804.06208)
53. Sun, Xiao, Liu and Wang, "Deep High-Resolution Representation Learning for Human Pose Estimation," 2019. [arXiv:1902.09212](https://arxiv.org/abs/1902.09212)
54. Eigen, Puhrsch and Fergus, "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network," 2014. [arXiv:1406.2283](https://arxiv.org/abs/1406.2283)
55. Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer," 2020. [arXiv:1907.01341](https://arxiv.org/abs/1907.01341)
56. Yang et al., "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data," 2024. [arXiv:2401.10891](https://arxiv.org/abs/2401.10891)
57. Godard, Mac Aodha and Brostow, "Unsupervised Monocular Depth Estimation with Left-Right Consistency," 2017. [arXiv:1609.03677](https://arxiv.org/abs/1609.03677)
58. Lucas and Kanade, "An Iterative Image Registration Technique with an Application to Stereo Vision," 1981.
59. Horn and Schunck, "Determining Optical Flow," 1981.
60. Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow," 2020. [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)
61. Keys, "Cubic Convolution Interpolation for Digital Image Processing," 1981.
62. Zhang, Liang, Van Gool and Timofte, "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution," 2021.
63. Zhang, "Making Convolutional Networks Shift-Invariant Again," 2019.
64. Hartley and Zisserman, *Multiple View Geometry in Computer Vision*, 2nd edition, 2004.
65. Schönberger and Frahm, "Structure-from-Motion Revisited," 2016.
66. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," 2020. [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)
67. Kerbl, Kopanas, Leimkühler and Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering," 2023. [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)
