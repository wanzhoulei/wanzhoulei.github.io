---
title: "Flow Matching and Few-Step Generation"
subtitle: "Continuous flows, flow matching and rectified flow, why they coincide with diffusion, and the distillation and consistency methods that cut sampling to one to four steps."
order: 4
updated: 2026-11-08
---

Flow matching trains a continuous normalizing flow by regressing a neural velocity field onto the velocity of a path you design between noise and data, with no ODE simulation during training. With a Gaussian source it describes the same model class as diffusion, and since SD3 and FLUX.1 it is the usual way large image generators are written down. Sampling still integrates an ODE with tens of network calls, so the second half covers methods that reach one to four steps by giving up some training simplicity, diversity or flexibility. Interviewers use this area to check that you can derive the conditional flow matching identity, convert between parameterizations, and say precisely why straight paths and distillation make sampling cheaper. DDPM basics, the SDE view and the CFG derivation are in [diffusion models](diffusion-models.html); GAN theory is in [GANs, VAEs and autoregressive models](gans-vaes-autoregressive.html).

**Convention.** Unless a section says otherwise, $t=0$ is noise and $t=1$ is data, as in Lipman et al., Liu et al. and the Flow Matching Guide. Noise is $\epsilon\sim\mathcal{N}(0,I)$ and data is $x_1\sim q$. SD3, the FLUX code and MeanFlow flip this ($t=0$ data, $t=1$ noise); sections that follow them say so.

## Continuous normalizing flows

A continuous normalizing flow (CNF) is an ODE $\frac{d}{dt}x_t=v_t(x_t)$ with a neural velocity field, started from $x_0\sim p_0=\mathcal{N}(0,I)$. Its flow map pushes $p_0$ forward to a density path $p_t$, and the pair satisfies the continuity equation

$$\partial_t p_t(x)+\nabla\cdot\big(p_t(x)\,v_t(x)\big)=0 .$$

It is conservation of probability mass: density in a small volume changes only through the net flux $p_tv_t$ across its boundary. We say $v_t$ *generates* $p_t$ when the two satisfy it.

**Log-density along a trajectory.** Expanding the divergence gives $\partial_tp_t+v_t\cdot\nabla p_t+p_t\nabla\cdot v_t=0$. The first two terms are the total derivative of $p_t(x_t)$ along a trajectory, so

$$\frac{d}{dt}\log p_t(x_t)=-(\nabla\cdot v_t)(x_t),\qquad\log p_1(x_1)=\log p_0(x_0)-\int_0^1(\nabla\cdot v_t)(x_t)\,dt .$$

**Why maximum likelihood is expensive.** The likelihood of one training image needs (i) a sequential ODE solve from the image back to noise, whose number of function evaluations (NFE) tends to grow as the learned field gets more complex; (ii) the divergence along the path, which costs $d$ vector-Jacobian products exactly or one with Hutchinson's unbiased but noisy estimator $\mathbb{E}_z[z^\top(\partial v/\partial x)z]$ (FFJORD); and (iii) backpropagation through the solve, usually by the adjoint method, a second ODE solve. One training step costs tens to hundreds of network calls. Flow matching keeps the model, replaces the objective with a regression at one random $t$ per example, and leaves the ODE only at sampling time.

::: key
A CNF is specified by its velocity field, and the continuity equation tells you which density path that field produces. Flow matching reverses the order: choose the path first, then regress the field that produces it.
:::

::: qa
**Why does the CNF log-likelihood contain a trace where discrete flows have a log-determinant?**

For one Euler step $x\mapsto x+h\,v(x)$ the Jacobian is $I+hJ$ with $J=\partial v/\partial x$, and $\log\det(I+hJ)=h\,\mathrm{tr}(J)+O(h^2)$. Composing infinitely many small steps turns the sum of log-determinants into the time integral of the trace, the divergence in the formula above. So CNFs need no triangular or coupling structure. The catch is that the exact trace still costs $d$ backward passes, so practical CNFs use Hutchinson's estimator and accept extra gradient variance.
:::

## Flow matching

### The objective we cannot compute

Given a density path $p_t$ from $\mathcal{N}(0,I)$ to $p_1\approx q$ and the field $u_t$ that generates it, we would regress

$$\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t\sim U[0,1],\;x\sim p_t}\big\lVert v^\theta_t(x)-u_t(x)\big\rVert^2 .$$

For a data distribution $q$, neither $p_t$ nor $u_t$ has a closed form.

### Marginal paths built from conditional paths

Condition on one data point. Pick a simple conditional path $p_t(x\mid x_1)$ from $\mathcal{N}(0,I)$ to $\mathcal{N}(x_1,\sigma_{\min}^2I)$ and a conditional field $u_t(x\mid x_1)$ that generates it. Then mix over data:

$$p_t(x)=\int p_t(x\mid x_1)\,q(x_1)\,dx_1,\qquad u_t(x)=\int u_t(x\mid x_1)\,\frac{p_t(x\mid x_1)\,q(x_1)}{p_t(x)}\,dx_1=\mathbb{E}\big[u_t(X_t\mid X_1)\,\big|\,X_t=x\big].$$

The marginal field is the posterior average of the conditional fields, and it generates the marginal path. Differentiate under the integral and use the conditional continuity equation:

$$\partial_tp_t(x)=\int\partial_tp_t(x\mid x_1)\,q(x_1)\,dx_1=-\nabla\cdot\int u_t(x\mid x_1)\,p_t(x\mid x_1)\,q(x_1)\,dx_1=-\nabla\cdot\big(u_t(x)\,p_t(x)\big).$$

### Conditional flow matching and the gradient identity

Replace the unknown marginal target with the conditional one:

$$\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t,\;x_1\sim q,\;x\sim p_t(\cdot\mid x_1)}\big\lVert v^\theta_t(x)-u_t(x\mid x_1)\big\rVert^2 .$$

**Claim (Lipman et al., Theorem 2).** If $p_t(x)>0$, the two losses differ by a constant independent of $\theta$, so their gradients agree.

**Proof.** Expand $\lVert v-u\rVert^2=\lVert v\rVert^2-2\langle v,u\rangle+\lVert u\rVert^2$. The last term has no $\theta$. The first has the same expectation in both losses because $p_t(x)$ is the $q$-mixture of $p_t(x\mid x_1)$. For the cross term, insert the definition of $u_t$ and swap integrals:

$$\mathbb{E}_{p_t(x)}\big\langle v^\theta_t(x),u_t(x)\big\rangle=\int\!\!\int\big\langle v^\theta_t(x),u_t(x\mid x_1)\big\rangle\,p_t(x\mid x_1)\,q(x_1)\,dx_1\,dx=\mathbb{E}_{q(x_1),\,p_t(x\mid x_1)}\big\langle v^\theta_t(x),u_t(x\mid x_1)\big\rangle .$$

Equivalently, least squares on a noisy target learns the conditional mean: with $U=u_t(X_t\mid X_1)$, $\mathcal{L}_{\mathrm{CFM}}=\mathcal{L}_{\mathrm{FM}}+\mathbb{E}_{t,X_t}\,\mathrm{tr}\,\mathrm{Cov}[U\mid X_t]$. The Guide extends the identity to any Bregman divergence.

**The algorithm** with the linear path: sample $x_1\sim q$, $\epsilon\sim\mathcal{N}(0,I)$, $t\sim U[0,1]$; set $x_t=t\,x_1+(1-t)\,\epsilon$; regress $v^\theta_t(x_t)$ onto $x_1-\epsilon$. To sample, integrate $dx/dt=v^\theta_t(x)$ from $\epsilon$ at $t=0$ to $t=1$.

::: pitfall
The CFM loss does not go to zero at the optimum. Its floor is the conditional-variance term, which is large at high noise, where many data points are compatible with the same $x_t$. A plateau is not evidence of underfitting, and loss values are not comparable across paths, couplings or timestep distributions.
:::

::: qa
**The CFM target depends on a data point $x_1$ that the model never sees. Why does regressing on it give the right field?**

Squared error is minimized by the conditional expectation of the target given the input, and the marginal velocity is exactly $\mathbb{E}[u_t(X_t\mid X_1)\mid X_t=x]$. So the population minimizer of CFM is the marginal field, and the losses differ only by the conditional variance, a constant in $\theta$. The argument depends on the loss: $\ell_1$ would learn a coordinate-wise conditional median, which in general does not generate $p_t$. Bregman divergences work because they are exactly the losses minimized by the conditional mean.
:::

::: {.qa asked="Xiaomi AIGC algorithm engineer, Nowcoder 2025"}
**Explain the "flow" mechanism in FLUX-style models: what is trained, and how is an image sampled?**

Black Forest Labs describe FLUX.1 as a rectified flow transformer but have not published its training recipe, so the training side rests on the closest published recipe, SD3's: conditional flow matching with the linear path in SD3's convention. Encode an image to a latent $x$, draw $\epsilon$ and $t$ (SD3 uses a logit-normal), form $z_t=(1-t)x+t\epsilon$, and regress onto $v=\epsilon-x$, so the network learns $\mathbb{E}[\epsilon-x\mid z_t]$. Sampling integrates $dz/dt=v_\theta$ from noise at $t=1$ down to $t=0$; the public reference code uses Euler steps on a time grid shifted toward high noise by an amount that grows with the number of image tokens. The model card says [dev] was trained with guidance distillation, and the public code passes the guidance scale to the network as an input, so each step needs one forward pass instead of two.
:::

## Probability paths

### Gaussian conditional paths

Lipman et al. use $p_t(x\mid x_1)=\mathcal{N}\big(x\mid\mu_t(x_1),\sigma_t(x_1)^2I\big)$ with $\mu_0=0$, $\sigma_0=1$, $\mu_1=x_1$, $\sigma_1=\sigma_{\min}$, produced by the flow $\psi_t(\epsilon)=\sigma_t\epsilon+\mu_t$. Its time derivative is $\sigma_t'\epsilon+\mu_t'$; substituting $\epsilon=(x-\mu_t)/\sigma_t$ gives

$$u_t(x\mid x_1)=\frac{\sigma_t'}{\sigma_t}\big(x-\mu_t\big)+\mu_t' .$$

Practical paths are affine, $x_t=\alpha_tx_1+\sigma_t\epsilon$ with $\alpha_0=0,\ \sigma_0=1,\ \alpha_1=1,\ \sigma_1\approx0$, and the target is $\dot\alpha_tx_1+\dot\sigma_t\epsilon$:

| Path | $\alpha_t$ | $\sigma_t$ | Target |
|---|---|---|---|
| Linear, "OT" (Lipman) | $t$ | $1-(1-\sigma_{\min})t$ | $x_1-(1-\sigma_{\min})\epsilon$ |
| Rectified flow | $t$ | $1-t$ | $x_1-\epsilon$ |
| Trigonometric (variance preserving) | $\sin\frac{\pi t}{2}$ | $\cos\frac{\pi t}{2}$ | $\frac{\pi}{2}\big(\cos\frac{\pi t}{2}\,x_1-\sin\frac{\pi t}{2}\,\epsilon\big)$ |

Lipman et al. write VE and VP diffusion paths in the same form and report that the linear path trained faster and sampled with fewer NFE in their experiments.

**v-prediction is a velocity.** In Salimans and Ho's angle form $z_\phi=\cos\phi\,x+\sin\phi\,\epsilon$, so $dz_\phi/d\phi=\cos\phi\,\epsilon-\sin\phi\,x$, which is their $v$. With $\phi=\frac{\pi}{2}(1-t)$, $z_\phi$ is the trigonometric path above and $u_t=\frac{dz_\phi}{d\phi}\frac{d\phi}{dt}=-\frac{\pi}{2}v$, so $v=-\frac{2}{\pi}u_t$. The sign comes from $\phi$ decreasing as $t$ runs from noise to data; with data at $t=0$ the factor is $+2/\pi$.

### What "OT" does and does not mean

The linear conditional path is the straight-line interpolation between noise and the point mass at $x_1$. The Guide shows with Jensen's inequality that the marginal field's kinetic energy is bounded by that of the conditional flows, that straight lines minimize this bound, and hence

$$\int_0^1\mathbb{E}\,\lVert u_t(x_t)\rVert^2\,dt\;\le\;\mathbb{E}\,\lVert x_1-\epsilon\rVert^2 .$$

This is optimal transport only when the target is a single point. For Gaussian paths the marginal field is a gradient field (Guide, eq. 4.79), so it is kinetic-optimal for its own path $p_t$, but that path is not the OT interpolation between $\mathcal{N}(0,I)$ and $q$.

### Stochastic interpolants

Albergo and Vanden-Eijnden reached the same quadratic velocity loss concurrently, for arbitrary pairs of densities. Their later framework uses $x_t=I(t,x_0,x_1)+\gamma(t)z$ with $\gamma(0)=\gamma(1)=0$, and learns both a velocity $b$ and a score $s=-\gamma^{-1}\mathbb{E}[z\mid x_t]$ by quadratic regression. The SDEs $dX_t=(b+\varepsilon(t)s)\,dt+\sqrt{2\varepsilon(t)}\,dW_t$ share the same marginals for any $\varepsilon(t)\ge0$, so deterministic versus stochastic sampling becomes a choice made at sampling time.

::: pitfall
"OT path" describes each conditional path, a straight line to one data point, not the learned map. With noise and data paired independently, the marginal flow is not the optimal transport map from $\mathcal{N}(0,I)$ to $q$, and its trajectories are curved.
:::

::: qa
**Linear path versus a variance-preserving path: if both are Gaussian, what actually changes?**

Both are $x_t=\alpha_tx_1+\sigma_t\epsilon$ and differ only in the curve $\alpha_t^2/\sigma_t^2$ (the SNR) and the overall scale of $x_t$. The Guide gives a closed-form scale-time change between any two such schedules, so a trained field can be sampled under another schedule and, in exact arithmetic, maps noise to the same samples. What changes in practice is how training samples are spread over noise levels (an implicit loss weighting), what the network outputs, and how curved the trajectories look as functions of $t$, which sets the per-step solver error. Neither family is uniformly straighter on real data.
:::

## Rectified flow, reflow and straightness

Liu, Gong and Liu reached the same objective from transport. Given any coupling $(X_0,X_1)$ of $\pi_0$ and $\pi_1$, fit

$$\min_v\int_0^1\mathbb{E}\,\big\lVert(X_1-X_0)-v(X_t,t)\big\rVert^2dt,\qquad X_t=tX_1+(1-t)X_0 .$$

With Gaussian $\pi_0$ and an independent coupling this is CFM with the linear path, called the 1-rectified flow.

**Non-crossing and marginals.** The lines $X_t$ can cross, but trajectories of a well-defined ODE cannot. At a crossing the learned field averages directions, so the flow rewires the lines: it keeps the marginals, $\mathrm{Law}(Z_t)=\mathrm{Law}(X_t)$ (Theorem 3.3), through a new deterministic coupling $(Z_0,Z_1)$. For every convex cost $c$, $\mathbb{E}\,c(Z_1-Z_0)\le\mathbb{E}\,c(X_1-X_0)$ (Theorem 3.5), but rectification targets no particular cost, so it is not guaranteed to reach the OT map for a given $c$.

**Straightness and reflow.** Let $S(Z)=\int_0^1\mathbb{E}\lVert(Z_1-Z_0)-\dot Z_t\rVert^2dt$, zero exactly when every trajectory is straight at constant speed. Reflow simulates the $k$-rectified flow to get pairs $(Z_0^k,Z_1^k)$ and retrains on them. These pairs come from a non-crossing flow, so their lines cross much less. Theorem 3.7: $\min_{k\le K}S(Z^k)\le\mathbb{E}\lVert X_1-X_0\rVert^2/K$.

**Why straight means few steps.** The local error of an Euler step of size $h$ is about $\frac{h^2}{2}\ddot z$, with acceleration $\ddot z=\partial_tv+(\nabla v)\,v$. A straight trajectory at constant speed has zero acceleration, so $z_1=z_0+v(z_0,0)$ is exact. Curvature sets the step count, not path length.

**In practice.** Liu et al. distill a final one-step map on the reflow pairs, only at the end, since distillation fits the current coupling without improving it. InstaFlow applied reflow and distillation to Stable Diffusion and reached one-step FID 13.1 on MS COCO 2014 at about 0.09 s per image. Lee et al. argue that one reflow round is enough in realistic settings. Minibatch OT (Tong et al.; Pooladian et al.) straightens without reflow: solve an assignment between noise and data samples inside each batch and train on matched pairs. Marginals stay correct, crossings and gradient variance drop, and fewer NFE are needed, though gains shrink in high dimension with small batches and when pairs must share a condition.

::: key
Every conditional path is straight. The learned marginal field is curved because it averages over crossing paths, and a better coupling (reflow, minibatch OT) is what removes the crossings.
:::

::: qa
**Why are the learned trajectories curved if every training path is a straight line?**

At $(x,t)$ the model predicts $\mathbb{E}[x_1-\epsilon\mid x_t=x]$, an average over all noise-data pairs whose lines pass through $x$. Near $t=0$ the posterior over $x_1$ is close to the whole data distribution, so the field is roughly $\mathbb{E}[x_1]-x$ and every trajectory first heads toward the data mean. As $t$ grows the posterior concentrates on one mode and the trajectory bends toward it. With a two-point dataset you see it exactly: all paths move toward the midpoint, then split.
:::

::: qa
**What does reflow cost, and what can go wrong?**

Each round solves the teacher ODE for every training pair, often millions of solves, then retrains. The new model learns the previous model's coupling, so its endpoint distribution is the previous model's samples rather than the data: errors and lost modes accumulate, and quality is bounded by the teacher. Liu et al. warn against many rounds, and Lee et al. find one round already gives nearly straight paths. Any teacher bias, including a baked-in guidance scale, is inherited.
:::

## Flow matching and diffusion are one model class

Take an affine Gaussian path $x_t=\alpha_tx_1+\sigma_t\epsilon$. Conditioning both the target and the path on $x_t=x$ gives two linear equations,

$$u_t(x)=\dot\alpha_t\,\hat x_1+\dot\sigma_t\,\hat\epsilon,\qquad x=\alpha_t\,\hat x_1+\sigma_t\,\hat\epsilon,$$

with $\hat x_1=\mathbb{E}[x_1\mid x_t=x]$ and $\hat\epsilon=\mathbb{E}[\epsilon\mid x_t=x]$, and Tweedie's formula adds $\nabla\log p_t(x)=-\hat\epsilon/\sigma_t$. Any one of velocity, $x_1$-prediction, $\epsilon$-prediction and score fixes the other three. Eliminating $\hat x_1$ gives eq. 4.78 of the Guide:

$$u_t(x)=\frac{\dot\alpha_t}{\alpha_t}x+\Big(\frac{\dot\alpha_t\sigma_t^2}{\alpha_t}-\dot\sigma_t\sigma_t\Big)\nabla\log p_t(x).$$

For the linear path: $\hat x_1=x+(1-t)v$, $\hat\epsilon=x-tv$, $v=(x-\hat\epsilon)/t$, $\nabla\log p_t(x)=(tv-x)/(1-t)$.

**Samplers coincide.** In diffusion notation $z_s=a_sx+b_s\epsilon$ (any schedule), rescale $\tilde z=z/(a_s+b_s)$ and set $\tau=b_s/(a_s+b_s)$, so $\tilde z=(1-\tau)x+\tau\epsilon$, a rectified-flow path with the same SNR (noise at $\tau=1$, as in SD3). A deterministic DDIM step $z_{s'}=a_{s'}\hat x+b_{s'}\hat\epsilon$ becomes

$$\tilde z_{\tau'}=(1-\tau')\hat x+\tau'\hat\epsilon=\tilde z_\tau+(\tau'-\tau)(\hat\epsilon-\hat x),$$

an Euler step of the rectified-flow ODE. DDIM on any Gaussian schedule is Euler on the rescaled linear path (Gao et al.).

**Losses coincide up to weighting.** In SD3's convention ($z_t=(1-t)x+t\epsilon$, $v=\epsilon-x$, $\lambda_t=2\log\frac{1-t}{t}$), writing the velocity through an $\epsilon$-prediction gives $\hat v-v=(\hat\epsilon-\epsilon)/(1-t)$. SD3 writes every objective as $-\frac12\mathbb{E}_t[w_t\lambda_t'\lVert\hat\epsilon-\epsilon\rVert^2]$; since $-\frac12\lambda_t'=\frac{1}{t(1-t)}$, matching $\frac{1}{(1-t)^2}$ gives $w_t=\frac{t}{1-t}=e^{-\lambda_t/2}$. SD3 lists the same $e^{-\lambda/2}$ for v-prediction on the cosine schedule (up to a constant). Changing variables from $t$ to $\lambda$ turns either loss into $\frac12\int w(\lambda)\,\mathbb{E}\lVert\hat\epsilon-\epsilon\rVert^2\,d\lambda$, so the expected objective over the implied denoiser is the same. Uniform $t$ samples $\lambda$ from different densities on the two schedules, but the factor $\lambda_t'$ compensates, so the density only changes the variance of the loss estimate (Gao et al.). What differs is the network's output and input scaling.

**What actually differs in practice.**

1. *Output parameterization.* $\hat x_1$ and $\hat\epsilon$ follow from $v$ without division; recovering $\hat x$ from $\hat\epsilon$ divides by $\alpha_t$, which vanishes at pure noise.
2. *Weighting and timestep density* over log-SNR (Kingma and Gao; SD3).
3. *Endpoints.* The linear path reaches zero SNR at the noise end; several DDPM schedules in use do not, which Lin et al. linked to Stable Diffusion's trouble with very bright or dark images.
4. *Generality.* CFM needs only samples of the source, so non-Gaussian sources and data-to-data couplings fit directly; the equivalence assumes a Gaussian source and independent coupling.

::: pitfall
"Flow matching samples faster because its paths are straight." The conditional paths are straight; the marginal ODE is not, and with the same SNR schedule DDIM on a diffusion model takes the same steps as Euler on the flow. Gao et al. also argue that on realistic data no schedule is uniformly straighter. Speed differences come from the schedule, the timestep allocation and the parameterization.
:::

::: qa
**Given a velocity prediction on the linear path, recover $\hat x_1$, $\hat\epsilon$ and the score. Where are the singularities?**

From $x_t=tx_1+(1-t)\epsilon$ and $v=\hat x_1-\hat\epsilon$: $\hat x_1=x_t+(1-t)v$, $\hat\epsilon=x_t-tv$, and $\nabla\log p_t=-\hat\epsilon/(1-t)$. The score blows up as $t\to1$ (data end) because $\sigma_t\to0$, whichever output you start from. In the other direction, $v=(x_t-\hat\epsilon)/t$ blows up as $t\to0$, so an $\epsilon$-prediction model gives no usable velocity at pure noise. Velocity is the only one of the three predictions that gives the other two ($\hat x_1$ and $\hat\epsilon$) without division at either end.
:::

::: qa
**Is uniform-$t$ rectified-flow training different from v-prediction diffusion?**

As an expected objective over the implied denoiser, no: written as a weighted $\epsilon$-MSE over log-SNR, both put weight $e^{-\lambda/2}$ on each noise level (SD3's weighting table; Gao et al.). What is not equal is the distribution of noise levels sampled: uniform $t$ induces different densities over $\lambda$, but the factor $|d\lambda/dt|$ reweights each sample, so only the estimator's variance changes. They also differ in the output target and the input scaling (a variance-preserving state versus a linear-path state), and these matter in practice: in SD3's Table 1 uniform rectified flow ranked well above v/cos but below the $\epsilon$/linear baseline, while rectified flow with logit-normal timesteps ranked first.
:::

::: {.qa asked="Image-generation vision intern, Nowcoder 2024"}
**What changed from SDXL to SD3, especially in the generative formulation?**

SDXL is a U-Net trained with $\epsilon$-prediction on a discrete variance-preserving DDPM schedule and sampled with DDIM or DPM-Solver-type samplers. SD3 switches to rectified flow: linear path, velocity target $\epsilon-x$, logit-normal timestep sampling, and a resolution-dependent timestep shift. The backbone becomes a transformer with separate weights for image and text tokens, on a 16-channel autoencoder. SD3's comparison found that the gain came from rectified flow combined with the non-uniform timestep density, and that rectified-flow variants degraded less when sampling steps were reduced.
:::

## Timestep sampling and resolution shift

*In this section $t$ is the noise fraction, as in SD3: $z_t=(1-t)x+t\epsilon$, $v=\epsilon-x$.*

**Why not uniform $t$.** Near either end the regression is easy: when $t\to0$, $z_t\approx x$ and $\epsilon$ can only be predicted by its mean; when $t\to1$, $z_t\approx\epsilon$ and $x$ can only be predicted by the data mean. The middle is where the model learns structure. Sampling $t\sim\pi$ equals uniform $t$ with the loss multiplied by $\pi(t)$, so the timestep density is a loss weighting.

**Logit-normal.** Draw $u\sim\mathcal{N}(m,s^2)$, set $t=\mathrm{sigmoid}(u)$:

$$\pi_{\mathrm{ln}}(t;m,s)=\frac{1}{s\sqrt{2\pi}}\,\frac{1}{t(1-t)}\exp\Big(-\frac{(\mathrm{logit}\,t-m)^2}{2s^2}\Big).$$

$m>0$ moves mass toward noise and $s$ sets the width. SD3 also tried mode sampling with heavy tails and CosMap; the best configuration was $m=0$, $s=1$, which also ranked first at 5 sampling steps.

**Resolution shift.** At a fixed $t$, a larger image keeps more information, because many correlated pixels carry noisy copies of the same signal. SD3 makes this exact for a constant image of value $c$ with $n$ pixels: averaging $z_t/(1-t)$ estimates $c$ with standard deviation $\frac{t}{(1-t)\sqrt n}$. Equal uncertainty at $n$ and $m$ pixels requires

$$\frac{t_m}{1-t_m}=\sqrt{\tfrac{m}{n}}\,\frac{t_n}{1-t_n}\quad\Longrightarrow\quad t_m=\frac{\alpha\,t_n}{1+(\alpha-1)\,t_n},\qquad\alpha=\sqrt{m/n},$$

a log-SNR shift of $-2\log\alpha$. With $\alpha>1$ every timestep moves toward noise, where global layout is decided. SD3 used $\alpha=3.0$ at $1024^2$ for training and sampling, chosen by a human preference study. FLUX's public sampling code uses the same map with $\alpha=e^\mu$, $\mu$ rising linearly from 0.5 at 256 image tokens to 1.15 at 4096. Hoogeboom et al. had shifted noise schedules for high-resolution pixel diffusion earlier.

::: qa
**Why does SD3 sample $t$ from a logit-normal instead of uniformly, and what does it do to the loss?**

Velocity regression is nearly trivial at both ends, where one of $x$ and $\epsilon$ is almost determined by $z_t$ and the other can only be predicted by its mean. A logit-normal centered at $t=0.5$ spends more of the training budget on the informative middle. Since sampling from $\pi$ equals uniform sampling with the loss reweighted by $\pi(t)$, this changes the loss weighting over log-SNR, not the model or its optimum in the infinite-capacity limit.
:::

::: qa
**Which way does the resolution shift move $t$, and why is the shift factor tuned rather than taken from the formula?**

For more pixels, $\alpha=\sqrt{m/n}>1$, and $t_m=\alpha t_n/(1+(\alpha-1)t_n)$ pushes every step toward noise, because averaging over more correlated pixels makes the same nominal noise level less destructive. The derivation assumes a constant image, which real images are not, so the formula is only a rough guide. SD3 therefore picked $\alpha$ by human preference (3.0 at $1024^2$), and FLUX ties it to token count with a tuned linear rule.
:::

## Why sampling is slow: solvers versus distillation

Sampling cost is NFE times one network call, and classifier-free guidance doubles the calls unless it is distilled. NFE is set by solver error: an order-$p$ method has global error $O(h^p)$ with constants that grow with the curvature of the trajectories.

Training-free solvers reduce NFE without touching the model: DDIM is first order, EDM's Heun sampler used 35 NFE on CIFAR-10, DPM-Solver integrates the linear part of the ODE exactly and gets good samples in about 10 to 20 NFE, and DPM-Solver++ stays stable under large guidance in 15 to 20 steps, where its authors report DDIM usually needs 100 to 250. Below roughly 10 NFE a few polynomial steps cannot follow curved trajectories, and further gains require changing the model.

| Method | What the student learns | Teacher | Main loss | Steps |
|---|---|---|---|---|
| Progressive distillation | two teacher steps as one | yes | regression | 4 to 8 |
| Reflow + distill (InstaFlow) | the teacher's noise-image coupling | yes | regression | 1 |
| Consistency distillation, LCM | ODE point to trajectory endpoint | yes | self-consistency | 1 to 4 |
| Consistency training (iCT, sCT) | same, no teacher | no | self-consistency | 1 to 2 |
| DMD, DMD2 | the output distribution | yes (scores) | score difference, plus regression or GAN | 1 to 4 |
| ADD, LADD | realistic, teacher-consistent samples | yes | GAN, plus score distillation in ADD | 1 to 4 |
| Shortcut, MeanFlow | jumps of any size | no | self-consistency, identity | 1 to 2 |

::: {.qa asked="Taobao/Tmall Group AIGC image algorithm, Nowcoder 2025"}
**Explain common fast samplers such as DDIM and DPM-Solver++, and how LCM and Turbo models differ from them.**

DDIM is a deterministic first-order solver of the probability-flow ODE: each step forms $\hat x$ and $\hat\epsilon$ and recombines them at the next noise level, which equals an Euler step on the rescaled linear path. DPM-Solver++ integrates the linear part exactly and approximates the data prediction with a higher-order (usually second-order multistep) expansion in half-log-SNR $\log(\alpha_t/\sigma_t)$, giving guided samples in about 15 to 20 steps. Both are training-free, so diversity, guidance scale and negative prompts are unchanged. LCM and SDXL-Turbo change the network: LCM is consistency distillation in latent space on the guided ODE (2 to 4 steps), and Turbo is adversarial diffusion distillation, a GAN loss plus a score-distillation loss (1 to 4 steps, no CFG). The price is extra training, some diversity, and a fixed or narrowed guidance range.
:::

::: qa
**Why can't a better ODE solver get you to one step?**

A solver sees the field only where it evaluates it. With one step, the error is set by the curvature over the whole interval, and higher-order schemes need more evaluations per step while their error constants grow when $h$ spans the whole range. The ODE's noise-to-data map is deterministic and smooth, so a network can represent it, but it must be learned, which is what distillation, consistency training and reflow do.
:::

## Few-step methods

### Progressive distillation

*Diffusion notation: $z_t=\alpha_tx+\sigma_t\epsilon$, data at $t=0$.* A student initialized from the teacher learns to make one DDIM step from $t$ to $t''=t-1/N$ land where two teacher DDIM steps land. Its step with $x$-prediction $\tilde x$ is $z_{t''}=\alpha_{t''}\tilde x+\frac{\sigma_{t''}}{\sigma_t}(z_t-\alpha_t\tilde x)$; solving for $\tilde x$ gives the regression target

$$\tilde x=\frac{z_{t''}-(\sigma_{t''}/\sigma_t)\,z_t}{\alpha_{t''}-(\sigma_{t''}/\sigma_t)\,\alpha_t}.$$

Each round halves $N$ and the student becomes the next teacher; Salimans and Ho went from up to 8192 steps to 4, with FID 3.0 on CIFAR-10 at 4 steps. Near pure noise $\hat x=(z_t-\sigma_t\hat\epsilon)/\alpha_t$ amplifies errors, so they introduced $v=\alpha_t\epsilon-\sigma_tx$, with $\hat x=\alpha_tz_t-\sigma_t\hat v$.

### Consistency models

*EDM notation: $x_t=x+t\,z$, $t\in[\epsilon,T]$, $\epsilon=0.002$, $T=80$.* The consistency function maps any point of a probability-flow ODE trajectory to its endpoint, $f(x_t,t)=x_\epsilon$, so it is constant along trajectories. The boundary condition $f(x,\epsilon)=x$ is built in with $f_\theta=c_{\mathrm{skip}}(t)\,x+c_{\mathrm{out}}(t)\,F_\theta(x,t)$, $c_{\mathrm{skip}}(\epsilon)=1$, $c_{\mathrm{out}}(\epsilon)=0$.

**Consistency distillation (CD).** With $\hat x^\phi_{t_n}$ one teacher ODE step from $x_{t_{n+1}}$ to $t_n$ and $\theta^-$ an EMA of $\theta$,

$$\mathcal{L}_{\mathrm{CD}}=\mathbb{E}\Big[\lambda(t_n)\,d\big(f_\theta(x_{t_{n+1}},t_{n+1}),\,f_{\theta^-}(\hat x^\phi_{t_n},t_n)\big)\Big].$$

If the loss is zero, the error to the true consistency function is $O((\Delta t)^p)$ for an order-$p$ solver. Multistep sampling alternates $f_\theta$ with re-noising.

**Consistency training (CT).** Replace the teacher's score with the one-sample estimate $\nabla\log p_t(x_t)\approx-(x_t-x)/t^2$. An Euler step of $dx/dt=-t\nabla\log p_t$ from $x_{t_{n+1}}=x+t_{n+1}z$ gives

$$x_{t_{n+1}}+(t_n-t_{n+1})\frac{x_{t_{n+1}}-x}{t_{n+1}}=x+t_nz,$$

so CT compares $f_\theta(x+t_{n+1}z,t_{n+1})$ with $f_{\theta^-}(x+t_nz,t_n)$ for the same $(x,z)$. No teacher, but a bias that vanishes only as the discretization is refined, hence a curriculum over $N$. Improved CT drops the EMA target, uses pseudo-Huber instead of LPIPS and a lognormal noise schedule, and reaches one-step FID 2.51 on CIFAR-10 and 3.25 on ImageNet 64.

**Latent consistency models.** LCM runs CD in Stable Diffusion's latent space on the guided ODE, $\tilde\epsilon=(1+\omega)\epsilon(z,c,t)-\omega\epsilon(z,\varnothing,t)$, with $\omega\sim U[2,14]$ fed to the student. Adjacent timesteps barely differ, so it links $t_{n+k}$ and $t_n$ with $k=20$. Distillation took 32 A100 GPU hours, and FID at 768 px fell from 34.2 at one step to 16.3 at two and 13.5 at four. LCM-LoRA packages it as a LoRA that transfers to other fine-tunes.

**Continuous-time consistency models (sCM).** Let $\Delta t\to0$. With squared $\ell_2$ and $\theta^-=\mathrm{sg}(\theta)$, $f_\theta(x_t,t)-f_{\theta^-}(x_{t-\Delta t},t-\Delta t)\approx\Delta t\,\frac{df}{dt}$, so the gradient becomes proportional to

$$\nabla_\theta\,\mathbb{E}\Big[f_\theta^\top\,\mathrm{sg}\Big(\frac{df_{\theta^-}}{dt}\Big)\Big],\qquad\frac{df}{dt}=\partial_tf+(\partial_xf)\,\frac{dx_t}{dt},$$

a Jacobian-vector product, with $dx_t/dt$ from the teacher (sCD) or the conditional velocity (sCT). There is no discretization error, but training was unstable. Lu and Song fix it with TrigFlow ($x_t=\cos t\,x+\sin t\,z$ for data $x$), the identity time transformation, positional time embeddings, adaptive double normalization, tangent normalization and warmup, adaptive weighting, and a JVP-capable attention kernel; at 1.5B parameters they reach two-step FID 1.88 on ImageNet 512.

### Distribution matching and score distillation

**DMD** trains a one-step generator $x=G_\theta(z)$ to minimize $\mathrm{KL}(p_{\mathrm{fake}}\Vert p_{\mathrm{real}})$. The explicit $\theta$-dependence of $\log p_{\mathrm{fake}}$ drops out because $\mathbb{E}_{p_\theta}[\nabla_\theta\log p_\theta]=0$, leaving

$$\nabla_\theta\mathrm{KL}=\mathbb{E}_z\Big[\big(s_{\mathrm{fake}}(x)-s_{\mathrm{real}}(x)\big)^\top\frac{\partial G_\theta}{\partial\theta}\Big].$$

Both scores are evaluated on noised samples: $s_{\mathrm{real}}$ from the frozen teacher, $s_{\mathrm{fake}}$ from a second diffusion model initialized from the teacher and trained online on generator outputs. Reverse KL is mode-seeking, so DMD adds an LPIPS regression onto teacher ODE outputs for fixed noises; without it, CIFAR-10 FID rose from 2.66 to 5.58 with visible mode collapse. DMD reached FID 2.62 on ImageNet 64 and 11.49 on zero-shot COCO-30k.

**DMD2** drops the regression set, which is costly and caps the student at the teacher. It uses a two time-scale rule (5 fake-score updates per generator update), a GAN loss through a classifier on the fake-score network's bottleneck using real images, and multi-step students trained on their own intermediate outputs. It reports FID 1.28 on ImageNet 64, better than its EDM teacher, 8.35 on zero-shot COCO 2014, and a 4-step SDXL student.

**Score distillation.** SDS, VSD and DMD all push samples along the difference between a "fake" score and the teacher's score. SDS (DreamFusion) optimizes a renderer $x=g(\theta)$ with

$$\nabla_\theta\mathcal{L}_{\mathrm{SDS}}=\mathbb{E}_{t,\epsilon}\Big[w(t)\big(\hat\epsilon_\phi(x_t;y,t)-\epsilon\big)\frac{\partial x}{\partial\theta}\Big],$$

dropping the denoiser's Jacobian. Since $\epsilon=-\sigma_t\nabla\log\mathcal{N}(x_t;\alpha_tx,\sigma_t^2I)$, this is the DMD gradient with the fake distribution set to a point mass at the current render: mode-seeking, and in practice it needs CFG around 100, giving oversaturated, low-diversity results. VSD (ProlificDreamer) treats $\theta$ as a distribution and learns the fake score with a LoRA of the teacher on renders, working at CFG 7.5. DMD and Diff-Instruct apply this VSD-type gradient to a generator network.

### Adversarial distillation

**ADD** initializes the student from the teacher and trains it from noised real images at four student timesteps, with zero terminal SNR so the last is pure noise. The loss is $\mathcal{L}_{\mathrm{adv}}+\lambda\mathcal{L}_{\mathrm{distill}}$ ($\lambda=2.5$). The discriminator is a frozen DINOv2 ViT with trainable heads, text and image conditioning, hinge loss and an R1 penalty. The distillation term re-noises the student output, lets the teacher denoise it, and pulls the student toward that result; with squared $\ell_2$ and a suitable weighting it is the SDS gradient. SDXL-Turbo is ADD applied to SDXL and runs with guidance disabled; the paper notes slightly lower diversity.

**LADD** uses the teacher's own features on re-noised latents (noise from a logit-normal, $m=1$, $s=1$) as the discriminator, so everything stays in latent space at high resolution and multiple aspect ratios. It trains on teacher-generated images, which are better text-aligned than real captioned data, and with that data a distillation loss gave no benefit. SD3-Turbo matches its teacher's image quality in four unguided steps, with weaker prompt alignment as the reported limitation.

### Teacher-free few-step flows

**Shortcut models** (flow matching time) condition on a step size: $x_{t+d}=x_t+d\,s(x_t,t,d)$. At $d=0$ they train with flow matching; for larger steps they enforce

$$s(x_t,t,2d)=\tfrac12\,s(x_t,t,d)+\tfrac12\,s(x'_{t+d},t+d,d),\qquad x'_{t+d}=x_t+d\,s(x_t,t,d),$$

for $d$ in powers of two from $1/128$ to $1$, with about a quarter of each batch on these bootstrap targets. One network, one run, step count chosen at inference; with DiT-XL on ImageNet 256 they report FID 10.6 at one step and 3.8 at 128.

**MeanFlow** (SD3 time) learns the average velocity $u(z_t,r,t)=\frac{1}{t-r}\int_r^tv(z_\tau,\tau)\,d\tau$. Multiply by $t-r$ and differentiate in $t$:

$$u+(t-r)\frac{du}{dt}=v\quad\Longrightarrow\quad u(z_t,r,t)=v(z_t,t)-(t-r)\big(v\,\partial_zu+\partial_tu\big).$$

The network regresses onto the stop-gradient right-hand side, with $v$ replaced by the conditional velocity $\epsilon-x$ and the total derivative computed as a JVP with tangent $(v,0,1)$. $r=t$ recovers flow matching; one-step sampling is $z_0=z_1-u(z_1,0,1)$. Trained from scratch, it reports one-step FID 3.43 on ImageNet 256.

::: key
Few-step methods learn something more global than the instantaneous velocity: an endpoint map (consistency), a finite jump (progressive distillation, shortcut, MeanFlow), or a match of the whole output distribution (DMD, adversarial). The more global the target, the more the method relies on a teacher, a self-consistency bootstrap, or a discriminator.
:::

::: qa
**Consistency distillation versus consistency training: where does each target come from, and what does each cost?**

CD gets the earlier trajectory point from one teacher ODE step, so its target follows the teacher's ODE up to solver error and quality is bounded by the teacher. CT uses the same clean sample and noise at two noise levels, which equals one Euler step with the single-sample score estimate; it needs no teacher but has a bias that only vanishes as steps shrink, hence a refinement curriculum. Song and Dhariwal argue that CT's EMA target is unnecessary and remove it, together with LPIPS, which they replace with pseudo-Huber to avoid biasing FID.
:::

::: qa
**Why move consistency models to continuous time, and why was that hard?**

Discrete CMs compare outputs at timesteps $\Delta t$ apart, with the earlier point from a solver, which adds discretization error and sensitivity to how $\Delta t$ is annealed. As $\Delta t\to0$ the gradient becomes $\mathbb{E}[f_\theta^\top\mathrm{sg}(df/dt)]$, with the tangent from a JVP, so there is no solver and no $\Delta t$ schedule. The tangent involves derivatives of the network in time and input, and with standard designs it blows up: EDM's time transform diverges at the noise end, Fourier time embeddings and some normalization layers amplify it, and FP16 JVPs overflow. sCM's TrigFlow, identity time transformation, tangent normalization and adaptive weighting address these.
:::

::: qa
**Why does DMD need a second, "fake" diffusion model, and why does it tend to drop modes?**

The KL gradient needs the score of the generator's own distribution, and a one-step generator has no tractable density, so that score is estimated by a diffusion model trained on generator samples and kept current as the generator moves. Reverse KL is mode-seeking: it is small when samples stay in high-density regions of the data even if whole modes are missing, and the score difference carries little signal about the relative mass of well-separated modes. DMD counters this with a regression loss onto teacher outputs; DMD2 replaces it with a GAN loss on real images.
:::

::: qa
**Why does ADD combine an adversarial and a distillation loss, and why could LADD drop the distillation loss?**

The ADD paper gives one reason per term. The adversarial loss forces each one- or two-step output onto the manifold of real images, avoiding the blur and artifacts typical of other distillation methods. The distillation loss lets the student use the frozen teacher's knowledge and keep the strong compositionality of large diffusion models. LADD trains on teacher-generated images, which are better aligned with their prompts than real captioned data, and reports that with this synthetic data the distillation loss gave no advantage, plausibly because the data and a discriminator built on the teacher's own features already carry the teacher signal.
:::

::: qa
**Why can MeanFlow train without a teacher?**

Every term in its target $v-(t-r)(v\,\partial_zu+\partial_tu)$ is either the instantaneous velocity, for which the CFM conditional velocity $\epsilon-x$ is an unbiased stand-in, or a derivative of the network itself, computed by one JVP under stop-gradient. So the target comes from data and the current network, as in flow matching, rather than from simulating a pretrained ODE. The costs are a JVP per step and a noisier target; in the paper's best setting, 75% of samples still use $r=t$, i.e. plain flow matching.
:::

## Trade-offs

**Diversity.** Reverse-KL and adversarial objectives are mode-seeking: DMD without regression collapsed modes, ADD reports slightly lower diversity, and the sCM paper reports that VSD-style distillation lowers recall (diversity), while two-step sCD keeps precision and recall close to the teacher's. Regression and consistency-distillation objectives inherit the teacher's coverage.

**Quality versus steps.** Flow and diffusion ODEs improve steadily with NFE. Consistency models gain most from one to two steps and then flatten or slightly worsen (LCM: FID 13.5 at four steps, 15.0 at eight), which motivated trajectory models such as CTM.

**Teacher dependence.** Progressive distillation, consistency distillation, reflow and regression-based DMD are bounded by the teacher; real data through a discriminator can lift the student to or past the teacher (DMD2 on ImageNet 64 FID; ADD-XL in human preference against 50-step SDXL). iCT, sCT, shortcut models and MeanFlow need no teacher.

**Guidance baked in.** Most distilled text-to-image students fix a guidance scale during distillation (DMD2 used 8 for its SDXL teacher; LADD trains on images generated at a fixed guidance). LCM and FLUX.1 [dev] take the scale as an input, and SDXL-Turbo disables guidance and negative prompts.

::: qa
**Your one-step distilled model has lost diversity. How do you diagnose it, and what do you change?**

Fix the prompt and vary the seed: if layouts and identities repeat, it is mode collapse rather than a quality problem, and recall or per-prompt diversity metrics will confirm it. Usual causes are a reverse-KL or GAN-dominated objective and a high baked-in guidance scale. Remedies are a mode-covering term (regression onto teacher ODE pairs, as in DMD), real data in the discriminator, lower or input-conditioned distillation guidance, and two to four steps instead of one. If diversity matters more than latency, a consistency-distilled model or a few-step solver on the teacher keeps the teacher's coverage.
:::

::: qa
**What do you lose when guidance is baked into a distilled model?**

Guidance sets the trade-off between prompt adherence and diversity, and distilling at one scale freezes it, including any oversaturation. Negative prompts work by replacing CFG's unconditional branch, which a single-pass model no longer has unless you run CFG on top at twice the cost. Conditioning the student on the scale, as LCM and FLUX.1 [dev] do, restores part of the control within the range seen during distillation. LoRAs and adapters trained on the base model may also behave differently on the guidance-distilled student.
:::

## Rapid-fire review

::: qa
**What is $\sigma_{\min}$ in Lipman's OT path for?**

It keeps $p_1(x\mid x_1)$ a proper narrow Gaussian. With $\sigma_{\min}\to0$ the target becomes the rectified-flow target $x_1-\epsilon$.
:::

::: qa
**Does flow matching need a Gaussian source?**

No. CFM only needs samples from source and target, which allows data-to-data transport. The equivalence with diffusion assumes a Gaussian source and independent coupling.
:::

::: qa
**What do stochastic interpolants add beyond flow matching?**

A learned score alongside the velocity, which gives a family of SDEs with the same marginals for any diffusion coefficient $\varepsilon(t)\ge0$; $\varepsilon=0$ is the ODE.
:::

::: qa
**What is a 2-rectified flow?**

A rectified flow retrained on (noise, sample) pairs generated by the 1-rectified flow. Its trajectories are much straighter, so one Euler step already gives reasonable samples.
:::

::: qa
**When does minibatch OT coupling stop helping?**

In high dimension a small batch approximates the true OT plan poorly, and in conditional generation only pairs with the same condition can be matched.
:::

::: qa
**Is DDIM an Euler method?**

Yes, on the rescaled state $z/(a_s+b_s)$, which follows the rectified-flow path with the same SNR. In the original coordinates it is a first-order exponential integrator.
:::

::: qa
**Does the resolution shift require retraining?**

It is a change of time grid, so it can be applied at sampling alone; FLUX's public code applies it at sampling (how FLUX was trained is not published). SD3 applies it at training too, so the network sees enough high-noise samples at high resolution.
:::

::: qa
**How is the consistency boundary condition enforced?**

By construction: $f_\theta=c_{\mathrm{skip}}(t)\,x+c_{\mathrm{out}}(t)\,F_\theta$ with $c_{\mathrm{skip}}(\epsilon)=1$ and $c_{\mathrm{out}}(\epsilon)=0$, so $f_\theta(x,\epsilon)=x$ for any network.
:::

::: qa
**What is the two time-scale update rule in DMD2?**

Several fake-score updates (5 on ImageNet and SDXL) per generator update, so the fake score tracks the generator's current distribution and the score difference stays accurate.
:::

::: qa
**What does LCM's skipping step do?**

It enforces consistency between $t_{n+k}$ and $t_n$ ($k=20$) instead of adjacent steps, which barely differ; a much larger $k$ increases the teacher solver's error.
:::

::: qa
**Why does MeanFlow need a Jacobian-vector product?**

Its target contains the total derivative $v\,\partial_zu+\partial_tu$ of the network's own output, which one forward-mode JVP with tangent $(v,0,1)$ computes.
:::

## References

1. Chen, Rubanova, Bettencourt, Duvenaud. Neural Ordinary Differential Equations, 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
2. Grathwohl et al. FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models, 2018. [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)
3. Lipman, Chen, Ben-Hamu, Nickel, Le. Flow Matching for Generative Modeling, 2022. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
4. Liu, Gong, Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, 2022. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
5. Albergo, Vanden-Eijnden. Building Normalizing Flows with Stochastic Interpolants, 2022. [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)
6. Albergo, Boffi, Vanden-Eijnden. Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, 2023. [arXiv:2303.08797](https://arxiv.org/abs/2303.08797)
7. Lipman et al. Flow Matching Guide and Code, 2024. [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)
8. Tong et al. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport, 2023. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
9. Pooladian et al. Multisample Flow Matching: Straightening Flows with Minibatch Couplings, 2023. [arXiv:2304.14772](https://arxiv.org/abs/2304.14772)
10. Lee, Lin, Fanti. Improving the Training of Rectified Flows, 2024. [arXiv:2405.20320](https://arxiv.org/abs/2405.20320)
11. Liu et al. InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, 2023. [arXiv:2309.06380](https://arxiv.org/abs/2309.06380)
12. Esser et al. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
13. Hoogeboom, Heek, Salimans. Simple Diffusion: End-to-End Diffusion for High Resolution Images, 2023. [arXiv:2301.11093](https://arxiv.org/abs/2301.11093)
14. Kingma, Gao. Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation, 2023. [arXiv:2303.00848](https://arxiv.org/abs/2303.00848)
15. Gao, Hoogeboom, Heek, De Bortoli, Murphy, Salimans. Diffusion Meets Flow Matching: Two Sides of the Same Coin (blog), 2024. [diffusionflow.github.io](https://diffusionflow.github.io/)
16. Lin, Liu, Li, Yang. Common Diffusion Noise Schedules and Sample Steps are Flawed, 2023. [arXiv:2305.08891](https://arxiv.org/abs/2305.08891)
17. Song, Meng, Ermon. Denoising Diffusion Implicit Models, 2020. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
18. Karras, Aittala, Aila, Laine. Elucidating the Design Space of Diffusion-Based Generative Models, 2022. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
19. Lu et al. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps, 2022. [arXiv:2206.00927](https://arxiv.org/abs/2206.00927)
20. Lu et al. DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models, 2022. [arXiv:2211.01095](https://arxiv.org/abs/2211.01095)
21. Salimans, Ho. Progressive Distillation for Fast Sampling of Diffusion Models, 2022. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
22. Song, Dhariwal, Chen, Sutskever. Consistency Models, 2023. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
23. Song, Dhariwal. Improved Techniques for Training Consistency Models, 2023. [arXiv:2310.14189](https://arxiv.org/abs/2310.14189)
24. Luo et al. Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference, 2023. [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)
25. Luo et al. LCM-LoRA: A Universal Stable-Diffusion Acceleration Module, 2023. [arXiv:2311.05556](https://arxiv.org/abs/2311.05556)
26. Kim et al. Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion, 2023. [arXiv:2310.02279](https://arxiv.org/abs/2310.02279)
27. Lu, Song. Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models, 2024. [arXiv:2410.11081](https://arxiv.org/abs/2410.11081)
28. Poole, Jain, Barron, Mildenhall. DreamFusion: Text-to-3D using 2D Diffusion, 2022. [arXiv:2209.14988](https://arxiv.org/abs/2209.14988)
29. Wang et al. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation, 2023. [arXiv:2305.16213](https://arxiv.org/abs/2305.16213)
30. Luo et al. Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models, 2023. [arXiv:2305.18455](https://arxiv.org/abs/2305.18455)
31. Yin et al. One-step Diffusion with Distribution Matching Distillation, 2023. [arXiv:2311.18828](https://arxiv.org/abs/2311.18828)
32. Yin et al. Improved Distribution Matching Distillation for Fast Image Synthesis, 2024. [arXiv:2405.14867](https://arxiv.org/abs/2405.14867)
33. Sauer, Lorenz, Blattmann, Rombach. Adversarial Diffusion Distillation, 2023. [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)
34. Sauer et al. Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation, 2024. [arXiv:2403.12015](https://arxiv.org/abs/2403.12015)
35. Frans, Hafner, Levine, Abbeel. One Step Diffusion via Shortcut Models, 2024. [arXiv:2410.12557](https://arxiv.org/abs/2410.12557)
36. Geng, Deng, Bai, Kolter, He. Mean Flows for One-step Generative Modeling, 2025. [arXiv:2505.13447](https://arxiv.org/abs/2505.13447)
37. Black Forest Labs. FLUX.1 [dev] model card, [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev); reference code, [github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)
38. Stability AI. SDXL-Turbo model card. [huggingface.co/stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)
39. Interview report (Taobao/Tmall Group AIGC image algorithm, first round), Nowcoder, 2025. [nowcoder.com/discuss/730099920601907200](https://www.nowcoder.com/discuss/730099920601907200)
40. Interview report (Xiaomi AIGC algorithm engineer, first round), Nowcoder, 2025. [nowcoder.com/feed/main/detail/ae505961cae04205abc9448399a1da64](https://www.nowcoder.com/feed/main/detail/ae505961cae04205abc9448399a1da64)
41. Interview report (vision algorithm intern, image generation team), Nowcoder, 2024. [nowcoder.com/feed/main/detail/2b94140c0cc54718b4cd1c8ba02d35a2](https://www.nowcoder.com/feed/main/detail/2b94140c0cc54718b4cd1c8ba02d35a2)
