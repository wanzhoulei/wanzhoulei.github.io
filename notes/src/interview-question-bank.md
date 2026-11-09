---
title: "Research Interview Question Bank"
subtitle: "Questions reported in research-scientist-intern interviews for generative modeling and computer vision, across companies, with short answers."
order: 0
updated: 2026-11-08
---

This page collects questions that candidates reported from research-scientist-intern interviews in generative modeling and computer vision, with a short model answer for each. It was compiled from public interview reports on Glassdoor, Blind, Reddit, 1point3acres, Nowcoder, Zhihu, Jointaro, PracHub and similar sites, mostly posted between 2019 and 2026. Many reports were visible only as search-engine snippets or login-wall summaries, so dates and roles are approximate, and every question below is paraphrased. Chinese reports usually come from "algorithm intern" roles on AIGC, multimodal and vision teams rather than from programs titled research intern; some English reports are from adjacent roles (ML intern, applied scientist intern, a few full-time research roles), and the badges say so.

**How to read the badges.** A question with an `asked` badge appeared in at least one report; the badge names the companies and, when the theme recurred, the approximate number of independent reports. A badge means the question has been asked somewhere, not that a company asks it every time. Questions without a badge were added because they test the same material; several of them come from curated Chinese AIGC question lists, which are written by bloggers and are weaker evidence than first-hand reports. The numbers in square brackets point to the [Sources](#sources) list at the end.

## How the interviews are structured

| Company / program | Rounds | Coding | Research talk / deep dive | Fundamentals | Notes |
|---|---|---|---|---|---|
| Google Research (research intern, student researcher) | 2 back-to-back video rounds, 45–60 min | None in most reports since about 2019; one undated snippet describes an engineer round on technical skills | Self-intro, then one paper in depth; CV items and open research questions | ML basics from your own sub-area: contrastive and InfoNCE losses, normalization layers, CNN parameter counts | Interviewers are research scientists from a nearby area, not the host. Passing puts you in a pool for host matching, which can take months [1–10] |
| Google DeepMind | Student researcher (2025–26): two ~30-min rounds with a manager. Research intern (2021–23): screen, then a technical quiz | Student researcher: none. Older intern loops: reading code for a convolution | Past paper plus open-ended background; which team papers interest you | Linear algebra, calculus, probability, big-O, ML | The two descriptions come from different programs and years, so neither is a reliable guide to the other. Full-time research roles are roughly half coding and half research, per the official blog [11–15] |
| Meta (FAIR, GenAI, Reality Labs) | 2–3 rounds of ~45 min: one research, one or two coding | Yes in most reports: LeetCode medium to hard, or ML coding (k-means, 2D convolution, pseudo-PyTorch), depending on the interviewer | Your publications, then a brainstorm on a research problem. Variants: pick one of three papers sent ahead; a 60-min imaginary project | Attention, CV basics, how hyperparameters scale with batch size | Reports conflict for Reality Labs: a 2022 Blind post had no LeetCode, while other Reality Labs reports had a separate coding interview. Decisions take one to two weeks or more [16–32] |
| NVIDIA Research | No fixed process; often 2–3 interviews with research scientists from different teams | Research teams: often none; one 2025 research-intern loop mixed in DL and DSA coding. DL and ML intern roles asked for conv2d, attention or DSA | A different paper of yours per interviewer, future work, and a distinct proposal for each interviewer | Basic ML; diffusion basics in DL-intern reports when the resume listed diffusion (two posts, possibly by one person) | A 2026 intern screen asked for a ~20-min structured walkthrough of your project [33–42] |
| Adobe Research | 1–2 conversational rounds, often with the would-be mentor | None in many reports. Exceptions: bilinear interpolation, an MoE layer, implementing a paper's core function | Overview of past work; ideas against a research agenda sent in advance; reading a team paper | Light; image processing for imaging teams | The planned project is often described during the interview [43–47] |
| Apple (MLR, AIML, other ML teams) | Screen, a 30-min talk or 1-h seminar, then 3–4 one-hour interviews | Easy to medium LeetCode in most reports, none in one; one 2025 loop had a probability round | Formal talk is common; interviewers perturb your project to test depth | DL fundamentals, BERT (NLP team), linear regression | Teams differ a lot [48–54] |
| Microsoft (MSR, applied scientist intern) | 2–4 rounds | One LeetCode-style round, low bar in a 2022 report; an easy problem plus metric pseudocode in 2026 | Project depth, ablation design, overlap with the team's project | Metrics, regularization, optimization, L1 vs L2, divergences | Teams select for fit with an ongoing project; no behavioral round in the 2026 report [26, 55–57] |
| Amazon (applied scientist intern) | Screen, coding, ML breadth and depth, ML design, Leadership Principles | Yes (DSA) | Deep questions on your own research | CV metrics (mAP, IoU); LLM training topics in 2026 | Behavioral (Leadership Principles) questions in every loop [58–61] |
| Qualcomm AI Research | Screen, paper review, coding (2024); one research plus one technical ML interview, 45 min each (2026) | ML, DL and generative-AI coding plus theory | Read a given paper and discuss it | Activations, transformers, quantization error, YOLO | [64, 65] |
| ByteDance / TikTok | US: two technical rounds. China: 2–3 technical rounds plus a manager round | China: nearly every technical round (LeetCode or PyTorch attention). US: a Transformer from scratch | Each round opens with a paper deep dive | Attention, BatchNorm, cross-entropy, VAE loss, KV cache, mixed precision | A 2026 third round was derivations from minute seven. No first-hand report for a Seed vision or video intern [63, 82, 84–87] |
| Alibaba (Taotian, Tongyi, Ant, Alibaba Cloud) | 2–3 technical rounds plus HR | One LeetCode problem in most rounds; open systems puzzles | About 30 min of project questions; Ant derived the paper's formulas | SD and DDPM, GAN training, why SD uses a VAE | A senior-manager round asked whether the work can ship and what value it brings [80, 91, 92, 101, 103] |
| Tencent (Youtu, multimodal, TME, Hunyuan) | 2–4 rounds | Youtu: max subarray. TME: every round. Multimodal (2026): none listed | 80-min paper grilling at Youtu; contribution in "three sentences" | DDPM vs DDIM, ControlNet, IP-Adapter, LoRA, CLIP, SAM | The Youtu interviewer advised bringing slides of your core work to later rounds [83, 97, 98, 122] |
| Kuaishou / Kling | 1–2 rounds, under an hour each | One LeetCode problem | Resume-based; a diffusion paper probed from motivation to results | Multi-head attention; VAE vs GAN vs diffusion vs flow; RL (GRPO) | Reports cover understanding and RL roles, not the Kling generation team [81, 99, 118, 119, 121] |
| Baidu | 2 rounds, ~1 h each | numpy convolution, cross-attention, LeetCode | Paper-centred | SD, DDPM vs DDIM, ControlNet, DreamBooth, LoRA, FLUX vs Wan | [89, 90, 106, 120] |
| Other Chinese companies (Meituan, Xiaohongshu, Xiaomi, SenseTime, Megvii, vivo, bigo) | 1–4 rounds | LeetCode in most; Megvii's SR team asked for numpy interpolation and a choice of im2col or deformable convolution | Paper walkthrough in most rounds | SD training, CLIP, BLIP-2, PSNR and SSIM, RAW noise | Business scenarios are common [93–96, 100, 105–107, 110, 111, 125] |
| Snap, Samsung Research America, Netflix | 1–3 rounds | Light: SGD in Python (Samsung); LeetCode medium (Netflix) | Research background; Netflix research internships may include a presentation | CV and ML basics | Thin evidence [62, 66–68] |
| OpenAI, Anthropic, Runway, Luma (full-time roles; no intern reports) | 3–6 rounds | Heavy: practical coding, PyTorch debugging, LeetCode | Research background; a GenAI project you led | KL divergence, attention variants, compute and memory estimates | Shown for contrast with intern loops [69–76] |

**Non-coding research loops.** The programs with mostly non-coding reports are Google Research's research-intern loop, the Google DeepMind student researcher loop in 2025–2026, most Adobe Research intern loops, and many NVIDIA Research teams. For Google Research, Chinese posters report no coding for research interns since about 2019 [8], English snippets describe two rounds on "general ML concepts" and your own research [3], and a 2026 Cloud research-intern report describes two video interviews on research and interests [5]; one undated Glassdoor snippet, however, describes a research round plus an engineer round on technical skills [3, 4]. The DeepMind student researcher reports say no coding or system design [12]; older DeepMind research-intern loops (2021–2023), a different program, did include a technical quiz and a code-reading question on a convolution [11]. NVIDIA reports from Zürich (2022) and January 2026 describe only research interviews [33, 34], while a Blind thread notes NVIDIA has no fixed process and puts little weight on LeetCode [40]. A 2022 Blind post from a PhD student who interned at Google, Meta Reality Labs Research and Microsoft said research internships "usually had no LeetCode rounds", with a single low-bar coding round at Microsoft [26]. "Non-coding" does not mean non-technical: these loops still test fundamentals from your own area, sometimes with derivations.

**Loops with coding.** Meta's research-intern loop usually pairs one research round with one or two coding rounds, though the 2022 Blind post above had none at Reality Labs Research [26]; a commenter in one 2023 thread advised "LC medium & hard regardless of career level" [21], and the "AI coding" round can be PyTorch or LeetCode depending on the interviewer [24]. Apple, Amazon and Microsoft add LeetCode-style rounds of easy to medium difficulty next to research and ML rounds [49, 52, 56, 58]. Qualcomm has a dedicated ML and generative-AI coding round [64]. NVIDIA deep-learning and ML intern roles, as distinct from NVIDIA Research teams, asked for conv2d, an attention module or algorithm problems [38, 39], and one 2025 research-intern loop also mixed in deep-learning and DSA coding [35]. At Chinese companies, coding is standard even for research-flavoured roles: usually one LeetCode-medium problem per round, and generative and multimodal teams increasingly ask for PyTorch attention or cross-attention by hand [80, 85, 90]. Full-time roles at OpenAI, Anthropic and generative-video startups are the most coding-heavy, including PyTorch debugging rounds [25, 69, 74, 76].

**What every loop shares.** Nearly every report spends a large share of the time on your own research. One Chinese poster with offers from several large companies described interviews as built on a roughly one-hour project deep dive, with LeetCode or fundamentals added if time remains [80]. Formal talks are less common for interns than for full-time roles: Apple (a 30-minute talk or a one-hour seminar) [48, 49] and some Netflix research internships [68] ask for one, as do Meta and DeepMind full-time research loops and, according to a prep guide, some Microsoft full-time roles [14, 15, 22, 30, 57]. Elsewhere the "talk" is an oral walkthrough in the first 10–20 minutes, and one Google student-researcher candidate was allowed to use slides [4]. After the interviews, matching is its own stage: at Google, passing puts you in a pool from which hosts pick, project search can run from January to April, and contacting teams proactively helped [7, 10]; at NVIDIA and Adobe the interviewer is often the would-be mentor [43]; at MSR teams select for overlap with an ongoing project [55].

## What interviewers look for

### Evaluation criteria

1. **Ownership and depth of your own work.** This is the one constant across companies. Interviewers ask why each design choice, what the limitations are and what you would do next, and some perturb the setup to see whether you understand the mechanism [51]. A Google candidate who had only partial involvement in the paper they presented did poorly [8]. A Tencent Youtu interviewer wanted the contribution in "three sentences" [83]; a ByteDance poster summarized rounds one and two as "what did you do" and round three as "why should I believe you are right" [86]; a TME candidate was questioned down to the experimental numbers [98]; an Ant interviewer had the candidate derive the paper's formulas [101].
2. **Research taste and the ability to generate ideas.** Most research loops include a "propose something" segment: a brainstorm after the publication questions at Meta [18], a distinct proposal for each interviewer at NVIDIA [34], ideas against an agenda sent in advance at Adobe [44], "what would you do here" at DeepMind [11], and a project you would bring to Google [8]. Designing experiments to test a hypothesis also came up [1].
3. **Fundamentals from your own sub-area.** Google interviewers draw fundamentals from the area you work in [1–3]. Attention is the most frequent single topic across all reports. Chinese AIGC interviews add model-version questions and ask for the intent behind each design change, for example SDXL versus SD 1.5 [80, 88]. A 2023 multi-company summary noted detailed questions on model choice and on the statistical assumptions behind a method [61].
4. **Hands-on implementation.** Coding questions favour from-scratch building blocks (attention, convolution, k-means, an MoE layer, SGD) over full training loops; debugging rounds on PyTorch code are appearing at GenAI-heavy companies [25, 69, 76].
5. **Communication.** Explaining your work to people outside your sub-area is part of the job [1–3]; a Runway candidate whose interviewer came from infrastructure struggled to get a modeling explanation across [74].
6. **Product sense (mostly Chinese companies, later rounds).** Can it ship, in which scenario, with what value: this came up in Taotian's senior-manager round, Ant's HR round, Meituan's business scenarios and Tencent's 2026 multimodal rounds [80, 95, 97, 101].

::: key
The deep dive on your own work carries the most weight in almost every loop, and its follow-ups (why, why not X, what if, what next) are predictable. Prepare those answers with evidence, not just the talk.
:::

### Common failure modes

- Presenting work you did not fully own [8], or explaining it at length when the interviewer wanted the claim [83].
- Coding: a candidate with a strong publication record failed Meituan three times on coding [95]; others ran out of time on a follow-up [74] or misread a variant of a known problem [23].
- A weak baseline that the interviewer criticized, and HR feedback that industry knowledge was thin [101].
- Weak PyTorch fluency in ML-debugging rounds [25]; leaning on the AI assistant in Meta's 2026 full-time "Coding with AI" round, where the interviewer told the candidate to read the code himself [23].
- Role labels that mislead: several "AIGC" intern roles turned out to be LLM or RAG interviews [102–104]. Use the reverse Q&A to learn what the team actually does.
- Over-reading one fast rejection: at least one candidate felt the interview was a quota-filling exercise [103].

::: pitfall
Treating the research round as the easy part. It is tempting to spend most preparation time on LeetCode and an evening on the talk, then fail to justify a design choice or an assumption under follow-up questions. The reports, and the advice of a Meta CV researcher [78], point the other way for research roles, except at companies where coding is usually a hard gate (Meta, most Chinese companies).
:::

### Tips from the reports

- Prepare a short slide deck of your core work and use it to steer the conversation; both a Chinese poster with several offers and the Tencent Youtu interviewer advised this [80, 83]. Keep a talk to about 20 minutes: problem, current approaches and their gaps, your solution, why it matters, then details and results [77]. A 2026 NVIDIA intern screen asked for motivation, pipeline, results and takeaways in about 20 minutes [41].
- Read the interviewers' and the team's recent papers, blog posts and talks [1–3, 13].
- Prepare two or three scoped project ideas tied to the host's recent work [34, 44].
- Know your sub-area: read widely, know the datasets and state-of-the-art numbers, and be able to write your pipeline and loss by hand [112].
- Be honest about what you do not know, and be succinct [1–3].
- A Meta CV researcher advised spending more preparation on research and ML fundamentals than on LeetCode [78]; for Meta itself and for most Chinese companies, coding is still a gate [21, 95].
- Referrals helped candidates get interviews; after that, performance and publications decide [26, 77].
- In the reverse Q&A, ask about the team's actual work, what answers they expected, and the number of remaining rounds [80, 83].

## Deep dive on your own research

These templates recur in nearly every loop. The answers describe how to answer well; the content has to come from your own work.

::: {.qa asked="Nearly every report: Google, Meta, NVIDIA, ByteDance (20+)"}
**Walk me through your paper: problem, motivation, method, results, and which part was yours.**

Open with one sentence on the problem and why it matters, then the specific gap in existing methods, then your key idea in one sentence before any detail. Give the method at the level of its main equation or diagram, then the strongest result as a number against the strongest baseline, then one limitation. Say early and plainly which parts you did yourself. Prepare a two-minute and a ten-minute version, and let the interviewer pull detail rather than narrating the paper in order. Pick one table and one figure you can describe from memory, with numbers.
:::

::: {.qa asked="Tencent Youtu CV intern, Nowcoder 2024"}
**State your contribution in three sentences.**

Sentence one: the problem and the gap. Sentence two: the mechanism that closes the gap, described by what it does rather than by its name. Sentence three: the evidence, either one number against the strongest baseline or the main theoretical result. Rehearse this until it is natural; it is also the core of your abstract. If an interviewer cuts you off, they usually want the claim, not the context.
:::

::: {.qa asked="Google RI (Reddit 2020), Alibaba, Ant (Nowcoder)"}
**Why did you choose this approach? I think method X would work better.**

Treat it as a technical question, not an attack. Restate what X assumes, then give the concrete reason you did not use it: an assumption your setting violates, a cost, a failure mode, or an experiment where X did worse, with the number. A useful shape is "X assumes A; in our setting A fails because B; we measured C." If you never tried X, say so, predict what would happen and describe the cheapest experiment that would tell. Concede the cases where X is better; that shows you mapped the design space.
:::

::: {.qa asked="Google RI (Blind 2019), Ant 2024, ByteDance 2026"}
**Justify the assumptions in your paper, or derive its main result on the spot.**

Know where every equation comes from: which assumption produces each step and where an approximation enters. Be able to rederive your loss or main bound on a whiteboard in about five minutes. For each assumption, know when it fails in practice and what breaks (for example, Gaussian noise assumed where real sensor noise is signal-dependent). If a proof has a gap or a step you took from another paper, say so before the interviewer finds it. The question checks whether you understand the method or only ran it.
:::

::: {.qa asked="NVIDIA phone screen; Google RI (snippet)"}
**What are the limitations of your work, and what would you do differently?**

Give two or three real limitations in order of importance, each with its cause and evidence: a failure case, a setting where the gain disappears, a compute cost. Separate limitations of the idea from limitations of execution (data, compute, time). For "differently", name one concrete change and what you learned that makes you believe in it now. Avoid generic answers like "more data" and avoid self-deprecation; a precise limitation answer reads as maturity.
:::

::: qa
**Show me where your method fails.**

Bring two or three failure examples you understand, each with a hypothesis for why it fails (outside the training distribution, an unmodeled degradation, a conditioning signal that is too weak). Explain how you would confirm the hypothesis: which probe or ablation. Failures that you have counted and grouped into categories are much more convincing than a single anecdote, and much more convincing than only showing successes.
:::

::: {.qa asked="Microsoft AS PhD intern, PracHub 2026"}
**Design an ablation study for your project.**

List the claims first, then for each claim the component responsible for it. Remove or replace one component at a time while holding data, compute and tuning budget fixed, and include a row where the baseline receives the same generic tricks you used. When two components might be redundant, run the two-by-two grid. Report variance over seeds, and use a metric that measures the claimed effect, not only the headline number. Finish by saying which ablation you would run first under a limited budget and why.
:::

::: {.qa asked="Apple ML intern, Jointaro 2024"}
**Suppose the setting changes: a new constraint, different data, or a tweak to your method. What happens?**

The interviewer is testing whether you understand the mechanism. Identify which assumption the change touches, predict the direction and rough size of the effect, and propose the cheapest experiment to check. "I don't know, but here is how I would find out" is a good answer when it is followed by a sound experiment. Prepare by listing the five most plausible perturbations of your own work (resolution, noise level, domain, compute budget, backbone) and thinking each through.
:::

::: {.qa asked="Google RI, NVIDIA RI, MSR intern (1p3a, Glassdoor)"}
**How would you extend your work to a new problem, and what is the future work?**

Give one near extension (same mechanism, new task) and one more ambitious one. For each, say what transfers, what new difficulty appears, and the first experiment with its success criterion. Connect to the team's interests if you know them. One concrete, well-reasoned direction beats a list of vague ones.
:::

::: {.qa asked="ByteDance LLM algorithm round 3, Nowcoder 2026"}
**How do you know the improvement came from your change and not from something else?**

Describe the controls: same data, same steps, same tuning budget, several seeds with error bars, and an ablation that removes only your change. Add a dose-response check: the gain should be largest on the subset where your mechanism should matter most. Name the confounders you ruled out, such as an under-tuned baseline or metric noise (FID on a few thousand samples can move by more than many reported gains). If you cannot fully separate the effect, say what remains uncertain.
:::

::: {.qa asked="Ant algorithm intern, Nowcoder 2024"}
**Your baseline looks weak. How would you make the comparison fair?**

Acknowledge the concern and explain how the baseline was obtained: official code or reported numbers, and how much tuning it received compared with your method. Propose the fix: reproduce the strongest recent baseline with an equal tuning budget and report both. If the data limited the baseline, say how you would collect or clean better data. Defensiveness tends to do more damage than the weak baseline itself.
:::

::: {.qa asked="Alibaba Taotian, Tencent, SenseTime (Nowcoder)"}
**Can your work ship? In which scenario, and what value would it bring?**

Name a concrete product scenario, the user-facing metric it would move, and what stands in the way: latency, memory, robustness, failure modes visible to users, data or licensing. Estimate the gap honestly and give a path to close it (a smaller backbone, distillation, a narrower first use case). This comes up mostly in manager rounds at Chinese companies, where an answer about value and risk matters more than one about novelty.
:::

::: qa
**Explain your research to someone outside your area.**

Start from a concrete input and output, then one analogy, then why current tools fail, what you changed, and one result stated in terms the listener cares about. Drop the notation and check understanding as you go. Rounds are sometimes run by an interviewer from a different sub-area, and an explanation that only works for specialists can sink an otherwise strong round.
:::

::: {.qa asked="Google, Meta, Adobe, Qualcomm (5+ reports)"}
**Read this paper (or one we sent in advance) and discuss it with me. Could its technique apply to other tasks?**

Organize the discussion as claim, mechanism, evidence, and the gap between them. Find the key equation or figure, state what is new relative to prior work, and check whether the experiments support the claim (baselines, ablations, choice of metrics). Identify an assumption that limits the method and propose a follow-up experiment. For transfer to other tasks, reduce the technique to what it assumes about the data and what it computes, then name one task that shares that structure and one where an assumption breaks. With an unseen paper, read the abstract, Figure 1, the method equations and the main table first, and summarize aloud before going into detail.
:::

::: {.qa asked="Apple, Netflix, NVIDIA (talk or walkthrough)"}
**Give a 20–30 minute talk on your research.**

Use the order problem, why it matters, current approaches and their shortfall, your idea, why it works (one key figure or equation), results, limitations and next steps. Put the takeaway of each slide in its title, plan roughly one slide per minute, and leave time for questions. Keep backup slides for the questions you expect (derivations, ablations, failure cases). For interns, the questions after the talk usually matter more than the slides themselves.
:::

## ML and deep-learning fundamentals

::: {.qa asked="ByteDance, Kling, TME, Meta, NVIDIA (10+ reports)"}
**How does scaled dot-product attention work, and why divide by $\sqrt{d_k}$?**

$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$: each output row is a convex combination of value vectors, weighted by how well its query matches each key. If the entries of $q$ and $k$ are independent with zero mean and unit variance, then $q^\top k$ has variance $d_k$. Without scaling, the logits grow with $d_k$, the softmax saturates toward one-hot, and its gradients become very small. Dividing by $\sqrt{d_k}$ keeps the logit variance near 1 at initialization.
:::

::: {.qa asked="ByteDance, Kling, TME, Amap (Nowcoder)"}
**Why use multiple heads, and where do Q, K and V come from in causal self-attention and in cross-attention?**

Each head projects into a subspace of dimension $d/h$ and can attend to different positions or relations; a single head has to average them into one distribution. The cost is about the same as one full-width head because the total width is unchanged; the heads are concatenated and mixed by an output projection. In causal self-attention, Q, K and V all come from the same sequence and scores for future positions are set to $-\infty$ before the softmax. In cross-attention, Q comes from the sequence being updated (decoder tokens, or image features in Stable Diffusion) and K and V come from the conditioning sequence (encoder output or text embeddings), so the output has the length of the query sequence.
:::

::: {.qa asked="ByteDance 2026, Alibaba Taotian, OpenAI applied RS"}
**What is the complexity of a transformer layer, which term dominates for long sequences, and how would you change attention for longer inputs?**

For sequence length $N$ and width $D$, the Q, K, V and output projections cost $4ND^2$ multiply-adds, a 4x-wide MLP costs $8ND^2$, and the attention scores plus the weighted sum cost $2N^2D$. So a layer is about $12ND^2 + 2N^2D$, and the quadratic term dominates once $N > 6D$; the attention matrix also needs $O(hN^2)$ memory unless it is computed in tiles. Options for long inputs: IO-aware exact attention (FlashAttention) cuts memory but not FLOPs; local, sliding-window or block-sparse patterns with a few global tokens; linear or low-rank approximations, usually at some quality cost; factorized attention (spatial then temporal for video); and positional-encoding scaling for longer context. Choose by whether the bottleneck is compute, activation memory or the KV cache.
:::

::: {.qa asked="ByteDance LLM algorithm, Nowcoder 2026"}
**What does a KV cache store, how large is it, and how do MQA and GQA change it?**

During autoregressive decoding the keys and values of past tokens do not change, so they are cached and each new step computes q, k and v only for the new token. This trades memory for compute: it removes recomputation of the prefix but stores $2 \times L \times h_{kv} \times d_h \times T \times B$ values ($L$ layers, $h_{kv}$ key-value heads, head dimension $d_h$, $T$ tokens, batch $B$). Multi-query attention shares a single K/V head across all query heads (cache divided by $h$); grouped-query attention uses $g$ groups (cache scaled by $g/h$). Since decoding is limited by memory bandwidth, a smaller cache speeds it up, at a small quality cost.
:::

::: {.qa asked="Alibaba Amap, TME, ByteDance (Nowcoder)"}
**Compare sinusoidal positional encodings with RoPE.**

Sinusoidal encodings add a fixed vector with frequencies $\omega_i = 10000^{-2i/d}$ to the input; the encoding at position $p+k$ is a rotation of the one at $p$, and there are no parameters, but position is absolute and mixed into the content once. RoPE rotates each two-dimensional pair of q and k by an angle $m\theta_i$ at position $m$, so $\langle R_m q, R_n k\rangle$ depends only on $n - m$: relative position enters every attention score without extra parameters. Longer contexts are reached by rescaling the frequencies (position interpolation, NTK-aware scaling). Vision models split the rotary channels across axes, for example 2D RoPE over height and width, or the temporal-height-width split of M-RoPE.
:::

::: {.qa asked="ByteDance, Alibaba, Ctrip, bigo (5 interviews, 2019–22)"}
**How does BatchNorm differ between training and inference? Why the learned scale and shift, and how many parameters does it have?**

In training, each channel is normalized with the mean and variance over $(N, H, W)$ of the current batch, and running averages are updated. At inference the running averages are used, so outputs are deterministic and independent of the batch. The learned $\gamma, \beta$ let the network undo or reshape the normalization so no representational power is lost. Parameters: $2C$ learned values plus $2C$ running statistics that are buffers, not trained. Statistics are per channel because channels carry different features with different scales while spatial positions share one filter. The batch noise gives a mild regularizing effect, but BatchNorm is not a substitute for regularization and degrades with small batches or a train-test shift.
:::

::: {.qa asked="Google intern, Glassdoor (role unclear)"}
**Compare BatchNorm, LayerNorm, InstanceNorm and GroupNorm.**

For an $(N, C, H, W)$ tensor they differ only in the axes averaged: BatchNorm over $(N, H, W)$ per channel; LayerNorm over $(C, H, W)$ per sample (over features per token in transformers); InstanceNorm over $(H, W)$ per sample and channel; GroupNorm over $(H, W)$ and a group of channels per sample. BatchNorm couples samples and depends on batch size; the others work with batch size one and in sequence models. InstanceNorm removes per-image contrast and style statistics, which is why style transfer uses it (AdaIN injects a style through its scale and shift). Diffusion U-Nets typically use GroupNorm, and transformers and DiTs use LayerNorm, with timestep-dependent scale and shift in DiT's adaLN.
:::

::: {.qa asked="ByteDance, iQIYI, aibee (5 interviews, 4 posts)"}
**Write softmax cross-entropy and its gradient. Why cross-entropy rather than MSE for classification, and what is focal loss?**

With $p = \mathrm{softmax}(z)$, $L = -\sum_k y_k \log p_k$ and $\partial L / \partial z = p - y$. The minus sign makes the negative log-likelihood non-negative. With MSE on softmax outputs, the gradient passes through the softmax Jacobian $\mathrm{diag}(p) - pp^\top$, which vanishes when the model is confidently wrong, so learning stalls exactly when it should be fastest; cross-entropy keeps the gradient $p - y$ and is the maximum-likelihood loss for a categorical model. Multi-label problems use an independent sigmoid with binary cross-entropy per label. Focal loss, $-\alpha_t (1 - p_t)^\gamma \log p_t$, down-weights easy examples so training focuses on hard ones; it was introduced for dense detection, where easy background anchors dominate, and $\gamma = 0$ recovers ($\alpha$-weighted) cross-entropy.
:::

::: {.qa asked="Microsoft AS, Baidu, Alibaba, ByteDance (4 reports)"}
**Compare L1 and L2, as losses and as regularizers. Where does Huber fit?**

As losses, L2 is the Gaussian maximum-likelihood loss and estimates the conditional mean; it punishes outliers heavily and its gradient shrinks near zero. L1 is the Laplace likelihood, estimates the conditional median, is robust to outliers and has a constant-magnitude gradient; many restoration networks train with L1 because it tends to give sharper results than L2, even when evaluated by PSNR. Huber is quadratic within $\delta$ and linear outside it, smooth near zero and robust far away. As regularizers, L2 (weight decay, a Gaussian prior) shrinks all weights smoothly, while L1 (a Laplace prior) sets many weights exactly to zero, because its subgradient does not vanish at zero; geometrically, the L1 ball has corners on the axes.
:::

::: {.qa asked="Qualcomm (Glassdoor), Horizon (Nowcoder 2022)"}
**List common activation functions with their pros and cons. Why do many ReLU units die during training?**

Sigmoid and tanh are smooth and bounded but saturate, which causes vanishing gradients, and sigmoid is not zero-centred. ReLU is cheap and does not saturate for positive inputs, but a unit whose input is negative for every example gets zero gradient. Leaky ReLU and PReLU keep a small slope for negative inputs. GELU and SiLU are smooth, slightly non-monotonic near zero, and standard in transformers and diffusion U-Nets. Units die when a large update (a high learning rate) or a strongly negative bias pushes the pre-activation below zero for all inputs; lower learning rates, careful initialization, normalization or leaky variants prevent it.
:::

::: {.qa asked="Tencent multimodal 2026, Horizon 2022 (Nowcoder)"}
**How should weights be initialized? Derive Kaiming initialization.**

The goal is to keep the variance of activations and gradients constant across layers. For $y = Wx$ with fan-in $n$ and zero-mean weights, $\mathrm{Var}(y_i) = n\,\mathrm{Var}(w)\,\mathbb{E}[x^2]$. If $x$ is the ReLU of a zero-mean symmetric pre-activation $u$, then $\mathbb{E}[x^2] = \tfrac{1}{2}\mathrm{Var}(u)$, so $\mathrm{Var}(w) = 2/n$ keeps the variance constant (He initialization); Xavier uses $2/(n_{\text{in}} + n_{\text{out}})$ for linear or tanh units. Initializing all weights to zero fails because of symmetry. Zero-initializing selected layers is a deliberate tool: the last layer of a residual branch, LoRA's $B$ matrix and ControlNet's zero convolutions all make a new path contribute nothing at first, so the block starts as the identity or the model starts as the pretrained one.
:::

::: {.qa asked="Meta research round, Glassdoor"}
**When you increase the batch size, how should you change the learning rate and other hyperparameters?**

For SGD, scaling the learning rate linearly with batch size, with a warmup, works up to a critical batch size; for Adam, square-root scaling is a common heuristic. Past the critical batch size, which relates to the gradient noise scale, larger batches save few optimization steps. With a fixed number of epochs or tokens, a larger batch means fewer steps, so define warmup and decay in samples or tokens rather than steps, and revisit weight decay and EMA decay, which act per step. Verify by checking that loss curves plotted against samples seen roughly overlap.
:::

::: {.qa asked="Amazon AS, Qualcomm, NVIDIA (3 reports)"}
**Explain the bias-variance trade-off and how you diagnose and fix overfitting.**

For squared error, expected error decomposes as $(\mathbb{E}\hat f - f)^2 + \mathrm{Var}(\hat f) + \sigma^2$: bias from a model too simple or too constrained, variance from sensitivity to the particular training set, and irreducible noise. Overfitting shows as falling training loss with rising validation loss. Fixes: more data or augmentation, weight decay, dropout, early stopping, a smaller model or fewer epochs, ensembling or weight EMA. Large networks can show double descent, so model size is not a monotone knob. In generative models, overfitting shows up as memorization, which you check by nearest-neighbour search of samples against the training set in a feature space.
:::

::: {.qa asked="ByteDance (2 reports), Tencent Hunyuan (Nowcoder)"}
**How does mixed-precision training work, and why loss scaling? What do gradient checkpointing and ZeRO save?**

Forward and backward run in FP16 or BF16 while an FP32 master copy of the weights and the optimizer states is kept for the update. FP16 has a narrow exponent range, so small gradients underflow to zero; multiplying the loss by a scale $S$ and unscaling the gradients before the update avoids that, and $S$ is lowered when infs or NaNs appear. BF16 has FP32's exponent range and usually needs no loss scaling. Gradient checkpointing stores only some activations and recomputes the rest during backward, cutting activation memory substantially for roughly one extra forward pass. ZeRO partitions optimizer states (stage 1), then gradients (stage 2), then parameters (stage 3) across data-parallel ranks, trading memory for communication.
:::

::: {.qa asked="Google RI, ByteDance, Huawei, SenseTime (4+ reports)"}
**How many parameters and FLOPs does a convolutional layer have? What do 1x1 and depthwise-separable convolutions save?**

A $k \times k$ convolution from $C_{\text{in}}$ to $C_{\text{out}}$ channels has $k^2 C_{\text{in}} C_{\text{out}}$ weights plus $C_{\text{out}}$ biases and costs $k^2 C_{\text{in}} C_{\text{out}} H_{\text{out}} W_{\text{out}}$ multiply-adds. A 1x1 convolution mixes channels at each pixel and is the cheap way to change width (bottlenecks). A depthwise-separable convolution is a per-channel $k \times k$ filter ($k^2 C_{\text{in}}$ weights) followed by a 1x1 pointwise convolution ($C_{\text{in}} C_{\text{out}}$); relative to a standard convolution the cost ratio is $1/C_{\text{out}} + 1/k^2$, roughly 8–9 times cheaper for $3 \times 3$. MobileNets are built on this; on GPUs the saving in wall-clock time is smaller because depthwise convolutions are memory-bound.
:::

::: {.qa asked="Apple ML intern 2024, Meta MLE 2026"}
**Derive linear regression, and explain the statistical basis of the MSE loss.**

Minimize $\|y - Xw\|^2$; setting the gradient $-2X^\top(y - Xw)$ to zero gives the normal equations $X^\top X w = X^\top y$, so $w = (X^\top X)^{-1} X^\top y$ when $X$ has full column rank. In practice you solve with QR or Cholesky rather than forming the inverse, or use gradient descent at scale; ridge regression adds $\lambda I$ to $X^\top X$. If $y = Xw + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, the log-likelihood is $-\|y - Xw\|^2 / (2\sigma^2)$ plus a constant, so minimizing MSE is maximum likelihood under Gaussian noise and estimates the conditional mean.
:::

Deeper: [Deep Learning Training](deep-learning-training.html) and [Vision Backbones](vision-backbones.html).

## Math and probability

::: {.qa asked="OpenAI RS, MSR intern, ByteDance (Glassdoor, Nowcoder)"}
**What is a divergence? Compute a KL divergence and relate KL to cross-entropy.**

A divergence $D(p \| q)$ is non-negative and zero only when $p = q$; it need not be symmetric or satisfy the triangle inequality. Examples: forward and reverse KL, Jensen-Shannon, total variation and the other f-divergences; the Wasserstein distance, and MMD with a characteristic kernel, are true metrics. $\mathrm{KL}(p \| q) = \mathbb{E}_p[\log p/q]$, and $H(p, q) = H(p) + \mathrm{KL}(p \| q)$, so minimizing cross-entropy in $q$ with the data distribution $p$ fixed is minimizing KL, which is maximum likelihood. For Gaussians, $\mathrm{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$. Fitting $q$ with forward KL is mode-covering; with reverse KL it is mode-seeking.
:::

::: {.qa asked="ByteDance algorithm intern, Nowcoder 2020"}
**A disease has 1% prevalence and a test is 99% accurate. If you test positive, how likely are you to be ill?**

Take sensitivity and specificity both 0.99. Then $P(+) = 0.01 \cdot 0.99 + 0.99 \cdot 0.01 = 0.0198$ and $P(\text{ill} \mid +) = 0.0099 / 0.0198 = 0.5$: the low base rate cancels the test's accuracy. A second independent positive multiplies the prior odds (now 1:1) by the likelihood ratio 99, giving 0.99. For the follow-up about repeated testing, use first-step analysis: if each test is positive independently with probability $p$, the expected number of tests until two consecutive positives is $(1 + p)/p^2$.
:::

::: {.qa asked="ByteDance algorithm intern, Nowcoder 2020"}
**Player A throws a die $n$ times and keeps the maximum; player B throws once. What is the probability that B's throw is larger?**

Let $X$ be A's maximum, so $P(X \le k) = (k/6)^n$. Then $P(Y > X) = \sum_{y=1}^{6} \frac{1}{6} P(X \le y - 1) = \frac{1}{6}\sum_{k=0}^{5} (k/6)^n$. For $n = 1$ this is $15/36 = 5/12$, the familiar two-dice answer, and it falls quickly as $n$ grows.
:::

::: {.qa asked="ByteDance algorithm intern, Nowcoder 2020"}
**Prove that the point minimizing the sum of squared distances to a set of points is their centroid.**

Let $f(x) = \sum_{i=1}^n \|x - p_i\|^2$. Then $\nabla f = 2\sum_i (x - p_i) = 0$ gives $x = \bar p$, and the Hessian $2nI$ is positive definite, so this is the unique global minimum. Equivalently, expanding around the mean gives $f(x) = n\|x - \bar p\|^2 + \sum_i \|p_i - \bar p\|^2$ because the cross term vanishes. This is why the k-means update uses the mean; with unsquared distances the minimizer is the geometric median instead.
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**What is the expected maximum of $n$ i.i.d. Uniform(0, 1) variables?**

$P(M \le x) = x^n$, so $\mathbb{E}[M] = \int_0^1 (1 - x^n)\,dx = 1 - \frac{1}{n+1} = \frac{n}{n+1}$. More generally the $k$-th smallest of $n$ uniforms has expectation $k/(n+1)$, so the points split the interval into $n + 1$ gaps of equal expected length.
:::

::: {.qa asked="ByteDance CV search 2023, Meta RS full-time 2024"}
**Sample an index from a given discrete distribution. How do you make it fast?**

Build the cumulative sums once, draw $u \sim U(0, 1)$, and binary-search for the first index whose cumulative weight reaches $u$ times the total: $O(n)$ setup and $O(\log n)$ per sample. For many samples from a fixed distribution, the alias method gives $O(1)$ per sample after $O(n)$ setup; when weights change, a Fenwick or segment tree supports $O(\log n)$ updates and samples. The Gumbel-max trick, $\arg\max_i(\log p_i + g_i)$ with Gumbel noise $g_i$, is another exact method and underlies the Gumbel-softmax relaxation.
:::

::: {.qa asked="Google DeepMind research intern, Glassdoor"}
**What is a Jacobian, and where does it show up in generative modeling?**

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J_{ij} = \partial f_i / \partial x_j$ is the best local linear approximation of $f$. For an invertible map $y = f(x)$, the change of variables gives $p_Y(y) = p_X(x)\,|\det J_f(x)|^{-1}$, which is the basis of normalizing flows; their layers are designed so the determinant is cheap, for example triangular Jacobians in coupling layers. In continuous-time flows the log-density evolves as $\frac{d}{dt}\log p(x_t) = -\mathrm{tr}(\partial v / \partial x)$. Backpropagation computes vector-Jacobian products without ever forming $J$.
:::

::: {.qa asked="Google DeepMind research intern, Glassdoor"}
**What does it cost to invert a matrix, how should you solve a linear system in practice, and how do you test linear independence?**

Inversion by Gaussian elimination or LU costs $O(n^3)$. You rarely need the inverse: solve $Ax = b$ with an LU factorization, with Cholesky when $A$ is symmetric positive definite (about half the cost and stable), with QR for least squares, or with conjugate gradients for large sparse SPD systems; the Woodbury identity handles low-rank updates. Vectors are linearly independent when the matrix they form has full column rank; numerically, check the smallest singular value relative to the largest (via SVD or pivoted QR) rather than an exact rank.
:::

Deeper: [Deep Learning Training](deep-learning-training.html) for losses as likelihoods, and [GANs, VAEs and Autoregressive Models](gans-vaes-autoregressive.html) for divergences in generative training.

## Diffusion and flow matching

Notation: DDPM's $\beta_t$, $\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s \le t} \alpha_s$, and $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$.

::: {.qa asked="NVIDIA DL intern, Alibaba Taotian, Baidu (3–4 reports)"}
**Explain how a diffusion model works.**

A fixed forward process adds Gaussian noise until the data become nearly $\mathcal{N}(0, I)$; in closed form, $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$. A network $\epsilon_\theta(x_t, t)$ is trained with $\mathbb{E}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ at random $t$; the optimal network gives the score, $\nabla \log p_t(x_t) \approx -\epsilon_\theta / \sqrt{1 - \bar\alpha_t}$. Sampling starts from noise and denoises step by step, with DDPM ancestral steps or an ODE solver. A complete answer names the design choices: noise schedule, prediction target, sampler, guidance, and pixel versus latent space.
:::

::: {.qa asked="JD multimodal 2026 (via interview repo)"}
**Derive the DDPM training objective.**

The ELBO gives $-\log p_\theta(x_0) \le \mathbb{E}_q\big[\mathrm{KL}(q(x_T|x_0) \| p(x_T)) + \sum_{t>1} \mathrm{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1)\big]$. The posterior $q(x_{t-1}|x_t, x_0)$ is Gaussian with mean $\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\epsilon\big)$. With $p_\theta$ Gaussian of fixed variance $\sigma_t^2$ and the mean parameterized the same way through $\epsilon_\theta$, each KL term equals $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar\alpha_t)}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ plus a constant. Ho et al. drop the weight to get the simple loss; this down-weights the small-$t$ terms, so the network spends more capacity on the harder, noisier denoising tasks, and samples improve.
:::

::: {.qa asked="NVIDIA DL intern (Jointaro, Glassdoor 2024–25)"}
**How does the noise schedule affect training?**

The schedule decides how training effort is spread over signal-to-noise ratios, $\mathrm{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$. Too much time at high SNR spends capacity on easy denoising; too little at low SNR hurts global structure. The linear schedule destroys information quickly at low resolution, which motivated the cosine schedule; higher resolutions need a shift toward more noise because neighbouring pixels are redundant, so the same per-pixel noise leaves more signal (SD3 shifts its timestep distribution for this reason). The final SNR should be zero, or there is a mismatch between training at $t = T$ and sampling from pure noise. Schedule and loss weighting interact, which is why EDM treats the training noise distribution and the sampling discretization as separate choices.
:::

::: {.qa asked="Tencent Youtu, Baidu, Alibaba, Xiaomi +3 (7 reports)"}
**DDPM vs DDIM: what is the difference, why can DDIM drop the Markov assumption, and which is a special case of which?**

The DDPM loss depends only on the marginals $q(x_t | x_0)$, so any forward process with the same marginals has the same training objective and the same trained network can be reused. DDIM defines a non-Markovian family with those marginals and samples with $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\,\hat\epsilon + \sigma_t z$, where $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$. Setting $\sigma_t^2 = \tilde\beta_t$ (the DDPM posterior variance, $\eta = 1$) recovers DDPM ancestral sampling, and $\sigma_t = 0$ gives deterministic DDIM, a discretization of the probability-flow ODE; so DDIM is the more general family and DDPM sampling is a special case. Deterministic DDIM tolerates skipping steps much better and gives an approximately invertible map from noise to image, which is used for inversion and editing.
:::

::: {.qa asked="Alibaba Taotian, image-gen intern (Nowcoder)"}
**Which samplers do you know (DDIM, DPM-Solver++, EDM's Heun, LCM, Turbo), and how do they differ?**

There are training-free solvers and distilled models. Among solvers, DDIM is a first-order ODE solver; DPM-Solver and DPM-Solver++ use the semi-linear structure of the diffusion ODE (an exponential integrator in log-SNR) to reach good quality in 10–20 steps, and the ++ variant predicts data, which is more stable under large guidance; EDM uses a second-order Heun method with step sizes concentrated at low noise; stochastic samplers (ancestral steps, EDM's churn) inject noise that corrects earlier errors but need more steps. Distilled models retrain the network for few steps: progressive distillation halves the step count repeatedly, consistency models and LCM learn to map any $x_t$ to $x_0$ in 1–4 steps, and adversarial diffusion distillation (SD-Turbo) adds a GAN loss. Distillation costs extra training and usually some diversity.
:::

::: {.qa asked="JD 2024, image-gen intern 2024 (Nowcoder)"}
**Classifier guidance vs classifier-free guidance: how do they work, and how is CFG trained and applied?**

Classifier guidance trains a classifier on noisy inputs and adds $w \nabla_{x_t} \log p_\phi(c | x_t)$ to the score. CFG trains one network with the condition dropped at random (often about 10% of the time), so it learns both conditional and unconditional predictions, and samples with $\tilde\epsilon = \epsilon_\theta(x_t, \varnothing) + w\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\big)$. Because $\nabla \log p(c|x) = \nabla \log p(x|c) - \nabla \log p(x)$, the guided score at each noise level is that of $p(x|c)\,p(c|x)^{w-1}$, an implicit classifier, so samples approximately follow this tilted density. Larger $w$ improves prompt adherence and fidelity at the cost of diversity, with oversaturation at high values; each step needs two forward passes unless guidance is distilled. Some papers write the scale as $1 + w$, so state your convention.
:::

::: {.qa asked="Alibaba Taotian AIGC, Nowcoder 2025"}
**What are the pros and cons of diffusion models?**

Pros: a stable regression loss with no adversarial game, good mode coverage, predictable scaling, flexible conditioning and guidance, likelihood bounds, and use as a prior for editing and inverse problems without retraining. Cons: slow iterative sampling (tens of network evaluations), large compute at high resolution and for video, no compact semantic latent like a GAN's or VAE's, only approximate inversion, artifacts at high guidance, and memorization risk on small or duplicated data. Latent diffusion, fast solvers and distillation address the cost side.
:::

::: {.qa asked="Xiaomi AIGC algorithm, Nowcoder 2025"}
**What is rectified flow, the training method behind SD3 and FLUX, and what does it gain over DDIM?**

Using the SD3 convention, rectified flow interpolates linearly, $x_t = (1 - t)\,x_0 + t\,\epsilon$, and trains $v_\theta(x_t, t)$ to regress the conditional velocity $\epsilon - x_0$; the minimizer is the marginal velocity field, whose ODE, integrated from $t = 1$ to $0$, carries noise to data. It belongs to the same family as diffusion (a particular schedule with a velocity target), and the loss is simple. Only the conditional paths are straight: the marginal trajectories are curved, and with a matched SNR schedule deterministic DDIM on a diffusion model takes the same steps as Euler on the flow, so practical gains come from the schedule, the timestep sampling and the target. Reflow (retraining on the model's own noise-image pairs) is what straightens the marginal trajectories for few-step sampling, and SD3 samples $t$ from a logit-normal distribution to emphasize intermediate noise levels.
:::

::: qa
**How do flow matching and diffusion relate? Are they different models?**

With Gaussian paths, both describe $x_t = \alpha_t x_0 + \sigma_t \epsilon$; they differ in the schedule $(\alpha_t, \sigma_t)$, the prediction target ($\epsilon$, $x_0$, $v$ or velocity) and the loss weighting over time. Each prediction can be converted into the others given $x_t$, and the probability-flow ODE of a diffusion model is a flow. Practical differences come from the schedule and weighting, not from the name. Flow matching is more general in one respect: it allows non-Gaussian sources and couplings, such as optimal-transport pairings or data-to-data translation.
:::

::: qa
**Once the diffusion training loss plateaus, should you stop training?**

Not on that basis alone. The loss averages over noise levels and is mostly irreducible error, the posterior variance of the target given $x_t$, which no model can remove, so real gains at the noise levels that matter for sample quality can be invisible in the total. Track the loss per timestep bucket and sample-based metrics (FID, CLIP score, a fixed set of prompts) computed with EMA weights; EMA models often keep improving after the raw loss flattens. On small datasets, also check for memorization.
:::

::: {.qa asked="Kuaishou multimodal round 2 (Zhihu snippet)"}
**How would you combine diffusion with DETR-style detection?**

Treat the box coordinates as the variable that is diffused: corrupt ground-truth boxes with Gaussian noise during training, and train a DETR-like decoder conditioned on image features to recover them; at inference, start from random boxes and refine them iteratively. This is the approach of DiffusionDet (Chen et al., 2022), which lets the number of boxes and refinement steps change at test time. A related, non-generative idea is DN-DETR's query denoising, which adds noised ground-truth boxes as extra queries to speed up training. Discuss what the generative view buys (iterative refinement, flexibility) against its extra inference cost.
:::

::: {.qa asked="Kling RL role (Nowcoder snippet)"}
**How do you fine-tune a diffusion model with reinforcement learning, for example with policy gradients or GRPO?**

Treat the denoising chain as an MDP: the state is $(x_t, t, c)$, the action is $x_{t-1}$, and the policy $p_\theta(x_{t-1} | x_t)$ is Gaussian, so its log-probability is tractable; the reward (aesthetics, text alignment, a human-preference model) arrives only at the final image. DDPO (Black et al., 2023) applies PPO-style updates with per-step importance ratios. GRPO-style methods sample a group of images per prompt and use group-normalized rewards as advantages, which removes the value network that GAE needs; for flow models with deterministic ODE samplers, Flow-GRPO converts the ODE into an SDE with the same marginals to obtain stochastic steps with likelihoods. Main risks are reward hacking and lost diversity, controlled with a KL penalty toward the reference model. Alternatives are backpropagating a differentiable reward through sampling (DRaFT, AlignProp) and Diffusion-DPO on preference pairs.
:::

Deeper: [Diffusion Models](diffusion-models.html) and [Flow Matching](flow-matching.html).

## GANs, VAEs and autoregressive models

::: {.qa asked="Alibaba, Kuaishou, ByteDance (5 interviews)"}
**Compare GANs, VAEs, normalizing flows and diffusion models for image synthesis.**

GANs sample in one step and give sharp images, but adversarial training is unstable, prone to mode collapse, and gives no likelihood. VAEs train stably on the ELBO and have a useful latent space, but samples are blurry when the decoder likelihood is a simple Gaussian. Normalizing flows give exact likelihoods, but invertibility and cheap Jacobians constrain the architecture and limit quality. Diffusion and flow-matching models train with a stable regression loss and lead on quality and coverage at scale, at the cost of many sampling steps. Current systems combine them: an autoencoder trained with perceptual and adversarial losses defines the latent space, a diffusion or flow model generates in it, and GAN losses reappear in few-step distillation.
:::

::: {.qa asked="Alibaba Cloud security vision, Nowcoder 2025"}
**Why are GANs hard to train, and what helps?**

Training solves a minimax game, not a minimization, so simultaneous gradient steps can oscillate or diverge. When the discriminator wins easily, for example when real and generated distributions barely overlap, the original loss saturates and the generator receives weak gradients; the non-saturating loss reduces this. Mode collapse happens when the generator maps many latents to a few outputs that fool the current discriminator. There is also no reliable loss value to monitor. Remedies include hinge or Wasserstein losses with gradient penalties, spectral normalization, R1 regularization, different learning rates for the two networks, an EMA of the generator, and differentiable augmentation (DiffAugment, ADA) when data are scarce.
:::

::: {.qa asked="Alibaba Cloud security vision, Nowcoder 2025"}
**What are the key ideas of StyleGAN and CycleGAN?**

StyleGAN maps $z$ through an MLP to an intermediate latent $w$ that is less entangled, injects $w$ into every layer by modulating feature statistics (AdaIN in the first version, weight demodulation in StyleGAN2), and adds per-layer noise for stochastic detail; this enables style mixing and a $W$ space suited to editing and inversion. CycleGAN translates between unpaired domains with generators $G: X \to Y$ and $F: Y \to X$, two discriminators, and a cycle-consistency loss $\|F(G(x)) - x\|_1 + \|G(F(y)) - y\|_1$ that constrains an otherwise underdetermined mapping. Its known limits: large geometric changes are hard, and the generator can hide information in imperceptible signals to satisfy the cycle loss.
:::

::: {.qa asked="ByteDance LLM algorithm round 3, Nowcoder 2026"}
**Write the VAE loss and explain the KL term. What happens if its weight is too large?**

The ELBO is $\log p(x) \ge \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) \| p(z))$. For a Gaussian encoder and a standard normal prior, $\mathrm{KL} = \frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)$, and gradients flow through $z = \mu + \sigma \odot \epsilon$. The KL term keeps posteriors close to the prior so that prior samples decode to valid images, and it regularizes the latent space. A large weight ($\beta > 1$) makes the latent carry less information: reconstructions blur, and in the extreme the posterior collapses to the prior, although factors can become more disentangled. Latent-diffusion autoencoders use a very small KL weight, because the diffusion model rather than the prior generates the latents.
:::

::: {.qa asked="Image-gen intern, company unnamed (Nowcoder 2024)"}
**In Stable Diffusion's VAE, what are the input and output shapes, and which losses constrain it?**

In SD 1.x and 2.x a $512 \times 512 \times 3$ image maps to a $64 \times 64 \times 4$ latent (8x downsampling per side); SD3 and FLUX use 16-channel latents for more detail. Training combines an L1 reconstruction loss, an LPIPS perceptual loss, a patch-based adversarial loss and a small KL penalty. The perceptual and adversarial terms keep the decoder from producing blur, and the KL term keeps the latent scale controlled. Latents are multiplied by a constant (0.18215 in SD 1.x) so they have roughly unit variance before diffusion.
:::

::: qa
**What is posterior collapse in a VAE, and how do you prevent it?**

With a powerful decoder, for example an autoregressive one, the model can explain $x$ without using $z$; the optimum then sets $q(z|x) = p(z)$ and the KL term goes to zero, so the latent carries no information. Fixes: anneal the KL weight up from zero, enforce a minimum KL per dimension ("free bits"), weaken the decoder or restrict its receptive field, or switch to a discrete bottleneck (VQ-VAE), which has no KL term pulling the posterior to the prior, though it has its own failure mode in unused codebook entries.
:::

::: qa
**Autoregressive models vs diffusion transformers for image generation: what are the trade-offs?**

Autoregressive models tokenize the image, often with a VQ tokenizer, and predict tokens in sequence with cross-entropy. They give exact likelihoods and share infrastructure with LLMs, which eases unified multimodal models, but decoding is sequential, raster order is arbitrary for images, and quantization limits fidelity. Diffusion transformers work on continuous latents, refine all positions in parallel over fewer steps than there are tokens, and currently lead on image quality, but they do not give exact likelihoods and sit apart from the LLM stack. Hybrids narrow the gap: masked parallel decoding (MaskGIT), next-scale prediction (VAR), and continuous-token autoregression with a small diffusion head per token (MAR).
:::

Deeper: [GANs, VAEs and Autoregressive Models](gans-vaes-autoregressive.html).

## Vision backbones and self-supervised learning

::: {.qa asked="Google research intern and PhD roles (Glassdoor)"}
**What is a contrastive loss? Write InfoNCE and explain where it is used.**

A contrastive loss pulls matching pairs together and pushes non-matching pairs apart in an embedding space. InfoNCE for a query $q$ with one positive $k^+$ among $N$ candidates is $L = -\log \frac{\exp(q \cdot k^+ / \tau)}{\sum_{i=1}^{N} \exp(q \cdot k_i / \tau)}$: a softmax cross-entropy where the positive is the correct class. Minimizing it maximizes a lower bound on mutual information, $I \ge \log N - L$, and the temperature $\tau$ controls how strongly hard negatives are weighted. It is used in SimCLR and MoCo (two augmentations of one image as positives), CLIP (image-text pairs) and CPC. It needs many negatives (large batches or a queue) and suffers from false negatives. The older margin form is $y\,d^2 + (1 - y)\max(0, m - d)^2$ for distance $d$ and label $y$ ($y = 1$ for a matching pair).
:::

::: {.qa asked="Xiaomi, TME, Tencent, image-gen intern (4 reports)"}
**How is CLIP trained? Write its loss.**

An image encoder and a text encoder map into a shared space with L2-normalized embeddings. For a batch of $N$ pairs, the logits are $s \cdot I T^\top$ (an $N \times N$ matrix, with a learned scale $s = 1/\tau$), and the loss is the average of the cross-entropy over rows (image to text) and over columns (text to image), with the diagonal as targets. In code: `logits = s * img @ txt.T; y = arange(N); loss = (ce(logits, y) + ce(logits.T, y)) / 2`. It was trained on about 400M web pairs and classifies zero-shot by comparing an image with prompts such as "a photo of a {class}". In generation it supplies text encoders (SD 1.x, SDXL), the CLIP score, and image embeddings for IP-Adapter; its weaknesses are counting, spatial relations and word order.
:::

::: {.qa asked="Google intern, Ant, Alibaba Amap (3 reports)"}
**What are the drawbacks of ViTs compared with CNNs, and vice versa?**

CNNs build in locality and translation equivariance, so they are data-efficient, scale linearly with pixel count and form multi-scale hierarchies naturally. ViTs lack these priors, so they need more data, strong augmentation or distillation (DeiT), or pretraining; attention is quadratic in the number of tokens, which limits resolution; large patches give coarse single-scale feature maps, a handicap for dense prediction; and position embeddings must be interpolated for new resolutions. ViTs in turn have a global receptive field from the first layer, scale better with data and compute, and use the same token interface as text, which simplifies multimodal models. Hybrids sit in between (Swin's windowed attention, ConvNeXt's modernized CNN). Whether CNN distillation still matters depends on the target: for on-device latency, it often does.
:::

::: {.qa asked="Xiaomi multimodal, image-gen intern (Nowcoder 2024)"}
**Describe the ViT architecture and how DiT adapts it for diffusion.**

ViT splits an image into $P \times P$ patches, embeds each linearly, adds position embeddings (and a class token), runs a stack of pre-norm transformer blocks, and classifies from the class token. DiT applies the same design to VAE latents (patch size 2 on a $32 \times 32 \times 4$ latent for 256-pixel images). The timestep and class embeddings condition every block through adaLN-Zero, which regresses a scale, shift and gate from the conditioning, with gates initialized to zero so each block starts as the identity. The final layer predicts noise (and variance) per patch, which is then unpatchified; sample quality improved steadily with model FLOPs. Text-to-image variants add cross-attention (PixArt-$\alpha$) or joint attention (MMDiT).
:::

::: {.qa asked="ByteDance 2020, Baidu 2024 (Nowcoder)"}
**Why does a U-Net downsample and upsample, and what do its skip connections do?**

The encoder downsamples to grow the receptive field and capture context cheaply; the decoder upsamples back to full resolution for dense prediction. Skip connections concatenate encoder features at each scale into the decoder, restoring the spatial detail lost in downsampling and shortening gradient paths. In diffusion U-Nets the multi-scale structure suits denoising at all noise levels: attention layers sit at the lower resolutions, and the timestep embedding enters every residual block. U-Net skips concatenate, while ResNet skips add.
:::

::: {.qa asked="Ctrip, bigo, NVIDIA (Nowcoder, PracHub)"}
**Why do residual connections work? Compare ResNet with VGG and DenseNet.**

A residual block computes $y = x + F(x)$, so the identity is easy to represent and extra depth does not have to hurt optimization; the Jacobian $I + \partial F / \partial x$ gives gradients a direct path through depth. ResNet fixed the degradation problem, in which deeper plain networks had higher training error. VGG stacks plain $3 \times 3$ convolutions, puts most of its parameters in fully connected layers, and becomes hard to train beyond roughly 20 layers. DenseNet concatenates all earlier feature maps within a block, which encourages feature reuse with fewer parameters but costs memory for the growing concatenations.
:::

::: {.qa asked="NIO 2024, Tencent multimodal 2026 (Nowcoder)"}
**How does SAM work, what prompts does it take, and how do SAM 2 and SAM 3 differ?**

SAM runs a large ViT image encoder once per image, encodes prompts (points, boxes, masks) with a light prompt encoder, and uses a small two-way transformer decoder to output several candidate masks with predicted IoU scores, which handles ambiguous prompts. It generalizes because of its promptable design and its data engine, which produced about 1.1B masks (SA-1B). SAM 2 extends this to video with a streaming memory of past frames that the current frame attends to, so it tracks objects through time. SAM 3 adds concept prompts, a short noun phrase or an exemplar image, and returns every matching instance, which brings open-vocabulary detection into the segmenter.
:::

::: {.qa asked="Qualcomm 2026, Glassdoor"}
**How does YOLO work?**

YOLO is a one-stage detector: a single forward pass predicts, at each cell of one or more feature grids, objectness, class probabilities and box parameters. YOLOv2 introduced anchors; later versions such as YOLOX and YOLOv8 are anchor-free. Training combines a box-regression loss (IoU-based variants such as CIoU in later versions) with objectness and classification losses, and multi-scale heads on FPN- or PAN-style necks handle object size. Non-maximum suppression removes duplicates, although some recent versions train one-to-one assignments to drop it. Compared with two-stage detectors such as Faster R-CNN, YOLO is faster and was historically weaker on small objects.
:::

::: qa
**Compare MAE and DINO as self-supervised pretraining methods.**

MAE masks about 75% of the patches, encodes only the visible ones, and reconstructs pixels with a light decoder; it is cheap to train and fine-tunes well, but its frozen features are weaker under linear probing. DINO trains a student to match an EMA teacher's softmax outputs across different crops, using centring and sharpening of the teacher outputs to avoid collapse without negatives; its frozen features are strong for k-NN retrieval and segmentation. DINOv2 scales the recipe with curated data, and its features are now used for feature-space generative metrics and for representation-alignment losses in diffusion training (REPA).
:::

Deeper: [Vision Backbones](vision-backbones.html).

## Restoration and super-resolution

::: {.qa asked="Megvii PhD SR (2021, 2022), Adobe RI (3 reports)"}
**Implement 2x bilinear interpolation. What are the pitfalls?**

Map each output pixel to input coordinates. With pixel-centre alignment (PyTorch's `align_corners=False`), $x = (j + 0.5)/s - 0.5$, clamped to the image; with `align_corners=True`, $x = j\,(W_{\text{in}} - 1)/(W_{\text{out}} - 1)$. With $x_0 = \lfloor x \rfloor$, $x_1 = x_0 + 1$ and $d_x = x - x_0$ (likewise for $y$), the value is $(1-d_x)(1-d_y)I[y_0, x_0] + d_x(1-d_y)I[y_0, x_1] + (1-d_x)d_y I[y_1, x_0] + d_x d_y I[y_1, x_1]$. In numpy, build index arrays once and gather instead of looping, clamp indices at the border, and work in float. The usual pitfall is the half-pixel convention, which differs across libraries and settings (OpenCV, PIL, PyTorch's `align_corners`) and produces small shifts that cost PSNR; also note that the operation is separable (one 1D pass per axis). One poster called this question a must for SR candidates.
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**What changes when you use interpolation to downsample instead of upsample?**

Downsampling must remove frequencies above the new Nyquist limit, or they alias into moiré and jagged edges. Bilinear or bicubic kernels with a fixed 2- or 4-pixel support do not low-pass enough at large factors; correct resizing widens the kernel by the scale factor (antialiased resizing, as in PIL, MATLAB's `imresize` or PyTorch's `antialias=True`). This matters for SR because low-resolution training images are usually made by downsampling: a mismatch between the training degradation and real inputs is a common reason models fail on real photos.
:::

::: {.qa asked="bigo image/CV, Nowcoder 2019"}
**Which upsampling layers do super-resolution networks use? Explain sub-pixel convolution.**

Pre-upsampling networks (SRCNN, VDSR) interpolate first and refine at high resolution, which is expensive. Post-upsampling networks keep computation at low resolution and upsample at the end with transposed convolution (prone to checkerboard artifacts when the kernel size is not divisible by the stride), resize-then-convolve (no checkerboard), or sub-pixel convolution. Sub-pixel convolution (pixel shuffle, from ESPCN) predicts $C r^2$ channels at low resolution and rearranges them into an $rH \times rW \times C$ output; it is cheap, and ICNR initialization reduces its artifacts. Progressive upsampling (LapSRN) and arbitrary-scale modules extend these.
:::

::: {.qa asked="Hikvision round 2, Nowcoder 2022"}
**How would you build super-resolution for arbitrary, including non-integer, scales?**

Replace the fixed upsampling layer with a function of continuous coordinates. Meta-SR predicts the upscaling filter weights from the fractional offset and the scale. LIIF learns a local implicit function: an MLP takes the nearest low-resolution feature, the relative coordinate of the query point and the cell size, and outputs RGB at any location; it is trained with random scales. A simple baseline is to super-resolve to the next integer scale and then downsample with antialiasing. Discuss behaviour at scales outside the training range and the per-pixel cost of the MLP.
:::

::: {.qa asked="Horizon Robotics, Nowcoder 2022"}
**How would you combine super-resolution with semantic segmentation?**

It works in both directions. Segmentation can guide SR: condition the SR network on semantic maps so generated textures match the class, as SFTGAN does with spatial feature transforms that modulate features by segmentation probabilities. SR can help segmentation: add an SR branch that shares the encoder during training to sharpen fine structure, as in dual super-resolution learning, or run SR before segmentation for small or distant objects. Say how you would test it: mIoU on small objects, and whether the gain survives against a strong high-resolution backbone at equal compute.
:::

::: {.qa asked="vivo written test, Nowcoder 2022"}
**What kinds of noise are in RAW sensor data, and how do you model and estimate them?**

The main sources are photon shot noise (Poisson, variance proportional to the signal), read noise (roughly Gaussian and signal-independent, from the readout electronics), row or banding noise, quantization noise, and fixed-pattern noise (dark current, pixel response non-uniformity). A common model is heteroscedastic Gaussian, $y = x + n$ with $n \sim \mathcal{N}(0, a x + b)$, where $a$ reflects the gain and $b$ the read-noise variance, both depending on ISO. Estimate them by calibration: flat-field captures at several exposures give a linear mean-variance relation, and dark frames give $b$; they can also be fitted from flat patches in a single image. After the ISP (demosaicing, tone mapping, compression), the noise becomes spatially correlated and harder to model, one reason denoising on RAW tends to work better.
:::

::: {.qa asked="vivo written test, Nowcoder 2022"}
**Which loss functions would you use for image denoising, and why?**

L2 targets the MMSE estimate and so aligns with PSNR, but it blurs fine texture; L1 and Charbonnier (a smooth L1) often converge better and keep edges sharper, and are common in restoration. Perceptual (VGG or LPIPS) and adversarial losses buy realism at some cost in fidelity, which is the perception-distortion trade-off; SSIM and MS-SSIM losses target structure. For RAW denoising, compute the loss after a simple ISP or a tone curve so dark regions are not under-weighted. Without clean targets, Noise2Noise shows that an L2 loss against a second noisy copy has the same minimizer when the noise is zero-mean and independent between the two copies.
:::

::: {.qa asked="ByteDance image 2021, OPPO 2021 (Nowcoder)"}
**Survey classical image denoising and filtering: mean, Gaussian, median, bilateral, non-local means, BM3D. Which edge operators do you know?**

Mean and Gaussian filters are linear low-pass filters that suppress Gaussian noise but blur edges. The median filter is non-linear, removes salt-and-pepper noise very well and preserves edges better. The bilateral filter weights neighbours by both spatial distance and intensity difference, so it smooths without crossing edges. Non-local means averages pixels whose surrounding patches are similar anywhere in a search window, and BM3D groups similar patches into 3D stacks, filters them jointly in a transform domain, then refines with a Wiener step; it remains a strong classical baseline. Edge operators: Roberts, Prewitt and Sobel estimate first derivatives, the Laplacian is a second derivative, and Canny adds smoothing, non-maximum suppression and hysteresis thresholds.
:::

::: {.qa asked="ByteDance image intern, Nowcoder 2020"}
**How would you remove a watermark from an image?**

It depends on what you know. If the same watermark appears on many images, as with stock photos, estimate the watermark and its alpha matte from the collection and invert the blend $J = \alpha W + (1 - \alpha) I$ (Dekel et al., 2017). For a single image, segment the watermark, then inpaint the region with a classical method (PatchMatch), a learned inpainting network, or diffusion inpainting with a mask. For semi-transparent marks, inverting the blend keeps the underlying content better than inpainting over it. Mention the legal and ethical side, and that robust watermarks are designed to resist exactly this.
:::

::: {.qa asked="ByteDance image intern, Nowcoder 2020"}
**How do RAW and RGB images differ, and which colour spaces do you know?**

RAW data are linear sensor readings with one colour per pixel behind a Bayer filter, 10–14 bits deep, before white balance or gamma. An sRGB image is the output of the ISP: black-level subtraction, demosaicing, white balance, a colour-correction matrix, tone mapping and gamma, denoising and sharpening, then 8-bit quantization and usually compression. Colour spaces: sRGB (gamma-encoded) and linear RGB; XYZ; YCbCr or YUV, which separate luma from chroma for compression (many SR papers evaluate on the Y channel); HSV and HSL for hue, saturation and brightness edits; and CIELAB, which is approximately perceptually uniform. Higher bit depth reduces banding in gradients and leaves headroom for editing.
:::

::: {.qa asked="Google Research Scientist, full-time (Glassdoor)"}
**Design an image-to-image translation network that uses a diffusion model.**

Choose by the data you have. With paired data, train a conditional diffusion model that concatenates the source image (or its latent) to the noisy target as extra input channels (SR3, Palette), or add a ControlNet-style branch to a pretrained text-to-image model to reuse its prior. With unpaired data, SDEdit noises the source to an intermediate level and denoises it with a target-domain model, where the starting noise level trades realism against faithfulness; inversion followed by editing is another route. Then discuss pixel versus latent space (an autoencoder may lose fine detail that restoration needs), how to enforce fidelity to the input, evaluation (fidelity metrics plus FID or a user study) and inference cost (distillation to a few steps).
:::

::: {.qa asked="Meituan CV intern, Nowcoder 2024"}
**Remove unwanted objects from street-view photos realistically. How would you do it?**

Detect and segment the objects to remove (an open-vocabulary detector plus SAM) and dilate the masks to cover shadows and reflections. Fill the holes with an inpainting model: a diffusion inpainting model with the mask as input, or a dual-branch design such as BrushNet that injects masked-image features; LaMa-style models with Fourier convolutions handle large masks on repetitive backgrounds quickly. A common failure is the model inventing a new object in the hole, which you counter with object-removal training data or by re-running the detector on the output. Evaluate with a human study and checks for hallucinated objects, and keep results consistent when several photos show the same place.
:::

Deeper: [Restoration and Super-Resolution](restoration-super-resolution.html).

## Text-to-image and multimodal models

::: {.qa asked="ByteDance, Baidu, Alibaba, Xiaomi, TME (8 reports)"}
**Describe Stable Diffusion's components and how the text prompt controls generation.**

There are four parts: a VAE that maps images to an 8x-downsampled latent and back; a text encoder (CLIP ViT-L/14 in SD 1.x, OpenCLIP in 2.x, two CLIP encoders in SDXL, plus T5 in SD3); a denoising U-Net of residual blocks and transformer blocks with self-attention and cross-attention; and a scheduler. The U-Net takes the noisy latent, the timestep (a sinusoidal embedding added inside the residual blocks) and the text tokens, and predicts the noise or $v$. Text acts through cross-attention: queries come from the image features and keys and values from the text-token embeddings, so each spatial location pulls in the words relevant to it, and the output keeps the spatial shape of the image features. CFG then amplifies the conditional signal. Training freezes the VAE and text encoder and trains the U-Net with the noise-prediction loss on image-caption pairs.
:::

::: {.qa asked="Alibaba Cloud, Baidu, Tencent Youtu (Nowcoder)"}
**Why does Stable Diffusion work in a VAE latent space rather than pixels, how does it differ from DDPM, and what could replace the VAE?**

Pixel-space diffusion spends much of its capacity on imperceptible high-frequency detail. A perceptually trained autoencoder compresses each side by 8x (48 times fewer values for a 4-channel latent) while keeping perceptual quality, so the diffusion model works on content and attention stays affordable; this was the latent-diffusion argument. DDPM, by contrast, is pixel-space and unconditional or class-conditional, while SD adds the latent space, text cross-attention and CFG. Replacements include VQ autoencoders, KL autoencoders with more channels (16 in SD3 and FLUX) for better detail, deep-compression autoencoders (32x or more per side) for high-resolution speed, latents aligned with semantic features, or pixel-space cascades (Imagen, DeepFloyd IF). The trade-off is the autoencoder's reconstruction ceiling against how easy its latent is to model.
:::

::: {.qa asked="Alibaba Taotian, Kunlun Wanwei, image-gen intern (3)"}
**What changed from SD 1.x to SD 2.x, SDXL and SD3, and why?**

SD 2.x switched to an OpenCLIP ViT-H text encoder, used v-prediction for the 768-pixel model, and trained on more heavily filtered data. SDXL has a U-Net about three times larger with more attention at lower resolutions, two text encoders (CLIP ViT-L and OpenCLIP ViT-bigG) plus a pooled embedding, size and crop conditioning so small or cropped training images do not teach artifacts, multi-aspect-ratio training, and an optional refiner for the final low-noise steps. SD3 replaced the U-Net with an MMDiT transformer, trained with rectified flow and logit-normal timestep sampling, used three text encoders (two CLIP plus T5-XXL) and a 16-channel VAE. The common intent: better prompt following through stronger text encoders, better handling of resolution and aspect ratio, and a backbone that scales.
:::

::: {.qa asked="Baidu multimodal intern, Nowcoder 2026"}
**How do MMDiT (SD3) and FLUX inject the text condition, and how does FLUX compare with Wan?**

In MMDiT, text and image tokens keep separate weights (projections, MLPs, norms) but are concatenated for a joint attention, so information flows in both directions; the pooled text embedding and the timestep enter through adaLN modulation. FLUX.1 is a rectified-flow transformer with double-stream MMDiT-style blocks followed by single-stream blocks that process the concatenated sequence with shared weights; T5-XXL token embeddings form the text sequence, a pooled CLIP vector and the timestep (and, in the guidance-distilled models, the guidance scale) enter through modulation, and rotary position embeddings encode the image-token coordinates. Wan 2.1 is a flow-matching video DiT with a causal 3D VAE in which umT5 text embeddings enter through cross-attention in each block and the timestep through adaLN-style modulation. In short: FLUX uses joint attention over text and image tokens, Wan uses cross-attention.
:::

::: {.qa asked="Baidu, ByteDance, Tencent, NIO +3 (7 reports)"}
**What is LoRA? Give the formula, the parameter count for a large layer, and how you choose the rank.**

Freeze $W \in \mathbb{R}^{d \times k}$ and learn $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ and $r \ll \min(d, k)$; the forward pass is $h = Wx + \frac{\alpha}{r} BAx$. That is $r(d + k)$ trainable parameters: for a $10^4 \times 10^4$ layer with $r = 100$, 2M instead of 100M (2%). After training, $W + \frac{\alpha}{r}BA$ can be merged, so inference costs nothing extra; for the same reason there is no nonlinearity inside the adapter. Start with a small rank (4–64 is common for a subject or style), raise it if the model underfits, and tune the learning rate together with $\alpha/r$; larger domain shifts need larger ranks. In diffusion models LoRA usually targets the attention projections. Captions for LoRA data typically come from a captioning model such as BLIP, and unlike prompt engineering, LoRA can teach concepts that text alone cannot express.
:::

::: {.qa asked="Baidu 2026, Tencent multimodal 2026 (Nowcoder)"}
**Why is LoRA's $B$ initialized to zero? Could you zero $A$ and initialize $B$ randomly instead?**

With $B = 0$ the update $\Delta W$ is zero, so fine-tuning starts exactly at the pretrained model, and the gradient $\partial L / \partial B = g\,(Ax)^\top$ (with $g = \partial L / \partial h$) is non-zero because $A$ is random. Swapping the roles also starts from $\Delta W = 0$ and trains, since then $\partial L / \partial A = B^\top g\, x^\top$ is non-zero. The two choices differ in their dynamics; one analysis (Hayou et al., 2024) found the standard choice tolerates larger learning rates. Zeroing both matrices fails: each gradient contains the other matrix, so both stay at zero.
:::

::: {.qa asked="Tencent Youtu, Baidu, image-gen intern (Nowcoder)"}
**How does ControlNet work, and why the zero convolutions?**

ControlNet copies the pretrained U-Net's encoder and middle block as a trainable branch, feeds it the control image (edges, depth, pose) through a small encoder, and adds its outputs to the frozen U-Net's skip connections through 1x1 convolutions initialized to zero. At the start the output is therefore unchanged, so the new branch cannot damage the pretrained model with random features, yet the zero convolutions still receive gradients because their inputs are non-zero. Because the base model stays frozen, the same ControlNet can be combined with other fine-tuned checkpoints. The cost is an extra encoder pass per step; T2I-Adapter is a lighter alternative.
:::

::: {.qa asked="Tencent Youtu, Baidu, Tencent MM, image-gen intern"}
**Compare DreamBooth, Textual Inversion, LoRA and IP-Adapter for subject or style personalization.**

Textual Inversion learns only a new token embedding: tiny, model unchanged, limited fidelity. DreamBooth fine-tunes the whole model on a few images with a rare identifier and a class-prior preservation loss to avoid forgetting the class: high fidelity, heavy, and prone to overfitting. LoRA applies a low-rank update (often with the DreamBooth objective): small files that can be combined, a middle ground. IP-Adapter needs no per-subject training: a trained adapter injects CLIP image embeddings through decoupled cross-attention (separate keys and values for image features, added to the text cross-attention), so it works from one reference image, with weaker identity fidelity than fine-tuning. Choose by how many reference images you have, the fidelity you need, and whether per-subject training is acceptable.
:::

::: {.qa asked="ByteDance MM, Xiaohongshu, image-gen intern (3)"}
**Explain BLIP-2's Q-Former and contrast it with LLaVA's connector.**

BLIP-2 freezes the image encoder and the LLM and trains a Q-Former, a small transformer with 32 learned query tokens that cross-attend to the image features. In stage one, with only the image encoder attached, image-text contrastive, image-grounded generation and image-text matching losses teach the queries to extract text-relevant visual information. In stage two, the query outputs are projected into the LLM's input space as soft prompts and trained with the language-modeling loss. LLaVA instead maps every CLIP patch token into the LLM embedding space with a simple projector (linear at first, a two-layer MLP in LLaVA-1.5) and fine-tunes on instruction data. The Q-Former compresses to a fixed, small number of tokens but can lose detail; MLP projectors keep all tokens and became the common choice.
:::

::: {.qa asked="Tencent multimodal 2026 (Nowcoder), JD 2026 (repo)"}
**What is M-RoPE in Qwen2-VL and Qwen2.5-VL, and how was position handled for images and video before it?**

Earlier multimodal LLMs flattened image patches into a one-dimensional sequence and gave them ordinary 1D RoPE indices, so 2D and temporal structure was only implicit. M-RoPE splits the rotary dimensions into three groups for temporal, height and width indices: text tokens use the same index in all three, which reduces to 1D RoPE; image patches share a temporal index and vary in height and width; video frames advance the temporal index. Qwen2.5-VL ties the temporal index to absolute time, so frame-rate changes do not distort timing, and its vision encoder runs at native resolution with 2D RoPE and window attention.
:::

::: {.qa asked="Alibaba Taotian AIGC, Nowcoder 2025"}
**How would you train a text-to-image model to render text in images accurately?**

First diagnose the causes: CLIP-style text encoders see subword tokens and spell poorly, the autoencoder loses small glyph detail, and few training captions transcribe the text shown in the image. Fixes follow from each cause: a character-aware or large T5-style text encoder (SD3 found that dropping T5 hurt typography, and character-aware encoders such as ByT5 spell better still); captions that quote the OCR'd text exactly; curated text-rich and synthetic rendered data; an autoencoder with more latent channels; explicit glyph or layout conditioning, as in TextDiffuser-style methods; and fine-tuning with an OCR-based reward. Measure OCR word accuracy and edit distance on prompts that quote text.
:::

::: {.qa asked="Meituan CV intern, Nowcoder 2024"}
**A merchant wants its billboard in a given style. Would you use a training-free style method or train your own?**

Training-free methods such as InstantStyle inject reference-image features only into the attention layers that carry style, building on IP-Adapter. They need no per-style training and respond quickly, but content from the reference can leak in and fidelity to a brand style is limited. Training a LoRA on 10–50 curated style images gives more consistent results at the cost of per-style training and data work. Decide by the number of styles, the turnaround time and how strict the consistency requirement is. In both cases keep the layout with a structural control (edges or layout via ControlNet) and composite the merchant's text and logo afterwards, since generated text and logos are unreliable.
:::

::: {.qa asked="Tencent Youtu CV intern, Nowcoder 2024"}
**Design a pipeline that generates 3D assets from text.**

There are three main routes. Optimization with a 2D prior: score distillation sampling (DreamFusion) optimizes a NeRF or 3D Gaussians so their renders score well under a text-to-image diffusion model; it is slow and suffers from multi-face (Janus) artifacts and oversaturation. Multi-view generation plus reconstruction: a multi-view diffusion model (MVDream, Zero123++) produces consistent views, and a feed-forward reconstruction model (LRM-style) lifts them to 3D in seconds. Native 3D generation: diffusion in a learned 3D latent (triplanes or a 3D VAE), limited by the size of 3D datasets compared with 2D ones. Discuss camera conditioning, view consistency, mesh and texture extraction, and evaluation (CLIP score on renders, user studies); the second route is a reasonable default on speed and quality.
:::

Deeper: [Text-to-Image and Multimodal Models](text-to-image-multimodal.html).

## Video generation

::: {.qa asked="Ant video intern 2024, Alibaba Taotian (Nowcoder)"}
**How are diffusion models and DiTs used for video generation, for example in Sora?**

A spatio-temporal VAE compresses the video (commonly 4x in time and 8x8 in space), the latent is cut into spacetime patches, and a DiT with full 3D attention or factorized spatial and temporal attention denoises them, conditioned on text through cross-attention or joint attention. Training uses diffusion or flow matching on a mix of images and videos at varied durations, resolutions and aspect ratios. OpenAI's Sora report described spacetime patches and native-aspect training; open models such as CogVideoX, HunyuanVideo and Wan follow similar recipes. The hard parts are the token count (a few seconds of 720p video is on the order of $10^4$–$10^5$ tokens), temporal consistency, motion quality and video captioning.
:::

::: {.qa asked="Company unnamed (via AIGC interview repo)"}
**Why do video VAEs use causal 3D convolutions, and how do 3D and (2+1)D convolutions compare?**

A per-frame image VAE gives no temporal compression (too many latents) and flickers because frames are encoded independently. 3D convolutions also compress time. Making them causal, by padding only on the past side in time, means the latent of frame $t$ depends only on frames up to $t$: the first frame is encoded on its own, so a single image is a valid one-frame video (useful for joint image-video training and for conditioning on a first frame), and long videos can be encoded chunk by chunk with cached context. With 4x temporal compression, an input of $1 + T$ frames gives $1 + T/4$ latent frames. A (2+1)D convolution, a 2D spatial convolution followed by a 1D temporal one, uses about $k^2 C^2 + k C^2$ weights instead of $k^3 C^2$, adds a non-linearity, and can start from image weights; a full 3D convolution models space-time interactions jointly at higher cost.
:::

::: qa
**Why does generating frames independently produce incoherent video, and how do architectures enforce temporal consistency?**

Independent sampling draws different noise and makes different choices of detail in every frame, so identity and texture drift and the video flickers. Architectures share information across time: temporal convolution or temporal attention layers inserted into an image model (VideoLDM, AnimateDiff), full spatio-temporal attention in video DiTs, which is the most consistent and the most expensive, and video VAEs that compress across time. Training on real video then teaches motion. Factorized attention is cheaper but weaker on long-range motion.
:::

::: qa
**How would you fine-tune a text-to-video model into a text-and-image-to-video model?**

Encode the conditioning frame with the video VAE and concatenate its latent (plus a mask channel marking known frames) to the noisy latent along the channel axis, expanding the first layer with zero-initialized weights; alternatively, replace the first latent frame with the clean conditioning latent, or add image features through cross-attention for semantics. Train on clips with the first frame given, and drop the image and text conditions at random so you can apply separate guidance scales. A common failure is a faithful first frame followed by drift; adding a little noise to the conditioning latent during training and training on enough motion helps.
:::

::: qa
**Do video diffusion models understand physics?**

Answer with evidence rather than opinion. They reproduce common dynamics well within their training distribution, but controlled studies find failures on unseen combinations, object permanence, conservation and contact, and suggest the models imitate similar training cases more than they learn general laws (Kang et al., 2024). Benchmarks such as Physics-IQ and VideoPhy make this measurable. Possible improvements include more diverse data, simulation data, 3D-aware conditioning and fine-tuning with physics-based rewards. Give both sides and say what experiment would change your mind.
:::

::: {.qa asked="Runway MLE, PracHub 2025"}
**Design a video search system, covering the model and the infrastructure.**

Start with requirements: query types (text, image, clip), latency, corpus size and freshness. Indexing: split videos into shots, sample frames, embed them with a vision-language model (per-frame CLIP-style embeddings or a video encoder for temporal semantics), add ASR transcripts, OCR text and metadata, and store embeddings in an approximate-nearest-neighbour index (HNSW or IVF-PQ) with shot timestamps. Querying: embed the query, retrieve candidates, re-rank with a cross-encoder or a multimodal LLM, and aggregate shots into videos. Evaluate with recall@k and nDCG offline and with engagement online. On the infrastructure side, cover the batch GPU embedding pipeline, incremental index updates, sharding and caching, and the cost of re-embedding everything when the model changes.
:::

::: {.qa asked="Tencent multimodal, Nowcoder 2026"}
**Track multiple faces through a video that contains shot cuts.**

Detect shot boundaries first (frame-difference or feature-difference statistics, or a learned detector). Within a shot, detect faces per frame and associate them with motion and overlap (a SORT-style Kalman filter with IoU matching) plus appearance embeddings. Across cuts, motion cues are meaningless, so link tracklets by face-recognition embeddings with clustering and a similarity threshold, and smooth over occlusions and profile views. Evaluate with IDF1 or HOTA and the number of identity switches.
:::

Deeper: [Video Generation](video-generation.html).

## Evaluation metrics

::: {.qa asked="Microsoft AS intern 2026, NIO 2024"}
**Explain precision, recall, ROC and AUC in your own words. When is a precision-recall curve more informative than ROC?**

Precision: of what you flagged, how much was right. Recall: of what was there, how much you found. ROC plots true-positive rate against false-positive rate across thresholds, and AUC is the probability that a random positive scores above a random negative. Under heavy class imbalance, ROC can look good because the false-positive rate divides by a huge number of negatives, while precision exposes the false positives directly. Generative-model precision and recall (Kynkäänniemi et al., 2019) reuse the idea: precision is the fraction of samples that land on the data manifold (quality), recall the fraction of the data manifold covered by samples (diversity). When asked to code it against a flaky API, validate inputs (missing or wrong-length results) and define the behaviour for empty denominators.
:::

::: {.qa asked="Amazon AS intern, Glassdoor 2023"}
**Define IoU and mAP for object detection.**

IoU is the area of the intersection of two boxes divided by the area of their union. A detection is a true positive if its IoU with an unmatched ground-truth box of the same class exceeds a threshold (0.5 for PASCAL VOC), with detections matched greedily in order of score. AP is the area under the precision-recall curve for one class (with interpolation), and mAP averages AP over classes; COCO also averages over IoU thresholds from 0.5 to 0.95 in steps of 0.05. For segmentation, mIoU computes IoU per class over the whole dataset and averages over classes.
:::

::: {.qa asked="bigo image/CV, Nowcoder 2019"}
**Define PSNR and SSIM. What do they miss?**

$\mathrm{PSNR} = 10 \log_{10}(\mathrm{MAX}^2 / \mathrm{MSE})$ in dB, a pure pixel-error measure. SSIM compares local luminance, contrast and structure, $\mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$, computed in local Gaussian windows and averaged. Both favour smooth, mean-like outputs and penalize plausible texture that is not pixel-aligned, so GAN and diffusion SR methods score worse on them while looking better (the perception-distortion trade-off). Report them with LPIPS and a no-reference or human evaluation, and state the conventions (Y channel or RGB, border cropping) because they change the numbers.
:::

::: {.qa asked="Amazon AS intern, CV PhD (1p3a 2020)"}
**Given two images, list every way you could measure how similar they are.**

Organize the answer by which differences should not count (shift, crop, lighting, style, viewpoint), because that choice selects the metric. Pixel-level: MSE and PSNR, L1, SSIM and MS-SSIM. Distribution-level: colour-histogram distances (chi-squared, earth mover's distance). Geometric: keypoint matching (SIFT, ORB) with geometric verification for the same scene seen differently. Learned: cosine similarity of CNN or ViT features, LPIPS for perceptual similarity, DINO features for instance-level similarity, CLIP for semantic similarity. Perceptual hashing for near-duplicate detection at scale.
:::

::: qa
**What does FID measure, and what are its pitfalls?**

FID fits Gaussians to Inception-v3 features of real and generated images and computes $\|\mu_r - \mu_g\|^2 + \mathrm{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\big)$. It is biased by sample size (compare only at equal $N$, commonly 50k), sensitive to resizing and compression details, tied to ImageNet-classifier features that can be insensitive to faces and text, and it mixes quality and diversity into one number. Complements: precision and recall, KID, Fréchet distance on DINOv2 features, and for text-to-image models CLIP score, VQA-based alignment scores and human preference.
:::

::: qa
**How would you evaluate a text-to-image model, or a character or style LoRA, in practice?**

Use several axes: image quality (FID or a DINOv2-based distance against a reference set, aesthetic predictors), text alignment (CLIP score, VQA-based scores, compositional benchmarks such as GenEval), human preference (pairwise studies, preference models), diversity, and safety. For a LoRA, also measure identity or style fidelity (face-recognition similarity, DINO or CLIP image similarity to the references), editability (text alignment over varied prompts) and copying of training images. Fix the prompts and seeds, and add a blind human comparison.
:::

::: qa
**How do you evaluate video generation?**

FVD computes a Fréchet distance on I3D video features and is more sensitive to per-frame content than to motion. VBench splits quality into dimensions such as subject consistency, motion smoothness, dynamic degree, aesthetics and text alignment. Per-frame metrics (FID, CLIP score) miss temporal artifacts, and static videos can score well on consistency, so pair consistency with a motion measure. Human evaluation remains the reference.
:::

Deeper: [Evaluation Metrics](evaluation-metrics.html).

## Research design and project proposals

::: {.qa asked="Meta, NVIDIA, Adobe, Google, GDM (9+ reports)"}
**Propose a research project you would do on our team, or bring a distinct idea for each interviewer.**

In two or three minutes, cover: the problem and why now (what recent result makes it tractable), the hypothesis, a method sketch that uses the team's assets, a first experiment that can fail within one or two weeks and its success metric, the main risk and a fallback, and the publishable outcome within the internship. Ground it in the team's recent papers without proposing their obvious next paper, and connect it to your own expertise. Prepare two or three scoped ideas at different risk levels. During the brainstorm, interviewers watch how you take pushback and refine the idea as much as the idea itself.
:::

::: {.qa asked="Google RI (Blind 2019), OpenAI applied RS"}
**Design an experiment to test a hypothesis.**

State the hypothesis as a falsifiable prediction with a direction. Identify the variable you manipulate, the confounders you hold fixed, and the smallest setting in which the effect should be visible. Pick the metric, the controls and baselines, and enough seeds or samples to detect the effect size you care about. Say in advance which result would refute the hypothesis. For example, testing whether hallucination falls with model size needs a fixed prompt set, models trained on matched data, a validated judge and confidence intervals.
:::

::: {.qa asked="Meta Reality Labs-type research intern (Glassdoor)"}
**Design a model for egocentric 3D perception.**

Clarify the task (hand or body pose, scene reconstruction, object localization), the sensors (camera rig, IMU, depth) and the compute budget (on-device or not). A reasonable design: per-frame features from a pretrained backbone, temporal fusion using camera poses from SLAM or visual-inertial odometry to lift features into a 3D or bird's-eye representation, and task heads on top; supervise with synthetic renders and real captures, plus multi-view consistency as self-supervision. Name the hard parts: fast head motion, seeing only parts of your own body, fisheye distortion and latency. Evaluate with 3D error metrics on held-out users and environments.
:::

::: {.qa asked="Adobe research intern (Glassdoor)"}
**Propose research ideas that combine a multimodal LLM with a video DiT.**

Start from what each model is good at: the MLLM at understanding and planning, the DiT at pixels. Candidate directions: the MLLM as a richer text encoder or prompt rewriter; the MLLM planning a shot list, layout or camera path that conditions the DiT; the MLLM as a critic or reward model for best-of-N selection or RL fine-tuning; or a unified transformer for both understanding and generation. For the one you pick, say what is new, the first experiment, and how you would evaluate it (VBench dimensions plus human studies). Expect follow-ups on compute and data.
:::

::: {.qa asked="Baidu multimodal intern, Nowcoder 2026"}
**For audio-driven talking-head or digital-human generation, what is unsolved, and what would be publishable?**

Map the pipeline first (audio features, then motion and expression, then rendering by 2D diffusion or 3D Gaussian avatars) and list where it fails: lip-sync precision against expressiveness, identity drift over long videos, natural head and body motion, real-time latency, generalization from one image, and evaluation (sync metrics such as LSE-C and LSE-D, FVD, user studies). A publishable idea targets one failure with a measurable gain, for example long-horizon identity consistency through a memory or reference-attention mechanism, or distillation to real time. Pick one and say how you would test it.
:::

::: {.qa asked="Google RI, Amazon AS, Meta RS FT (5 reports)"}
**Open-ended ML or CV system design, such as language detection, or a robot that identifies books and shelves them.**

Clarify the objective, inputs, constraints (latency, accuracy, scale) and success metric. Then cover data (sources, labels, how to bootstrap), a simple baseline before the main model, training and offline evaluation, online evaluation, failure modes and monitoring, and how you would iterate. For the book-shelving robot: perception (detect spines, OCR the titles, match against the catalogue), localization of shelf slots, grasp planning and error recovery, evaluated by end-to-end task success. Interviewers grade the structure and the trade-offs more than the specific model.
:::

::: {.qa asked="GDM research intern; Ant 2024; Xiaohongshu 2024"}
**Which recent papers from our team or the field interest you, and where is the field going?**

Pick one or two team papers you have actually read; say what is new in each, one limitation, and a follow-up you would test. For the direction of the field, give a view with reasons and evidence, in the shape "X scaled, but Y is still unsolved because Z", and state your uncertainty. Opinion questions (one report asked whether CNN distillation still matters in the era of large models) are best answered with conditions: when it matters, when it does not, and what evidence would settle it.
:::

Deeper: the design choices worth proposing to vary are laid out in [Video Generation](video-generation.html) and [Text-to-Image and Multimodal Models](text-to-image-multimodal.html).

## Behavioral and motivation

::: {.qa asked="Adobe, OpenAI, Luma (3 reports)"}
**Why this company or lab, and where do your research interests lie?**

Be specific: name the team's recent work that connects to yours, what this team has that others do not (data, compute, a product where the research would land), and what you want to learn. Add one or two sentences on your research direction and how the internship fits it. Generic prestige answers do not help.
:::

::: {.qa asked="OpenAI RS, Tencent Youtu (Glassdoor, Nowcoder)"}
**What are your long-term research goals? Academia or industry?**

Give an honest, coherent arc: the problems you want to work on over the next few years and why. Being undecided between academia and industry is fine if you say what would decide it. Connect the answer to the role, and avoid suggesting you would drop the project halfway for something else.
:::

::: {.qa asked="OpenAI applied RS (forum 2025)"}
**How do you balance research exploration with product timelines?**

Use a concrete example: how you time-boxed exploration, defined milestones and a fallback deliverable, and communicated early when a direction was not working. Show that you can separate exploratory work, which can fail, from commitments, which should not.
:::

::: {.qa asked="Amazon AS (Leadership Principles), Meta RS FT 2026"}
**Describe working with others on a hard problem, or resolving a disagreement.**

Use situation, task, action, result, with your own actions at the centre, a measurable outcome and what you learned. Avoid blaming others. Amazon maps these answers to its Leadership Principles, so prepare stories that show ownership, customer focus and diving into detail.
:::

::: {.qa asked="NVIDIA research intern (PracHub, 1p3a 2026)"}
**Describe your day-to-day research workflow. How would AI agents change it?**

Be concrete: how you choose the next experiment, track runs, read papers and write. Then say where agents help (literature triage, boilerplate code, launching sweeps, plotting) and where they do not (choosing problems, judging whether a result is real), and how you verify what an agent produces.
:::

::: qa
**What questions do you have for us?**

Use the time. Ask what the team actually works on and with which stack (role titles can mislead), what a successful intern produced last year, the publication policy, how involved the mentor is, compute access, and what the remaining process looks like. Posters also recommend asking what answer the interviewer was looking for on a question you found hard.
:::

Deeper: see [What interviewers look for](#what-interviewers-look-for) above.

## ML coding round

The most frequent ML-coding request across all reports is multi-head attention, followed by convolution and classic algorithms written from scratch. Write shapes in comments, test on a tiny input, and compare with the library function when you can.

::: {.qa asked="ByteDance, NVIDIA, OpenAI, Meta, Baidu +4 (9+ reports)"}
**Implement multi-head attention in PyTorch, supporting masks and cross-attention. Then build a transformer block.**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_context=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_context or d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, context=None, mask=None):
        # x: (B, N, d_model); context: (B, M, d_context), None for self-attention
        # mask: bool, broadcastable to (B, h, N, M); True = may attend
        B, N, _ = x.shape
        ctx = x if context is None else context
        q = self.q(x).view(B, N, self.h, self.dk).transpose(1, 2)        # (B, h, N, dk)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / self.dk ** 0.5                # (B, h, N, M)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v                              # (B, h, N, dk)
        return self.out(out.transpose(1, 2).reshape(B, N, -1))
```

A causal mask is `torch.tril(torch.ones(N, N, dtype=torch.bool))`; a row that is fully masked produces NaNs, so padding masks need care. Cross-attention is the same module with `context` set to the text or encoder tokens. Mention attention dropout, the $O(NMd)$ cost, and that production code calls `F.scaled_dot_product_attention`. A pre-norm transformer block is then `x = x + attn(ln1(x))` followed by `x = x + mlp(ln2(x))`, with an MLP of `Linear(d, 4d)`, GELU, `Linear(4d, d)`; a DiT block adds timestep-dependent scale, shift and gate around each sub-layer.
:::

::: {.qa asked="NVIDIA, Meta, Baidu, Xiaomi, Megvii (5 reports)"}
**Implement 2D convolution without calling a convolution layer.**

```python
import torch
import torch.nn.functional as F

def conv2d(x, w, b=None, stride=1, padding=0):
    # x: (B, C_in, H, W); w: (C_out, C_in, kh, kw); cross-correlation, as in PyTorch
    B, C, H, W = x.shape
    C_out, _, kh, kw = w.shape
    x = F.pad(x, (padding, padding, padding, padding))
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    # im2col: one strided slice per kernel offset -> (B, C, kh, kw, H_out, W_out)
    cols = x.new_empty(B, C, kh, kw, H_out, W_out)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = x[:, :, i:i + stride * H_out:stride, j:j + stride * W_out:stride]
    cols = cols.reshape(B, C * kh * kw, H_out * W_out)
    out = w.reshape(C_out, -1) @ cols                     # (B, C_out, H_out * W_out)
    if b is not None:
        out = out + b.view(1, -1, 1)
    return out.view(B, C_out, H_out, W_out)
```

State the output-size formula $\lfloor (H + 2p - k)/s \rfloor + 1$, note that deep-learning "convolution" does not flip the kernel, and give the cost $B\,C_{\text{out}} C_{\text{in}} k_h k_w H_{\text{out}} W_{\text{out}}$ multiply-adds. The im2col form trades memory for a single large matrix multiply. Verify with `torch.allclose(conv2d(x, w, b, 2, 1), F.conv2d(x, w, b, 2, 1), atol=1e-5)`. Dilation and groups are natural follow-ups.
:::

::: {.qa asked="LLM algorithm intern, company unnamed (via repo)"}
**Write a DDPM training step and the ancestral sampler.**

```python
import torch, torch.nn.functional as F

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
abar = torch.cumprod(alphas, dim=0)

def train_step(model, x0, opt):
    t = torch.randint(0, T, (x0.shape[0],), device=x0.device)
    eps = torch.randn_like(x0)
    a = abar.to(x0.device)[t].view(-1, *[1] * (x0.dim() - 1))
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    loss = F.mse_loss(model(xt, t), eps)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

@torch.no_grad()
def sample(model, shape, device):
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        eps = model(x, torch.full((shape[0],), t, device=device, dtype=torch.long))
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + betas[t].sqrt() * torch.randn_like(x) if t > 0 else mean
    return x
```

Timesteps here run from 0 to $T-1$. The sampler uses $\sigma_t^2 = \beta_t$; the posterior variance $\tilde\beta_t$ is the other choice from the DDPM paper. Follow-ups to expect: keep an EMA of the weights for sampling, optionally clip the implied $\hat x_0$, implement CFG by running the conditional and unconditional inputs as one doubled batch, and switch to DDIM with a subsequence of timesteps to sample in far fewer steps.
:::

::: {.qa asked="Insta360 multimodal (Nowcoder snippet)"}
**Implement a LoRA linear layer.**

```python
import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))     # same as nn.Linear's default
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self):
        self.base.weight += self.scale * self.B @ self.A
        return self.base
```

Explain why $B$ starts at zero (the model starts exactly at the pretrained one), why the adapter has no non-linearity (it must stay mergeable), and how to wrap the attention projections of an existing model by replacing those `nn.Linear` modules. Only $A$ and $B$ go to the optimizer.
:::

::: {.qa asked="Meta RS intern, Meta MLE 2026 (Glassdoor, PracHub)"}
**Implement k-means from scratch.**

Initialize $k$ centroids, preferably with k-means++ (each new centroid sampled with probability proportional to the squared distance to the nearest centroid already chosen). Then alternate: assign each point to its nearest centroid, computing all squared distances at once as $\|x\|^2 - 2xc^\top + \|c\|^2$, and move each centroid to the mean of its points; stop when assignments stop changing or the objective improves by less than a tolerance. Re-seed empty clusters, for example at the point farthest from its centroid. Neither step can increase the within-cluster sum of squares, so the algorithm converges, but only to a local optimum; use several restarts. Each iteration costs $O(nkd)$.
:::

::: {.qa asked="Xiaomi, Baidu, Ctrip, Horizon (4 reports)"}
**Implement IoU and non-maximum suppression.**

For boxes $(x_1, y_1, x_2, y_2)$, the intersection's top-left corner is the elementwise max of the two top-left corners and its bottom-right corner the min of the bottom-right ones; its area is $\max(0, x_2 - x_1)\max(0, y_2 - y_1)$, and IoU is the intersection divided by the sum of areas minus the intersection. Broadcasting gives the full $N \times M$ IoU matrix at once. NMS: sort by score, keep the top box, drop remaining boxes whose IoU with it exceeds a threshold, and repeat; the worst case is $O(N^2)$. Run it per class (or offset boxes by class index so classes never overlap); Soft-NMS decays scores instead of dropping boxes. Rotated boxes need a polygon intersection, for example clipping one rectangle against the other (Sutherland-Hodgman). For mIoU over a batch, accumulate per-class intersections and unions before dividing.
:::

::: {.qa asked="Adobe research intern, London (Glassdoor)"}
**Implement a Mixture-of-Experts layer for a transformer.**

A router (a linear layer from $d$ to $E$ experts) gives logits per token; apply softmax, keep the top $k$ (1 or 2) and renormalize their weights. Send each token to its chosen expert FFNs and sum the weighted outputs; in an interview, a loop over experts with boolean masks is acceptable, and you can mention that production code sorts tokens by expert, uses grouped matrix multiplies, and needs all-to-all communication under expert parallelism. Add a capacity factor that drops or reroutes overflow tokens, and a load-balancing loss such as Switch Transformer's $E \sum_i f_i P_i$, where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ its mean router probability; without it, routing tends to collapse onto a few experts.
:::

::: {.qa asked="Meta GenAI, OpenAI RE, Luma RE (3 reports)"}
**Debug this PyTorch training script: the loss does not go down, or becomes NaN.**

Work one hypothesis at a time and say each aloud. First try to overfit a single small batch: if that fails, the bug is in the model or the loss rather than in the data or the optimizer. Then check the usual suspects: misaligned or unnormalized data and labels; broadcasting errors (a $(B, 1)$ tensor minus a $(B,)$ tensor silently becomes $(B, B)$); a softmax applied before a loss that expects logits, or over the wrong dimension; a missing `zero_grad`; the wrong `train()` or `eval()` mode for dropout and BatchNorm; a stray `detach` or `no_grad` that cuts the graph; an inverted attention mask; in-place operations that break autograd; learning-rate or warmup problems; and FP16 overflow. Confirm each fix with a quick test before moving on.
:::

::: {.qa asked="OpenAI RE (Aced, ~2025)"}
**Add a KV cache to a transformer decoder. What changes?**

Keep per-layer key and value tensors of shape $(B, h, T, d_h)$. A prefill pass runs the whole prompt once and fills the cache; each decoding step then computes q, k and v for the new token only, appends k and v, and attends over the full cache. Positions must be offset by the current cache length (apply RoPE to keys before caching them), the causal mask is unnecessary for a single new query, and batched decoding must track per-sequence lengths and padding. Memory grows linearly with length, so pre-allocate. Test that logits with and without the cache match to numerical tolerance.
:::

::: {.qa asked="Runway (Glassdoor 2025–26)"}
**Estimate the compute and memory needed to train a given network.**

Memory: with mixed-precision Adam, each parameter costs about 16 bytes (2 for BF16 weights, 2 for gradients, 4 for the FP32 master copy, 8 for the two Adam moments) before activations; activations scale with batch size, sequence length, width and depth, often dominate, and shrink with checkpointing. Compute: training a transformer costs about $6 \times \text{parameters} \times \text{tokens}$ FLOPs (2 for the forward pass, 4 for the backward); for a CNN, sum per-layer multiply-adds, double for FLOPs, and triple for forward plus backward. Divide by achievable throughput, typically 30–50% of peak. Example: a 1B-parameter model on 100B tokens needs $6 \times 10^{20}$ FLOPs; at an effective $4 \times 10^{14}$ FLOP/s per GPU (about 40% of an H100's dense BF16 peak) that is $1.5 \times 10^6$ GPU-seconds, roughly 420 GPU-hours.
:::

::: {.qa asked="OpenAI RS (Glassdoor)"}
**Express a 1-nearest-neighbour classifier as a feed-forward network.**

With Euclidean distance, $\arg\min_i \|x - x_i\|^2 = \arg\max_i \big(x_i^\top x - \tfrac{1}{2}\|x_i\|^2\big)$, because $\|x\|^2$ is the same for every $i$. So a linear layer with weight rows $x_i$ and biases $-\tfrac{1}{2}\|x_i\|^2$, followed by an argmax (a winner-take-all layer, or a softmax with temperature near zero for a differentiable version), selects the nearest training point, and a second linear layer whose weights are the one-hot labels outputs its class. This is also attention with the training points as keys and their labels as values, in the zero-temperature limit.
:::

Deeper: [Vision Backbones](vision-backbones.html) for attention and convolution arithmetic, and [Diffusion Models](diffusion-models.html) for the sampler.

## Sources

"Full" means the whole post was read; "snippet" means only a search-engine excerpt or a login-wall summary was visible. Where no URL was recorded during collection, the entry says so rather than guessing one. Entries marked curated are blogger-written question lists, not first-hand reports.

1. Blind, "Google research internship interview". <https://www.teamblind.com/post/google-research-internship-interview-nclkxdkb>. 2019 (snippet).
2. Blind, "Research internship interview with Google". <https://www.teamblind.com/post/research-internship-interview-with-google-nrek4e8d>. 2022 (snippet).
3. Glassdoor, Google Research Intern interview questions. <https://www.glassdoor.com/Interview/Google-Research-Intern-Interview-Questions-EI_IE9079.0,6_KO7,22.htm>. Various years (snippet).
4. Glassdoor, other Google and DeepMind role pages (Student Researcher, Research Scientist, PhD intern, "Google Intern"). URL not recorded. Undated (snippet).
5. AceOffer, Google Cloud research intern report. URL not recorded. 2026 (snippet).
6. Google blog, "Getting to know a research intern". <https://blog.google/inside-google/life-at-google/getting-to-know-research-intern-pau/>. 2019 (snippet).
7. Blind, Google research-intern host-matching threads. URL not recorded. 2021–2023 (snippet).
8. 1point3acres, Google research-intern threads 898599, 582418, 953975, 943143, 733331 and 683562. URL not recorded. 898599 from 2022, others undated (snippet).
9. Reddit thread on a Google research-intern interview, reposted on Tencent Cloud. URL not recorded. 2020 (snippet).
10. uscardforum, thread 151564 on Google intern host matching. URL not recorded. 2023 (snippet).
11. Glassdoor, Google DeepMind Research Intern page and reviews. <https://clear.glassdoor.nl/Interview/Google-DeepMind-Research-Intern-Interview-Questions-EI_IE1596815.0,15_KO16,31.htm>, <https://www.glassdoor.ie/Interview/Google-DeepMind-Interview-E1596815-RVW58586975.htm>, <https://www.glassdoor.com/Interview/Google-DeepMind-Interview-E1596815-RVW72610017.htm>, <https://www.glassdoor.co.in/Interview/Google-DeepMind-Interview-E1596815-RVW71106586.htm>, <https://www.glassdoor.com/Interview/Google-DeepMind-Interview-E1596815-RVW41884680.htm>. 2021–2023 (snippet).
12. Glassdoor, Google DeepMind Student Researcher reviews. <https://www.glassdoor.com.au/Interview/Google-DeepMind-Interview-E1596815-RVW96434887.htm>, <https://static.glassdoor.be/Interview/Google-DeepMind-Interview-E1596815-RVW99465599.htm>. 2025–2026 (snippet).
13. Google DeepMind blog, "My journey from DeepMind intern to mentor". <https://deepmind.google/discover/blog/my-journey-from-deepmind-intern-to-mentor>. Undated (snippet).
14. Google DeepMind blog, "Applying for technical roles". <https://deepmind.google/discover/blog/applying-for-technical-roles>. Undated (snippet).
15. 1point3acres, Google DeepMind Research Scientist onsite experience. <https://www.1point3acres.com/interview/thread/1186080>. Undated (snippet).
16. Glassdoor, Meta Research Intern questions. <https://static-pc.glassdoor.fr/Interview/Meta-Research-Intern-Interview-Questions-EI_IE40772.0,4_KO5,20_IP3.htm>. December 2025 report (snippet).
17. Glassdoor, Meta Research Scientist Intern questions. <https://static.glassdoor.com.ar/Interview/Meta-Research-Scientist-Intern-Interview-Questions-EI_IE40772.0,4_KO5,30_IP9.htm>, <https://static-pc.glassdoor.fr/Interview/Meta-Research-Scientist-Intern-Interview-Questions-EI_IE40772.0,4_KO5,30_IP2.htm>. February 2023 report (snippet).
18. Glassdoor, Meta research-intern and research-scientist reviews. <https://www.glassdoor.ca/Interview/Meta-Interview-E40772-RVW94704627.htm>, <https://www.glassdoor.co.nz/Interview/Meta-Interview-E40772-RVW97460370.htm>, <https://www.glassdoor.co.in/Interview/Meta-Interview-E40772-RVW76926865.htm>, <https://www.glassdoor.com/Interview/Meta-Interview-E40772-RVW86235295.htm>, <https://www.glassdoor.com/Interview/Meta-Interview-E40772-RVW103016814.htm>, <https://www.glassdoor.co.nz/Interview/Meta-Interview-E40772-RVW98875976.htm>. 2023–2026 (snippet).
19. Glassdoor, Meta Research Scientist questions. <https://static-pc.glassdoor.fr/Interview/Meta-Research-Scientist-Interview-Questions-EI_IE40772.0,4_KO5,23_IP2.htm>. Various years (snippet).
20. Glassdoor, Meta Research Intern and Computer Vision AI Research Intern pages. <https://static-pc.glassdoor.de/Interview/Meta-Research-Intern-Interview-Questions-EI_IE40772.0,4_KO5,20.htm>, <https://www.glassdoor.com/Interview/Meta-Computer-Vision-AI-Research-Intern-Interview-Questions-EI_IE40772.0,4_KO5,39.htm>, <https://www.glassdoor.com/Interview/Meta-Interview-E40772-RVW75782100.htm>. Undated (snippet).
21. Blind, "Meta Research Intern Interviews". <https://www.teamblind.com/post/meta-research-intern-interviews-fqwzuaj0>. 2023-12-20 (full).
22. Blind, "Research Scientist, Machine Learning (PhD) Interview - Meta". <https://www.teamblind.com/post/research-scientist-machine-learning-phd-interview-meta-gbxha4qh>. 2024-10-26 (full).
23. Reddit (snapshot), "[Meta] ML Research Scientist Interview Experience (New 'Coding with AI' Round)". <https://reddit.sentinel-team.org/posts/1r37w7q/snapshots/2026-02-13T07%3A00%3A23.975077Z>. Early 2026 (full).
24. Blind, "Meta: AI research EM Interview: AI Design, AI research Design". <https://www.teamblind.com/post/what-the-f-meta-ai-research-em-interview-ai-design-ai-research-design-f1ttrnh8>. 2024-03-09 (full).
25. Blind, "ML coding interview at OpenAI and Meta GenAI". <https://www.teamblind.com/post/ml-coding-interview-at-openai-and-meta-genai-dhq0gysn>. 2025-01-16 (full).
26. Blind, "FANG research internships during PhD". <https://www.teamblind.com/post/fang-research-internships-during-phd-vrqnf2m2>. 2022-02-20 (full).
27. Blind, Meta research-intern decision and return-offer threads. <https://www.teamblind.com/post/internship-offer-chance-at-meta-after-one-week-qve8kei6>, <https://www.teamblind.com/post/Return-Internship-Offer-for-Research-Scientist-Intern-ATggFkWD>. Undated (snippet).
28. PracHub, Meta machine-learning questions (MLE and DS tags). <https://prachub.com/companies/meta/categories/machine-learning>. 2025–2026 (full).
29. 1point3acres, Meta Reality Labs PhD research intern. <https://www.1point3acres.com/bbs/thread-960664-1-1.html>. About 2022–2023 (snippet).
30. 1point3acres, Meta research scientist interview, April. <https://www.1point3acres.com/bbs/thread-906962-1-1.html>. 2022 (snippet).
31. 1point3acres, Meta FAIR research intern with timeline. <https://www.1point3acres.com/bbs/thread-1097553-1-1.html>. 2024 (snippet).
32. 1point3acres, Meta research-scientist PhD intern and coding-screen threads. <https://www.1point3acres.com/bbs/thread-1105125-1-1.html>, <https://www.1point3acres.com/bbs/thread-1046619-1-1.html>. About 2024–2025 (snippet).
33. Glassdoor, NVIDIA Research Scientist Intern questions. <https://clear.glassdoor.nl/Interview/NVIDIA-Research-Scientist-Intern-Interview-Questions-EI_IE7633.0,6_KO7,32.htm>. December 2022 report (snippet).
34. Glassdoor, NVIDIA review RVW96839348. <https://clear.glassdoor.nl/Interview/NVIDIA-Interview-E7633-RVW96839348.htm>. January 2026 (snippet).
35. Glassdoor, NVIDIA review RVW97047784. <https://www.glassdoor.ca/Interview/NVIDIA-Interview-E7633-RVW97047784.htm>. April 2025 (snippet).
36. Glassdoor, NVIDIA Research Internship questions and reviews. <https://clear.glassdoor.nl/Interview/NVIDIA-Research-Internship-Interview-Questions-EI_IE7633.0,6_KO7,26.htm>, <https://www.glassdoor.com/Interview/NVIDIA-Interview-E7633-RVW103044139.htm>, <https://www.glassdoor.co.nz/Interview/NVIDIA-Interview-E7633-RVW102538139.htm>. 2020–2026 (snippet).
37. Glassdoor, NVIDIA Deep Learning Intern questions and review. <https://www.glassdoor.ca/Interview/NVIDIA-Deep-Learning-Intern-Interview-Questions-EI_IE7633.0,6_KO7,27.htm>, <https://www.glassdoor.ca/Interview/NVIDIA-Interview-E7633-RVW58641382.htm>. 2024–2025 (snippet).
38. Jointaro, NVIDIA Deep Learning Intern (Canada). <https://www.jointaro.com/interviews/companies/nvidia/experiences/deep-learning-intern-canada-november-12-2024-no-offer-positive-8af2f65a>. 2024-11-12 (full).
39. Jointaro, NVIDIA Machine Learning Intern (US). <https://www.jointaro.com/interviews/companies/nvidia/experiences/machine-learning-intern-united-states-december-4-2024-accepted-offer-positive-f72dc535/>. 2024-12-04 (full).
40. Blind, NVIDIA research-intern threads. <https://www.teamblind.com/post/NVIDIA-Research-Intern---Interview-Process-Gmr8wG85>, <https://www.teamblind.com/post/nvidia-interview-help-please-a2wdvp2d>, <https://www.teamblind.com/post/terrible-intern-interview-experience-at-nvidia-vkoo62xn>. 2023 and undated (snippet).
41. PracHub, NVIDIA machine-learning questions. <https://prachub.com/companies/nvidia/categories/machine-learning>. 2025–2026 (full; the research-intern item was truncated).
42. 1point3acres, NVIDIA research intern technical phone screen. <https://www.1point3acres.com/interview/thread/1189479>. 2026-09-14 (snippet).
43. Glassdoor, Adobe Research Intern pages and reviews. <https://static.glassdoor.de/Interview/Adobe-Research-Intern-Interview-Questions-EI_IE1090.0,5_KO6,21_IP2.htm>, <https://www.glassdoor.com/Interview/Adobe-Interview-E1090-RVW4808805.htm>, <https://www.glassdoor.sg/Interview/Adobe-Interview-E1090-RVW31216631.htm>, <https://www.glassdoor.co.uk/Interview/Adobe-Interview-E1090-RVW58089576.htm>, <https://static-pc.glassdoor.fr/Interview/Adobe-Interview-E1090-RVW101456095.htm>. 2013–2025 (snippet).
44. Glassdoor, Adobe research-scientist-intern reviews. <https://www.glassdoor.com/Interview/Adobe-Interview-E1090-RVW92950333.htm>, <https://www.glassdoor.ca/Interview/Adobe-Interview-E1090-RVW94560054.htm>, <https://www.glassdoor.ie/Interview/Adobe-Interview-E1090-RVW82635617.htm>, <https://www.glassdoor.ca/Interview/Adobe-Interview-E1090-RVW73331213.htm>, <https://www.glassdoor.co.nz/Interview/Adobe-Interview-E1090-RVW83540032.htm>, <https://www.glassdoor.co.uk/Interview/Adobe-Interview-E1090-RVW103060096.htm>, <https://static.glassdoor.at/Interview/Adobe-Research-Intern-Interview-Questions-EI_IE1090.0,5_KO6,21_IP5.htm>. 2023–2025 (snippet).
45. Jointaro, Adobe Machine Learning Research Intern. <https://www.jointaro.com/interviews/companies/adobe/experiences/machine-learning-research-intern-united-states-november-1-2021-accepted-offer-positive-2590d478/>. 2021-11 (full).
46. 1point3acres, Adobe Research internship interview experience. <https://www.1point3acres.com/interview/thread/1077093>. 2024 (snippet).
47. 1point3acres, Adobe tag page (research scientist intern 2021, PhD research intern 2022). <https://www.1point3acres.com/bbs/tag/adobe-1016-18.html>. 2021–2022 (snippet).
48. Glassdoor, Apple reviews describing a 30-minute research talk. <https://www.glassdoor.com.hk/Interview/Apple-Interview-E1138-RVW66240619.htm>, <https://www.glassdoor.com/Interview/Apple-Interview-E1138-RVW85642157.htm>. Undated (snippet).
49. Jointaro, Apple Machine Learning Intern. <https://www.jointaro.com/interviews/companies/apple/experiences/machine-learning-intern-united-states-february-4-2024-no-offer-neutral-dce1c499>. 2024-02 (full).
50. Jointaro, Apple Machine Learning Intern (NLP and speech team). <https://www.jointaro.com/interviews/companies/apple/experiences/machine-learning-intern-united-states-february-1-2023-no-offer-positive-69dfa35d>. 2023-02 (full).
51. Jointaro, Apple Machine Learning Intern (Culver City). <https://www.jointaro.com/interviews/companies/apple/experiences/machine-learning-intern-culver-city-ca-may-6-2024-no-offer-positive-473cd940>. 2024-05 (full).
52. Jointaro, Apple Machine Learning Intern (Sunnyvale). <https://www.jointaro.com/interviews/companies/apple/experiences/machine-learning-intern-sunnyvale-california-february-1-2025-accepted-offer-positive-d9e0913c>. 2025-02 (full).
53. 1point3acres, Apple AIML summer internship video interview. <https://www.1point3acres.com/interview/thread/1113143>. 2025 (snippet).
54. Glassdoor, Apple Machine Learning Research Intern page and reviews. <https://clear.glassdoor.nl/Interview/Apple-Machine-Learning-Research-Intern-Interview-Questions-EI_IE1138.0,5_KO6,38.htm>, <https://www.glassdoor.sg/Interview/Apple-Interview-E1138-RVW103104655.htm>, <https://www.glassdoor.co.uk/Interview/Apple-Interview-E1138-RVW74485718.htm>. Undated (snippet).
55. Glassdoor, Microsoft Research Intern pages and reviews. <https://static.glassdoor.at/Interview/Microsoft-Research-Intern-Interview-Questions-EI_IE1651.0,9_KO10,25_IP2.htm>, <https://www.glassdoor.com/Interview/Microsoft-Interview-E1651-RVW101009814.htm>. 2015–2026 (snippet).
56. PracHub, "Microsoft Intern Applied Scientist: Three Back-to-Back Virtual Onsites". <https://prachub.com/interview-experiences/microsoft-intern-applied-scientist-interview-experience-three-back-to-back-virtual-onsites-in-one-day>. 2026-02 (full).
57. Aced (formerly Exponent), Microsoft Applied Scientist interview guide. <https://www.tryexponent.com/guides/microsoft-applied-scientist-as-interview>. Undated (snippet; a guide, not a first-hand report).
58. Glassdoor, Amazon Applied Scientist (computer vision) pages and reviews. <https://www.glassdoor.co.uk/Interview/Amazon-Applied-Scientist-Computer-Vision-Interview-Questions-EI_IE6036.0,6_KO7,40.htm>, <https://www.glassdoor.com/Interview/Amazon-Interview-E6036-RVW44153298.htm>, <https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW29201660.htm>, <https://www.glassdoor.com/Interview/Amazon-Interview-E6036-RVW103606232.htm>. 2019–2026 (snippet).
59. LeetCode Discuss, Amazon research and applied-scientist intern posts. <https://leetcode.com/discuss/interview-experience/1826385/>, <https://leetcode.com/discuss/post/8530116/amazon-applied-scientist-intern-6-months-hdbp/>. 2022 and 2026 (snippet).
60. 1point3acres, Amazon applied scientist intern (PhD, computer vision). <https://www.1point3acres.com/bbs/thread-594260-1-1.html>. 2020 (snippet).
61. 1point3acres, summary of ML research-intern interviews at Amazon, Meta, Apple and TikTok. <https://www.1point3acres.com/bbs/thread-1042225-1-1.html>. 2023 (snippet).
62. Glassdoor, Snap reviews and Research Engineer (computer vision) page. <https://clear.glassdoor.nl/Interview/Snap-Interview-E671946-RVW59663263.htm>, <https://www.glassdoor.co.in/Interview/Snap-Interview-E671946-RVW62577187.htm>, <https://clear.glassdoor.nl/Interview/Snap-Research-Engineer-Computer-Vision-Interview-Questions-EI_IE671946.0,4_KO5,38.htm>. 2017 and 2022 (snippet).
63. Glassdoor, ByteDance and TikTok reviews. <https://www.glassdoor.com/Interview/ByteDance-Interview-E1624196-RVW104672982.htm>, <https://www.glassdoor.com/Interview/ByteDance-Interview-E1624196-RVW74118351.htm>, <https://www.glassdoor.com/Interview/TikTok-Interview-E2230881-RVW96573809.htm>. Undated (snippet).
64. Jointaro, Qualcomm Machine Learning Research Intern. <https://www.jointaro.com/interviews/companies/qualcomm/experiences/machine-learning-research-intern-united-states-january-1-2024-no-offer-positive-7e5f92ab/>. 2024-01 (full).
65. Glassdoor, Qualcomm research and ML intern pages. <https://static.glassdoor.de/Interview/Qualcomm-Research-Intern-Interview-Questions-EI_IE640.0,8_KO9,24.htm>, <https://www.glassdoor.ca/Interview/Qualcomm-Machine-Learning-Engineer-Intern-Interview-Questions-EI_IE640.0,8_KO9,41.htm>, <https://www.glassdoor.sg/Interview/Qualcomm-Interview-E640-RVW93598584.htm>. 2012–2026 (snippet).
66. Glassdoor, Samsung Research America reviews. <https://www.glassdoor.com/Interview/Samsung-Research-America-Interview-E295294-RVW93370563.htm>, <https://www.glassdoor.ie/Interview/Samsung-Research-America-Interview-E295294-RVW83495595.htm>, <https://www.glassdoor.co.uk/Interview/Samsung-Research-America-Interview-E295294-RVW10468112.htm>, <https://www.glassdoor.co.uk/Interview/Samsung-Research-America-Interview-E295294-RVW17051603.htm>. 2016–2024 (snippet).
67. Jointaro, Netflix Machine Learning Intern. <https://www.jointaro.com/interviews/companies/netflix/experiences/machine-learning-intern-united-states-february-1-2025-accepted-offer-positive-fe5e8c4f/>. 2025-02 (full).
68. 1point3acres and Glassdoor, Netflix ML research internship. <https://www.1point3acres.com/interview/thread/1118026>, <https://www.glassdoor.com.au/Interview/Netflix-Interview-E11891-RVW82601099.htm>. Undated (snippet).
69. Aced (formerly Exponent), OpenAI Research Engineer interview experience. <https://www.aced.io/experiences/openai-machine-learning-engineer-interview-c7b6a2>. About 2025–2026 (full).
70. PracHub, OpenAI Research Scientist interview with a card-game design round. <https://prachub.com/interview-experiences/openai-research-scientist-interview-experience-new-card-game-coding-round-that-tests-extensible-ood-design>. 2026-09 (full).
71. Glassdoor, OpenAI Research Scientist questions. <https://static.glassdoor.ch/Interview/OpenAI-Research-Scientist-Interview-Questions-EI_IE2210885.0,6_KO7,25.htm>. Undated (snippet).
72. tianpan.co forum, "Collecting OpenAI Interview Questions - 2025". <https://tianpan.co/forum/t/collecting-openai-interview-questions-2025-edition-share-your-experience/38?page=3>. 2025-09-21 (full; possibly second-hand).
73. Glassdoor and interview guides, Anthropic research roles. <https://static.glassdoor.ie/Interview/Anthropic-Research-Engineer-Scientist-Interview-Questions-EI_IE8109027.0,9_KO10,37.htm>, <https://www.tryexponent.com/blog/anthropic-interview-process>, <https://www.cleverprep.com/companies/anthropic/research-scientist>. 2024–2026 (snippet; partly secondary).
74. PracHub, Runway Machine Learning Engineer interview experience. <https://prachub.com/interview-experiences/runway-machine-learning-engineer-interview-experience-two-phone-screens-rejected-in-ml-system-design>. 2025-08 (full).
75. Glassdoor, Runway reviews. <https://www.glassdoor.com/Interview/Runway-NYC-Interview-E5638023-RVW71894353.htm>, <https://www.glassdoor.co.in/Interview/Runway-NYC-Interview-E5638023-RVW104089250.htm>, <https://www.glassdoor.co.uk/Interview/Runway-NYC-Interview-E5638023-RVW104047694.htm>. 2025–2026 (snippet).
76. Glassdoor, Luma AI questions and review. <https://www.glassdoor.com/Interview/Luma-AI-Interview-Questions-E8587007.htm>, <https://www.glassdoor.co.uk/Interview/Luma-AI-Interview-E8587007-RVW103600503.htm>. 2025–2026 (snippet).
77. Substack, "AI Research Internship Hunt as a CS PhD Student". <https://yongzx.substack.com/p/ai-research-internship-search-as>. Spring 2024 (full).
78. Rora, "2025 Technical Interview Guide for AI Researchers". <https://teamrora.com/post/the-2025-technical-interview-guide-for-ai-researchers>. 2025 (snippet).
79. Attention-implementation question banks tagged ByteDance, Apple and Uber. <https://prachub.com/interview-questions/implement-multi-head-self-attention-from-scratch-with-projections-and-masking>, <https://dataford.io/questions/implement-a-multi-head-attention-mechanism-from-scratch-in-pytorch>, <https://www.1point3acres.com/interview/thread/1168284>. Undated (snippet).
80. Nowcoder, summary of AIGC vision interview questions across several large companies. <https://www.nowcoder.com/discuss/730099920601907200>. 2025-03-14 (full).
81. Nowcoder, Kuaishou multimodal LLM intern, round 1 (Kling video-generation team). <https://www.nowcoder.com/feed/main/detail/58f352c813e44f00a2bd3f0c68df9b73>. About 2025-03-20 (full).
82. Nowcoder, ByteDance CV search early campus round 1. <https://www.nowcoder.com/discuss/520573268923826176>. 2023-08-13 (full).
83. Nowcoder, Tencent Youtu computer-vision round 1. <https://www.nowcoder.com/feed/main/detail/8eb1af0a75b34d50b30fbea4149b7022>. 2024-04-02 (full).
84. Nowcoder, ByteDance AIGC daily intern. <https://www.nowcoder.com/feed/main/detail/579091c3eea64dc0a1a785dcb220435c>. 2024-12-10 (full).
85. Nowcoder, ByteDance multimodal LLM campus interviews, with two reposts by the same author. <https://www.nowcoder.com/feed/main/detail/ce451a35177e487f916fe34374ad9df8>, <https://www.nowcoder.com/feed/main/detail/5e1ebd8041044003a0b55469961a9b6f>, <https://www.nowcoder.com/feed/main/detail/8fcf609e1034407488070e4b5b6ed76e>. 2025-09-24, 2025-04-18, 2026-03-18 (full).
86. Nowcoder, ByteDance LLM algorithm round 3 with on-the-spot derivations. <https://www.nowcoder.com/discuss/946798743163088896>. 2026-11-02 (full for the visible part; rest paywalled).
87. Nowcoder, ByteDance algorithm intern rounds 1–3. <https://www.nowcoder.com/discuss/353156438598033408>. 2020-05-15 (full).
88. Nowcoder, vision algorithm intern on an image-generation team (company not named). <https://www.nowcoder.com/feed/main/detail/2b94140c0cc54718b4cd1c8ba02d35a2>. 2024-05-27 (full).
89. Nowcoder, algorithm intern interviews at JD, NIO, Baidu and Kunlun Wanwei. <https://www.nowcoder.com/feed/main/detail/c64ad6f358f04455b639e7952cb3b494>. 2024-03-08 (full).
90. Nowcoder, Baidu multimodal algorithm daily intern, round 2. <https://www.nowcoder.com/feed/main/detail/32dd1dff10bd4cce90d314be29793000>. 2026-04-14 (full).
91. Nowcoder, Alibaba Taotian algorithm interviews. <https://www.nowcoder.com/feed/main/detail/52ce1365203c41968a2d661d8d34f661>. 2024-05-02 (full).
92. Nowcoder, Alibaba Tongyi Lab summer intern. <https://www.nowcoder.com/feed/main/detail/ed5f9ca1295947a0aea44b92cda15ba3>. 2024-04-08 (full).
93. Nowcoder, Xiaomi AIGC algorithm engineer. <https://www.nowcoder.com/feed/main/detail/ae505961cae04205abc9448399a1da64>. 2025-01-22 (full).
94. Nowcoder, Xiaomi multimodal algorithm, round 1. <https://www.nowcoder.com/feed/main/detail/0c6da046c76341189fbb9c7b99b1bc03>. 2024-03-29 (full).
95. Nowcoder, Meituan computer-vision intern, round 1. <https://www.nowcoder.com/feed/main/detail/0d1475aec6ec4f4d92a16bc71d7bd7f9>. 2024-04-29 (full).
96. Nowcoder, Xiaohongshu summer intern, computer vision and multimodal. <https://www.nowcoder.com/discuss/612973780121477120>. 2024-04-25 (full).
97. Nowcoder, Tencent multimodal algorithm rounds 1–2. <https://www.nowcoder.com/feed/main/detail/51ecce470a9444d7b5e1cde1f038b72d>. 2026-10-05 (full).
98. Nowcoder, Tencent Music Entertainment summer intern (NLP research). <https://www.nowcoder.com/discuss/744319938189438976>. 2025-04-22 (full).
99. Nowcoder, Kuaishou multimodal LLM algorithm engineer, round 1. <https://www.nowcoder.com/discuss/942532364704534528>. 2026-10-21 (full).
100. Nowcoder, SenseTime multimodal LLM algorithm intern, round 1. <https://www.nowcoder.com/feed/main/detail/bc02d116feb54a46a8b85a46f620f9e6>. 2026-04-19 (full).
101. Nowcoder, Ant Group algorithm interviews (three technical rounds plus HR). <https://www.nowcoder.com/discuss/608236811491573760>. 2024-04-12 (full).
102. Nowcoder, Dewu AIGC algorithm daily intern. <https://www.nowcoder.com/feed/main/detail/2262d4a3647746f1b941ed237b89e4d7>. 2024-04-17 (full).
103. Nowcoder, Ele.me AIGC summer intern, round 1. <https://www.nowcoder.com/discuss/611682506647117824>. 2024-04-21 (full).
104. Nowcoder, G-bits AIGC algorithm intern. <https://www.nowcoder.com/feed/main/detail/f750e3920fa5428e90b4a7ebfb19239a>. 2024-08-12 (full).
105. Nowcoder, Megvii PhD algorithm researcher. <https://www.nowcoder.com/discuss/353158334100480000>. 2021-08-25 (full).
106. Nowcoder, computer-vision algorithm interviews at about 20 companies. <https://www.nowcoder.com/discuss/828122>. 2022-01-19 (full).
107. Nowcoder, image-processing and computer-vision algorithm interviews. <https://www.nowcoder.com/discuss/353155640900132864>. 2019-10-03 (full).
108. Nowcoder, image and computer-vision early-batch interviews. <https://www.nowcoder.com/discuss/353155460939325440>. 2021-08-19 (full).
109. Nowcoder, NetEase Games image algorithm, round 1. <https://www.nowcoder.com/discuss/762583>. 2021-09-29 (full).
110. Nowcoder, SenseTime CV algorithm researcher (three rounds plus HR). <https://www.nowcoder.com/discuss/353155147922612224>. 2019-07-22 (full).
111. Nowcoder, SenseTime Research vision algorithm researcher. <https://www.nowcoder.com/discuss/353155362536759296>. 2019-09-04 (full).
112. Nowcoder, advice on image-algorithm campus recruiting. <https://www.nowcoder.com/discuss/353155828444241920>. 2019-10-28 (full).
113. Nowcoder, AIGC algorithm engineer question list, 2025 edition. <https://www.nowcoder.com/discuss/741096126551457792>. 2025-04-14 (full; curated).
114. GitHub, WeThinkIn AIGC-Interview-Book. <https://github.com/WeThinkIn/AIGC-Interview-Book>. Accessed 2026-11-08 (full for selected files; mostly curated, with some reposted interview reports).
115. Nowcoder, vision algorithm intern interviews at Meituan, NetEase, Ctrip, Hikvision and Huawei. <https://www.nowcoder.com/discuss/353156432784728064>. 2020-05-13 (full).
116. Zhihu, diffusion and AIGC high-frequency interview questions. <https://zhuanlan.zhihu.com/p/1894357747270665620>. About 2025 (snippet; curated).
117. CSDN, diffusion and AIGC interview question summary. <https://blog.csdn.net/2401_84033492/article/details/147162622>. About 2025 (snippet; curated).
118. CSDN, Kuaishou AIGC algorithm interview questions. <https://blog.csdn.net/m0_59596990/article/details/140566133>. About 2024 (snippet).
119. Zhihu, Kuaishou large-model interview. <https://zhuanlan.zhihu.com/p/1928556109037281822>. About 2025 (snippet).
120. Zhihu, Baidu multimodal high-frequency interview questions. <https://zhuanlan.zhihu.com/p/713494385>. About 2024 (snippet).
121. Nowcoder, Kuaishou Kling policy-gradient role, round 1. URL not recorded. Undated (snippet).
122. Nowcoder, Tencent Hunyuan large-model interviews. <https://www.nowcoder.com/feed/main/detail/42fba762c2374372bcae72245f45f942>. About 2025 (snippet).
123. Nowcoder, Insta360 multimodal LLM algorithm engineer, round 1. <https://www.nowcoder.com/discuss/858836573159206912>. About 2026 (snippet).
124. Nowcoder, DeepSeek LLM algorithm written test. <https://www.nowcoder.com/discuss/913482312753418240>. 2026-07 (snippet).
125. Nowcoder, Xiaohongshu algorithm rounds 1–4 plus HR. <https://www.nowcoder.com/discuss/734876>. 2021-09-07 (full).
