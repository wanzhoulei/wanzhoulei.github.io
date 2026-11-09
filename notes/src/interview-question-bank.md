---
title: "Research Interview Question Bank"
subtitle: "Questions reported in research-scientist-intern interviews for generative modeling and computer vision, across companies, with model answers."
order: 0
updated: 2026-11-08
---

This page collects questions that candidates reported from research-scientist-intern interviews in generative modeling and computer vision, with a short model answer for each. It was compiled from public interview reports on Glassdoor, Blind, Reddit, 1point3acres, Nowcoder, Zhihu, Jointaro, PracHub and similar sites, mostly posted between 2019 and 2026. Many reports were visible only as search-engine snippets or login-wall summaries, so dates and roles are approximate, and every question below is paraphrased. Chinese reports usually come from "algorithm intern" roles on AIGC, multimodal and vision teams rather than from programs titled research intern; some English reports are from adjacent roles (ML intern, applied scientist intern, a few full-time research roles), and the badges say so.

**How to read the badges.** A question with an `asked` badge appeared in at least one report; the badge names the companies and, when the theme recurred, the approximate number of independent reports. A badge means the question has been asked somewhere, not that a company asks it every time. A few badges instead say "curated" (a blogger-written question list), "job ad", "advice thread" or "prep guide"; they name where the question came from, not an interview report. Questions without a badge were added because they test the same material; several of them come from curated Chinese AIGC question lists, which are written by bloggers and are weaker evidence than first-hand reports. The numbers in square brackets point to the [Sources](#sources) list at the end.

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

::: {.qa asked="Meituan CV intern, Nowcoder 2024"}
**Your paper was rejected. What did the reviewers object to, and what will you do with it now?**

The interviewer is checking whether you can separate valid criticism from noise and turn it into a plan. Cover, in order:

1. The paper's claim in one sentence.
2. The two or three main objections, grouped by type (novelty relative to a named prior work, a missing baseline, weak evaluation, unclear writing), stated fairly and in the reviewers' terms.
3. Your assessment of each: which you agree with and why, and which came from a misreading that your writing allowed.
4. What has changed since: the new experiment, baseline or rewrite for each objection, with its result if you have it.
5. The plan: resubmit with those changes, or narrow or reframe the claim if an objection hit the core.

If a criticism was right, say how it changed your claim; that reads as maturity. Avoid blaming reviewers or the process, and avoid claiming every objection was wrong. Expect the follow-up "do you still believe the main result?", and answer it with evidence.
:::

::: {.qa asked="DJI, Nowcoder 2022"}
**How would you evaluate the academic value of your own paper?**

Use the criteria a careful reviewer would, and apply them to yourself honestly.

1. **Novelty**: the precise difference from the closest prior work, stated as a delta ("X did A under assumption B; we remove B").
2. **Significance**: who would change what they do because of it. A practitioner who would adopt the method, a researcher who would build on the finding, a benchmark that changes how a task is measured.
3. **Rigor**: how strong the evidence is: baseline strength, ablations, seeds and variance, whether the metric measures the claim.
4. **Generality**: whether the insight holds beyond your datasets and setting, or only there.
5. **Uptake**, stated modestly: follow-up work, independent reproductions, use of your code. Do not make citation counts the answer.
6. **Limitations**, and the one experiment that would most raise the paper's value.

A good closing line has the shape "the main value is X; the weakest part is Y." Avoid both inflation and false modesty.
:::

::: {.qa asked="ByteDance 2026, Tencent multimodal 2026 (Nowcoder)"}
**How did you build and clean your dataset, and how do you estimate how many good samples your filters threw away?**

Walk through the pipeline in order: sources and licensing (with provenance recorded), deduplication (exact hashes, then near-duplicates by perceptual hash or embedding, also against your evaluation sets), then each filter as rule, threshold, reason and share removed. Store scores, not pass or fail flags, so thresholds can move later.

To estimate false rejections, label a uniform random sample of $n$ items from a filter's rejected pool. If $k$ are good, $\hat p = k/n$, the number of good items lost is about $N_{\text{rej}}\hat p$, and a 95% interval is roughly $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$; if $k = 0$, the rule of three gives an upper bound near $3/n$. Since filters run in sequence, also score one random sample of raw data with every filter, so earlier filters do not hide later ones' losses. The decisive test is training small models at several thresholds under equal compute.

On "why not the strongest model, or an LLM or VLM API?": APIs judge semantic criteria well, but weigh cost at your scale, rate limits, silent model updates, licence terms and reproducibility. A common compromise: label a sample with the API, check agreement with humans, and distil it into a cheap classifier for the full corpus.

[more](video-generation.html)
:::

::: {.qa asked="Meta RS FT 2026, SenseTime 2026, Xiaohongshu 2024"}
**What was the hardest problem in your project, or in reproducing a method, and how did you solve it?**

Pick a technical difficulty, not a scheduling one, and tell it as a debugging story with evidence.

1. **Symptom**, with numbers: the loss diverged at a certain step, or you were 2 dB below the reported result.
2. **Hypotheses**, and the experiments that told them apart: overfitting a single batch, comparing intermediate outputs against a reference implementation layer by layer, swapping one component at a time.
3. **Root cause and fix**, with before-and-after results over several seeds.
4. **What you would do differently.**

For reproductions, the usual causes are worth naming: undocumented details (EMA, warmup, learning-rate schedule, data preprocessing and normalization range), evaluation-protocol mismatches (metric implementation, resizing before FID, PSNR on the Y channel versus RGB, border cropping, test split), seed variance larger than the gap you are chasing, and numerical precision (fp16 overflow, TF32 defaults, library version changes). A strong lesson is often "first reproduce the released checkpoint's numbers with my own evaluation code, then train." Avoid stories where luck or someone else solved it.
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

::: {.qa asked="Amazon AS, NVIDIA, Qualcomm, ByteDance, Taotian (5)"}
**Walk through the transformer architecture end to end. Why does it capture long-range dependencies better than an RNN?**

A transformer maps token vectors to contextual vectors through a stack of identical blocks: every position reads from every other position (self-attention), then each position is transformed on its own (an MLP), with a residual connection around both. Any two tokens interact in one attention step instead of through a chain of recurrent updates.

End to end, for a language model:

1. **Input.** An embedding table $E \in \mathbb{R}^{V \times d}$ maps token IDs to vectors. Attention is permutation-equivariant, so order is injected: learned or sinusoidal vectors added at the input, or rotary embeddings applied to queries and keys in every layer.
2. **Block.** In the now-standard pre-norm form, $h = x + \mathrm{MHA}(\mathrm{LN}(x))$, then $x' = h + \mathrm{FFN}(\mathrm{LN}(h))$ with $\mathrm{FFN}(u) = W_2\,\phi(W_1 u)$ of hidden width about $4d$. The original model was post-norm, $\mathrm{LN}(x + \mathrm{Sublayer}(x))$, which normalizes the residual path and is harder to train deep without warmup (Xiong et al., 2020).
3. **Masks.** A padding mask hides pad tokens; a causal mask stops position $t$ from attending to later positions.
4. **Layouts.** Encoder-decoder (the original model, T5): a bidirectional encoder and a causal decoder that cross-attends to it. Encoder-only (BERT, ViT): bidirectional, for representations. Decoder-only (GPT, Llama): one causal stack trained on next-token prediction, the LLM default because one objective on raw text both scales and generates.
5. **Output.** A final norm, a linear map to $V$ logits (sometimes tied to $E$), softmax, and cross-entropy against the input shifted by one; with teacher forcing all positions train in parallel.

**Why long range works better.** In an RNN, token $i$ reaches step $j$ through $j - i$ state updates, so $\partial h_j / \partial h_i$ is a product of $j - i$ Jacobians that tends to shrink or grow geometrically, and the whole prefix must fit in a fixed-size state. In self-attention every pair of positions is connected within one layer (maximum path length $O(1)$ versus $O(n)$ for recurrence, Table 1 of Vaswani et al., 2017), lookup is by content rather than distance, and past tokens are kept individually. Training is also parallel across positions instead of $O(n)$ sequential steps, which made very large corpora practical.

**Costs.** Attention needs $O(n^2 d)$ compute and, unless tiled, $O(n^2)$ memory per layer; the KV cache grows with context while an RNN's state does not; and with no built-in bias toward order or locality, positions must be encoded and generalization past the trained length is not automatic.

[more](vision-backbones.html)
:::

::: {.qa asked="Meta research round, Glassdoor (topic only)"}
**What are the main architectural choices in Llama, and why were they made?**

Llama is a decoder-only transformer built from changes that had already worked elsewhere. The more distinctive decision is training relatively small models on far more tokens than is compute-optimal, so they are cheap to serve.

Architecture (LLaMA, Touvron et al., 2023; kept in Llama 2 and 3):

- **Pre-normalization with RMSNorm.** Each sub-layer normalizes its input rather than its output, which leaves the residual path an identity and stabilizes training. RMSNorm divides by the root mean square without subtracting the mean, which is cheaper than LayerNorm.
- **SwiGLU feed-forward**, $W_2\big(\mathrm{SiLU}(W_1 x) \odot W_3 x\big)$. Gated MLPs gave lower perplexity than ReLU or GELU MLPs at equal parameter count (Shazeer, 2020); with three matrices the hidden width is $\tfrac{2}{3} \cdot 4d$ to match a $4d$ two-matrix MLP.
- **Rotary position embeddings** in every layer instead of absolute position embeddings, so attention scores depend on relative offsets. Llama 3 raises the RoPE base frequency to 500,000 for longer contexts.
- **No bias terms** in the linear layers of the reference implementation.
- **Grouped-query attention.** Llama 2 uses it only in the 34B and 70B models; Llama 3 uses 8 key-value heads at all sizes (8B, 70B, 405B). It shrinks the KV cache and speeds up decoding at a small quality cost.

**Tokenizer.** LLaMA and Llama 2 use SentencePiece BPE with 32k tokens, split numbers into single digits and fall back to bytes for unknown UTF-8. Llama 3 moves to 128K tokens (100K from tiktoken plus 28K for non-English text), which raised English compression from 3.17 to 3.94 characters per token, so the same compute covers more text.

**Data and scale.** LLaMA trained the 7B and 13B models on 1.0T tokens and the 33B and 65B models on 1.4T. The paper notes that Hoffmann et al. recommend a 10B model on 200B tokens, yet its 7B model was still improving after 1T, and LLaMA-13B outperformed GPT-3 on most benchmarks. Llama 2 used 2T tokens and doubled the context to 4k. Llama 3's 405B model used 15.6T tokens, near the optimum of the authors' own scaling laws, while its smaller models were trained far beyond it. Training past the compute-optimal point costs more training compute per unit of loss but pays off when the model serves many queries. Llama 3 also stays dense rather than mixture-of-experts, for training stability, and masks attention between documents packed into one sequence.
:::

::: {.qa asked="Apple ML research intern (NLP team), Jointaro 2023"}
**Explain BERT: its architecture, pretraining objectives, and how it differs from GPT.**

BERT (Devlin et al., 2018) is a transformer encoder pretrained to recover masked tokens from context on both sides, then fine-tuned with a small task head. GPT is a transformer decoder trained to predict the next token from the left context only.

**Architecture.** BERT-Base has 12 layers, hidden size 768 and 12 heads (110M parameters), chosen to match the original GPT's size; BERT-Large has 24 layers, hidden size 1024 and 16 heads (340M). Inputs are WordPiece tokens (30,000 vocabulary), up to 512 positions, with `[CLS]` at the start and `[SEP]` after each segment. Each input vector is the sum of a token, a learned position and a segment (A or B) embedding. Attention is unmasked.

**Masked language modeling.** Choose 15% of positions; replace 80% of those with `[MASK]`, 10% with a random token, and leave 10% unchanged; predict the original tokens with cross-entropy at the chosen positions only. The split exists because `[MASK]` never appears during fine-tuning; since the model cannot tell which tokens were altered, it must keep a good representation of every token. The cost is that only 15% of positions give a loss, a weaker signal per sequence than next-token prediction.

**Next-sentence prediction.** Half the time segment B follows A in the corpus, otherwise it is a random sentence; a classifier on the `[CLS]` output predicts which. RoBERTa (Liu et al., 2019) found that removing NSP matched or slightly improved downstream results, and that BERT was undertrained: dynamic masking, more data, larger batches and longer training helped.

**Fine-tuning.** Add one output layer, on `[CLS]` for classification or on every token for tagging and span extraction, and train all weights.

**Versus GPT.** A causal mask makes every position a training target and makes generation natural, left to right. Bidirectional context gives BERT stronger representations for understanding tasks at the same size, but MLM does not define a left-to-right factorization of the sequence probability, so BERT does not generate text directly. BERT also learns `[CLS]`, `[SEP]` and segment embeddings during pretraining, whereas the original GPT added them only for fine-tuning. Today encoders are used mostly for embeddings, retrieval and classification, and decoder-only models for generation.
:::

::: {.qa asked="Amazon AS intern, Glassdoor 2026"}
**What does a tokenizer do, and how does byte-pair encoding work?**

A tokenizer maps text to integer IDs from a fixed vocabulary, and back. Subword tokenizers sit between characters (long sequences) and words (huge vocabulary, unknown words): frequent strings become single tokens and rare words split into pieces, so nothing is out of vocabulary.

**BPE training** (Sennrich et al., 2016, adapted from a compression algorithm):

1. Pre-split the corpus into words, count them, and write each word as a sequence of base symbols (characters or bytes).
2. Count adjacent symbol pairs, weighted by word frequency.
3. Merge the most frequent pair into a new symbol, record the merge, and rewrite the words.
4. Repeat until the vocabulary reaches its target size.

Example with word counts `cat` 6, `cats` 2, `hat` 4, `that` 3. The pair counts are (a, t) 15, (c, a) 8, (h, a) 7, (t, h) 3 and (t, s) 2, so the first merge is `at`. Then (c, at) has 8 and (h, at) has 7, so the next merges are `cat`, then `hat`. **Encoding** applies the merges to new text in the order they were learned: `chats` goes from c h a t s to c h at s to c hat s, giving [c, hat, s]. An unseen word still gets a segmentation.

**Variants.** Byte-level BPE (GPT-2) starts from the 256 byte values, so any string is encodable, and pre-splits with a regex so merges do not cross word and punctuation boundaries. WordPiece (BERT) picks merges by likelihood gain instead of raw frequency and, at inference, takes the longest matching vocabulary entry first. Unigram (Kudo, 2018) starts from a large vocabulary, prunes pieces whose removal costs the least corpus likelihood, and segments with Viterbi. SentencePiece runs BPE or Unigram on raw text with the space as a symbol, so no language-specific pre-tokenizer is needed.

**Vocabulary size.** A larger vocabulary gives shorter sequences (less attention compute, more text per context) but larger embedding and output matrices, and rare tokens receive few updates.

**Side effects.** Numbers can be chunked inconsistently, hiding digit alignment, which is why LLaMA splits numbers into single digits. Languages underrepresented in the tokenizer's training data need more tokens per word, so they cost more and fit less in context. The model sees tokens, not letters, which makes spelling and character counting hard, and " the" and "the" are different tokens, so a trailing space in a prompt can change the output.
:::

::: {.qa asked="Tencent multimodal, Nowcoder 2026"}
**How does an LLM turn hidden states into token probabilities, and how does decoding choose the next token?**

The final hidden state at position $t$, after a last normalization, is multiplied by an output matrix $W_U \in \mathbb{R}^{V \times d}$ to give one logit per vocabulary entry, and a softmax turns the logits into a next-token distribution. Decoding is the rule that picks a token from that distribution; greedy, beam, top-k and top-p use the same network and differ only in that rule.

**Probabilities.** $z_t = W_U h_t$ and $p(v \mid x_{<t}) = \mathrm{softmax}(z_t / T)_v$, with temperature $T = 1$ in training. The original transformer ties $W_U$ to the input embedding; LLaMA keeps them separate. By the chain rule, $\log p(x_{1:n}) = \sum_t \log p(x_t \mid x_{<t})$; we work in log space because a product of thousands of probabilities underflows. Training minimizes the mean negative log-probability, and perplexity is its exponential. Dividing logits by $T$ keeps the ranking but changes the shape: $T \to 0$ approaches the argmax, and $T > 1$ flattens the distribution.

**Decoding rules.**

- *Greedy*: the argmax at each step. Deterministic and cheap, but it does not find the most likely sequence and tends to repeat itself in open-ended text.
- *Beam search*: keep the $B$ partial sequences with the highest total log-probability. It suits tasks with nearly one right answer (translation, speech recognition). Summed log-probabilities favor short outputs unless length-normalized, and for open-ended text high-likelihood outputs tend to be bland and repetitive (Holtzman et al., 2019).
- *Top-k sampling*: renormalize over the $k$ most likely tokens. A fixed $k$ is too loose when the distribution is peaked and too tight when it is flat.
- *Top-p (nucleus) sampling*: keep the smallest set of tokens whose total probability reaches $p$, so the candidate set adapts to the model's confidence.

**Why the settings change diversity.** Sampling the full distribution gives diversity but also visits the low-probability tail, and that adds up over a long output: if the tail holds 5% of the mass at every step, the chance of never sampling it in 100 tokens is $0.95^{100} \approx 0.006$. Lower temperature and truncation cut the tail (more coherent, less diverse); higher temperature and larger $k$ or $p$ do the reverse.

```python
import torch

def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0):
    # logits: (V,) for one position; returns a token id
    if temperature == 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        p_sorted, idx = probs.sort(descending=True)
        drop = p_sorted.cumsum(0) - p_sorted >= top_p   # mass before this token already >= p
        p_sorted[drop] = 0.0
        probs = torch.zeros_like(probs).scatter(0, idx, p_sorted)
    return int(torch.multinomial(probs / probs.sum(), 1))
```

Per step this costs $O(V \log V)$ for the sort ($O(V)$ without top-p) and $O(V)$ memory.
:::

::: {.qa asked="Tencent multimodal, Nowcoder 2026"}
**What gating mechanisms appear in neural networks, and what does a gate buy you?**

A gate is a learned factor, usually a sigmoid in $(0, 1)$, sometimes a softmax over options or a scalar such as $\tanh(\alpha)$, that multiplies another signal and so decides how much of it passes. Gates give input-dependent multiplicative interactions, control over what is kept or overwritten, and a clean way to start a new path switched off.

Where they appear:

- **LSTM and GRU.** The LSTM computes forget, input and output gates $f_t, i_t, o_t = \sigma(\cdot)$ and updates $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$, $h_t = o_t \odot \tanh(c_t)$. The direct path through the cell has Jacobian $\mathrm{diag}(f_t)$, so with the forget gate near 1, information and gradients survive many steps. The GRU's update gate interpolates between the old state and a candidate.
- **Highway networks** (Srivastava et al., 2015): $y = H(x)\,T(x) + x\,(1 - T(x))$, with the gate's bias initialized negative so layers start by mostly carrying their input. ResNets drop the gate and keep a fixed identity path.
- **Gated linear units**: $(xW) \odot \sigma(xV)$ (Dauphin et al., 2016) and SwiGLU, $(\mathrm{SiLU}(xW) \odot xV)W_2$, which improved transformer feed-forward layers at equal parameter count (Shazeer, 2020) and is used in PaLM and Llama.
- **Channel gates**: Squeeze-and-Excitation rescales each feature map by a sigmoid of a small MLP on pooled features.
- **Mixture-of-experts routers**: a softmax with top-$k$ selection decides which experts process each token (Shazeer et al., 2017), so capacity grows without growing per-token compute; an auxiliary loss balances the load.
- **Conditioning gates.** DiT's adaLN-Zero regresses a per-channel gate from the timestep and class embedding and multiplies each residual branch by it; the regressing layer is zero-initialized, so every block starts as the identity. Flamingo inserts cross-attention layers into a frozen language model as $y \leftarrow y + \tanh(\alpha)\,\mathrm{xattn}(y, \text{visual})$ with a per-layer scalar $\alpha$ initialized at 0, so training starts from exactly the pretrained LM.

**What a gate buys:** input-dependent multiplicative routing that an additive layer can only approximate with more units; near-identity gradient paths when a carry gate is open; safe insertion of new modules into a pretrained network; and conditional computation when the gate is hard (top-$k$).

**Caveats.** A saturated sigmoid passes almost no gradient. Zero-initialize the gate or the branch, not both in a way that blocks learning: in Flamingo the gradient for $\alpha$ is $\tanh'(0) = 1$ times the branch output, which is nonzero. MoE routers can collapse onto a few experts.
:::

::: {.qa asked="Amazon AS intern 2026, Microsoft AS intern 2026"}
**Describe the LLM training pipeline after pretraining: what do SFT, RLHF with PPO, DPO and GRPO each do?**

SFT imitates demonstrations; RLHF learns a reward from human comparisons and maximizes it with PPO; DPO fits the same preference objective directly, with no reward model and no sampling; GRPO is online RL that replaces PPO's value network with a group baseline.

**SFT.** Train on (prompt $x$, response $y$) pairs in the chat template with cross-entropy on response tokens only, $-\sum_t \log\pi_\theta(y_t \mid x, y_{<t})$; masking the prompt stops the model from learning to write user turns. It is only as good as the demonstrations.

**Reward model.** From comparisons $(x, y_w, y_l)$, train a scalar-output model (typically the SFT model with a new head) under Bradley-Terry, $P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$, by minimizing $-\log\sigma(r_w - r_l)$.

**PPO.** Maximize $\mathbb{E}_{y \sim \pi_\theta}[r(x, y)] - \beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ with $\pi_{\text{ref}}$ the SFT model. In practice the KL is a per-token penalty $-\beta \log(\pi_\theta / \pi_{\text{ref}})$ added to the reward, advantages $A_t$ come from a learned value function, and the update maximizes $\mathbb{E}[\min(\rho_t A_t, \mathrm{clip}(\rho_t, 1 - \varepsilon, 1 + \varepsilon) A_t)]$ with $\rho_t = \pi_\theta / \pi_{\text{old}}$. The KL keeps the policy where the reward model is reliable. The cost is four models in memory (policy, reference, reward, value).

**DPO.** The KL-regularized objective equals $-\beta\,\mathrm{KL}\big(\pi \,\|\, \pi_{\text{ref}} e^{r/\beta} / Z\big) + \beta \log Z(x)$ with $Z(x) = \sum_y \pi_{\text{ref}}(y|x) e^{r(x,y)/\beta}$, so its maximizer is $\pi^* = \pi_{\text{ref}} e^{r/\beta} / Z$. Inverting, $r = \beta \log(\pi^* / \pi_{\text{ref}}) + \beta \log Z(x)$. Bradley-Terry uses only $r_w - r_l$, so $Z$ cancels and the policy is fit directly:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\Big[\log\sigma\Big(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\Big].
$$

It is offline and stable and needs only the policy and a frozen reference, but it learns only from the given pairs, and since the loss sees only the margin, the chosen response's likelihood can fall too.

**GRPO** (Shao et al., 2024). For each prompt, sample $G$ outputs, score them (often with a checkable reward such as a correct answer), and give every token of output $i$ the advantage $\hat A_i = (r_i - \mathrm{mean}(r)) / \mathrm{std}(r)$. The objective is PPO's clipped surrogate averaged over the group, with the KL to the reference added to the loss. The group mean replaces the value network, saving memory and avoiding a per-token critic trained from one end-of-sequence reward.

**Trade-offs.** PPO is the most general but expensive; DPO is simple but offline; GRPO is online, lighter than PPO, and suits verifiable rewards. All can over-optimize their reward, and $\beta$ limits how far the model moves.

[more](text-to-image-multimodal.html)
:::

::: {.qa asked="NVIDIA AI intern, Jointaro 2023"}
**How would you train an LLM to perform a given list of tasks?**

Treat it as a measurement problem first: define each task's metric and a held-out set, get a prompting baseline, and fine-tune only where the baseline falls short, adding data where the evaluation shows failures.

1. **Clarify.** What are the tasks, their input and output formats and metrics? Do latency or cost limits fix the model size? How much labeled data exists per task? Must the model keep its general abilities, and will one model serve every task?
2. **Baseline.** Prompt a strong instruction-tuned model zero- and few-shot, with retrieval if the tasks need knowledge. This sets the bar, sometimes solves easy tasks outright, and shows where training is needed.
3. **Data per task.** Convert existing labeled data to instruction format with several prompt templates per task, as in instruction tuning (FLAN, Wei et al., 2021); generate synthetic examples with a stronger model and filter them (exact-match checks, unit tests, a judge model, deduplication); write examples by hand for hard or subjective tasks; remove overlap with the evaluation sets.
4. **Mixture.** Set per-task sampling weights so large tasks do not swamp small ones (for example, cap each task's share), and mix in general instruction data to limit forgetting.
5. **Training.** SFT with loss on response tokens. Full fine-tuning has the most capacity; LoRA is cheaper, allows one adapter per task, and tends to learn less but forget less of the base model (Biderman et al., 2024). Use a small learning rate and a few epochs, and pick checkpoints on per-task validation.
6. **Evaluation.** Per-task metrics on held-out data (exact match, F1, pass@k, a calibrated judge model or humans), plus general benchmarks to catch regressions. Look at the worst task and at individual failures, not only the average, and feed them back into the data.
7. **Beyond SFT.** Where quality is subjective, collect preference pairs and use DPO; where correctness can be checked automatically (math, code, structured outputs), use RL with verifiable rewards such as GRPO.

What the interviewer is checking: that you start simple, fix data and metrics before methods, and plan for tasks interfering with each other.
:::

::: {.qa asked="Microsoft AS intern, PracHub 2026"}
**What is an LLM agent, and how would you build and evaluate one?**

An agent is an LLM in a loop: it chooses an action (a tool call, code to run, a click), observes the result, and continues until it decides the task is done. Compared with a single call, it has state, tools and many steps, so errors can compound and evaluation has to judge outcomes over whole trajectories.

**Components.**

- *Policy*: the LLM, with a system prompt that describes the tools. With function calling, tools are declared as schemas, the model emits a structured call, the runtime executes it, and the result is appended to the context.
- *Reasoning and acting*: ReAct (Yao et al., 2022) interleaves reasoning traces with actions, so the reasoning updates the plan and the actions fetch information.
- *Memory*: the context window for the current task (with summarization when it fills up), plus an external store of notes or past results retrieved when relevant.
- *Planning*: implicit step by step, an explicit plan then execution, or sub-agents for sub-tasks, with checks and retries.

**Building.** Start with one loop and a few well-documented tools; keep anything that needs no judgment in ordinary code; validate tool arguments; set step, time and cost limits; log full trajectories.

**Failure modes.** Errors compound: at 95% per-step reliability, a 20-step task succeeds about $0.95^{20} \approx 36\%$ of the time if errors are independent. Others: wrong tool arguments or invented tools, loops, losing the instructions as the context grows, declaring success early, and prompt injection from tool outputs such as web pages.

**Evaluation.** Measure task success on held-out tasks with automatic checks of the end state, for example tests that must pass, as in SWE-bench (Jimenez et al., 2023), or web tasks with checkable outcomes, as in WebArena (Zhou et al., 2023). Run each task several times and report the mean and the spread, since runs are stochastic and reliability matters as much as best-case success. Read and categorize failed trajectories, track cost (tokens, tool calls) and latency, and ablate tools and prompts.

**Safety.** Run tools in a sandbox with least-privilege credentials and network allowlists, require confirmation for irreversible actions, treat tool outputs as untrusted data rather than instructions, and keep audit logs.
:::

::: {.qa asked="MSR intern, ByteDance 2020, Meituan 2022"}
**Explain SGD with momentum, RMSProp and Adam. When would you choose each, and how do you control how fast Adam converges?**

Momentum averages gradient directions over time; RMSProp divides each coordinate by a running estimate of its gradient magnitude; Adam does both and corrects the bias from starting the averages at zero.

$$
\begin{aligned}
&\text{Momentum:} && v_t = \mu v_{t-1} + g_t, \quad \theta_t = \theta_{t-1} - \eta v_t \\
&\text{RMSProp:} && s_t = \rho s_{t-1} + (1-\rho) g_t^2, \quad \theta_t = \theta_{t-1} - \eta\, g_t / (\sqrt{s_t} + \epsilon) \\
&\text{Adam:} && m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \quad \theta_t = \theta_{t-1} - \alpha\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)
\end{aligned}
$$

Momentum's effective step is $\eta/(1-\mu)$ along consistent directions, while components that flip sign cancel. RMSProp's per-coordinate step is about $\eta$ whatever the gradient scale. **Bias correction**: from $m_0 = 0$, $m_t = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i$, whose weights sum to $1 - \beta_1^t$, so for stationary gradients $\mathbb{E}[m_t] = (1 - \beta_1^t)\,\mathbb{E}[g]$ and $\hat m_t = m_t/(1-\beta_1^t)$ removes the shrinkage; likewise $\hat v_t = v_t/(1-\beta_2^t)$. **AdamW** applies weight decay as $\theta \leftarrow \theta - \eta\lambda\theta$ outside the adaptive step; an L2 term added to the gradient gets divided by $\sqrt{\hat v}$, so weights with large gradients are barely regularized.

**Choosing.** SGD with momentum and a step or cosine schedule remains strong for CNN image classification and stores one state per parameter instead of two; Wilson et al. (2017) found adaptive methods generalizing worse than SGD on several benchmarks. AdamW is the default for transformers and diffusion models, whose blocks have very different gradient scales that no single global step size suits. RMSProp is now used less than Adam, which with $\beta_1 = 0$ is RMSProp plus bias correction.

**Controlling Adam's speed.** Ignoring $\epsilon$, each coordinate moves about $\alpha$ per step, so:

- the **learning rate and schedule** are the main lever; warmup avoids large steps while $v$ is poorly estimated;
- **$\beta_1$** sets inertia: higher is smoother but reacts more slowly;
- **$\beta_2$** sets the memory of gradient scale: 0.999 averages about 1000 steps and can lag a sudden rise in gradient scale, giving oversized steps and loss spikes, which is why many LLM recipes use 0.95;
- **$\epsilon$** is the gradient scale below which normalization stops; raising it makes small-gradient coordinates take smaller, SGD-like steps;
- **gradient clipping** caps the step during spikes.

To slow Adam down on purpose, for stability or gentle fine-tuning: lower the peak learning rate, lengthen warmup, raise $\beta_1$ or $\epsilon$, and clip.

[more](deep-learning-training.html)
:::

::: {.qa asked="ByteDance algorithm intern, Nowcoder 2020"}
**What causes vanishing and exploding gradients, and how do you handle them?**

Backpropagation multiplies one Jacobian per layer (or per time step in an RNN), so the gradient's norm changes geometrically with depth: if each factor shrinks it a little it vanishes, and if each grows it a little it explodes.

**Mechanism.** For $h_\ell = \phi(a_\ell)$ with $a_\ell = W_\ell h_{\ell-1}$,

$$
\frac{\partial\mathcal{L}}{\partial h_0} = J_1^\top J_2^\top \cdots J_L^\top\,\frac{\partial\mathcal{L}}{\partial h_L}, \qquad J_\ell = \mathrm{diag}\big(\phi'(a_\ell)\big)\,W_\ell .
$$

The norm of the product lies between $\prod_\ell \sigma_{\min}(J_\ell)$ and $\prod_\ell \sigma_{\max}(J_\ell)$ times $\|\partial\mathcal{L}/\partial h_L\|$, so a typical per-layer factor $\gamma$ compounds to about $\gamma^L$: $0.9^{50} \approx 0.005$ and $1.1^{50} \approx 117$. The causes:

- saturating activations: $\sigma' \le 1/4$, and $\tanh'$ is near 0 for large inputs;
- weight scale: an initialization that does not preserve variance, or weights that grow during training;
- RNNs: the same $W_{hh}$ appears at every step; Pascanu et al. (2012) show that if $|\phi'| \le \gamma$ and the largest eigenvalue magnitude of $W_{hh}$ is below $1/\gamma$, long-range gradient terms vanish, and exceeding $1/\gamma$ is necessary for them to explode.

**Diagnosis.** Log per-layer gradient norms. Vanishing shows as tiny norms in early layers and a loss that plateaus from the start; exploding as gradient-norm spikes, loss spikes, then NaNs.

**Fixes, and why they work.**

1. ReLU-family activations: derivative exactly 1 on active units.
2. Variance-preserving initialization (He, Xavier; orthogonal for RNNs), so each layer's Jacobian roughly preserves norms at step 0.
3. Normalization layers rescale activations at every layer, so drift in weight scale does not compound.
4. Residual connections: $h_{\ell+1} = h_\ell + F(h_\ell)$ has Jacobian $I + \partial F/\partial h$, which gives an identity path from the loss to every layer.
5. Gating in LSTMs and GRUs: the additive cell update gives the direct path Jacobian $\mathrm{diag}(f_t)$, close to the identity when the forget gate is open.
6. Global-norm clipping, $g \leftarrow g \min(1, c/\|g\|)$, bounds the step during spikes and keeps its direction.
7. Smaller learning rates with warmup, truncated backpropagation through time, and bf16 instead of fp16 to avoid overflow.

Clipping treats only explosions; vanishing gradients need changes to the architecture or the initialization.

[more](deep-learning-training.html)
:::

::: {.qa asked="ByteDance CV 2023, Horizon 2022 (Nowcoder)"}
**How do dropout and pooling layers behave in the forward and backward passes, and how does dropout change between training and inference?**

None of these layers has parameters, and their backward passes only route the upstream gradient: dropout through the same random mask used in the forward pass, max pooling to the input that won each window, and average pooling evenly across each window.

**Dropout (inverted).** Forward in training: draw $m_i \sim \mathrm{Bernoulli}(1-p)$ and output $y = m \odot x / (1-p)$, so $\mathbb{E}[y] = x$. Backward: for a fixed mask $y$ is linear in $x$, so $\partial\mathcal{L}/\partial x = m \odot (\partial\mathcal{L}/\partial y) / (1-p)$; the mask must be stored or regenerated from the same seed, and dropped units get zero gradient for that step. Inference: the identity, with no mask and no rescaling, because the $1/(1-p)$ was applied during training. The original formulation left training unscaled and multiplied by $1-p$ at test time instead; the two agree in expectation. Variances do not match between the modes, which can shift the statistics a later BatchNorm collected. Leaving dropout on at test time and averaging several passes (Monte Carlo dropout, Gal and Ghahramani, 2016) gives a cheap uncertainty estimate.

**Max pooling.** Forward: $y_j = \max_{i \in R_j} x_i$, storing the argmax $i^*(j)$ of each window. Backward: $\partial\mathcal{L}/\partial x_i = \sum_{j:\, i^*(j) = i} \partial\mathcal{L}/\partial y_j$, so the winner receives the whole gradient and the rest of the window receives zero (a subgradient; ties go to one index by convention).

**Average pooling.** Forward: $y_j = \frac{1}{|R_j|}\sum_{i \in R_j} x_i$. Backward: every input in window $j$ receives $\frac{1}{|R_j|}\,\partial\mathcal{L}/\partial y_j$; global average pooling gives each of the $HW$ positions $1/(HW)$ of the gradient.

With overlapping windows (for example $3 \times 3$ with stride 2), an input belongs to several windows and the gradients from all of them add up. Pooling behaves the same in training and inference.

```python
import numpy as np

def dropout_fwd(x, p, rng, train=True):
    if not train or p == 0:
        return x, None                                # inference: identity
    mask = (rng.random(x.shape) >= p) / (1 - p)      # keep w.p. 1-p, pre-scaled
    return x * mask, mask

def dropout_bwd(dy, mask):
    return dy if mask is None else dy * mask         # the same mask as forward

def maxpool1d_fwd(x, k, s):
    starts = np.arange(0, len(x) - k + 1, s)
    idx = starts + np.array([np.argmax(x[i:i + k]) for i in starts])
    return x[idx], idx                               # keep argmax indices

def maxpool1d_bwd(dy, idx, n):
    dx = np.zeros(n)
    np.add.at(dx, idx, dy)                           # shared argmax: gradients add
    return dx
```

Both passes are $O(n)$ in time; dropout stores an $n$-sized mask and max pooling one index per output.

[more](deep-learning-training.html)
:::

::: {.qa asked="Alibaba, Nowcoder 2022"}
**What is the difference between empirical and structural risk minimization, why does early stopping work, and what does information theory say about generalization?**

ERM picks the hypothesis with the lowest training loss in a fixed class; SRM also pays for the capacity of the class. Early stopping is a capacity control indexed by training time. Information-theoretic bounds say an algorithm generalizes when its output carries little information about its particular training sample.

**ERM.** Choose $h \in \mathcal{H}$ minimizing $\hat R_n(h) = \frac{1}{n}\sum_i \ell(h(x_i), y_i)$. This works when $\hat R_n$ is uniformly close to the true risk $R$ over $\mathcal{H}$, which needs $\mathcal{H}$ small relative to $n$; in a rich class ERM drives $\hat R_n$ down while $R$ stays high.

**SRM** (Vapnik). Take nested classes $\mathcal{H}_1 \subset \mathcal{H}_2 \subset \cdots$ of growing capacity and minimize training loss plus a capacity penalty. For 0-1 loss and VC dimension $d$, with probability at least $1-\delta$ every $h \in \mathcal{H}$ satisfies
$$
R(h) \le \hat R_n(h) + O\Big(\sqrt{\big(d\log(n/d) + \log(1/\delta)\big)/n}\Big),
$$
and SRM picks the class minimizing this bound. Regularized ERM, $\hat R_n(h) + \lambda\Omega(h)$, is the practical form, and tuning $\lambda$ or model size on validation data is SRM done empirically. For deep networks such bounds are vacuous, since networks can fit random labels (Zhang et al., 2016).

**Early stopping.** Stop when validation loss stops improving and restore the best checkpoint. The weights reachable within $t$ steps of the initialization form a set that grows with $t$, so $t$ indexes a nested family, as in SRM. For a quadratic loss with Hessian eigenvalues $\lambda_i$, gradient descent from 0 with step $\eta$ has fit the component along eigenvector $i$ by the factor $1 - (1 - \eta\lambda_i)^t$, while L2 with strength $\alpha$ fits it by $\lambda_i/(\lambda_i + \alpha)$; when $\eta\lambda_i \ll 1$ and $\lambda_i \ll \alpha$, both are about linear in $\lambda_i$ and agree for $\alpha \approx 1/(\eta t)$. High-curvature directions, which the data pin down, are fit early; low-curvature ones, where noise lives, late. For deep networks this is an analogy, and epoch-wise double descent can make the first validation minimum misleading.

**Information theory.** Xu and Raginsky (2017) treat the algorithm as a channel from the sample $S$ of $n$ i.i.d. points to the output $W$. If $\ell(w, Z)$ is $\sigma$-sub-Gaussian for every $w$ (a loss in $[a, b]$ qualifies with $\sigma = (b-a)/2$), then $|\mathbb{E}[R(W) - \hat R_S(W)]| \le \sqrt{2\sigma^2 I(S; W)/n}$. With $k$ hypotheses, $I(S; W) \le \log k$ recovers the classical finite-class rate; for deterministic training with continuous outputs $I(S; W)$ is typically infinite, so the bound suits noisy algorithms. The information-bottleneck account (Shwartz-Ziv and Tishby, 2017) proposed that networks fit and then compress their representations, driving generalization; Saxe et al. (2018) found compression depends on the nonlinearity (seen with tanh, not ReLU) and no clear causal link to generalization. Treat it as a lens, not an explanation.

[more](deep-learning-training.html)
:::

::: {.qa asked="ByteDance, Huawei, SenseTime (Nowcoder 2019–22)"}
**How do you handle class imbalance?**

Decide first what you are optimizing. A cross-entropy model is tuned for accuracy under the training class frequencies, so imbalance causes trouble mainly when the metric or the deployment distribution weights rare classes more than the training data does, and when rare classes have too few examples to learn from. Then change the data, the loss or the decision rule.

1. **Metrics.** Accuracy is misleading (99% by always predicting the majority class). Use per-class precision and recall, balanced accuracy (mean per-class recall), macro-F1 and PR-AUC, and look at the confusion matrix.
2. **Resampling.** Oversample rare classes (repeats can overfit; augment them), undersample frequent ones (throws data away), or sample batches class-balanced. Resample only the training split, after splitting; duplicating before the split leaks copies into validation. Keep the test set at the real distribution.
3. **Loss weighting.** Weight class $c$ in the cross-entropy by about $1/n_c$, or by the inverse of the "effective number" $(1 - \beta^{n_c})/(1 - \beta)$ of Cui et al. (2019), which saturates for large classes. Focal loss down-weights easy examples, which are mostly from the majority class.
4. **Prior correction.** A cross-entropy model estimates $p_{\text{train}}(y \mid x) \propto p(x \mid y)\,\pi_y$. If the deployment priors are $\pi'_y$ and $p(x \mid y)$ is unchanged (label shift), Bayes' rule gives $p'(y \mid x) \propto p_{\text{train}}(y \mid x)\,\pi'_y / \pi_y$, so subtract $\log\pi_y$ from the logits (and add $\log\pi'_y$). For balanced error this is post-hoc logit adjustment, $\arg\max_y f_y(x) - \tau\log\pi_y$, and Menon et al. (2020) show it is consistent; their training-time version adds $\tau\log\pi_y$ to the logits inside the softmax loss. In binary problems the same idea is threshold moving: choose the threshold on validation data for the target metric or cost.
5. **Decoupled training.** Kang et al. (2019) found that ordinary instance-balanced sampling learns the best representations; then retrain only the classifier with class-balanced sampling, or rescale the classifier's weight norms.
6. **More data.** Collect rare examples with active learning or targeted labeling; for extremely rare positives, consider an anomaly-detection framing.

**Pitfalls.** Reweighting and resampling distort the predicted probabilities, so undo them with the prior correction or recalibrate on a validation set drawn from the true distribution; do not tune thresholds on the test set; and report results per class, since an average hides the rare classes.
:::

::: {.qa asked="NVIDIA 2025, ByteDance MM 2025, Qualcomm"}
**Compare data, tensor and pipeline parallelism. How do DeepSpeed and Megatron-LM differ?**

Data parallelism splits the batch and replicates the model; tensor parallelism splits each layer's matrices; pipeline parallelism splits the layers into stages. They trade memory, communication volume and idle time differently, so large runs combine them.

**Data parallelism.** Each of $N$ GPUs holds a full replica and runs its own micro-batch; gradients are averaged with an all-reduce (about $2\Psi$ elements per GPU per step for $\Psi$ parameters), overlapped with backward. Per-GPU memory does not shrink. **ZeRO** and **FSDP** remove the redundancy by sharding optimizer states, then gradients, then parameters across the data-parallel ranks; full sharding brings model states to $16\Psi/N$ bytes per GPU for mixed-precision Adam at about $3\Psi$ communication, since weights are all-gathered in both forward and backward. Every GPU still runs whole layers.

**Tensor parallelism** (Megatron-LM, Shoeybi et al., 2019). In the MLP $Y = \mathrm{GeLU}(XA)\,B$, split $A$ by columns, $A = [A_1, A_2]$, so each GPU computes $\mathrm{GeLU}(XA_i)$ without communication (the nonlinearity is elementwise). Split $B$ by rows, so $Y = \mathrm{GeLU}(XA_1)B_1 + \mathrm{GeLU}(XA_2)B_2$ needs one all-reduce. Attention is split by heads the same way, giving two all-reduces forward and two backward per layer. Weights, compute and the activations inside each block are divided across GPUs, but the communication is frequent and blocking.

**Pipeline parallelism.** Contiguous layers go to $K$ stages and the batch to $M$ micro-batches. With GPipe's schedule (all forwards, then all backwards) the idle fraction is $(K-1)/(M+K-1)$ and activations for all $M$ micro-batches are kept. The 1F1B schedule alternates one forward and one backward per stage after a warmup: the same bubble, but at most $K$ micro-batches in flight. Giving each device $v$ interleaved chunks cuts the bubble by $v$ at $v$ times the communication (Narayanan et al., 2021). Only stage-boundary activations cross GPUs, point to point, which is cheap enough to span nodes; keeping the bubble small needs $M \gg K$ and so a large global batch.

**Other axes.** Context parallelism splits long sequences across GPUs (for example ring attention); expert parallelism spreads MoE experts across GPUs with all-to-all token exchange.

**DeepSpeed versus Megatron-LM.** DeepSpeed is a library built around ZeRO: sharding, offload to CPU or NVMe (ZeRO-Offload, ZeRO-Infinity), little change to an existing PyTorch model, plus a pipeline engine. Megatron-LM is a transformer training codebase with hand-written tensor-parallel layers, pipeline schedules, sequence parallelism and fused kernels: high throughput, but the model must be written in its style. They are often combined (Megatron-DeepSpeed) in "3D parallelism": Narayanan et al. found tensor parallelism works best within a node, where NVLink absorbs its all-reduces, and pipeline parallelism across nodes, with data parallelism over the remaining GPUs.

[more](deep-learning-training.html)
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

::: {.qa asked="Google DeepMind research intern, Glassdoor"}
**What is the difference between a PDF and a PMF? Derive the mean and variance of the binomial distribution.**

A PMF gives probabilities of a discrete variable directly, $p(x) = P(X = x)$, with values in $[0, 1]$ that sum to 1. A PDF is a density of a continuous variable: probabilities come from integrals, $P(X \in A) = \int_A f(x)\,dx$, so every single point has probability 0, $f$ can exceed 1 (Uniform(0, 0.5) has density 2), $f$ has units of one over $x$, and $f = F'$ for the CDF $F$. Under a change of variables $y = g(x)$, a PMF just relabels outcomes, while a density picks up a Jacobian factor, $f_Y(y) = f_X(x)\,|dx/dy|$.

**Binomial.** $X \sim \mathrm{Bin}(n, p)$ counts successes in $n$ independent trials: $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$.

*Indicators.* Write $X = \sum_{i=1}^n I_i$ with $I_i \sim \mathrm{Bernoulli}(p)$. Then $\mathbb{E}[I_i] = p$ and, since $I_i^2 = I_i$, $\mathrm{Var}(I_i) = p - p^2 = p(1-p)$. Linearity gives $\mathbb{E}[X] = np$ (no independence needed), and independence makes all covariances zero, so $\mathrm{Var}(X) = np(1-p)$.

*Direct sums.* Using $k\binom{n}{k} = n\binom{n-1}{k-1}$,

$$
\mathbb{E}[X] = \sum_{k=1}^{n} n\binom{n-1}{k-1} p^k (1-p)^{n-k} = np \sum_{j=0}^{n-1}\binom{n-1}{j} p^j (1-p)^{n-1-j} = np.
$$

Using $k(k-1)\binom{n}{k} = n(n-1)\binom{n-2}{k-2}$ the same way gives $\mathbb{E}[X(X-1)] = n(n-1)p^2$, so

$$
\mathrm{Var}(X) = \mathbb{E}[X(X-1)] + \mathbb{E}[X] - (\mathbb{E}[X])^2 = n(n-1)p^2 + np - n^2p^2 = np(1-p).
$$

The variance is largest at $p = 1/2$, and the standard deviation of the fraction $X/n$ is $\sqrt{p(1-p)/n}$: the error bar on an accuracy measured on $n$ test examples.
:::

::: {.qa asked="ByteDance, Nowcoder 2022"}
**Two players take turns tossing a fair coin, and the first to get heads wins. What is the probability that the first player wins?**

$2/3$. With heads probability $p$ the answer is $1/(2-p)$.

*Series.* The first player wins on toss 1, 3, 5, and so on, that is, after an even number $2k$ of tails followed by a head:

$$
P = \sum_{k=0}^{\infty} (1-p)^{2k}\,p = \frac{p}{1 - (1-p)^2} = \frac{p}{p(2-p)} = \frac{1}{2-p},
$$

which is $\tfrac{1/2}{3/4} = \tfrac{2}{3}$ for a fair coin.

*First-step analysis.* Let $P$ be the probability that the player about to toss eventually wins. They win now with probability $p$; otherwise the other player becomes the one about to toss, and our player then wins with probability $1 - P$. So $P = p + (1-p)(1-P)$, which rearranges to $P(2-p) = 1$.

Sanity checks: $P \to 1/2$ as $p \to 0$ (the game lasts many rounds and moving first barely matters), $P \to 1$ as $p \to 1$, and a simulation of $10^6$ games gives about 0.667. Common follow-ups: the expected number of tosses is $1/p = 2$ (geometric), and with three players the first wins with probability $p/(1 - (1-p)^3) = 4/7$.
:::

::: {.qa asked="SenseTime, Nowcoder 2022"}
**Without a calculator, how do you decide whether $a^b$ or $b^a$ is larger?**

For $a, b > 0$, take logarithms and divide by $ab$:

$$
a^b > b^a \iff b\ln a > a\ln b \iff \frac{\ln a}{a} > \frac{\ln b}{b}.
$$

So compare $f(x) = \ln x / x$ at the two points. Its derivative $f'(x) = (1 - \ln x)/x^2$ is positive for $x < e$ and negative for $x > e$, so $f$ rises to its maximum $1/e$ at $x = e$ and then decreases toward 0; it is negative for $x < 1$.

- **Both at least $e$**: if $e \le a < b$ then $f(a) > f(b)$, so $a^b > b^a$; the smaller base with the larger exponent wins. Examples: $3^4 = 81 > 4^3 = 64$, and $10^{11} > 11^{10}$.
- **Both at most $e$**: if $0 < a < b \le e$, $f$ is increasing, so $a^b < b^a$.
- **$e^\pi$ versus $\pi^e$**: $f$ peaks at $e$, so $f(e) > f(\pi)$ and $e^\pi > \pi^e$ (23.14 versus 22.46).
- **One on each side of $e$**: monotonicity does not decide, so evaluate $f$. $f(4) = 2\ln 2/4 = f(2)$, so $2^4 = 4^2$. Because $f$ decreases past $e$, for $b > e$ we get $2^b > b^2$ exactly when $b > 4$: $2^3 = 8 < 3^2 = 9$ but $2^5 = 32 > 5^2 = 25$. With $\ln 2 \approx 0.693$ and $\ln 3 \approx 1.099$, $f(3) \approx 0.366 > f(2) \approx 0.347$ confirms $3^2 > 2^3$.
- **$a < 1 < b$**: $a^b < 1 < b^a$ directly.

The same reduction handles large exponents, such as $3^{100}$ versus $100^3$: compare $100\ln 3 \approx 110$ with $3\ln 100 \approx 13.8$.
:::

::: {.qa asked="SenseTime, Nowcoder 2022"}
**100 lamps start off; student $k$ toggles every lamp whose number is a multiple of $k$. Which lamps are on at the end?**

With students $k = 1, \dots, 100$, the lamps left on are the perfect squares 1, 4, 9, 16, 25, 36, 49, 64, 81 and 100: ten lamps.

Lamp $n$ is toggled once by each student $k$ that divides $n$, and every divisor of $n \le 100$ is a student, so the lamp ends on exactly when $n$ has an odd number of divisors. Divisors come in pairs $(d, n/d)$, and the two members of a pair differ unless $d = n/d$, that is, $d = \sqrt n$. So the count is odd if and only if $n$ is a perfect square. Equivalently, for $n = \prod_i p_i^{e_i}$ the number of divisors is $\prod_i (e_i + 1)$, which is odd exactly when every $e_i$ is even.

For $N$ lamps the answer is the $\lfloor\sqrt N\rfloor$ perfect squares. Follow-ups: the lamps toggled exactly twice are the primes (25 of them up to 100), since a prime's only divisors are 1 and itself. A simulation in which student $k$ visits only multiples of $k$ costs $\sum_k N/k = O(N \log N)$.
:::

::: {.qa asked="bigo image/CV, Nowcoder 2019"}
**What is the expected length of a random chord of a circle?**

It depends on what "random chord" means; different natural sampling schemes give different distributions of chords (Bertrand's paradox). For radius $R$, the three standard schemes give $4R/\pi \approx 1.27R$, $\pi R/2 \approx 1.57R$ and $4R/3 \approx 1.33R$. A chord at distance $r$ from the centre has length $L = 2\sqrt{R^2 - r^2}$; the schemes differ in the distribution they induce on the chord.

1. **Two uniform endpoints on the circle.** Fix one endpoint by symmetry; the central angle $\theta$ to the other is uniform on $[0, \pi]$ after folding, and $L = 2R\sin(\theta/2)$:
$$
\mathbb{E}[L] = \frac{1}{\pi}\int_0^\pi 2R\sin\frac{\theta}{2}\,d\theta = \frac{2R}{\pi}\Big[-2\cos\frac{\theta}{2}\Big]_0^\pi = \frac{4R}{\pi}.
$$
2. **Uniform distance from the centre.** Pick a random radius, a point uniform along it, and the chord perpendicular to the radius there, so $r \sim U[0, R]$:
$$
\mathbb{E}[L] = \frac{1}{R}\int_0^R 2\sqrt{R^2 - r^2}\,dr = \frac{2}{R}\cdot\frac{\pi R^2}{4} = \frac{\pi R}{2},
$$
since the integral is the area of a quarter disk.
3. **Midpoint uniform in the disk.** The midpoint's distance from the centre has density $2r/R^2$:
$$
\mathbb{E}[L] = \int_0^R 2\sqrt{R^2 - r^2}\,\frac{2r}{R^2}\,dr = \frac{4}{R^2}\Big[-\tfrac{1}{3}(R^2 - r^2)^{3/2}\Big]_0^R = \frac{4R}{3}.
$$

A Monte Carlo check with $2 \times 10^6$ chords per scheme gives 1.273, 1.571 and 1.333 for $R = 1$. The classic version of the paradox asks for the probability that a chord is longer than the side $\sqrt3 R$ of the inscribed equilateral triangle: $1/3$, $1/2$ and $1/4$ under the three schemes. In the interview, state your assumption first and then compute; the point being tested is that "uniform" needs a specified measure.
:::

::: {.qa asked="Alibaba Tongyi Lab intern, Nowcoder 2024"}
**Compute $\sqrt{x}$ by binary search instead of Newton's method. How do you set the bounds for very large or very small inputs?**

Bisect on $y$ for the root of $y^2 = x$, keeping $\mathit{lo} \le \sqrt x \le \mathit{hi}$. Bounds that always work are $[\min(x, 1), \max(x, 1)]$: for $x \ge 1$, $1 \le \sqrt x \le x$, and for $0 < x < 1$, $x < \sqrt x < 1$. The common bug is $\mathit{hi} = x$, which excludes the answer when $x < 1$.

**Stopping.** Use a relative tolerance, $\mathit{hi} - \mathit{lo} \le \mathrm{tol} \cdot \mathit{lo}$. An absolute tolerance is meaningless for tiny inputs (with tolerance $10^{-9}$, any value below $10^{-9}$ passes for $x = 10^{-20}$, whose root is $10^{-10}$) and unreachable for huge ones (adjacent doubles near $10^{150}$ are about $10^{134}$ apart, so the loop never ends). Each step halves the interval, so it takes about $\log_2\!\big((\max(x,1) - \min(x,1))/(\mathrm{tol}\sqrt x)\big)$ iterations: 40 for $x = 2$ and about 540 for $x = 10^{\pm 300}$ at $\mathrm{tol} = 10^{-12}$. An iteration cap guards against a tolerance below machine precision.

**Shrinking the range.** To make the count independent of magnitude, split off the exponent: write $x = m \cdot 2^e$ with $e$ even and $m \in [0.5, 2)$ (`math.frexp`, moving one factor of 2 into $m$ when $e$ is odd), bisect for $\sqrt m$ in $[0.7, 1.5]$, and return $\sqrt m \cdot 2^{e/2}$ (`math.ldexp`). About 40 iterations then reach $10^{-12}$ for any input.

**Overflow.** Compare `mid > x / mid` rather than `mid * mid > x`; in the integer version this matters in fixed-width languages, where `mid * mid` overflows.

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

Time is $O(\log(\mathrm{range}/\mathrm{tol}))$ iterations for the float version and $O(\log n)$ for the integer one, with $O(1)$ memory. For comparison, Newton's update $y \leftarrow (y + x/y)/2$ doubles the number of correct digits per step once close, so after the exponent reduction about 5 steps from $y = 1$ reach double precision; bisection gains one bit per step but needs only comparisons and always stays inside its bracket.
:::

::: {.qa asked="NetEase Games image algorithm, Nowcoder 2021"}
**How do you test whether a function is convex, and how do Newton and quasi-Newton methods use the Hessian?**

**Convexity tests** for $f$ on a convex domain:

1. Definition: $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$ for $\lambda \in [0, 1]$; equivalently, $t \mapsto f(x + tv)$ is convex for every $x, v$, which reduces any test to one dimension.
2. First order: $f(y) \ge f(x) + \nabla f(x)^\top(y - x)$, so tangent planes lie below $f$ and a stationary point is a global minimum.
3. Second order: $\nabla^2 f(x) \succeq 0$ everywhere, checked with eigenvalues or a Cholesky factorization (leading principal minors certify definiteness; semidefiniteness needs all principal minors nonnegative).
4. Composition rules: nonnegative sums, affine precomposition, pointwise maxima, and $h(g(x))$ with $h$ convex nondecreasing and $g$ convex preserve convexity, which covers norms, log-sum-exp and softmax cross-entropy in the logits.

Numerically, one point with a negative Hessian eigenvalue disproves convexity; sampling cannot prove it.

**Newton's method.** Minimize the quadratic model $f(x) + g^\top p + \tfrac12 p^\top H p$: its gradient $g + Hp$ vanishes at $p = -H^{-1}g$, so $x_{k+1} = x_k - H_k^{-1}g_k$. With $e_k = x_k - x^*$ and $\nabla f(x^*) = 0$, $g_k = \int_0^1 H(x^* + te_k)\,e_k\,dt$; if $H_k \succeq \mu I$ and $H$ is $M$-Lipschitz,

$$
\|e_{k+1}\| = \Big\|H_k^{-1}\int_0^1 \big(H_k - H(x^* + te_k)\big)e_k\,dt\Big\| \le \frac{1}{\mu}\int_0^1 M(1-t)\|e_k\|^2\,dt = \frac{M}{2\mu}\|e_k\|^2 ,
$$

so convergence is quadratic near the minimizer, and affine invariance makes it insensitive to conditioning. The cost is $O(d^2)$ memory and $O(d^3)$ per solve. Far from the minimizer it can fail: with an indefinite $H$ the step may head uphill or toward a saddle, and it can diverge even on a convex function (for $f(x) = \sqrt{1 + x^2}$ the iteration is $x \mapsto -x^3$, divergent for $|x| > 1$). Fixes: a line search (damped Newton), trust regions, or Levenberg-Marquardt damping $(H + \lambda I)^{-1}$.

**Quasi-Newton.** Build curvature from gradients alone. With $s_k = x_{k+1} - x_k$ and $y_k = g_{k+1} - g_k$, impose the secant condition $B_{k+1}s_k = y_k$. BFGS updates the inverse approximation $C_k \approx H^{-1}$ by a rank-two correction,

$$
C_{k+1} = (I - \rho_k s_k y_k^\top)\,C_k\,(I - \rho_k y_k s_k^\top) + \rho_k s_k s_k^\top, \qquad \rho_k = \frac{1}{y_k^\top s_k},
$$

which stays positive definite when $y_k^\top s_k > 0$ (a Wolfe line search ensures this), costs $O(d^2)$ per step and converges superlinearly. L-BFGS keeps only the last $m$ pairs (typically 3 to 20) and applies $C_k$ to a gradient with a two-loop recursion in $O(md)$. Minibatch noise corrupts the $(s, y)$ pairs, so in deep learning L-BFGS is used mainly for full-batch problems, and curvature enters through Hessian-vector products or structured approximations.
:::

::: {.qa asked="NetEase Games image algorithm, Nowcoder 2021"}
**What do the eigenvectors of a covariance matrix mean, and how does the SVD relate to PCA?**

The eigenvectors are orthogonal directions along which the projected coordinates are uncorrelated, ordered by the variance they carry; the eigenvalues are those variances. PCA keeps the top ones, and the SVD of the centred data matrix computes the same thing more stably.

**Derivation.** Let $X \in \mathbb{R}^{n \times d}$ be centred data (rows are samples) and $\Sigma = X^\top X/(n-1)$. The variance of the projection onto a unit vector $u$ is $\mathrm{Var}(Xu) = u^\top\Sigma u$. Maximizing it subject to $u^\top u = 1$ with the Lagrangian $u^\top\Sigma u - \lambda(u^\top u - 1)$ gives $\Sigma u = \lambda u$, and at such a point the objective equals $u^\top\Sigma u = \lambda$. So the best direction is the top eigenvector and the variance captured is the top eigenvalue; requiring orthogonality to earlier components yields the next eigenvectors in order. Because $\Sigma$ is symmetric positive semidefinite, its eigenvectors are orthogonal and its eigenvalues are nonnegative. $\sum_i \lambda_i = \mathrm{tr}\,\Sigma$ is the total variance, so $\lambda_i / \sum_j \lambda_j$ is the explained-variance ratio. In the eigenbasis the covariance is the diagonal $\Lambda$; for Gaussian data the eigenvectors are the axes of the density's ellipsoids, with semi-axes proportional to $\sqrt{\lambda_i}$.

**SVD.** $X = USV^\top$ gives $X^\top X = VS^2V^\top$, so $\Sigma = V\,\tfrac{S^2}{n-1}\,V^\top$: the right singular vectors are the principal directions, $\lambda_i = s_i^2/(n-1)$, and the projected data (scores) are $XV = US$. Forming $X^\top X$ squares the condition number, so small-variance directions lose about twice as many digits; the SVD works on $X$ directly. For the top $k$ components of large data, use a truncated or randomized SVD; when $d \gg n$, the $n \times n$ Gram matrix $XX^\top$ has the same nonzero eigenvalues and is cheaper.

**Whitening.** $z = \Lambda^{-1/2}V^\top x$ has identity covariance (PCA whitening). $V\Lambda^{-1/2}V^\top x$ (ZCA) also whitens but stays as close as possible to the original coordinates, so whitened images still resemble images. Add a small $\epsilon$ to each $\lambda_i$ so near-zero directions, mostly noise, are not blown up.

**Pitfalls.** Forgetting to centre; mixing units, since PCA is not scale-invariant (standardize first); reading meaning into eigenvector signs, which are arbitrary; and expecting PCA to find nonlinear structure, since it uses only second moments.
:::

::: {.qa asked="Alibaba, Nowcoder 2022"}
**Prove that gradient descent converges, and state the rate.**

Assume $f$ is $L$-smooth ($\nabla f$ is $L$-Lipschitz) and use step $\eta = 1/L$. Then: for $f$ bounded below by $f^*$, $\min_{k<K}\|\nabla f(x_k)\|^2 \le 2L(f(x_0) - f^*)/K$; for convex $f$, $f(x_K) - f^* \le L\|x_0 - x^*\|^2/(2K)$; for $\mu$-strongly convex $f$, $f(x_K) - f^* \le (1 - \mu/L)^K(f(x_0) - f^*)$.

**Descent lemma.**

$$
f(y) - f(x) - \nabla f(x)^\top(y - x) = \int_0^1 \big(\nabla f(x + t(y - x)) - \nabla f(x)\big)^\top(y - x)\,dt \le \int_0^1 Lt\,\|y - x\|^2\,dt = \frac{L}{2}\|y - x\|^2 .
$$

With $y = x - \eta\nabla f(x)$ this gives $f(y) \le f(x) - \eta(1 - L\eta/2)\|\nabla f(x)\|^2$, so any $\eta < 2/L$ decreases $f$, and $\eta = 1/L$ maximizes the guaranteed decrease: $f(x_{k+1}) \le f(x_k) - \frac{1}{2L}\|g_k\|^2$ with $g_k = \nabla f(x_k)$.

**Nonconvex.** Summing over $k = 0, \dots, K-1$ telescopes: $\frac{1}{2L}\sum_k \|g_k\|^2 \le f(x_0) - f(x_K) \le f(x_0) - f^*$. The minimum is at most the average, which gives the first rate: $\|\nabla f\| \le \epsilon$ within $O(1/\epsilon^2)$ steps, at a stationary point that may be a saddle.

**Convex.** Convexity gives $f(x_k) - f^* \le g_k^\top(x_k - x^*)$. Expanding $\|x_{k+1} - x^*\|^2 = \|x_k - x^*\|^2 - \tfrac{2}{L}g_k^\top(x_k - x^*) + \tfrac{1}{L^2}\|g_k\|^2$ and combining with the descent step,

$$
f(x_{k+1}) - f^* \le g_k^\top(x_k - x^*) - \frac{1}{2L}\|g_k\|^2 = \frac{L}{2}\Big(\|x_k - x^*\|^2 - \|x_{k+1} - x^*\|^2\Big).
$$

Summing telescopes to $\sum_{k=1}^{K}(f(x_k) - f^*) \le \frac{L}{2}\|x_0 - x^*\|^2$, and since $f(x_k)$ is nonincreasing, $K(f(x_K) - f^*)$ is at most that sum: the $O(1/K)$ rate.

**Strongly convex.** Minimizing both sides of $f(y) \ge f(x) + g^\top(y - x) + \frac{\mu}{2}\|y - x\|^2$ over $y$ gives $f^* \ge f(x) - \|g\|^2/(2\mu)$, i.e., $\|g\|^2 \ge 2\mu(f(x) - f^*)$ (the Polyak-Łojasiewicz inequality). Substituting into the descent step, $f(x_{k+1}) - f^* \le (1 - \mu/L)(f(x_k) - f^*)$: linear convergence, $O(\kappa\log(1/\epsilon))$ steps with $\kappa = L/\mu$. The proof uses only the PL inequality, so it also covers nonconvex functions that satisfy it.

For context, Nesterov acceleration improves these to $O(1/K^2)$ and a factor $1 - 1/\sqrt\kappa$ per step, while SGD on convex problems is slower, $O(1/\sqrt K)$, because its gradient noise does not vanish at the optimum.
:::

::: {.qa asked="OpenAI RE (Aced, ~2025)"}
**Implement all_gather across several nodes. If the links between nodes are noisy, derive a worst-case bound on the number of rounds needed for every node to get every value within a target error, improve the bound, and then use the fact that the values are floating-point numbers to design a better algorithm.**

The report gives no noise model, so state one. The point to reach: sending a real value and averaging copies needs rounds proportional to $1/\varepsilon^2$, while sending a float's bits, decoding each exactly and forwarding it needs rounds proportional to the number of bits.

**Setup.** Node $i$ of $n$ holds $x_i \in [-R, R]$. Per round, each node sends one symbol $|s| \le A$ to a neighbour, which receives $s + z$, $z \sim \mathcal{N}(0, \sigma^2)$ independent. Ring all_gather takes $n - 1$ steps, each node forwarding to its successor the value it received in the previous step (its own first): $n(n-1)$ value transmissions in all.

**Naive bound.** Send $(A/R)\,x$ $k$ times, average, forward the estimate. One hop leaves Gaussian error of variance $(R\sigma/A)^2/k$, and $n - 1$ hops add $n - 1$ such errors. A Gaussian tail and a union bound over the $n(n-1)$ (value, receiver) pairs give

$$
2n(n-1)\exp\Big(-\frac{k\varepsilon^2}{2(n-1)(R\sigma/A)^2}\Big) \le \delta
\iff k \ge 2(n-1)\Big(\frac{R\sigma}{A\varepsilon}\Big)^2 \ln\frac{2n(n-1)}{\delta},
$$

for a total of $(n-1)k = O\big(n^2 (R\sigma/A\varepsilon)^2 \log(n/\delta)\big)$ rounds.

**Improving it.** Chebyshev's inequality would give a factor $1/\delta$; the Gaussian tail makes it $\log(1/\delta)$. Fewer hops per value (direct links, or a tree of depth $\log_2 n$) remove the factor $n - 1$ from $k$. But $1/\varepsilon^2$ remains, because averaging shrinks the error only as $1/\sqrt{k}$: each extra bit of precision costs four times the rounds.

**Using floats.** A float32 is a 32-bit string. Send each bit as $\pm A$, $r$ times, and decode by the sign of the average:

$$
P(\text{bit wrong}) = Q\big(A\sqrt{r}/\sigma\big) \le \tfrac{1}{2} e^{-rA^2/2\sigma^2}.
$$

Decoded bits are forwarded clean, so noise no longer accumulates over hops. A union bound over $b$ bits in each of the $n(n-1)$ transmissions needs $r \ge (2\sigma^2/A^2) \ln\big(b\,n(n-1)/2\delta\big)$, a total of $(n-1)\,b\,r = O\big(n b (\sigma/A)^2 \log(nb/\delta)\big)$ rounds, and with probability at least $1 - \delta$ every float arrives exactly. The target error only sets $b$: 32 bits for an exact copy, or about $\log_2(2R/\varepsilon)$ in fixed point. With $n = 8$, $\delta = 10^{-3}$, $A = \sigma$ and $R/\varepsilon = 10^3$, averaging needs about $2.3 \times 10^7$ symbols per value even over one hop; bits with repetition need $11 \times 26 = 286$.

**Further.** Protect bits unequally (a sign or exponent error is catastrophic, the last mantissa bits barely matter), and replace repetition with an error-correcting code, which costs a constant number of channel uses per bit. Nothing does much better: pinning a worst-case value to within $\varepsilon$ means telling apart about $R/\varepsilon$ values, roughly $\log_2(R/\varepsilon)$ bits, and a Gaussian channel with power $A^2$ carries at most $C = \frac{1}{2}\log_2(1 + A^2/\sigma^2)$ bits per use, so any scheme needs about $\log_2(R/\varepsilon)/C$ rounds per hop. If the channel instead flips bits with probability $p$, majority voting over $r$ copies fails with probability at most $e^{-2r(1/2 - p)^2}$ (Hoeffding), and the rest carries over.
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

::: {.qa asked="Curated diffusion lists (Zhihu, CSDN 2025)"}
**Why does DDPM add noise in many small steps whose size grows over time, instead of one large step or equal steps?**

Small steps keep each reverse step easy to model. Growing steps spread the information loss evenly, because a fixed amount of added noise destroys far more information while the image is still clean than once it is already noisy.

1. *Why many small steps.* Each reverse conditional $q(x_{t-1} \mid x_t)$ is modeled as a Gaussian, which is accurate only when $\beta_t$ is small: forward and reverse kernels then share a functional form, the argument DDPM takes from Sohl-Dickstein et al. (2015). With a single step, $q(x_0 \mid x_1)$ is essentially the data distribution, and a Gaussian fit to it is a blurry average. VDM (Kingma et al., 2021) also shows that, for a good enough denoiser, doubling the number of steps lowers the diffusion loss.
2. *Why growing steps.* Write $\lambda_t = \log\frac{\bar\alpha_t}{1-\bar\alpha_t}$ and $\mathrm{SNR}_t = e^{\lambda_t}$. Since $1-\bar\alpha_t = (1-\bar\alpha_{t-1}) + \bar\alpha_{t-1}\beta_t$, the log-SNR drop per step is exactly
$$\lambda_{t-1} - \lambda_t = \log\big(1 + \beta_t\,\mathrm{SNR}_{t-1}\big) - \log(1 - \beta_t).$$
At high SNR (early steps) a tiny $\beta_t$ already moves $\lambda$ a lot; at low SNR (late steps) $\beta_t$ must be large to make progress. With DDPM's linear schedule ($T = 1000$, $\beta_t$ from $10^{-4}$ to $0.02$) the drop is about 0.8 at the start and about 0.01 to 0.02 in the middle and at the end.
3. *Why not equal steps.* A constant $\beta$ forces a bad choice. With $\beta = 10^{-4}$, $\bar\alpha_T \approx 0.90$ after 1000 steps, nowhere near $\mathcal{N}(0, I)$. With $\beta = 0.01$, $\bar\alpha_T \approx 4 \times 10^{-5}$, but the first step already adds noise of standard deviation 0.1 to data in $[-1, 1]$, so the last reverse step must remove that much noise in one jump, where fine texture is decided. The linear schedule starts at standard deviation 0.01 (about one 8-bit grey level) and ends with $\sqrt{\bar\alpha_T} \approx 0.006$; Ho et al. picked it from constant, linear and quadratic candidates, all constrained so the prior term $L_T$ is near zero.

What ultimately matters is how steps are distributed over log-SNR; the linear schedule still spends its last steps near zero SNR, which the cosine schedule corrects (see the noise-schedule question in this section).

[more](diffusion-models.html)
:::

::: {.qa asked="Kling RL role (Nowcoder snippet)"}
**What is generalized advantage estimation (GAE), and how does it relate to RL fine-tuning of diffusion models?**

GAE (Schulman et al., 2015) estimates the advantage $A(s_t, a_t)$ as an exponentially weighted sum of one-step TD errors of a learned value function $V$; the weight $\lambda$ trades bias against variance.

1. TD residual: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, an advantage estimate that is unbiased only if $V$ is exact.
2. Summing $k$ residuals telescopes the intermediate values: $\sum_{l=0}^{k-1}\gamma^l \delta_{t+l} = \sum_{l=0}^{k-1}\gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t)$, the $k$-step estimate. Averaging these with weights $(1-\lambda)\lambda^{k-1}$ gives
$$\hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l}.$$
3. $\lambda = 0$ gives $\delta_t$ (TD(0)): low variance, biased when $V$ is wrong. $\lambda = 1$ gives $\sum_l \gamma^l r_{t+l} - V(s_t)$, the Monte Carlo return minus a baseline: unbiased for any $V$ but high variance. The paper notes that $\gamma < 1$ biases the gradient whatever $V$ is, while $\lambda < 1$ does so only when $V$ is inaccurate; its best runs used $\lambda$ between 0.92 and 0.99.

In diffusion fine-tuning, an episode is the denoising chain, intermediate rewards are zero, the only reward $R$ comes from the final image, and $\gamma = 1$. Then $\delta_t = V(s_{t+1}) - V(s_t)$ until the last step, and with $K$ steps left,
$$\hat A_t = \lambda^{K-1} R + (1-\lambda)\sum_{l=1}^{K-1} \lambda^{l-1} V(s_{t+l}) - V(s_t).$$
For $\lambda < 1$ the real signal $R$ is down-weighted by $\lambda^{K-1}$ at early steps (about 0.08 for $\lambda = 0.95$ and 50 steps), so the estimate leans on a critic that must predict the final image's reward from a noisy latent, a hard regression. Most methods therefore use $\lambda = 1$, so $\hat A_t = R - b$ is the same for every step of a trajectory, and replace the critic with a cheap baseline $b$: DDPO (Black et al., 2023) normalizes rewards with a running mean and standard deviation tracked per prompt, and GRPO-style methods use the statistics of a group of samples for the same prompt. A critic can still help: DPOK (Fan et al., 2023) learned $V(x_t, c)$ and reported higher final rewards.

[more](text-to-image-multimodal.html)
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

SAM runs a large ViT image encoder once per image, encodes prompts (points, boxes, masks) with a light prompt encoder, and uses a small two-way transformer decoder to output several candidate masks with predicted IoU scores, which handles ambiguous prompts. It generalizes because of its promptable design and its data engine, which produced about 1.1B masks (SA-1B). SAM 2 extends this to video with a streaming memory of past frames that the current frame attends to, so it tracks objects through time. SAM 3 adds concept prompts (a short noun phrase, image exemplars given as positive or negative boxes on the image, or both) and returns every matching instance, which brings open-vocabulary detection into the segmenter.
:::

::: {.qa asked="Qualcomm 2026, Glassdoor"}
**How does YOLO work?**

YOLO is a one-stage detector: a single forward pass predicts, at each cell of one or more feature grids, objectness, class probabilities and box parameters. YOLOv2 introduced anchors; later versions such as YOLOX and YOLOv8 are anchor-free. Training combines a box-regression loss (IoU-based variants such as CIoU in later versions) with objectness and classification losses, and multi-scale heads on FPN- or PAN-style necks handle object size. Non-maximum suppression removes duplicates, although some recent versions train one-to-one assignments to drop it. Compared with two-stage detectors such as Faster R-CNN, YOLO is faster and was historically weaker on small objects.
:::

::: qa
**Compare MAE and DINO as self-supervised pretraining methods.**

MAE masks about 75% of the patches, encodes only the visible ones, and reconstructs pixels with a light decoder; it is cheap to train and fine-tunes well, but its frozen features are weaker under linear probing. DINO trains a student to match an EMA teacher's softmax outputs across different crops, using centring and sharpening of the teacher outputs to avoid collapse without negatives; its frozen features are strong for k-NN retrieval and segmentation. DINOv2 scales the recipe with curated data, and its features are now used for feature-space generative metrics and for representation-alignment losses in diffusion training (REPA).
:::

::: {.qa asked="CV interview advice thread (Reddit 2025), not a report"}
**How does DETR detect objects without anchors or non-maximum suppression?**

DETR (Carion et al., 2020) treats detection as direct set prediction: it outputs a fixed-size set of boxes and trains with a one-to-one matching loss, so duplicates are penalized during training instead of removed afterwards.

1. *Architecture.* A CNN backbone (ResNet-50 in the base model) gives a feature map; a 1x1 convolution reduces its channels, the map is flattened into tokens with positional encodings, and a transformer encoder applies global self-attention. The decoder takes $N = 100$ learned object queries that attend to each other and to the image; a shared FFN maps each query to a class distribution, including a "no object" class $\varnothing$, and a normalized box (center, width, height).
2. *Matching.* Pad the ground truth to $N$ entries with $\varnothing$ and find the permutation with the lowest total cost using the Hungarian algorithm. For a real object the cost is $-\hat p_{\sigma(i)}(c_i) + \mathcal{L}_{\text{box}}$, with $\mathcal{L}_{\text{box}} = \lambda_{\text{iou}}\mathcal{L}_{\text{GIoU}} + \lambda_{L1}\lVert b_i - \hat b_{\sigma(i)}\rVert_1$ (weights 2 and 5); GIoU adds a scale-invariant term, since L1 alone treats small and large boxes differently.
3. *Loss.* Cross-entropy on all $N$ predictions, with the $\varnothing$ term down-weighted by 10 for class imbalance, plus the box loss on matched pairs, applied after every decoder layer.

Why no NMS: each object is matched to exactly one query, so a second query predicting the same object is trained toward $\varnothing$, and decoder self-attention lets queries coordinate which one claims it. The paper tested NMS on its outputs: it helped after the first decoder layer and slightly lowered AP after the last.

Weaknesses: slow convergence (500 epochs to reach 42.0 AP on COCO, comparable to a Faster R-CNN baseline of the same size) and small objects (5.5 points lower $\text{AP}_S$ than that baseline), because attention over a single low-resolution map is expensive and starts out nearly uniform. Deformable DETR attends to a few learned sampling points around a reference point on multi-scale features and reports better accuracy, especially on small objects, with 10x fewer epochs. DN-DETR attributes slow convergence to unstable bipartite matching and adds noised ground-truth boxes as extra queries to reconstruct; DINO builds on this with contrastive denoising and better query initialization, reaching 49.4 AP in 12 epochs with ResNet-50.
:::

::: {.qa asked="SenseTime 2019, Momo 2021 (Nowcoder)"}
**What changed from MobileNet v1 to MobileNet v2, and why?**

V1 replaced standard convolutions with depthwise-separable ones. V2 kept them but rebuilt the block around where nonlinearities lose information, giving inverted residuals with linear bottlenecks.

*MobileNet v1 (Howard et al., 2017).* A plain stack of 3x3 depthwise then 1x1 pointwise convolutions, each followed by BatchNorm and ReLU, with strided depthwise layers for downsampling and no residual connections (28 layers counting both kinds). With $3 \times 3$ kernels this costs 8 to 9 times less than standard convolution. Two global knobs trade accuracy for cost: a width multiplier $\alpha$ thins every layer (cost and parameters scale by about $\alpha^2$) and a resolution multiplier $\rho$ shrinks the input (cost scales by $\rho^2$).

*MobileNet v2 (Sandler et al., 2018).* Each block is

1. a 1x1 expansion from $d'$ to $t\,d'$ channels (expansion factor $t = 6$), then ReLU6;
2. a 3x3 depthwise convolution in the wide space, then ReLU6;
3. a 1x1 projection to $d''$ narrow channels with no activation;
4. a shortcut between the narrow ends when the stride is 1 and $d' = d''$.

The reasons:

- *Linear bottleneck.* The paper argues that useful activations lie on a low-dimensional manifold, and ReLU preserves information about it only when that manifold sits in a low-dimensional subspace of a wider space; in a narrow layer ReLU discards too much. So the nonlinearities act in the expanded space and the projection stays linear. Its ablation found that adding a nonlinearity to the bottleneck lowered accuracy.
- *Inverted residual.* ResNet bottlenecks are wide-narrow-wide with skips between wide tensors; v2 is narrow-wide-narrow with skips between the narrow tensors, which carry the information while the expansion is an implementation detail. The ablation found shortcuts between bottlenecks better than between expansions, and memory drops because the wide intermediate tensor can be computed in pieces and never fully stored.
- *ReLU6*, clipped at 6, for robustness in low-precision computation.

The result in the v2 paper's ImageNet table: 72.0% top-1 with 3.4M parameters and 300M multiply-adds, against 70.6% with 4.2M and 575M for v1, and lower latency on a phone CPU core (75 ms against 113 ms).

[more](vision-backbones.html)
:::

::: {.qa asked="Xiaomi multimodal, Nowcoder 2024"}
**Which image, video and text encoders do you know, and when would you pick each?**

Choose by what the downstream task needs from the features (language-aligned semantics, dense spatial detail, motion, or reconstruction) and by whether the encoder stays frozen.

*Image encoders.*

- CNNs (ResNet, ConvNeXt): multi-scale feature pyramids and cost linear in pixel count, so they suit detection, segmentation and on-device latency.
- CLIP and SigLIP: global semantics aligned with text; the usual vision tower of VLMs and the basis of retrieval, zero-shot classification and text-image scores. SigLIP's pairwise sigmoid loss needs no batch-wide softmax normalization, so it scales to larger batches and also does better at small ones.
- DINOv2: self-supervised features (trained on the curated LVD-142M set) that are strong when frozen, notably for segmentation, depth and correspondence; also used for feature-space metrics and representation alignment in diffusion.
- MAE: a cheap pretraining recipe that fine-tunes well but gives weaker frozen features; SAM's image encoder is an MAE-pretrained ViT-H.

*Video encoders.*

- I3D inflates 2D ImageNet filters into 3D; SlowFast pairs a low-frame-rate pathway for semantics with a lightweight high-frame-rate pathway for motion.
- TimeSformer (divided space-time attention) and ViViT (factorized variants) are video transformers; VideoMAE pretrains them by masking 90 to 95% of space-time tubes.
- Per-frame CLIP embeddings pooled over time are a strong, cheap baseline for retrieval and captioning but ignore motion order.
- In generation, the video "encoder" is a causal 3D VAE built for reconstruction, not semantics.

*Text encoders.*

- BERT-style bidirectional encoders for classification and retrieval.
- T5 encoders (T5-XXL; umT5 for multilingual prompts) as frozen conditioning for diffusion; Imagen and SD3 found them better than CLIP alone on complex prompts and written text.
- The CLIP text encoder: aligned with images, but limited to 77 tokens and weak on word order and attribute binding.
- Decoder-only LLMs or VLMs used as encoders (hidden states from several layers) for world knowledge and long prompts.

Rules of thumb: image-text matching, CLIP or SigLIP; dense prediction or frozen probing, DINOv2 or a CNN pyramid; motion understanding, a video-native encoder; diffusion conditioning, T5 or an LLM plus a pooled CLIP vector; tight latency, a CNN or a small ViT. Check the input resolution and token count, which set the cost.

[more](vision-backbones.html)
:::

::: {.qa asked="Tencent multimodal, Nowcoder 2026"}
**How was SAM 3 trained, and how was its concept-prompt data built?**

SAM 3 (Carion et al., 2025, arXiv 2511.16719) adds an open-vocabulary detector to SAM 2's tracker. Much of its gain comes from a data engine that uses models as annotators and verifiers, so human effort goes to the hard cases.

*Architecture.* A shared Perception Encoder backbone encodes the image and the noun-phrase prompt; image exemplars (a box with a positive or negative label) are encoded from ROI-pooled features and appended to the text tokens. A fusion encoder conditions the image features on these prompt tokens by cross-attention, and a DETR-style decoder with learned queries predicts boxes, scores and masks. The key addition is a *presence token*: one global token predicts whether the concept appears at all, each query only predicts whether it is a match given presence, and the final score is the product, which separates recognition (global context) from localization (local evidence). For video, a tracker inherited from SAM 2 propagates masklets with a memory bank; they are matched to new detections by IoU, and the tracker is periodically re-prompted with confident detections.

*Training* has four stages: Perception Encoder pretraining, detector pretraining, detector fine-tuning, then tracker training with the backbone frozen.

*Data engine.* Media are mined with a curated ontology, a model proposes noun phrases, and SAM 3 (initially SAM 2 prompted by an off-the-shelf detector) proposes masks. Two checks follow: mask verification (accept or reject each mask) and exhaustivity verification (are all instances masked?). Failures go to humans, who correct the masks.

1. Phase 1: human verifiers only.
2. Phase 2: Llama 3.2 fine-tuned on the phase-1 human labels becomes an AI verifier for both checks, roughly doubling throughput; a Llama-based pipeline proposes phrases, including hard negatives the current SAM 3 wrongly fires on.
3. Phase 3: 15 domains, and long-tail concepts from a 22.4M-node ontology built on Wikidata.
4. Phase 4: video.

*What it released.* SA-Co/HQ (5.2M images, 4M unique phrases), a synthetic set labeled without humans (38M phrases, 1.4B masks), SA-Co/VIDEO (52.5K videos) and the SA-Co benchmark (207K concepts, exhaustive masks on 120K images and 1.7K videos), scored by cgF1, the product of localization F1 and image-level presence accuracy (MCC). In ablations, hard negatives raised that presence accuracy from 0.44 to 0.68, and the presence head added 1.5 cgF1.
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

::: {.qa asked="Megvii CV algorithm, Nowcoder 2022"}
**Implement nearest-neighbour upsampling. How does it differ from bilinear?**

Nearest neighbour copies, for each output pixel, the input pixel whose area contains the output pixel's centre; bilinear blends the four surrounding input pixels. With the pixel-centre convention, output column $j$ maps to $u = (j + 0.5)\,W_{\text{in}}/W_{\text{out}} - 0.5$, and rounding $u$ equals $\lfloor (j + 0.5)\,W_{\text{in}}/W_{\text{out}} \rfloor$:

```python
import numpy as np
def nearest_resize(img, out_h, out_w):
    H, W = img.shape[:2]
    ys = np.minimum(((np.arange(out_h) + 0.5) * H / out_h).astype(int), H - 1)
    xs = np.minimum(((np.arange(out_w) + 0.5) * W / out_w).astype(int), W - 1)
    return img[ys[:, None], xs[None, :]]   # (H, W) or (H, W, C); dtype is kept
```

This matches PyTorch's `F.interpolate(mode="nearest-exact")`. The legacy `mode="nearest"` uses $\lfloor j\,W_{\text{in}}/W_{\text{out}} \rfloor$, which picks different pixels when downsampling or at non-integer scales. For an integer factor $s$ both reduce to `img.repeat(s, 0).repeat(s, 1)`. Time and memory are $O(H_{\text{out}} W_{\text{out}} C)$: one gather per output value.

How it differs from bilinear:

- *Kernel.* A box (frequency response a sinc) instead of a triangle. The output is piecewise constant, so edges turn into staircases and smooth gradients into blocks; bilinear is continuous but blurrier.
- *Values.* Nearest creates no new values, which makes it the right choice for label maps, segmentation masks, index or ID images and pixel art, where bilinear would invent invalid in-between labels. Bilinear suits natural images and feature maps.
- *Downsampling.* Nearest takes one sample per output pixel with no averaging, so it aliases even more than bilinear; prefilter or area-average first.
- *Gradient.* Backpropagation sums the gradients over each copied block. Nearest upsampling followed by a convolution is a standard way to avoid the checkerboard artifacts of transposed convolutions.

[more](restoration-super-resolution.html)
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**What is deformable convolution, and how would you implement it?**

A deformable convolution samples its kernel taps at learned, input-dependent, fractional positions instead of a fixed grid, so the receptive field can follow object shape or motion. A standard $k \times k$ convolution computes $y(p_0) = \sum_n w_n\, x(p_0 + p_n)$ over the $K = k^2$ grid offsets $p_n$; the deformable version is
$$y(p_0) = \sum_{n=1}^{K} w_n\, x(p_0 + p_n + \Delta p_n)\, \Delta m_n .$$

1. The offsets $\Delta p_n$ come from a parallel ordinary convolution on the same input with $2K$ output channels, one $(\Delta y, \Delta x)$ pair per tap and output location. It is zero-initialized, so training starts from a regular convolution (Dai et al., 2017).
2. The positions are fractional, so $x$ is read by bilinear interpolation, which is differentiable with respect to both $x$ and the offsets.
3. DCNv2 (Zhu et al., 2018) adds the modulation $\Delta m_n \in [0, 1]$ (DCNv1 has $\Delta m_n = 1$): the offset branch outputs $3K$ channels, the last $K$ through a sigmoid, so taps can also be turned down. It also makes all 3x3 convolutions in three stages of the backbone deformable rather than a few.

An im2col-style implementation builds the $K$ sampling grids, gathers $K$ bilinearly shifted copies of the input with `grid_sample`, and contracts them with the weights (stride 1, dilation 1):

```python
import torch, torch.nn.functional as F
def deform_conv2d(x, weight, offset, mask=None, pad=1):
    B, C, H, W = x.shape                      # offset: (B, 2K, Ho, Wo) as (dy, dx) pairs
    O, _, kh, kw = weight.shape               # mask:   (B, K, Ho, Wo) or None
    Ho, Wo = offset.shape[-2:]
    ky, kx = torch.meshgrid(torch.arange(kh), torch.arange(kw), indexing="ij")
    oy, ox = torch.meshgrid(torch.arange(Ho), torch.arange(Wo), indexing="ij")
    py = (oy + ky.reshape(-1, 1, 1) - pad).to(x) + offset[:, 0::2]   # (B, K, Ho, Wo)
    px = (ox + kx.reshape(-1, 1, 1) - pad).to(x) + offset[:, 1::2]
    grid = torch.stack([2 * px / (W - 1) - 1, 2 * py / (H - 1) - 1], dim=-1)
    cols = F.grid_sample(x, grid.view(B, -1, Wo, 2), mode="bilinear",
                         padding_mode="zeros", align_corners=True)
    cols = cols.view(B, C, kh * kw, Ho, Wo)
    if mask is not None:
        cols = cols * mask[:, None]
    return torch.einsum("bckhw,ock->bohw", cols, weight.flatten(2))
```

It matches `torchvision.ops.deform_conv2d` (same offset layout, zero outside the image) and reduces to `F.conv2d` when the offsets are zero. Cost: the same $O(B H_o W_o K C O)$ multiply-adds as a convolution, plus the offset convolution and the bilinear gathers; memory $O(B C K H_o W_o)$ for the sampled columns, $K$ times the input. Irregular memory access makes it slower than its FLOP count suggests, so production code uses fused CUDA kernels. In video SR (TDAN, EDVR) the offsets are predicted from the neighbouring and reference frames to align the neighbour's features, which is equivalent to $K$ flow warps followed by a convolution.

[more](restoration-super-resolution.html)
:::

::: {.qa asked="iQIYI, Nowcoder 2022"}
**How would you make a super-resolution model fast enough to deploy?**

Start from a latency budget on the target device and a profile, then cut cost where the profile says it goes. FLOPs and parameter counts often disagree with measured latency; the RepVGG paper makes this point, noting that multi-branch blocks slow inference and depthwise convolutions raise memory-access cost.

1. *Measure.* Fix the target (phone NPU, server GPU, real-time video at a given resolution) and a quality floor measured on your own data, then profile per layer: compute-bound convolutions, memory-bound element-wise ops, data movement, pre- and post-processing.
2. *Architecture.* Keep all computation at low resolution and upsample once at the end with pixel shuffle, which saves a factor $s^2$ over networks that interpolate first; FSRCNN's redesign of SRCNN along these lines reported a speed-up of more than 40x. Use few channels and blocks, plain 3x3 convolutions that the hardware runs well, no attention at high resolution, and a global residual over a cheap interpolation so the network predicts only detail.
3. *Free at inference.* Structural re-parameterization: train multi-branch blocks (3x3, 1x1, identity) and fold them into a single 3x3 convolution for deployment. Knowledge distillation from a larger teacher.
4. *Compression.* Channel pruning, then int8 quantization, with quantization-aware training if post-training quantization costs too much quality. SR outputs are dense intensities, so quantization error shows up directly as banding or lost PSNR.
5. *System.* Run the network on the luma (Y) channel only and upsample chroma bilinearly; tile large images with overlap; fuse operators and compile with a vendor runtime such as TensorRT; use fp16; for video, reuse computation across frames or skip static regions.
6. *Re-validate on the device:* quality-latency curves, seams between tiles, and behaviour on real degraded inputs, not only bicubic benchmarks.

[more](restoration-super-resolution.html)
:::

::: {.qa asked="vivo written test, Nowcoder 2022"}
**How do dual cameras improve image quality, and how does that differ from taking two shots with one camera?**

Two cameras capture two views at the same instant from different positions, often with different optics or sensors; two shots from one camera share viewpoint and optics but are separated in time. Dual cameras therefore exchange motion problems for parallax problems, and each setup buys different gains.

*What a second camera adds.*

- Wide plus telephoto: optical zoom at the tele focal length, and tele detail fused into the centre of the wide image at intermediate zoom levels.
- Colour plus monochrome: the mono sensor has no colour filter array, so each pixel gathers more light and needs no demosaicing; fusing its luminance with the colour camera's chroma lowers noise and improves detail in low light.
- Stereo: disparity between the views gives depth, used for synthetic background blur and segmentation.
- Simultaneous captures with different exposures, giving HDR without motion between them.

*What a burst from one camera adds.* Averaging $N$ aligned frames with independent noise divides the noise standard deviation by $\sqrt N$; exposure bracketing gives HDR; hand tremor shifts the frames by sub-pixel amounts that multi-frame super-resolution can exploit (Wronski et al., 2019). The problems are temporal: hand shake and moving objects require robust alignment and a merge that rejects mismatched pixels, or the result shows ghosting.

*Problems specific to two cameras.* Parallax depends on depth, so alignment is per-pixel disparity estimation, not a global warp. Regions occluded in one view must come from the other camera alone. The cameras differ in focal length, distortion, colour response, noise and focus, so they need calibration and photometric matching, and seams can appear where the tele field of view ends. Depth from a short baseline is coarse, so synthetic blur fails around hair and thin structures. The two approaches are complementary: a system can burst on each camera and then fuse across cameras.
:::

::: {.qa asked="ByteDance graphics/image, Nowcoder 2021"}
**Which classical image-enhancement algorithms do you know?**

Group them by what they correct: global contrast, local contrast, tone, illumination, colour cast, sharpness and haze.

1. *Histogram equalization.* Map intensity $r$ through its cumulative distribution, $s = T(r) = (L-1)F_r(r)$. For a continuous density and increasing $T$, $p_s(s) = p_r(r)\,\lvert dr/ds\rvert = p_r(r)/\big((L-1)p_r(r)\big) = 1/(L-1)$, a uniform output. The discrete version $s_k = \mathrm{round}\big((L-1)\sum_{j \le k} n_j / N\big)$ is only approximately flat. Fails: it over-stretches large uniform areas, amplifies noise, and shifts colours if applied per RGB channel, so apply it to luminance.
2. *CLAHE.* Equalize per tile, clip each tile's histogram at a limit and redistribute the excess (which bounds the slope of the mapping and so the noise gain), and interpolate the tile mappings bilinearly to hide seams. Fails: halos and boosted noise in flat regions at high clip limits.
3. *Gamma and tone curves.* $s = r^\gamma$ with $\gamma < 1$ brightens shadows; an S-curve adds mid-tone contrast. Cheap and global, so they cannot fix local problems.
4. *Retinex.* Model $I = R \cdot L$ (reflectance times illumination), estimate $L$ with a large Gaussian blur, and output $\log I - \log(G_\sigma * I)$; multi-scale versions average several $\sigma$. Fails: halos at strong edges where the smooth illumination estimate is wrong, washed-out colours, noise boosted in dark areas.
5. *Unsharp masking.* $I + k\,(I - G_\sigma * I)$ adds back high frequencies. Fails: overshoot halos at edges and amplified noise.
6. *White balance.* Gray world scales each channel so the channel means are equal; white patch scales so the brightest pixel becomes white. Fails: scenes dominated by one colour (grass, sunsets) or with clipped highlights.
7. *Dark-channel dehazing* (He et al., 2009). With the haze model $I = J t + A(1-t)$ and the observation that haze-free outdoor patches outside the sky have some channel close to zero, estimate $t(x) = 1 - \omega \min_c \min_{y \in \Omega(x)} I^c(y)/A^c$ with $\omega = 0.95$ to keep a little haze, take $A$ from the pixels with the brightest dark channel, refine $t$ with soft matting, and recover $J = (I - A)/\max(t, t_0) + A$ with $t_0 = 0.1$. Fails: large white or sky-coloured objects such as snow or white walls.

These methods are fast and interpretable, and remain useful for preprocessing, data augmentation and baselines.
:::

::: {.qa asked="OPPO 2021, NetEase 2021 (Nowcoder)"}
**How do you convolve an image with a very large kernel quickly?**

Direct convolution of an $H \times W$ image with a $k \times k$ kernel costs $O(HWk^2)$ per channel. Faster methods exploit structure in the kernel or move to the frequency domain.

1. *Separable and low-rank kernels.* If $K = u v^\top$ has rank 1 (Gaussian, box), filter the rows with $v$ and the columns with $u$: $O(2HWk)$. A general kernel can be approximated by its top $r$ singular components, $K \approx \sum_{i=1}^{r} \sigma_i u_i v_i^\top$, at $O(2rHWk)$.
2. *FFT.* By the convolution theorem, $x * K = \mathcal{F}^{-1}(\mathcal{F}x \cdot \mathcal{F}K)$, costing $O(N \log N)$ with $N = (H+k-1)(W+k-1)$, independent of $k$ otherwise. The DFT computes circular convolution, so zero-pad both arrays to at least $(H+k-1) \times (W+k-1)$ to avoid wrap-around (reflect-pad the image first if you want mirrored borders), and pick sizes with small prime factors. For very large images use overlap-add or overlap-save in blocks. For small kernels direct or Winograd convolution is faster; the crossover depends on the hardware, so benchmark.
3. *Box filters with an integral image.* With $S(x, y) = \sum_{x' \le x,\, y' \le y} I(x', y')$, any box sum is $S(x_2, y_2) - S(x_1 - 1, y_2) - S(x_2, y_1 - 1) + S(x_1 - 1, y_1 - 1)$: $O(1)$ per pixel for any box size.
4. *Gaussians from repeated boxes.* A box of odd width $w$ has variance $(w^2 - 1)/12$; by the central limit theorem, $n$ passes approximate a Gaussian of variance $n(w^2 - 1)/12$, and three passes are usually close enough.
5. *Recursive (IIR) Gaussian filters.* A few-tap causal and anticausal recursion along each axis: $O(HW)$ independent of $\sigma$.
6. *Pyramids.* Downsample, filter with a smaller kernel, upsample: a blur of $\sigma$ becomes $\sigma/2^l$ at level $l$ on $4^{-l}$ of the pixels, at some accuracy cost.

In deep networks, large receptive fields come from dilated convolutions ($k$ taps spanning $d(k-1)+1$ pixels), stacks of small kernels, or large depthwise kernels (ConvNeXt's 7x7, RepLKNet's re-parameterized 31x31), which cost $O(HWk^2C)$ instead of $O(HWk^2C^2)$ for $C$ channels.
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

::: {.qa asked="Image-gen intern 2024, ByteDance MM 2025, Xiaohongshu"}
**Compare BLIP, BLIP-2, ALBEF, LLaVA and MiniGPT-4.**

They trace a path from vision-language models pretrained end to end for retrieval and VQA (ALBEF, BLIP) to vision encoders attached to a frozen or fine-tuned LLM for open-ended chat (BLIP-2, MiniGPT-4, LLaVA). Compare them on what is trained, where vision and text meet, and the training signal.

| Model | Where vision meets text | What is trained | Objectives and data |
|---|---|---|---|
| ALBEF (2021) | ViT-B/16; text encoder (first 6 BERT layers), then a cross-attention fusion encoder (last 6) | everything | contrastive ITC before fusion, ITM with hard negatives, MLM, momentum distillation |
| BLIP (2022) | unimodal encoders, an image-grounded encoder and an image-grounded decoder sharing all weights except self-attention | everything | ITC, ITM and LM; CapFilt cleans and extends the captions |
| BLIP-2 (2023) | Q-Former with 32 learned queries between a frozen ViT and a frozen LLM | Q-Former and projection | stage 1 ITC, ITG and ITM; stage 2 LM |
| MiniGPT-4 (2023) | BLIP-2's frozen ViT and Q-Former, one linear layer into frozen Vicuna | the linear layer | about 5M caption pairs, then 3,500 curated detailed descriptions |
| LLaVA (2023) | every CLIP ViT-L/14 patch token, linearly projected into Vicuna | projection, then projection and LLM | 595K filtered CC3M pairs, then 158K GPT-4-generated instruction samples |

Points worth making:

- *ALBEF* aligns the unimodal embeddings before the fusion encoder, so cross-attention starts from comparable features; momentum distillation uses an EMA model's soft predictions as extra targets to tolerate noisy web captions.
- *BLIP* also fixes the data: a captioner (the decoder fine-tuned on COCO) writes captions for web images, using nucleus sampling because diverse captions helped more than beam search, and a filter (the ITM encoder) drops mismatched web and synthetic captions.
- *MiniGPT-4* showed that one linear layer on BLIP-2's Q-Former already gives fluent chat with a strong LLM, and that a small, clean second stage cures the repetitive, fragmented output of caption-only training.
- *LLaVA* keeps every patch token and fine-tunes the LLM on synthetic instruction data, the recipe most later open VLMs followed (with an MLP projector from LLaVA-1.5).

The trade-offs: retrieval-style models are small and rank well but do not converse; LLM-based models generate and reason but are larger; and compressing to a few tokens (Q-Former) versus keeping every patch (projector) trades cost against fine detail.

[more](text-to-image-multimodal.html)
:::

::: {.qa asked="Image-gen intern, company unnamed (Nowcoder 2024)"}
**How do you caption LoRA training data, and why use a captioner such as BLIP?**

Captions decide what the LoRA attaches to which words. Whatever in the images the caption does not explain must be absorbed by the new weights and becomes tied to the tokens that are always present (the trigger word, or the model's default output). Whatever the caption describes stays attributed to those words and remains controllable from the prompt. The rule: caption what should vary, and leave out what the LoRA should learn.

1. *Subject or character.* Use a rare identifier token plus the class noun (the DreamBooth convention, "a [V] dog"); describe pose, background, lighting, clothing and framing so they stay promptable; do not describe the identity features themselves, or they get attributed to those descriptive words instead of the trigger.
2. *Style.* Describe the content (objects, layout) so it can change, and leave out style words, so the style becomes the default or attaches to the trigger.
3. *Why a captioner.* Hand-captioning hundreds of images is slow and inconsistent. BLIP writes short, plain captions such as "a dog sitting on a couch", close to how people write prompts, which keeps training captions and inference prompts in the same distribution. Tag-style captioners suit checkpoints trained on tag prompts, and VLM captioners give dense descriptions that suit models with T5 or LLM text encoders.
4. *Fit the text encoder.* CLIP encoders read 77 tokens and truncate the rest, so long VLM captions lose their tail on SD 1.x and SDXL; T5-based models (SD3, FLUX) can use long, detailed captions.
5. *Always review.* Captioners hallucinate, miss the subject's distinctive attributes and repeat phrasings. Check a sample, fix errors, insert the trigger, and delete descriptions of what should be learned.

Test the result with prompts that vary what you captioned (pose, background) and prompts without the trigger, to check that the concept is controllable and has not leaked into the base model's behaviour.

[more](text-to-image-multimodal.html)
:::

::: {.qa asked="Xiaohongshu CV/multimodal intern (2021, 2024)"}
**How are vision and language aligned and fused in multimodal models, and does alignment actually help?**

Alignment makes the two modalities' representations comparable, so matched image and text land close together; fusion lets tokens of one modality read the other so the model can reason jointly. Most models use both.

*Alignment.*

- Contrastive dual encoders (CLIP, SigLIP, ALBEF's ITC): separate encoders and a symmetric InfoNCE or sigmoid loss on pooled embeddings. Cheap at retrieval time because embeddings are precomputed, but no token-level interaction.
- Image-text matching (ITM): a binary head on fused features, sharper with hard negatives.
- Projector pretraining (LLaVA stage 1): train only the projection, with a captioning loss, so visual features land in the LLM's input space.

*Fusion.*

- Single-stream: concatenate image and text tokens in one transformer, with shared or modality-specific weights.
- Cross-attention: text tokens attend to image features (ALBEF's and BLIP's fusion encoders). Flamingo inserts cross-attention layers into a frozen LLM with $\tanh(\alpha)$ gates initialized at $\alpha = 0$, so the LLM is unchanged at the start, and a Perceiver Resampler that reduces each image to 64 tokens.
- Query bottleneck: BLIP-2's Q-Former condenses the image into 32 tokens given to the LLM as soft prompts.
- Projector into the LLM: a linear or MLP map of every patch token; fusion then happens in the LLM's self-attention.

*Does alignment help?* It does when fusion must start from mismatched spaces or a frozen model. In ALBEF, adding the contrastive loss to MLM and ITM raised VQA from 71.40 to 73.29 and NLVR2 from 77.51 to 79.88. BLIP-2 without its representation-learning stage did much worse on zero-shot VQA, and with OPT it suffered catastrophic forgetting. LLaVA trained on ScienceQA without its alignment pretraining dropped from 90.92% to 85.81%. It is not always needed: Prismatic VLMs (Karamcheti et al., 2024) found that with LLaVA-1.5-style data, skipping the separate projector-alignment stage did not hurt (single-stage training scored slightly higher) and saved 20 to 25% of compute. So alignment matters most when downstream data are small or the LLM is frozen, and less when a large, diverse instruction set trains the LLM end to end. Contrastive alignment alone also does not give token-level grounding; that needs fusion.

[more](text-to-image-multimodal.html)
:::

::: {.qa asked="Tencent multimodal 2026 (Nowcoder), JD 2026 (repo)"}
**Describe the Qwen2.5-VL architecture.**

Qwen2.5-VL (Qwen Team, 2025, arXiv 2502.13923) is a ViT, an MLP merger and a Qwen2.5 LLM, released at 3B, 7B and 72B. Its changes from Qwen2-VL target efficiency at native resolution and timing in video.

1. *Vision encoder.* A ViT trained from scratch, with the same configuration in all three sizes (hidden size 1280, 32 layers, 16 heads) and 14x14 patches on images resized to multiples of 28. To keep cost linear in image size, most layers use window attention with windows of at most 112x112 pixels (8x8 patches); only four layers (indices 7, 15, 23 and 31) use full attention. It uses 2D RoPE, and RMSNorm and SwiGLU as in the LLM. For video, two consecutive frames form one 3D patch, halving the token count.
2. *Merger.* Features of each 2x2 group of neighbouring patches are concatenated and projected by a two-layer MLP to the LLM width (2048, 3584 or 8192), so the LLM receives one token per 28x28 pixels.
3. *LLM.* Initialized from Qwen2.5 and using M-RoPE with temporal, height and width indices. The temporal index is tied to absolute time, so the spacing between indices encodes real time, and training samples videos at dynamic frame rates.
4. *Native coordinates.* Boxes and points use the input image's actual pixel coordinates instead of normalized ones, so the model learns scale.

Training: the ViT is first trained alone (CLIP-style pretraining, then alignment with the LLM with only the ViT updated), then all parameters are unfrozen for multimodal pretraining at sequence length 8,192, then a long-context phase at 32,768 with more video and agent data, then SFT and DPO with the ViT frozen.

Points to discuss: window attention gives up some global interaction for linear cost, and the four full-attention layers restore it; the 4x token merge keeps long documents and videos affordable; absolute-time M-RoPE supports temporal grounding without timestamp tokens or extra heads.
:::

::: {.qa asked="Curated AIGC lists (Nowcoder 2025, AIGC repo)"}
**What are the ways to inject a control condition into a DiT, including a video DiT, and does a modern DiT still need RoPE?**

Match the mechanism to the shape of the condition: a global vector, a token sequence, or a spatially aligned map.

1. *adaLN(-Zero) modulation* for global signals (timestep, class, pooled text, frame rate, motion score, guidance scale): an MLP regresses per-block scale, shift and gate. Almost free, but one affine map for all tokens cannot carry spatial or word-level information.
2. *Cross-attention* to a token sequence (text tokens, image-encoder tokens of a reference image): cost $O(NM)$ with $M \ll N$ condition tokens, which stay fixed across layers. PixArt-$\alpha$ and Wan use it for text.
3. *Joint (in-context) attention* over concatenated tokens: MMDiT's text tokens, CogVideoX's text tokens with expert adaLN, FLUX.1 Kontext's context-image tokens. Two-way interaction and conditions of any size or aspect ratio, at $(N+M)^2$ attention cost; Kontext found it worked better than channel concatenation for editing.
4. *Channel concatenation* for pixel-aligned inputs (mask and masked image for inpainting, the low-resolution input for SR, the first-frame latent plus a mask for image-to-video): widen the patch embedding with zero-initialized weights. The cheapest option, but only for aligned conditions.
5. *ControlNet-style branches* for structural maps (depth, pose, edges): a trainable copy of the first blocks whose outputs are added to the frozen model through zero-initialized layers. PixArt-$\delta$ adapted this to a DiT and chose a copy of the first 13 blocks. The base stays frozen and composable, at extra compute per step; lighter adapters add a small encoder's features to the tokens.

In a video DiT, per-frame structural controls (pose or depth sequences) go through channel concatenation or a control branch, and camera control often uses per-pixel Plücker ray embeddings.

*RoPE.* DiT used fixed 2D sine-cosine embeddings, which must be interpolated or extrapolated at other sizes, changing what each position means. RoPE encodes relative offsets inside attention along each axis, so variable resolution, aspect ratio and duration come naturally, and an offset can tag context tokens (Kontext puts context-image tokens at a constant offset on a virtual time axis). CogVideoX uses 3D RoPE (channels split 3:3:2 over $x$, $y$, $t$) and found it converged faster than sinusoidal absolute embeddings; HunyuanVideo and FLUX also use RoPE. A DiT does not strictly need it (SD3 uses 2D sinusoidal embeddings on an extended grid), but it is the common default for mixed-size image and video training.

[more](text-to-image-multimodal.html)
:::

::: {.qa asked="Curated AIGC list, Nowcoder 2025"}
**How does ReferenceNet differ from ControlNet, and how do you preserve the input image's features in image-to-image or image-to-video generation?**

ControlNet injects structure that is pixel-aligned with the output; ReferenceNet injects appearance from an image that is related to the output but not aligned with it.

*ControlNet* copies the U-Net encoder, feeds it a control map (edges, depth, pose) and adds its outputs to the frozen U-Net through zero convolutions. The addition is location by location, so it makes sense only when control pixel $(i, j)$ corresponds to output pixel $(i, j)$.

*ReferenceNet* (Animate Anyone, Hu et al., 2023) is a copy of the denoising U-Net without temporal layers, initialized from the same Stable Diffusion weights and trained separately, that encodes the reference image. In each self-attention layer of the denoiser, the reference feature map is concatenated with the denoiser's along the width, self-attention runs over both, and only the denoiser's half is kept. Every output location can pull detail from anywhere in the reference, which handles a character whose pose differs from the reference, and shared architecture and initialization put both feature maps in the same space. The reference features are extracted once per generation, not at every denoising step, so the inference overhead is modest. CLIP image features also enter by cross-attention for semantics, and pose enters through a light convolutional pose guider added to the noisy latent. The authors found ControlNet a poor fit here because the reference is not spatially aligned with the target.

*Tools for preserving the input:*

1. Channel concatenation of the image latent (inpainting, instruction editing, image-to-video with a mask): strong pixel fidelity where the input is aligned.
2. Reference attention (ReferenceNet, or the denoiser's own keys and values from a pass over the reference): detail without alignment.
3. Image embeddings in cross-attention (IP-Adapter's decoupled cross-attention; CLIP image features in SVD and Wan): identity and style, but CLIP sees a low-resolution image and encodes semantics, so fine detail is lost.
4. Inversion (DDIM or null-text) followed by editing, or SDEdit from a chosen noise level: the starting strength trades faithfulness against change.
5. For image-to-video, first-frame conditioning with mild noise augmentation of the conditioning latent, so the model neither ignores nor merely copies it.
6. Per-subject fine-tuning (LoRA, DreamBooth) when zero-shot preservation is not enough.
:::

::: {.qa asked="JD algorithm intern, Nowcoder 2024"}
**Describe DeepFloyd IF's structure and its advantages.**

DeepFloyd IF is an openly released, pixel-space cascade in the style of Imagen: a frozen T5 text encoder conditions three diffusion U-Nets that generate at 64x64 and then upsample to 256x256 and 1024x1024.

1. *Text encoder:* frozen T5-v1.1-XXL. Per the model card, its per-token outputs pass through a linear projection into hybrid self- and cross-attention in the U-Net, and an attention-pooled summary is added to the timestep embedding.
2. *Stage I:* 64x64 text-to-image, up to 4.3B parameters (IF-I-XL), trained with 1,000 diffusion steps, a cosine schedule and 10% caption dropout for classifier-free guidance.
3. *Stage II:* 64 to 256 super-resolution conditioned on the text and the stage-I image, up to 1.2B parameters (IF-II-L).
4. *Stage III:* 256 to 1024. The release used the Stable Diffusion x4 upscaler here, with its own stage-III model listed as coming later.

The repository reports a zero-shot FID-30K of 6.66 on COCO. Imagen (Saharia et al., 2022) explains the design: scaling the frozen text encoder helped more than scaling the U-Net, raters preferred T5-XXL to CLIP text encoders on hard compositional prompts, and super-resolution stages trained with noise-conditioning augmentation (the low-resolution input is noised and the noise level is passed as a condition) tolerate artifacts from the stage before.

*Advantages.*

- Language understanding from a large T5, which helps compositional prompts and written text; SD3 later measured that T5 matters most for typography.
- No autoencoder bottleneck: generating in pixels avoids the latent's reconstruction limit on small glyphs and fine detail.
- Modularity: each stage can be replaced or fine-tuned separately, and stage I is cheap per step at 64x64.

*Costs.* Three models run in sequence, each with its own steps and guidance; the super-resolution U-Nets work at high resolution in pixel space, which is expensive; layout errors made at 64x64 cannot be fixed later; T5-XXL alone is a multi-billion-parameter encoder; and the cascade needs per-stage tuning of noise augmentation and guidance. Later latent transformers with T5 encoders (SD3, FLUX) kept the language-model conditioning and dropped the cascade.
:::

::: {.qa asked="Xiaomi multimodal, Nowcoder 2024"}
**Survey current text-to-image and image-to-image methods.**

Organize methods by two axes: what is generated (pixels, continuous latents or discrete tokens) and how (iterative denoising or flow, versus autoregressive or masked token prediction); then by how the condition enters.

*Text-to-image families.*

1. Latent diffusion with a U-Net and cross-attention: Stable Diffusion 1.x and 2.x, SDXL. A mature ecosystem of LoRAs and ControlNets; weaker prompt following than newer models.
2. Latent transformers trained with diffusion or flow matching: PixArt-$\alpha$ (cross-attention), SD3 (MMDiT, rectified flow), FLUX.1 (double- then single-stream blocks). They scale predictably, follow text better with T5 encoders, and are the current default among open models.
3. Pixel-space cascades: Imagen, DeepFloyd IF. No autoencoder ceiling, but several expensive stages.
4. Token models over a VQ tokenizer: Parti (autoregressive) and Muse (masked, parallel decoding). They reuse language-model machinery; quality is limited by the tokenizer, and autoregressive decoding is sequential.
5. Unified models that both understand and generate: Chameleon (discrete tokens), Transfusion (language-model loss plus diffusion in one transformer), Janus, BAGEL. They add world knowledge and interleaved text-image output.

*Image-to-image methods.*

1. SDEdit: noise the input to an intermediate level and denoise; one knob trades faithfulness for realism, and no training is needed.
2. Inversion-based editing: DDIM or null-text inversion, then attention control such as Prompt-to-Prompt; good preservation, but slower and sensitive to guidance.
3. Instruction editing: InstructPix2Pix (synthetic before-and-after pairs, source image concatenated on channels, separate image and text guidance) and successors trained on larger edit datasets.
4. Structure and reference control: ControlNet and T2I-Adapter for spatial structure, IP-Adapter for appearance from an image.
5. Inpainting and outpainting with a mask, the mask and masked image concatenated on channels.
6. In-context editing: the source image becomes context tokens of a DiT (FLUX.1 Kontext) or is read by a VLM alongside the VAE (Qwen-Image's editor).

Compare them on fidelity to the input versus how far an edit can go, whether training per task or subject is needed, step count and latency, and text rendering or multi-image consistency if the product needs them. For evaluation, pair human preference or FID with alignment scores (CLIP or VQA-based) and preservation metrics for edits.

[more](text-to-image-multimodal.html)
:::

::: {.qa asked="Curated AIGC interview guide, Nowcoder 2024–25"}
**What is aspect-ratio bucketing, and why is it used?**

Training images come in many shapes, but a batch tensor needs one. Bucketing defines a set of resolutions with different aspect ratios and roughly equal pixel counts, assigns each image to the closest one, and draws every batch from a single bucket, so images keep their framing with little cropping and no padding.

SDXL's version: buckets with pixel counts as close to $1024^2$ as possible and height and width in multiples of 64; each batch comes from one bucket, with the bucket alternating between steps; and the bucket size $(h_{\text{tgt}}, w_{\text{tgt}})$ is given to the model as a Fourier-embedded condition, so the user can request an aspect ratio. SDXL applies this as a fine-tuning stage after pretraining at a fixed resolution and aspect ratio.

A typical implementation:

1. Enumerate buckets: for each width in steps of 64, take the height (also a multiple of 64) whose area is closest to the target, up to a maximum aspect ratio.
2. Put each image in the bucket with the nearest aspect ratio, resize it to cover the bucket, and crop the small remainder.
3. Sample batches one bucket at a time, choosing buckets in proportion to their size, and shuffle within buckets.

Sides must be divisible by the model's total downsampling factor: the VAE's 8x times the U-Net's internal downsampling (3 halvings in SD 1.x, giving 64; 2 in SDXL, giving 32) or times the patch size in a DiT. Because the pixel count is about constant, cost per step and the effective noise level stay comparable across buckets.

*Why not center-crop to squares?* Crops cut off heads and feet, and the model learns that framing: SDXL showed earlier Stable Diffusion models generating cut-off subjects. Square training also loses the composition of wide and tall images and leaves the model untrained at non-square sizes. Padding wastes compute and can teach borders. Alternatives: SDXL's crop conditioning (keep random crops but give the model the crop offsets, then request $(0, 0)$ at inference), and in transformers, packing variable-size token sequences (NaViT-style, as in CogVideoX's frame packing). Watch for rare buckets that give small or repetitive batches.

[more](text-to-image-multimodal.html)
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

::: {.qa asked="Alibaba Taotian AIGC, Nowcoder 2025"}
**What is your technical assessment of SD3 and Sora?**

Assess both the same way: what is claimed, what the evidence supports, what is unknown, and what you would test. The contrast is that SD3 is a paper with controlled ablations and released weights, while the Sora report describes the approach qualitatively without model details.

*SD3 (Esser et al., 2024).*

- Claims: rectified flow with logit-normal timestep sampling beats other diffusion formulations; MMDiT (separate weights per modality, joint attention) beats DiT-style, cross-attention and UViT backbones; quality scales predictably.
- Evidence: 61 training formulations ranked across datasets and metrics, where rectified flow with lognorm(0, 1) ranked consistently well; a scaling study up to 8B parameters and $5 \times 10^{22}$ training FLOPs in which lower validation loss correlated with GenEval and human preference; ablations supporting the 16-channel autoencoder, QK-normalization for stability, a 50/50 mix of synthetic and original captions, and T5 (dropping it at inference left aesthetics unchanged at a 50% win rate but cut typography to 38%).
- Limits: the formulation study ran at moderate scale, human-preference results depend on the prompt set, and the three text encoders (including a 4.7B-parameter T5) are heavy.

*Sora (technical report, 2024).*

- Claims: a video compression network plus a diffusion transformer on spacetime patches, trained jointly on images and videos at native durations, resolutions and aspect ratios; up to a minute of high-definition video; descriptive re-captioning and GPT prompt expansion; quality improving with training compute; 3D consistency and object permanence that hold "often, though not always".
- Evidence: qualitative samples, including the same prompt at base, 4x and 16x compute and a square-crop versus native-aspect comparison.
- Unknown: size, data, objective details and quantitative metrics; the report says model and implementation details are not included.
- Admitted limits: physics such as glass shattering, state changes such as eating, and incoherence in long samples.

*Comparison and tests.* Both bet on the same ingredients: transformers over patch tokens, synthetic captions and scale. SD3 backs its bet with controlled evidence, while Sora's claims of emergent simulation are demonstrations, and later benchmarks found physical accuracy largely unrelated to visual realism. To test either, I would measure prompt adherence on compositional and text-rendering prompts, compare held-out loss with human preference across scales, and, for video, use controlled physical scenarios and long-duration consistency checks.

[more](video-generation.html)
:::

::: {.qa asked="Curated AIGC interview guide, Nowcoder 2024–25"}
**Describe CogVideoX's architecture, and how Sora handled training data and captions.**

*CogVideoX (Yang et al., 2024)* is a text-to-video diffusion transformer released at 2B and 5B parameters, with four main components.

1. *3D causal VAE.* 3D convolutions with all temporal padding on the past side compress 8x8 in space and 4x in time. It is trained with L1, LPIPS and KL losses (plus a 3D GAN loss later), first on 17-frame clips, then on 161 frames split across devices, which causality makes cheap. Their ablation found 16x16x8 compression hard to converge even with more channels.
2. *Expert transformer.* T5 text embeddings and patchified video latents are concatenated into one sequence. Because the two modalities have different feature statistics, separate vision and text "expert" adaLN modulations, both driven by the timestep, are applied to their tokens; at equal parameter count this beat MMDiT in their ablation.
3. *3D full attention* over all text and video tokens. They found separate spatial and temporal attention less stable and worse for large motion, because a moving object's tokens in adjacent frames cannot attend to each other directly.
4. *3D RoPE.* 1D RoPE on $x$, $y$ and $t$ using 3/8, 3/8 and 2/8 of the channels, extended rather than rescaled for new sizes; it converged faster than sinusoidal absolute embeddings.

Training uses v-prediction with zero terminal SNR; mixed-duration batches (multi-resolution frame packing, after NaViT's sequence packing); progressive resolution from 256 to 512 to 768 px; and explicit uniform sampling, which gives each data-parallel rank its own interval of timesteps so the batch covers $[1, T]$ evenly and the loss is more stable. Captions come from a pipeline in which a video model writes a short caption, CogVLM writes dense captions of frames, and GPT-4 summarizes them; a fine-tuned Llama 2 and then an end-to-end CogVLM2-Caption model replaced the expensive steps.

*Sora, per its report,* trains on images and videos at native durations, resolutions and aspect ratios instead of resizing, cropping or trimming them, which the report says improves framing and composition and lets one model sample from widescreen to vertical. It captions all training videos with a highly descriptive captioner (DALL-E 3's re-captioning technique), reported to improve text fidelity and overall quality, and at inference GPT expands short user prompts into the long captions the model was trained on. Data sources, size and filtering are not disclosed.

[more](video-generation.html)
:::

::: {.qa asked="Curated AIGC interview repo (video section)"}
**U-Net or DiT for video diffusion: what are the trade-offs?**

U-Nets inflated from image models are cheap to start from and efficient at high resolution; DiTs treat every token the same way, allow full space-time attention and scale more predictably, at a cost quadratic in the token count. Most recent large video models are DiTs.

*Inflated U-Nets* (Video Diffusion Models, Make-A-Video, AnimateDiff, Stable Video Diffusion) add temporal layers (temporal attention, temporal convolutions or both) after the spatial layers of an image U-Net, often initialized as identities or with zeroed outputs.

- Pros: they reuse a strong image prior and, in AnimateDiff's plug-in motion module, its community fine-tunes; the multiscale hierarchy puts attention at low resolution, keeping high-resolution cost manageable; convolutions add locality and translation equivariance, which helps data efficiency.
- Cons: temporal attention links only tokens at the same spatial position and temporal convolutions are local, so large motion must propagate through many layers; global temporal context and long clips are limited; heterogeneous blocks are harder to scale and parallelize; and text enters only through cross-attention.

*Video DiTs* (CogVideoX, HunyuanVideo, Wan, and Sora as described in its report) cut a compressed spacetime latent (a causal 3D VAE in the open models) into patch tokens and run a uniform transformer.

- Pros: full 3D attention lets any token reach any other, which CogVideoX found better and more stable than factorized attention for large motion; text can join the attention; variable resolution, duration and aspect ratio come from token sequences, packing and 3D RoPE; scaling with compute is predictable (SD3's image scaling study, Sora's samples at base, 4x and 16x compute); and LLM tooling applies (FlashAttention, sequence and context parallelism).
- Cons: attention cost grows with the square of the token count, so DiTs rely on strong VAE compression (commonly 4x8x8) and 2x2 patches, and long or high-resolution clips need sparse or windowed attention, caching or distillation; with no built-in multiscale prior they typically need more data and compute.

*Choosing.* With a limited budget and an existing image U-Net ecosystem (short clips, stylized animation, image-to-video), inflating is practical. For a new foundation model trained at scale, a DiT with full or sparse 3D attention is the default. Hybrids exist, such as factorized-attention DiTs (Latte compares several variants) and windowed attention in restoration transformers (SeedVR).

[more](video-generation.html)
:::

::: {.qa asked="Curated AIGC interview repo (video section)"}
**In image-to-video generation, why doesn't a faithful first frame guarantee consistent later frames?**

The condition constrains only the frame it is attached to. Every later frame is sampled by the model, so the first frame's appearance must be carried forward through the network while the motion prior moves content around, and several things break that chain.

1. *Appearance must travel through attention.* With channel concatenation or first-latent replacement, the image enters at frame 0, and later frames see it only through attention. Under factorized attention, temporal attention links only tokens at the same position, so once an object moves its appearance must hop through other tokens (CogVideoX's argument for full 3D attention). Even with full attention, a logo or a face covers few tokens, so errors there cost little in a loss averaged over the clip.
2. *New content has no source.* Turning heads, camera motion and disocclusion reveal regions the first frame never showed; the model has to invent them, and its guess need not match the identity.
3. *Latents are not frame-aligned.* In a causal VAE with 4x temporal compression, the first latent encodes one frame alone and each later latent encodes four frames together, so even a perfect first latent does not determine the others; the model must translate it.
4. *Train-test mismatch.* In training, the condition is the clip's own first frame, from the same camera and distribution. At inference users bring sharp stills, generated images or unusual styles, and the motion prior pulls later frames back toward the training distribution (generic faces, video-like texture). A high text-guidance scale also pushes frames toward the prompt and away from the image.
5. *Accumulation.* In long or autoregressive generation, each chunk is conditioned on generated frames, so drift compounds.

Fixes aimed at these causes:

- Give every frame direct access to the condition: repeat the image latent along time (SVD), reference attention from every frame to the condition image's features (ReferenceNet-style), and image embeddings in cross-attention for identity (CLIP features in SVD and Wan).
- Noise-augment the conditioning latent in training so the model treats it as evidence rather than copying it; too little noise gives static videos, too much loses detail.
- Separate image and text guidance scales, or stronger guidance for later frames (SVD raises guidance linearly across frames).
- Training clips with real motion whose first frame matches the rest; keyframe or last-frame conditioning so later frames also have an anchor; identity or consistency rewards in post-training.

[more](video-generation.html)
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

::: {.qa asked="Meta AI research design round, Blind 2024"}
**What is the difference between offline and online metrics, and how do you choose them?**

Offline metrics are computed on fixed held-out data before deployment; online metrics measure what users do on live traffic, usually in an A/B test. Offline metrics are cheap, fast and reproducible, so they drive iteration; online metrics are closer to the goal, but slow, noisy and risky, so they decide launches.

They disagree for four main reasons:

1. **Proxy mismatch.** The offline label (a click, a rating) is not the goal (long-term satisfaction, retention).
2. **Feedback loops.** Logged data come from the current system, so items it never showed have no labels; offline evaluation then favours models that resemble the incumbent. Counterfactual estimators such as inverse propensity scoring correct this partly.
3. **Distribution shift.** Traffic changes over time, and partly because of the model itself.
4. **System effects.** Latency, diversity, interactions between items on one page, and novelty effects do not appear in per-item offline scores.

Choose one primary metric tied to the goal, a few guardrails that must not regress (latency, errors, safety, revenue), and diagnostic metrics for debugging. The primary metric must be sensitive enough to move within the test's duration, which a power analysis checks. Then validate the offline metric: across past launches, check whether offline deltas predicted the sign and ranking of the online deltas. If they did not, fix the offline metric before using it as a gate.

Examples. Ranking model: offline NDCG@k and log loss on logged impressions; online click-through, dwell time and retention, with latency as a guardrail. Image generator: offline FID or a DINOv2-based distance on a prompt suite, alignment scores and blind side-by-side ratings; online download and regeneration rates and user reports, with cost per image as a guardrail.
:::

::: {.qa asked="Curated AIGC list, Nowcoder 2025"}
**How would you evaluate a diffusion model's generation quality in production?**

In production, evaluation is a release gate plus continuous monitoring, always relative to the model currently serving. It has four layers.

1. **Versioned prompt suites with fixed seeds**: prompts sampled from real (anonymized) traffic and stratified by category; targeted suites for known failures (text rendering, hands, counting, multiple people); and a red-team suite for safety. Keep a hidden held-out suite, because teams overfit to any suite they can see.
2. **Automatic metrics on each suite**: prompt alignment (VQA-based scores), aesthetic and technical quality predictors, artifact detectors (OCR match for requested text, hand and face checks), safety classifiers (nudity, violence, real-person likeness, logos), and diversity across seeds. Compare per category against production; averages hide regressions.
3. **Human side-by-side**: blind pairwise ratings against production with confidence intervals. These calibrate the automatic metrics; recheck that calibration whenever the model family changes.
4. **Online**: an A/B test on acceptance signals (downloads, saves, shares), dissatisfaction signals (regenerations, edits, abandonment, reports) and retention, with guardrails on p95 latency, GPU cost per output and safety-filter trigger rate.

Release through gates: no category regresses beyond a set margin, safety does not get worse, then a canary rollout. After launch, monitor prompt-distribution drift (new trends, languages), filter hit rates and complaint categories, and refresh the suites from new traffic. The main failure mode is optimizing an automatic metric until it stops tracking human judgment.

[more](evaluation-metrics.html)
:::

::: {.qa asked="Apple I2V research-intern job ad (not an interview)"}
**What is LPIPS, and how is it computed?**

LPIPS (Zhang et al., 2018) is a full-reference perceptual distance: compare deep features of two aligned images, with channel weights trained so the distance agrees with human similarity judgments.

1. Pass both images $x, x_0$ through a pretrained network (AlexNet, VGG or SqueezeNet) and take activations from $L$ layers, $\hat y^l, \hat y^l_0 \in \mathbb{R}^{H_l \times W_l \times C_l}$.
2. Unit-normalize the feature vector at each spatial position across channels.
3. Scale channels by non-negative learned weights $w_l$, take the squared $\ell_2$ difference, average over positions and sum over layers:

$$d(x, x_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \big\lVert w_l \odot (\hat y^l_{hw} - \hat y^l_{0,hw}) \big\rVert_2^2 .$$

With all $w_l = 1$ this is a cosine distance, since $\lVert a - b \rVert^2 = 2 - 2\cos(a, b)$ for unit vectors.

The weights are trained on BAPPS, a dataset of human judgments on $64 \times 64$ patches covering traditional distortions, CNN-based distortions and real algorithm outputs. In the 2AFC task, people see a reference and two distorted versions and pick the closer one. A small network maps the pair of distances $(d_0, d_1)$ to the probability of that choice and is trained with cross-entropy. Three variants: *lin* freezes the backbone and learns only $w$ (1,472 parameters for VGG), *tune* fine-tunes the whole pretrained network, and *scratch* trains from random initialization. The official code expects RGB inputs scaled to $[-1, 1]$.

Limitations: it needs a pixel-aligned reference, so small shifts change the score; it compares pairs, not distributions (unlike FID); scores differ by backbone, so report which you used; and because it is differentiable and often used as a training loss, evaluating with the same LPIPS flatters models trained on it.

[more](evaluation-metrics.html)
:::

::: {.qa asked="NVIDIA (PracHub 2026)"}
**What is model calibration, and how do you measure and fix it?**

A classifier is calibrated if, among predictions made with confidence $p$, a fraction $p$ is correct: $P(\hat Y = Y \mid \hat P = p) = p$. Calibration is separate from accuracy; a model can be accurate and still overconfident.

**Measuring it.** A reliability diagram bins predictions by confidence and plots accuracy against mean confidence per bin; perfect calibration is the diagonal. Expected calibration error summarizes it with $M$ bins $B_m$:

$$\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big| .$$

ECE has known flaws: it depends on the number and type of bins, sparse high-confidence bins are noisy, the usual version looks only at the top-class probability, and it is not a proper scoring rule. A model that predicts the base rate for every input has zero ECE and no skill, so report NLL or the Brier score alongside it. Nixon et al. (2019) found that the ranking of recalibration methods changes with these choices, and recommend adaptive bins and class-conditional variants.

**Why modern networks are overconfident.** Guo et al. (2017) found that greater depth and width, BatchNorm and less weight decay go with worse calibration, and observed that test NLL can overfit while test error keeps falling: the network keeps enlarging logits on examples it already classifies correctly.

**Fixing it.**

- Temperature scaling: $q = \mathrm{softmax}(z/T)$ with one scalar $T > 0$ fit by minimizing NLL on a held-out validation set. The argmax does not change, so accuracy is unchanged; $T > 1$ softens overconfident outputs. It was the most effective post-hoc method on most datasets in Guo et al.
- Platt scaling, $\sigma(az + b)$ for binary scores, and isotonic regression (a monotone, non-parametric map that needs more validation data).
- Training-time methods: label smoothing improves calibration (Müller et al., 2019), and deep ensembles give well-calibrated uncertainty (Lakshminarayanan et al., 2017).

Post-hoc calibration fit in-distribution degrades under dataset shift; in Ovadia et al. (2019), methods that average over several models, such as ensembles, held up better.
:::

::: {.qa asked="NVIDIA (PracHub 2026)"}
**What is data or model drift, and how do you detect and handle it?**

Drift means the production distribution moves away from the training distribution, so performance decays silently. Factor $p(x, y)$ to name the kinds:

- **Covariate shift**: $p(x)$ changes and $p(y \mid x)$ does not (a new camera, a new region).
- **Label shift**: $p(y)$ changes and $p(x \mid y)$ does not (a disease becomes more common in winter).
- **Concept drift**: $p(y \mid x)$ itself changes (fraud patterns adapt, the meaning of a label changes).

**Detecting it.**

1. Inputs: per-feature two-sample tests (Kolmogorov–Smirnov for continuous features, chi-squared for categorical ones) or the population stability index $\sum_b (q_b - p_b)\ln(q_b/p_b)$ over bins. For images or text, embed with a fixed encoder and use MMD with a permutation test, or train a domain classifier to separate reference from current data: an AUC well above 0.5 signals shift, and its most confident examples show what changed. Rabanser et al. (2018) found two-sample tests on representations from a pretrained classifier the best of the detectors they compared. Correct for multiple testing when monitoring many features.
2. Outputs: track the predicted-class and confidence distributions. To estimate label shift without labels, black-box shift estimation (Lipton et al., 2018) uses the source joint $C_{ij} = P(\hat y = i, y = j)$. Under label shift, $q(\hat y = i) = \sum_j C_{ij} w_j$ with $w_j = q(y = j)/p(y = j)$, so solve $\hat\mu = C w$.
3. Delayed labels: when labels arrive, track accuracy and calibration per slice. Concept drift is visible only here, or through audited samples.

**Handling it.** Alert on shifts that hurt performance, not on every shift. For covariate shift, reweight training data by $w(x) = q(x)/p(x)$; a domain classifier $c(x) = P(\text{target} \mid x)$ gives $w(x) = \frac{c(x)}{1 - c(x)}\cdot\frac{n_s}{n_t}$. For label shift, rescale outputs, $p_{\text{new}}(y \mid x) \propto w_y\, p(y \mid x)$. Otherwise retrain on a sliding window, on a schedule or when triggered, validate on recent data, deploy in shadow mode or as a canary, and keep a rollback.
:::

::: {.qa asked="NVIDIA data-science question (PracHub 2025)"}
**How would you preprocess and cross-validate a medical image classifier?**

The biggest risks are leakage and shift between sites, not the backbone. Split by patient, validate on data from another site, and report metrics at the clinical operating point.

**Preprocessing.**

- CT values are in Hounsfield units: clip to a task window (a lung or a soft-tissue window) and rescale. MRI intensities are not standardized across scanners, so normalize per volume (percentile clipping, z-score within a body mask), optionally after bias-field correction. For X-rays, handle bit depth and inverted DICOM photometric interpretation.
- Resample 3D volumes to a common voxel spacing and crop to the anatomy.
- Augment within anatomical limits: small rotations and scaling, contrast and noise jitter, mild elastic deformation. Avoid left-right flips when laterality matters, and crops that can cut out the lesion.
- Look for shortcuts such as burned-in text, markers or scanner signatures. Zech et al. (2018) found that pneumonia CNNs could identify the hospital system that produced a chest X-ray, and performed significantly worse on outside hospitals in 3 of 5 comparisons.

**Cross-validation.**

- Group by patient: follow-up scans, both eyes or several slices of one patient must stay in the same fold, or near-duplicates leak across folds. Use stratified group k-fold (stratify by label, group by patient).
- Fit normalization statistics and any feature selection inside each training fold only.
- Tune hyperparameters with nested CV, or keep a test set you use once.
- Report the mean and spread across folds; with small datasets folds vary a lot.
- Hold out whole sites or scanners (leave-one-site-out) for external validation.

**Imbalance and metrics.** Reweight or rebalance during training, but evaluate at the natural prevalence. Report AUROC and AUPRC, sensitivity at a fixed specificity chosen for the use case (screening favours sensitivity), bootstrap confidence intervals over patients, calibration (clinicians read the probabilities), and subgroup results by sex, age and scanner.
:::

::: {.qa asked="ByteDance multimodal campus, Nowcoder 2025"}
**For e-commerce image generation, how would you design the prompts and filter bad outputs?**

Let the product photo, not the text, carry the product: the prompt describes the scene and style, conditioning preserves the product, automatic filters remove failures, and human rejections feed back into templates and filters.

**Prompts and conditioning.**

1. Templates built from structured attributes (category, material, colour, audience) plus a scene and style chosen per category, with negative prompts for common defects.
2. An LLM rewrites templates into detailed captions in the style the model was trained on, with instructions not to add product attributes (no new text, logos or accessories). Generate several variants per product.
3. Fidelity comes from conditioning: segment the product from the merchant photo and generate the background around it, so product pixels stay exact; or use an image prompt or structural control (edges, depth); or a per-product LoRA for high-value items.

**Filters, cheap to expensive.**

- Product fidelity: embedding similarity (DINO or CLIP) between the product region and the source, masked LPIPS, colour difference, mask IoU for shape.
- Text and logos: OCR against the brand text; detect garbled text and spurious logos.
- Quality: artifact detectors (hands, warped geometry), no-reference quality and aesthetic scores.
- Compliance: safety classifiers and platform rules (no misleading claims, watermarks or other brands' logos).
- Brief compliance: VQA checks such as "is the product on a kitchen counter?".

Tune each threshold against a labelled set, measuring the filter's precision and recall, and rank the survivors. Have humans spot-check a sample per category, with more review for high-traffic items. Log rejection reasons, cluster them, and fix the template, negative prompt or filter responsible; accepted and rejected pairs can later train a preference-tuned model. Online, compare click-through, conversion and return rate against the original photos. Returns matter: an image that flatters the product too much raises them.

[more](evaluation-metrics.html)
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

::: {.qa asked="Runway (Glassdoor 2025–26)"}
**Design a production system around a generative image or video model.**

**Requirements.** Who calls it (consumer app, API, internal tool), the outputs (resolution, duration), the latency target (seconds for an image preview, minutes may be acceptable for a video), peak throughput, the budget per generation, and safety and legal constraints.

**Data and training loop.** Licensed data with recorded provenance, a curation and captioning pipeline, a base model, then fine-tunes for product styles. Product logs (pairwise choices, regenerations, downloads) become preference data for reward models or preference fine-tuning, and every new checkpoint passes an offline evaluation gate before it ships.

**Request path.** API, then prompt safety check, then optional LLM prompt rewriting, then a job queue, GPU workers, output safety check, and storage behind a CDN.

**Serving levers.**

- Fewer steps: better solvers, step distillation, guidance distillation (which removes the second classifier-free-guidance pass).
- Cheaper steps: lower-resolution previews followed by an upscaler, cached text-encoder outputs and reuse of features across steps, FP8 or INT8 quantization, fused attention kernels, sequence parallelism across GPUs for long videos.
- Batching: group requests with the same shape and step count; dynamic batching trades latency for throughput.
- Queues: asynchronous jobs with progress updates for video, priority tiers, autoscaling on queue depth, warm GPU pools (loading a model takes a long time), cheaper preemptible GPUs for batch work.

Cost per output is about GPU-seconds per output times the GPU price, divided by utilization, so step distillation and higher utilization cut cost directly.

**Safety.** Prompt and output classifiers, likeness and IP checks, provenance metadata or watermarks, rate limits and abuse monitoring.

**Evaluation and monitoring.** Offline gates and A/B tests (see the production-evaluation question above), plus p95 latency, failure and out-of-memory rates, GPU utilization, safety-filter rates and prompt drift. Store the seed, model version and configuration with every output so support can reproduce it.

**Failure modes and iteration.** Traffic spikes at launches, over-blocking filters, preview and final outputs that differ. Mine failure clusters from logs, fix them with targeted data or fine-tuning, and re-run the gates.
:::

::: {.qa asked="Meta RS full-time, Glassdoor"}
**Design a system that detects harmful posts.**

**Requirements.** Start from the policy: a taxonomy of harm types (violence, hate, self-harm, nudity, spam, scams) with an action for each (remove, reduce distribution, add a warning, send to review). Posts are multimodal and multilingual. Decide where enforcement happens (before publishing or after) and the error trade-off per class: the most severe classes need very high recall, while borderline speech needs high precision because wrongful removal is costly.

**Data.** Labels come from trained reviewers following written guidelines; measure agreement between reviewers. User reports are plentiful but noisy and biased toward visible content. Prevalence is low, so random samples contain few positives; label with active learning around the decision boundary, and estimate prevalence with score-stratified sampling and importance weights so the estimate stays unbiased.

**Model, as stages.**

1. Cheap matching: perceptual hashes of known violating images and videos, URL and text blocklists.
2. A multimodal classifier (text and image encoders, fused) with one head per policy class, plus context features: account age, past violations, how the post spreads. Many harmful memes are harmful only through the combination of text and image.
3. An LLM given the policy text can classify new or rare policy areas zero-shot, and can be distilled into cheaper models once validated.
4. Per-class thresholds: automatic action above a high threshold, human review in a middle band, with the review queue ordered by expected harm times expected reach.

**Evaluation.** Offline: precision and recall at the operating thresholds, per class and per language. Online: prevalence (the share of views that land on violating content, estimated by sampling), time to action, reviewer and appeal overturn rates, and false-positive rates for different groups and dialects as a fairness guardrail.

**Failure modes and iteration.** Adversarial evasion (misspellings, text inside images), new slang and events, missing cultural context, removal of news or counter-speech, label noise. Retrain often, red-team regularly, and keep a fast path for adding new hashes during a crisis.
:::

::: {.qa asked="Meta RS full-time, Reddit 2026"}
**Design a recommender for places to visit.**

**Clarify first.** Tourists planning a trip or locals choosing something now; the surface (map, feed, search); and what success is (a visit, a save, a good rating). Location, time and opening hours constrain this recommender far more than a feed of videos.

**Data.** Places: category, coordinates, hours, price, photos, reviews, popularity. Users: visits inferred from location history (opt-in only), check-ins, saves, searches, ratings. Context: current location, time, weather, at home or travelling, group size. A visit is noisy implicit feedback; dwell time and revisits are stronger signals.

**Candidate generation.** Filter to places reachable in reasonable travel time and open at the relevant time, then union several sources: popular nearby places per category, co-visitation ("people who visited A also visited B"), a two-tower model (user and context tower against a place tower) served from an approximate-nearest-neighbour index, and places similar to past favourites. A few hundred candidates.

**Ranking.** Predict the probability of a visit, or expected satisfaction, from travel time, opening hours, the user's category affinity, place quality (a Bayesian average of ratings, so places with three reviews do not top the list), price fit and time-of-day fit, using gradient-boosted trees or a neural ranker, optionally with several objectives (visit, save, rating).

**Re-ranking.** Diversity (not five cafés), removing closed places, route feasibility for trip itineraries, and a small exploration budget for new places.

**Cold start.** New users: context, popularity and a few onboarding questions. New places: content features (category, photo and text embeddings) plus exploration.

**Evaluation.** Offline: recall@k and NDCG on later visits with a time-based split, keeping in mind that logged data are biased by what was shown and where. Online: visit and save rates, navigation requests, retention, diversity; latency as a guardrail.

**Failure modes.** Popularity bias that floods the same places, stale data (closed businesses), seasonality, sparse data in rural areas, and privacy of location history.
:::

::: {.qa asked="Microsoft AS intern, PracHub 2026"}
**Design a retrieval-augmented generation (RAG) system.**

RAG retrieves passages at query time and has an LLM answer conditioned on them (Lewis et al., 2020), so answers can use private or fresh data and cite sources. Many failures start in retrieval, so design and evaluate retrieval and generation separately.

1. **Ingestion and chunking.** Parse PDFs, HTML and tables; keep metadata (source, date, permissions). Split along document structure into chunks of a few hundred tokens with small overlap, and prepend the title and section path. Small chunks lose context; large ones dilute the embedding and waste context.
2. **Indexing.** Dense embeddings in an approximate-nearest-neighbour index plus a BM25 index: dense retrieval handles paraphrases, BM25 exact names, codes and rare terms. Merge them with reciprocal rank fusion or a weighted score.
3. **Query processing.** Rewrite conversational turns into standalone queries, split multi-hop questions, and apply metadata filters (permissions, dates).
4. **Reranking.** A cross-encoder reads query and passage together and cuts the top 50–100 to 5–10; more accurate than the bi-encoder, at a cost per pair.
5. **Prompting.** Instruct the model to answer only from the context, cite chunk IDs and say when the answer is not there. Keep the context short and put the best passages first: Liu et al. (2023) found models use information at the start or end of a long context better than in the middle.
6. **Checking.** Verify that each cited passage supports its claim.

**Evaluation.** Retrieval: recall@k and MRR or nDCG on questions from real queries with labelled relevant passages (synthetic questions generated from chunks can bootstrap this). Generation: faithfulness to the retrieved context, answer relevance and correctness, scored by LLM judges calibrated against human labels; reference-free frameworks such as RAGAS (Es et al., 2023) cover the first two. Online: thumbs up or down, follow-up questions, escalations.

**Failure modes.** The fact was never retrieved (chunking split it, vocabulary mismatch); it was retrieved but ignored or overridden by the model's memory; documents conflict or are stale; permissions leak; documents carry hidden instructions (prompt injection). For freshness, index incrementally, propagate deletions, store timestamps, and version the index, since a new embedding model means re-embedding the corpus.
:::

::: {.qa asked="Netflix ML intern, Jointaro 2025"}
**How would you combine an LLM with a knowledge graph, for example for named-entity resolution?**

Use each for what it does well. The knowledge graph holds canonical entities, identifiers, typed relations and constraints; the LLM reads messy text and judges whether two descriptions refer to the same thing. For entity resolution, put the LLM inside the classic blocking, matching and clustering pipeline, with the graph supplying candidates, context and constraints.

1. **Blocking.** For each mention (a title or a person's name in a document or a partner feed), generate candidates cheaply: alias tables, fuzzy string matching, nearest neighbours over embeddings of entity names and descriptions. Recall here caps the recall of the whole system.
2. **Context from the graph.** For each candidate, pull a compact subgraph: type, dates, related entities. This separates entities that share a name.
3. **Matching.** Prompt the LLM with the mention's context and the serialized candidates, and constrain the output to a candidate ID or "none" (a new entity), so it cannot invent entities. At scale, a fine-tuned cross-encoder handles most cases and the LLM only the uncertain ones. Shuffle candidate order, since LLMs can favour the first option.
4. **Constraints and clustering.** Reject matches that violate graph constraints (type mismatch, impossible dates). For deduplicating records, make decisions transitive by clustering pairwise match scores rather than accepting pairs independently.
5. **Review.** Send low-confidence or high-impact merges to people, and keep provenance, because wrong merges are hard to undo.

The integration also runs in the other two directions: retrieve graph facts as context so the LLM answers with fewer hallucinations, and use the LLM to extract candidate triples from text that are validated before entering the graph.

**Evaluation.** Pairwise precision, recall and F1 on labelled pairs; cluster-level metrics such as B-cubed for deduplication; linking accuracy including "none" cases; and cost per resolved entity. Cache results and call the LLM only where cheaper models are unsure.
:::

::: {.qa asked="Meta Reality Labs-type intern; Ant digital human"}
**How does visual SLAM work, and where do learned 3D methods fit in?**

Visual SLAM estimates the camera trajectory and a scene map together: poses come from aligning images to the map, and the map from triangulating with the poses.

1. **Tracking.** Estimate each frame's pose against the map. Feature-based methods match keypoint descriptors (such as ORB) to map points, solve PnP inside RANSAC and refine the reprojection error. Direct methods minimize the photometric error of pixels with gradient over pose and depth; DSO (Engel et al., 2016), a direct odometry system, also models exposure and vignetting.
2. **Mapping and bundle adjustment.** Select keyframes, triangulate new points, and jointly optimize poses $T_i$ and points $X_j$ by minimizing $\sum_{ij} \rho\big(\lVert u_{ij} - \pi(T_i X_j) \rVert^2\big)$, with observed pixel $u_{ij}$, projection $\pi$ and robust loss $\rho$; solve with Levenberg–Marquardt, eliminating the points by a Schur complement.
3. **Loop closure.** Recognize a revisited place, estimate the relative pose, and optimize the pose graph to spread accumulated drift. ORB-SLAM (Mur-Artal et al., 2015) uses one set of ORB features for tracking, mapping, relocalization and loop closing.

A single camera fixes geometry only up to scale; stereo, depth or an IMU makes it metric. Feature-based methods tolerate lighting changes and wide baselines but give sparse maps; direct methods use more of the image but are sensitive to photometric changes and need small motion between frames.

**Learned methods.**

- Learned features, matchers and depth priors inside classical pipelines.
- Differentiable SLAM: DROID-SLAM (Teed and Deng, 2021) updates poses and per-pixel depth through a dense bundle adjustment layer.
- Neural-field maps: iMAP (Sucar et al., 2021) uses one MLP as the whole RGB-D map; NICE-SLAM (Zhu et al., 2021) uses hierarchical feature grids.
- Gaussian-splatting maps: SplaTAM (Keetha et al., 2023) for RGB-D, and Gaussian Splatting SLAM (Matsuki et al., 2023), monocular, which tracks by optimizing the pose against the Gaussians and runs live at about 3 fps.
- Reconstruction priors: DUSt3R (Wang et al., 2023) regresses pointmaps from uncalibrated image pairs; MASt3R-SLAM (Murai et al., 2024) builds monocular dense SLAM at about 15 fps on such a prior.

Classical tracking suits tight compute budgets, learned components add robustness, and neural or Gaussian maps matter when the map must be rendered.
:::

::: {.qa asked="Ant digital human (Nowcoder roundup 2025)"}
**What are the main error sources in 3D Gaussian splatting?**

3D Gaussian splatting (Kerbl et al., 2023) fits anisotropic Gaussians with opacity and spherical-harmonic colour to posed images, starting from a structure-from-motion (SfM) point cloud and alpha-blending depth-sorted splats. Most errors come from its inputs, its heuristic density control or its rendering approximations.

1. **Poses and initialization.** Pose errors produce blur and doubled edges that the optimizer patches with extra Gaussians, and textureless or shiny regions get few SfM points. Remedies: joint pose refinement or, for video, pose-free methods such as COLMAP-Free 3DGS (Fu et al., 2023).
2. **Densification heuristics.** Gaussians whose average view-space position gradient exceeds a threshold are cloned if small or split if large, and opacities are reset towards zero every 3,000 iterations to remove floaters near the cameras. These threshold-based rules can miss fine detail, inflate memory and still leave floaters. 3DGS-MCMC (Kheradmand et al., 2024) recasts densification and pruning as MCMC state transitions.
3. **Aliasing across scales.** Changing focal length or distance from training gives high-frequency artifacts when zooming in and aliasing or dilation artifacts when zooming out. Mip-Splatting (Yu et al., 2023) adds a 3D smoothing filter bounded by the training views' sampling rate and replaces the 2D dilation with a 2D Mip filter.
4. **Limited view dependence.** Low-order spherical harmonics cannot represent sharp specular highlights or reflections, so optimization fakes them with extra Gaussians, which corrupts geometry.
5. **Sparse views.** With few images the Gaussians overfit the training views. Depth priors help; FSGS (Zhu et al., 2023) adds monocular depth guidance and grows Gaussians from sparse points.
6. **Appearance changes.** Exposure, white-balance and lighting differences get baked in as floaters or colour shifts; per-image appearance embeddings or colour corrections absorb them.
7. **Sorting and popping.** Splats are sorted by one depth per Gaussian rather than per pixel, so blending order can flip as the camera rotates. StopThePop (Radl et al., 2024) re-sorts hierarchically per pixel to remove this.

To diagnose, render depth, opacity and per-view error maps, and test on views off the training trajectory.
:::

::: {.qa asked="Ant digital human (Nowcoder roundup 2025)"}
**How do multi-view stereo and NeRF differ in where they sample 3D points?**

MVS samples a fixed grid of depth hypotheses in the reference camera's frustum, scores each by how consistent the warped source views look, and picks one depth per pixel. NeRF samples points along each pixel's ray, at positions that change every iteration, and integrates a learned density and colour with volume-rendering weights.

**MVS (plane sweep).**

1. Choose $D$ depth hypotheses for the reference view, uniform in depth or inverse depth between $d_{\min}$ and $d_{\max}$. MVSNet used $D = 256$ planes at 2 mm spacing on DTU.
2. Warp each source image or feature map into the reference view with the homography of a fronto-parallel plane at that depth.
3. Score photometric consistency (NCC, or the variance of warped features across views in MVSNet) to build a cost volume over pixels and depths.
4. Regularize the volume (semi-global matching or a 3D CNN), take the argmin or a soft-argmin expectation per pixel, then fuse depth maps across views with consistency checks.

The score comes from matching, not rendering.

**NeRF.**

1. For a ray $r(t) = o + t d$, split $[t_n, t_f]$ into $N_c$ bins and draw one uniform sample per bin (stratified sampling).
2. Query the coarse MLP for $\sigma_i, c_i$ and compute $w_i = T_i\big(1 - e^{-\sigma_i \delta_i}\big)$ with $T_i = \exp\big(-\sum_{j<i}\sigma_j\delta_j\big)$; the pixel colour is $\sum_i w_i c_i$.
3. Normalize the $w_i$ into a piecewise-constant density along the ray, draw $N_f$ more samples by inverse-transform sampling, and evaluate the fine MLP on all $N_c + N_f$ points (64 and 128 in the original paper).

**Trade-offs.** Learned MVS is feed-forward and outputs depth directly, but needs texture and roughly Lambertian surfaces and handles occlusion through heuristics. NeRF models occlusion through transmittance and handles view-dependent appearance, but needs per-scene optimization and many network queries per ray, and samples in empty space are wasted, hence hierarchical sampling. They combine well: MVS depth can concentrate NeRF samples near surfaces, and cost volumes can condition generalizable radiance fields.
:::

::: {.qa asked="Ant digital human round 2 (Nowcoder roundup 2025)"}
**What is your view of digital humans: where is the value, and where is the field going?**

Give a structured opinion with stated uncertainty, in three parts.

**Where the value is**, ordered by how forgiving the use case is: dubbing and lip-sync translation of existing video, training and explainer videos with a presenter, livestream hosts and customer-service avatars that run around the clock, virtual idols and entertainment, avatars for telepresence and VR. The common value is a lower cost per minute of presenter video, and scaling across languages and personalized versions.

**What blocks adoption.** Realism beyond the lips (emotion, gaze, gestures, body motion); identity and stability over long sessions; latency for interactive use, where speech recognition, an LLM, text-to-speech and rendering add up; GPU cost per streamed hour; consent and likeness rights, deepfake misuse and disclosure rules; and audience trust, since viewers may engage less with a host they know is synthetic.

**Where it is going.** End-to-end audio-to-video diffusion models with longer consistency, real-time 3D Gaussian avatars built from short captures, full-duplex conversational agents driving the avatar, and shared evaluation. Name what you would watch to update your view: cost per minute against a human presenter, and engagement and conversion of synthetic hosts relative to human ones.

For the technical pipeline and open problems, use the audio-driven talking-head entry above rather than repeating it here.
:::

::: {.qa asked="Curated AIGC list, Nowcoder 2025"}
**Where are diffusion models most useful in products today?**

Answer with criteria first, then examples that meet them. Diffusion models fit best where:

1. **Errors are tolerable**: the output is judged by taste rather than correctness, and a bad sample can be discarded.
2. **A person stays in the loop**, choosing among samples or editing them, so the model supplies drafts rather than final answers.
3. **The output is worth more than it costs**: GPU-seconds per sample are cheap next to a designer's or photographer's hour.
4. **Volume and variation matter**: many variants per audience, language or format.
5. **Control is available**: a reference image, a mask or a layout makes the output predictable enough to use.

Examples that fit: editing tools (inpainting, outpainting, background replacement, object removal), advertising and e-commerce creatives with variants per audience, concept art and storyboards, stock imagery, short-form video and effects, and restoration such as upscaling, where conditioning on the input keeps outputs close to it. Outside media, diffusion models generate candidate molecules and protein structures and synthetic training data, where downstream tests screen the outputs.

Weaker fits: anything that must be exactly right (dense text, charts, diagrams, medical images used for diagnosis, where hallucinated detail is dangerous), interactive uses with tight latency budgets unless the model is distilled to few steps, and domains with high legal risk (real people's likeness, trademarks).
:::

::: {.qa asked="Tencent multimodal, Nowcoder 2026"}
**How could generative AI be used in games, and what do player and payment distributions look like?**

**Uses.** Asset production (concept art, textures, sprites, 3D assets, character animation), dialogue and voice for non-player characters, procedural quests and levels, player-facing creation tools (avatars, skins), localization and voice dubbing, marketing creatives with many variants, and automated play-testing agents. The constraints are consistency with the art direction, runtime latency and on-device compute, IP and licensing, controllability, safety filters for player-facing generation, and player acceptance of AI-made content.

**Distributions.** Payment is heavy-tailed: most players never pay, and a small share of payers brings in a large share of revenue. Playtime is skewed too, and many players leave in the first days. (Quote exact shares only from data you have seen.)

**Modelling implications.**

- **Zero inflation.** Use a hurdle model, $P(\text{pay} > 0)$ times $\mathbb{E}[\text{amount} \mid \text{pay} > 0]$, or a Tweedie loss, which models exact zeros and positive amounts together.
- **Tail.** Spending among payers is often close to lognormal or heavier. Model $\log(\text{amount})$, check the tail on a log-log plot of the complementary CDF, and expect sample means to be unstable: report medians or trimmed means with bootstrap intervals. A/B tests on revenue need large samples, winsorization or variance reduction.
- **Lifetime value.** Predict from early behaviour (first-day sessions, progression, first purchase), handle players who are still active as censored, and evaluate by how much revenue the top-ranked players capture rather than by MSE, which the largest spenders dominate.
- **Imbalance.** Payers are rare, so use class weights or stratified sampling and evaluate with precision-recall curves.

Add a responsibility point: avoid targeting vulnerable heavy spenders, and note that rules on paid random rewards differ by country.
:::

::: {.qa asked="Alibaba Taotian senior-manager round, Nowcoder 2025"}
**How could your work connect to our business, for example outfit recommendation in live-stream shopping?**

A manager asking this wants evidence that you can think in terms of their product. Structure:

1. **Understand the business first.** Say, or ask, who the user is (a viewer in a live stream), what decision is being made (which items to show or bundle while the host presents), the metric (click-through to the item, add-to-cart, conversion, return rate) and the constraints (real time, a catalogue that changes per stream).
2. **Map your technique to one stage of that pipeline.** For example: recognizing the item the host is wearing and retrieving it and compatible items; generating try-on images on different body types; enhancing low-quality stream frames or product crops so recognition works.
3. **Evidence path.** An offline metric for that stage (retrieval recall@k, try-on fidelity in a user study), then an online A/B test on conversion and returns.
4. **Cost and risk.** Latency per frame, GPU cost per streamed hour, misrepresentation (a flattering try-on raises returns), privacy of user photos.
5. **A small first step**: one category, or offline generation before the stream rather than live.

Ask questions back; it is a conversation. Avoid claiming your method transfers unchanged, and avoid research jargon.
:::

::: {.qa asked="SenseTime leader round, Nowcoder 2019"}
**What are the challenges of knowledge distillation, especially for video models, and how would you improve it?**

Knowledge distillation trains a small student to match a teacher. The main difficulties are the capacity gap, choosing what to match and, for video, the cost of running the teacher and keeping the student temporally consistent.

**Mechanism.** With student logits $z_s$, teacher logits $z_t$ and temperature $T$:

$$\mathcal{L} = (1-\lambda)\,\mathrm{CE}\big(y, \mathrm{softmax}(z_s)\big) + \lambda\, T^2\, \mathrm{KL}\big(\mathrm{softmax}(z_t/T)\,\big\|\,\mathrm{softmax}(z_s/T)\big).$$

A higher $T$ exposes the teacher's relative probabilities over wrong classes. The gradient of the KL term with respect to $z_{s,i}$ is $(q_i - p_i)/T$, where $q$ and $p$ are the student's and teacher's softened distributions, and for large $T$ the difference $q_i - p_i$ itself shrinks like $1/T$, so the soft gradients scale as $1/T^2$; the $T^2$ factor keeps their weight stable as $T$ changes (Hinton et al., 2015).

**General challenges.**

- **Capacity gap.** A stronger teacher is not always a better one. Mirzadeh et al. (2019) found student accuracy drops when the gap is large and inserted an intermediate-size teacher assistant; Cho and Hariharan (2019) found a chain of distillation steps ineffective in their experiments and instead stopped the teacher's training early. The evidence is mixed, so test both.
- **What to match.** Logits carry little information for dense prediction or generation. Feature distillation (FitNets, Romero et al., 2014) matches intermediate features through a learned adapter, but needs layer correspondences.
- **Teacher training choices.** A teacher trained with label smoothing distils much less effectively (Müller et al., 2019).

**Video-specific challenges.**

- **Teacher cost.** Precompute and cache teacher outputs, distil on shorter or lower-resolution clips first, then fine-tune at full size.
- **Temporal consistency.** Frame-wise losses can make the student flicker. Match the teacher's frame-to-frame feature differences or spatio-temporal attention, add flow-warped consistency losses, and evaluate temporal stability, not only accuracy.
- **Architecture mismatch.** A factorized or 2D student does not share layers with a full 3D teacher, so feature matching needs adapters.

For video generation, distillation usually reduces sampling steps rather than parameters; progressive distillation, for example, repeatedly halves the step count (Salimans and Ho, 2022).
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

::: {.qa asked="MSR intern, Glassdoor"}
**Which ML frameworks and tools have you used?**

Group the answer by layer and attach depth to each item, so the interviewer can calibrate.

1. **Core framework you use daily**, with what you have built in it: custom modules and losses, mixed precision, multi-GPU training, custom data pipelines, profiling.
2. **Libraries relevant to this team**, such as model hubs, diffusion or vision libraries, with a concrete use.
3. **Infrastructure**: cluster schedulers, experiment tracking, containers, version control, testing and reproducibility habits.
4. **Lighter experience**, labelled as such: "used JAX for one project; I would need a short ramp-up."

Anchor each line with one thing you built, because follow-up questions or a coding round will test the claim. If the team uses a stack you have not used, say how quickly you learned the last new one, with evidence. Avoid listing every tool you have touched.
:::

::: {.qa asked="Anthropic (secondary prep guide)"}
**Why do you want to work on AI safety?**

Give a specific reason grounded in something you have observed or worked on, then show you know what the work involves.

1. **The origin**: a concrete failure mode you have seen in models, such as confident but wrong outputs, a metric the model learned to exploit, or behaviour that changed under distribution shift, and why it worried you.
2. **The research areas** it connects to (evaluations, interpretability, robustness, alignment training, misuse prevention), and which of them you want to work on.
3. **What you bring**: relevant skills from your own work, such as careful evaluation, uncertainty and calibration, or experience with generative models.
4. **What you want to learn** in the role.

Avoid reciting the organization's mission back, apocalyptic or dismissive framing, and expertise you cannot back up. Prepare for the follow-up "which safety problem do you think is underrated, and how would you test it?", with one experiment you could actually run.
:::

::: {.qa asked="Meta RS FT 2026 (Reddit), HR rounds (Nowcoder 2022)"}
**Describe your work style. How would colleagues describe you?**

Pick two or three traits that matter for research collaboration and back each with a behaviour and a short example rather than an adjective: how you plan (weekly goals, small experiments that can fail fast), how you communicate (written updates, sharing negative results early), how you take feedback. For "how would colleagues describe you", paraphrase real feedback you have received from an advisor, a co-author or a past manager; it is more credible than self-description and should match what your references would say. Add one genuine growth area and what you are doing about it. Avoid generic lists ("hard-working, team player"), claiming no weaknesses, and weaknesses that are disguised strengths.
:::

::: {.qa asked="Tencent Youtu, Dewu (Nowcoder 2024)"}
**You are already interning somewhere. Why are you looking again?**

Answer honestly and keep it forward-looking. Cover the timing (when the current internship ends and when this one would start), what you are gaining from the current role, and what you want next that this team offers: a different problem area, more research or more product exposure, a chance to publish, or this team's specific work. Speak positively about the current employer, disclose nothing confidential, and say how you would handle any overlap (notice period, finishing your project responsibly, a pending return offer). Avoid complaints about the current team, and avoid anything that suggests you would leave this role halfway as well.
:::

::: {.qa asked="Tencent Youtu CV intern, Nowcoder 2024"}
**How is your English, and can you work in it every day?**

Answer factually, with evidence: the language of your degree and research group, papers you have written in English, talks or teaching in English, international collaborations, and how much you use it day to day. Be honest about the level, for example "fluent for technical work, less practised in fast informal conversation", and be ready to switch to English on the spot if asked, since that is the real test. Mention a habit that prevents misunderstandings, such as summarizing decisions in writing after meetings. Avoid overclaiming, and avoid answering with a test score alone.
:::

::: {.qa asked="Meituan 2024, SenseTime 2019, HR rounds (Nowcoder)"}
**What other offers do you have, why did you turn one down, and how would you choose between us and another company?**

Be truthful and discreet. Structure:

1. **Status**: say which processes or offers you have and their deadlines, without compensation details or confidential information unless you choose to share them.
2. **Your criteria, ranked**: fit of the research problem, the mentor and team, what past interns produced, publication policy, compute, timing and location, a path to full-time.
3. **The declined offer**, explained through those criteria ("the role was mostly product engineering"; "the timing did not fit"), never by criticizing the company.
4. **Choosing between teams**: what would make you pick this one, and what information you still need to decide. Call it your first choice only if it is.

Avoid bluffing about offers (timelines and claims can be checked) and crude bargaining. Raise deadlines early and ask whether the process can accommodate them.
:::

::: {.qa asked="Meituan 2024, SenseTime 2019 (Nowcoder)"}
**Do you prefer research or product work?**

Give an honest preference with a reason, then show that you value the other side. Describe the kind of work that motivates you (open problems, or seeing users rely on something you built), the balance you want (for example, research grounded in real data and real constraints), and one example where a product constraint improved your research, or research changed a product decision. Match the answer to the role without misrepresenting yourself: if the team is mostly product work and you want only research, that mismatch matters to both sides, and it is better found now. Avoid treating product work as "just engineering", and avoid "either is fine" with nothing behind it. (How to balance the two under deadlines is a separate question in this section.)
:::

::: {.qa asked="ByteDance AIGC intern, Nowcoder 2024"}
**How will you balance the internship with your school work?**

Give specifics rather than reassurance: your remaining obligations (courses, teaching, exams, thesis milestones, paper deadlines) and roughly how many hours they take, whether your advisor has agreed to the internship, the schedule you can commit to (days on site, hours per week, full-time or part-time), and any rules that cap your hours during term, such as work-authorization limits. Name the known conflicts and say how you would handle them: telling the manager weeks in advance, shifting hours, front-loading work. Support it with a time you handled overlapping commitments. Avoid promising hours you cannot sustain, and avoid hiding a conflict you already know about.
:::

::: {.qa asked="HR rounds, Nowcoder 2022"}
**Tell me about a time you broke a rule.**

The question tests judgment and honesty, so choose a real, low-stakes case where a rule or process conflicted with its own purpose, for example an outdated procedure that would have blocked a deadline for no benefit, and where you acted openly. Structure it as: the situation; the rule and what it was meant to protect; why following it literally would have hurt the goal; what you did, including whom you told or asked and how you limited the risk; the outcome; and what you would do differently, often "raise it earlier" or "get the rule changed instead". Never use a story involving safety, research integrity, privacy, security or the law, or one where you hid what you did. "I have never broken a rule" sounds evasive, and bragging about it sounds worse.
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

::: {.qa asked="Qualcomm 2026 (Glassdoor), NVIDIA (PracHub 2026)"}
**Quantize an FP32 tensor to int8: compute the scale and zero point, and explain which errors quantization introduces.**

Quantization stores a real value as an integer through an affine map, $x \approx s\,(q - z)$, with a float scale $s$ and an integer zero point $z$. You choose $s$ and $z$ from the range you decide to represent. The errors are rounding inside that range, clipping outside it, and what both turn into after integer arithmetic and later layers.

```python
import numpy as np

def quantize(x, bits=8, symmetric=False):
    if symmetric:                                  # zero point fixed at 0; codes [-127, 127]
        qmin, qmax = -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
        scale, zp = max(np.abs(x).max(), 1e-12) / qmax, 0
    else:                                          # map [lo, hi] onto [-128, 127]
        qmin, qmax = -2 ** (bits - 1), 2 ** (bits - 1) - 1
        lo, hi = min(x.min(), 0.0), max(x.max(), 0.0)   # keep 0 exactly representable
        scale = max(hi - lo, 1e-12) / (qmax - qmin)
        zp = int(np.clip(np.round(qmin - lo / scale), qmin, qmax))
    q = np.clip(np.round(x / scale) + zp, qmin, qmax).astype(np.int8)
    return q, scale, zp

def dequantize(q, scale, zp):
    return (q.astype(np.float32) - zp) * scale
```

**Choosing the map.** Asymmetric: for a range $[\ell, u]$ widened to contain 0, $s = (u - \ell)/255$ and $z = \mathrm{round}(-128 - \ell/s)$; zero stays exact, which matters for padding and one-sided post-ReLU activations. Symmetric: $z = 0$ and $s = \max|x|/127$. A nonzero weight zero point adds cross terms to the integer matmul $\sum_k (q_{x,k} - z_x)(q_{w,k} - z_w)$, so weights are usually symmetric, with one scale per output channel because channel ranges differ.

**The errors.**

1. Rounding: inside the range the error lies in $[-s/2, s/2]$ and is close to uniform when values span many steps, so its variance is $\frac{1}{s}\int_{-s/2}^{s/2} e^2\,de = s^2/12$; each extra bit halves $s$, about 6 dB of signal-to-noise.
2. Clipping: values outside the range saturate. A wider range clips less but raises $s$ for every value, so calibration data picks the range (min/max, a percentile, minimum MSE, or a KL criterion).
3. Outliers: one value of 100 among 10,000 unit-Gaussian values raises the symmetric int8 MSE from about $8 \times 10^{-5}$ to $5 \times 10^{-2}$; finer-grained scales, clipping, or higher precision for outlier channels limit the damage.
4. Accumulation: sums of int8 products need 32-bit accumulators, and requantizing with the combined scale $s_x s_w / s_y$ rounds again.
5. Propagation: errors compound through layers, and rounded weights shift output means (bias correction re-centres them). Fold BatchNorm into the weights first, and keep sensitive layers (first, last, softmax, normalization) in higher precision if needed.

**PTQ vs QAT.** Post-training quantization only calibrates ranges. Quantization-aware training trains through quantize-dequantize operations with the straight-through estimator, taking $\partial\,\mathrm{round}(u)/\partial u = 1$ inside the clipping range and 0 outside; it costs training time and usually loses less accuracy at low bit widths. Quantizing is $O(N)$ in time and memory.
:::

::: {.qa asked="Adobe research intern (Glassdoor)"}
**You have limited time to implement the core function of a paper. How do you approach it?**

Aim for a small, correct, tested version of the key equation first and generalize only after it works, saying your assumptions aloud as you go. A structure that fits most of these rounds:

1. **Pin down the contract.** Inputs and outputs with shapes and dtypes, whether it must be batched or differentiable, and whether a reference output exists.
2. **Find the defining equation** and write each symbol's shape in a comment. Name the ambiguities papers leave open (sum or mean, the softmax axis, indexing, a missing constant) and state your choice.
3. **Write the minimal version** close to the equation, so the interviewer can check it line by line; loops are fine.
4. **Test on a tiny input at once:** a hand-computed case, an invariant (shape, rows summing to 1, an identity case such as zero noise), loop against vectorized, or a gradient check.
5. **Then extend:** vectorize, batch, add masks and numerical stability, re-running the same tests after each change.

Avoid scaffolding (data loaders, configs, a training loop) before the core function, optimizing before it is correct, and long silences. If time runs out, leave the rest as clear pseudocode and say how you would test it.
:::

::: {.qa asked="Samsung Research America 2024; AIGC interview repo"}
**Implement minibatch SGD from scratch, for example for linear regression.**

Each epoch reshuffles the data, walks through it in minibatches, computes the gradient of the batch loss and steps against it. For $L(w) = \frac{1}{2|B|}\|X_B w - y_B\|^2$ the batch gradient is $g = X_B^\top (X_B w - y_B)/|B|$.

```python
import numpy as np

def sgd_linreg(X, y, lr=0.003, batch=32, epochs=500, beta=0.9, tol=1e-6, patience=10, seed=0):
    rng = np.random.default_rng(seed)
    X = np.hstack([X, np.ones((len(X), 1))])        # bias as an extra feature
    w, v = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    best, bad = np.inf, 0
    for epoch in range(epochs):
        eta = lr / (1 + 0.05 * epoch)                # decay so the gradient noise averages out
        idx = rng.permutation(len(X))                # reshuffle every epoch
        for s in range(0, len(X), batch):
            j = idx[s:s + batch]
            g = X[j].T @ (X[j] @ w - y[j]) / len(j)  # grad of 0.5 * mean squared error
            v = beta * v + g                         # heavy-ball momentum
            w = w - eta * v
        loss = 0.5 * np.mean((X @ w - y) ** 2)       # use a held-out set in practice
        best, bad = (loss, 0) if loss < best * (1 - tol) else (best, bad + 1)
        if bad == patience:                          # no real progress for `patience` epochs
            break
    return w
```

**Learning rate.** The full-batch loss has the constant Hessian $H = X^\top X / n$, and one gradient step gives $w_{t+1} - w^* = (I - \eta H)(w_t - w^*)$. This contracts only if $|1 - \eta\lambda_i| < 1$ for every eigenvalue, that is $\eta < 2/\lambda_{\max}$, and the slowest direction converges at a rate set by $\lambda_{\max}/\lambda_{\min}$, which is why you standardize features first. Heavy-ball momentum widens the stable range to $\eta < 2(1+\beta)/\lambda_{\max}$. With minibatches and a constant step, the iterate settles in a noise ball whose size grows with $\eta$ and shrinks with batch size; a decaying step with $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ (such as $\eta_t \propto 1/t$, as above) removes it.

**Convergence check.** Stop when the held-out loss has not improved for a few epochs (comparing consecutive epochs alone stops too early, because SGD noise makes the loss go up and down), and on small problems compare with `np.linalg.lstsq`; on synthetic data the snippet matches it to within the noise level of the data.

**Follow-up: an extremum of a 1D function.** Iterate $x \leftarrow x - \eta f'(x)$ (plus sign for a maximum), with $f'(x) \approx (f(x+h) - f(x-h))/2h$ if no derivative is given, until $|f'(x)|$ is below a tolerance; this finds a local extremum near the start. Newton's method, $x \leftarrow x - f'(x)/f''(x)$, converges quadratically near the extremum but needs $f'' \ne 0$ and can head for a maximum.

Each epoch costs $O(nd)$ time, with $O(d)$ memory beyond the data.

[more](deep-learning-training.html)
:::

::: {.qa asked="Apple ML intern 2024, Meta MLE 2026"}
**Implement k-nearest-neighbour search and classification over embeddings.**

Compute all query-to-database squared distances with one matrix multiply, using $\|q - x\|^2 = \|q\|^2 - 2q^\top x + \|x\|^2$, take the $k$ smallest per query with a partial sort, and vote. Process queries in chunks so memory stays bounded.

```python
import numpy as np

def knn_search(Q, X, k, cosine=False, chunk=1024):
    # Q: (m, d) queries, X: (n, d) database -> (m, k) indices, nearest first
    if cosine:                                     # cosine = L2 ranking on unit vectors
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    x2, out = (X ** 2).sum(1), []
    for s in range(0, len(Q), chunk):              # memory O(chunk * n), not O(m * n)
        d = x2[None, :] - 2 * Q[s:s + chunk] @ X.T # ||q||^2 is constant per row: dropped
        part = np.argpartition(d, k - 1, axis=1)[:, :k]          # unordered top-k, O(n) per row
        order = np.argsort(np.take_along_axis(d, part, 1), axis=1)
        out.append(np.take_along_axis(part, order, 1))
    return np.vstack(out)

def knn_classify(Q, X, labels, k):
    preds = []
    for row in labels[knn_search(Q, X, k)]:        # neighbour labels, nearest first
        counts = np.bincount(row)
        preds.append(next(c for c in row if counts[c] == counts.max()))  # tie: nearest wins
    return np.array(preds)
```

Dropping $\|q\|^2$ does not change the ranking for a given query. For unit vectors $\|q - x\|^2 = 2 - 2\cos(q, x)$, so cosine search is L2 search after normalization. The expanded form can give slightly negative values or reorder near-ties through floating-point cancellation; clamp at 0 if you need the distances themselves. Ties in the vote go here to the class of the nearest tied neighbour; distance-weighted votes are an alternative, and $k$ is chosen by cross-validation (odd $k$ avoids ties in binary problems).

Brute force costs $O(mnd)$ time, which is one large matrix multiply and fast on a GPU, plus $O(n)$ per query for the partial sort and $O(k \log k)$ to order the top $k$. For millions of vectors or more, use approximate search: an inverted-file index (cluster the database with k-means and scan only the few nearest clusters), product quantization (compress vectors to short codes and approximate distances with lookup tables), or a graph index such as HNSW (greedy search through a navigable neighbour graph). All trade recall for speed and memory, so measure recall@k against brute force on a sample.
:::

::: {.qa asked="Meta MLE 2026 (PracHub); AIGC interview repo"}
**Implement the forward and backward pass of a two-layer MLP in NumPy.**

The forward pass caches its intermediate values; the backward pass applies the chain rule in reverse order. Every gradient has the shape of the quantity it differentiates, which fixes where the transposes go. Forward: $z_1 = XW_1 + b_1$, $h = \max(z_1, 0)$, $z_2 = hW_2 + b_2$, $p = \mathrm{softmax}(z_2)$, $L = -\frac{1}{B}\sum_i \log p_{i, y_i}$. Backward:

1. Softmax with cross-entropy: for one row, $L_i = -z_{y_i} + \log\sum_j e^{z_j}$, so $\partial L_i/\partial z_j = p_j - \mathbf{1}[j = y_i]$. Over the batch, $\delta_2 = (P - Y)/B$ with $Y$ one-hot, shape $(B, C)$.
2. Second layer: $\partial L/\partial W_2 = h^\top \delta_2$ of shape $(H, C)$, $\partial L/\partial b_2 = \sum_i \delta_{2,i}$, and $\partial L/\partial h = \delta_2 W_2^\top$ of shape $(B, H)$.
3. ReLU: $\delta_1 = (\delta_2 W_2^\top) \odot \mathbf{1}[z_1 > 0]$.
4. First layer: $\partial L/\partial W_1 = X^\top \delta_1$ and $\partial L/\partial b_1 = \sum_i \delta_{1,i}$.

```python
import numpy as np

def mlp_loss_and_grads(X, y, W1, b1, W2, b2):
    # X: (B, D); y: (B,) int labels; W1: (D, H); W2: (H, C)
    B = len(X)
    z1 = X @ W1 + b1                               # (B, H)
    h = np.maximum(z1, 0)                          # ReLU
    z2 = h @ W2 + b2                               # (B, C) logits
    z2 = z2 - z2.max(1, keepdims=True)             # stable softmax
    p = np.exp(z2); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(B), y]).mean()
    d2 = p.copy(); d2[np.arange(B), y] -= 1; d2 /= B   # dL/dz2 = (p - onehot) / B
    dW2, db2 = h.T @ d2, d2.sum(0)
    dz1 = (d2 @ W2.T) * (z1 > 0)                   # back through W2, then the ReLU mask
    dW1, db1 = X.T @ dz1, dz1.sum(0)
    return loss, [dW1, db1, dW2, db2]

def grad_check(f, params, grads, eps=1e-6):       # f(params) -> loss; central differences
    for P, G in zip(params, grads):
        i = tuple(np.random.randint(s) for s in P.shape)
        old = P[i]; P[i] = old + eps; lp = f(params)
        P[i] = old - eps; lm = f(params); P[i] = old
        num = (lp - lm) / (2 * eps)
        print(abs(num - G[i]) / max(1e-12, abs(num) + abs(G[i])))   # ~1e-7 or less in float64
```

Call the check as `grad_check(lambda ps: mlp_loss_and_grads(X, y, *ps)[0], params, grads)`. The central difference has $O(\epsilon^2)$ truncation error, so use float64; a relative error near $10^{-6}$ or below means the gradient is right, while a perturbation that pushes a pre-activation across 0 can produce a spurious mismatch at the ReLU kink. Forward and backward each cost $O(B(DH + HC))$ multiply-adds, and the cached activations take $O(B(H + C))$ memory.

[more](deep-learning-training.html)
:::

::: {.qa asked="Runway MLE, PracHub 2025"}
**Write an n-gram language model class with fit and generate methods. How do you pick n?**

An n-gram model estimates $P(w_t \mid w_{t-n+1}, \dots, w_{t-1})$ from counts of each $(n-1)$-token context followed by each next token. Fitting is counting, generation is repeated sampling from the counts of the current context, and $n$ is chosen by held-out perplexity: a larger $n$ fits the training text better but sees each context fewer times.

```python
import math, random
from collections import Counter, defaultdict

class NGramLM:
    def __init__(self, n, k=0.1):                  # k: add-k smoothing constant
        self.n, self.k = n, k
        self.counts, self.vocab = defaultdict(Counter), {"</s>"}

    def grams(self, toks):                         # (context, next token) pairs, padded
        toks = ["<s>"] * (self.n - 1) + list(toks) + ["</s>"]
        return [(tuple(toks[i - self.n + 1:i]), toks[i]) for i in range(self.n - 1, len(toks))]

    def fit(self, corpus):                         # corpus: list of token lists
        for toks in corpus:
            self.vocab.update(toks)
            for ctx, w in self.grams(toks):
                self.counts[ctx][w] += 1
        return self

    def prob(self, ctx, w):
        c = self.counts.get(ctx, Counter())
        return (c[w] + self.k) / (sum(c.values()) + self.k * len(self.vocab))

    def perplexity(self, corpus):
        pairs = [g for toks in corpus for g in self.grams(toks)]
        return math.exp(-sum(math.log(self.prob(c, w)) for c, w in pairs) / len(pairs))

    def generate(self, max_len=30, rng=random):
        out = ["<s>"] * (self.n - 1)
        for _ in range(max_len):
            c = self.counts.get(tuple(out[len(out) - self.n + 1:]))
            if not c:
                break
            w = rng.choices(list(c), weights=list(c.values()))[0]
            if w == "</s>":
                break
            out.append(w)
        return out[self.n - 1:]
```

The $n-1$ start tokens give the first words a full context, and the end token lets the model learn when to stop. The maximum-likelihood estimate $c(\text{ctx}, w)/c(\text{ctx})$ gives probability zero, and infinite perplexity, to any n-gram missing from training, so evaluation needs smoothing. Add-$k$ (shown) is the simplest and spreads too much mass over unseen words; interpolation mixes orders, $\lambda_1 P_1 + \dots + \lambda_n P_n$ with the weights tuned on held-out data; backoff uses the longest context that has counts and discounts so the distribution still sums to 1; Kneser-Ney smoothing is usually the strongest of the classic methods.

**Choosing n.** Fit $n = 1, \dots, 5$ on the training split and pick the lowest validation perplexity $\exp\big(-\frac{1}{N}\sum_t \log P(w_t \mid \text{ctx}_t)\big)$. Training perplexity keeps falling as $n$ grows, while validation perplexity turns back up once most contexts are seen only once or never, because the number of possible contexts grows like $V^{n-1}$ while the data stays fixed.

Fitting costs $O(Nn)$ time for $N$ tokens and memory proportional to the number of distinct n-grams. Cache each context's total count to make `prob` $O(1)$ (the snippet recomputes it); a generation step costs time proportional to the number of continuations of the context.
:::

::: {.qa asked="Meta RS full-time, Blind 2024"}
**Compute the dot product of two sparse vectors efficiently.**

Store only the nonzeros and touch only them. With hash maps, iterate over the shorter vector and look indices up in the other, $O(\min(m, n))$ expected time. With sorted index arrays, walk both with two pointers in $O(m + n)$, which needs no hashing and is cache-friendly. When one vector is much shorter, binary-search its indices in the longer one, $O(m \log n)$.

```python
from bisect import bisect_left

def dot_dict(a, b):                        # a, b: {index: value}; expected O(min(m, n))
    if len(a) > len(b):
        a, b = b, a                        # iterate over the shorter one
    return sum(v * b[i] for i, v in a.items() if i in b)

def dot_two_pointer(ia, va, ib, vb):       # sorted index arrays with parallel values; O(m + n)
    i = j = 0
    s = 0.0
    while i < len(ia) and j < len(ib):
        if ia[i] == ib[j]:
            s += va[i] * vb[j]; i += 1; j += 1
        elif ia[i] < ib[j]:
            i += 1
        else:
            j += 1
    return s

def dot_skewed(ia, va, ib, vb):            # len(ia) = m << len(ib) = n; O(m log n)
    s, lo = 0.0, 0
    for k, v in zip(ia, va):
        lo = bisect_left(ib, k, lo)        # search only to the right of the previous match
        if lo < len(ib) and ib[lo] == k:
            s += v * vb[lo]
    return s
```

Follow-ups usually probe the representation and the skew:

- **Very different lengths.** Two pointers cost $O(n)$ even when $m = 3$. Binary search costs $O(m \log n)$; galloping (exponential) search from the previous match costs $O(m \log(n/m))$, which is asymptotically no worse than either.
- **One vector dense.** Iterate over the sparse one and index the dense array: $O(m)$.
- **One vector against many.** Convert it once to a hash map or dense array. For many sparse vectors against many queries, build an inverted index from coordinate to (vector id, value), the same structure that sparse retrieval uses.
- **Storage.** Coordinate lists (index, value) for vectors; CSR (row pointers, column indices, values) for matrices, where a sparse matrix-vector product is a loop of sparse dot products. Hash maps cost more memory and have poor locality; sorted arrays are compact.
- **Scale.** Shard by index range, compute partial sums per shard and add them; accumulate in float64 when there are many terms.
:::

::: {.qa asked="NVIDIA (PracHub 2026); Luma RE"}
**Work out the tensor shapes through an MLP and explain PyTorch broadcasting.**

`nn.Linear(d_in, d_out)` stores a weight of shape `(d_out, d_in)` and a bias of shape `(d_out,)`, and computes $y = xW^\top + b$ over the last dimension only, so leading batch and sequence dimensions pass through unchanged. Broadcasting aligns shapes from the right and stretches size-1 dimensions; it is convenient and a common source of silent bugs.

**Shapes.** For $x$ of shape $(B, T, d)$, a transformer MLP is `Linear(d, 4d)` with weight $(4d, d)$, giving $(B, T, 4d)$, then an activation, then `Linear(4d, d)` with weight $(d, 4d)$, back to $(B, T, d)$. It has $8d^2 + 5d$ parameters and costs $8BTd^2$ multiply-adds; the backward pass needs the $(B, T, 4d)$ pre-activations. The weight gradient of the first layer is $\delta^\top x$ after flattening $B \cdot T$ into one axis: a sum of outer products over all tokens.

**Broadcasting rules.** Compare shapes from the last dimension backwards; two sizes are compatible when they are equal or one of them is 1, and missing leading dimensions count as 1. The result takes the larger size in each position, and size-1 dimensions are expanded as views with stride 0, without copying.

```python
import torch, torch.nn as nn

B, T, d = 8, 16, 64
x = torch.randn(B, T, d)
fc1, fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
print(fc1.weight.shape, fc1.bias.shape)         # (256, 64), (256,): weight is (out, in)
h = torch.relu(fc1(x))                          # x @ W.T + b on the last dim -> (8, 16, 256)
print(fc2(h).shape)                             # (8, 16, 64)
print(torch.allclose(torch.einsum("btd,od->bto", x, fc1.weight) + fc1.bias, fc1(x), atol=1e-5))

pred, target = torch.randn(B, 1), torch.randn(B)
print((pred - target).shape)                    # (8, 8): silent broadcasting bug
print((pred.squeeze(-1) - target).shape)        # (8,)

p = torch.rand(B, 10)
p = p / p.sum(dim=1, keepdim=True)              # (8, 10) / (8, 1): rows sum to 1
scale = torch.rand(B)
img = torch.randn(B, 3, 32, 32) * scale.view(B, 1, 1, 1)   # one scale per sample
```

**Classic bugs.** A $(B, 1)$ prediction minus a $(B,)$ target becomes $(B, B)$, and its mean is a plausible-looking but wrong loss. Normalizing without `keepdim` divides a $(B, C)$ tensor by a $(B,)$ one: an error when $B \ne C$, and a silent division by the wrong row's sum when $B = C$. A per-sample scalar needs `view(B, 1, 1, 1)` before multiplying an image batch. In-place operations cannot grow their left operand, so `a += b` fails when the broadcast shape differs from `a`'s. Batched matmul broadcasts the leading dimensions, $(B, h, N, d)$ @ $(B, h, d, M) \to (B, h, N, M)$, and `einsum` makes a contraction explicit. `expand` returns a view, while `repeat` copies memory. Assert shapes at module boundaries and name the dimensions in comments.
:::

::: {.qa asked="NVIDIA ML intern, Jointaro 2024"}
**Decide whether a point lies inside a convex hull, and code it.**

In 2D, build the hull with the monotone-chain algorithm in $O(n \log n)$ and test the point with cross-product signs: a point is inside a counter-clockwise convex polygon exactly when it is on the left of, or on, every edge. A binary search over the fan of triangles around one vertex does this in $O(\log h)$ for a hull of $h$ vertices. In $d$ dimensions, skip the hull and solve a linear-programming feasibility problem.

```python
def cross(o, a, b):                            # > 0 if o -> a -> b turns left
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull(pts):                          # Andrew's monotone chain: CCW, O(n log n)
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2:
        return pts
    def chain(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()                        # drop right turns and collinear points
            h.append(p)
        return h[:-1]
    return chain(pts) + chain(pts[::-1])       # lower hull + upper hull

def in_convex_polygon(hull, p):                # CCW hull with >= 3 vertices; O(log h)
    h0, n = hull[0], len(hull)
    if cross(h0, hull[1], p) < 0 or cross(h0, hull[-1], p) > 0:
        return False                           # outside the fan of wedges around h0
    lo, hi = 1, n - 1                          # find the wedge (h0, hull[lo], hull[lo + 1])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cross(h0, hull[mid], p) >= 0:
            lo = mid
        else:
            hi = mid
    return cross(hull[lo], hull[lo + 1], p) >= 0   # boundary counts as inside
```

The first check confines $p$ to the angle at `hull[0]` spanned by the polygon; the binary search finds the triangle `(hull[0], hull[lo], hull[lo+1])` whose angular sector contains $p$; then $p$ is inside if it lies on the inner side of that triangle's outer edge. The `>= 0` comparisons count the boundary as inside; use `> 0` for strict interior. Integer coordinates make the test exact, while floats need a tolerance or exact predicates. If all points are collinear, the hull has fewer than three vertices, and the test becomes whether $p$ lies on the segment. For a single query, the $O(h)$ check against every edge is enough.

In $d$ dimensions, $p$ lies in the hull of $x_1, \dots, x_n$ exactly when some $\lambda \ge 0$ satisfies $\sum_i \lambda_i = 1$ and $\sum_i \lambda_i x_i = p$, which is a feasibility LP with $n$ variables and $d + 1$ equality constraints:

```python
import numpy as np
from scipy.optimize import linprog

def in_hull_lp(X, p):                          # X: (n, d) points, p: (d,)
    A = np.vstack([X.T, np.ones(len(X))])      # sum_i lam_i x_i = p and sum_i lam_i = 1
    return linprog(np.zeros(len(X)), A_eq=A, b_eq=np.append(p, 1.0), bounds=(0, None)).status == 0
```

Building the hull is impractical in high dimensions, because a polytope with $n$ vertices can have on the order of $n^{\lfloor d/2 \rfloor}$ facets. For many queries in low dimensions, precompute the facet inequalities $Ax \le b$ or a Delaunay triangulation and locate the containing simplex.
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**How does PyTorch autograd work? How do you take second derivatives, why call a module instead of its forward method, and what are ReLU's backward caveats?**

Autograd records the graph while the forward pass runs. Each differentiable operation on a tensor that requires gradients creates a node, the tensor's `grad_fn`, which links to the nodes of its inputs and saves whatever its backward needs (a multiplication saves both operands, ReLU its output). `loss.backward()` walks this graph in reverse topological order, computes vector-Jacobian products, and adds the results into `.grad` of the leaf tensors, usually the parameters.

Gradients accumulate, hence `zero_grad()` between steps (and gradient accumulation over micro-batches for free). Non-leaf tensors keep no `.grad` unless you call `retain_grad()`; the graph is freed after `backward()` unless `retain_graph=True`; a non-scalar output needs an explicit `grad_outputs` vector. Saved tensors carry version counters, so modifying one in place before the backward pass raises an error.

**Second derivatives.** With `create_graph=True`, autograd also records the backward computation, so the returned gradient is itself differentiable:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
(g,) = torch.autograd.grad(x ** 3, x, create_graph=True)   # 3x^2 = 12, itself differentiable
(h,) = torch.autograd.grad(g, x)                             # 6x = 12

w, v = torch.randn(5, requires_grad=True), torch.randn(5)
(gw,) = torch.autograd.grad((w ** 4).sum(), w, create_graph=True)
(hvp,) = torch.autograd.grad(gw @ v, w)                      # Hessian-vector product, no Hessian built
print(torch.allclose(hvp, 12 * w.detach() ** 2 * v))         # True
```

Gradient penalties (WGAN-GP, R1), curvature estimates and MAML-style meta-learning use this; each extra order costs roughly another backward pass plus the memory of the larger graph.

**Module call versus `forward`.** `module(x)` goes through `nn.Module.__call__`, which runs the forward pre-hooks, `forward`, the forward hooks, and sets up backward hooks. Calling `module.forward(x)` silently skips them: feature-extraction hooks do not fire, and the older `torch.nn.utils.spectral_norm`, which recomputes the normalized weight in a pre-hook, uses a stale weight.

**ReLU caveats.**

1. The derivative at exactly 0 is undefined; any value in $[0, 1]$ is a valid subgradient, and PyTorch uses 0. It matters only with exact zeros (padding, clamped or quantized inputs), where it decides whether gradient flows.
2. In-place ReLU saves memory and is safe after a linear or convolution layer, since ReLU's backward needs only its own output. After an operation that saved its output for its backward (sigmoid, tanh, exp), overwriting that output raises the in-place modification error.
3. Dead units: a unit whose pre-activation is negative for every input gets zero gradient and does not recover. Large learning rates and negative biases cause it; smaller steps, careful initialization, LeakyReLU or GELU avoid it.
:::

::: {.qa asked="Company unnamed (via AIGC interview repo)"}
**Implement 1D rotary position embedding (RoPE).**

RoPE splits the channels of each query and key into pairs and rotates pair $i$ by the angle $m\theta_i$, where $m$ is the token's position and $\theta_i = 10000^{-2i/d}$. Rotations compose, so for each pair $(R(m\theta_i)q^{(i)})^\top R(n\theta_i)k^{(i)} = q^{(i)\top} R((n - m)\theta_i)\,k^{(i)}$, because $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$. Summing over pairs, the attention score depends on the positions only through the offset $n - m$.

```python
import torch

def rope(x, pos, base=10000.0):
    # x: (..., T, d), d even; pos: (T,) integer positions (offset them when using a KV cache)
    # Half-split pairing: dimension i rotates with dimension i + d/2.
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)   # (d/2,) frequencies
    ang = pos.float()[:, None] * theta[None, :]                         # (T, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# check: the score depends only on the offset n - m
q, k = torch.randn(64), torch.randn(64)
def score(m, n):
    return rope(q[None], torch.tensor([m])) @ rope(k[None], torch.tensor([n])).T
print(torch.allclose(score(3, 10), score(103, 110), atol=1e-4))   # True
print(torch.allclose(score(3, 10), score(3, 11), atol=1e-4))      # False
```

**Pairing convention.** The RoFormer paper (Su et al., 2021) pairs adjacent channels $(2i, 2i+1)$; the half-split form above, pairing $i$ with $i + d/2$, is common in open-source LLM code. The two differ by a fixed permutation of channels, so either works when training from scratch, but a pretrained checkpoint only works with its own convention unless its query and key projection rows are permuted to match.

**Practical points.** Apply RoPE per head after the q and k projections, and not to v. With a KV cache, pass absolute positions offset by the cache length and cache the keys after rotation. Precompute the cos and sin tables, and compute the angles in float32, since large positions lose precision in half precision. Rotation preserves norms, and position 0 is the identity, which are two more cheap tests. The cost is $O(Td)$ elementwise work per head.
:::

::: {.qa asked="ByteDance/OPPO/Momo image roles, Nowcoder 2021"}
**Implement a ResNet basic block as a reusable module.**

A basic block computes $y = \mathrm{ReLU}(F(x) + S(x))$, where $F$ is conv3x3, BN, ReLU, conv3x3, BN, and the shortcut $S$ is the identity when the shapes match, or a strided 1x1 convolution followed by BN (a projection) when the stride or channel count changes.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, zero_init=True):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)  # BN follows: no bias
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if zero_init:
            nn.init.zeros_(self.bn2.weight)           # residual branch starts at zero
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:              # projection when the shape changes
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))         # ReLU after the addition

def make_stage(c_in, c_out, n_blocks, stride):
    blocks = [BasicBlock(c_in, c_out, stride)] + [BasicBlock(c_out, c_out) for _ in range(n_blocks - 1)]
    return nn.Sequential(*blocks)
```

The convolutions have no bias because the following BN's shift makes it redundant. The stride sits on the first convolution, so only the first block of a stage downsamples. Setting the last BN's scale $\gamma$ to zero (Goyal et al., 2017) makes every residual branch output zero at initialization, so the block starts as the identity followed by a ReLU, which eases early training of deep networks; torchvision exposes this as `zero_init_residual`. The pre-activation variant (He et al., 2016) orders each branch as BN, ReLU, conv and drops the ReLU after the addition, so the shortcut path stays a pure identity.

The bottleneck block used in ResNet-50 and deeper replaces the two 3x3 convolutions with a 1x1 reduction to width $w$, a 3x3 convolution at width $w$, and a 1x1 expansion to $4w$, which is far cheaper than two 3x3 convolutions at width $4w$. A basic block has $9c_{\text{in}}c_{\text{out}} + 9c_{\text{out}}^2$ convolution weights, plus $c_{\text{in}}c_{\text{out}}$ for a projection, and costs that many multiply-adds per output pixel. A quick test: `make_stage(64, 128, 2, 2)` has the same parameter count as torchvision's `resnet18().layer2` (525,568), and a zero-initialized block in eval mode returns `relu(x)`.

[more](vision-backbones.html)
:::

::: {.qa asked="ByteDance, OPPO, Momo image roles (Nowcoder 2021)"}
**Implement mean, median and Gaussian filters and a mosaic effect, and make the Gaussian filter fast.**

Compute the mean (box) filter from an integral image, at $O(1)$ per pixel for any radius; the Gaussian filter as two 1D passes, at $O(k)$ instead of $O(k^2)$ per pixel for a $k \times k$ kernel; the median by sorting each window, or with a sliding histogram for 8-bit images; and the mosaic by replacing blocks with their means.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

def box_filter(img, r):                        # mean over a (2r+1)^2 window; O(1) per pixel
    k, (H, W) = 2 * r + 1, img.shape
    p = np.pad(img, r, mode="reflect").astype(np.float64)
    S = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))   # integral image, zero first row/col
    return (S[k:k + H, k:k + W] - S[:H, k:k + W] - S[k:k + H, :W] + S[:H, :W]) / k ** 2

def gaussian_filter(img, sigma):               # separable: two 1D passes, O(k) per pixel
    r = int(np.ceil(3 * sigma))
    g = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2)); g /= g.sum()
    p = np.pad(img, r, mode="reflect").astype(np.float64)
    p = windows(p, 2 * r + 1, axis=0) @ g      # vertical pass   -> (H, W + 2r)
    return windows(p, 2 * r + 1, axis=1) @ g   # horizontal pass -> (H, W)

def median_filter(img, r):                     # sorts every window: O(k^2 log k) per pixel
    p = np.pad(img, r, mode="reflect")
    return np.median(windows(p, (2 * r + 1, 2 * r + 1)), axis=(-2, -1))

def mosaic(img, b):                            # replace each b x b block by its mean
    out = img.astype(np.float64).copy()
    for i in range(0, img.shape[0], b):
        for j in range(0, img.shape[1], b):
            out[i:i + b, j:j + b] = out[i:i + b, j:j + b].mean(axis=(0, 1))
    return out
```

**Mean filter.** With the integral image $S$ (each entry the sum of everything above and to the left), any window sum takes four lookups, so the cost does not depend on the radius. Accumulate in float64 or int64 to avoid overflow and precision loss on large images.

**Fast Gaussian.** The kernel factorizes, $e^{-(x^2 + y^2)/2\sigma^2} = e^{-x^2/2\sigma^2} e^{-y^2/2\sigma^2}$, so filtering columns and then rows with the 1D kernel equals the 2D convolution. Truncate at about $3\sigma$ and renormalize. For large $\sigma$, three successive box filters approximate a Gaussian closely (repeated convolution of boxes tends to a Gaussian) at $O(1)$ per pixel each; recursive (IIR) Gaussian approximations, FFT convolution for very large kernels, and blurring a downsampled image are other options, and production code adds fixed-point arithmetic and SIMD or GPU kernels.

**Median.** It is non-linear and not separable. For 8-bit images, keep a 256-bin histogram of the window and update it as the window slides by removing the column that leaves and adding the one that enters, $O(k)$ per pixel; keeping one histogram per column brings this to $O(1)$. The window view in the snippet uses $O(HWk^2)$ memory, so tile large images.

**Borders and colour.** Reflect padding avoids seams; zero padding darkens the borders of a blur. Apply the filters per channel for colour images (the mosaic already handles an extra channel axis). The three filters match `scipy.ndimage` (`uniform_filter`, `gaussian_filter` with the same truncation radius, `median_filter`) with `mode="mirror"` to floating-point precision, which is a convenient test.

[more](restoration-super-resolution.html)
:::

::: {.qa asked="Ant digital human (Nowcoder roundup 2025)"}
**Write pseudocode for quadrilinear interpolation in PyTorch or CUDA.**

Quadrilinear interpolation is linear interpolation along each of four axes. Find the grid cell that contains the point, compute the fractional offset along each axis, and take a weighted sum of the cell's $2^4 = 16$ corners, where each corner's weight is the product of four 1D weights. For a point $x$ in index coordinates, let $i_a = \lfloor x_a \rfloor$, clamped so that $i_a + 1$ is still on the grid, and $f_a = x_a - i_a$. Then

$$
v(x) = \sum_{c \in \{0, 1\}^4} \Big( \prod_{a=1}^{4} f_a^{c_a} (1 - f_a)^{1 - c_a} \Big)\, G[i_1 + c_1, \dots, i_4 + c_4].
$$

The weights are non-negative and sum to 1, since their sum is $\prod_a (f_a + 1 - f_a)$. Equivalently, perform 15 linear interpolations: 8 along the first axis, then 4, 2 and 1.

```python
import itertools, torch

def quadrilinear(grid, pts):
    # grid: (C, D0, D1, D2, D3), each D_a >= 2; pts: (N, 4) continuous index coordinates
    C, *D = grid.shape
    hi = torch.tensor(D, device=pts.device) - 1
    pts = torch.minimum(pts.clamp(min=0), hi)                 # clamp to the grid (border mode)
    i0 = torch.minimum(pts.floor().long(), hi - 1)            # lower corner; i0 + 1 stays valid
    f = pts - i0                                              # (N, 4) fractions in [0, 1]
    strides = torch.tensor([D[1] * D[2] * D[3], D[2] * D[3], D[3], 1], device=pts.device)
    flat, out = grid.reshape(C, -1), 0
    for corner in itertools.product((0, 1), repeat=4):        # the 16 cell corners
        c = torch.tensor(corner, device=pts.device)
        w = torch.where(c.bool(), f, 1 - f).prod(dim=1)       # product of 1D weights, (N,)
        out = out + flat[:, ((i0 + c) * strides).sum(1)] * w  # gather (C, N), weight, add
    return out.T                                              # (N, C)
```

Autograd gives gradients with respect to both the grid (for a learned lookup table) and the coordinates. To test it, check that a multilinear function such as $x_1 x_2 x_3 x_4 + x_1$ is reproduced exactly, that grid nodes return their stored values, and that fixing the fourth coordinate to an integer matches `F.grid_sample` trilinear sampling with `align_corners=True`.

**CUDA.** Launch one thread per output point, or per (point, channel) pair for coalesced reads when channels are stored last:

```text
n = blockIdx.x * blockDim.x + threadIdx.x;  if n >= N: return
for a in 0..3:  x = clamp(pts[n][a], 0, D[a] - 1);  i[a] = min(floor(x), D[a] - 2);  f[a] = x - i[a]
for ch in 0..C-1:
    acc = 0
    for corner in 0..15:                       # bit a of corner chooses i[a] or i[a] + 1
        w = 1;  off = 0
        for a in 0..3:
            b = (corner >> a) & 1
            w *= b ? f[a] : 1 - f[a];  off += (i[a] + b) * stride[a]
        acc += w * grid[ch][off]
    out[n][ch] = acc
```

The backward kernel scatters $w \cdot \partial L/\partial v$ into the 16 corners with `atomicAdd`, because many points can fall in the same cell. Hardware texture filtering covers at most three dimensions and uses low-precision weights, so a 4D lookup either does all 16 reads by hand or combines two trilinear texture fetches with a manual interpolation along the fourth axis. Uses include 4D lookup tables for colour enhancement (RGB plus a context channel), space-time feature grids, and two-plane light fields $(u, v, s, t)$. The cost is $O(16NC)$ and is dominated by memory reads.
:::

::: {.qa asked="Alibaba Taotian round 2, Nowcoder 2025"}
**Count node popularity over a high-volume stream of events.**

Clarify the query first: exact counts or only the top $k$, all-time or a sliding window, event rate and number of distinct nodes, latency, and how much error is acceptable. For exact counts, shard a hash map by node id and merge per-shard top-$k$ lists. When the number of distinct nodes makes that too large, use mergeable sketches (Count-Min for counts, Misra-Gries or Space-Saving for heavy hitters) kept per time bucket.

**Exact design.** Ingest through a durable log partitioned by a hash of the node id, so each node's count lives on exactly one worker, and pre-aggregate per micro-batch to cut traffic. Each worker keeps a hash map from node to count ($O(1)$ per event) and a periodically refreshed local top $k$ ($O(U \log k)$ with a heap over $U$ local nodes). Merging the per-shard lists gives the exact global top $k$, because no node is split across shards; very hot nodes can be split over sub-keys and summed at query time. Checkpoint state with the log offsets so a failed worker can replay.

**Approximate counts.** A Count-Min sketch keeps $d$ rows of $w$ counters, one hash function per row; an event increments one counter per row, and a query returns the minimum over rows, which cannot be below the true count. In one row the overcount is the total $X$ of other nodes that hash to the same counter, and with pairwise-independent hashes $\mathbb{E}[X] \le N/w$ for $N$ events. With $w = \lceil e/\varepsilon \rceil$, Markov's inequality gives $P(X > \varepsilon N) \le 1/e$; rows are independent, so all $d$ rows exceed $\varepsilon N$ with probability at most $e^{-d} \le \delta$ for $d = \lceil \ln(1/\delta) \rceil$. Memory is $O(\varepsilon^{-1} \log(1/\delta))$ whatever the number of nodes, and shard sketches add elementwise.

```python
import math, random

P = (1 << 61) - 1                                    # prime for the hash family

class CountMin:
    def __init__(self, eps=1e-3, delta=1e-4, seed=0):
        self.w, self.d = math.ceil(math.e / eps), math.ceil(math.log(1 / delta))
        rng = random.Random(seed)                    # same seed on every shard -> mergeable
        self.ab = [(rng.randrange(1, P), rng.randrange(P)) for _ in range(self.d)]
        self.table = [[0] * self.w for _ in range(self.d)]

    def _cols(self, key):                            # key: integer node id
        return [((a * key + b) % P) % self.w for a, b in self.ab]

    def add(self, key, count=1):
        for row, col in zip(self.table, self._cols(key)):
            row[col] += count

    def query(self, key):                            # never below the true count
        return min(row[col] for row, col in zip(self.table, self._cols(key)))

    def merge(self, other):                          # sketches of two shards add elementwise
        for r1, r2 in zip(self.table, other.table):
            for j, v in enumerate(r2):
                r1[j] += v
```

**Heavy hitters.** Misra-Gries keeps $k - 1$ counters: increment a tracked node, insert a new one if a slot is free, otherwise decrement all. Each decrement round cancels $k$ distinct events, so it happens at most $N/k$ times: counts are low by at most $N/k$, and every node above frequency $N/k$ survives. Space-Saving instead replaces the smallest of $k$ counters and inherits its count plus one, overestimating by at most $N/k$.

**Windows and trade-offs.** For a sliding window, keep per-minute buckets (maps or sketches) in a ring buffer and sum the last $W$; decayed scores are the alternative. Exact counting needs memory proportional to the number of distinct nodes; sketches use fixed memory, with an additive error up to $\varepsilon N$ that swamps rare nodes but does not matter for the top $k$.
:::

::: {.qa asked="OpenAI RE (Aced, ~2025)"}
**Given a Python package, find the latest Python version that supports it, using far fewer calls to the expensive compatibility check than there are versions. Then make the solution work in the general case.**

Binary search needs a monotone predicate, so first agree with the interviewer on what is monotone. Parse versions into integer tuples, since string comparison puts "3.10" before "3.9". If support is one contiguous range, keep the releases at or above a version known to work; `works(v)` is then True up to the answer and False after it, and a last-True binary search needs $\lceil \log_2(N+1) \rceil$ checks. The report does not spell out the general case. One version that needs a hierarchical search: support is not monotone over the flat list (a patch release breaks the package) but is monotone level by level.

Assume that at every level a group (a major, or a minor within a major) contains a supported release only if its first release is supported, and that groups whose first release works come before those whose first release fails; later releases inside a group may still be broken. Binary-search the majors by their first releases, recurse into the last major that passes, and repeat for minors and patches. No later group contains a supported release, so the answer lies in the last passing group, where the same assumption holds one level down.

```python
def last_true(keys, pred):                     # pred is True on a prefix of keys, then False
    lo, hi = 0, len(keys)                      # invariant: True on keys[:lo], False on keys[hi:]
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(keys[mid]):
            lo = mid + 1
        else:
            hi = mid
    return lo - 1                              # -1 if no key satisfies pred

def first_release(tree, prefix):               # oldest version under a node of the tree
    while tree:
        k = min(tree)
        prefix, tree = prefix + (k,), tree[k]
    return prefix

def latest_supported(tree, works, prefix=()):
    # tree: nested dict major -> minor -> patch -> {} holding releases at or above a version
    # known to work; works(version_tuple) -> bool is the expensive check.
    if not tree:
        return prefix
    keys = sorted(tree)                        # numeric order, so 3.10 sorts after 3.9
    i = last_true(keys, lambda k: works(first_release(tree[k], prefix + (k,))))
    return None if i < 0 else latest_supported(tree[keys[i]], works, prefix + (keys[i],))
```

On releases 3.6.0 to 3.13.2 with support up to 3.11.4 and a broken 3.9.3, the flat search stops at 3.9.2 and the hierarchical one returns 3.11.4. It uses $O(\log M + \log m + \log p)$ checks for $M$ majors, $m$ minors and $p$ patches, which is $O(\log N)$, and fetches only the version lists along its path. For the earliest supporting version, mirror it: represent each group by its latest release and search for the first True. With no structure at all, the worst case needs every version checked, since the one supported version could be anywhere, so stating the assumption is part of the answer.
:::

::: {.qa asked="OpenAI RS, PracHub 2026"}
**Implement a card game whose rules arrive in stages: rank hands (three of a kind beats a pair, which beats a high card), then add a flush between three of a kind and pair, then add a rule that discards some hands before comparison. Return a log of the game, and keep the design open to new rules.**

The round tests whether the first part's design absorbs the later parts without a rewrite; the reporter warned that changing the design afterwards was very painful. Separate the card model, hand evaluation, comparison, discard policy, game flow and logging, and make the hand categories data: an ordered list of rules, each a function that returns a tie-break key when the hand matches and `None` otherwise.

```python
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class Card:
    rank: int                                     # 2..13 with J, Q, K = 11, 12, 13
    suit: str

def ranks(hand):
    return tuple(sorted((c.rank for c in hand), reverse=True))

def of_a_kind(n):                                 # one factory gives pair and three of a kind
    def match(hand):
        hits = [r for r, k in Counter(c.rank for c in hand).items() if k >= n]
        return (max(hits),) + ranks(hand) if hits else None
    return match

def flush(hand):
    return ranks(hand) if len({c.suit for c in hand}) == 1 else None

class Game:
    def __init__(self, rules, keep=lambda hand: True):
        self.rules, self.keep = rules, keep       # rules: [(name, match)], strongest first

    def evaluate(self, hand):                     # -> (strength, tiebreak), category name
        for i, (name, match) in enumerate(self.rules):
            if (tb := match(hand)) is not None:
                return (len(self.rules) - i, tb), name

    def play(self, hands):                        # hands: {player: [Card]} -> winners, log
        log, scores = [], {}
        for p, hand in hands.items():
            if not self.keep(hand):
                log.append(f"{p} discards {ranks(hand)}")
                continue
            scores[p], name = self.evaluate(hand)
            log.append(f"{p}: {name}")
        best = max(scores.values(), default=None)
        winners = sorted(p for p, s in scores.items() if s == best)
        return winners, log + [f"winners: {winners}"]

V1 = [("three of a kind", of_a_kind(3)), ("pair", of_a_kind(2)), ("high card", ranks)]
V2 = V1[:1] + [("flush", flush)] + V1[1:]         # part 2: one new entry, nothing else changes
keep_pairs = lambda hand: of_a_kind(2)(hand) is not None   # part 3: a swappable discard policy
```

A hand's score is the tuple (category strength, tie-break), and Python compares tuples lexicographically, so comparing two hands needs no category-specific code; part 2 is one new list entry and part 3 one new function passed to `Game`. The log is returned rather than printed, so tests can assert on it. Ask about what the reports leave open: the hand size (three cards is the smallest that allows three of a kind), aces, whether three of a kind counts as a pair for the discard rule (here it does), tie-breaking within a category, and what happens when every hand is discarded (here, no winner).

Test each category boundary (pair against high card, flush against pair, three of a kind against flush), ties, and the all-discarded case, plus a regression test that adding the flush leaves comparisons between non-flush hands unchanged. A single `compare` function with nested cases for each pair of categories grows quadratically and must be edited for every new rule; a subclass per category fits poorly because a hand's category is computed from its cards, not declared. Natural follow-ups, such as a straight or a different discard rule chosen at runtime, then stay small, local changes.
:::

::: {.qa asked="Anthropic CodeSignal (Glassdoor, guides)"}
**In a timed CodeSignal assessment, implement a simplified banking system whose requirements arrive in levels. How do you design and test it?**

The reports describe building bank-account functionality in Python, with requirements that arrive in stages. That matches CodeSignal's published format for project-based tasks: one task in four progressive levels, at most 90 minutes, partial credit per passing test, and method signatures fixed by earlier levels, so later levels force you to refactor your own code. Unofficial reconstructions on practice sites (not confirmed) list accounts and deposits, then transfers and a top-spenders ranking, then payments with delayed cashback, then account merges. Design so those later levels are cheap:

- Take a timestamp in every operation from level 1, and settle pending timed events (a min-heap by due time, with a sequence number for ties) at the start of every public method.
- Route every balance change through one internal method that also records history and updates aggregates such as outgoing totals.
- Validate in helpers (missing account, self-transfer, insufficient funds) and return exactly what the specification says; `None`, `False` and `""` are different answers to the tests.
- Give every ranking an explicit secondary key, such as amount descending then id ascending.

```python
import heapq
from collections import defaultdict

class Bank:                                       # names and return values are illustrative
    def __init__(self):
        self.balance, self.outgoing = {}, defaultdict(int)
        self.history = defaultdict(list)          # account -> [(timestamp, balance)]
        self.pending, self.seq = [], 0            # heap of (due, seq, account, amount)

    def _settle(self, ts):                        # first line of every public method
        while self.pending and self.pending[0][0] <= ts:
            due, _, acc, amount = heapq.heappop(self.pending)
            if acc in self.balance:
                self._change(due, acc, amount)

    def _change(self, ts, acc, amount):
        self.balance[acc] += amount
        self.history[acc].append((ts, self.balance[acc]))

    def schedule(self, due, acc, amount):         # e.g. a delayed credit
        heapq.heappush(self.pending, (due, self.seq, acc, amount))
        self.seq += 1

    def transfer(self, ts, src, dst, amount):
        self._settle(ts)
        if src == dst or {src, dst} - self.balance.keys() or self.balance[src] < amount:
            return None
        self._change(ts, src, -amount)
        self._change(ts, dst, amount)
        self.outgoing[src] += amount
        return self.balance[src]

    def top_spenders(self, ts, n):
        self._settle(ts)
        return sorted(self.balance, key=lambda a: (-self.outgoing[a], a))[:n]   # explicit tie-break
```

**Process.** Read the whole prompt, including any overview of later levels, then implement one level at a time, running the tests after each method and submitting often. Add your own quick checks for missing or duplicate accounts, self-transfers, insufficient funds, ranking ties, and an event due exactly at an operation's timestamp; typos and wrong return types cost more here than algorithmic choices. A scheduled event costs $O(\log P)$ and a ranking $O(A \log A)$ for $A$ accounts.
:::

Deeper: [Vision Backbones](vision-backbones.html) for attention and convolution arithmetic, and [Diffusion Models](diffusion-models.html) for the sampler.

### Algorithm problems from the reports

These are the specific algorithm problems named in the reports. Problems of this kind were standard in the Chinese reports, even for research-flavoured roles, and also appeared in some Meta, Microsoft, Apple and Adobe loops. Each answer gives the standard approach, its complexity and the likely follow-up.

::: {.qa asked="Kuaishou Kling multimodal intern, Nowcoder 2025"}
**Find the longest palindromic substring.**

Every palindrome is symmetric about a centre, which is either a character (odd length) or the gap between two characters (even length). There are $2n - 1$ centres, so expand outward from each one and keep the longest match.

1. Index the centres by $c = 0, \dots, 2n-2$: start with `lo = c // 2`, `hi = (c + 1) // 2`; these are equal for a character centre and adjacent for a gap.
2. While both ends are inside the string and `s[lo] == s[hi]`, move them outward.
3. The palindrome is then `s[lo+1:hi]`; compare its length with the best so far.

This is correct because the expansion from a centre stops exactly at the longest palindrome with that centre, and every palindrome has some centre.

```python
def longest_palindrome(s):
    best_lo, best_hi = 0, 0                    # half-open interval [lo, hi)
    for c in range(2 * len(s) - 1):            # n character centres and n - 1 gaps
        lo, hi = c // 2, (c + 1) // 2
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            lo, hi = lo - 1, hi + 1
        if hi - lo - 1 > best_hi - best_lo:
            best_lo, best_hi = lo + 1, hi
    return s[best_lo:best_hi]
```

Time $O(n^2)$ in the worst case (a string of one repeated letter), space $O(1)$. Manacher's algorithm gets $O(n)$: it reuses the radius of the mirror-image centre inside the rightmost palindrome found so far, so the right boundary only moves forward. Common follow-ups: count all palindromic substrings (the same expansion, adding one per successful step), or the longest palindromic subsequence, which needs an $O(n^2)$ interval DP instead.
:::

::: {.qa asked="Tencent Youtu 2024, JD 2026 (Nowcoder)"}
**Find the maximum subarray sum, then return the subarray itself.**

Kadane's algorithm: let `cur` be the best sum of a subarray that ends at index $i$. Either that subarray is `a[i]` alone or it extends the best one ending at $i-1$, so `cur = a[i] + max(cur, 0)`. The answer is the largest `cur` seen. To return the subarray, remember where the current run started and copy the run's start and end whenever it beats the best.

1. When `cur < 0`, a negative prefix can only hurt, so restart at $i$ (record `l = i`).
2. Otherwise extend.
3. Update the best sum and its `(l, i)` when `cur` is larger.

Starting from `a[0]` rather than 0 handles the all-negative case: the answer is then the largest single element, not an empty subarray.

```python
def max_subarray(a):
    best, best_l, best_r = a[0], 0, 0
    cur, l = a[0], 0
    for i in range(1, len(a)):
        if cur < 0:
            cur, l = a[i], i                  # start a new run at i
        else:
            cur += a[i]
        if cur > best:
            best, best_l, best_r = cur, l, i
    return best, a[best_l:best_r + 1]
```

Time $O(n)$, space $O(1)$ besides the returned slice. Ask whether an empty subarray is allowed (then the answer is `max(best, 0)`). Follow-ups: the circular version is the larger of the ordinary answer and `total - (minimum subarray sum)`, except when every element is negative; the 2D maximum-sum rectangle fixes a pair of rows, collapses the columns into a 1D array and runs Kadane, in $O(m^2 n)$.
:::

::: {.qa asked="Meituan CV intern, Nowcoder 2024"}
**Generate all permutations of a list.**

Backtracking: build the permutation one position at a time, choosing any element not yet used, recurse, then undo the choice. Every permutation corresponds to exactly one root-to-leaf path in this search tree, so each is produced once.

For the variant with duplicate values, sort first and add one rule: skip `nums[i]` if it equals `nums[i-1]` and `nums[i-1]` is not currently used. That forces equal values to be placed in their left-to-right order, so each distinct arrangement appears exactly once. The snippet below already includes this rule; on distinct input it never fires.

```python
def permutations(nums):
    nums, out, path = sorted(nums), [], []
    used = [False] * len(nums)
    def backtrack():
        if len(path) == len(nums):
            out.append(path[:])
            return
        for i, x in enumerate(nums):
            if used[i] or (i > 0 and x == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True; path.append(x)
            backtrack()
            path.pop(); used[i] = False
    backtrack()
    return out
```

Time $O(n \cdot n!)$ (there are $n!$ leaves and copying each costs $O(n)$), extra space $O(n)$ for the recursion and `path`, plus the output. An empty list gives `[[]]`. An alternative is to swap elements into position in place, which avoids the `used` array; with duplicates it needs a per-level set of values already tried. Follow-ups: the next permutation in lexicographic order (find the rightmost ascent, swap with the smallest larger element to its right, reverse the suffix), or the $k$-th permutation directly via the factorial number system.
:::

::: {.qa asked="Alibaba Taotian, Nowcoder 2024"}
**Count the paths from the top-left to the bottom-right of a grid moving only right or down.**

A cell can only be entered from the cell above or the cell to its left, so the number of paths into it is the sum of those two counts:

$$
P(i, j) = P(i-1, j) + P(i, j-1), \qquad P(0, 0) = 1 .
$$

Without obstacles there is a closed form. Every path from the corner of an $m \times n$ grid consists of exactly $m - 1$ down moves and $n - 1$ right moves in some order, and a path is determined by which of the $m + n - 2$ steps are the down moves, giving $\binom{m+n-2}{m-1}$ (`math.comb(m + n - 2, m - 1)`).

The obstacles variant needs the DP: set $P = 0$ on blocked cells. Keep a single row, since `dp[j]` already holds the value from the row above and `dp[j-1]` the value to the left.

```python
def unique_paths(grid):                  # grid[i][j] == 1 marks an obstacle
    n = len(grid[0])
    dp = [1] + [0] * (n - 1)
    for row in grid:
        for j in range(n):
            if row[j]:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j - 1]       # above (old dp[j]) + left (dp[j-1])
    return dp[-1]
```

Time $O(mn)$, space $O(n)$. A blocked start or end gives 0. Python integers do not overflow, but in C++ or Java the counts exceed 64 bits for moderately large grids, so interviewers often ask for the answer modulo a prime. Follow-ups: the minimum-cost path (replace the sum with `cost + min(...)`), or allowing all four directions, which makes path counting ill-posed (cycles) and turns the task into BFS shortest paths.
:::

::: {.qa asked="Dewu AIGC intern, Nowcoder 2024"}
**Find the k-th largest element of an array.**

Two standard answers. A size-$k$ min-heap keeps the $k$ largest values seen so far; its root is the $k$-th largest. Quickselect partitions around a pivot and recurses into only the side that contains the answer.

Heap: push the first $k$ elements, then for each later element larger than the root, replace the root. Time $O(n \log k)$, space $O(k)$, and it works on a stream.

Quickselect: pick a random pivot $p$ and split into `greater`, `equal` and `less`. If $k \le$ `len(greater)`, the answer is in `greater`; if $k \le$ `len(greater) + len(equal)`, it is $p$; otherwise search `less` for rank $k - $ `len(greater) - len(equal)`. The three-way split keeps it fast when many values are equal.

```python
import heapq, random

def kth_largest_heap(a, k):
    h = a[:k]
    heapq.heapify(h)
    for x in a[k:]:
        if x > h[0]:
            heapq.heapreplace(h, x)
    return h[0]

def kth_largest(a, k):                      # quickselect, 1 <= k <= len(a)
    while True:
        p = random.choice(a)
        greater = [x for x in a if x > p]
        n_eq = sum(x == p for x in a)
        if k <= len(greater):
            a = greater
        elif k <= len(greater) + n_eq:
            return p
        else:
            k -= len(greater) + n_eq
            a = [x for x in a if x < p]
```

A random pivot discards a constant fraction of the array in expectation, so the expected work is a geometric series, $O(n)$; the worst case is $O(n^2)$. This version copies the array ($O(n)$ space); an in-place Lomuto or Hoare partition gives $O(1)$ extra space. Median of medians guarantees worst-case $O(n)$ but has a large constant. Follow-up: top-$k$ from a stream or across machines uses the heap, merging per-shard top-$k$ lists.
:::

::: {.qa asked="Alibaba Tongyi 2024, ByteDance AIGC 2024 (Nowcoder)"}
**Find the shortest path in a grid with obstacles, and return the path.**

With unit-cost moves, breadth-first search finds shortest paths: it removes cells from the queue in non-decreasing order of distance, so the first time it reaches the goal is along a shortest path. To return the path rather than its length, store each cell's parent when it is first discovered and walk back from the goal.

1. Put the start in the queue with parent `None`. The `parent` dict doubles as the visited set.
2. Pop a cell; if it is the goal, follow parents back to the start and reverse.
3. Otherwise push each in-bounds, free, unvisited neighbour and record its parent.
4. If the queue empties, the goal is unreachable.

```python
from collections import deque

def shortest_path(grid, start, goal):          # grid[r][c] == 1 is a wall
    R, C = len(grid), len(grid[0])
    if grid[start[0]][start[1]] or grid[goal[0]][goal[1]]:
        return None
    parent, q = {start: None}, deque([start])
    while q:
        cell = q.popleft()
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return path[::-1]
        r, c = cell
        for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nb[0] < R and 0 <= nb[1] < C and not grid[nb[0]][nb[1]] and nb not in parent:
                parent[nb] = cell
                q.append(nb)
    return None
```

Time and space $O(RC)$. Mark cells visited when they are pushed, not when popped, or a cell can enter the queue many times. Follow-ups: weighted cells need Dijkstra ($O(V \log V)$ with a heap on a grid); A\* adds an admissible heuristic such as the Manhattan distance for 4-connected unit moves and usually expands far fewer cells; 8-connected moves with diagonal cost $\sqrt 2$ use the octile distance as heuristic; bidirectional BFS searches from both ends and pays off most when the branching factor is high (on an open grid it only about halves the explored area); "may remove up to $k$ walls" adds $k$ to the BFS state.
:::

::: {.qa asked="Alibaba Tongyi Lab intern, Nowcoder 2024"}
**3Sum: find all unique triplets that sum to zero.**

Sort, fix the smallest element `nums[i]`, and solve two-sum on the suffix with two pointers. Sorting also makes deduplication easy, because equal values become adjacent.

1. For each `i`, skip it if `nums[i] == nums[i-1]` (that first value was already handled); stop once `nums[i] > 0`.
2. Set `lo = i + 1`, `hi = n - 1`. If the sum is too small, `nums[lo]` cannot reach zero with any remaining partner (all are at most `nums[hi]`), so drop it with `lo += 1`; if too large, drop `nums[hi]` symmetrically.
3. On a hit, record it, then move `lo` past all copies of its value and step `hi` down.

```python
def three_sum(nums):
    nums, out = sorted(nums), []
    for i in range(len(nums) - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                out.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                hi -= 1
    return out
```

Time $O(n^2)$, extra space $O(1)$ apart from the sort and the output. After a hit, the new `lo` value is strictly larger, so no duplicate triplet is produced for this `i`. A hash-set two-sum is also $O(n^2)$ but makes deduplication messier. Follow-ups: 3Sum closest (track the best $|s - \text{target}|$ with the same pointers), and $k$-Sum by recursing down to two-sum, in $O(n^{k-1})$.
:::

::: {.qa asked="G-bits 2024, ByteDance 2020, Baidu 2022 (Nowcoder)"}
**Search for a target in a rotated sorted array.**

Binary search still works because, for any `mid`, at least one of the halves `[lo, mid]` and `[mid, hi]` is sorted. Compare `a[lo]` with `a[mid]` to find which half is sorted, check whether the target lies within that half's range, and discard the other half accordingly.

1. If `a[mid]` is the target, return `mid`.
2. If `a[lo] <= a[mid]`, the left half is sorted: go left exactly when `a[lo] <= target < a[mid]`.
3. Otherwise the right half is sorted: go right exactly when `a[mid] < target <= a[hi]`.

```python
def search_rotated(a, target, allow_dups=False):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == target:
            return mid
        if allow_dups and a[lo] == a[mid] == a[hi]:
            lo, hi = lo + 1, hi - 1           # cannot tell which half is sorted
        elif a[lo] <= a[mid]:                 # left half sorted
            if a[lo] <= target < a[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                 # right half sorted
            if a[mid] < target <= a[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```

Time $O(\log n)$ for distinct values, space $O(1)$. With duplicates, `a[lo] == a[mid] == a[hi]` gives no information (in `[1, 1, 1, 0, 1]` the dip could be on either side), so the code shrinks both ends by one; the worst case becomes $O(n)$, and no algorithm can avoid that on inputs that are all equal except for one element. Follow-up: find the rotation point (the minimum) by comparing `a[mid]` with `a[hi]`, then binary search the correct sorted piece.
:::

::: {.qa asked="Xiaohongshu CV intern, Nowcoder 2024"}
**Generate all subsets of a set.**

Each element is either in or out, so there are $2^n$ subsets. Two standard ways to list them: a bitmask from 0 to $2^n - 1$, where bit $j$ says whether `nums[j]` is included, or backtracking that records the current path at every node of the search tree and only extends it with elements to the right of the last one chosen (which avoids producing the same set in two orders).

```python
def subsets_bitmask(nums):
    n = len(nums)
    return [[nums[j] for j in range(n) if m >> j & 1] for m in range(1 << n)]

def subsets(nums):                          # also handles duplicate values
    nums, out, path = sorted(nums), [], []
    def backtrack(start):
        out.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue                    # same value already tried at this depth
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()
    backtrack(0)
    return out
```

Time $O(n \cdot 2^n)$ (each of the $2^n$ subsets is copied in $O(n)$), extra space $O(n)$ for the recursion besides the output. For input with duplicates (the usual follow-up), sort and skip a value that equals its left neighbour at the same depth: at that depth the earlier copy already started every subset that this one would. The bitmask version cannot dedupe this way; it would need a set of tuples. Other follow-ups: subsets of size exactly $k$ (stop recursing at depth $k$), or subsets summing to a target (prune once the sum exceeds it, if all values are positive).
:::

::: {.qa asked="Xiaohongshu CV intern, Nowcoder 2024"}
**Search a 2D sorted matrix.**

Clarify which kind of sorted matrix it is; the two common versions need different algorithms.

1. Each row is sorted and each row starts after the previous row ends. Reading the matrix row by row then gives one sorted array of length $mn$, so binary search over an index $k$ and read element `M[k // n][k % n]`. Time $O(\log mn)$.
2. Rows and columns are sorted separately (the first element of a row can be smaller than the last element of the row above). Start at the top-right corner. It is the largest value in its row and the smallest in its column, so if it is larger than the target, the whole column can be dropped, and if smaller, the whole row can. Each step removes a row or a column. Time $O(m + n)$.

```python
def search_flat(M, target):                 # version 1
    if not M or not M[0]:
        return False
    m, n = len(M), len(M[0])
    lo, hi = 0, m * n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        v = M[mid // n][mid % n]
        if v == target:
            return True
        lo, hi = (mid + 1, hi) if v < target else (lo, mid - 1)
    return False

def search_staircase(M, target):            # version 2
    r, c = 0, (len(M[0]) - 1 if M else -1)
    while r < len(M) and c >= 0:
        if M[r][c] == target:
            return True
        if M[r][c] > target:
            c -= 1
        else:
            r += 1
    return False
```

Both use $O(1)$ space. For version 2, binary searching every row costs $O(m \log n)$, which beats the staircase only when $m$ is much smaller than $n$. The staircase walk also counts the elements $\le$ a value in $O(m + n)$, which is the building block for "the $k$-th smallest element of a sorted matrix" (binary search on the value).
:::

::: {.qa asked="SenseTime multimodal intern, Nowcoder 2026"}
**Reverse a linked list, reading the input and printing the output yourself.**

In an ACM-style round you get no function signature: read stdin, build the list, solve, print exactly the expected format. Ask for the input format; here the first token is $n$ and the next $n$ tokens are the values. Reading all of stdin at once with `split()` tolerates any mix of spaces and newlines.

The reversal walks the list once and turns each `next` pointer around, keeping three references: `prev` (the already reversed part), `head` (the current node) and `nxt` (the rest, saved before the pointer is overwritten).

```python
import sys

class Node:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt

def reverse(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev, head = head, nxt
    return prev

def main():
    data = sys.stdin.read().split()
    n = int(data[0]) if data else 0
    head = None
    for v in reversed(data[1:n + 1]):       # build front to back
        head = Node(int(v), head)
    head, out = reverse(head), []
    while head:
        out.append(str(head.val))
        head = head.next
    print(" ".join(out))

if __name__ == "__main__":
    main()
```

Time $O(n)$, extra space $O(1)$ for the reversal. The recursive version (reverse the tail, then set `head.next.next = head` and `head.next = None`) uses $O(n)$ stack, and CPython's default recursion limit of 1000 makes it fail on long lists. Test the empty list and a single node. Follow-ups: reverse only positions $m$ to $n$, or reverse in groups of $k$; both reuse the same three-pointer loop with careful reconnection at the boundaries.
:::

::: {.qa asked="Baidu 2024, ByteDance 2020 (Nowcoder)"}
**Count the islands in a grid, then count connected regions in an image where neighbours join if their pixel values differ by at most a threshold.**

Both are counts of connected components in a grid graph. Scan every cell; each time an unvisited cell that can start a region is found, increment the count and flood-fill everything reachable from it. Only the edge test differs: for islands, two 4-neighbours are connected when both are land; for the image, when $|I(p) - I(q)| \le t$ (every pixel then belongs to some region).

```python
from collections import deque

def count_regions(img, thr=None):
    # thr is None: islands, img holds 0/1. Otherwise: 4-neighbours join if |diff| <= thr.
    R, C = len(img), (len(img[0]) if len(img) else 0)
    seen = [[False] * C for _ in range(R)]
    count = 0
    for r in range(R):
        for c in range(C):
            if seen[r][c] or (thr is None and not img[r][c]):
                continue
            count += 1
            seen[r][c] = True
            q = deque([(r, c)])
            while q:
                y, x = q.popleft()
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < R and 0 <= nx < C and not seen[ny][nx]:
                        if (img[ny][nx] if thr is None
                                else abs(int(img[ny][nx]) - int(img[y][x])) <= thr):
                            seen[ny][nx] = True
                            q.append((ny, nx))
    return count
```

Time and space $O(RC)$. Practical points: recursive DFS on a large image can recurse once per pixel and overflow the stack, hence the explicit queue; `int(...)` matters for `uint8` NumPy arrays, where `a - b` wraps around; the threshold relation is not transitive, so one region can contain pixels whose values differ by much more than $t$ through a chain of small steps. Union-find is the alternative: union each pixel with its right and lower neighbours when they pass the test and count roots; it suits the follow-up where land cells arrive one at a time and the count is needed after each.
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**Compute the median of every sliding window.**

Keep the current window's values in a sorted structure that supports insert, delete and access to the middle. The window changes by one insertion and one deletion per step, so recomputing from scratch ($O(k \log k)$ per window) is wasteful.

The shortest correct version keeps a sorted Python list and uses binary search to insert the incoming value and delete the outgoing one:

```python
from bisect import bisect_left, insort

def window_medians(a, k):
    win = sorted(a[:k])
    out = []
    for i in range(k, len(a) + 1):
        mid = k // 2
        out.append(win[mid] if k % 2 else (win[mid - 1] + win[mid]) / 2)
        if i == len(a):
            break
        del win[bisect_left(win, a[i - k])]  # remove the value leaving the window
        insort(win, a[i])
    return out
```

The searches are $O(\log k)$ but list insertion and deletion shift elements, so the worst case is $O(nk)$ time (fast in practice because the shift is a memory move), with $O(k)$ space.

For guaranteed $O(n \log k)$, use two heaps: a max-heap for the lower half and a min-heap for the upper half, balanced so the lower one has the same size or one more. The median is the top of the lower heap (or the mean of both tops). Heaps cannot delete an arbitrary element, so deletion is lazy: record the outgoing value in a hash map of pending removals, adjust a balance counter for the side it was on, and pop such values only when they reach a heap's top.

For median filtering of 8-bit images, use a 256-bin histogram of the window instead. Sliding the window one pixel updates only the entering and leaving column or row, and the median is found by accumulating bin counts up to half the window size, so the cost no longer grows as $k^2 \log k$ with the window size.
:::

::: {.qa asked="Megvii PhD researcher, Nowcoder 2021"}
**In one pass, find the maximum difference a[j] - a[i] with i < j.**

For each $j$, the best $i$ is the position of the smallest element before $j$. So scan left to right while keeping the running minimum of the prefix: at each $j$, the candidate is `a[j] - min(a[:j])`, then update the minimum. This is the "best time to buy and sell a stock once" problem.

```python
def max_diff(a):                            # requires len(a) >= 2
    lo, best = a[0], float("-inf")
    for x in a[1:]:
        best = max(best, x - lo)            # sell at x, bought at the cheapest earlier price
        lo = min(lo, x)
    return best
```

Time $O(n)$, space $O(1)$. Order matters: compute the candidate before updating the minimum, or $i = j$ would be allowed. For a strictly decreasing array the answer is negative; the stock version takes `max(0, best)` because you may choose not to trade. To return the indices, also store the index of the running minimum and the pair at each improvement.

Follow-ups: unlimited transactions (the profit is the sum of all positive day-to-day increases); at most $k$ transactions (a DP over day, transactions used and whether you hold a share, $O(nk)$); a cooldown or a fee per transaction (the same hold/not-hold state machine with one extra term).
:::

::: {.qa asked="Kuaishou AIGC (CSDN snippet)"}
**Find the first missing positive integer in O(n) time and O(1) extra space.**

The answer lies in $1, \dots, n+1$ for an array of length $n$, because $n$ slots can hold at most the values $1, \dots, n$. So use the array itself as the hash table: put each value $v$ with $1 \le v \le n$ at index $v - 1$, then the first index $i$ with `a[i] != i + 1` gives the answer $i + 1$; if none, the answer is $n + 1$.

1. For each $i$, while `a[i]` is in range and its target slot does not already hold the same value, swap it into its slot.
2. Scan for the first mismatch.

```python
def first_missing_positive(a):              # modifies a in place
    n = len(a)
    for i in range(n):
        while 1 <= a[i] <= n and a[a[i] - 1] != a[i]:
            j = a[i] - 1
            a[i], a[j] = a[j], a[i]
    for i in range(n):
        if a[i] != i + 1:
            return i + 1
    return n + 1
```

Each swap puts one value into its final slot, where it stays, so there are at most $n$ swaps in total and the nested loop is $O(n)$ overall. The condition `a[a[i] - 1] != a[i]` stops infinite swapping on duplicates. Assign through the saved index `j`: `a[i], a[a[i] - 1] = ...` in Python reassigns `a[i]` before computing the second target. The alternative is index marking: replace out-of-range values with $n + 1$, then for each value $v \le n$ make `a[v-1]` negative; the first positive slot gives the answer. Both modify the input, which "O(1) extra space" allows; if the input must be kept, say so and copy it.
:::

::: {.qa asked="Kuaishou AIGC (CSDN snippet)"}
**Merge overlapping intervals.**

Sort by start. Then any interval that overlaps the current merged block must come right after it in this order, so a single sweep suffices: if the next interval starts at or before the end of the last merged block, extend that block's end; otherwise start a new block.

Correctness: after sorting, the starts are non-decreasing, so once an interval starts after the current block's end, no later interval can reach back into that block, and the block is final.

```python
def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)  # max: the new interval may lie inside
        else:
            out.append([s, e])
    return out
```

Time $O(n \log n)$ for the sort, $O(n)$ for the output. Two details: take the `max` of the ends, since an interval can be contained in the block; and ask whether touching intervals such as `[1, 2]` and `[2, 3]` should merge (closed intervals: yes, `<=`; half-open: use `<`). Follow-ups: insert one interval into an already sorted, disjoint list in $O(n)$ without re-sorting; the minimum number of meeting rooms (sort start and end events, sweep with $+1/-1$, take the maximum running count, or keep a min-heap of end times); the intersection of two sorted interval lists with two pointers.
:::

::: {.qa asked="ByteDance CV search, Nowcoder 2023"}
**Word break: return every way to split a string into dictionary words.**

Define `go(i)` as the list of all sentences that split the suffix `s[i:]`. It tries every dictionary word that starts at $i$ and prefixes it to every sentence of the remaining suffix. Memoizing on $i$ means each suffix is solved once, which matters on inputs like `"aaaa...ab"`, where plain DFS re-explores the same failing suffixes exponentially often.

1. Optionally first run the boolean DP: `ok[i]` is true if `s[i:]` can be split. If `ok[0]` is false, return `[]` before building anything.
2. `go(len(s)) = [""]`; otherwise loop over end positions `j` up to the longest word length and recurse when `s[i:j]` is a word.

```python
from functools import lru_cache

def word_break_all(s, words):
    words = set(words)
    L = max(map(len, words), default=0)
    @lru_cache(maxsize=None)
    def go(i):                              # all sentences for s[i:]
        if i == len(s):
            return [""]
        res = []
        for j in range(i + 1, min(len(s), i + L) + 1):
            if s[i:j] in words:
                res += [s[i:j] + (" " + rest if rest else "") for rest in go(j)]
        return res
    return go(0)
```

The output itself can be exponential: for `"a" * n` with words `{"a", "aa"}` the number of sentences is a Fibonacci number, so no algorithm is polynomial in $n$ alone. The cost is $O(nL)$ substring checks (each costing up to $O(L)$ to hash) plus time proportional to the total length of all sentences built. The memo stores sentence lists for every suffix, which is where the memory goes; the feasibility pass keeps that from being wasted when there is no answer at all. Recursion depth is up to $n$. The common follow-up is just the yes/no or count version, which is the $O(nL)$ DP without building strings.
:::

::: {.qa asked="Baidu, Nowcoder 2022"}
**Edit distance where insertion, deletion and substitution have different costs.**

Let $D(i, j)$ be the cheapest way to turn the first $i$ characters of $a$ into the first $j$ characters of $b$. The last operation of an optimal edit script either deletes $a_i$, inserts $b_j$, or aligns $a_i$ with $b_j$ (free if equal, a substitution otherwise), so

$$
D(i, j) = \min\big(D(i-1, j) + c_{\text{del}},\; D(i, j-1) + c_{\text{ins}},\; D(i-1, j-1) + c_{\text{sub}}\,[a_i \ne b_j]\big),
$$

with $D(i, 0) = i\,c_{\text{del}}$ and $D(0, j) = j\,c_{\text{ins}}$. Different costs change nothing structurally; if $c_{\text{sub}} > c_{\text{ins}} + c_{\text{del}}$, the DP simply never uses substitutions. To list the operations, walk back from $(m, n)$, at each cell choosing a predecessor whose value plus step cost equals the current value.

```python
def edit_distance(a, b, ins=1, dele=1, sub=1):
    m, n = len(a), len(b)
    D = [[i * dele] + [0] * n for i in range(m + 1)]
    D[0] = [j * ins for j in range(n + 1)]
    cost = lambda i, j: 0 if a[i - 1] == b[j - 1] else sub
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i][j] = min(D[i-1][j] + dele, D[i][j-1] + ins, D[i-1][j-1] + cost(i, j))
    ops, i, j = [], m, n
    while i or j:
        if i and j and D[i][j] == D[i-1][j-1] + cost(i, j):
            if a[i-1] != b[j-1]:
                ops.append(f"sub {a[i-1]}->{b[j-1]}")
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i-1][j] + dele:
            ops.append(f"del {a[i-1]}"); i -= 1
        else:
            ops.append(f"ins {b[j-1]}"); j -= 1
    return D[m][n], ops[::-1]
```

Time $O(mn)$. The distance alone needs only two rows, $O(\min(m, n))$ space after swapping so the shorter string indexes the columns (and swapping the insertion and deletion costs with it). Recovering the operations in linear space needs Hirschberg's divide and conquer. With floating-point costs, store back-pointers instead of testing equality.
:::

::: {.qa asked="Meta RS full-time, Reddit 2026"}
**Find the diameter of a binary tree, counted in nodes rather than edges.**

The longest path in a tree bends at exactly one highest node. At that node it goes down the deepest path of the left subtree and the deepest path of the right subtree. So one post-order DFS that returns each subtree's height (in nodes) and updates a global best with `left + right + 1` at every node finds it.

The trap in the report: the common LeetCode version counts edges. A path with $k$ nodes has $k - 1$ edges, so the node count is `left + right + 1` with heights measured in nodes, while the edge count is `left + right`. Fix the convention before coding and check it on a single node (1 node, 0 edges) and an empty tree (0 nodes).

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def diameter_nodes(root):
    best = 0
    def height(node):                       # nodes on the longest downward path
        nonlocal best
        if node is None:
            return 0
        l, r = height(node.left), height(node.right)
        best = max(best, l + r + 1)         # longest path that bends at node
        return 1 + max(l, r)
    height(root)
    return best
```

Time $O(n)$, space $O(h)$ for the recursion, where $h$ is the height; a degenerate (linked-list-shaped) tree with many thousands of nodes exceeds Python's recursion limit, so the follow-up is an iterative post-order traversal with an explicit stack. For an $n$-ary tree, keep the two largest child heights. For a general tree given as an edge list, run BFS from any node to the farthest node $u$, then BFS from $u$; the farthest distance is the diameter.
:::

::: {.qa asked="Meta RS full-time, Reddit 2026"}
**Find the length of the longest valid parentheses substring.**

Stack approach: push indices of `'('`. Keep, at the bottom of the stack, the index of the last position that cannot be part of any valid substring, starting with the sentinel $-1$. On `')'`, pop: if the stack becomes empty, this `')'` is unmatched and becomes the new bottom; otherwise the valid substring ending here starts right after the index now on top, so its length is `i - stack[-1]`.

Two-pass counters with $O(1)$ space: scan left to right counting `open` and `close`. When they are equal, the scanned stretch since the last reset is balanced, giving length `2 * close`; when `close > open`, reset both to 0. This misses cases like `"(()"` where opens never get matched down to equality, so repeat the scan right to left with the roles swapped.

```python
def longest_valid(s):
    stack, best = [-1], 0
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)             # unmatched ')' is the new barrier
            else:
                best = max(best, i - stack[-1])
    return best

def longest_valid_O1(s):
    best = 0
    for seq, op in ((s, "("), (s[::-1], ")")):
        o = c = 0
        for ch in seq:
            o, c = (o + 1, c) if ch == op else (o, c + 1)
            if o == c:
                best = max(best, 2 * c)
            elif c > o:
                o = c = 0
    return best
```

Both run in $O(n)$ time; the stack uses $O(n)$ space and the counters $O(1)$. A third option is the DP where `dp[i]` is the length of the longest valid substring ending at $i$. Edge cases: the empty string, `")("`, and `"()(())"` (adjacent valid blocks must join).
:::

::: {.qa asked="Microsoft AS intern, PracHub 2026"}
**Find all pairs of elements with the minimum absolute difference.**

Sort the array. The closest pair must then be adjacent: if $x < z < y$ in sorted order, then $y - x = (y - z) + (z - x)$ exceeds both $y - z$ and $z - x$, so a non-adjacent pair can never be strictly closest. One pass computes the minimum gap between neighbours and a second collects every neighbouring pair with that gap.

```python
def min_diff_pairs(a):
    a = sorted(a)
    gaps = [y - x for x, y in zip(a, a[1:])]
    if not gaps:
        return []
    best = min(gaps)
    return [[x, y] for x, y in zip(a, a[1:]) if y - x == best]
```

Time $O(n \log n)$ for the sort, $O(n)$ extra space. The pairs come out in ascending order, which is the usual required format. Clarify duplicates: with distinct values (the common statement) the above is complete; if values repeat and the minimum gap is 0, a value occurring $c$ times forms $\binom{c}{2}$ pairs by index, of which adjacency lists only $c - 1$, so ask whether pairs are by value or by index. If the values are bounded integers, counting sort or radix sort makes it $O(n + \text{range})$. Follow-ups: the minimum difference between elements of two arrays (sort both and advance two pointers), and the $k$-th smallest pair distance (binary search on the distance, counting pairs within it with a sliding window, $O(n \log n + n \log D)$ for value range $D$).
:::

::: {.qa asked="Adobe research intern (Glassdoor)"}
**Find the second-smallest element of an array.**

One pass with two variables, `first` (smallest so far) and `second`. For each `x`: if `x < first`, the old minimum becomes the runner-up; otherwise if `x` beats `second`, it becomes the runner-up. First clarify duplicates: for `[1, 1, 2]`, is the answer 1 (second in sorted order) or 2 (second-smallest distinct value)? The flag below handles both, and the function returns `None` when no second element exists.

```python
def second_smallest(a, distinct=True):
    first = second = float("inf")
    for x in a:
        if x < first:
            first, second = x, first
        elif x < second and (x != first or not distinct):
            second = x
    return None if second == float("inf") else second
```

Time $O(n)$, space $O(1)$; about $2n$ comparisons in the worst case. The follow-up is the minimum number of comparisons. Run a knockout tournament: $n - 1$ comparisons find the minimum, since every other element must lose once. The second smallest lost only to the minimum, so it is among the at most $\lceil \log_2 n \rceil$ elements the minimum beat directly, and $\lceil \log_2 n \rceil - 1$ more comparisons find the smallest of them. Total: $n + \lceil \log_2 n \rceil - 2$, and this is known to be optimal in the worst case. Implementing it needs the list of opponents each winner beat, $O(n)$ extra space. Generalization: the $k$-th smallest via quickselect (expected $O(n)$) or a size-$k$ max-heap ($O(n \log k)$).
:::

::: {.qa asked="Adobe research intern (Glassdoor)"}
**Solve the 0/1 knapsack problem.**

Let $dp_i(c)$ be the best value using the first $i$ items with total weight at most $c$. Item $i$ is either skipped or taken once:

$$
dp_i(c) = \max\big(dp_{i-1}(c),\; dp_{i-1}(c - w_i) + v_i\big) \quad \text{if } c \ge w_i .
$$

Only the previous row is needed, so keep one array and iterate the capacity downward: when `dp[c]` is updated, `dp[c - w]` still holds the previous row's value, which means item $i$ is not used twice. Iterating upward would give the unbounded knapsack instead. To return the chosen items, record whether item $i$ was taken at capacity $c$, then walk back from the last item at capacity $W$.

```python
def knapsack(weights, values, W):
    dp = [0] * (W + 1)
    took = [[False] * (W + 1) for _ in weights]
    for i, (w, v) in enumerate(zip(weights, values)):
        for c in range(W, w - 1, -1):       # downward: each item at most once
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
                took[i][c] = True
    items, c = [], W
    for i in range(len(weights) - 1, -1, -1):
        if took[i][c]:
            items.append(i)
            c -= weights[i]
    return dp[W], items[::-1]
```

Time $O(nW)$; space $O(W)$ for the value and $O(nW)$ bits for reconstruction. This is pseudo-polynomial: $W$ takes only $\log W$ bits to write down, so the running time is exponential in the input size, consistent with knapsack being NP-hard. Follow-ups: when $W$ is huge but values are small, run the DP over value (minimum weight for each total value); for $n$ around 40, meet in the middle ($2^{n/2}$ subsets per half, sort one side and binary search); the fractional version is solved greedily by value per unit weight.
:::

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
