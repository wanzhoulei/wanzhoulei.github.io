---
title: "Large Language Models: Pretraining, Fine-Tuning and Alignment"
subtitle: "A guided reading of how LLMs are trained, adapted and served: next-token training and tokenization, data and scaling laws, decoding and speculative sampling, LoRA, SFT, reward models, PPO, DPO and GRPO, quantization and distillation, retrieval, evaluation, and vision-language models, with every derivation shown."
order: 0.52
updated: 2026-11-15
---

A large language model (LLM) is a transformer trained to predict the next token of text, first on trillions of tokens of general data (pretraining), then on much smaller, curated data that teaches it to follow instructions and to prefer responses people like (fine-tuning and alignment). Almost everything an interviewer asks about LLMs comes down to a small set of ideas: a single loss (next-token cross-entropy), how compute should be split between model size and data, a sampling rule at inference, a cheap way to fine-tune (LoRA), and a way to optimize a preference signal without letting the model drift too far (a KL penalty, which DPO turns into a closed form). This page derives each of them. The network itself (attention, positional encodings, the block, the KV cache, FlashAttention, mixture-of-experts) is covered on [transformers](transformers.html); here it is a black box that maps a prefix to a distribution over the next token.

**Roadmap.** Section 1 fixes the training objective, perplexity and tokenization, and contrasts GPT, BERT and T5. Section 2 covers pretraining: the data pipeline, sequence packing, the $6ND$ compute rule, the Kaplan and Chinchilla scaling laws with a worked budget, why Llama-style models are trained far past the Chinchilla point, and emergent abilities. Section 3 covers decoding: temperature, greedy and beam search, top-k and top-p, repetition penalties, and speculative decoding with a proof that it samples exactly from the large model. Section 4 covers adaptation without and with gradients: prompting, chain of thought, full fine-tuning, LoRA, QLoRA, adapters and prefix tuning. Section 5 covers alignment: SFT, the Bradley–Terry reward model, RLHF with PPO, a step-by-step derivation of DPO, GRPO, reward over-optimization, and AI feedback. Section 6 covers inference efficiency: quantization, distillation and batching. Section 7 covers retrieval-augmented generation and tools, Section 8 evaluation and failure modes, Section 9 vision-language models (CLIP and LLaVA), and Section 10 has three short from-scratch implementations. Practice questions follow each section; a rapid-fire review and references close the page.

## Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{V}$, $V$ | the vocabulary (set of tokens) and its size |
| $x = (x_1, \dots, x_n)$ | a token sequence of length $n$; $x_{<t} = (x_1, \dots, x_{t-1})$ is the prefix before position $t$ |
| $\theta$ | model parameters |
| $p_\theta(x_t \mid x_{<t})$ | the model's next-token distribution; written $\pi_\theta$ when the model is treated as a policy (Section 5) |
| $z_t \in \mathbb{R}^V$ | the logits at position $t$; $p_\theta(\cdot \mid x_{<t}) = \mathrm{softmax}(z_t)$ |
| $T$ | temperature, in sampling (Section 3) and in distillation (Section 6) |
| $N$, $D$, $C$ | number of parameters, number of training tokens, training compute in FLOPs |
| $L$ | loss in nats per token |
| $\alpha_N$, $\alpha_D$, $c_N$, $c_D$, $E$ | exponents, coefficients and irreducible loss of a scaling law $L = E + c_N N^{-\alpha_N} + c_D D^{-\alpha_D}$ |
| $x$ (prompt), $y$ (response) | in Section 5 a sequence is a prompt $x$ followed by a response $y = (y_1, \dots, y_m)$ |
| $\pi_{\text{ref}}$ | a frozen reference policy, usually the SFT model |
| $r(x, y)$, $r_\phi$ | a scalar reward for response $y$ to prompt $x$; $r_\phi$ is a learned reward model |
| $y_w$, $y_l$ | the preferred ("winner") and rejected ("loser") response in a comparison |
| $\beta$ | strength of the KL penalty toward $\pi_{\text{ref}}$ (RLHF, DPO, GRPO) |
| $\sigma(u) = 1/(1 + e^{-u})$ | the logistic sigmoid |
| $\rho_t$, $\hat A_t$, $\varepsilon$ | PPO's importance ratio $\pi_\theta / \pi_{\text{old}}$, advantage estimate and clip range |
| $G$ | GRPO group size (outputs sampled per prompt) |
| $W_0 \in \mathbb{R}^{d \times k}$; $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ | a frozen weight matrix and the two LoRA factors of rank $r$ |
| $\alpha$ | LoRA scale numerator; the update is multiplied by $\alpha / r$ |
| $p$, $q$ | target and draft next-token distributions in speculative decoding |
| $\gamma$ | number of draft tokens per verification step in speculative decoding |
| $s$, $z$ | scale and zero point of a quantizer (Section 6 only) |

Two letters are reused with different meanings in different sections: $r$ is the LoRA rank, while a reward is always written with its arguments, $r(x, y)$; and $A$ is a LoRA factor, while an advantage always carries a hat and a subscript, $\hat A_t$. All logarithms are natural unless a base is shown.

## 1. Language modeling

### The chain rule and the training loss

Any probability distribution over sequences can be written, with no assumption at all, as a product of next-token conditionals:

$$
p(x_1, \dots, x_n) = \prod_{t=1}^{n} p(x_t \mid x_{<t}).
$$

Reason: apply the definition of conditional probability, $p(a, b) = p(a)\,p(b \mid a)$, repeatedly from left to right. An autoregressive language model parametrizes each factor with one network that sees only the prefix. In a transformer this is enforced with a causal attention mask, so a single forward pass over a length-$n$ sequence returns all $n$ conditionals at once; feeding the true prefix at every position during training is called teacher forcing.

Training is maximum likelihood. Taking the log turns the product into a sum (and avoids underflow, since a product of thousands of probabilities is far below the smallest float), and averaging over positions gives the per-token loss

$$
\mathcal{L}(\theta) = -\frac{1}{n}\sum_{t=1}^{n} \log p_\theta(x_t \mid x_{<t}).
$$

Each term is a softmax cross-entropy with the observed next token as the class label. With logits $z_t$ and target token $v = x_t$,

$$
-\log \mathrm{softmax}(z_t)_v = -z_{t,v} + \log \sum_{u \in \mathcal{V}} e^{z_{t,u}},
\qquad
\frac{\partial}{\partial z_t}\big(-\log \mathrm{softmax}(z_t)_v\big) = \mathrm{softmax}(z_t) - e_v,
$$

where $e_v$ is the one-hot vector for $v$. The gradient is "predicted distribution minus target", the same as for any classifier (derived on [deep learning training](deep-learning-training.html)).

**What the loss optimizes.** Averaged over the data distribution $p_{\text{data}}$, the expected loss is a cross-entropy, which splits as

$$
\mathbb{E}_{x \sim p_{\text{data}}}\big[-\log p_\theta(x)\big] = H(p_{\text{data}}) + \mathrm{KL}(p_{\text{data}} \,\|\, p_\theta).
$$

The entropy $H(p_{\text{data}})$ does not depend on $\theta$, so maximum likelihood minimizes the forward KL from the data to the model. Forward KL punishes the model heavily wherever it puts near-zero probability on something that occurs in the data, so a model that cannot represent everything spreads its mass out ("mode-covering"). That is one reason the raw distribution has a long, unreliable tail, and why truncated sampling (Section 3) helps.

**Exposure bias.** During training the model always conditions on true prefixes; at generation time it conditions on its own samples. Errors can therefore push it into prefixes unlike anything it was trained on. This mismatch motivates some later ideas on this page: on-policy distillation (Section 6) and RL fine-tuning on the model's own samples (Section 5).

### Perplexity

Perplexity is the exponential of the per-token loss:

$$
\mathrm{PPL} = \exp\Big(-\frac{1}{n}\sum_{t=1}^{n}\log p_\theta(x_t \mid x_{<t})\Big) = 2^{\text{(loss in bits)}}.
$$

It reads as an effective branching factor: a model that is uniform over $V$ tokens has $\mathrm{PPL} = V$, and a perfect model has $\mathrm{PPL} = 1$.

**Worked example.** A model assigns probabilities $\tfrac12, \tfrac14, \tfrac12, \tfrac18$ to the four tokens of a sentence. In bits, the losses are $1, 2, 1, 3$, with mean $1.75$ bits, so $\mathrm{PPL} = 2^{1.75} = 3.36$. In nats the mean loss is $1.75 \ln 2 = 1.213$ and $e^{1.213} = 3.36$, the same number: on this sentence the model is about as uncertain as a uniform choice among 3.36 tokens.

**Perplexity depends on the tokenizer.** Per-token perplexity can only be compared between models that share a tokenizer, because a tokenizer with longer tokens has fewer, harder predictions. To compare across tokenizers, convert the total loss to bits per byte (or per character) of the underlying text. Example: a 20-byte sentence. Tokenizer A splits it into 5 tokens at 2.0 nats each (PPL $e^{2.0} = 7.39$); tokenizer B into 8 tokens at 1.4 nats each (PPL $e^{1.4} = 4.06$). B looks better per token, but the totals are $10$ and $11.2$ nats, i.e. $10/(20 \ln 2) = 0.72$ and $11.2/(20 \ln 2) = 0.81$ bits per byte, so A is the better model of the text.

::: pitfall
"Model X has lower perplexity than model Y" means nothing unless both were evaluated on the same text with the same tokenizer and the same context length. Lower perplexity on web text also does not by itself mean a more useful assistant: alignment changes behaviour in ways that perplexity does not measure.
:::

### Tokenization

A tokenizer maps text to a sequence of integer IDs from a fixed vocabulary, and back. Characters give long sequences (more attention compute, less text per context window); whole words give a huge vocabulary and no way to spell an unseen word. Subword tokenizers sit in between: frequent strings become single tokens, rare words split into pieces, and nothing is out of vocabulary.

**Byte-pair encoding (BPE)**, introduced for neural machine translation by Sennrich, Haddow and Birch (2016) from an old compression algorithm, learns its vocabulary bottom-up:

1. Pre-split the training corpus into words, count each word, and write each word as a sequence of base symbols (characters, or bytes).
2. Count every adjacent pair of symbols, weighting each occurrence by its word's count.
3. Merge the most frequent pair into a new symbol, record the merge, and rewrite every word with it.
4. Repeat until the vocabulary reaches its target size.

**Worked merge example.** Corpus word counts: `cat` 6, `cats` 2, `hat` 4, `that` 3. Base symbols are `a c h s t`.

| Step | Pair counts (highest first) | Merge |
|---|---|---|
| 1 | (a,t) 15, (c,a) 8, (h,a) 7, (t,h) 3, (t,s) 2 | `a t` → `at` |
| 2 | (c,at) 8, (h,at) 7, (t,h) 3, (at,s) 2 | `c at` → `cat` |
| 3 | (h,at) 7, (t,h) 3, (cat,s) 2 | `h at` → `hat` |
| 4 | (t,hat) 3, (cat,s) 2 | `t hat` → `that` |

Check step 1 by hand: (a,t) occurs once in each of the four words, so its count is $6 + 2 + 4 + 3 = 15$; (h,a) occurs in `hat` and `that`, so $4 + 3 = 7$. After four merges the words are `cat`, `cat s`, `hat`, `that`, and the vocabulary is the five base symbols plus `at`, `cat`, `hat`, `that`.

**Encoding** new text applies the learned merges in the order they were learned (earliest merge first), not by looking for the longest match. `chats` starts as `c h a t s`; merge 1 gives `c h at s`; merge 2 (`c at`) does not apply because `c` is followed by `h`; merge 3 gives `c hat s`; merge 4 needs a `t` before `hat` and does not apply. The result is `[c, hat, s]`. An unseen word always gets some segmentation, in the worst case into base symbols.

**Byte-level BPE** (GPT-2) uses the 256 byte values as base symbols, so any string, in any script or with emoji, can be encoded without an "unknown" token. GPT-2's vocabulary has 50,257 entries: 256 bytes, 50,000 merges and one end-of-text token. It also pre-splits text with a regular expression so that merges do not join letters with digits or punctuation (a leading space stays attached to the word after it); without this, `dog`, `dog.` and `dog!` would each waste a vocabulary slot.

**Other subword schemes.** WordPiece (BERT) picks the merge that most increases the training-data likelihood rather than the most frequent pair, and at inference takes the longest matching vocabulary entry first. Unigram (Kudo, 2018) works top-down: it starts from a large candidate vocabulary, repeatedly removes the pieces whose removal costs the least corpus likelihood, and segments new text with the Viterbi algorithm. SentencePiece (Kudo and Richardson, 2018) runs BPE or Unigram directly on raw text, treating the space as an ordinary symbol, so no language-specific pre-tokenizer is needed. Special tokens (beginning and end of sequence, padding, and role markers for chat templates) are added to the vocabulary by hand.

**Vocabulary size is a trade-off.**

- *Larger vocabulary, shorter sequences.* Each token covers more text, so the same context window and the same training compute cover more text. Llama 3 moved from LLaMA's 32,000 tokens to about 128K and reports that this raised English compression from 3.17 to 3.94 characters per token.
- *Larger vocabulary, more parameters.* The input embedding and the output (unembedding) matrix each have $V \times d$ entries. For Llama 3 8B, $128{,}256 \times 4{,}096 = 525$ million per matrix; with separate input and output matrices that is about 1.05 billion, roughly 13% of the 8.03 billion parameters. LLaMA 7B's 32,000-token vocabulary needs 131 million per matrix. The output softmax also costs $O(Vd)$ per token.
- *Larger vocabulary, rarer tokens.* Tokens that almost never occur in training receive almost no gradient, so their embeddings stay poorly trained and can behave erratically when they do appear.

**Side effects interviewers like to ask about.** Numbers may be chunked inconsistently (`1234` as `12 34` in one place and `123 4` in another), which hides digit alignment; LLaMA splits every number into single digits for this reason. Text in languages underrepresented in the tokenizer's training data needs more tokens per word, so it costs more and fits less in context. The model sees tokens, not letters, which makes spelling and counting characters hard. And ` the` and `the` are different tokens, so a trailing space at the end of a prompt can change the output.

### Three families: GPT, BERT and T5

All three are transformers pretrained on unlabeled text; they differ in which tokens each position may see and what is predicted.

- **Decoder-only, causal (GPT).** GPT-1 (Radford et al., 2018) pretrained a 12-layer decoder with the next-token loss on books, then fine-tuned it on each task. GPT-2 (Radford et al., 2019) scaled to 1.5B parameters on WebText and showed tasks could be performed zero-shot from a prompt. GPT-3 (Brown et al., 2020), with 175B parameters trained on 300B tokens, showed few-shot in-context learning (Section 4). Every position is a training target and generation is natural, left to right.
- **Encoder-only, bidirectional (BERT).** Devlin et al. (2019) mask 15% of tokens and predict them from both sides. Strong representations for classification and retrieval, but no left-to-right factorization of the sequence probability, so no direct generation.
- **Encoder-decoder (T5).** Raffel et al. (2020) cast every task as text in, text out. Pretraining uses span corruption: about 15% of tokens are dropped in spans with mean length 3, each span is replaced by a sentinel token in the encoder input, and the decoder outputs the sentinels each followed by its missing span. The data was C4, a cleaned Common Crawl. T5 encoders are still used as text encoders in image generators such as Imagen, SD3 and FLUX (see [text-to-image](text-to-image-multimodal.html)).

Decoder-only models now dominate generation: the objective gives a training signal at every position, one model serves every task through prompting, and caching keys and values makes generation cheap (see [transformers](transformers.html)).

::: {.qa asked="Amazon AS intern, Glassdoor 2026"}
**What does a tokenizer do, and how does byte-pair encoding work?**

A tokenizer maps text to integer IDs from a fixed vocabulary and back; subword tokenizers keep frequent strings whole and split rare words, so nothing is out of vocabulary. BPE training counts adjacent symbol pairs weighted by word frequency, merges the most frequent pair, records the merge and repeats until the vocabulary is full (with `cat` 6, `cats` 2, `hat` 4, `that` 3: `at`, then `cat`, `hat`, `that`). Encoding replays merges in learned order (`chats` → `c hat s`). Byte-level BPE starts from 256 bytes so any string is encodable; WordPiece picks merges by likelihood gain; Unigram prunes a large vocabulary and segments with Viterbi; SentencePiece works on raw text. Vocabulary size trades shorter sequences against bigger embedding and output matrices and under-trained rare tokens. Side effects: inconsistent number chunking, costlier underrepresented languages, poor spelling and character counting, and sensitivity to trailing spaces.

:::

::: {.qa asked="Apple ML research intern (NLP team), Jointaro 2023"}
**Explain BERT: its architecture, pretraining objectives, and how it differs from GPT.**

BERT is a transformer encoder with bidirectional attention. BERT-Base has 12 layers, hidden size 768 and 12 heads (110M parameters, chosen to match the original GPT); BERT-Large has 24 layers, hidden size 1024 and 16 heads (340M). Inputs are WordPiece tokens (30,000 vocabulary), up to 512 positions, with `[CLS]` first and `[SEP]` after each segment; each input vector sums token, learned position and segment embeddings.

Masked language modeling chooses 15% of positions, replaces 80% of them with `[MASK]`, 10% with a random token and leaves 10% unchanged, and predicts the originals there. The split exists because `[MASK]` never appears in fine-tuning; since the model cannot tell which tokens were altered, it must represent every token well. Only 15% of positions give a loss, a weaker signal per sequence than next-token prediction. Next-sentence prediction classifies from `[CLS]` whether segment B followed A; RoBERTa (Liu et al., 2019) found removing it matched or slightly helped and that BERT was undertrained. Fine-tuning adds one output layer and trains all weights.

Versus GPT: the causal mask makes every position a target and generation natural; bidirectional context gives better representations for understanding tasks at equal size, but masked LM defines no left-to-right factorization, so BERT does not generate directly. Encoders are now used mostly for embeddings, retrieval and classification.

:::

::: qa
**Two models report perplexities of 8.1 and 9.5 on the same test set. Which is better?**

Not answerable without more information. If they use different tokenizers, per-token perplexity is not comparable: convert each model's total loss on the same text to bits per byte (total nats divided by $\ln 2$ times the number of bytes). Also check that the context length and evaluation protocol (sliding window or not, padding masked) match, and that the test set is not in either model's training data. Even then, perplexity measures how well a model predicts that text, not how well it follows instructions.
:::

## 2. Pretraining

### The data pipeline

Pretraining data is mostly web text, plus code, books, scientific papers, encyclopedias, math and multilingual sources. Much of the engineering effort in pretraining goes into cleaning and mixing this data. A typical pipeline:

1. **Extraction.** Pull the main text out of HTML (dropping menus, ads and boilerplate) and identify the language, usually with a small fastText-style classifier.
2. **Heuristic filters.** Rules on document length, mean word length, the fraction of symbols or non-alphabetic words, repeated lines and n-grams, and the presence of common stop words remove spam, tables of numbers and machine-generated junk. C4 (used for T5) and Gopher's MassiveText are well-known rule sets.
3. **Model-based quality filters.** Train a classifier to separate a trusted reference set from raw crawl and keep pages it scores highly; GPT-3 did this with WebText, books and Wikipedia as the positive class. CCNet (Wenzek et al., 2020) instead keeps pages with low perplexity under a small n-gram model trained on Wikipedia. FineWeb-Edu (Penedo et al., 2024) had an LLM rate the educational value of a sample of pages, then trained a small classifier on those ratings to filter the whole crawl.
4. **Deduplication.** Exact duplicates are removed by hashing documents, lines or paragraphs; near-duplicates (the same article with different headers, templated pages) with MinHash, described below. Lee et al. (2022) found, for example, one 61-word sentence repeated over 60,000 times in C4; deduplicating made models emit memorized text about ten times less often and reach the same accuracy in fewer steps.
5. **Safety and privacy.** Filter toxic content and remove personal information such as emails and phone numbers.
6. **Decontamination.** Remove documents that overlap benchmark test sets, for example by n-gram overlap (GPT-3 used 13-grams), so that evaluation scores are not inflated (Section 8).
7. **Mixing.** Choose how often to sample each source. Weights are deliberately not proportional to size: in GPT-3, filtered Common Crawl (410B tokens) made up 60% of training examples and was seen 0.44 times over the 300B-token run, while Wikipedia (3B tokens) made up 3% and was seen 3.4 times. Many recent recipes also raise the share of high-quality and domain data (code, math) near the end of training, as the learning rate decays.

**How much repetition is acceptable?** When unique data runs out, Muennighoff et al. (2023) found that up to about 4 epochs of repeated data gave a loss almost the same as fresh data at a fixed compute budget, with the value of further repetition decaying toward zero.

**MinHash, in a few lines.** Represent each document by its set $S$ of shingles (for example word 5-grams). The Jaccard similarity of two documents is $J = |S_1 \cap S_2| / |S_1 \cup S_2|$. For a random hash function $h$, let $m(S) = \arg\min_{u \in S} h(u)$. Then

$$
P\big[m(S_1) = m(S_2)\big] = J.
$$

Reason: the element of $S_1 \cup S_2$ with the smallest hash is equally likely to be any element of the union, and the two minima coincide exactly when that element lies in the intersection. A signature of $k$ independent min-hashes estimates $J$ by the fraction of agreeing entries. To avoid comparing all pairs, locality-sensitive hashing splits the signature into $b$ bands of $r$ entries and flags a pair as a candidate if any band matches entirely, which happens with probability

$$
P(\text{candidate}) = 1 - (1 - J^{r})^{b}.
$$

With $k = 100$ split as $b = 20$ bands of $r = 5$: $J = 0.5$ gives $0.47$, $J = 0.7$ gives $0.975$ and $J = 0.8$ gives $0.9996$. The curve is a soft step near $(1/b)^{1/r} = 0.05^{0.2} = 0.55$; candidates are then checked with the exact Jaccard similarity.

### Sequence packing and document masking

Training batches have a fixed sequence length (for example 4,096 or 8,192 tokens), but documents vary from a few dozen tokens to hundreds of thousands. Padding short documents wastes compute. **Packing** concatenates tokenized documents, separated by an end-of-sequence token, into one stream and cuts it into fixed-length rows, so almost no position is padding. Long documents are split across rows.

With an ordinary causal mask, a token in a packed row can attend to the end of an unrelated previous document. **Document masking** (intra-document causal masking) prevents this: token $i$ may attend to token $j$ only if $j \le i$ and both belong to the same document, so the mask is block-diagonal with a causal triangle in each block. Position indices are often reset at each document start. For three documents of lengths 3, 2 and 3 packed into 8 positions, a full causal mask allows $8 \cdot 9 / 2 = 36$ (query, key) pairs, while document masking allows $6 + 3 + 6 = 15$:

$$
M = \begin{bmatrix}
1&0&0&0&0&0&0&0\\
1&1&0&0&0&0&0&0\\
1&1&1&0&0&0&0&0\\
0&0&0&1&0&0&0&0\\
0&0&0&1&1&0&0&0\\
0&0&0&0&0&1&0&0\\
0&0&0&0&0&1&1&0\\
0&0&0&0&0&1&1&1
\end{bmatrix}.
$$

Llama 3 used such a mask, and Zhao et al. (2024) found intra-document masking improved language modeling and downstream results compared with plain causal masking over packed rows. Attention kernels with variable-length support compute this block-diagonal attention without materializing the mask. In fine-tuning, packing several conversations into one row needs both the attention mask and the loss mask (Section 5).

### Training compute: $C \approx 6ND$

For a dense transformer with $N$ parameters trained on $D$ tokens:

1. Forward pass: each parameter takes part in one multiply-add per token, which is 2 FLOPs, so about $2N$ FLOPs per token.
2. Backward pass: each weight matrix needs one matrix product for the gradient with respect to its input and one for the gradient with respect to its weights, each as expensive as the forward product, so about $4N$ FLOPs per token.
3. Total: $C \approx 6N$ FLOPs per token, so $C \approx 6ND$.

This ignores the attention-score computation, which Kaplan et al. (2020) count as $2\,n_{\text{layer}}\, n_{\text{ctx}}\, d_{\text{model}}$ extra forward FLOPs per token. Relative to $2N \approx 2 \cdot 12\, n_{\text{layer}} d_{\text{model}}^2$ this is a fraction $n_{\text{ctx}}/(12\, d_{\text{model}})$: for GPT-3 ($n_{\text{ctx}} = 2048$, $d_{\text{model}} = 12288$) about 1.4%. **Check:** GPT-3 175B on 300B tokens gives $6 \times 175 \times 10^9 \times 300 \times 10^9 = 3.15 \times 10^{23}$ FLOPs; the paper reports $3.14 \times 10^{23}$.

Inference needs only the forward pass: about $2N$ FLOPs per generated token. This asymmetry drives the "train smaller models longer" argument below.

### Scaling laws: Kaplan versus Chinchilla

**Kaplan et al. (2020)** found that, when the other two are not the bottleneck, test loss falls as a power law in non-embedding parameters $N$, data $D$ and compute $C$, over many orders of magnitude: $L(N) \propto N^{-0.076}$, $L(D) \propto D^{-0.095}$ and $L(C) \propto C^{-0.050}$. Doubling $N$ multiplies the loss by $2^{-0.076} = 0.949$ (about 5% lower), and doubling $D$ by $2^{-0.095} = 0.936$. Architectural details such as depth versus width mattered little at fixed $N$. Their compute-optimal recipe grew model size much faster than data, $N_{\text{opt}} \propto C^{0.73}$: with 10 times the compute, use 5.4 times the parameters and only $10^{0.27} = 1.86$ times the data, training large models and stopping well before convergence. GPT-3 (175B parameters, 300B tokens, under 2 tokens per parameter) reflects this view.

**Hoffmann et al. (2022), "Chinchilla"**, trained over 400 models from 70M to over 16B parameters on 5B to 500B tokens and estimated the compute-optimal $N$ and $D$ three ways: the minimum of training curves at fixed compute, fitted parabolas of loss against size at fixed compute (IsoFLOP profiles), and a fitted parametric loss. All three gave $N_{\text{opt}} \propto C^{a}$ and $D_{\text{opt}} \propto C^{b}$ with $a \approx b \approx 0.5$ (0.50 and 0.50; 0.49 and 0.51; 0.46 and 0.54): **parameters and tokens should grow together**, about 20 tokens per parameter. To test it they trained Chinchilla, 70B parameters on 1.4T tokens (exactly 20 per parameter), with the same compute budget as Gopher (280B parameters, 300B tokens). Chinchilla outperformed Gopher, GPT-3 and other larger models, reaching 67.5% on MMLU, more than 7 points above Gopher. The $6ND$ rule gives $5.0 \times 10^{23}$ FLOPs for Gopher and $5.9 \times 10^{23}$ for Chinchilla, the same order; the rule of thumb is not exact.

**Deriving the compute-optimal allocation.** Take the parametric form Chinchilla fitted,

$$
L(N, D) = E + c_N N^{-\alpha_N} + c_D D^{-\alpha_D},
$$

where $E$ is the irreducible loss (the entropy of text), the second term is the error from finite model size, and the third the error from finite data. Minimize it subject to $6ND = C$.

1. Substitute $D = C/(6N)$: $L(N) = E + c_N N^{-\alpha_N} + c_D (6N/C)^{\alpha_D}$.
2. Set the derivative to zero: $-\alpha_N c_N N^{-\alpha_N - 1} + \alpha_D c_D (6/C)^{\alpha_D} N^{\alpha_D - 1} = 0$.
3. Multiply by $N$: $\alpha_N c_N N^{-\alpha_N} = \alpha_D c_D (6N/C)^{\alpha_D} = \alpha_D c_D D^{-\alpha_D}$. At the optimum, the two reducible loss terms stand in the fixed ratio $\alpha_D : \alpha_N$.
4. Solve for $N$: $N^{\alpha_N + \alpha_D} = \frac{\alpha_N c_N}{\alpha_D c_D} (C/6)^{\alpha_D}$, so

$$
N_{\text{opt}} = \Big(\frac{\alpha_N c_N}{\alpha_D c_D}\Big)^{\frac{1}{\alpha_N + \alpha_D}} \Big(\frac{C}{6}\Big)^{\frac{\alpha_D}{\alpha_N + \alpha_D}},
\qquad
D_{\text{opt}} = \frac{C}{6 N_{\text{opt}}} \propto \Big(\frac{C}{6}\Big)^{\frac{\alpha_N}{\alpha_N + \alpha_D}}.
$$

So the exponent $a = \alpha_D/(\alpha_N + \alpha_D)$ equals $\tfrac12$ exactly when the two terms decay at the same rate, and then the optimal ratio $D/N$ is constant. Chinchilla's own fit reported $\alpha_N = 0.34$, $\alpha_D = 0.28$ ($a = 0.45$ from these rounded values; the paper gives 0.46), but a replication by Besiroglu et al. (2024) found those fitted values inconsistent with the paper's other two approaches and refit them as $E = 1.82$, $c_N = 482.01$, $c_D = 2085.43$, $\alpha_N = 0.3478$, $\alpha_D = 0.3658$, giving $a = 0.51$. With the refit constants, the optimum at $C = 10^{23}$ is $N = 29.4$B and $D = 566$B, about 19 tokens per parameter, in line with the 20-token rule.

**Worked compute budget.** You have $C = 10^{23}$ FLOPs. With $D = 20N$, $C = 6N \cdot 20N = 120 N^2$, so $N = \sqrt{10^{23}/120} = 28.9$B parameters and $D = 577$B tokens. At an effective $4 \times 10^{14}$ FLOP/s per H100 (about 40% of its dense BF16 peak), training takes $10^{23} / (4 \times 10^{14}) = 2.5 \times 10^8$ GPU-seconds, which is 69,444 GPU-hours or 2.8 days on 1,024 GPUs. Memory: with mixed-precision Adam at about 16 bytes per parameter (2 for BF16 weights, 2 for gradients, 4 for an FP32 master copy, 8 for the two Adam moments), the model and optimizer state need $28.9 \times 10^9 \times 16 = 462$ GB before activations, so they must be sharded across at least six 80 GB GPUs (see [deep learning training](deep-learning-training.html) for ZeRO and parallelism).

**Why the two papers disagreed.** Hoffmann et al. point out that Kaplan et al. used one fixed learning-rate schedule and token budget for all runs, so the loss at intermediate points of a long cosine schedule overestimated what a run sized for that many tokens would reach; this undervalued data. Their own runs were also larger. Later work identified further causes. Pearce and Song (2024) attribute much of the gap to Kaplan counting only non-embedding parameters at small scale: simulating the Chinchilla study under those conditions gives exponents close to Kaplan's. Porian et al. (2024) point to the last layer's compute cost, the warmup length and optimizer settings not tuned per scale, and with these corrected they obtained close agreement with Chinchilla.

### Why Llama-style models train far past the Chinchilla point

Chinchilla answers "what is the lowest loss for a given *training* budget?" A deployed model also pays about $2N$ FLOPs for every token it ever generates. If it will serve trillions of tokens, a smaller model trained on more data can be cheaper overall, even though it costs more training compute to reach the same loss.

**Worked example.** Use the refit law and the $C = 10^{23}$ optimum above: $N = 29.4$B, $D = 566$B, loss $L^* = 2.032$. Ask what a model of half the size needs to reach the same loss. Solve $E + c_N N^{-\alpha_N} + c_D D^{-\alpha_D} = L^*$ for $D$ with $N = 14.7$B: $D = 1.43$T tokens (97 per parameter), and training compute $6ND = 1.26 \times 10^{23}$, 26% more. Each generated token then costs $2 \times 14.7$B fewer FLOPs, so the extra $2.6 \times 10^{22}$ training FLOPs are repaid after $2.6 \times 10^{22} / (2 \times 14.7 \times 10^9) \approx 8.8 \times 10^{11}$ tokens served, under a trillion. A quarter-size model (7.4B) would need 10.1T tokens and 4.5 times the training compute, repaid after about $7.8 \times 10^{12}$ tokens. A widely used model serves far more than that, and a smaller model also has lower latency, needs less memory, and fits on cheaper hardware.

This is the LLaMA strategy. LLaMA (Touvron et al., 2023) trained its 7B and 13B models on 1.0T tokens and its 33B and 65B models on 1.4T, noting that Hoffmann et al. recommend a 10B model on 200B tokens while its 7B model was still improving after 1T; LLaMA-13B outperformed GPT-3 on most benchmarks. For Llama 3, Meta wrote that the Chinchilla-optimal amount for an 8B model is about 200B tokens, yet both the 8B and 70B models "continued to improve log-linearly" up to 15T tokens, about 1,875 tokens per parameter for the 8B model. Sardana et al. (2023) formalized this by adding inference cost to the Chinchilla objective: with large expected demand (around a billion requests), the optimum shifts to smaller models trained longer. They also found quality kept improving up to 10,000 tokens per parameter, though fits made from data at typical ratios overestimated the benefit of the extra tokens at such extreme ratios, so the numbers in the worked example are illustrative.

::: key
**How to say it in an interview.** "Training compute is about $6ND$. Kaplan's laws suggested putting most new compute into parameters; Chinchilla re-did the analysis with schedules matched to each run's length and found parameters and tokens should grow equally, about 20 tokens per parameter, and showed it by beating Gopher with a model four times smaller on the same compute. But Chinchilla only minimizes training cost. Inference costs $2N$ per token forever, so if a model will serve trillions of tokens you train a smaller one far past 20 tokens per parameter, which is what LLaMA did with up to 15 trillion tokens for an 8B model."
:::

### Emergent abilities and the measurement critique

Wei et al. (2022) called an ability **emergent** if it is absent in smaller models and present in larger ones, so that it cannot be predicted by extrapolating small-model performance; examples included multi-digit arithmetic and some BIG-bench tasks, where accuracy sat near chance and then rose sharply past some scale. Schaeffer, Miranda and Koyejo (2023) argued that many of these jumps come from the metric rather than the model: nonlinear or discontinuous metrics such as exact match produce sharp jumps, while continuous metrics such as per-token log-likelihood or edit distance on the same outputs change smoothly and predictably.

**Worked example.** Suppose the answer is 10 tokens long, each token is right with probability $p$ independently, and scoring is exact match, so the score is $p^{10}$. As $p$ improves smoothly through 0.5, 0.7, 0.8, 0.9, 0.95 and 0.99, exact match goes 0.001, 0.028, 0.107, 0.349, 0.599 and 0.904. On a log-scale x-axis of model size this looks like nothing, then a sudden take-off. The critique does not prove that every jump is an artifact; the practical lesson is to track a continuous metric alongside the headline one before claiming emergence.

::: {.qa asked="Meta research intern/RS, Glassdoor (topic only)"}
**What are the main architectural choices in Llama, and why were they made?**

The reported question was only "a research question about Llama", so be ready for architecture, data or scaling. Llama is a decoder-only transformer built from changes that had worked elsewhere (details on [transformers](transformers.html)): pre-normalization with RMSNorm (keeps the residual path an identity, stabilizes training, cheaper than LayerNorm); SwiGLU feed-forward layers with hidden width $\tfrac23 \cdot 4d$ to match parameter count, since gated MLPs gave lower perplexity; rotary position embeddings (Llama 3 raises the base to 500,000 for longer context); no biases; grouped-query attention (Llama 2 at 34B and 70B, Llama 3 at all sizes) to shrink the KV cache. Tokenizer: 32K SentencePiece BPE with split digits and byte fallback, then about 128K tokens in Llama 3 (3.17 to 3.94 English characters per token).

The more distinctive choice is data: LLaMA trained 7B and 13B on 1.0T tokens and 33B and 65B on 1.4T, and LLaMA-13B beat GPT-3 on most benchmarks; Llama 3 trained on over 15T tokens, far past the roughly 200B that is Chinchilla-optimal for 8B, because inference cost dominates for a widely served model. Llama 3 also stays dense rather than mixture-of-experts for training stability, and masks attention across packed documents.

:::

::: {.qa asked="Runway (Glassdoor 2025–26)"}
**Estimate the compute and memory needed to train a given network.**

Compute: a dense transformer costs about $6ND$ FLOPs ($2N$ per token forward, $4N$ backward). For other networks, sum per-layer multiply-adds, double for FLOPs, and triple for forward plus backward. Divide by achievable throughput, typically 30–50% of peak. Example: 1B parameters on 100B tokens is $6 \times 10^{20}$ FLOPs; at an effective $4 \times 10^{14}$ FLOP/s per H100 that is $1.5 \times 10^6$ GPU-seconds, about 420 GPU-hours.

Memory: mixed-precision Adam needs about 16 bytes per parameter (2 for BF16 weights, 2 for gradients, 4 for the FP32 master weights, 8 for the two moments), so 16 GB for a 1B model before activations. Activations scale with batch size, sequence length, width and depth, often dominate, and shrink with activation checkpointing; model and optimizer state shrink per GPU with ZeRO or FSDP sharding. If asked for the compute-optimal size for a budget, use $C = 6N \cdot 20N$, so $N = \sqrt{C/120}$.
:::

::: qa
**Kaplan said grow the model faster than the data; Chinchilla said grow them equally. What changed, and which should you use?**

Kaplan's runs shared one learning-rate schedule and token budget, so a run's loss partway through a long cosine schedule was taken as the loss a shorter, properly scheduled run would reach; that overestimates the loss at small $D$ and so undervalues data. Kaplan also counted only non-embedding parameters and worked at smaller scale, both of which later analyses (Pearce and Song; Porian et al.) showed bias the exponent toward model size. Chinchilla matched schedules to each run, used larger models, and found $N_{\text{opt}} \propto C^{0.5}$, about 20 tokens per parameter. Use Chinchilla to minimize training compute for a target loss; if the model will be served heavily, add inference cost ($2N$ per token) and train a smaller model longer.
:::

## 3. Decoding

The network outputs logits $z_t = W_U h_t$ (the final hidden state times the unembedding matrix), one per vocabulary entry. A **decoding rule** turns them into the next token. Greedy, beam, top-k and top-p all use the same network and differ only in this rule.

### Temperature

$$
p_T(v) = \frac{\exp(z_v / T)}{\sum_{u} \exp(z_u / T)} \;\propto\; p(v)^{1/T},
$$

where $p = p_1$ is the untempered softmax. The second form follows because $\exp(z_v/T) = (\exp z_v)^{1/T}$ and the normalizer of $p$ cancels. Temperature keeps the ranking but changes the shape: $T \to 0$ approaches the argmax, $T = 1$ is the trained distribution, and $T \to \infty$ approaches uniform.

**Worked example** (used for the rest of this section). Five candidate tokens with $p = (0.50, 0.20, 0.15, 0.10, 0.05)$.

| Rule | Resulting distribution |
|---|---|
| $T = 1$ | $(0.50, 0.20, 0.15, 0.10, 0.05)$ |
| $T = 0.5$ (square, renormalize) | $(0.769, 0.123, 0.069, 0.031, 0.008)$ |
| $T = 2$ (square root, renormalize) | $(0.340, 0.215, 0.186, 0.152, 0.107)$ |
| top-k, $k = 2$ | $(0.714, 0.286, 0, 0, 0)$ |
| top-p, $p = 0.8$ | $(0.588, 0.235, 0.176, 0, 0)$ |

Check $T = 0.5$: the squares are $0.25, 0.04, 0.0225, 0.01, 0.0025$, summing to $0.325$, and $0.25/0.325 = 0.769$.

### Greedy and beam search

**Greedy** takes the argmax at every step. It is deterministic and cheap but does not find the most likely sequence. Example: step 1 offers A (0.5), B (0.4), C (0.1). After A the best continuation has probability 0.4; after B, one continuation has 0.9. Greedy picks A and ends with $0.5 \times 0.4 = 0.20$, while B then that continuation has $0.4 \times 0.9 = 0.36$.

**Beam search** keeps the $B$ partial sequences with the highest total log-probability, extends each by every token, and keeps the best $B$ again. With $B = 2$ the example keeps A and B after step 1 and finds the 0.36 sequence. Beam search is still approximate, and sums of log-probabilities favor short outputs, so scores are usually divided by a power of the length. It suits tasks with essentially one right answer (translation, speech recognition). For open-ended text, Holtzman et al. (2020) showed that maximizing likelihood with greedy or beam search produces bland, repetitive text ("degeneration"), and that human text is not the most likely text: its per-token probabilities fluctuate. Pure sampling, at the other extreme, keeps drawing from the unreliable tail: if the tail holds 5% of the mass at every step, the chance of never touching it in 100 tokens is $0.95^{100} = 0.006$.

### Top-k and top-p (nucleus) sampling

**Top-k** keeps the $k$ most likely tokens and renormalizes. With $k = 2$: $0.5/0.7 = 0.714$ and $0.2/0.7 = 0.286$. A fixed $k$ is too loose when the distribution is peaked and too tight when it is flat.

**Top-p** (nucleus sampling, Holtzman et al., 2020) keeps the smallest set of most likely tokens whose total probability is at least $p$, then renormalizes:

$$
\mathcal{V}^{(p)} = \text{smallest top set with } \sum_{v \in \mathcal{V}^{(p)}} p(v) \ge p,
\qquad
\tilde p(v) = \frac{p(v)\,\mathbf{1}[v \in \mathcal{V}^{(p)}]}{\sum_{u \in \mathcal{V}^{(p)}} p(u)}.
$$

In the example the cumulative sums are $0.50, 0.70, 0.85, 0.95, 1.00$, so $p = 0.8$ keeps three tokens (sum $0.85$) and gives $(0.588, 0.235, 0.176)$, while $p = 0.9$ keeps four. The candidate set adapts to the model's confidence: one token when the model is sure, many when it is not. Libraries usually apply temperature first and truncation after.

### Repetition penalties

Language models tend to fall into loops, because once a phrase has appeared, repeating it becomes more likely. Common fixes act on the logits of tokens already generated:

- **CTRL's penalty** (Keskar et al., 2019) divides the logit of every previously generated token by $\theta > 1$ (they used about 1.2). Logits $(2.0, 1.0)$ with the first token already used become $(1.667, 1.0)$, moving its probability from 0.731 to 0.661. Dividing a negative logit would raise it, so implementations multiply negative logits by $\theta$ instead.
- **Frequency and presence penalties** (in the OpenAI API) subtract $\alpha_{\text{freq}} \cdot c_v + \alpha_{\text{pres}} \cdot \mathbf{1}[c_v > 0]$ from the logit of token $v$, where $c_v$ is how often $v$ has appeared so far.
- **No-repeat n-gram blocking** forbids any token that would complete an n-gram already in the output; trigram blocking is common in summarization.

All three also penalize legitimate repetition, such as a name, a variable in code, or JSON keys, so they are tuned per application.

### Speculative decoding

Decoding one token from a large model is limited by memory bandwidth, not arithmetic: every step reads all the weights to produce one token (Section 6). Verifying several tokens in one forward pass costs about the same time as generating one. Speculative decoding (Leviathan, Kalman and Matias, 2023; Chen et al., 2023, who call it speculative sampling) exploits this:

1. A small **draft** model with next-token distribution $q$ proposes $\gamma$ tokens autoregressively.
2. The large **target** model scores the prefix plus all $\gamma$ drafts in one forward pass, giving its distribution $p$ at each of the $\gamma + 1$ positions.
3. Walk through the drafts in order. Accept draft token $x$ with probability $\min\big(1, p(x)/q(x)\big)$.
4. At the first rejection, sample a replacement from the residual distribution $p'(x) = \max(0, p(x) - q(x)) / \sum_u \max(0, p(u) - q(u))$ and discard the remaining drafts. If all $\gamma$ are accepted, sample one extra token from the target's distribution at position $\gamma + 1$, which is already computed.

Each target call yields between 1 and $\gamma + 1$ tokens.

**Theorem.** Every emitted token is distributed exactly according to $p$, so the output has exactly the distribution of sampling from the target model alone.

**Proof** for one position (the multi-token case follows by induction, since each accepted or resampled token is then distributed as $p$ given the prefix emitted so far, which is the same prefix the target conditioned on).

1. The draft proposes $x$ with probability $q(x)$ and it is accepted with probability $\min(1, p(x)/q(x))$, so $P(\text{propose } x \text{ and accept}) = q(x)\min\big(1, p(x)/q(x)\big) = \min\big(q(x), p(x)\big)$.
2. The total rejection probability is $1 - \sum_u \min(p(u), q(u)) = \sum_u \big(p(u) - \min(p(u), q(u))\big) = \sum_u \max\big(0, p(u) - q(u)\big)$, using $\sum_u p(u) = 1$ and $a - \min(a, b) = \max(0, a - b)$.
3. After a rejection the output is $x$ with probability $\max(0, p(x) - q(x)) / \sum_u \max(0, p(u) - q(u))$. Multiplying by step 2, the denominator cancels: $P(\text{reject, then } x) = \max\big(0, p(x) - q(x)\big)$.
4. Adding the two disjoint ways to output $x$: $\min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)$. $\blacksquare$

The draft's quality affects only speed, never the output distribution. If $p = q$ nothing is ever rejected; if the draft is poor, most steps yield one token and time is wasted on drafting.

**How fast?** The acceptance probability at a position is $\alpha = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p, q)$, since $\mathrm{TV}(p, q) = \sum_x \max(0, p(x) - q(x))$. If acceptances are independent with the same $\alpha$, the number of tokens per target call is a geometric variable capped at $\gamma + 1$, with mean

$$
\mathbb{E}[\text{tokens per call}] = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}.
$$

If one draft step costs a fraction $c$ of a target step, Leviathan et al. give the expected wall-time improvement as $(1 - \alpha^{\gamma+1}) / \big((1 - \alpha)(\gamma c + 1)\big)$.

**Worked example.** Target $p = (0.6, 0.3, 0.1)$ and draft $q = (0.3, 0.5, 0.2)$ over three tokens. Acceptance probabilities: $\min(1, 2) = 1$, $\min(1, 0.6) = 0.6$, $\min(1, 0.5) = 0.5$. Then $\alpha = 0.3 + 0.3 + 0.1 = 0.7$. The residual is $\max(0, p - q) = (0.3, 0, 0)$, normalized to $(1, 0, 0)$. Output probabilities: token 1, $0.3 \times 1$ (proposed and accepted) $+\; 0.3 \times 1$ (any rejection, then the residual) $= 0.6$; token 2, $0.5 \times 0.6 = 0.3$; token 3, $0.2 \times 0.5 = 0.1$, which is exactly $p$. With $\alpha = 0.7$ and $\gamma = 4$, each target call yields $(1 - 0.7^5)/0.3 = 2.77$ tokens on average; with $c = 0.05$ the wall-time gain is $2.77 / 1.2 = 2.31\times$.

The papers report 2–3$\times$ speedups on T5-XXL (Leviathan et al.) and 2–2.5$\times$ on the 70B Chinchilla model in a distributed setup (Chen et al.), with unchanged outputs. With temperature or top-p, apply the same transformation to both $p$ and $q$ before the acceptance test; with greedy decoding the rule reduces to accepting a draft token exactly when it equals the target's argmax. Gains shrink at large batch sizes, where decoding is no longer memory-bound.

::: {.qa asked="Tencent multimodal, Nowcoder 2026"}
**How does an LLM turn hidden states into token probabilities, and how does decoding choose the next token?**

The final hidden state (after a last normalization) times the unembedding matrix $W_U \in \mathbb{R}^{V \times d}$ gives logits, and $\mathrm{softmax}(z/T)$ the next-token distribution ($T = 1$ in training). The original transformer ties $W_U$ to the input embedding; LLaMA keeps them separate. Decoding rules then pick a token: greedy (argmax; cheap, not globally optimal, repetitive), beam search (best $B$ partial sequences; good for translation, bland for chat, needs length normalization), top-k (fixed candidate count), top-p (smallest set with mass $\ge p$, adapting to confidence). Lower temperature and tighter truncation cut the unreliable tail (more coherent, less diverse). On $p = (0.5, 0.2, 0.15, 0.1, 0.05)$: $T = 0.5$ gives $(0.769, 0.123, \dots)$, top-2 gives $(0.714, 0.286)$, top-p 0.8 gives $(0.588, 0.235, 0.176)$. Code is in Section 10.

:::

::: qa
**Prove that speculative decoding samples exactly from the target model.**

A draft $x \sim q$ is accepted with probability $\min(1, p(x)/q(x))$, so $P(\text{accept } x) = \min(p(x), q(x))$. The total rejection probability is $1 - \sum_u \min(p(u), q(u)) = \sum_u \max(0, p(u) - q(u))$. On rejection we sample from $\max(0, p - q)$ normalized by that same sum, so $P(\text{reject then } x) = \max(0, p(x) - q(x))$. Summing, $\min(p, q) + \max(0, p - q) = p$. Each emitted token is therefore distributed as $p$ given the prefix, and by induction the whole sequence is distributed as target-only sampling. The acceptance rate is $1 - \mathrm{TV}(p, q)$, which sets the speed but never the output.
:::

::: qa
**Why is beam search good for translation but bad for a chatbot?**

In translation the conditional distribution is concentrated on a few correct outputs, and the most likely one is usually good, so searching for high likelihood helps. In open-ended generation many continuations are acceptable and the most likely ones are generic and repetitive; Holtzman et al. found human text sits well below the maximum-likelihood text in probability. Sampling with top-p or a temperature keeps diversity while cutting the unreliable tail. Beam search also needs length normalization, since summed log-probabilities favor short outputs.
:::

## 4. Adaptation: prompting and fine-tuning

### In-context learning and prompting

GPT-3 (Brown et al., 2020) showed that a large enough model can perform a new task from a description and a few examples in its prompt (zero-, one- or few-shot), with no gradient update; this is **in-context learning**. What the demonstrations contribute is not obvious: Min et al. (2022) found that replacing their labels with random ones barely hurt across 12 models, and that what mattered was showing the label space, the distribution of inputs and the format. Few-shot results are also sensitive to the choice, order and wording of examples; Zhao et al. (2021) traced part of this to biases toward labels that are frequent or recent in the prompt. Several accounts of the mechanism have been proposed (implicit Bayesian inference over latent tasks, Xie et al.; induction heads that copy patterns, Olsson et al.; attention implementing gradient-descent-like updates in simple settings, von Oswald et al.), none of them complete.

**Chain of thought (CoT).** Wei et al. (2022) put worked reasoning in the few-shot examples, so the model writes intermediate steps before the answer; with eight such exemplars, PaLM 540B reached the then state of the art on GSM8K math word problems, surpassing a fine-tuned GPT-3 with a verifier, and the gains appeared mainly in large models. Kojima et al. (2022) obtained large gains without any exemplars by appending "Let's think step by step." Self-consistency (Wang et al., 2023) samples several chains and takes the majority answer, adding 17.9 points on GSM8K over greedy CoT. Two reasons CoT helps: each generated token is one more forward pass, so a long chain buys more computation before the answer; and intermediate results are written into the context, where the answer can condition on them. Reasoning models trained with RL on verifiable rewards (Section 5) learn to produce such chains themselves.

### Full fine-tuning versus parameter-efficient fine-tuning

Full fine-tuning updates every weight. For a 6.74B-parameter model with mixed-precision Adam that is about $16 \times 6.74 \times 10^9 = 108$ GB for weights, gradients and optimizer state before activations, and a full copy of the model per task. **Parameter-efficient fine-tuning (PEFT)** freezes the base model and trains a small number of new or selected parameters: the frozen BF16 weights take 13.5 GB, the trainable part and its optimizer state are tiny, and each task is stored as a file of megabytes. Biderman et al. (2024) compared the two on code and math: LoRA learned substantially less than full fine-tuning, especially with continued pretraining on billions of tokens, but forgot less of the base model's other abilities; full fine-tuning learned weight changes with rank 10 to 100 times higher than typical LoRA ranks. So full fine-tuning suits large domain shifts when compute allows, and PEFT suits many tasks on one base model, small datasets and limited memory.

### LoRA

**Idea.** Hu et al. (2022) hypothesized that the weight change needed to adapt a pretrained model has low rank (building on Aghajanyan et al., 2020, who found fine-tuning works in a low-dimensional subspace). So freeze $W_0 \in \mathbb{R}^{d \times k}$ and learn the change as a product of two thin matrices:

$$
h = W_0 x + \frac{\alpha}{r} B A x,
\qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k).
$$

**Initialization.** $A$ is random (Gaussian in the paper) and $B = 0$, so $\Delta W = BA = 0$ and training starts exactly from the pretrained model. With $g = \partial \mathcal{L} / \partial h$, the chain rule gives

$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r}\, g\,(Ax)^\top,
\qquad
\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r}\, B^\top g\, x^\top.
$$

At initialization $\partial \mathcal{L}/\partial A = 0$ (because $B = 0$) but $\partial \mathcal{L}/\partial B \ne 0$ (because $Ax \ne 0$), so the first step moves $B$, and from then on both move. Zeroing both matrices would fail: each gradient contains the other matrix, so both stay zero forever. Zeroing $A$ and randomizing $B$ also starts from $\Delta W = 0$ and trains, with different dynamics.

**The $\alpha/r$ scale.** The paper sets $\alpha$ to the first $r$ tried and does not tune it; the scale reduces the need to retune the learning rate when $r$ changes. Kalajdzievski (2023) argued for $\alpha/\sqrt{r}$ (rsLoRA) so that larger ranks keep learning.

**Parameter savings.** One matrix costs $r(d + k)$ trainable parameters instead of $dk$. For $d = k = 4096$ and $r = 8$: $8 \times 8192 = 65{,}536$ against $16{,}777{,}216$, which is 0.39%. For a 7B LLaMA-style model with 32 layers:

- LoRA on the query and value projections only: $2 \times 32 \times 65{,}536 = 4.19$M parameters, 0.062% of 6.74B.
- LoRA on all seven projections per layer (q, k, v, o of size $4096 \times 4096$; gate and up $4096 \to 11008$; down $11008 \to 4096$): $8 \times (4 \times 8192 + 3 \times 15104) = 624{,}640$ per layer, 19.99M in total, 0.30%.

On GPT-3 175B the paper reports 10,000 times fewer trainable parameters, training memory down from 1.2 TB to 350 GB, and, with $r = 4$ on the query and value projections only, a checkpoint of 35 MB instead of 350 GB.

**Merging.** After training, $W = W_0 + \frac{\alpha}{r}BA$ has the same shape as $W_0$, so inference has no extra latency; subtracting the product restores the base model for switching tasks. This is why the adapter has no nonlinearity inside it. Alternatively, a server can keep the base model shared and apply many adapters unmerged, one per request.

**Choices.** The paper adapted mostly $W_q$ and $W_v$ and found a rank as small as 1 to 4 sufficed there; under a fixed parameter budget, adapting more matrix types at low rank beat adapting one type at higher rank. Common practice now applies LoRA to all linear layers, starts with $r$ between 8 and 64, raises it if the model underfits (bigger domain shifts need more), and usually uses a larger learning rate than full fine-tuning.

**QLoRA** (Dettmers et al., 2023) stores the frozen base model in 4 bits and trains LoRA adapters in 16 bits; gradients flow through the frozen, dequantized weights into the adapters. Three ingredients: the 4-bit NormalFloat (NF4) data type, whose 16 levels are quantiles of a normal distribution, suited to roughly normally distributed weights; double quantization, which quantizes the per-block scales themselves: with blocks of 64 weights, an FP32 scale per block costs $32/64 = 0.5$ bits per parameter, while 8-bit scales with a second-level FP32 scale per 256 blocks cost $8/64 + 32/(64 \cdot 256) = 0.127$ bits; and paged optimizers that move optimizer state to CPU memory during spikes. A 6.74B model then needs $6.74 \times 10^9 \times 4.127 / 8 = 3.5$ GB for its weights. QLoRA fine-tuned a 65B model on one 48 GB GPU, and its Guanaco models reached 99.3% of ChatGPT's level on the Vicuna benchmark after 24 hours on one GPU.

::: pitfall
LoRA is not free at inference unless you merge, and merging into a quantized base model requires re-quantizing the merged weights, which adds error. Also, "rank 8 is enough" is a finding for narrow tasks on large models; it need not hold for injecting a new domain, where Biderman et al. found LoRA learns substantially less than full fine-tuning.
:::

### Adapters, prefix tuning and prompt tuning

- **Adapters** (Houlsby et al., 2019) insert a small bottleneck MLP (down-project, nonlinearity, up-project, residual connection, initialized near the identity) twice in every layer. On GLUE they came within 0.4% of full fine-tuning while adding 3.6% parameters per task. They add sequential layers, so inference is slower, and the nonlinearity prevents merging.
- **Prefix tuning** (Li and Liang, 2021) keeps the model frozen and learns continuous "virtual token" vectors prepended to the keys and values at every layer, reparametrized through an MLP during training for stability. With 0.1% of the parameters it matched fine-tuning on table-to-text and summarization with full data and did better in low-data settings.
- **Prompt tuning** (Lester et al., 2021) learns soft prompt embeddings at the input only; it closes the gap to full fine-tuning as models exceed billions of parameters.

::: {.qa asked="Baidu, ByteDance, Tencent, NIO +3 (7 reports)"}
**What is LoRA? Give the formula, the parameter count for a large layer, and how you choose the rank.**

Freeze $W_0 \in \mathbb{R}^{d \times k}$ and learn $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$; the forward pass is $h = W_0 x + \frac{\alpha}{r} BAx$. That is $r(d + k)$ trainable parameters: for a $10^4 \times 10^4$ layer with $r = 100$, 2M instead of 100M (2%); for $4096 \times 4096$ with $r = 8$, 65,536 (0.39%). After training, merge $W_0 + \frac{\alpha}{r}BA$, so inference costs nothing extra, which is why the adapter has no nonlinearity. Choose the rank by starting small (8–64 is common), raising it if the model underfits or the domain shift is large, and tuning the learning rate together with $\alpha/r$; more target matrices at low rank often beat fewer at high rank. In diffusion models LoRA usually targets the attention projections ([text-to-image](text-to-image-multimodal.html)).
:::

::: {.qa asked="Baidu 2026, Tencent multimodal 2026 (Nowcoder)"}
**Why is LoRA's $B$ initialized to zero? Could you zero $A$ and initialize $B$ randomly instead?**

With $B = 0$, $\Delta W = 0$, so fine-tuning starts exactly at the pretrained model, and $\partial \mathcal{L}/\partial B = \frac{\alpha}{r} g (Ax)^\top$ is non-zero because $A$ is random, so training proceeds. Swapping roles also starts at $\Delta W = 0$ and trains, since then $\partial \mathcal{L}/\partial A = \frac{\alpha}{r} B^\top g x^\top \ne 0$; the two choices differ in dynamics (Hayou et al., 2024, found the standard choice tolerates larger learning rates). Zeroing both fails: each gradient contains the other matrix, so both stay zero.
:::

::: {.qa asked="NVIDIA AI intern, Jointaro 2023"}
**How would you train an LLM to perform a given list of tasks?**

Treat it as a measurement problem first. Clarify each task's format and metric, latency and cost limits (which fix the model size), available labels, and whether general abilities must be kept. Baseline: prompt a strong instruction-tuned model zero- and few-shot, with retrieval if knowledge is needed. Data: convert labeled data to instruction format with several templates per task (as in FLAN), generate and filter synthetic examples with a stronger model (exact-match checks, unit tests, a judge, deduplication), write hard cases by hand, and decontaminate against evaluation sets. Mixture: cap each task's share and mix in general instruction data to limit forgetting. Training: SFT with the loss on responses only; full fine-tuning for capacity, LoRA for cost, per-task adapters and less forgetting. Evaluation: per-task held-out metrics plus general benchmarks, with attention to the worst task. Beyond SFT: DPO where quality is subjective, RL with verifiable rewards (GRPO) where correctness can be checked.

:::

## 5. Alignment

A pretrained model continues text; it does not reliably follow instructions, and it imitates whatever is common online. InstructGPT (Ouyang et al., 2022) established the recipe that most later work modifies: (1) supervised fine-tuning (SFT) on demonstrations written by labelers, (2) a reward model trained on labelers' rankings of model outputs, (3) reinforcement learning against that reward with PPO. In their human evaluations, outputs of the 1.3B InstructGPT model were preferred to those of the 175B GPT-3. The ingredients came from earlier work: Christiano et al. (2017) learned rewards from human comparisons for Atari and simulated robotics, and Ziegler et al. (2019) and Stiennon et al. (2020) applied the method to language with a KL penalty. Today the stages are typically SFT, then preference optimization (RLHF or DPO), then, for reasoning, RL with automatically checkable rewards (GRPO).

### Supervised fine-tuning with loss masking

Each example is a prompt $x$ and a response $y$ rendered in a chat template, with special tokens marking the system, user and assistant turns and an end-of-turn token. The loss is the next-token loss on **response tokens only**:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{m}\sum_{t=1}^{m} \log \pi_\theta(y_t \mid x, y_{<t}).
$$

In code, the labels are the input IDs with every prompt position set to the ignore index ($-100$ for PyTorch's cross-entropy). Reasons to mask the prompt:

- The prompt is given at inference, so its likelihood is irrelevant; training on it teaches the model to write user turns and system prompts.
- Prompts are often much longer than responses (retrieved documents, few-shot examples), and would dominate the gradient.
- Packed or multi-turn rows must mask the loss on every non-assistant segment, and train every assistant turn including its end-of-turn token, so the model learns to stop.

For a 10-token row with a 6-token prompt, the loss averages over 4 positions. Averaging per token across a batch weights long responses more than averaging per sequence; either is used, but the choice changes what the model is pushed toward. Curated quality can matter more than quantity: LIMA (Zhou et al., 2023) fine-tuned a 65B LLaMA on 1,000 curated examples and was judged equivalent or better than GPT-4 in 43% of comparisons, which the authors take as evidence that most knowledge comes from pretraining. SFT on facts the model does not already know is learned slowly and, once learned, increases its tendency to hallucinate (Gekhman et al., 2024).

### The reward model and Bradley–Terry

Labelers see a prompt and two responses and pick the better one, giving triples $(x, y_w, y_l)$. The **Bradley–Terry** model assumes each response has a latent score $r(x, y)$ and

$$
P(y_w \succ y_l \mid x) = \frac{e^{r(x, y_w)}}{e^{r(x, y_w)} + e^{r(x, y_l)}} = \sigma\big(r(x, y_w) - r(x, y_l)\big),
$$

where the second form follows by dividing the numerator and denominator by $e^{r(x, y_w)}$. A random-utility reading: if a labeler's utility is $r$ plus independent standard Gumbel noise for each response, the probability of preferring $y_w$ is exactly this logistic function, since the difference of two independent Gumbel variables is logistic.

**The loss.** A reward model $r_\phi$ (usually the SFT model with the unembedding replaced by a scalar head on the last token) is fit by maximum likelihood of the observed choices:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\,\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big].
$$

With $\Delta = r_\phi(x, y_w) - r_\phi(x, y_l)$, $\frac{d}{d\Delta}\big(-\log\sigma(\Delta)\big) = -(1 - \sigma(\Delta)) = -\sigma(-\Delta)$: the push is strongest when the model ranks the pair the wrong way and fades as the margin grows. Example: $\Delta = 1$ gives $P = \sigma(1) = 0.731$, loss $0.313$ and derivative $-0.269$.

Two practical points. Only differences of rewards enter, so adding any function of $x$ alone to the reward leaves the loss unchanged; rewards are usually normalized (for example to mean zero on a reference set) before RL. InstructGPT showed labelers $K = 4$ to 9 responses to rank, giving $\binom{K}{2}$ comparisons per prompt; because these are strongly correlated, it trained on all of a prompt's comparisons as one batch element (one forward pass per response), which avoided the overfitting seen when they were shuffled into separate examples. It used a 6B reward model.

### RLHF with PPO

**Objective.** Maximize reward while staying close to the reference (SFT) policy:

$$
\max_{\theta}\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] - \beta\, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big).
$$

The KL term matters for three reasons: the reward model is only accurate near the outputs it was trained on, so unconstrained optimization finds its errors (see over-optimization below); without it the policy can collapse onto a few high-reward outputs; and it preserves fluency. InstructGPT's "PPO-ptx" variant also adds $\gamma_{\text{ptx}}\, \mathbb{E}_{x \sim \text{pretrain}}[\log \pi_\theta(x)]$ to reduce regressions on public benchmarks.

**As a token-level RL problem.** The state is $s_t = (x, y_{<t})$, the action is the next token, the transition appends it, and the episode ends at end-of-sequence. The per-token reward is

$$
r_t = -\beta \log\frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)} + \mathbf{1}[t = m]\; r_\phi(x, y).
$$

Why this implements the objective: the log-probability of a sequence is the sum of token log-probabilities, so $\sum_t r_t = r_\phi(x, y) - \beta \log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$, and its expectation over $y \sim \pi_\theta$ is $\mathbb{E}[r_\phi] - \beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$. The per-token penalty is a one-sample estimate of the sequence KL, spread over its tokens.

**Policy gradient with a critic.** The gradient of the expected return is $\mathbb{E}\big[\sum_t \nabla_\theta \log \pi_\theta(y_t \mid s_t)\, \hat A_t\big]$, where the advantage $\hat A_t$ estimates how much better token $y_t$ was than the policy's average at $s_t$. Subtracting a baseline $b(s_t)$ adds no bias because $\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a \mid s)]\,b(s) = b(s)\,\nabla \sum_a \pi(a \mid s) = 0$. PPO learns a **value model** $V_\psi(s_t)$ (InstructGPT initialized it from the reward model) and computes advantages with generalized advantage estimation, usually with no discounting: $\delta_t = r_t + V_\psi(s_{t+1}) - V_\psi(s_t)$ and $\hat A_t = \sum_{l \ge 0} \lambda^l \delta_{t+l}$ (the GAE question in the [question bank](interview-question-bank.html) derives this). The value model is trained by regression onto the returns $\hat A_t + V_{\psi_{\text{old}}}(s_t)$.

**The clipped surrogate.** PPO (Schulman et al., 2017) samples responses with the current policy, freezes a copy $\pi_{\text{old}}$, and takes several minibatch steps on

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\; \mathrm{clip}(\rho_t, 1 - \varepsilon, 1 + \varepsilon)\, \hat A_t\big)\Big],
\qquad
\rho_t = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{old}}(y_t \mid s_t)},
$$

with $\varepsilon = 0.2$ in the PPO paper. At $\theta = \theta_{\text{old}}$ the gradient of $\rho_t \hat A_t$ equals the policy gradient (importance sampling), but the surrogate is only trustworthy near $\theta_{\text{old}}$. The clip removes the incentive to push $\rho_t$ beyond $[1 - \varepsilon, 1 + \varepsilon]$ in the direction that helps, and the min keeps the pessimistic value. Three cases with $\varepsilon = 0.2$:

- $\hat A_t > 0$, $\rho_t = 1.5$: the min is $1.2\hat A_t$, constant in $\theta$, so the gradient is zero; a good token already made much more likely is not pushed further.
- $\hat A_t < 0$, $\rho_t = 0.5$: $\min(0.5\hat A_t, 0.8\hat A_t) = 0.8\hat A_t$ (since $\hat A_t < 0$), constant, zero gradient; a bad token already suppressed enough is left alone.
- $\hat A_t < 0$, $\rho_t = 1.5$: $\min(1.5\hat A_t, 1.2\hat A_t) = 1.5\hat A_t$, unclipped, so the gradient still pushes the probability down; an update that made a bad token more likely is always corrected.

The full loss is $-\mathcal{L}^{\text{CLIP}}$ plus a weighted value loss. Practical additions include normalizing rewards and advantages, and adapting $\beta$ to hit a target KL (Ziegler et al. did this).

**Four models.** PPO for LLMs holds the policy (trained), the value model (trained), the reward model (frozen) and the reference model (frozen). For a 7B policy and a 7B value model, the two trained models need about 16 bytes per parameter and the two frozen ones 2, so $(16 + 16 + 2 + 2) \times 7 \times 10^9 = 252$ GB before activations. And every iteration must first generate responses autoregressively, which often dominates wall-clock time.

::: key
**How to say it in an interview.** "RLHF maximizes a learned reward minus $\beta$ times the KL to the SFT model. In practice the KL becomes a per-token penalty, $-\beta \log(\pi_\theta/\pi_{\text{ref}})$, summed with the reward model's score at the last token; summed over tokens that's exactly the sequence-level objective. A value model gives per-token advantages through GAE, and PPO's clipped ratio objective lets us take several gradient steps per batch of samples without moving too far. The cost is four models in memory and generation inside the training loop."
:::

### DPO: deriving the closed form

Direct Preference Optimization (Rafailov et al., 2023) observes that the KL-regularized objective above has a closed-form optimal policy, and that substituting it into the Bradley–Terry likelihood gives a loss on the policy alone. Every step:

**Step 1: the optimal policy.** The objective separates over prompts, so fix $x$ and drop it from the notation. For any distribution $\pi$ over responses,

$$
\begin{aligned}
J(\pi) &= \sum_y \pi(y)\, r(y) - \beta \sum_y \pi(y) \log\frac{\pi(y)}{\pi_{\text{ref}}(y)} && \text{(definition of the objective and the KL)} \\
&= -\beta \sum_y \pi(y)\Big[\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} - \frac{r(y)}{\beta}\Big] && \text{(factor out } -\beta\text{)} \\
&= -\beta \sum_y \pi(y) \log\frac{\pi(y)}{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}} && (r/\beta = \log e^{r/\beta}).
\end{aligned}
$$

Define the normalizer $Z = \sum_y \pi_{\text{ref}}(y)\, e^{r(y)/\beta}$ and the distribution $\pi^*(y) = \pi_{\text{ref}}(y)\, e^{r(y)/\beta} / Z$. Substituting $\pi_{\text{ref}}(y)\, e^{r(y)/\beta} = Z\,\pi^*(y)$:

$$
J(\pi) = -\beta \sum_y \pi(y) \log\frac{\pi(y)}{\pi^*(y)} + \beta \log Z \sum_y \pi(y) = -\beta\, \mathrm{KL}(\pi \,\|\, \pi^*) + \beta \log Z.
$$

$Z$ does not depend on $\pi$, and $\mathrm{KL} \ge 0$ with equality only at $\pi = \pi^*$ (Gibbs' inequality), so the unique maximizer is

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\big(r(x, y)/\beta\big).
$$

The optimal policy reweights the reference by the exponentiated reward; $\beta$ sets how sharply.

**Step 2: invert for the reward.** Take logs and rearrange:

$$
r(x, y) = \beta \log\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).
$$

Any reward can be written through its own optimal policy this way. $Z(x)$ is a sum over all possible responses and cannot be computed, but it depends only on $x$.

**Step 3: plug into Bradley–Terry; $Z$ cancels.** Bradley–Terry uses only the reward difference between two responses to the same prompt:

$$
\begin{aligned}
P(y_w \succ y_l \mid x) &= \sigma\Big(\beta \log\frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) - \beta \log\frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \beta \log Z(x)\Big) \\
&= \sigma\Big(\beta \log\frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big).
\end{aligned}
$$

**Step 4: fit the policy by maximum likelihood.** Replace $\pi^*$ with the trainable $\pi_\theta$ and maximize the likelihood of the observed preferences:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\Big(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)\Big].
$$

This is the reward-model loss with the reward replaced by the **implicit reward** $\hat r_\theta(x, y) = \beta \log(\pi_\theta(y \mid x)/\pi_{\text{ref}}(y \mid x))$: training a policy this way is the same as training a reward model whose optimal policy is that policy. Each sequence log-probability is a sum over response tokens (prompt masked, as in SFT).

**The gradient.** With margin $u = \hat r_\theta(x, y_w) - \hat r_\theta(x, y_l)$, $\frac{d}{du}(-\log\sigma(u)) = -\sigma(-u)$, so

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}\Big[\sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big)\big(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\big)\Big].
$$

It raises the likelihood of the chosen response and lowers that of the rejected one, weighted by how wrongly the implicit reward currently orders the pair.

**Worked example.** At initialization $\pi_\theta = \pi_{\text{ref}}$, the margin is 0 and the loss is $\log 2 = 0.693$. With $\beta = 0.1$, suppose training has raised $\log \pi_\theta(y_w) - \log \pi_{\text{ref}}(y_w)$ to $+2.0$ and lowered the same quantity for $y_l$ to $-1.0$. The margin is $0.1 \times 3 = 0.3$, the loss is $-\log\sigma(0.3) = 0.554$, and the pair's gradient weight is $\sigma(-0.3) = 0.426$.

**Properties and limits.**

- Offline and simple: no sampling during training, no reward or value model; the reference log-probabilities can be precomputed once, so only the policy needs to be in memory during training.
- It learns only from the given pairs. Pairs generated by other models are off-policy; iterative or online DPO samples fresh pairs from the current policy and labels them with a reward model or judge.
- The loss sees only the margin, so the chosen response's likelihood can fall too, as long as the rejected one falls faster; this is commonly observed in practice.
- If a pair is always preferred the same way (deterministic preferences), the maximum-likelihood margin is infinite, so the policy pushes $\pi_\theta(y_l) \to 0$ whatever $\beta$ is; Azar et al. (2023) identified this weakness and proposed IPO, which regresses the margin to a finite target.
- Diffusion-DPO adapts the derivation to image diffusion models (see [text-to-image](text-to-image-multimodal.html)).

::: key
**How to say it in an interview.** "The KL-regularized reward objective is maximized by $\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}$; that's $-\beta\,\mathrm{KL}(\pi \| \pi^*)$ plus a constant. Inverting gives $r = \beta \log(\pi^*/\pi_{\text{ref}}) + \beta \log Z(x)$. Bradley–Terry only uses reward differences on the same prompt, so the intractable $Z(x)$ cancels, and I can fit the policy directly with a logistic loss on $\beta$ times the difference of log-ratios. DPO is a reward model whose optimal policy is the model itself."
:::

### GRPO: group-relative advantages, no critic

PPO's value model is as large as the policy, and with a single reward at the end of a response it must learn per-token values from a weak signal. Group Relative Policy Optimization (Shao et al., 2024, introduced in DeepSeekMath) replaces it with a baseline computed from several samples of the same prompt:

1. For each prompt $q$, sample $G$ outputs $o_1, \dots, o_G$ from $\pi_{\text{old}}$ and score them, giving rewards $r_1, \dots, r_G$.
2. Give every token of output $i$ the same advantage, the group-normalized reward (outcome supervision): $\hat A_{i,t} = \big(r_i - \mathrm{mean}(r)\big)/\mathrm{std}(r)$.
3. Maximize PPO's clipped objective averaged over the group, with the KL to the reference in the loss rather than in the reward:

$$
\mathcal{J}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat A_{i,t},\, \mathrm{clip}(\rho_{i,t}, 1 - \varepsilon, 1 + \varepsilon)\hat A_{i,t}\big) - \beta\, \mathbb{D}_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big]\Big)\Bigg].
$$

The per-token KL is estimated as $\mathbb{D}_{\text{KL}} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$, evaluated at the sampled token. It is unbiased: under sampling from $\pi_\theta$, $\mathbb{E}[\pi_{\text{ref}}/\pi_\theta] = \sum_v \pi_{\text{ref}}(v) = 1$, so the expectation is $1 + \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}}) - 1$. And it is never negative, because $u - \log u - 1 \ge 0$ for all $u > 0$. Putting the KL in the loss keeps the advantages purely about the reward. DeepSeekMath used $G = 64$ samples per question, a policy learning rate of $10^{-6}$, KL coefficient 0.04, maximum length 1,024 and batch size 1,024.

**Worked example.** $G = 4$ answers to a math question, one correct: rewards $(1, 0, 0, 0)$, mean $0.25$, population standard deviation $\sqrt{0.25 \times 0.75} = 0.433$. Advantages: $+1.73$ for every token of the correct answer and $-0.58$ for every token of each wrong one (with the sample standard deviation, $0.577$, they are $+1.5$ and $-0.5$; implementations differ). If all four were correct, or all wrong, every advantage would be zero and the prompt would give no gradient; DAPO (Yu et al., 2025) filters such groups out ("dynamic sampling").

**Known biases.** Liu et al. (2025) showed that the per-output $1/|o_i|$ average creates an incentive toward long wrong answers: a negative advantage is spread over more tokens, so each token of a longer wrong answer is penalized less, while correct answers are rewarded more per token when short. Their Dr. GRPO removes the length normalization and the division by the standard deviation (which up-weights questions that are nearly always or nearly never solved).

**RL with verifiable rewards.** GRPO is typically paired with rewards from a checker rather than a learned model: exact match against a reference answer, unit tests for code, a format check. DeepSeek-R1-Zero applied GRPO with such rule-based rewards directly to a base model, and its responses grew longer and more reflective during training. Checkers are harder to exploit than a reward model, though not impossible (tests that do not cover the specification, answer-format tricks).

### Best-of-n, over-optimization and reward hacking

The simplest way to use a reward model is **best-of-n**: sample $n$ responses and return the highest-scoring one. Gao et al. (2022) compute its KL from the base policy as $\log n - (n - 1)/n$: 0.64 nats for $n = 4$, 1.84 for $n = 16$, 3.17 for $n = 64$.

A reward model is a proxy, and optimizing a proxy hard enough finds its errors (Goodhart's law). Gao et al. measured this with a synthetic setup in which a large "gold" reward model stood in for humans. Writing $d = \sqrt{\mathrm{KL}(\pi \,\|\, \pi_{\text{init}})}$, the gold reward followed $R_{\text{bon}}(d) = d(\alpha_{\text{bon}} - \beta_{\text{bon}} d)$ for best-of-n and $R_{\text{RL}}(d) = d(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d)$ for RL: it rises, peaks and then falls while the proxy reward keeps rising. The coefficients changed smoothly with reward-model size, larger reward models over-optimizing less.

Reward hacking in LLMs shows up as length (reward models and judges tend to prefer longer answers, and RLHF gains can partly come from length; Singhal et al., 2023), sycophancy (agreeing with the user's stated view; Sharma et al., 2023), confident-sounding but wrong answers, and, with programmatic rewards, gaming the checker. Mitigations: the KL penalty and early stopping chosen on held-out evaluations not used as the reward; reward-model ensembles (Coste et al., 2023); explicit length control; refreshing the reward model with comparisons of the current policy's outputs; verifiable rewards where possible; and reading samples. The same failure modes, and the same group-relative methods, appear in RL fine-tuning of diffusion models; see the post-training section of [text-to-image](text-to-image-multimodal.html).

### AI feedback and Constitutional AI

Human labels are slow and expensive. **Constitutional AI** (Bai et al., 2022) replaced human labels for harmlessness with AI feedback guided by a short written list of principles (the "constitution"). In a supervised stage, the model critiques and revises its own responses to harmful prompts according to the principles, and is fine-tuned on the revisions. In an RL stage, a model compares pairs of responses according to the principles; these AI preferences train a preference model, which is then used for RL ("RL from AI feedback", RLAIF), while helpfulness labels still came from humans. Lee et al. (2023) later found RLAIF comparable to RLHF on summarization and dialogue. The catch is that the judging model's biases (Section 8) become the training signal.

::: {.qa asked="Amazon AS intern 2026, Microsoft AS intern 2026"}
**Describe the LLM training pipeline after pretraining: what do SFT, RLHF with PPO, DPO and GRPO each do?**

SFT imitates demonstrations: cross-entropy on response tokens, prompt masked. RLHF trains a reward model with the Bradley–Terry loss $-\log\sigma(r_w - r_l)$, then maximizes $\mathbb{E}[r] - \beta\,\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ with PPO: a per-token KL penalty, advantages from a value model, the clipped ratio objective, and four models in memory. DPO uses the closed-form optimum $\pi^* \propto \pi_{\text{ref}}e^{r/\beta}$ to write the reward as $\beta\log(\pi/\pi_{\text{ref}})$ plus a prompt-only term that cancels in Bradley–Terry, giving a logistic loss on the policy alone: offline, stable, two models, but limited to the given pairs. GRPO is online RL without a critic: $G$ samples per prompt, advantage $(r_i - \mathrm{mean})/\mathrm{std}$ for every token, PPO's clipped objective plus a KL term in the loss; it suits checkable rewards. All can over-optimize their reward; $\beta$ limits how far the policy moves.

:::

::: qa
**Derive the DPO loss from the KL-regularized RLHF objective.**

For a fixed prompt, $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \| \pi_{\text{ref}}) = -\beta\sum_y \pi\log\frac{\pi}{\pi_{\text{ref}}e^{r/\beta}}$. Define $Z = \sum_y \pi_{\text{ref}}e^{r/\beta}$ and $\pi^* = \pi_{\text{ref}}e^{r/\beta}/Z$; then the objective is $-\beta\,\mathrm{KL}(\pi \| \pi^*) + \beta\log Z$, maximized uniquely at $\pi = \pi^*$. Taking logs, $r = \beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z(x)$. Bradley–Terry gives $P(y_w \succ y_l) = \sigma(r_w - r_l)$, in which $\beta\log Z(x)$ appears with opposite signs and cancels. Replacing $\pi^*$ by $\pi_\theta$ and maximizing the preference likelihood gives $\mathcal{L} = -\mathbb{E}\log\sigma\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$.
:::

::: qa
**In GRPO, what happens when every sample in a group gets the same reward, and why does the paper put the KL in the loss instead of the reward?**

All advantages $r_i - \mathrm{mean}(r)$ are zero (and the standard deviation is zero, so implementations add a small constant), so the prompt contributes no policy gradient and its samples are wasted; DAPO's dynamic sampling drops such groups and samples more prompts. The KL goes in the loss so the advantage stays a pure function of the group's rewards; with a per-token KL in the reward, the advantage would vary per token and the group normalization would mix reward and penalty. The paper estimates the per-token KL with $\pi_{\text{ref}}/\pi_\theta - \log(\pi_{\text{ref}}/\pi_\theta) - 1$, which is unbiased and never negative.
:::

## 6. Efficiency at inference

**Why decoding is memory-bound.** Generating one token at batch size 1 reads every weight once and does about 2 FLOPs per weight. A 6.74B model in BF16 is 13.5 GB; at an H100's memory bandwidth of about 3.35 TB/s, just reading it takes $13.5 / 3{,}350 \text{ s} \approx 4.0$ ms, so at most about 250 tokens per second, whatever the arithmetic speed. The ratio of FLOPs to bytes is about 1, while the H100 can do about $989 / 3.35 \approx 295$ dense BF16 FLOPs per byte moved, so the arithmetic units sit mostly idle until a batch of hundreds of sequences shares each weight read (the KV cache, which is per sequence, adds further reads). Processing the prompt (prefill) handles many tokens at once and is compute-bound. Two consequences: making weights smaller speeds up decoding almost proportionally, and batching is the main lever for throughput.

### Quantization

A quantizer stores a real value $x$ as an integer $q$ through an affine map:

$$
q = \mathrm{clip}\big(\mathrm{round}(x / s) + z,\; q_{\min},\; q_{\max}\big),
\qquad
\hat x = s\,(q - z).
$$

For int8, **asymmetric** quantization maps a range $[x_{\min}, x_{\max}]$ (widened to contain 0) onto $[-128, 127]$ with $s = (x_{\max} - x_{\min})/255$ and $z = \mathrm{round}(-128 - x_{\min}/s)$, so 0 is represented exactly. **Symmetric** quantization fixes $z = 0$ and $s = \max|x| / 127$. Inside the range the rounding error lies in $[-s/2, s/2]$; when values span many steps it is close to uniform, with variance $\frac{1}{s}\int_{-s/2}^{s/2} e^2\, de = s^2/12$. Each extra bit halves $s$.

**Worked example (symmetric).** $x = (0.52, -1.214, 0.033, 2.54)$: $s = 2.54/127 = 0.02$; $x/s = (26, -60.7, 1.65, 127)$; $q = (26, -61, 2, 127)$; $\hat x = (0.52, -1.22, 0.04, 2.54)$. The errors $(0, -0.006, +0.007, 0)$ are all within $s/2 = 0.01$, but the small value $0.033$ is off by 21%.

**Worked example (asymmetric).** Post-ReLU activations in $[0, 6]$: $s = 6/255 = 0.0235$ and $z = -128$, so 0 maps to $-128$ exactly and 2.5 maps to $\mathrm{round}(106.25) - 128 = -22$, decoded as $106 \times 0.0235 = 2.494$. A symmetric quantizer would need $s = 6/127 = 0.047$, twice the step, and waste every negative code.

**Granularity.** A weight matrix with one row in $[-0.1, 0.1]$ and another in $[-10, 10]$, quantized with one scale per tensor, gets $s = 10/127 = 0.079$; the small row $(0.05, -0.1, 0.02)$ becomes codes $(1, -1, 0)$, decoded as $(0.079, -0.079, 0)$, errors as large as the weights. With one scale per output channel, the small row gets $s = 0.1/127 = 0.00079$, codes $(63, -127, 25)$, errors below $0.0004$. Per-channel weight scales are cheap because they factor out of the matrix product: for output $j$, $\sum_k x_k W_{jk} \approx s_x s_{w,j} \sum_k q_{x,k}\, q_{w,jk}$, an integer dot product accumulated in 32 bits followed by one float multiply. Scales that vary along the summed (input) dimension do not factor out, which is why outliers in activation channels are the hard case. Low-bit weight schemes go finer still, with one scale per group of 64 or 128 weights.

**Weight-only versus weight-and-activation.** Weight-only quantization (for example 4-bit weights with 16-bit activations) stores weights in low precision and dequantizes them inside the matrix-multiply kernel. It saves memory and bandwidth, and therefore speeds up memory-bound decoding (13.5 GB becomes about 3.4 GB at 4 bits, about 1 ms per token at the bandwidth above), but the arithmetic is still in 16 bits. Quantizing activations too (W8A8, in int8 or FP8) lets the matrix multiplies use low-precision tensor cores, which helps compute-bound prefill and large batches, but activations are harder: LLMs develop a few hidden dimensions with very large values. LLM.int8() (Dettmers et al., 2022) handles them with vector-wise scales plus a mixed-precision split, computing the outlier dimensions (above a magnitude threshold, 6 by default) in 16 bits and over 99.9% of values in 8 bits. SmoothQuant (Xiao et al., 2023) uses the identity $XW = (X\,\mathrm{diag}(s)^{-1})(\mathrm{diag}(s)\,W)$ to divide each activation channel $j$ by a factor $s_j$ and multiply the weights that read that channel by the same factor, moving the difficulty from activations to weights so both fit in int8.

**GPTQ** (Frantar et al., 2023) quantizes one layer at a time to minimize $\lVert WX - \hat W X \rVert^2$ on a few calibration inputs $X$. It quantizes the weights column by column and, after each column, updates the not-yet-quantized columns to compensate for the error just made, using the inverse of the Hessian $H = 2XX^\top$ of this squared error (the Optimal Brain Quantization update), organized in batched, numerically stable form. It quantized 175B-parameter models to 3 or 4 bits in about four GPU hours with negligible accuracy loss.

**AWQ** (Lin et al., 2024) observes that protecting about 1% of salient weights greatly reduces quantization error, and that salient weight channels are best identified by the magnitude of the activations they multiply, not by the weights themselves. Rather than keeping those weights in higher precision (awkward for hardware), it scales salient input channels of $W$ up by a factor before quantization and folds the inverse into the preceding operation, choosing the factors by a small search on calibration data. It uses no backpropagation or reconstruction, so it overfits the calibration set less.

### Distillation

Knowledge distillation trains a small student to match a large teacher. Hinton, Vinyals and Dean (2015) soften both distributions with a temperature $T$: with teacher logits $v$, student logits $z$, $p = \mathrm{softmax}(v/T)$ and $q = \mathrm{softmax}(z/T)$,

$$
\mathcal{L} = (1 - \lambda)\,\mathrm{CE}\big(y, \mathrm{softmax}(z)\big) + \lambda\, T^2\, \mathrm{KL}(p \,\|\, q).
$$

A higher $T$ exposes the teacher's relative probabilities among wrong answers ("a 2 looks more like a 3 than a 7"), which carry information that one-hot labels lack.

**Why $T^2$.** Derive the gradient of the soft term:

1. $\mathrm{KL}(p \| q) = \sum_j p_j \log p_j - \sum_j p_j \log q_j$, and only the second sum depends on $z$.
2. $\log q_j = z_j/T - \log\sum_k e^{z_k/T}$, so $\partial \log q_j / \partial z_i = (\delta_{ij} - q_i)/T$.
3. Hence $\partial\, \mathrm{KL} / \partial z_i = -\sum_j p_j (\delta_{ij} - q_i)/T = (q_i - p_i)/T$, using $\sum_j p_j = 1$.
4. For large $T$, $e^{z_i/T} \approx 1 + z_i/T$. If the logits have zero mean over the $V$ classes, $q_i \approx (1 + z_i/T)/V$ and $p_i \approx (1 + v_i/T)/V$, so $\partial\, \mathrm{KL}/\partial z_i \approx (z_i - v_i)/(V T^2)$.

The soft-target gradients shrink like $1/T^2$, so multiplying by $T^2$ keeps their weight relative to the hard-label term roughly constant when $T$ is tuned. Step 4 also shows that at high temperature, distillation reduces to matching zero-mean logits with a squared error. A numerical check with four logits: the gradient norm of the KL term falls from 0.355 at $T = 1$ to 0.0013 at $T = 16$, while $T^2$ times it stays between 0.32 and 0.40.

**For LLMs.** Word-level distillation matches the teacher's next-token distribution at every position of some text. Sequence-level distillation (Kim and Rush, 2016) trains the student on the teacher's generated outputs; fine-tuning on outputs of a stronger model is this. Both train on prefixes the student would not produce itself (exposure bias). On-policy distillation lets the student generate and the teacher score each token: GKD (Agarwal et al., 2024) trains on student-generated sequences, and MiniLLM (Gu et al., 2024) minimizes the reverse KL $\mathrm{KL}(q \| p)$, which is mode-seeking, so a small student concentrates on outputs it can model well instead of spreading mass over everything the teacher might say.

### Batching and the paged KV cache

Since decoding is memory-bound, servers batch many requests. Static batching waits for the slowest sequence in a batch. **Continuous (iteration-level) batching**, introduced in the Orca system (Yu et al., 2022), schedules at every decoding step, adding new requests and retiring finished ones. The KV cache (size and the MQA/GQA reductions on [transformers](transformers.html)) then becomes the memory bottleneck, and its size per request is unknown in advance. Reserving the maximum length per request wastes memory through fragmentation. **PagedAttention** (vLLM; Kwon et al., 2023) stores each sequence's cache in fixed-size blocks located through a per-sequence block table, like virtual-memory pages, giving near-zero waste and letting sequences share blocks (parallel samples of one prompt, beam search, a common system prompt) with copy-on-write. vLLM reported 2–4$\times$ the throughput of FasterTransformer and Orca at the same latency. Prefix caching reuses the cache of shared prompt prefixes across requests.

::: {.qa asked="Qualcomm 2026 (Glassdoor), NVIDIA SWE (PracHub 2026)"}
**Quantize an FP32 tensor to int8: compute the scale and zero point, and explain which errors quantization introduces.**

Asymmetric: widen $[x_{\min}, x_{\max}]$ to contain 0, $s = (x_{\max} - x_{\min})/255$, $z = \mathrm{round}(-128 - x_{\min}/s)$, $q = \mathrm{clip}(\mathrm{round}(x/s) + z)$; zero is exact. Symmetric: $z = 0$, $s = \max|x|/127$; preferred for weights, since a nonzero weight zero point adds cross terms to the integer matmul, with one scale per output channel. Errors: rounding within $\pm s/2$ (variance $s^2/12$); clipping outside the calibrated range (calibration trades clipping against step size); outliers that inflate $s$ for everything sharing it; requantization after 32-bit accumulation with scale $s_x s_w / s_y$; and propagation through layers (bias correction, sensitive layers kept in higher precision). Quantization-aware training, with the straight-through estimator, loses less than post-training calibration at low bit widths. Code is in the [question bank](interview-question-bank.html).

:::

::: qa
**Why does 4-bit weight-only quantization speed up LLM decoding even though the matrix multiplies still run in 16 bits?**

At small batch sizes decoding is memory-bound: each step reads every weight once and does about 2 FLOPs per weight, so time is set by bytes moved, not FLOPs. Storing weights in 4 bits cuts the bytes about fourfold (13.5 GB to 3.4 GB for a 7B model, about 4 ms to 1 ms per token at 3.35 TB/s); dequantizing inside the kernel is cheap compute on otherwise idle units. At large batches or in prefill, where the work is compute-bound, the gain shrinks, and activation quantization (W8A8 or FP8) is what speeds up the arithmetic.
:::

::: qa
**Why is the soft-target term in distillation multiplied by $T^2$?**

Its gradient with respect to a student logit is $(q_i - p_i)/T$, and for large $T$ the softened probabilities differ by about $(z_i - v_i)/(VT)$, so the gradient is about $(z_i - v_i)/(VT^2)$. Without correction the soft term's influence would fall as $1/T^2$ whenever $T$ is raised; multiplying by $T^2$ keeps the balance with the hard-label loss stable while $T$ is tuned.
:::

## 7. Retrieval and tools

### Retrieval-augmented generation (RAG)

A model's knowledge is frozen at training time, cannot cite sources, and is unreliable for rare facts. RAG retrieves passages at query time and conditions the answer on them. Lewis et al. (2020) combined a dense retriever (DPR: a BERT query encoder searching Wikipedia passage embeddings) with a BART generator and trained the query encoder and generator end to end, treating the retrieved passage as a latent variable. Production systems usually keep the retriever and the LLM separate:

1. **Chunking.** Parse documents (PDF, HTML, tables), keep metadata (source, date, permissions), split along structure into chunks of a few hundred tokens with small overlap, and prepend the title and section path. Small chunks lose context; large ones blur the embedding and waste context.
2. **Embedding.** A bi-encoder maps queries and chunks separately into vectors, compared by cosine similarity. It is trained contrastively (InfoNCE with in-batch and hard negatives; DPR used BM25 results as hard negatives; derivation on [vision backbones](vision-backbones.html)). Separate encoding is what makes retrieval fast: chunks are embedded once, offline.
3. **Approximate nearest-neighbour (ANN) index.** Exact search costs $O(Md)$ per query over $M$ chunks. IVF indexes cluster the vectors with k-means and search only the few clusters nearest the query; product quantization (PQ) splits each vector into sub-vectors and stores each as the index of one of 256 centroids, so distances become table lookups; HNSW (Malkov and Yashunin) builds a layered proximity graph and searches it greedily from the top layer, with high recall at the cost of memory. FAISS (Johnson et al.) implements these. Example: 10M chunks of 768-dimensional float32 vectors take $10^7 \times 768 \times 4 = 30.7$ GB; PQ with 96 one-byte codes per vector takes 0.96 GB.
4. **Hybrid search.** Dense retrieval handles paraphrase; lexical BM25 handles exact names, codes and rare terms. Merge rankings with reciprocal rank fusion, $\mathrm{score}(d) = \sum_{\text{rankers}} 1/(k + \mathrm{rank}(d))$ with $k = 60$: a chunk ranked 1st and 3rd scores $1/61 + 1/63 = 0.03227$, one ranked 2nd by both $2/62 = 0.03226$.
5. **Reranking.** A cross-encoder reads the query and a chunk together and scores relevance; more accurate than the bi-encoder but too slow for the whole corpus, so it reorders the top 50 to 100 down to 5 to 10.
6. **Generation.** Instruct the model to answer from the context, cite chunk IDs and say when the answer is absent; put the best passages first. Liu et al. (2023, "Lost in the Middle") found models use information at the start or end of a long context better than in the middle.

**Failure modes.** The fact was never retrieved (split across chunks, vocabulary mismatch, missing metadata filter); it was retrieved but ignored or overridden by the model's own memory; sources conflict or are stale; permissions leak across users; retrieved documents carry instructions (prompt injection). Evaluate retrieval (recall@k, MRR, nDCG on labeled queries) and generation (faithfulness to the context, answer correctness) separately, since a bad answer can come from either.

**RAG or fine-tuning?** Retrieval for knowledge that is large, changing, private or must be cited; fine-tuning for behavior, format and style. Fine-tuning is a poor way to add new facts: they are learned slowly and increase hallucination (Gekhman et al., 2024).

### Function calling and agents

With **function calling**, tools are declared to the model as schemas (name, description, typed JSON arguments). The model emits a structured call, the runtime executes it, the result is appended to the context, and generation continues. Valid output can be enforced with constrained decoding, which masks tokens that would break the schema's grammar. Models learn tool use from fine-tuning on tool-use transcripts; Toolformer (Schick et al., 2023) generated such data itself, keeping sampled API calls only when their results lowered the loss on the following tokens. An **agent** runs this in a loop: act, observe, decide whether to continue. ReAct (Yao et al., 2023) interleaves written reasoning with actions. Errors compound: at 95% per-step reliability, a 20-step task with independent errors succeeds $0.95^{20} = 36\%$ of the time.

::: {.qa asked="Microsoft AS intern, PracHub 2026"}
**Design a retrieval-augmented generation (RAG) system.**

Follow the pipeline above: structure-aware chunking with metadata; dense embeddings in an ANN index plus BM25, fused by reciprocal rank; query rewriting for conversational and multi-hop questions; cross-encoder reranking of the top 50–100 to 5–10; a prompt that demands answers from the context with chunk citations and permits "not found"; optional claim-by-claim verification. Evaluate retrieval (recall@k, nDCG) and generation (faithfulness, correctness, with LLM judges checked against humans) separately, then online. For freshness, index incrementally, propagate deletions and version the index (a new embedding model means re-embedding everything). Watch for unretrieved facts, ignored context, stale or conflicting sources, permission leaks and prompt injection.

:::

::: {.qa asked="Microsoft AS intern, PracHub 2026"}
**What is an LLM agent, and how would you build and evaluate one?**

An LLM in a loop that chooses actions (tool calls, code, clicks), observes results and continues until done. Components: the policy (LLM plus a system prompt describing tools as schemas), interleaved reasoning and acting (ReAct), memory (the context window, summarized when full, plus an external store), and planning (implicit, plan-then-execute, or sub-agents), with checks and retries. Build with one loop and a few well-documented tools, keep deterministic steps in ordinary code, validate arguments, cap steps, time and cost, and log full trajectories. Failure modes: compounding errors ($0.95^{20} \approx 36\%$), wrong arguments or invented tools, loops, losing instructions in a long context, declaring success early, and prompt injection from tool outputs. Evaluate task success on held-out tasks with automatic end-state checks (tests that must pass, as in SWE-bench), several runs per task to measure reliability, cost and latency, and categorized failure trajectories. Sandbox tools, use least-privilege credentials, confirm irreversible actions, and treat tool outputs as data, not instructions.
:::

## 8. Evaluation and failure modes

### Benchmarks and contamination

Standard benchmarks include MMLU (multiple-choice questions in 57 subjects; Hendrycks et al., 2021), GSM8K (grade-school math word problems; Cobbe et al., 2021) and HumanEval (164 hand-written Python problems checked by unit tests; Chen et al., 2021). For code, **pass@k** is the probability that at least one of $k$ samples passes. Estimating it as $1 - (1 - c/n)^k$ from $c$ correct out of $n$ samples is biased (by Jensen's inequality), so Chen et al. use

$$
\text{pass@}k = 1 - \binom{n - c}{k} \Big/ \binom{n}{k},
$$

the probability that a random $k$-subset of the $n$ samples contains no correct one, subtracted from 1. With $n = 10$ and $c = 3$: pass@1 $= 0.3$ and pass@5 $= 1 - 21/252 = 0.917$. Open-ended chat is evaluated by pairwise human votes aggregated with a Bradley–Terry model (Chatbot Arena), the same model as the reward model in Section 5, or by an LLM judge.

**Contamination** means test items appeared in the training data, inflating scores. Lee et al. (2022) found train-test overlap affecting over 4% of the validation set of standard datasets. Detection and prevention: n-gram overlap checks between training data and test sets (GPT-3 used 13-grams), canary strings embedded in benchmark files so their presence in a corpus can be detected, comparing accuracy on problems published before and after the training cutoff, and benchmarks refreshed with new problems over time (LiveCodeBench). A large drop on a freshly written, matched version of a benchmark is a warning sign.

### Hallucination

A hallucination is fluent output that is false or unsupported by the given source; in summarization, intrinsic hallucinations contradict the source and extrinsic ones cannot be verified from it (Maynez et al., 2020). Causes:

- **The objective.** Next-token training rewards plausible continuations, and the model must output something. Kalai et al. (2025) argue that pretraining errors arise as errors in a binary "is this valid?" classification: for facts seen rarely in training, such as a birthday mentioned once, the model has little to separate the true value from plausible ones.
- **Evaluation and post-training reward guessing.** Benchmarks graded right or wrong give zero for "I don't know", so guessing scores better than abstaining (Kalai et al.).
- **Fine-tuning on unknown facts** teaches the model to answer confidently beyond its knowledge (Gekhman et al., 2024).
- **Decoding and snowballing.** Sampling from the tail introduces errors, and models tend to commit to an early mistake and then make further errors consistent with it (Zhang et al., 2023).
- **Retrieval failures**: missing, wrong or conflicting context.

Mitigations: retrieval with citations and checks that each claim is supported; rewarding abstention and penalizing confident errors in training and evaluation; tools for computation and lookup; lower temperature for factual tasks; and consistency checks, such as SelfCheckGPT (Manakul et al., 2023), which samples several answers and flags facts that vary across them.

### Calibration

A model is calibrated if, among answers given with confidence $c$, a fraction $c$ is correct; expected calibration error averages the gap between accuracy and confidence over confidence bins (the [question bank](interview-question-bank.html) covers ECE and temperature scaling). For LLMs: Kadavath et al. (2022) found large pretrained models well calibrated on multiple-choice and true/false questions presented in the right format, and able to estimate the probability that their own answer is true. The GPT-4 technical report found its pretrained model well calibrated on MMLU, and its calibration reduced after post-training. For free-form answers, token likelihood mixes up meaning with wording and length; semantic entropy (Kuhn et al., 2023) samples several answers, groups those with the same meaning, and measures uncertainty over the groups.

### LLM-as-a-judge

Using a strong LLM to grade outputs is cheap and scales. Zheng et al. (2023) found GPT-4's judgments agreed with human preferences over 80% of the time, the same level as agreement between humans, and documented the biases: **position** (favoring the first or second answer shown), **verbosity** (favoring longer answers), **self-enhancement** (favoring the judge's own outputs), and limited ability to grade math and reasoning. Mitigations: judge both orders and count only consistent wins, give a reference answer for math, ask for reasoning before the verdict, control for length (as in length-controlled AlpacaEval; Dubois et al., 2024), use several judges from different families, and validate against a set of human labels.

::: qa
**Why do LLMs hallucinate, and what would you do about it in a deployed assistant?**

Next-token training rewards plausible text, and for facts seen rarely the model cannot separate the true value from plausible alternatives; post-training and benchmarks that score abstention as wrong reward guessing; fine-tuning on facts the model does not know teaches confident answering; sampling and snowballing compound early errors. In deployment: ground answers with retrieval and require citations, verify claims against the retrieved text, let the model abstain and reward it for doing so, use tools for arithmetic and lookups, lower the temperature for factual queries, and monitor with sampled-consistency checks and human review of flagged cases.
:::

::: qa
**You use GPT-4 as a judge to compare two models. What can go wrong?**

Position bias (swap orders and count only consistent wins), verbosity bias (the longer model wins; control or regress on length), self-enhancement (a judge from the same family as one model favors it; use several judges), weak grading of math and code (give references or execute tests), and drift if the judge model changes. Validate the judge against a few hundred human labels on your own distribution and report agreement, and report uncertainty, since small win-rate differences are often within noise.
:::

## 9. Vision-language models

**CLIP** (Radford et al., 2021) trains an image encoder and a text encoder on about 400M image-caption pairs from the web. For a batch of $N$ pairs, let $u_i$ and $w_j$ be the L2-normalized image and text embeddings and $\tau$ a learned temperature. The loss is a symmetric InfoNCE, a cross-entropy over each row and each column of the $N \times N$ similarity matrix with the matching pair as the correct class:

$$
\mathcal{L} = \frac{1}{2N}\sum_{i=1}^{N}\Big[-\log\frac{e^{u_i^\top w_i/\tau}}{\sum_{j} e^{u_i^\top w_j/\tau}} - \log\frac{e^{u_i^\top w_i/\tau}}{\sum_{j} e^{u_j^\top w_i/\tau}}\Big].
$$

The temperature was initialized at 0.07 (a logit scale of $1/0.07 = 14.3$) and clipped so the scale never exceeds 100; the batch size was 32,768, so each image is classified against 32,767 negative captions. Zero-shot classification embeds prompts such as "a photo of a {class}" and picks the most similar; the best CLIP model reached 76.2% top-1 on ImageNet this way without using its training labels. Known weaknesses are counting, spatial relations and word order. CLIP encoders are reused as text encoders and image-feature extractors in image generation ([text-to-image](text-to-image-multimodal.html)); the InfoNCE derivation and its mutual-information bound are on [vision backbones](vision-backbones.html).

**LLaVA** (Liu et al., 2023) connects a frozen CLIP ViT-L/14 image encoder to an LLM (Vicuna) with a trainable projection (a single linear layer): each patch feature is mapped into the LLM's word-embedding space and the resulting visual tokens are inserted into the input sequence alongside the text tokens. Training has two stages: (1) feature alignment on 595K image-caption pairs filtered from CC3M, training only the projection; (2) end-to-end instruction tuning on 158K multimodal instruction examples generated by a text-only GPT-4 from captions and bounding boxes, updating the projection and the LLM with the vision encoder still frozen. The loss is the SFT loss on answer tokens only. LLaVA-1.5 (Liu et al., 2024) replaced the linear projection with a two-layer MLP, used a 336-pixel CLIP encoder ($(336/14)^2 = 576$ visual tokens per image), added academic VQA data, and trained in about a day on one 8-A100 node with 1.2M public examples. Alternatives compress the image to fewer tokens (BLIP-2's Q-Former with 32 learned queries) or inject it through gated cross-attention (Flamingo); see [text-to-image and multimodal](text-to-image-multimodal.html).

::: {.qa asked="Xiaomi, TME, Tencent, image-gen intern (4 reports)"}
**How is CLIP trained? Write its loss.**

Two encoders map images and captions to L2-normalized embeddings. For a batch of $N$ pairs, the logits are $s \cdot U W^\top$ with a learned scale $s = 1/\tau$ (initialized at $1/0.07$, capped at 100), and the loss averages the cross-entropy over rows (image to text) and columns (text to image) with the diagonal as targets: `logits = s * img @ txt.T; y = arange(N); loss = (ce(logits, y) + ce(logits.T, y)) / 2`. Trained on about 400M web pairs with batch 32,768; zero-shot classification compares an image with class prompts. It supplies text encoders and image embeddings to diffusion models and the CLIP score; it is weak at counting, spatial relations and word order.
:::

::: qa
**How does LLaVA turn an image into something an LLM can read, and what is trained in each stage?**

A frozen CLIP ViT encodes the image into patch features; a projection (linear in LLaVA, a two-layer MLP in LLaVA-1.5) maps each into the LLM's token-embedding space, giving a sequence of visual tokens (576 for a 336-pixel image with 14-pixel patches) inserted into the input alongside the text tokens. Stage 1 trains only the projection on image-caption pairs, so visual tokens land where the LLM can use them. Stage 2 trains the projection and the LLM on visual instruction data with the loss on answer tokens only, keeping the vision encoder frozen.
:::

## 10. From-scratch code

All three snippets were run with Python 3 and PyTorch 2.11; the printed results are shown in the text.

### A LoRA linear layer

```python
import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)              # W and bias stay frozen
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))    # random, like nn.Linear
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero: BA = 0 at start
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merge(self):                                        # one plain Linear for inference
        out = nn.Linear(self.base.in_features, self.base.out_features)
        out.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        out.bias.copy_(self.base.bias)
        return out

torch.manual_seed(0)
base = nn.Linear(64, 32)
lora = LoRALinear(base, r=4, alpha=8)
x = torch.randn(10, 64)
print(torch.equal(lora(x), base(x)))                       # True: starts at the base model

opt = torch.optim.SGD([lora.A, lora.B], lr=0.1)            # train only A and B
for _ in range(20):
    loss = (lora(x) - 1.0).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(lora.B.abs().max().item() > 0)                       # True: B has moved
print(torch.allclose(lora(x), lora.merge()(x), atol=1e-5)) # True: merged == unmerged
print(sum(p.numel() for p in [lora.A, lora.B]), base.weight.numel())  # 384 vs 2048
```

Output: `True`, `True`, `True`, `384 2048`. The first check confirms that with $B = 0$ the wrapped layer reproduces the base layer exactly; the second that $B$ received gradient even though it started at zero; the third that the merged single `nn.Linear` gives the same outputs as base plus adapter; the last that rank 4 trains $4 \times (64 + 32) = 384$ parameters against 2,048 frozen ones. To adapt a real model, replace its attention and MLP `nn.Linear` modules with this wrapper and pass only `A` and `B` to the optimizer.

### Top-p sampling

```python
import torch

def top_p_filter(probs, p):
    """Keep the smallest set of most-likely tokens whose mass reaches p; renormalize."""
    sorted_p, idx = probs.sort(descending=True)
    mass_before = sorted_p.cumsum(-1) - sorted_p          # mass of strictly better tokens
    sorted_p[mass_before >= p] = 0.0                      # drop once p is already reached
    out = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
    return out / out.sum(-1, keepdim=True)

def sample(logits, temperature=1.0, top_p=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(top_p_filter(probs, top_p), 1)

probs = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
print(top_p_filter(probs, 0.8))                   # [0.588, 0.235, 0.176, 0, 0]
print(top_p_filter(probs, 0.5))                   # [1, 0, 0, 0, 0]
torch.manual_seed(0)
draws = torch.stack([sample(probs.log(), top_p=0.8) for _ in range(20000)]).flatten()
print(torch.bincount(draws, minlength=5) / 20000)          # close to the filtered probs
```

Output: `[0.5882, 0.2353, 0.1765, 0, 0]`, `[1, 0, 0, 0, 0]`, and empirical frequencies `[0.5885, 0.2326, 0.1789, 0, 0]`. The test `mass_before >= p` keeps a token as long as the mass of the tokens ranked above it is still below $p$, which is exactly "the smallest set reaching $p$"; the top token is always kept. With $p = 0.5$ only the first token survives, because it alone reaches 0.5. Sorting costs $O(V \log V)$ per step.

### The DPO loss on toy logits

```python
import torch, torch.nn.functional as F

def seq_logprob(logits, tokens, mask):
    # logits: (B, T, V) predicting tokens (B, T); mask is 1 on response tokens only
    lp = torch.log_softmax(logits, -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)

def dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1):
    margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))   # implicit reward difference
    return -F.logsigmoid(margin).mean(), margin

torch.manual_seed(0)
B, T, V = 4, 6, 11
tok_w, tok_l = torch.randint(V, (B, T)), torch.randint(V, (B, T))
mask = torch.ones(B, T); mask[:, :2] = 0                  # first 2 tokens are the prompt
ref_logits_w, ref_logits_l = torch.randn(B, T, V), torch.randn(B, T, V)
pol_logits_w = ref_logits_w.clone().requires_grad_()       # policy starts at the reference
pol_logits_l = ref_logits_l.clone().requires_grad_()

with torch.no_grad():
    ref_w, ref_l = seq_logprob(ref_logits_w, tok_w, mask), seq_logprob(ref_logits_l, tok_l, mask)
opt = torch.optim.SGD([pol_logits_w, pol_logits_l], lr=1.0)
for step in range(51):
    loss, margin = dpo_loss(seq_logprob(pol_logits_w, tok_w, mask),
                            seq_logprob(pol_logits_l, tok_l, mask), ref_w, ref_l)
    if step % 25 == 0:
        print(step, round(loss.item(), 4), round(margin.mean().item(), 3))
    opt.zero_grad(); loss.backward(); opt.step()
```

Output: `0 0.6931 0.0`, `25 0.5894 0.22`, `50 0.5075 0.414`. At step 0 the policy equals the reference, so every margin is zero and the loss is $\log 2$. Gradient steps raise the chosen sequences' log-probability relative to the reference and lower the rejected ones', so the mean margin grows and the loss falls. In a real model the logits come from two forward passes (chosen and rejected) of the policy, and the reference log-probabilities are computed once with `torch.no_grad()` or precomputed.

::: {.qa asked="Insta360 multimodal (Nowcoder snippet)"}
**Implement a LoRA linear layer.**

Use the class above: freeze the base layer, add $A \in \mathbb{R}^{r \times k}$ (random) and $B \in \mathbb{R}^{d \times r}$ (zero), and return `base(x) + (alpha / r) * x @ A.T @ B.T`. Be ready to explain why $B$ starts at zero (the model starts exactly at the pretrained one, and $B$ still gets gradient), why there is no nonlinearity (so the update can be merged into $W_0$), how to test it (equal to the base at initialization, merged equals unmerged), and how to wrap an existing model's projections. Only $A$ and $B$ go to the optimizer.
:::

## Rapid-fire review

::: qa
**What should the initial loss of a language model with a 50,257-token vocabulary be?**

About $\ln 50{,}257 = 10.8$ nats, from near-uniform predictions. Much higher means badly scaled output logits; much lower at step 0 suggests a bug such as label leakage through the attention mask.
:::

::: qa
**What is perplexity, and when is it not comparable?**

The exponential of the mean per-token negative log-likelihood, an effective branching factor. It is not comparable across tokenizers (use bits per byte), context lengths or test sets.
:::

::: qa
**What goes wrong if you pack documents without document masking?**

Tokens attend to unrelated earlier documents in the same row, which adds distracting context during training; intra-document masking (a block-diagonal causal mask) removes it and was found to help.
:::

::: qa
**Compute-optimal tokens for a 7B model under Chinchilla, and what a deployed 7B model actually gets?**

About $20 \times 7\text{B} = 140\text{B}$ tokens. Deployed models are trained far longer (LLaMA 7B on 1T, Llama 3 8B on over 15T) because inference, about $2N = 14$ GFLOPs per token for 7B, dominates lifetime cost.
:::

::: qa
**What is temperature 0, and does it make output deterministic?**

The limit $T \to 0$ is greedy decoding. In principle deterministic, but batched GPU inference can change floating-point summation order, so near-ties can flip between runs.
:::

::: qa
**What makes speculative decoding fast, and what limits it?**

Verifying $\gamma$ draft tokens in one target pass costs about as much as generating one token, because decoding is memory-bound. The gain depends on the acceptance rate $1 - \mathrm{TV}(p, q)$ and the draft's cost, and shrinks at large batch sizes.
:::

::: qa
**Why mask prompt tokens in SFT?**

The prompt is given at inference, so its likelihood is irrelevant; training on it teaches the model to write user turns, and long prompts would dominate the gradient.
:::

::: qa
**Why is the KL penalty needed in RLHF?**

The reward model is accurate only near the data it was trained on; without the penalty the policy drifts into regions where the reward model is wrong (reward hacking), collapses onto a few outputs, and loses fluency. $\beta$ sets how far it may move.
:::

::: qa
**What does $\beta$ do in DPO?**

It is the KL strength of the objective DPO solves: the implicit reward is $\beta\log(\pi_\theta/\pi_{\text{ref}})$, so a smaller $\beta$ allows larger departures from the reference for the same preference margin.
:::

::: qa
**Why does GRPO not need a value model?**

It uses the mean reward of $G$ samples of the same prompt as the baseline and divides by their standard deviation, so the advantage comes from the group rather than from a learned per-token value.
:::

::: qa
**What does best-of-16 cost and how far does it move from the base policy?**

Sixteen generations plus sixteen reward-model calls per prompt, and a KL of $\log 16 - 15/16 = 1.84$ nats by the formula Gao et al. use.
:::

::: qa
**Why does LoRA have zero inference overhead?**

The update $\frac{\alpha}{r}BA$ is linear and has the shape of $W_0$, so it can be added into the weights once after training.
:::

::: qa
**Weight-only or weight-and-activation quantization: which speeds up what?**

Weight-only (for example 4-bit weights, 16-bit activations) cuts memory traffic and speeds up memory-bound decoding at small batch. Weight-and-activation (W8A8, FP8) uses low-precision arithmetic and speeds up compute-bound prefill and large batches, but must handle activation outliers.
:::

::: qa
**What is the pass@k estimator for code generation?**

Draw $n \ge k$ samples, count $c$ correct, and use $1 - \binom{n-c}{k}/\binom{n}{k}$; with $n = 10$, $c = 3$, pass@5 is 0.917.
:::

::: qa
**In one sentence, the critique of emergent abilities.**

Many sharp jumps come from discontinuous metrics like exact match ($p^{L}$ rises abruptly while per-token accuracy $p$ improves smoothly), and tend to disappear under continuous metrics.
:::

::: qa
**In-context learning or fine-tuning?**

In-context learning needs no training and adapts instantly, but costs context length on every call and is sensitive to the examples' choice and order. Fine-tuning costs training and data but makes the behavior cheap and consistent at inference.
:::

## References

1. Radford, Narasimhan, Salimans, Sutskever. Improving Language Understanding by Generative Pre-Training. 2018. [OpenAI](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
2. Radford, Wu, Child, Luan, Amodei, Sutskever. Language Models are Unsupervised Multitask Learners. 2019. [OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. Brown et al. Language Models are Few-Shot Learners. 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
4. Devlin, Chang, Lee, Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
5. Raffel et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. 2020. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
6. Sennrich, Haddow, Birch. Neural Machine Translation of Rare Words with Subword Units. 2016. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
7. Kudo, Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. 2018. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
8. Kaplan et al. Scaling Laws for Neural Language Models. 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
9. Hoffmann et al. Training Compute-Optimal Large Language Models. 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
10. Besiroglu, Erdil, Barnett, You. Chinchilla Scaling: A replication attempt. 2024. [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)
11. Pearce, Song. Reconciling Kaplan and Chinchilla Scaling Laws. 2024. [arXiv:2406.12907](https://arxiv.org/abs/2406.12907)
12. Porian, Wortsman, Jitsev, Schmidt, Carmon. Resolving Discrepancies in Compute-Optimal Scaling of Language Models. 2024. [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)
13. Sardana, Portes, Doubov, Frankle. Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws. 2023. [arXiv:2401.00448](https://arxiv.org/abs/2401.00448)
14. Touvron et al. LLaMA: Open and Efficient Foundation Language Models. 2023. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
15. Grattafiori et al. (Llama Team, Meta). The Llama 3 Herd of Models. 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783); Meta, Introducing Meta Llama 3, 2024. [blog](https://ai.meta.com/blog/meta-llama-3/)
16. Muennighoff et al. Scaling Data-Constrained Language Models. 2023. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)
17. Lee et al. Deduplicating Training Data Makes Language Models Better. 2022. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
18. Wenzek et al. CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data. 2020. [arXiv:1911.00359](https://arxiv.org/abs/1911.00359)
19. Penedo et al. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. 2024. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
20. Zhao et al. Analysing The Impact of Sequence Composition on Language Model Pre-Training. 2024. [arXiv:2402.13991](https://arxiv.org/abs/2402.13991)
21. Wei et al. Emergent Abilities of Large Language Models. 2022. [arXiv:2206.07682](https://arxiv.org/abs/2206.07682)
22. Schaeffer, Miranda, Koyejo. Are Emergent Abilities of Large Language Models a Mirage? 2023. [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)
23. Holtzman, Buys, Du, Forbes, Choi. The Curious Case of Neural Text Degeneration. 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
24. Keskar, McCann, Varshney, Xiong, Socher. CTRL: A Conditional Transformer Language Model for Controllable Generation. 2019. [arXiv:1909.05858](https://arxiv.org/abs/1909.05858)
25. Leviathan, Kalman, Matias. Fast Inference from Transformers via Speculative Decoding. 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
26. Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper. Accelerating Large Language Model Decoding with Speculative Sampling. 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
27. Min et al. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? 2022. [arXiv:2202.12837](https://arxiv.org/abs/2202.12837)
28. Zhao, Wallace, Feng, Klein, Singh. Calibrate Before Use: Improving Few-Shot Performance of Language Models. 2021. [arXiv:2102.09690](https://arxiv.org/abs/2102.09690)
29. Xie, Raghunathan, Liang, Ma. An Explanation of In-context Learning as Implicit Bayesian Inference. 2021. [arXiv:2111.02080](https://arxiv.org/abs/2111.02080); Olsson et al. In-context Learning and Induction Heads. 2022. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895); von Oswald et al. Transformers learn in-context by gradient descent. 2023. [arXiv:2212.07677](https://arxiv.org/abs/2212.07677)
30. Wei et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
31. Kojima, Gu, Reid, Matsuo, Iwasawa. Large Language Models are Zero-Shot Reasoners. 2022. [arXiv:2205.11916](https://arxiv.org/abs/2205.11916)
32. Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
33. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
34. Aghajanyan, Zettlemoyer, Gupta. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. 2020. [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)
35. Kalajdzievski. A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. 2023. [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)
36. Hayou, Ghosh, Yu. The Impact of Initialization on LoRA Finetuning Dynamics. 2024. [arXiv:2406.08447](https://arxiv.org/abs/2406.08447)
37. Biderman et al. LoRA Learns Less and Forgets Less. 2024. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
38. Dettmers, Pagnoni, Holtzman, Zettlemoyer. QLoRA: Efficient Finetuning of Quantized LLMs. 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
39. Houlsby et al. Parameter-Efficient Transfer Learning for NLP. 2019. [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
40. Li, Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation. 2021. [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
41. Lester, Al-Rfou, Constant. The Power of Scale for Parameter-Efficient Prompt Tuning. 2021. [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)
42. Ouyang et al. Training language models to follow instructions with human feedback. 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
43. Christiano et al. Deep reinforcement learning from human preferences. 2017. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
44. Ziegler et al. Fine-Tuning Language Models from Human Preferences. 2019. [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)
45. Stiennon et al. Learning to summarize from human feedback. 2020. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
46. Schulman, Wolski, Dhariwal, Radford, Klimov. Proximal Policy Optimization Algorithms. 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
47. Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
48. Azar et al. A General Theoretical Paradigm to Understand Learning from Human Preferences. 2023. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
49. Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (introduces GRPO). 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
50. DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
51. Liu et al. Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO). 2025. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
52. Yu et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. 2025. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
53. Zhou et al. LIMA: Less Is More for Alignment. 2023. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
54. Gekhman et al. Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? 2024. [arXiv:2405.05904](https://arxiv.org/abs/2405.05904)
55. Gao, Schulman, Hilton. Scaling Laws for Reward Model Overoptimization. 2022. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)
56. Singhal, Goyal, Xu, Durrett. A Long Way to Go: Investigating Length Correlations in RLHF. 2023. [arXiv:2310.03716](https://arxiv.org/abs/2310.03716)
57. Sharma et al. Towards Understanding Sycophancy in Language Models. 2023. [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)
58. Coste, Anwar, Kirk, Krueger. Reward Model Ensembles Help Mitigate Overoptimization. 2023. [arXiv:2310.02743](https://arxiv.org/abs/2310.02743)
59. Bai et al. Constitutional AI: Harmlessness from AI Feedback. 2022. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
60. Lee et al. RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. 2023. [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)
61. Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
62. Kim, Rush. Sequence-Level Knowledge Distillation. 2016. [arXiv:1606.07947](https://arxiv.org/abs/1606.07947)
63. Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD). 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
64. Gu, Dong, Wei, Huang. MiniLLM: On-Policy Distillation of Large Language Models (first titled "Knowledge Distillation of Large Language Models"). 2024. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
65. Dettmers, Lewis, Belkada, Zettlemoyer. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
66. Xiao et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
67. Frantar, Ashkboos, Hoefler, Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
68. Lin et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
69. Yu, Jeong, Kim, Kim, Chun. Orca: A Distributed Serving System for Transformer-Based Generative Models. OSDI 2022. [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)
70. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
71. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
72. Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
73. Malkov, Yashunin. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. 2016. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
74. Johnson, Douze, Jégou. Billion-scale similarity search with GPUs (FAISS). 2017. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
75. Cormack, Clarke, Büttcher. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR 2009.
76. Liu et al. Lost in the Middle: How Language Models Use Long Contexts. 2023. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
77. Schick et al. Toolformer: Language Models Can Teach Themselves to Use Tools. 2023. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
78. Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models. 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
79. Hendrycks et al. Measuring Massive Multitask Language Understanding. 2021. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)
80. Cobbe et al. Training Verifiers to Solve Math Word Problems (GSM8K). 2021. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
81. Chen et al. Evaluating Large Language Models Trained on Code (HumanEval, pass@k). 2021. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
82. Jain et al. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. 2024. [arXiv:2403.07974](https://arxiv.org/abs/2403.07974)
83. Maynez, Narayan, Bohnet, McDonald. On Faithfulness and Factuality in Abstractive Summarization. 2020. [arXiv:2005.00661](https://arxiv.org/abs/2005.00661)
84. Kalai, Nachum, Vempala, Zhang. Why Language Models Hallucinate. 2025. [arXiv:2509.04664](https://arxiv.org/abs/2509.04664)
85. Zhang, Press, Merrill, Liu, Smith. How Language Model Hallucinations Can Snowball. 2023. [arXiv:2305.13534](https://arxiv.org/abs/2305.13534)
86. Manakul, Liusie, Gales. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. 2023. [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
87. Kadavath et al. Language Models (Mostly) Know What They Know. 2022. [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)
88. OpenAI. GPT-4 Technical Report. 2023. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)
89. Kuhn, Gal, Farquhar. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. 2023. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
90. Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
91. Dubois, Galambosi, Liang, Hashimoto. Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. 2024. [arXiv:2404.04475](https://arxiv.org/abs/2404.04475)
92. Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
93. Liu, Li, Wu, Lee. Visual Instruction Tuning (LLaVA). 2023. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
94. Liu, Li, Li, Lee. Improved Baselines with Visual Instruction Tuning (LLaVA-1.5). 2024. [arXiv:2310.03744](https://arxiv.org/abs/2310.03744)
95. Liu et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. 2019. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
96. Rae et al. Scaling Language Models: Methods, Analysis & Insights from Training Gopher. 2021. [arXiv:2112.11446](https://arxiv.org/abs/2112.11446)
97. Wei et al. Finetuned Language Models Are Zero-Shot Learners (FLAN). 2021. [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)
98. Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? 2023. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
99. Li, Li, Savarese, Hoi. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
100. Alayrac et al. Flamingo: a Visual Language Model for Few-Shot Learning. 2022. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
