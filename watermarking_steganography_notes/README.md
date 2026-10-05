# Watermarking LLM Text Meets Provably Secure Steganography

Research notes exploring the question: has anyone combined the idea of
watermarking LLM-generated text with the cryptographic approach to
steganography that Luis von Ahn worked on around 2002?

**Short answer: yes, and quite literally.** The modern cryptographic school of
LLM watermarking descends directly from the Hopper–Langford–von Ahn line of
provably secure steganography, and by 2024 the two ideas had fused into single
papers.

## The 2002 side: provably secure steganography

- **Hopper, Langford, von Ahn, "Provably Secure Steganography" (CRYPTO 2002;
  journal version IEEE Trans. Computers 2009).** First rigorous,
  complexity-theoretic definition of steganographic security: stegotext must be
  *computationally indistinguishable* from the covertext distribution to anyone
  without the key. Their construction assumes a sampler for the covertext
  channel — a big assumption in 2002, when nobody could sample realistic
  English text. https://link.springer.com/chapter/10.1007/3-540-45708-9_6
- **von Ahn & Hopper, "Public-Key Steganography" (EUROCRYPT 2004).** Extends
  the framework so sender and receiver need not share a secret key.

The framework sat mostly theoretical for ~15 years because the "channel
sampler" it presupposed didn't exist. Then language models became exactly that
sampler.

## The bridge: LLMs as the covertext sampler

- **Ziegler, Deng, Rush, "Neural Linguistic Steganography" (EMNLP 2019).**
  Arithmetic coding over GPT-2's next-token distribution — practical but not
  provably secure.
- **Kaptchuk, Jois, Green, Rubin, "Meteor" (ACM CCS 2021).**
  https://eprint.iacr.org/2021/686 — the first construction to make the
  Hopper–Langford–von Ahn program *practical* by using GPT-2 as the channel
  sampler. Provably indistinguishable from honest model output; encodes ~160
  message bytes per 300–350 words of GPT-2 text.
- **Schroeder de Witt et al., "Perfectly Secure Steganography Using Minimum
  Entropy Coupling" (ICLR 2023).** https://arxiv.org/abs/2210.14889 —
  information-theoretic (not just computational) security via minimum entropy
  coupling, again with generative models as the channel.

## The watermarking side, and the fusion

- **Kirchenbauer et al., "A Watermark for Large Language Models" (ICML 2023).**
  The well-known green-list/red-list scheme — statistical, *not* cryptographic,
  and it perturbs the output distribution.
- **Christ, Gunn, Zamir, "Undetectable Watermarks for Language Models"
  (COLT 2024).** https://proceedings.mlr.press/v247/christ24a.html — imports
  the 2002 steganographic security definition into watermarking: the watermark
  is undetectable (output distribution provably unchanged) without the secret
  key, under standard cryptographic assumptions (PRFs). The paper explicitly
  frames undetectability as the steganography definition ported to
  watermarking. This is the cleanest "von Ahn 2002 meets LLM watermarking"
  citation trail.
- **Zamir, "Excuse me, sir? Your language model is leaking (information)"
  (arXiv:2401.10360, 2024; OpenReview version titled "Undetectable
  Steganography for Language Models").** https://arxiv.org/abs/2401.10360 —
  the full fusion: takes the Christ–Gunn–Zamir *watermark* and upgrades it to
  *steganography*, embedding an arbitrary secret payload in LLM output such
  that, without the key, distinguishing payload-carrying responses from clean
  ones is provably infeasible. Uses error-correcting codes with feedback to
  handle the low-entropy stretches of LLM text. Also flags the unsettling
  dual-use reading: a model could covertly exfiltrate information through
  seemingly clean responses.
- **Fairoze et al., "Publicly-Detectable Watermarking for Language Models"
  (IACR Communications in Cryptology, 2024).** https://cic.iacr.org/p/1/4/31 —
  embeds publicly verifiable cryptographic *signatures* into LLM text using
  steganography-style rejection sampling; a public-key flavor reminiscent of
  von Ahn & Hopper 2004.
- **Cohen et al., "Watermarking Language Models for Many Adaptive Users"
  (2024).** https://eprint.iacr.org/2024/759 — multi-user, zero-bit and
  multi-bit undetectable watermarks in the same cryptographic tradition.

## Active directions (2024–2026)

- **Robustness vs. undetectability tension.** Provably secure stego (Meteor and
  descendants) is fragile: paraphrasing or even small edits destroy the
  payload. "Robust Steganography from Large Language Models"
  (https://arxiv.org/abs/2504.08977) and Alkaid
  (https://arxiv.org/abs/2603.06169) attack the edit-robustness problem;
  watermarking papers attack it from the other side with error-correcting
  codes.
- **Capacity in low-entropy text.** LLMs often generate near-deterministic
  continuations, starving stego schemes of embedding room — e.g., provably
  secure linguistic steganography via range coding
  (https://arxiv.org/abs/2604.08052) and list decoding
  (https://arxiv.org/abs/2604.21394).
- **AI-safety flip side.** The same machinery enables covert collusion between
  LLM agents: "Secret Collusion among AI Agents"
  (https://arxiv.org/abs/2402.07510) and "Hidden in Plain Text"
  (https://arxiv.org/abs/2410.03768) study steganographic collusion as a risk.

## The pleasing symmetry

Watermarking and steganography are duals sharing one mechanism (keyed
pseudorandom biasing of token sampling):

| | Hidden message | Adversary's goal |
|---|---|---|
| **Steganography** (von Ahn 2002) | arbitrary payload | *detect* that a message exists |
| **Watermarking** (Christ–Gunn–Zamir 2024) | one bit ("model-generated") | *remove* the mark / evade detection |

Von Ahn's 2002 framework needed an efficient sampler of realistic covertext,
which didn't exist then. LLMs are that sampler. So the modern cryptographic
watermarking literature isn't merely *analogous* to the 2002 work — it is the
2002 work, finally instantiated, with the roles of message and channel
reshuffled.

## Worked example

[`toy_demo/`](toy_demo/) carries out the first project idea below: a tested
implementation of a distortion-free watermark and a Meteor-style steganographic
channel that share one PRF, with the headline measurement confirmed — **payload
capacity equals text entropy** (97–102%), and watermark detection power grows
with length and entropy. It runs on real distilgpt2: a 31-byte secret hides in
plausible English and recovers exactly.

## Fun follow-up project ideas

1. Reimplement a toy Christ–Gunn–Zamir watermark on a small open model and then
   apply Zamir's ECC-with-feedback trick to turn it into a payload channel;
   measure bits/token vs. text entropy. *(Done — see [`toy_demo/`](toy_demo/).)*
2. Measure how many bits of a Meteor-style payload survive an
   LLM-paraphrase attack, versus a CGZ-style watermark's one bit.
3. Historical note: von Ahn's other famous line (CAPTCHA, also ~2002-2003) was
   "prove a human generated this"; watermarking is "prove a machine generated
   this." Two duals from the same person's early work, twenty years apart.
