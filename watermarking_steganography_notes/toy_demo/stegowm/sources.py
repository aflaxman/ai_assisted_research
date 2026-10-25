"""Distribution sources that plug into the watermark and stego channels.

A source only has to answer one question: given the tokens chosen so far, what is
the next-token distribution? The synthetic source ignores the context and instead
replays a pre-drawn sequence of distributions with a controllable entropy level,
which lets us put entropy on the x-axis of every experiment. The optional GPT-2
source returns a real (top-k truncated) language-model distribution, so we can
hide a message inside genuine English.
"""

from __future__ import annotations

import numpy as np


class SyntheticSource:
    """Replays a fixed sequence of categorical distributions.

    The distributions are drawn from a Dirichlet whose concentration sets the
    entropy: small ``alpha`` -> peaky, low-entropy steps (like a confident LM);
    large ``alpha`` -> flat, high-entropy steps. Because the sequence is fixed and
    context-independent, encoder and decoder always see identical distributions.
    """

    def __init__(self, vocab: int, n_steps: int, alpha: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dists = rng.dirichlet(np.full(vocab, alpha), size=n_steps)
        self.vocab = vocab

    def dist(self, context: list[int]) -> np.ndarray:
        t = len(context)
        if t >= len(self.dists):
            # Loop if a caller asks for more steps than we pre-drew.
            t = t % len(self.dists)
        return self.dists[t]


class MixedEntropySource:
    """Alternates confident and uncertain steps, mimicking real LM text.

    Roughly ``low_frac`` of positions are near-deterministic (one dominant token)
    and the rest are high-entropy. This reproduces the "low-entropy tendency" that
    the steganography literature calls out as the main obstacle to capacity.
    """

    def __init__(self, vocab: int, n_steps: int, low_frac: float = 0.6, seed: int = 0):
        rng = np.random.default_rng(seed)
        dists = []
        for _ in range(n_steps):
            if rng.random() < low_frac:
                p = rng.dirichlet(np.full(vocab, 0.03))  # peaky
            else:
                p = rng.dirichlet(np.full(vocab, 1.5))   # broad
            dists.append(p)
        self.dists = np.array(dists)
        self.vocab = vocab

    def dist(self, context: list[int]) -> np.ndarray:
        t = len(context) % len(self.dists)
        return self.dists[t]


class GPT2Source:
    """Real GPT-2 next-token distribution, truncated to top-k and renormalized.

    Lazily imports torch/transformers so the rest of the demo runs without them.
    Deterministic given context, so it works for both encoding and decoding.
    """

    def __init__(self, model_name: str = "distilgpt2", top_k: int = 64,
                 temperature: float = 1.0, prompt: str = "\n"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.top_k = top_k
        self.temperature = temperature
        self.prompt_ids = self.tok.encode(prompt)
        self.vocab = self.model.config.vocab_size

    def dist(self, context: list[int]) -> np.ndarray:
        torch = self._torch
        ids = self.prompt_ids + list(context)
        with torch.no_grad():
            logits = self.model(torch.tensor([ids])).logits[0, -1]
        logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1).numpy()
        # Top-k truncation keeps the fixed-point cumulative small and stable.
        keep = np.argpartition(probs, -self.top_k)[-self.top_k:]
        trunc = np.zeros_like(probs)
        trunc[keep] = probs[keep]
        trunc /= trunc.sum()
        return trunc

    def decode_text(self, tokens: list[int]) -> str:
        return self.tok.decode(tokens)
