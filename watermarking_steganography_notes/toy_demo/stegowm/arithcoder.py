"""Incremental integer arithmetic coder (Witten-Neal-Cleary style).

This is the engine that makes the steganographic channel both efficient and
stall-free. A per-token "commit the common prefix" scheme throws away the
low-order bits of every token; arithmetic coding instead carries that
undetermined state forward in a persistent range register, so the bits are
reclaimed by later tokens. That carry-forward is exactly Meteor's "randomness
reuse", and it is why the payload rate reaches the entropy of the text.

Two directions, mirror images of each other:

- ``decode`` turns a bit stream into tokens using per-step distributions. Fed a
  (pseudo)random stream, its tokens are distributed like the model -- this is the
  sender generating stegotext.
- ``encode`` turns tokens back into the bit stream -- this is the receiver
  recovering the payload.

Distributions are supplied per step as integer cumulative frequencies summing to
``TOTAL`` (see ``stegowm.sampler.integer_cumulative`` with ``precision=FREQ_BITS``).
"""

from __future__ import annotations

CODE_BITS = 32
TOP = 1 << CODE_BITS
MASK = TOP - 1
HALF = TOP >> 1
QUARTER = TOP >> 2
THREE_QUARTER = 3 * QUARTER

FREQ_BITS = 16
TOTAL = 1 << FREQ_BITS


class BitReader:
    def __init__(self, bits):
        self.bits = bits
        self.pos = 0
        self.consumed = 0  # counts real bits read (past the end reads as 0)

    def read(self) -> int:
        if self.pos < len(self.bits):
            b = int(self.bits[self.pos])
            self.consumed += 1
        else:
            b = 0
        self.pos += 1
        return b


def decode(model, cipher, n_tokens=None, min_consumed=None, max_tokens=200_000):
    """Bits -> tokens. Stop after ``n_tokens`` or once ``min_consumed`` real bits
    have been consumed by renormalization (whichever is given).

    Returns ``(tokens, consumed_bits)``.
    """
    from .sampler import integer_cumulative

    reader = BitReader(cipher)
    low, high = 0, MASK
    code = 0
    for _ in range(CODE_BITS):
        code = (code << 1) | reader.read()

    tokens: list[int] = []
    context: list[int] = []
    while True:
        if n_tokens is not None and len(tokens) >= n_tokens:
            break
        if min_consumed is not None and reader.consumed >= min_consumed:
            break
        if len(tokens) >= max_tokens:
            break

        C = integer_cumulative(model.dist(context), FREQ_BITS)
        rng = high - low + 1
        scaled = ((code - low + 1) * TOTAL - 1) // rng
        # Find symbol whose cumulative interval contains ``scaled``.
        import numpy as np
        i = int(np.searchsorted(C, scaled, side="right") - 1)
        cl, ch = int(C[i]), int(C[i + 1])

        high = low + (rng * ch) // TOTAL - 1
        low = low + (rng * cl) // TOTAL
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                code -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                low -= QUARTER
                high -= QUARTER
                code -= QUARTER
            else:
                break
            low = (low << 1) & MASK
            high = ((high << 1) | 1) & MASK
            code = ((code << 1) | reader.read()) & MASK

        tokens.append(i)
        context.append(i)
    return tokens, reader.consumed


def encode(model, tokens):
    """Tokens -> bits (the canonical minimal encoding).

    Returns ``(bits, bits_per_token)`` where ``bits_per_token[t]`` is how many
    output bits had been emitted after token ``t`` minus after ``t-1``.
    """
    from .sampler import integer_cumulative

    low, high = 0, MASK
    pending = 0
    out: list[int] = []
    per_token: list[int] = []

    def emit(bit: int):
        out.append(bit)
        for _ in range(pending_ref[0]):
            out.append(1 - bit)
        pending_ref[0] = 0

    pending_ref = [0]
    context: list[int] = []
    for i in tokens:
        C = integer_cumulative(model.dist(context), FREQ_BITS)
        cl, ch = int(C[i]), int(C[i + 1])
        rng = high - low + 1
        high = low + (rng * ch) // TOTAL - 1
        low = low + (rng * cl) // TOTAL
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                pending_ref[0] += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low = (low << 1) & MASK
            high = ((high << 1) | 1) & MASK
        context.append(i)
        per_token.append(len(out))

    # Convert cumulative lengths to per-token deltas.
    deltas = [per_token[0]] + [per_token[t] - per_token[t - 1]
                               for t in range(1, len(per_token))] if per_token else []
    return out, deltas
