"""Keyed pseudorandom functions built on HMAC-SHA256.

The whole watermarking/steganography family in this demo rests on one idea:
replace the true randomness a language model would use to sample tokens with
*pseudorandom* bits derived from a secret key. To anyone without the key the
bits are indistinguishable from random, so the model's output distribution is
untouched. To anyone with the key the bits are perfectly reproducible, which is
what makes a hidden signal recoverable.

This module provides the two things the rest of the code needs:

- ``keystream(key, nbits)``: a reproducible pseudorandom bit string, used as the
  one-time pad that hides the steganographic payload.
- ``uniforms(key, context, n)``: reproducible Uniform(0, 1) draws keyed on the
  generation context, used by the Gumbel watermark.
"""

from __future__ import annotations

import hashlib
import hmac
import struct

import numpy as np


def _prg_bytes(key: bytes, nonce: bytes, nbytes: int) -> bytes:
    """Expand (key, nonce) into ``nbytes`` pseudorandom bytes via HMAC in CTR mode."""
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        block = hmac.new(key, nonce + struct.pack(">Q", counter), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:nbytes])


def keystream(key: bytes, nbits: int, nonce: bytes = b"stego") -> np.ndarray:
    """Return ``nbits`` reproducible pseudorandom bits as a 0/1 uint8 array."""
    nbytes = (nbits + 7) // 8
    raw = np.frombuffer(_prg_bytes(key, nonce, nbytes), dtype=np.uint8)
    bits = np.unpackbits(raw)[:nbits]
    return bits.astype(np.uint8)


def uniforms(key: bytes, context: bytes, n: int) -> np.ndarray:
    """Return ``n`` reproducible Uniform(0, 1) draws keyed on (key, context).

    Used by the Gumbel watermark to score each candidate token at one position.
    Two positions with different context get independent-looking draws; the same
    context always reproduces the same draws, which is what the detector relies
    on.
    """
    # 8 bytes -> one 53-bit-precision float in [0, 1).
    raw = _prg_bytes(key, b"wm|" + context, 8 * n)
    ints = np.frombuffer(raw, dtype=">u8").astype(np.uint64)
    # Top 53 bits give a uniform double, avoiding the exact endpoints.
    return (ints >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a 0/1 array into bytes (right-padded with zeros)."""
    return np.packbits(bits.astype(np.uint8)).tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Unpack bytes into a 0/1 uint8 array."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)
