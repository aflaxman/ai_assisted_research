"""Correctness gates for the sampler, the stego channel, and the watermark."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stegowm import stego, watermark  # noqa: E402
from stegowm.sampler import entropy_bits, integer_cumulative  # noqa: E402
from stegowm.sources import SyntheticSource  # noqa: E402


def test_integer_cumulative_exact_and_positive():
    rng = np.random.default_rng(1)
    P = 16
    for _ in range(300):
        v = rng.integers(2, 40)
        p = rng.dirichlet(np.full(v, rng.uniform(0.05, 3.0)))
        C = integer_cumulative(p, P)
        assert C[0] == 0 and C[-1] == (1 << P)
        widths = np.diff(C)
        assert np.all(widths[p > 0] >= 1)
        assert widths.sum() == (1 << P)


def test_stego_roundtrip_exact():
    """Encode then decode must return the exact payload, over many settings."""
    rng = np.random.default_rng(2)
    for trial in range(30):
        vocab = int(rng.integers(3, 40))
        alpha = float(rng.uniform(0.3, 3.0))
        src = SyntheticSource(vocab, n_steps=4000, alpha=alpha, seed=trial)
        n = int(rng.integers(1, 20))
        payload = bytes(rng.integers(0, 256, size=n).tolist())
        key = bytes(rng.integers(0, 256, size=16).tolist())
        tokens = stego.encode_message(src, key, payload)
        out = stego.decode_message(src, key, tokens)
        assert out == payload, f"trial {trial}: {out!r} != {payload!r}"


def test_stego_roundtrip_constant_distribution():
    """A constant (context-independent) distribution must not stall."""
    p = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])

    class Fixed:
        def dist(self, context):
            return p

    payload = b"steganography"
    key = b"fixed-key-000001"
    tokens = stego.encode_message(Fixed(), key, payload)
    assert stego.decode_message(Fixed(), key, tokens) == payload


def test_stego_is_distortion_free():
    """With a random key the stego token histogram must match the model's."""
    vocab = 6
    p = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])

    class Fixed:
        def dist(self, context):
            return p

    counts = np.zeros(vocab)
    trials = 4000
    rng = np.random.default_rng(7)
    for _ in range(trials):
        key = bytes(rng.integers(0, 256, size=16).tolist())
        payload = bytes(rng.integers(0, 256, size=6).tolist())
        tokens = stego.encode_message(Fixed(), key, payload)
        counts[tokens[0]] += 1  # first token of each independent run
    emp = counts / counts.sum()
    tv = 0.5 * np.abs(emp - p).sum()
    assert tv < 0.05, f"total variation {tv:.3f} too high (not distortion-free)"


def test_capacity_tracks_entropy():
    """Mean committed bits per token should sit within ~0.5 bits of entropy."""
    for alpha in (0.5, 1.0, 3.0):
        src = SyntheticSource(64, n_steps=4000, alpha=alpha, seed=5)
        H = np.mean([entropy_bits(src.dist(list(range(t)))) for t in range(400)])
        payload = bytes(range(120))  # long enough to average out
        from stegowm.prf import bytes_to_bits, keystream
        header = np.zeros(stego.HEADER_BITS, dtype=np.uint8)
        plain = np.concatenate([header, bytes_to_bits(payload)])
        cipher = np.bitwise_xor(plain, keystream(b"k" * 16, len(plain)))
        res = stego.embed_bits(src, cipher)
        bits_per_token = float(np.mean(res.committed_per_token))
        assert abs(bits_per_token - H) < 0.5, (alpha, bits_per_token, H)


def test_watermark_detects_and_is_distortion_free():
    src = SyntheticSource(50, n_steps=4000, alpha=1.0, seed=3)
    key = b"watermark-key-01"
    toks = watermark.generate(src, key, n_tokens=400)
    res = watermark.detect(src, key, toks)
    assert res["p_value"] < 1e-6, res

    rng = np.random.default_rng(9)
    p_values = []
    for _ in range(20):
        random_toks = [int(rng.integers(0, 50)) for _ in range(400)]
        p_values.append(watermark.detect(src, key, random_toks)["p_value"])
    assert np.mean(np.array(p_values) < 0.01) < 0.15, np.mean(p_values)


if __name__ == "__main__":
    test_integer_cumulative_exact_and_positive()
    print("cumulative ok")
    test_stego_roundtrip_exact()
    print("roundtrip ok")
    test_stego_roundtrip_constant_distribution()
    print("constant-dist ok")
    test_capacity_tracks_entropy()
    print("capacity ok")
    test_watermark_detects_and_is_distortion_free()
    print("watermark ok")
    test_stego_is_distortion_free()
    print("distortion-free ok")
    print("ALL PASSED")
