"""Hide a secret message inside real English, then recover it.

Generates text from distilgpt2 whose token distribution is untouched -- to any
reader without the key it is an ordinary (if rambly) GPT-2 sample -- yet it
carries a hidden payload that the key holder extracts exactly. Also shows the
distortion-free watermark on the same model.

Run: python experiments/demo_gpt2.py
Requires torch + transformers (see requirements-gpt2.txt).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stegowm import stego, watermark  # noqa: E402
from stegowm.sources import GPT2Source  # noqa: E402


def main():
    key = b"correct horse battery staple!!"[:16]
    secret = b"Meet at the old mill, midnight."

    print("Loading distilgpt2 ...")
    src = GPT2Source(model_name="distilgpt2", top_k=64, temperature=1.0,
                     prompt="The weather report for today is")

    print(f"\nSecret payload ({len(secret)} bytes): {secret!r}\n")

    tokens = stego.encode_message(src, key, secret)
    text = src.decode_text(tokens)
    print("=" * 70)
    print("STEGOTEXT (looks like an ordinary GPT-2 sample):")
    print("=" * 70)
    print(text)
    print("=" * 70)
    print(f"\n{len(tokens)} tokens carry {len(secret)*8} payload bits "
          f"({len(secret)*8/len(tokens):.2f} bits/token).")

    recovered = stego.decode_message(src, key, tokens)
    print(f"\nRecovered with key : {recovered!r}")
    print(f"Exact match        : {recovered == secret}")

    wrong = stego.decode_message(src, b"the-wrong-key-XX", tokens)
    print(f"Recovered w/ wrong key (garbage): {wrong!r}")

    print("\n" + "=" * 70)
    print("WATERMARK on the same model")
    print("=" * 70)
    wm_tokens = watermark.generate(src, key, n_tokens=120)
    print(src.decode_text(wm_tokens))
    hit = watermark.detect(src, key, wm_tokens)
    print(f"\nDetect (own key)   : z={hit['z']:.1f}  p={hit['p_value']:.2e}")
    miss = watermark.detect(src, b"some-other-keyXX", wm_tokens)
    print(f"Detect (other key) : z={miss['z']:.1f}  p={miss['p_value']:.2f}")


if __name__ == "__main__":
    main()
