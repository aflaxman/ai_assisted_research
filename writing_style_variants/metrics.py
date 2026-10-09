"""Deterministic text metrics that approximate ASD-STE100 style rules.

These are heuristics. They count surface features (sentence length, semicolons,
contractions, Latin abbreviations, a passive-voice pattern, hedges), not meaning.
Code blocks and inline code are removed first, because the PR text says code
keeps its own conventions.
"""
from __future__ import annotations

import re

FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-_.]*")

CONTRACTION_RE = re.compile(
    r"\b(?:\w+n[’']t|\w+[’']re|\w+[’']ve|\w+[’']ll|\w+[’']d|I[’']m|it[’']s|that[’']s|"
    r"there[’']s|here[’']s|what[’']s|let[’']s|who[’']s|he[’']s|she[’']s|where[’']s|how[’']s)\b",
    re.I,
)
LATIN_RE = re.compile(r"\b(?:e\.g\.|i\.e\.|etc\.?|vs\.?|cf\.|et al\.?|viz\.|N\.B\.|ca\.|via)(?=[\s,;:)]|$)", re.I)
HEDGE_RE = re.compile(
    r"\b(?:maybe|might|may|probably|perhaps|possibly|seems?|appears?|likely|unlikely|"
    r"I think|I believe|I suspect|I guess|I assume|presumably|arguably|could be|my guess|"
    r"judgment|judgement|guess)\b",
    re.I,
)
PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|get|gets|got)\s+(?:\w+ly\s+)?"
    r"(?:\w+ed|\w+en|built|made|done|given|taken|kept|sent|set|put|run|read|seen|shown|"
    r"known|found|held|left|lost|met|paid|said|sold|told|thought|written|broken|chosen|"
    r"hidden|thrown|caught|bought|brought|taught|understood|meant|dealt|felt|hit|cut|"
    r"split|spent|lent|bound|struck)\b",
    re.I,
)
MERMAID_RE = re.compile(r"```\s*mermaid", re.I)
BOX_CHARS = set("│┌┐└┘├┤┬┴┼─╭╮╰╯═║╔╗╚╝▶►")
ARROW_RE = re.compile(r"-{2,}>|={2,}>|──+>|<-{2,}")


def strip_code(text: str) -> str:
    text = FENCE_RE.sub(" ", text)
    return INLINE_CODE_RE.sub("ID", text)


def sentences(text: str) -> list[list[str]]:
    """Split prose into sentences and return the words of each one.

    Headings and table rows are skipped. A line break ends a sentence.
    """
    out: list[list[str]] = []
    for line in strip_code(text).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("|") or set(s) <= set("-*_= "):
            continue
        s = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", s)  # bullet or number marker
        s = re.sub(r"\*\*|__|\*", "", s)  # bold and italics markers
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(`])", s):
            words = WORD_RE.findall(piece)
            if words:
                out.append(words)
    return out


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def text_metrics(text: str) -> dict:
    prose = strip_code(text)
    sents = sentences(text)
    n_sent = len(sents)
    lens = [len(s) for s in sents]
    n_words = sum(lens)
    n_syll = sum(syllables(w) for s in sents for w in s)
    fk_grade = (
        0.39 * (n_words / n_sent) + 11.8 * (n_syll / n_words) - 15.59 if n_sent and n_words else float("nan")
    )
    return {
        "words": n_words,
        "sentences": n_sent,
        "mean_sentence_words": (n_words / n_sent) if n_sent else float("nan"),
        "max_sentence_words": max(lens) if lens else 0,
        "pct_sentences_over_20": (100 * sum(1 for n in lens if n > 20) / n_sent) if n_sent else float("nan"),
        "fk_grade": fk_grade,
        "semicolons": prose.count(";"),
        "contractions": len(CONTRACTION_RE.findall(prose)),
        "latin_abbreviations": len(LATIN_RE.findall(prose)),
        "passive_hits": len(PASSIVE_RE.findall(prose)),
        "passive_per_100_sentences": (100 * len(PASSIVE_RE.findall(prose)) / n_sent) if n_sent else float("nan"),
        "hedges": len(HEDGE_RE.findall(prose)),
        "inline_code": len(INLINE_CODE_RE.findall(FENCE_RE.sub(" ", text))),
        "code_blocks": len(FENCE_RE.findall(text)),
        "bullets": len(re.findall(r"^\s*(?:[-*+]|\d+[.)])\s+", text, re.M)),
        "headers": len(re.findall(r"^\s*#{1,6}\s+", text, re.M)),
        "has_diagram": bool(MERMAID_RE.search(text) or sum(c in BOX_CHARS for c in text) >= 3 or len(ARROW_RE.findall(text)) >= 3),
        "wrapped_in_code_fence": text.strip().startswith("```"),
        "chars": len(text),
    }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(text_metrics(sys.stdin.read()), indent=2))
