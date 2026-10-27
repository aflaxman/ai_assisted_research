"""Writing-rule variants under test.

Each variant is a short text appended to the Claude Code system prompt
(`claude -p --append-system-prompt`). The two PR texts are copied verbatim
from ihmeuw/vivarium-suite PR #304 (branch pnast/ai-tools-public/mic-7606/writing-style).
"""

# PR #304: the one-line rule added to each agent definition (verbatim).
PR_AGENT_LINE = (
    "Write prose for people in plain technical English. Use short sentences in the "
    "active voice and simple tenses, and one name for each thing. Do not use "
    "semicolons, Latin abbreviations, or metaphors. Put code identifiers and paths "
    "in backticks."
)

# PR #304: tools/ai-tools/hooks/writing-style.txt, the SessionStart hook text (verbatim, 427 chars).
PR_HOOK_TEXT = (
    "Prose for people in this project is plain technical English. Sentences are short, "
    "in the active voice, and in simple tenses. Each thing has one name. The text has no "
    "semicolons, contractions, Latin abbreviations, or metaphors. Code identifiers and "
    "paths are in backticks. A guess or a judgment is marked as one, and a reason that "
    "the source does not give is not stated as fact. Code and quoted text keep their own "
    "conventions."
)

# PR #304, eval round 5: the same hook text with the standard named. The PR reports that
# this version scored lower on clarity, so the shipped text does not name ASD-STE100.
PR_HOOK_NAMED = PR_HOOK_TEXT.replace(
    "plain technical English.",
    "plain technical English, based on ASD-STE100 Simplified Technical English.",
    1,
)

# Karpathy (x.com/karpathy, 2 Oct): "Ask your LLM to explain something in ASD-STE100".
STE_NAMED = "Write in ASD-STE100 (Simplified Technical English)."

# Karpathy: "ask for '80% of the way to ASD-STE100' because the spec is quite stringent".
STE_80PCT = "Write 80% of the way to ASD-STE100 (Simplified Technical English)."

# Karpathy: "Instead of writing, ask your LLM to create a diagram."
DIAGRAM = (
    "Prefer a diagram over prose. When the content has a structure, a sequence, or "
    "dependencies, show it as a Mermaid or plain-text diagram, and keep the prose "
    "around it short."
)

VARIANTS = {
    "none": {
        "label": "V0 no rules (control)",
        "source": "control",
        "text": None,
    },
    "pr_agent_line": {
        "label": "V1 PR one-line agent rule",
        "source": "PR #304 (shipped, agents)",
        "text": PR_AGENT_LINE,
    },
    "pr_hook_text": {
        "label": "V2 PR hook text (427 chars)",
        "source": "PR #304 (shipped, SessionStart hook)",
        "text": PR_HOOK_TEXT,
    },
    "pr_hook_named": {
        "label": "V3 PR hook text + names ASD-STE100",
        "source": "PR #304 eval round 5 (rejected)",
        "text": PR_HOOK_NAMED,
    },
    "ste_named": {
        "label": "V4 'Write in ASD-STE100'",
        "source": "Karpathy thread",
        "text": STE_NAMED,
    },
    "ste_80pct": {
        "label": "V5 '80% of the way to ASD-STE100'",
        "source": "Karpathy thread",
        "text": STE_80PCT,
    },
    "diagram": {
        "label": "V6 prefer a diagram",
        "source": "Karpathy thread",
        "text": DIAGRAM,
    },
}

ORDER = list(VARIANTS)
