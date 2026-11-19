"""Freeze broader prompt splits for the longer run; no model calls or GPU work."""

import itertools
import json
import random

from pilot import ROOT, write_json

ROLES = {
    "train": [
        "gardener",
        "merchant",
        "soldier",
        "musician",
        "parent",
        "messenger",
        "innkeeper",
        "weaver",
        "servant",
        "watchmaker",
        "traveller",
        "painter",
        "shepherd",
        "cook",
        "miller",
        "courtier",
    ],
    "validation": ["astronomer", "stonemason", "bookseller", "ferryman"],
    "test": ["glassmaker", "mapmaker", "bellringer", "locksmith"],
}
SETTINGS = {
    "train": [
        "an orchard at dusk",
        "a market after closing",
        "a courtyard before dawn",
        "a roadside shelter",
        "a room above a tavern",
        "a deserted workshop",
        "a village square",
        "the steps of a household chapel",
    ],
    "validation": ["a windswept observatory", "a bridge during a thaw"],
    "test": ["a lantern-lit harbour", "an empty theatre after the audience has left"],
}
DILEMMAS = {
    "train": [
        "You have lost money entrusted to you. Decide to confess rather than blame someone else.",
        "You discover an unfair payment in your favour. Decide whether to return it.",
        "A friend asks you to repeat a rumour you cannot verify. Choose what to tell them.",
        "Someone who once mocked you now needs help. Decide whether to help them.",
        "You must leave home for work but fear disappointing a loved one. Resolve what to say.",
        "An old promise conflicts with a new opportunity. "
        "Make a clear choice and explain its cost.",
        "You are praised for a task that another person helped complete. Decide how to respond.",
        "You find a damaged keepsake of someone you miss. "
        "Decide what to preserve and what to let go.",
    ],
    "validation": [
        "You discover that a trusted teacher concealed a mistake. "
        "Decide how to seek an explanation.",
        "A stranger offers to repay a debt you had forgotten. Decide how to receive the offer.",
        "You must admit that a cherished plan is no longer possible. Choose a concrete next step.",
        "An estranged relative sends an invitation. "
        "Decide whether to accept while keeping a boundary.",
    ],
    "test": [
        "You learn that your proudest achievement caused an unintended injury. "
        "Decide how to make amends.",
        "You can expose a deception only by admitting an embarrassing truth about yourself. "
        "Make a choice.",
        "An unexpected act of kindness undermines a prejudice you have long held. "
        "Decide how to respond.",
        "A community asks you to accept an honour you believe belongs to someone absent. "
        "Decide what to say.",
    ],
}


def prepare():
    counts = {"train": 256, "validation": 16, "test": 16}
    rng = random.Random(20261119)
    result = {}
    for split, count in counts.items():
        combinations = list(itertools.product(ROLES[split], SETTINGS[split], DILEMMAS[split]))
        selected = rng.sample(combinations, count)
        result[split] = [
            {"id": f"{split}-{i:04}", "prompt": f"Speak as a {role} in {setting}. {dilemma}"}
            for i, (role, setting, dilemma) in enumerate(selected)
        ]
    flattened = [r["prompt"] for rows in result.values() for r in rows]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Prompt splits overlap")
    write_json(ROOT / "ddp-prompts.json", result)
    print(json.dumps({split: len(rows) for split, rows in result.items()}))


if __name__ == "__main__":
    prepare()
