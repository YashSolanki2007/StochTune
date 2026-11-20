"""High-recall candidate search followed by an explicit, versioned topic review."""

import re

# This search is deliberately permissive. Inclusion requires topic-review.json;
# a keyword match alone NEVER makes a reference eligible for this experiment.
CANDIDATE = re.compile(
    r"\b(python|javascript|typescript|java|programming|debug|sql|html|css|regex|"
    r"coding|code|algorithm|recursion|compiler|linked list|binary tree|data structure|"
    r"pandas|numpy|pytorch|tensorflow|django|flask|github|gitlab|rust|golang|nodejs|"
    r"node\.js|react|flutter|swift|kotlin|laravel|php|pinescript|ruby|perl|bash|"
    r"powershell|docker|kubernetes|terraform|cmake|pypi|npm|microservices|fastapi|"
    r"webapp|web app|software development|software engineer|dataframe|matlab|octave)\b|c\+\+|c#",
    re.IGNORECASE,
)


def opening_key(prompt):
    """Conservatively keep continuations of the same opening in one group."""
    opening = prompt.split("<extra_id_1>")[0]
    return " ".join(re.findall(r"[a-z0-9]+", opening.casefold()))


def prompt_groups(rows):
    """Group same and near-identical openings; no responses enter this decision."""
    rows = sorted(rows, key=lambda row: row["id"])
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys = [opening_key(row["prompt"]) for row in rows]
    tokens = [set(key.split()) for key in keys]
    for i in range(len(rows)):
        for j in range(i):
            same = keys[i] == keys[j]
            near = min(len(tokens[i]), len(tokens[j])) >= 10 and (
                len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j]) >= 0.92
            )
            if same or near:
                parent[root(i)] = root(j)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    return list(groups.values())
