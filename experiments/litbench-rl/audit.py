"""Independent frozen rubric judge, adapted to fiction rather than factual answers."""

from pathlib import Path

_here = Path(__file__).resolve().parent
_impl = _here / "_audit_impl.py"
if not _impl.exists():
    _impl = _here.parent / "helpsteer-rl-v2/audit.py"
_source = _impl.read_text().replace("Evaluate an assistant answer.", "Evaluate a fictional story.")
exec(compile(_source, str(_impl), "exec"), globals())
RUBRICS = globals()["RUBRICS"]
RUBRICS["coherence"] = (
    "Internally consistent characters, events and narrative; clear development; "
    "no excessive repetition."
)
RUBRICS["readability"] = (
    "Fluent, readable prose appropriate to the fictional genre; no reward for length alone."
)
RUBRICS["prompt_fulfillment"] = (
    "Develops the requested fictional premise and constraints. "
    "Fiction need not be true in the real world."
)
