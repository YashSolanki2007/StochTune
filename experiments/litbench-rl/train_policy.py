"""Run the existing tested quality-plus-word-KL trainer with LitBench data/reward."""

from pathlib import Path

_here = Path(__file__).resolve().parent
_impl = _here / "_policy_impl.py"
if not _impl.exists():
    _impl = _here.parent / "helpsteer-rl-v2/train_policy.py"
_source = _impl.read_text().replace(
    'ROOT / "data/reward_validation.jsonl"', 'ROOT / "data/reward_audit.jsonl"'
)
exec(compile(_source, str(_impl), "exec"), globals())
