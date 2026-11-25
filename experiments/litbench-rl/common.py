"""Reuse the tested utilities with this experiment's own ROOT and data."""

from pathlib import Path

_here = Path(__file__).resolve().parent
_impl = _here / "_common_impl.py"
if not _impl.exists():
    _impl = _here.parent / "helpsteer-rl-v2/common.py"
exec(compile(_impl.read_text(), str(_impl), "exec"), globals())
# Local CPU preparation uses shared imports; staged runs copy them alongside us.
import sys  # noqa: E402

sys.path.append(str(_here.parent / "helpsteer-rl-v2"))
