"""Build a source-only transfer bundle; never include credentials or model weights."""

import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root / "dist" / "stochastic-llm-source.tar.gz"
destination.parent.mkdir(exist_ok=True)
included = [root / name for name in ("pyproject.toml", "README.md", ".gitignore")]
for name in ("src", "configs", "slurm", "scripts", "tests"):
    included.extend(path for path in (root / name).rglob("*") if path.is_file())
with tarfile.open(destination, "w:gz") as bundle:
    for path in sorted(included):
        if "__pycache__" not in path.parts and not any(p.endswith(".egg-info") for p in path.parts):
            bundle.add(path, arcname=str(Path("stochastic-llm") / path.relative_to(root)))
print(destination)
