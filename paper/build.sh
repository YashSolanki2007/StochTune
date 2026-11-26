#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
if command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --keep-intermediates --outdir build main.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
else
  printf '%s\n' 'Install Tectonic or a LaTeX distribution with latexmk.' >&2
  exit 1
fi
cp build/main.pdf controlled-stochasticity.pdf
