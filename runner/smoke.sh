#!/usr/bin/env bash
# Dev/CPU smoke test on the box. Results are NOT publishable (subset, CPU, no verification).
set -euo pipefail
cd "$(dirname "$0")"
M=${1:-qwen2.5:0.5b}; N=${2:-4}; T=${3:-1}
name=${M%%:*}; tag=${M##*:}
sha=${GGUF_SHA256:-}  # sha256 of the GGUF blob (Ollama model layer digest)
pb=$(echo "$tag" | tr -dc '0-9.' ); ver=$(ollama --version 2>/dev/null | awk '{print $NF}')
python3 -m lab --model "$M" --family qwen2.5 --params-b "${pb:-0}" --quant Q4_K_M --gguf-sha256 "$sha" \
  --engine ollama --engine-version "$ver" --tier-gb 0 --channel dev --per-category "$N" --trials "$T" --out ../results/dev
