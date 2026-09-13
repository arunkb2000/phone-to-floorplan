#!/usr/bin/env bash
# Fetch pinned model weights into ./weights (Hugging Face, revision-pinned). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_TELEMETRY=1
W="$PWD/weights"; mkdir -p "$W"
dl() { uv run hf download "$1" --revision "$2" --local-dir "$W/$3" >/dev/null && echo "ok $1@$2 -> weights/$3"; }
# Metric monocular depth (indoor). Small for speed, Large for the final numbers.
dl depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf main da2_metric_indoor_small
dl depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf main da2_metric_indoor_large
# Open-vocabulary detector for damage regions and door/window verification.
dl google/owlv2-base-patch16-ensemble main owlv2_base
(cd "$W" && find . -type f \( -name '*.safetensors' -o -name '*.bin' \) -exec shasum -a 256 {} \; | sort -k2 > SHA256SUMS && echo "wrote weights/SHA256SUMS")
