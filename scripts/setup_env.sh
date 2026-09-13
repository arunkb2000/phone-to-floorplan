#!/usr/bin/env bash
# Clean-machine setup: Python 3.12 venv via uv + locked deps. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv python install 3.12 >/dev/null 2>&1 || true
uv sync --frozen 2>/dev/null || uv sync
echo "env ready: $(uv run python -c 'import torch,sys;print(sys.version.split()[0], "torch", torch.__version__, "mps", torch.backends.mps.is_available())')"
