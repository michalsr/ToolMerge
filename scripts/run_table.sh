#!/usr/bin/env bash
# Reproduce one paper row locally.
#
# Usage:
#   ./scripts/run_table.sh lvb/qwen3_8
#   ./scripts/run_table.sh m2m/qwen3_8 max_final_k=8 data.start_idx=0 data.end_idx=10
#
# Loads .env if present, validates the named config exists, and runs the CLI.

set -euo pipefail

ROOT="$(cd "$(dirname "$(realpath "$0")")"/.. && pwd)"

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset>/<answerer>_<K> [extra OmegaConf overrides ...]" >&2
  exit 1
fi

NAME="$1"
shift
CFG="$ROOT/configs/${NAME}.yaml"

if [ ! -f "$CFG" ]; then
  echo "Unknown config: ${NAME} (no such file at ${CFG})" >&2
  echo "Available:" >&2
  ( cd "$ROOT" && find configs -mindepth 2 -name '*.yaml' | sort ) >&2
  exit 1
fi

cd "$ROOT"
exec python -m toolmerge.run run "config=${CFG}" "$@"
