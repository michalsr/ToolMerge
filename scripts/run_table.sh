#!/usr/bin/env bash
# Reproduce one paper table row locally.
#
# Usage:
#   ./scripts/run_table.sh table2_lvb_qwen3_8
#   ./scripts/run_table.sh table3_m2m_qa_qwen3_8 max_final_k=8 data.start_idx=0 data.end_idx=10
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
  echo "Usage: $0 <table-config-name> [extra OmegaConf overrides ...]" >&2
  exit 1
fi

NAME="$1"
shift
CFG="$ROOT/configs/tables/${NAME}.yaml"

if [ ! -f "$CFG" ]; then
  echo "Unknown table config: ${NAME} (no such file at ${CFG})" >&2
  echo "Available:" >&2
  ls "$ROOT/configs/tables/" >&2
  exit 1
fi

cd "$ROOT"
exec python -m toolmerge.run run "config=${CFG}" "$@"
