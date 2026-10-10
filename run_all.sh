#!/usr/bin/env bash
# runs every step in order. each step resumes where it stopped, so the script can be started again after a cut-off.
set -euo pipefail
export PYTHONPATH=.
export PYSERINI_CACHE="${PYSERINI_CACHE:-$HOME/.cache/pyserini}"

if ! curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" > /dev/null; then
  echo "no ollama server found, start it and pull qwen2.5:7b-instruct and gemma3:4b first" >&2
  exit 1
fi

step() {
  echo "[run] python $*"
  python "$@"
}

step src/00_env_check.py
step src/01_data_audit.py
step src/02_runs_miracl.py
step src/03_runs_e5.py
step src/04_coverage.py
step src/05_bounds.py
step src/06_llm_validate.py split
step src/06_llm_validate.py calibrate
step src/06_llm_validate.py test
step src/07_llm_fill.py
step src/08_ciral.py --search
step src/09_stats.py
step src/10_figures.py
