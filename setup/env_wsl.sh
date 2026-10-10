#!/usr/bin/env bash
# creates the holes env inside wsl2 ubuntu, safe to re-run after a failure
set -euo pipefail

if ! command -v conda >/dev/null 2>&1 && [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  echo "[env] installing Miniconda to ~/miniconda3"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  "$HOME/miniconda3/bin/conda" init bash
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

if ! conda env list | grep -q '^holes '; then
  conda create -n holes python=3.12 -y
fi
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
conda activate holes
conda install -c conda-forge openjdk=21.0.10 maven -y
pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu==1.15.1 pyserini==2.4.0 ir_measures==0.4.3 datasets==5.1.0 pandas==3.0.6 numpy==2.5.3 \
  scipy==1.18.1 matplotlib==3.11.2 pyyaml==6.0.3 tqdm==4.70.1 pytest==9.1.1 sentence-transformers==6.1.0 \
  transformers==5.19.0 huggingface_hub==1.33.0 requests==2.34.2

mkdir -p "$HOME/.cache/pyserini/indexes" "$HOME/.cache/huggingface"
grep -q PYSERINI_CACHE "$HOME/.bashrc" || echo 'export PYSERINI_CACHE=$HOME/.cache/pyserini' >> "$HOME/.bashrc"
grep -q HF_HOME "$HOME/.bashrc" || echo 'export HF_HOME=$HOME/.cache/huggingface' >> "$HOME/.bashrc"
echo "[env] done. Activate with: conda activate holes"
