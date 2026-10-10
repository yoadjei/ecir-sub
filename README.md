# judgement holes in miracl and ciral

code to measure how much incomplete relevance judgements affect cross-language comparisons on miracl and ciral: judged@10 per language and system, ndcg@10 under lower, condensed and optimistic treatments, which gaps and rankings are identified under both bounds, hole filling with a small local llm judge, and ciral's shallow versus deep judgements.

## requirements

- linux or wsl2, conda, no gpu
- about 50 GB of free disk. dense indexes are 5 to 11 GB each and can be deleted once a run is finished
- network for the first run, pyserini downloads the prebuilt indexes and qrels
- a local ollama server for the judge, with the two candidate models pulled: `ollama pull qwen2.5:7b-instruct` and `ollama pull gemma3:4b`. set `OLLAMA_HOST` if it is not on localhost:11434

## setup

```bash
bash setup/env_wsl.sh
conda activate holes
```

tested on ubuntu 24.04 under wsl2 with 8 cpu cores and 12 GB of ram, python 3.12.15, jdk 21.0.10 and ollama 0.40.0. package versions are pinned in the setup script.

## run order

`bash run_all.sh` runs every step below in order and can be started again after a cut-off. to run one step, call it from the repo root as `PYTHONPATH=. python src/<script>`. most take `--langs` and `--systems`, see `--help`.

```bash
python src/00_env_check.py
python src/01_data_audit.py
python src/02_runs_miracl.py
python src/03_runs_e5.py
python src/04_coverage.py
python src/05_bounds.py
python src/06_llm_validate.py split
python src/06_llm_validate.py calibrate
python src/06_llm_validate.py test
python src/07_llm_fill.py
python src/08_ciral.py --search
python src/09_stats.py
python src/10_figures.py
```

| script | writes |
|---|---|
| 00 | results/ENV.txt |
| 01 | results/data_audit.csv |
| 02 | runs/miracl.*.txt, results/repro_check.csv |
| 03 | runs/miracl.*.e5small.txt |
| 04 | results/coverage*.csv, coverage_model.txt |
| 05 | results/bounds*.csv, gaps.csv, pairs.csv |
| 06 | results/llm_splits/, llm_calibration.csv, llm_model_choice.txt, llm_validation.csv |
| 07 | qrels/miracl.*.filled.txt, results/filled*.csv, gaps_filled.csv, pairs_filled.csv |
| 08 | results/ciral*.csv, qrels/ciral.*, runs/ciral.* |
| 09 | results/table*.csv, mde.csv, summary_stats.txt, numbers.md, tables/*.tex |
| 10 | figures/ |

## resuming

dense search (02), the e5 runs (03) and the judge (06, 07) can be stopped at any time. dense search keeps a `.partial` file and carries on after the last complete query, the e5 embeddings are saved in chunks, and every judged passage is cached in `results/llm_labels.jsonl`. dense search needs no network once the index and the model are in the local cache.

## the judge

qwen2.5:7b-instruct through ollama, temperature 0, seed 13, with the prompt in `src/prompts/judge.txt`. `results/llm_labels.jsonl` holds every label it produced, keyed by model, prompt hash, query id and passage id. with it in place, steps 06 and 07 reuse the labels and need no model calls. the fill samples 50 queries per language, `--sample-queries` changes that and a larger sample reuses the labels of a smaller one.

## tests

```bash
PYTHONPATH=. python -m pytest tests -q
```

the tests build their own small data and need the conda environment from the setup step.
