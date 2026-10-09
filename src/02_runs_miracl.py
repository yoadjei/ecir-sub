"""make trec runs and check them against the published two-click reproduction scores.

usage: PYTHONPATH=. python src/02_runs_miracl.py [--langs yo sw] [--systems bm25 mdpr]
bm25 runs use the pyserini cli, dense runs use src/02b_resume_faiss.py, which keeps a
.partial file and resumes after a cut-off. skips a run file that is already finished
(a .partial file is never finished). writes results/repro_check.csv.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import ir_measures
import pandas as pd
from ir_measures import nDCG

from src.utils.config import cfg, langs, path
from src.utils.io import pyserini_qrels, read_qrels, read_run_scores, write_qrels

TOLERANCE = 0.01
# yo has no lucene analyzer, 2cr runs it on pretokenized topics instead
PRETOKENIZED = {"yo"}


def run_path(lang: str, system: str):
    return path("runs") / f"miracl.{lang}.{system}.txt"


def cli_command(lang: str, system: str) -> list[str]:
    s = cfg()["systems"][system]
    topics = f"miracl-v1.0-{lang}-dev"
    out = str(run_path(lang, system))
    hits = str(cfg()["hits"])
    if s["kind"] == "lucene":
        analyzer = ["--pretokenized"] if lang in PRETOKENIZED else ["--language", lang]
        return [sys.executable, "-m", "pyserini.search.lucene", *analyzer,
                "--topics", topics, "--index", f"miracl-v1.0-{lang}",
                "--output", out, "--bm25", "--hits", hits, "--batch-size", "64", "--threads", "8"]
    if s["kind"] == "faiss":
        return [sys.executable, str(Path(__file__).with_name("02b_resume_faiss.py")),
                "--lang", lang, "--system", system]
    raise ValueError(f"system {system} is not produced by this script")


def ensure_qrels(lang: str):
    q = path("qrels") / f"miracl.{lang}.dev.txt"
    if not q.exists():
        write_qrels(pyserini_qrels(f"miracl-v1.0-{lang}-dev"), q)
    return q


def evaluate(lang: str, system: str) -> float:
    qrels = read_qrels(ensure_qrels(lang))
    run = read_run_scores(run_path(lang, system))
    agg = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)
    return float(agg[nDCG @ 10])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--systems", nargs="*", default=["bm25", "mdpr", "mcontriever"])
    args = ap.parse_args()
    target_langs = args.langs or langs()
    ref = pd.read_csv(path("results") / "reference_2cr.csv", comment="#")
    rows = []
    for lang in target_langs:
        info = cfg()["languages"][lang]
        for system in args.systems:
            kind = cfg()["systems"][system]["kind"]
            if kind == "faiss" and not info["dense"]:
                continue
            if kind == "lucene" and not info["bm25"]:
                continue
            out = run_path(lang, system)
            if out.exists():
                print(f"[runs] exists, skipping: {out.name}")
            else:
                cmd = cli_command(lang, system)
                print("[runs] " + " ".join(cmd))
                subprocess.run(cmd, check=True)
            got = evaluate(lang, system)
            r = ref[(ref.lang == lang) & (ref.system == system)]
            expected = float(r.ndcg10_ref.iloc[0]) if len(r) and pd.notna(r.ndcg10_ref.iloc[0]) else float("nan")
            diff = abs(got - expected) if expected == expected else float("nan")
            rows.append({"lang": lang, "system": system, "ndcg10_ours": round(got, 4),
                         "ndcg10_ref": expected, "abs_diff": round(diff, 4) if diff == diff else "",
                         "ok": (diff <= TOLERANCE) if diff == diff else "no_ref"})
            print(rows[-1])
    df = pd.DataFrame(rows)
    out = path("results") / "repro_check.csv"
    if out.exists():
        old = pd.read_csv(out)
        redone = df.set_index(["lang", "system"]).index
        df = pd.concat([old[~old.set_index(["lang", "system"]).index.isin(redone)], df])
    df.sort_values(["lang", "system"]).to_csv(out, index=False)
    bad = df[df.ok.eq(False)]
    print(f"[runs] wrote {out}; {len(bad)} run(s) differ from reference by > {TOLERANCE}")


if __name__ == "__main__":
    main()
