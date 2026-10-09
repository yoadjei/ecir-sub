"""record the environment and check pyserini works on the smallest miracl index.

writes results/ENV.txt, exits non-zero if bm25 search on miracl-v1.0-yo fails.
"""
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys

from src.utils.config import ROOT, path


# import name -> pip name, where they differ
_DIST_NAMES = {"faiss": "faiss-cpu", "sentence_transformers": "sentence-transformers"}


def _ver(mod: str) -> str:
    try:
        return importlib.metadata.version(_DIST_NAMES.get(mod, mod))
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "present")
    except Exception as exc:  # keep the error name for ENV.txt
        return f"MISSING ({exc.__class__.__name__})"


def main() -> int:
    lines = [f"python: {sys.version.split()[0]}",
             f"platform: {platform.platform()}"]
    java = subprocess.run(["java", "-version"], capture_output=True, text=True)
    lines.append("java: " + (java.stderr or java.stdout).splitlines()[0])
    for mod in ["pyserini", "faiss", "torch", "ir_measures", "transformers",
                "sentence_transformers", "datasets", "pandas", "numpy", "scipy", "matplotlib"]:
        lines.append(f"{mod}: {_ver(mod)}")
    # the repo and the home folder can sit on different disks, indexes and caches go to home
    lines.append(f"disk_free_repo_gb: {shutil.disk_usage(str(ROOT)).free / 1e9:.1f}")
    lines.append(f"disk_free_home_gb: {shutil.disk_usage(os.path.expanduser('~')).free / 1e9:.1f}")
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            mem_kb = int(next(l for l in fh if l.startswith("MemTotal")).split()[1])
        lines.append(f"ram_total_gb: {mem_kb / 1e6:.1f}")
    except Exception:  # noqa: BLE001
        lines.append("ram_total_gb: unknown")

    # smoke test, one bm25 query on yoruba
    try:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher.from_prebuilt_index("miracl-v1.0-yo")
        searcher.set_language("yo")
        hits = searcher.search("Nigeria", k=3)
        lines.append(f"bm25_smoke_yo: OK ({len(hits)} hits, top docid {hits[0].docid})")
        ok = len(hits) > 0
    except Exception as exc:  # noqa: BLE001
        lines.append(f"bm25_smoke_yo: FAILED ({exc})")
        ok = False

    out = path("results") / "ENV.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
