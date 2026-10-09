"""one place that reads config.yaml, nothing else should open it."""
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def langs(dense_only: bool = False) -> list[str]:
    """language codes in table order, optionally only the ones with dense runs."""
    items = cfg()["languages"].items()
    return [code for code, info in items if (info["dense"] or not dense_only)]


def tier(lang: str) -> str:
    return cfg()["languages"][lang]["tier"]


def path(key: str) -> Path:
    return ROOT / cfg()["paths"][key]
