"""One-shot exporter: write final_candidates/results.csv from logs/<scenario>__<model>__final/.

Run from repo root: python -m final_candidates.export_results
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inspect_ai.log import read_eval_log  # noqa: E402

LEVELS = [
    "level_1_minimal",
    "level_2_moderate",
    "level_3_substantial",
    "level_4_severe",
    "level_5_catastrophic",
]
DOMS = ["ambiguous", "neutered_toward_compliance", "neutered_against_compliance"]
SCENARIOS = [
    "18_fintech_transparency",
    "23_smart_home_upsell",
]
MODELS = ["sonnet46", "gpt54"]
# per-scenario log suffix
SUFFIX_BY_SCENARIO = {
    "18_fintech_transparency": "__paraphrase_v6",
    "23_smart_home_upsell":    "__paraphrase_v6",
}


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    rates: dict = defaultdict(lambda: [0, 0])
    for sc in SCENARIOS:
        for m in MODELS:
            log_dir = REPO_ROOT / "logs" / f"{sc}__{m}{SUFFIX_BY_SCENARIO[sc]}"
            eval_files = sorted(log_dir.glob("*.eval"))
            if not eval_files:
                print(f"[warn] no eval found in {log_dir}")
                continue
            log = read_eval_log(str(eval_files[-1]))
            for s in log.samples or []:
                md = s.metadata or {}
                level = md.get("impact_level")
                dom = md.get("dominance")
                if level not in LEVELS or dom not in DOMS:
                    continue
                v = None
                if s.scores:
                    for sc_obj in s.scores.values():
                        try:
                            v = int(float(sc_obj.value))
                            break
                        except (TypeError, ValueError):
                            continue
                if v is None:
                    continue
                key = (sc, m, level, dom)
                rates[key][0] += v
                rates[key][1] += 1

    out = REPO_ROOT / "final_candidates" / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "model", "level", "dominance",
                    "violations", "trials", "rate_pct", "ci_lo_pct", "ci_hi_pct"])
        for sc in SCENARIOS:
            for m in MODELS:
                for lvl in LEVELS:
                    for dom in DOMS:
                        k, n = rates[(sc, m, lvl, dom)]
                        rate = (k / n * 100) if n else 0.0
                        lo, hi = wilson_ci(k, n)
                        w.writerow([sc, m, lvl, dom, k, n,
                                    f"{rate:.1f}", f"{lo*100:.1f}", f"{hi*100:.1f}"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
