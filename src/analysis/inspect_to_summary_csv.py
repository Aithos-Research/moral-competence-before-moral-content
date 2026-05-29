"""Convert inspect_ai eval logs to the cube_run summary.csv format.

Aggregates a single `logs/<scenario>__<model>__<suffix>/<run>.eval` (which
contains all 5 paraphrases × 5 levels × 3 dominances) into the
`paraphrase,impact_level,dominance,n_trials,n_success,n_violations,verdict_rate,pareto_dominated_action`
schema produced by `src.perturbations.cube_run`. Writes the CSV under
`results/exp2/<timestamp>_<scenario>_<safe_model>/summary.csv` so
`src.analysis.plot_cube` can read it directly.

Usage:
    python -m src.analysis.inspect_to_summary_csv \
        --log-dir logs/18_fintech_transparency__sonnet46__paraphrase \
        --scenario fintech_transparency \
        --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
PARAPHRASES = ["canonical", "clinical_log", "terminal_compact", "conversational", "structured_xml"]


def aggregate_log(log_dir: Path) -> dict:
    """Returns rates[(paraphrase, level, dominance)] = {n_trials, n_violations, polarity}."""
    eval_files = sorted(log_dir.glob("*.eval"))
    if not eval_files:
        raise SystemExit(f"no .eval found in {log_dir}")
    log = read_eval_log(str(eval_files[-1]))
    cells: dict = defaultdict(lambda: {"n_trials": 0, "n_violations": 0, "polarity": ""})
    for s in log.samples or []:
        md = s.metadata or {}
        para = md.get("paraphrase")
        level = md.get("impact_level")
        dom = md.get("dominance")
        polarity = md.get("pareto_dominated_action") or ""
        if level not in LEVELS or dom not in DOMS or para not in PARAPHRASES:
            continue
        v = None
        if s.scores:
            for sc in s.scores.values():
                try:
                    v = int(float(sc.value))
                    break
                except (TypeError, ValueError):
                    continue
        if v is None:
            continue
        key = (para, level, dom)
        cells[key]["n_trials"] += 1
        cells[key]["n_violations"] += v
        cells[key]["polarity"] = polarity
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True, help="logs/<scenario>__<model>__<suffix>/")
    ap.add_argument("--scenario", required=True, help="short name, e.g. fintech_transparency")
    ap.add_argument("--model", required=True, help="display name, e.g. claude-sonnet-4-6")
    args = ap.parse_args()

    cells = aggregate_log(Path(args.log_dir))
    if not cells:
        raise SystemExit(f"no cells aggregated from {args.log_dir}")

    safe_model = args.model.replace("/", "-").replace(":", "-")
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = REPO_ROOT / "results" / "exp2" / f"{ts}_{args.scenario}_{safe_model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "summary.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "paraphrase", "impact_level", "dominance",
            "n_trials", "n_success", "n_violations",
            "verdict_rate", "pareto_dominated_action",
        ])
        for para in PARAPHRASES:
            for level in LEVELS:
                for dom in DOMS:
                    c = cells.get((para, level, dom))
                    if c is None:
                        continue
                    n = c["n_trials"]
                    k = c["n_violations"]
                    w.writerow([
                        para, level, dom, n, n, k,
                        f"{(k / n) if n else 0:.6f}",
                        c["polarity"],
                    ])

    print(f"wrote {out_csv}")
    print(f"results dir: {out_dir}")


if __name__ == "__main__":
    main()
