"""Build per-model cube dashboards across all 3 scenarios with paper-spec metrics.

Reads each model's (ambiguous + neutered) eval log dirs, merges into one
summary.csv, calls plot_cube to render, then copies the dashboard PNG to
final_candidates/plots/cube_dashboards/.

Run from repo root:
    python -m final_candidates.build_all_cube_dashboards
"""
from __future__ import annotations

import csv
import datetime
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inspect_ai.log import read_eval_log  # noqa: E402

MODELS = [
    ("sonnet46",    "claude-sonnet-4-6"),
    ("gpt54",       "gpt-5.4"),
    ("gemini25pro", "gemini-2.5-pro"),
    ("mistral2512", "mistral-large-2512"),
    ("glm5p1",      "glm-5p1"),
    ("dsv4pro",     "deepseek-v4-pro"),
    ("kimik2p6",    "kimi-k2p6"),
    ("qwen3p6plus", "qwen3p6-plus"),
    ("llama",       "llama-fireworks-deployed"),
]

SCENARIOS = [
    # (log_scenario_prefix, suffix, display_name, dashboard_tag)
    ("13_chemical_spill",      "chem_full",        "chemical_spill",  "chemical_spill"),
    ("18_fintech_transparency","fintech_full",     "fintech_v7",      "fintech_v7"),
    ("23_smart_home_upsell",   "smart_home_full",  "smart_home_v7",   "smart_home_v7"),
]

LEVELS = ["level_1_minimal", "level_2_moderate", "level_3_substantial",
          "level_4_severe", "level_5_catastrophic"]
DOMS = ["ambiguous", "neutered_toward_compliance", "neutered_against_compliance"]
PARAS = ["canonical", "clinical_log", "terminal_compact", "conversational", "structured_xml"]


def aggregate(log_dir: Path) -> dict:
    cells: dict = defaultdict(lambda: {"n_trials": 0, "n_violations": 0, "polarity": ""})
    evs = sorted(log_dir.glob("*.eval"))
    if not evs:
        return cells
    log = read_eval_log(str(evs[-1]))
    for s in log.samples or []:
        md = s.metadata or {}
        para, level, dom = md.get("paraphrase"), md.get("impact_level"), md.get("dominance")
        if para not in PARAS or level not in LEVELS or dom not in DOMS:
            continue
        v = None
        if s.scores:
            for sc in s.scores.values():
                try:
                    v = int(float(sc.value)); break
                except (TypeError, ValueError):
                    continue
        if v is None:
            continue
        c = cells[(para, level, dom)]
        c["n_trials"] += 1
        c["n_violations"] += v
        c["polarity"] = md.get("pareto_dominated_action") or ""
    return cells


def write_summary(cells: dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paraphrase", "impact_level", "dominance",
                    "n_trials", "n_success", "n_violations",
                    "verdict_rate", "pareto_dominated_action"])
        for para in PARAS:
            for level in LEVELS:
                for dom in DOMS:
                    c = cells.get((para, level, dom))
                    if c is None or c["n_trials"] == 0:
                        continue
                    n, k = c["n_trials"], c["n_violations"]
                    w.writerow([para, level, dom, n, n, k,
                                f"{k/n:.6f}", c["polarity"]])


def main() -> None:
    dest = REPO_ROOT / "final_candidates" / "plots" / "cube_dashboards" / "paper_metrics"
    dest.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for scen_prefix, suffix, scen_display, dash_tag in SCENARIOS:
        print(f"\n=== {scen_display} ===")
        for tag, disp in MODELS:
            amb = REPO_ROOT / "logs" / f"{scen_prefix}__{tag}__{suffix}__ambiguous"
            neu = REPO_ROOT / "logs" / f"{scen_prefix}__{tag}__{suffix}__neutered"
            if not (amb.exists() and neu.exists()):
                print(f"  [{tag}] skip (missing log dir)")
                continue

            cells = aggregate(amb)
            for k, v in aggregate(neu).items():
                if k in cells:
                    cells[k]["n_trials"] += v["n_trials"]
                    cells[k]["n_violations"] += v["n_violations"]
                    cells[k]["polarity"] = cells[k]["polarity"] or v["polarity"]
                else:
                    cells[k] = v
            if not cells:
                print(f"  [{tag}] skip (empty)")
                continue

            rd = REPO_ROOT / "results" / "exp2" / f"{ts}_{scen_display}_{disp}"
            write_summary(cells, rd / "summary.csv")
            (rd / "summary.json").write_text(json.dumps({
                "model": disp,
                "scenario": scen_display,
                "n_per_cell": "100 amb / 20 neut",
            }, indent=2), encoding="utf-8")

            r = subprocess.run(
                [sys.executable, "-m", "src.analysis.plot_cube",
                 "--results-dir", str(rd)],
                cwd=str(REPO_ROOT), capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"  [{tag}] ERR plot_cube: {r.stderr.strip()[:300]}")
                continue

            src = rd / "plots" / "dashboard.png"
            dst = dest / f"dashboard_{dash_tag}_{tag}.png"
            csv_dst = dest / f"summary_{dash_tag}_{tag}.csv"
            if src.exists():
                shutil.copyfile(src, dst)
                shutil.copyfile(rd / "summary.csv", csv_dst)
                # Pull the metric line
                triple = {}
                for line in r.stdout.splitlines():
                    for k in ("R  Robustness", "M  Monotonicity", "D  Decisiveness",
                              "P  Pareto", "C  Competence"):
                        if line.strip().startswith(k):
                            triple[k.split()[0]] = line.strip().split(":")[-1].strip()
                print(f"  [{tag}] -> {dst.relative_to(REPO_ROOT)}   "
                      + " ".join(f"{k}={v}" for k, v in triple.items()))
            else:
                print(f"  [{tag}] warn: dashboard.png not produced")


if __name__ == "__main__":
    main()
