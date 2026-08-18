"""Regenerate all 27 per-model dashboard PNGs from the already-committed
summary_<scenario>_<model>.csv cell-level files, using the corrected
robustness_R in src/analysis/competence.py.

No raw eval logs / inspect_ai needed: build_all_cube_dashboards.py only uses
those to build summary.csv in the first place, and that file already exists
for every (scenario, model) pair under final_candidates/plots/cube_dashboards/
paper_metrics/. This script just re-feeds those existing CSVs through
plot_cube.py's own dashboard renderer so panel (D)'s R/M/D/P/C numbers pick
up the fix; panels A-C (raw verdict-rate heatmap/trajectories/Pareto) are
untouched since they don't go through robustness_R at all.

Run from repo root:
    python -m final_candidates.regen_all_dashboards --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_METRICS = REPO_ROOT / "final_candidates" / "plots" / "cube_dashboards" / "paper_metrics"

MODELS = [
    ("sonnet46", "claude-sonnet-4-6"),
    ("gpt54", "gpt-5.4"),
    ("gemini25pro", "gemini-2.5-pro"),
    ("mistral2512", "mistral-large-2512"),
    ("glm5p1", "glm-5p1"),
    ("dsv4pro", "deepseek-v4-pro"),
    ("kimik2p6", "kimi-k2p6"),
    ("qwen3p6plus", "qwen3p6-plus"),
    ("llama", "llama-fireworks-deployed"),
]
SCENARIOS = ["chemical_spill", "fintech_v7", "smart_home_v7"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PAPER_METRICS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ok, missing, failed = [], [], []
    for scen in SCENARIOS:
        for tag, disp in MODELS:
            src_csv = PAPER_METRICS / f"summary_{scen}_{tag}.csv"
            if not src_csv.exists():
                missing.append(src_csv.name)
                continue
            with tempfile.TemporaryDirectory() as td:
                rd = Path(td)
                shutil.copyfile(src_csv, rd / "summary.csv")
                (rd / "summary.json").write_text(json.dumps({
                    "model": disp, "scenario": scen,
                    "n_per_cell": "100 amb / 20 neut",
                }, indent=2), encoding="utf-8")

                r = subprocess.run(
                    [sys.executable, "-m", "src.analysis.plot_cube", "--results-dir", str(rd)],
                    cwd=str(REPO_ROOT), capture_output=True, text=True,
                )
                if r.returncode != 0:
                    failed.append((tag, scen, r.stderr.strip()[:300]))
                    continue

                dash_src = rd / "plots" / "dashboard.png"
                dash_dst = args.out_dir / f"dashboard_{scen}_{tag}.png"
                if dash_src.exists():
                    shutil.copyfile(dash_src, dash_dst)
                    ok.append(dash_dst.name)
                else:
                    failed.append((tag, scen, "dashboard.png not produced"))

    print(f"Regenerated {len(ok)}/27 dashboards into {args.out_dir}")
    if missing:
        print("Missing source CSVs:", missing)
    if failed:
        print("Failed:")
        for tag, scen, err in failed:
            print(f"  {tag}/{scen}: {err}")


if __name__ == "__main__":
    main()
