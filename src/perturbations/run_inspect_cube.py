"""Driver to run the chemical_spill cube via inspect_ai's Python API.

Bypasses inspect_ai's CLI task-glob loader, which on Windows + Python 3.11
trips ``NotImplementedError: Non-relative patterns are unsupported`` when
``Path.glob()`` is handed an absolute path.

Usage (from repo root):

    python -m src.perturbations.run_inspect_cube --model anthropic/claude-sonnet-4-6 --epochs 1
    python -m src.perturbations.run_inspect_cube --model anthropic/claude-sonnet-4-6 --epochs 10 --max-connections 10
    python -m src.perturbations.run_inspect_cube --model openai/gpt-5.4              --epochs 10 --max-connections 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from inspect_ai import eval as inspect_eval  # noqa: E402

from src.perturbations.inspect_cube import chemical_spill_cube  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run the chemical_spill cube via inspect_ai.")
    p.add_argument("--model", required=True,
                   help='e.g. "anthropic/claude-sonnet-4-6" or "openai/gpt-5.4"')
    p.add_argument("--epochs", type=int, default=1,
                   help="Repetitions per cell (default: 1 for smoke test).")
    p.add_argument("--turns", type=int, default=6,
                   help="Max consecutive model turns per trial (default: 6).")
    p.add_argument("--max-connections", type=int, default=5,
                   help="Concurrent API connections (default: 5).")
    p.add_argument("--log-dir", default="logs",
                   help="Where inspect_ai writes the .eval file (default: ./logs).")
    p.add_argument("--scenario", default="13_chemical_spill",
                   help='Scenario folder under prompts/ (default: "13_chemical_spill"). '
                        'e.g. "14_annuity_manipulation".')
    p.add_argument("--dominance-filter", default=None,
                   choices=[None, "ambiguous", "neutered"],
                   help='Restrict cells loaded: "ambiguous" (1 dom), "neutered" (2 doms), or all.')
    args = p.parse_args()

    inspect_eval(
        tasks=chemical_spill_cube(
            turns=args.turns,
            scenario=args.scenario,
            dominance_filter=args.dominance_filter,
        ),
        model=args.model,
        epochs=args.epochs,
        max_connections=args.max_connections,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
