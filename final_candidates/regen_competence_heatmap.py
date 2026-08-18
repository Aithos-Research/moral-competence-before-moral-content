"""Regenerate figures/competence_heatmap_okabe_ito.png for the PCAIDE paper.

The original figure-generation script for this specific plot was not checked
into this repo (only the per-model dashboards from plot_cube.py are). This
reconstructs the same 3-panel layout from the corrected competence_summary.csv
(see src/analysis/competence.py's robustness_R finite-K normalization fix),
using a colormap extracted pixel-by-pixel from the *existing* published PNG's
colorbar, so colors match the original as closely as possible without the
original source.

Run from repo root:
    python -m final_candidates.regen_competence_heatmap --out <path.png>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "final_candidates" / "plots" / "cube_dashboards" / "paper_metrics" / "competence_summary.csv"

# Colors sampled directly from the colorbar of the published
# competence_heatmap_okabe_ito.png (256 steps, index 0 = value 0.0,
# index 255 = value 1.0). Endpoints land exactly on Okabe-Ito bluish-green
# (#009E73, value=1.0) and yellow (#F0E442, value=0.75); low end runs
# through a darkened vermillion rather than pure #D55E00.
_HEX = ("8e2a00,8f2b00,902b00,912c00,912c00,922d00,932e00,942e00,952f00,963000,963000,973100,983100,"
        "993200,9a3300,9b3300,9b3400,9c3500,9d3500,9e3600,9f3600,a03700,a13800,a13800,a23900,a33900,"
        "a43a00,a53b00,a63b00,a63c00,a73d00,a83d00,a93e00,aa3e00,ab3f00,ac4000,ac4000,ad4100,ae4100,"
        "af4200,b04300,b14300,b14400,b24500,b34500,b44600,b54600,b64700,b64800,b74800,b84900,b94a00,"
        "ba4a00,bb4b00,bc4b00,bc4c00,bd4d00,be4d00,bf4e00,c04e00,c14f00,c15000,c25000,c35100,c45200,"
        "c55200,c65300,c75300,c75400,c85500,c95500,ca5600,cb5600,cc5700,cc5800,cd5800,ce5900,cf5a00,"
        "d05a00,d15b00,d15b00,d25c00,d35d00,d45d00,d55e00,d55f01,d56001,d66202,d66302,d66403,d66504,"
        "d76704,d76805,d76905,d76a06,d86c07,d86d07,d86e08,d86f09,d97109,d9720a,d9730a,da740b,da760c,"
        "da770c,da780d,db790d,db7b0e,db7c0f,db7d0f,dc7e10,dc8011,dc8111,dc8212,dd8312,dd8513,dd8614,"
        "dd8714,de8815,de8a15,de8b16,de8c17,df8d17,df8f18,df9019,df9119,e0921a,e0941a,e0951b,e0961c,"
        "e1971c,e1991d,e19a1d,e19b1e,e29c1f,e29e1f,e29f20,e2a021,e3a121,e3a322,e3a422,e3a523,e4a624,"
        "e4a824,e4a925,e4aa25,e5ab26,e5ad27,e5ae27,e5af28,e6b029,e6b229,e6b32a,e6b42a,e7b52b,e7b72c,"
        "e7b82c,e7b92d,e8ba2e,e8bc2e,e8bd2f,e8be2f,e9bf30,e9c131,e9c231,e9c332,eac432,eac633,eac734,"
        "eac834,ebc935,ebcb36,ebcc36,ebcd37,ecce37,ecd038,ecd139,ecd239,edd33a,edd53a,edd63b,edd73c,"
        "eed83c,eeda3d,eedb3e,eedc3e,efdd3f,efdf3f,efe040,efe141,f0e241,f0e442,ede343,e9e243,e6e144,"
        "e2e045,dedf46,dade46,d7dd47,d3db48,cfda49,cbd949,c8d84a,c4d74b,c0d64c,bcd54d,b8d44d,b5d34e,"
        "b1d24f,add150,a9cf50,a6ce51,a2cd52,9ecc53,9acb53,97ca54,93c955,8fc856,8bc757,88c657,84c458,"
        "80c359,7cc25a,78c15a,75c05b,71bf5c,6dbe5d,69bd5d,66bc5e,62bb5f,5eb960,5ab861,57b761,53b662,"
        "4fb563,4bb464,48b364,44b265,40b166,3cb067,38ae67,35ad68,31ac69,2dab6a,29aa6b,26a96b,22a86c,"
        "1ea76d,1aa66e,17a56e,13a36f,0fa270,0ba171,08a071,049f72,009e73")


def okabe_ito_reconstructed_cmap() -> ListedColormap:
    hexes = _HEX.split(",")
    rgb = np.array([[int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] for h in hexes])
    return ListedColormap(rgb, name="okabe_ito_reconstructed")


PAPER_NAME = {
    "Claude Sonnet 4.6": "Claude Sonnet 4.6",
    "GPT-5.4": "GPT-5.4",
    "Gemini 2.5 Pro": "Gemini 2.5 Pro",
    "Mistral Large 2512": "Mistral Large 2512",
    "GLM 5.1": "GLM 5.1",
    "DeepSeek V4 Pro": "DeepSeek V4 Pro",
    "Kimi K2.6": "Kimi K2.6",
    "Qwen 3.6 Plus": "Qwen 3.6 Plus",
    "Llama (Fireworks)": "Llama 3.3",
}
SCENARIOS = ["Chemical spill", "Fintech transparency", "Smart-home upsell"]
COMPONENTS = ["R", "M", "D", "P"]  # P is displayed/colored as max(0, P)
COMPONENT_LABELS = ["Robustness", "Monotonicity", "Decisiveness", "Pareto viability"]


def load_data() -> dict:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    data = {}
    for r in rows:
        model = PAPER_NAME[r["model"]]
        vals = [float(r["R"]), float(r["M"]), float(r["D"]), max(0.0, float(r["P"]))]
        data.setdefault(r["scenario"], {})[model] = vals
    return data


def luminance(rgb) -> float:
    r, g, b = rgb[:3]
    return 0.299 * r * 255 + 0.587 * g * 255 + 0.114 * b * 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "final_candidates" / "plots" /
                     "cube_dashboards" / "paper_metrics" / "competence_heatmap_okabe_ito_updated.png")
    args = ap.parse_args()

    data = load_data()
    models = sorted(PAPER_NAME.values())
    cmap = okabe_ito_reconstructed_cmap()

    fig = plt.figure(figsize=(3267 / 200, 1880 / 200), dpi=200)
    fig.patch.set_facecolor("white")

    panel_lefts = [0.1264, 0.3961, 0.6659]
    panel_bottom, panel_height, panel_width = 0.1133, 0.8330, 0.2298

    axes = []
    for i, (scen, left) in enumerate(zip(SCENARIOS, panel_lefts)):
        ax = fig.add_axes([left, panel_bottom, panel_width, panel_height])
        grid = np.array([data[scen][m] for m in models])  # (9 models, 4 components)

        n_rows, n_cols = grid.shape
        mesh = ax.pcolormesh(np.arange(n_cols + 1), np.arange(n_rows + 1), grid,
                              cmap=cmap, vmin=0.0, vmax=1.0,
                              edgecolors="white", linewidth=3)
        ax.invert_yaxis()
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)

        for r in range(n_rows):
            for c in range(n_cols):
                val = grid[r, c]
                rgba = cmap(val)
                color = "white" if luminance(rgba) < 140 else "black"
                ax.text(c + 0.5, r + 0.5, f"{val:.2f}", ha="center", va="center",
                         color=color, fontsize=15)

        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(COMPONENT_LABELS, rotation=30, ha="right", fontsize=13)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        if i == 0:
            ax.set_yticklabels(models, fontsize=13)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
        ax.set_title(scen, fontsize=19, pad=14)
        axes.append(ax)

    cax = fig.add_axes([0.9339, panel_bottom, 0.0104, panel_height])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Score (higher is better)", fontsize=14, rotation=270, labelpad=18)
    cbar.ax.tick_params(labelsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
