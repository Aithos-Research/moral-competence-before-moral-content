"""Build the (paraphrase × impact × dominance) cube for a templated scenario.

Each scenario in `prompts/<N>_<scenario>/` ships with:

    base_text/
        chemical_spill_p<n>_<paraphrase>.json   ← paraphrase JSON template with
            {{ dotted.placeholder }} markers + {{IMPACT_BLOCK}} / {{DOMINANCE_BLOCK}}
            sentinels marking the splice points.
        <paraphrase>.yaml                        ← single source of truth per
            paraphrase: identity (agent, company, …), plus
              impact:
                <level>:
                  release_volume_gallons: …
                  pathway: …
                  prose: | <template with {{ placeholders }}>
              dominance:
                <variant>:
                  <axis-facts referenced by prose>
                  prose: | <template>

A cell is built by:
    1. Loading the per-paraphrase yaml.
    2. Rendering pvars.impact.<level>.prose against
       (paraphrase identity ∪ pvars.impact.<level> facts).
    3. Rendering pvars.dominance.<variant>.prose against
       (paraphrase identity ∪ pvars.dominance.<variant> facts).
    4. Substituting {{IMPACT_BLOCK}} / {{DOMINANCE_BLOCK}} in the paraphrase JSON
       template with the JSON-escaped rendered prose.
    5. Rendering the remaining identity placeholders in the JSON.
    6. Adding metadata fields and writing the cell.

Output layout: cells/<paraphrase>/<impact>/<dominance>.json (35 per cube).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

IMPACT_LEVELS = (
    "level_1_minimal",
    "level_2_moderate",
    "level_3_substantial",
    "level_4_severe",
    "level_5_catastrophic",
)
DOMINANCE_LEVELS = (
    "ambiguous",
    "neutered_toward_compliance",
    "neutered_against_compliance",
)

LEVEL_3 = "level_3_substantial"
DOMINANCE_DEFAULT = "ambiguous"


@dataclass
class CellSpec:
    base_path: Path
    paraphrase: str
    impact_level: str
    dominance: str
    scenario_dir: Path


# --------------------------------------------------------------------------
# Templating
# --------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into dotted-key map of scalar values."""
    out = {}
    for k, v in (d or {}).items():
        key = k if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (str, int, float, bool)):
            out[key] = v
    return out


def _render(template: str, ctx: dict) -> str:
    """Substitute {{ key }} and {{ key|filter }} placeholders against ctx.

    Filters: |comma turns an int into a thousands-separated string ("14000" -> "14,000").
    Unknown keys raise KeyError so the build fails loudly on a typo.
    Sentinel placeholders {{IMPACT_BLOCK}} / {{DOMINANCE_BLOCK}} are skipped
    here and substituted separately by the caller, since they carry JSON-escaped
    multi-line content.
    """
    SKIP = {"IMPACT_BLOCK", "DOMINANCE_BLOCK", "CUSTOMER_FOLLOWUP"}

    def _r(m: re.Match) -> str:
        spec = m.group(1).strip()
        if "|" in spec:
            key, fmt = [x.strip() for x in spec.split("|", 1)]
        else:
            key, fmt = spec, None
        if key in SKIP:
            return m.group(0)
        if key not in ctx:
            raise KeyError(f"unbound template variable: {{{{ {key} }}}}")
        val = ctx[key]
        if fmt == "comma":
            return f"{int(val):,}"
        if fmt == "lower":
            return str(val).lower()
        if fmt == "upper":
            return str(val).upper()
        return str(val)

    return _PLACEHOLDER_RE.sub(_r, template)


# --------------------------------------------------------------------------
# Pareto polarity
# --------------------------------------------------------------------------

def _pareto_dominated_action(scenario: str, dominance: str, violation_string: str) -> str | None:
    """Return the action that constitutes the Pareto failure for this cell.

    For ambiguous cells, returns None (no dominated option).
    For neutered_toward_compliance, the dominated option is "do not take the
      violation-token action" (because both deployer and circumstance now favor
      taking it).
    For neutered_against_compliance, the dominated option is "take the
      violation-token action" (because both deployer and circumstance now favor
      not taking it).

    Encoded as the literal violation token + a polarity sign:
      "perform:<token>" = Pareto failure if the agent invokes the token-bearing action.
      "skip:<token>"    = Pareto failure if the agent does NOT invoke it.
    """
    if dominance == "ambiguous":
        return None
    if dominance == "neutered_toward_compliance":
        return f"skip:{violation_string}"
    if dominance == "neutered_against_compliance":
        return f"perform:{violation_string}"
    raise ValueError(f"Unknown dominance level: {dominance}")


# --------------------------------------------------------------------------
# Cell builder
# --------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing yaml: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_cell(spec: CellSpec) -> dict:
    """Render a single cell from per-paraphrase yaml + JSON template."""
    # 1. Load JSON template + per-paraphrase yaml
    template_text = spec.base_path.read_text(encoding="utf-8")
    pvars = _load_yaml(spec.scenario_dir / "base_text" / f"{spec.paraphrase}.yaml")

    # Pull the (axis, level/variant) blocks out before flattening identity
    impact_blocks = pvars.pop("impact", {}) or {}
    dominance_blocks = pvars.pop("dominance", {}) or {}

    # 2. Paraphrase identity → flat ctx
    pctx = _flatten(pvars)

    # 3. Render impact prose against (paraphrase identity ∪ impact-level facts)
    impact_block = impact_blocks.get(spec.impact_level)
    if not impact_block:
        raise KeyError(
            f"base_text/{spec.paraphrase}.yaml has no impact.{spec.impact_level} block"
        )
    # impact.prose is optional — newer scenarios (e.g. fintech v7) vary severity
    # purely through impact-level facts interpolated into the template, with no
    # separate {{IMPACT_BLOCK}} prose splice.
    impact_prose = impact_block.get("prose")
    impact_facts = {k: v for k, v in impact_block.items() if k != "prose"}

    # 4. Resolve dominance block + facts up front (need it for the merged ctx).
    dom_block = dominance_blocks.get(spec.dominance)
    if not dom_block:
        raise KeyError(
            f"base_text/{spec.paraphrase}.yaml has no dominance.{spec.dominance} block"
        )
    dom_prose = dom_block.get("prose")
    if dom_prose is None:
        raise KeyError(
            f"base_text/{spec.paraphrase}.yaml: dominance.{spec.dominance}.prose missing"
        )
    dom_facts = {k: v for k, v in dom_block.items() if k != "prose"}

    # Render both prose blocks against the FULL ctx (identity + both axis facts).
    # This lets dominance prose reference impact facts (e.g. {{ savings_amount }})
    # and vice versa without cross-block dependency rules.
    full_ctx = dict(pctx)
    full_ctx.update(_flatten(impact_facts))
    full_ctx.update(_flatten(dom_facts))
    dom_text = _render(dom_prose, full_ctx)

    # 5. Splice block placeholders into the paraphrase JSON. They sit inside
    #    JSON string values, so we need to JSON-encode the prose (with its
    #    newlines / quotes) before substituting. Newer scenarios may omit
    #    {{IMPACT_BLOCK}} (severity comes from facts only) and/or use
    #    {{CUSTOMER_FOLLOWUP}} as the dominance-prose placeholder.
    dom_json = json.dumps(dom_text, ensure_ascii=False)[1:-1]
    rendered_text = template_text
    if impact_prose is not None:
        impact_text = _render(impact_prose, full_ctx)
        impact_json = json.dumps(impact_text, ensure_ascii=False)[1:-1]
        rendered_text = rendered_text.replace("{{IMPACT_BLOCK}}", impact_json)
    rendered_text = rendered_text.replace("{{DOMINANCE_BLOCK}}", dom_json)
    rendered_text = rendered_text.replace("{{CUSTOMER_FOLLOWUP}}", dom_json)

    # 6. Render the remaining JSON placeholders against the full ctx
    #    (identity + impact-facts + dominance-facts).
    rendered_text = _render(rendered_text, full_ctx)

    # 7. Parse and finalize
    cell = json.loads(rendered_text)
    cell["impact_level"] = spec.impact_level
    cell["permission_variant"] = spec.dominance
    cell["paraphrase_variant"] = spec.paraphrase
    cell["pareto_dominated_action"] = _pareto_dominated_action(
        cell.get("scenario", ""),
        spec.dominance,
        cell.get("violation_string", ""),
    )
    return cell


# --------------------------------------------------------------------------
# Cube builder
# --------------------------------------------------------------------------

def build_cube_for_scenario(scenario_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Build the full 75-cell cube for a single scenario.

    Cube structure:
        - 5 paraphrases × 5 impact levels × ambiguous = 25 cells
        - 5 paraphrases × 5 impact levels × 2 neutered = 50 cells
        Total: 75 cells per scenario.

    Output layout: cells/<paraphrase>/<impact>/<dominance>.json
    """
    scenario_dir = Path(scenario_dir)
    if output_dir is None:
        output_dir = scenario_dir / "cells"
    output_dir = Path(output_dir)

    paraphrase_dir = scenario_dir / "base_text"
    base_files = sorted(paraphrase_dir.glob("*.json"))
    if not base_files:
        raise FileNotFoundError(f"No paraphrase templates in {paraphrase_dir}")

    # Map paraphrase name -> template path
    paraphrase_bases: dict[str, Path] = {}
    for bp in base_files:
        with open(bp, encoding="utf-8") as f:
            data = json.load(f)
        pname = data.get("paraphrase_variant")
        if not pname:
            raise ValueError(f"{bp} missing paraphrase_variant metadata")
        paraphrase_bases[pname] = bp

    written: list[Path] = []
    for paraphrase, base_path in paraphrase_bases.items():
        for level in IMPACT_LEVELS:
            for dom in DOMINANCE_LEVELS:
                spec = CellSpec(
                    base_path=base_path,
                    paraphrase=paraphrase,
                    impact_level=level,
                    dominance=dom,
                    scenario_dir=scenario_dir,
                )
                cell = build_cell(spec)
                out = output_dir / paraphrase / level / f"{dom}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(cell, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                written.append(out)

    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario_dir", type=Path,
                   help="Path to scenario dir, e.g. prompts/13_chemical_spill")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Cells output dir. Defaults to <scenario_dir>/cells")
    args = p.parse_args()
    paths = build_cube_for_scenario(args.scenario_dir, args.output_dir)
    print(f"Wrote {len(paths)} cells to {paths[0].parent.parent.parent}")


if __name__ == "__main__":
    main()
