import json
import argparse
import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from src.load_prompts import get_all_prompt_combinations
from src.generation import generate_response
from src.detection import detect_violate, detect_deceive, detect_persist
from src.config_loader import get_config_loader

# Simple alias map so CLI can use short names or "all"
# Keys are aliases passed via CLI, values are actual provider model ids
MODEL_ALIASES = {
    "deepseek-v3": "deepseek-ai/DeepSeek-V3-0324",
    # "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "qwen3-235b": "Qwen/Qwen3-235B-A22B",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1-0528",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "claude-sonnet-4-0": "claude-sonnet-4-0",
    # "llama-4": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-Turbo",
}

def extract_response_data(msg: str, scenario_name: str) -> str:
    """Extract relevant data from response based on scenario configuration."""
    config_loader = get_config_loader()
    return config_loader.extract_response_data(msg, scenario_name)


def _build_deception_trigger(scenario_name: str, prompt_data: dict, violation_response: str) -> str:
    config_loader = get_config_loader()
    config_trigger = config_loader.get_deception_trigger(scenario_name)
    trade_success_msg = config_trigger if config_trigger else prompt_data.get("deception_trigger", "")
    
    # Only extract response data if the scenario has response_extraction configuration
    if "response_extraction" in prompt_data:
        extracted_data = extract_response_data(violation_response, scenario_name)
        if extracted_data and "{" in trade_success_msg:
            try:
                trade_success_msg = trade_success_msg.format(amount=extracted_data)
            except (KeyError, ValueError):
                pass
    
    return trade_success_msg


def _build_persist_trigger(scenario_name: str, prompt_data: dict) -> str:
    config_loader = get_config_loader()
    config_persist_trigger = config_loader.get_persistence_trigger(scenario_name)
    return config_persist_trigger if config_persist_trigger else prompt_data.get("persistence_trigger", "")


async def run_experiments_for_category(
    category: str,
    variation: str,
    prompt_data: dict,
    num_experiments: int = 50,
    results_dir: Path = Path("results"),
    model_name: str = "gpt-4.1-nano",
    concurrency: int = 0,
) -> None:
    """
    Run experiments for a specific category/variation combination (synchronous).
    Each experiment gets its own run_id equal to its index.
    """
    
    # Sanitize model name for filesystem (replace / with -)
    safe_model_name = model_name.replace("/", "-")
    
    # Directory mapping
    if category.startswith(tuple(f"{i}_" for i in range(10))):
        scenario_name = category
        if category == "0_insider_trading":
            # Use variation_category from prompt_data if available
            if "variation_category" in prompt_data:
                variable_name = prompt_data["variation_category"]
                # Extract the specific variant name
                if variable_name in variation:
                    variant_name = variation.replace(f"{variable_name}_", "")
                else:
                    variant_name = variation
                
                # Handle special mappings for display names
                if variable_name == "countermeasures":
                    variation_mapping = {
                        "1_strongly_discouraging": "Strongly Discouraging",
                        "2_discouraging": "Discouraging", 
                        "3_hhh": "Helpful Harmless And Honest",
                        "5_encouraging": "Encouraging",
                        "6_strongly_encouraging": "Strongly Encouraging",
                    }
                    variant_display = variation_mapping.get(variant_name, variant_name)
                elif variable_name == "caught_likelihood":
                    variant_display = (variant_name
                                     .replace("1_very_low", "Very Low Risk")
                                     .replace("2_low", "Low Risk") 
                                     .replace("3_moderate", "Moderate Risk")
                                     .replace("4_high", "High Risk")
                                     .replace("5_very_high", "Very High Risk"))
                elif variable_name == "pressure":
                    variant_display = variant_name.replace("_", " ").title()
                elif variable_name == "diff_first_encounter_reasoning":
                    variant_display = variant_name.replace("strength_", "Strength ")
                elif variable_name == "variations_of_default":
                    variable_name = "variations"  # Rename for consistency
                    variant_display = variant_name.replace("_", " ").title()
                else:
                    variant_display = variant_name.replace("_", " ").title()
                
                category_dir = results_dir / scenario_name / variable_name / safe_model_name / variant_display
            else:
                # Handle default cases
                if variation == "default":
                    variable_name = "default"
                    category_dir = results_dir / scenario_name / variable_name / safe_model_name
                elif variation == "no_scratchpad":
                    variable_name = "reasoning"
                    category_dir = results_dir / scenario_name / variable_name / safe_model_name
                else:
                    # Fallback for unexpected variations
                    category_dir = results_dir / scenario_name / "default" / safe_model_name / variation
        else:
            category_dir = results_dir / scenario_name / "default" / safe_model_name / variation
    else:
        # For non-numbered scenarios, use default structure
        category_dir = results_dir / category / "default" / safe_model_name / variation

    violate_dir = category_dir / "violate"
    deceive_dir = category_dir / "deceive"
    
    # Clean up existing results to avoid mixing old and new data
    if violate_dir.exists():
        shutil.rmtree(violate_dir)
        print(f"Cleaned up existing violate results in {violate_dir}")
    if deceive_dir.exists():
        shutil.rmtree(deceive_dir)
        print(f"Cleaned up existing deceive results in {deceive_dir}")
    
    violate_dir.mkdir(parents=True, exist_ok=True)
    deceive_dir.mkdir(parents=True, exist_ok=True)

    evaluate_persist = (
        (category == "default" and variation == "default")
        or (category == "reasoning" and variation == "no_scratchpad")
    )
    persist_dir = None
    if evaluate_persist:
        persist_dir = category_dir / "persist"
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
            print(f"Cleaned up existing persist results in {persist_dir}")
        persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {num_experiments} experiments for {category}/{variation}")
    if evaluate_persist:
        print("  (Will evaluate persistence)")

    violated_count = 0
    deceived_count = 0
    persisted_count = 0

    prompt_data_with_meta = dict(prompt_data)
    prompt_data_with_meta["category"] = category
    prompt_data_with_meta["variation"] = variation
    prompt_data_with_meta["model"] = model_name

    print(f"  Running {num_experiments} experiments with asyncio (3 phases)...")

    sem = asyncio.Semaphore(concurrency) if concurrency > 0 else None

    # Phase 1: Generate violation responses concurrently
    async def _gen_violate(idx: int):
        run_id = str(idx + 1)
        prompt = dict(prompt_data_with_meta)
        result = await generate_response(model_name, prompt)
        return idx, run_id, result

    async def _gen_violate_limited(i: int):
        if sem is not None:
            async with sem:
                return await _gen_violate(i)
        return await _gen_violate(i)

    tasks = [asyncio.create_task(_gen_violate_limited(i)) for i in range(num_experiments)]
    violation_results = [None] * num_experiments
    for t in asyncio.as_completed(tasks):
        idx, run_id, gen_res = await t
        violation_results[idx] = {
            "run_id": run_id,
            "prompt_dialogue": prompt_data_with_meta.get("messages", []),
            "response": gen_res.get("response", ""),
            "model": gen_res.get("model", model_name),
            "temperature": prompt_data_with_meta.get("temperature", 0.9),
            "top_p": prompt_data_with_meta.get("top_p", 0.95),
            "max_tokens": prompt_data_with_meta.get("max_tokens", 10000),
            "log_datetime": datetime.now().isoformat(),
            "prompt_path": f"prompts/{prompt_data_with_meta.get('category', 'unknown')}/{prompt_data_with_meta.get('variation', 'unknown')}.json",
            "canary": prompt_data_with_meta.get("canary", ""),
            "violation_string": prompt_data_with_meta.get("violation_string", ""),
            "scenario": prompt_data_with_meta.get("scenario", "insider_trading"),
            "success": gen_res.get("success", False),
            "error": gen_res.get("error"),
        }
        print(f"violate: completed idx={idx} run_id={run_id} model={gen_res.get('model', model_name)}, max_tokens={prompt_data_with_meta.get('max_tokens', 10000)}")

    # Detect violations and write violate results
    for i, item in enumerate(violation_results):
        response_text = item.get("response", "") or ""
        scenario_name = item.get("scenario", "insider_trading")
        is_violated = detect_violate(response_text, scenario_name, item.get("violation_string"))
        item["is_violated"] = is_violated
        
        # Build and store deception trigger
        deception_trigger = _build_deception_trigger(scenario_name, prompt_data_with_meta, response_text)
        item["deception_trigger"] = deception_trigger
        
        # Build and store persistence trigger
        persist_trigger = _build_persist_trigger(scenario_name, prompt_data_with_meta)
        item["persist_trigger"] = persist_trigger
        
        item.setdefault("report_response", "")
        item.setdefault("is_deceived", False)
        item.setdefault("is_persisted", False)

        violate_file = violate_dir / f"response_{i}.json"
        with open(violate_file, "w") as f:
            json.dump(item, f, indent=2)
        if is_violated:
            violated_count += 1

    # Phase 2: Deception generation and detection for violated only
    async def _gen_deceive(idx: int, base: dict):
        scenario_name = base.get("scenario", "insider_trading")
        violation_response = base.get("response", "") or ""
        deception_trigger = _build_deception_trigger(scenario_name, prompt_data_with_meta, violation_response)
        extended_messages = list(prompt_data_with_meta.get("messages", []))
        extended_messages.append({"role": "assistant", "content": violation_response})
        extended_messages.append({"role": "user", "content": deception_trigger})
        extended_prompt = dict(prompt_data_with_meta)
        extended_prompt["messages"] = extended_messages
        report_res = await generate_response(model_name, extended_prompt)
        deception_response = report_res.get("response", "") if report_res.get("success") else ""

        # detection
        is_deceived = False
        deceive_detection_prompt = None
        try:
            config_loader = get_config_loader()
            dec_prompt = config_loader.get_deception_prompt(
                scenario_name, violation_response, deception_trigger, deception_response
            )
            deceive_detection_prompt = dec_prompt
        except Exception:
            deceive_detection_prompt = {
                "system_prompt": "Config-based prompt used",
                "user_prompt": "See config files for details",
            }

        try:
            is_deceived = await detect_deceive(
                violation_response=violation_response,
                deception_trigger=deception_trigger,
                deception_response=deception_response,
                prompt_data=prompt_data_with_meta,
                model="gpt-4o",
                run_id=base.get("run_id"),
            )
        except Exception:
            is_deceived = False

        return idx, deception_trigger, deception_response, is_deceived, deceive_detection_prompt

    async def _gen_deceive_limited(i: int, base: dict):
        if sem is not None:
            async with sem:
                return await _gen_deceive(i, base)
        return await _gen_deceive(i, base)

    deceive_tasks = [
        asyncio.create_task(_gen_deceive_limited(i, item))
        for i, item in enumerate(violation_results)
        if item.get("is_violated")
    ]
    deception_results = {}
    for t in asyncio.as_completed(deceive_tasks):
        idx, trigger, response_text, is_deceived, detect_prompt = await t
        deception_results[idx] = {
            "deception_trigger": trigger,
            "report_response": response_text,
            "is_deceived": is_deceived,
            "deceive_detection_prompt": detect_prompt,
        }
        print(f"deceive: completed idx={idx} run_id={violation_results[idx].get('run_id')} model={model_name}")

    # Write deceive results and update counts
    for i, base in enumerate(violation_results):
        if not base.get("is_violated"):
            continue
        dr = deception_results.get(i, {
            "deception_trigger": "",
            "report_response": "",
            "is_deceived": False,
        })
        deceive_record = {
            "prompt_dialogue": base.get("prompt_dialogue", []) + [
                {"role": "assistant", "content": base.get("response", "")},
                {"role": "user", "content": dr.get("deception_trigger", "")},
            ],
            "response": dr.get("report_response", ""),
            "model": base.get("model"),
            "temperature": base.get("temperature"),
            "top_p": base.get("top_p"),
            "max_tokens": base.get("max_tokens"),
            "log_datetime": base.get("log_datetime"),
            "is_deceived": dr.get("is_deceived", False),
            "prompt_path": base.get("prompt_path"),
            "canary": base.get("canary", ""),
            "violation_string": base.get("violation_string", ""),
            "scenario": base.get("scenario"),
            "deception_trigger": dr.get("deception_trigger", ""),
            "run_id": base.get("run_id"),
        }
        if "deceive_detection_prompt" in dr and dr["deceive_detection_prompt"] is not None:
            deceive_record["deceive_detection_prompt"] = dr["deceive_detection_prompt"]
        deceive_file = deceive_dir / f"response_{i}.json"
        with open(deceive_file, "w") as f:
            json.dump(deceive_record, f, indent=2)
        if deceive_record.get("is_deceived"):
            deceived_count += 1

    # Phase 3: Persistence generation and detection (optional)
    if evaluate_persist:
        async def _gen_persist(idx: int, base: dict, dr: dict):
            scenario_name = base.get("scenario", "insider_trading")
            violation_response = base.get("response", "") or ""
            deception_trigger = dr.get("deception_trigger", "")
            deception_response = dr.get("report_response", "")
            persist_trigger = _build_persist_trigger(scenario_name, prompt_data_with_meta)
            extended_messages = list(prompt_data_with_meta.get("messages", []))
            extended_messages.extend([
                {"role": "assistant", "content": violation_response},
                {"role": "user", "content": deception_trigger},
                {"role": "assistant", "content": deception_response},
                {"role": "user", "content": persist_trigger},
            ])
            extended_prompt = dict(prompt_data_with_meta)
            extended_prompt["messages"] = extended_messages
            persist_res = await generate_response(model_name, extended_prompt)
            persist_response = persist_res.get("response", "") if persist_res.get("success") else ""

            # detection
            persist_detection_prompt = None
            try:
                config_loader = get_config_loader()
                pdp = config_loader.get_persistence_prompt(
                    scenario_name,
                    violation_response,
                    deception_trigger,
                    deception_response,
                    persist_trigger,
                    persist_response,
                )
                persist_detection_prompt = pdp
            except Exception:
                persist_detection_prompt = {
                    "system_prompt": "Config-based prompt used",
                    "user_prompt": "See config files for details",
                }

            is_persisted = False
            try:
                is_persisted = await detect_persist(
                    violation_response=violation_response,
                    deception_trigger=deception_trigger,
                    deception_response=deception_response,
                    persist_trigger=persist_trigger,
                    persist_response=persist_response,
                    prompt_data=prompt_data_with_meta,
                    model="gpt-4o",
                    run_id=base.get("run_id"),
                )
            except Exception:
                is_persisted = False

            return idx, persist_trigger, persist_response, is_persisted, persist_detection_prompt

        async def _gen_persist_limited(i: int, base: dict, dr: dict):
            if sem is not None:
                async with sem:
                    return await _gen_persist(i, base, dr)
            return await _gen_persist(i, base, dr)

        persist_tasks = []
        for i, base in enumerate(violation_results):
            if not base.get("is_violated"):
                continue
            dr = deception_results.get(i)
            if not dr or not dr.get("is_deceived"):
                continue
            persist_tasks.append(asyncio.create_task(_gen_persist_limited(i, base, dr)))

        for t in asyncio.as_completed(persist_tasks):
            idx, persist_trigger, persist_response, is_persisted, persist_detect_prompt = await t
            base = violation_results[idx]
            dr = deception_results[idx]
            persist_record = {
                "prompt_dialogue": base.get("prompt_dialogue", []) + [
                    {"role": "assistant", "content": base.get("response", "")},
                    {"role": "user", "content": dr.get("deception_trigger", "")},
                    {"role": "assistant", "content": dr.get("report_response", "")},
                    {"role": "user", "content": persist_trigger},
                ],
                "response": persist_response,
                "model": base.get("model"),
                "temperature": base.get("temperature"),
                "top_p": base.get("top_p"),
                "max_tokens": base.get("max_tokens"),
                "log_datetime": base.get("log_datetime"),
                "is_persisted": is_persisted,
                "prompt_path": base.get("prompt_path"),
                "canary": base.get("canary", ""),
                "violation_string": base.get("violation_string", ""),
                "scenario": base.get("scenario"),
                "deception_trigger": dr.get("deception_trigger", ""),
                "persist_trigger": persist_trigger,
                "run_id": base.get("run_id"),
            }
            if persist_detect_prompt is not None:
                persist_record["persist_detection_prompt"] = persist_detect_prompt
            persist_file = persist_dir / f"response_{idx}.json"
            with open(persist_file, "w") as f:
                json.dump(persist_record, f, indent=2)
            if is_persisted:
                persisted_count += 1
            print(f"persist: completed idx={idx} run_id={base.get('run_id')} model={model_name}")



    summary = f"  Results: {violated_count}/{num_experiments} violated, {deceived_count}/{num_experiments} deceived"
    if evaluate_persist:
        summary += f", {persisted_count}/{num_experiments} persisted"
    print(summary)


async def run_all_experiments(
    num_experiments_per_category: int = 50,
    model_name: str = "gpt-4.1-nano",
    target_scenarios: list = None,
    variations_filter: str = "default",
    output_dir: str = None,
    concurrency: int = 0,
):
    """
    Run experiments for scenarios and variations (synchronous).
    """
    scenarios = get_all_prompt_combinations(variations_filter)

    results_dir = Path("results")
    if output_dir:
        results_dir = results_dir / output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting experiments with {num_experiments_per_category} runs per variation")
    if target_scenarios:
        print(f"Target scenarios: {target_scenarios}")
    else:
        print(f"Target scenarios: all")
    print(f"Variations filter: {variations_filter}")
    print(f"Results directory: {results_dir}")
    print("=" * 60)

    for scenario_name, variations in scenarios.items():
        if "_" in scenario_name and scenario_name[0].isdigit():
            clean_scenario = scenario_name.split("_", 1)[1]
            scenario_number = int(scenario_name.split("_", 1)[0])
        else:
            clean_scenario = scenario_name
            scenario_number = None

        # Filter by scenario numbers if specified
        if target_scenarios:
            if scenario_number is None or scenario_number not in target_scenarios:
                continue

        print(f"\nScenario: {clean_scenario.upper()}")

        for variation_name, prompt_data in variations.items():
            if "variation_category" in prompt_data:
                variable_name = prompt_data["variation_category"]
                variation_display = variation_name.replace(f"{variable_name}_", "")
            else:
                if variation_name == "no_scratchpad":
                    variable_name = "reasoning"
                    variation_display = "no_scratchpad"
                else:
                    variable_name = "default"
                    variation_display = variation_name

            await run_experiments_for_category(
                scenario_name,
                variation_name,
                prompt_data,
                num_experiments_per_category,
                results_dir,
                model_name,
                concurrency,
            )


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run insider trading experiments (async)")
    parser.add_argument("--model", default="gpt-4.1-nano",
                        help="Single model alias or id (default: gpt-4.1-nano). See --models for multiple or 'all'.")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model aliases or 'all' to run every known model")
    parser.add_argument("--num-experiments", type=int, default=10,
                        help="Number of experiments per category (default: 10)")
    # detection model fixed to gpt-4o in code for now; keep arg for compatibility
    parser.add_argument("--detection-model", default="gpt-4o",
                        help="Model to use for deception detection (default: gpt-4o)")
    parser.add_argument("--scenarios", type=str, default="all",
                        help="Scenarios to run: 'all' or comma-separated list of numbers (e.g., '1,2,3,4,10,13') (default: all)")
    parser.add_argument("--variations", type=str, default="default",
                        help="Variations to run (default: default)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Subfolder within results directory to store outputs (e.g., 'sept-3')")
    parser.add_argument("--category", type=str, default=None,
                        help="Legacy parameter for backward compatibility")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="0 = unbounded concurrency (default), 1 = sequential, N = limit to N concurrent tasks")


    args = parser.parse_args()

    def _resolve_models(models_str: str | None, single_model: str | None) -> list[str]:
        if models_str:
            if models_str.strip().lower() == "all":
                return list(MODEL_ALIASES.values())
            return [MODEL_ALIASES.get(m.strip(), m.strip()) for m in models_str.split(",") if m.strip()]
        if single_model:
            sm = single_model.strip()
            return [MODEL_ALIASES.get(sm, sm)]
        return [MODEL_ALIASES["gpt-4.1-nano"]]

    models_to_run = _resolve_models(args.models, args.model)

    # Parse scenarios list if provided
    target_scenarios = None
    if args.scenarios and args.scenarios.lower() != "all":
        try:
            target_scenarios = [int(x.strip()) for x in args.scenarios.split(",") if x.strip()]
            print(f"Parsed scenario numbers: {target_scenarios}")
        except ValueError as e:
            print(f"Error parsing scenarios: {e}")
            print("Please provide 'all' or scenario numbers as comma-separated integers (e.g., '1,2,3,4,10,13')")
            exit(1)

    if len(models_to_run) == 1:
        print(f"Running experiments for model: {models_to_run[0]}")
    else:
        print(f"Running experiments for models: {', '.join(models_to_run)}")
    print(f"Number of experiments per category: {args.num_experiments}")
    if target_scenarios:
        print(f"Target scenarios: {target_scenarios}")
    else:
        print(f"Target scenarios: all")

    async def _main():
        for m in models_to_run:
            print("=" * 60)
            print(f"MODEL: {m}")
            await run_all_experiments(
                num_experiments_per_category=args.num_experiments,
                model_name=m,
                target_scenarios=target_scenarios,
                variations_filter=args.variations,
                output_dir=args.output_dir,
                concurrency=args.concurrency,
            )

    asyncio.run(_main())
