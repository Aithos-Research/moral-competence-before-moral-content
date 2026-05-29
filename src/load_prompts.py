import json
from pathlib import Path

def get_all_prompt_combinations(variations_filter: str = "all") -> dict:
    """Get all prompt combinations from the prompts folder.
    
    Args:
        variations_filter: "default", "all", or specific variation name
    """
    prompts_dir = Path("prompts")
    
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")
    
    prompt_combinations = {}
    
    # Get scenario variations (subdirectories)
    for scenario_dir in prompts_dir.iterdir():
        if scenario_dir.is_dir():
            scenario_name = scenario_dir.name
            scenario_variations = {}
            
            # Extract scenario name from numbered folders (e.g., "0_insider_trading" -> "insider_trading")
            if '_' in scenario_name and scenario_name[0].isdigit():
                scenario_key = scenario_name.split('_', 1)[1]  # Get part after first underscore
            else:
                scenario_key = scenario_name
            
            # Load direct JSON files (like default.json, no_scratchpad.json)
            for variation_file in scenario_dir.glob("*.json"):
                variation_name = variation_file.stem
                
                # Apply variations filter
                if variations_filter == "default" and variation_name != "default":
                    continue
                
                with open(variation_file, 'r') as f:
                    prompt_data = json.load(f)
                    # Set the scenario field in prompt data if not present
                    if "scenario" not in prompt_data:
                        prompt_data["scenario"] = scenario_key
                    scenario_variations[variation_name] = prompt_data
            
            # Load nested variation folders (like countermeasures/, pressure/, etc.)
            if variations_filter == "all":
                for variation_dir in scenario_dir.iterdir():
                    if variation_dir.is_dir():
                        variation_category = variation_dir.name
                        for nested_file in variation_dir.glob("*.json"):
                            nested_name = nested_file.stem
                            combined_key = f"{variation_category}_{nested_name}"
                            
                            with open(nested_file, 'r') as f:
                                prompt_data = json.load(f)
                                # Set the scenario field in prompt data if not present
                                if "scenario" not in prompt_data:
                                    prompt_data["scenario"] = scenario_key
                                # Add variation category info
                                prompt_data["variation_category"] = variation_category
                                scenario_variations[combined_key] = prompt_data
            
            if scenario_variations:
                prompt_combinations[scenario_name] = scenario_variations
    
    return prompt_combinations

def get_scenario_prompts(scenario_name: str) -> dict:
    """Get all prompt variations for a specific scenario."""
    prompts_dir = Path("prompts") / scenario_name
    
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {prompts_dir}")
    
    scenario_prompts = {}
    for variation_file in prompts_dir.glob("*.json"):
        with open(variation_file, 'r') as f:
            scenario_prompts[variation_file.stem] = json.load(f)
    
    return scenario_prompts

def list_scenarios() -> list:
    """List all available scenarios."""
    prompts_dir = Path("prompts")
    scenarios = []
    
    for scenario_dir in prompts_dir.iterdir():
        if scenario_dir.is_dir():
            scenarios.append(scenario_dir.name)
    
    return scenarios

if __name__ == "__main__":
    prompts = get_all_prompt_combinations()
    print("Prompt Structure:")
    for category, variations in prompts.items():
        print(f"{category}: {list(variations.keys())}")