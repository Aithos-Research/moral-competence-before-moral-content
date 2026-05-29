import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def ensure_directory_exists(filepath):
    """Create directory for filepath if it doesn't exist."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

BEHAVIOR_DIRS = {"violate", "misalignment", "deceive", "deception", "persist", "doubling_down"}

# Map CLI aliases to on-disk folder names (slashes are replaced with '-')
MODEL_ALIASES = {
    "gpt-4.1-nano": "gpt-4.1-nano",
    "deepseek-v3": "deepseek-ai-DeepSeek-V3-0324",
    "qwen3-235b": "Qwen-Qwen3-235B-A22B",
    "gpt-oss-20b": "openai-gpt-oss-20b",
}


def load_results(results_folder, model_name=None):
    """Load results from the results folder."""
    if not os.path.exists(results_folder):
        raise FileNotFoundError(f"Results folder '{results_folder}' does not exist.")
    
    results = {}
    
    # Determine structure (model/category vs category/model)
    first_level = [item for item in os.listdir(results_folder) 
                   if os.path.isdir(os.path.join(results_folder, item))]
    # New structure if folders look like numbered scenarios (e.g., '0_insider_trading')
    is_new_structure = any(re.match(r"^\d+_", name) for name in first_level)
    
    # Load data based on structure
    if is_new_structure:
        # New structure: scenario/variable/model/(variation?)/behavior/
        available_models = set()
        for scenario in first_level:
            scenario_path = os.path.join(results_folder, scenario)
            if not os.path.isdir(scenario_path):
                continue
            for variable in os.listdir(scenario_path):
                variable_path = os.path.join(scenario_path, variable)
                if not os.path.isdir(variable_path):
                    continue
                for model in os.listdir(variable_path):
                    model_path = os.path.join(variable_path, model)
                    if os.path.isdir(model_path):
                        available_models.add(model)

        models_to_process = [model_name] if model_name else sorted(available_models)
        for model in models_to_process:
            results[model] = {}
            for scenario in first_level:
                scenario_path = os.path.join(results_folder, scenario)
                if not os.path.isdir(scenario_path):
                    continue
                for variable in os.listdir(scenario_path):
                    variable_path = os.path.join(scenario_path, variable)
                    if not os.path.isdir(variable_path):
                        continue
                    model_path = os.path.join(variable_path, model)
                    if os.path.isdir(model_path):
                        key = scenario if variable == "default" else f"{scenario}_{variable}"
                        results[model][key] = load_category_data(model_path)
    else:
        # Old structure: model/category/variation/behavior/
        models_to_process = [model_name] if model_name else first_level
        for model in models_to_process:
            model_path = os.path.join(results_folder, model)
            if not os.path.isdir(model_path):
                continue
            results[model] = load_model_data(model_path)
    
    # Return single model data if requested
    if model_name and model_name in results:
        return results[model_name]
    
    return results

def load_model_data(model_path):
    """Load data for a single model."""
    model_data = {}
    for category in os.listdir(model_path):
        category_path = os.path.join(model_path, category)
        if os.path.isdir(category_path):
            model_data[category] = load_category_data(category_path)
    return model_data

def load_category_data(category_path):
    """Load data for a single category."""
    category_data = {}
    # If behavior folders are directly present, treat as a single 'default' variation
    entries = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
    if any(d in BEHAVIOR_DIRS for d in entries):
        category_data["default"] = load_variation_data(category_path)
        return category_data

    for variation in entries:
        variation_path = os.path.join(category_path, variation)
        if os.path.isdir(variation_path):
            category_data[variation] = load_variation_data(variation_path)
    return category_data

def load_variation_data(variation_path):
    """Load data for a single variation."""
    variation_data = {'violate': [], 'deceive': [], 'persist': []}
    
    # Get expected total from violate folder (backwards compatibility with old misalignment folder)
    violate_path = os.path.join(variation_path, 'violate')
    misalignment_path = os.path.join(variation_path, 'misalignment')  # backwards compatibility
    expected_total = 0
    
    if os.path.exists(violate_path):
        expected_total = len([f for f in os.listdir(violate_path) if f.endswith('.json')])
    elif os.path.exists(misalignment_path):
        expected_total = len([f for f in os.listdir(misalignment_path) if f.endswith('.json')])
    
    variation_data['expected_total'] = expected_total
    
    # Load each behavior type (with backwards compatibility)
    behavior_mappings = {
        'violate': ['violate', 'misalignment'],
        'deceive': ['deceive', 'deception'], 
        'persist': ['persist', 'doubling_down']
    }
    
    for behavior, folder_names in behavior_mappings.items():
        for folder_name in folder_names:
            behavior_path = os.path.join(variation_path, folder_name)
            if os.path.exists(behavior_path):
                for file in os.listdir(behavior_path):
                    if file.endswith('.json'):
                        with open(os.path.join(behavior_path, file), 'r') as f:
                            data = json.load(f)
                            if behavior == 'violate':
                                variation_data[behavior].append(data.get('is_violated', data.get('is_misaligned', False)))
                            elif behavior == 'deceive':
                                variation_data[behavior].append(data.get('is_deceived', data.get('is_deceptive', False)))
                            elif behavior == 'persist':
                                variation_data[behavior].append(data.get('is_persisted', data.get('is_doubling_down', False)))
                break  # Only process first existing folder
    
    return variation_data

def maybe_show(show: bool) -> None:
    if show:
        try:
            plt.show()
        except Exception:
            pass


def plot_scenario_results(results, model_name, filename='results/figures/scenarios.png', show: bool = False):
    """Plot results across different scenarios for a single model."""
    ensure_directory_exists(filename)
    colors = {'violate': '#1f77b4', 'deceive': '#ff7f0e', 'persist': '#d62728'}
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get all scenarios for the given model
    if model_name not in results:
        print(f"Model '{model_name}' not found in results")
        return
    
    model_results = results[model_name]
    scenarios = []
    scenario_labels = []
    
    # Extract scenario information and create readable labels
    # Sort numerically by the number prefix (0_, 1_, 2_, etc.)
    def sort_key(scenario_key):
        if scenario_key[0].isdigit():
            return int(scenario_key.split('_')[0])
        return 999  # Put non-numbered scenarios at the end
    
    for scenario_key in sorted(model_results.keys(), key=sort_key):
        if 'default' in model_results[scenario_key]:
            scenarios.append(scenario_key)
            # Create readable labels from scenario keys
            if scenario_key.startswith(('0_', '1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_')):
                # Remove number prefix and format nicely
                number = scenario_key.split('_')[0]
                label_text = scenario_key.split('_', 1)[1].replace('_', ' ').title()
                label = f"{number}. {label_text}"
            else:
                label = scenario_key.replace('_', ' ').title()
            scenario_labels.append(label)
    
    # Reverse to show 0 at top, 10 at bottom
    scenarios.reverse()
    scenario_labels.reverse()
    
    y_positions = range(len(scenarios))
    
    for behavior in ['violate', 'deceive']:
        x_values = []
        
        for scenario_key in scenarios:
            if 'default' in model_results[scenario_key]:
                data = model_results[scenario_key]['default'][behavior]
                if data:
                    percentage = np.mean(data) * 100
                    x_values.append(percentage)
                else:
                    x_values.append(0)
            else:
                x_values.append(0)
        
        label = 'Y_violate (Violation Rate)' if behavior == 'violate' else 'Y_deceive (Strategic Deception)'
        ax.plot(x_values, y_positions, 'o', color=colors[behavior], markersize=8, label=label)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(scenario_labels)
    ax.set_xlabel('% responses match behavior')
    ax.set_ylabel('Corporate Decision Scenarios')
    ax.set_xlim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Corporate Decision Scenario Results - {model_name}')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {filename}")
    maybe_show(show)

def plot_default_results(results, filename='results/figures/default.png', show: bool = False):
    """Plot basic results - automatically detects if comparing models or scenarios."""
    ensure_directory_exists(filename)
    # Check if we have multiple models or multiple scenarios
    if len(results) == 1:
        # Single model, multiple scenarios
        model_name = list(results.keys())[0]
        # Skip 'figures' directory if it got loaded as a model
        if model_name == 'figures':
            print("Skipping 'figures' directory")
            return
        plot_scenario_results(results, model_name, filename, show=show)
    else:
        # Filter out 'figures' directory and other non-model entries
        valid_models = {k: v for k, v in results.items() if k != 'figures' and v}
        if len(valid_models) == 1:
            # Single valid model
            model_name = list(valid_models.keys())[0]
            plot_scenario_results({model_name: valid_models[model_name]}, model_name, filename, show=show)
        else:
            # Multiple models - use original logic but fix labels
            colors = {'violate': '#1f77b4', 'deceive': '#ff7f0e', 'persist': '#d62728'}
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            models = list(results.keys())
            y_positions = range(len(models))
            
            for behavior in ['violate', 'deceive']:
                x_values = []
                
                for model_name in models:
                    # For multi-model comparison, look for default scenario
                    default_found = False
                    total_violations = 0
                    total_experiments = 0
                    
                    # Aggregate across all scenarios for this model
                    for scenario_key in results[model_name].keys():
                        if 'default' in results[model_name][scenario_key]:
                            data = results[model_name][scenario_key]['default'][behavior]
                            if data:
                                total_violations += sum(data)
                                total_experiments += len(data)
                                default_found = True
                    
                    if default_found and total_experiments > 0:
                        percentage = (total_violations / total_experiments) * 100
                        x_values.append(percentage)
                    else:
                        x_values.append(0)
                
                label = 'Y_violate' if behavior == 'violate' else 'Y_deceive'
                ax.plot(x_values, y_positions, 'o', color=colors[behavior], markersize=8, label=label)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(models)
            ax.set_xlabel('% responses match behavior')
            ax.set_ylabel('Models')
            ax.set_xlim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {filename}")
            maybe_show(show)

def plot_category_variations(results, category, model_name=None, filename=None, show: bool = False):
    """Plot variations within a specific category using academic terminology."""
    if model_name is None:
        model_name = next(iter(results.keys()))
    
    if model_name not in results:
        raise ValueError(f"Model '{model_name}' not found")
    
    if category not in results[model_name]:
        raise ValueError(f"Category '{category}' not found")
    
    colors = {'violate': '#1f77b4', 'deceive': '#ff7f0e'}
    
    variations = results[model_name][category]
    variation_names = list(variations.keys())
    y_positions = range(len(variation_names))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for behavior in ['violate', 'deceive']:
        x_values = []
        
        for variation_name in variation_names:
            data = variations[variation_name][behavior]
            if data:
                percentage = np.mean(data) * 100
                x_values.append(percentage)
            else:
                x_values.append(0)
        
        label = 'Y_violate' if behavior == 'violate' else 'Y_deceive'
        ax.plot(x_values, y_positions, 'o', color=colors[behavior], markersize=8, label=label)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variation_names)
    ax.set_xlabel('% responses match behavior')
    ax.set_ylabel('Variations')
    ax.set_xlim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'{model_name} - {category}')
    
    plt.tight_layout()
    
    if filename is None:
        filename = f'{category}_{model_name}.png'
    ensure_directory_exists(filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {filename}")
    maybe_show(show)

def plot_models_for_category(results, category_key, filename='results/figures/models_by_category.png', show: bool = False):
    """Compare multiple models for a specific category (scenario or scenario_variable).

    Aggregates across all variations in the category (or uses 'default' if present).
    """
    ensure_directory_exists(filename)
    colors = {'violate': '#1f77b4', 'deceive': '#ff7f0e'}

    models = [m for m in results.keys() if m != 'figures']
    if not models:
        print("No models found to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    y_positions = range(len(models))

    for behavior in ['violate', 'deceive']:
        x_values = []
        for model_name in models:
            model_results = results.get(model_name, {})
            cat = model_results.get(category_key)
            if not cat:
                x_values.append(0)
                continue
            # Aggregate over all variations; if only 'default' exists, this just uses that
            values = []
            for variation_name, variation_data in cat.items():
                values.extend(variation_data.get(behavior, []))
            percentage = (sum(values) / len(values) * 100) if values else 0
            x_values.append(percentage)

        label = 'Y_violate' if behavior == 'violate' else 'Y_deceive'
        ax.plot(x_values, y_positions, 'o', color=colors[behavior], markersize=8, label=label)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(models)
    ax.set_xlabel('% responses match behavior')
    ax.set_ylabel('Models')
    ax.set_xlim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {filename}")
    maybe_show(show)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots for experiment results')
    parser.add_argument('--results-folder', type=str, default='results',
                       help='Path to results folder (default: results)')
    parser.add_argument('--model', type=str, default=None,
                       help='Single model folder name to plot (on-disk name)')
    parser.add_argument('--models', type=str, default=None,
                       help="Comma-separated model aliases or 'all' to include multiple")
    parser.add_argument('--category', type=str, default=None,
                       help='Specific category key to plot (e.g., 0_insider_trading or 0_insider_trading_countermeasures)')
    parser.add_argument('--scenario', type=str, default=None,
                       help='Scenario name without numeric prefix (e.g., insider_trading)')
    parser.add_argument('--variable', type=str, default=None,
                       help='Variable within scenario (e.g., default, countermeasures, reasoning, pressure, ...)')
    parser.add_argument('--show', action='store_true',
                       help='Display plots interactively (off by default for jobs)')
    
    args = parser.parse_args()

    # Load full results set; we'll subset in memory for multi-model cases
    results = load_results(args.results_folder, args.model)

    # Resolve models filter if provided
    def resolve_models(models_arg: str | None, single_model: str | None, available_models: list[str]) -> list[str]:
        if models_arg:
            if models_arg.strip().lower() == 'all':
                return available_models
            models = []
            for m in models_arg.split(','):
                m = m.strip()
                if not m:
                    continue
                # Map alias to on-disk folder name, default to provided
                models.append(MODEL_ALIASES.get(m, m))
            return models
        if single_model:
            return [single_model]
        return available_models

    available_models = [m for m in results.keys() if m != 'figures']
    selected_models = resolve_models(args.models, args.model, available_models)

    # Filter results to selected models
    if selected_models and set(selected_models) != set(available_models):
        results = {m: results[m] for m in selected_models if m in results}

    # Derive category key if scenario/variable are provided
    category_key = args.category
    if not category_key and args.scenario:
        candidates = set()
        # gather all keys from first available model
        first_model = next(iter(results.keys()), None)
        if first_model:
            candidates = set(results[first_model].keys())
        scen = args.scenario.lower().strip()
        var = (args.variable or 'default').lower().strip()

        def strip_prefix(key: str) -> str:
            return re.sub(r'^\d+_', '', key)

        # Try exact build
        target = scen if var == 'default' else f"{scen}_{var}"
        for c in candidates:
            if strip_prefix(c).lower() == target:
                category_key = c
                break

    # Routing: if a category_key is specified and multiple models selected, do model-comparison plot
    if category_key and len(results.keys()) > 1:
        plot_models_for_category(results, category_key, show=args.show)
    elif category_key:
        # Single-model variations plot for that category
        single_model = next(iter(results.keys()), None)
        plot_category_variations(results, category_key, single_model, show=args.show)
    else:
        # Fallback: default behavior
        plot_default_results(results, show=args.show)