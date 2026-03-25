#!/usr/bin/env python3
"""Generate config files and sbatch submission script for the full two-turn-nc matrix.

Turn 1 models: 0.6nt, 1.7nt, 4nt, 8nt, 14nt, 0.6t, 1.7t
Turn 2 models: 0.6nt, 1.7nt, 4nt, 8nt, 14nt, 0.6t, 1.7t, 4t, 8t, 14t

Filtering rules:
- Same-mode pairs (nt->nt or t->t): turn 2 size must be >= turn 1 size
- Cross-mode pairs (nt->t or t->nt): all combinations valid
- Existing completed runs are excluded
"""

import os
import yaml
from pathlib import Path

# Model abbreviation -> (HF name, enable_thinking)
MODEL_MAP = {
    "0.6nt": ("Qwen/Qwen3-0.6B", False),
    "1.7nt": ("Qwen/Qwen3-1.7B", False),
    "4nt": ("Qwen/Qwen3-4B", False),
    "8nt": ("Qwen/Qwen3-8B", False),
    "14nt": ("Qwen/Qwen3-14B", False),
    "0.6t": ("Qwen/Qwen3-0.6B", True),
    "1.7t": ("Qwen/Qwen3-1.7B", True),
    "4t": ("Qwen/Qwen3-4B", True),
    "8t": ("Qwen/Qwen3-8B", True),
    "14t": ("Qwen/Qwen3-14B", True),
}

TURN1_MODELS = ["0.6nt", "1.7nt", "4nt", "8nt", "14nt", "0.6t", "1.7t"]
TURN2_MODELS = ["0.6nt", "1.7nt", "4nt", "8nt", "14nt", "0.6t", "1.7t", "4t", "8t", "14t"]

# Size ordering for same-mode filtering
SIZE_ORDER = {"0.6": 0, "1.7": 1, "4": 2, "8": 3, "14": 4}


def parse_model(abbr: str) -> tuple[str, str]:
    """Parse abbreviation into (size, mode). E.g. '1.7nt' -> ('1.7', 'nt')."""
    if abbr.endswith("nt"):
        return abbr[:-2], "nt"
    else:
        return abbr[:-1], "t"


def is_valid_pair(t1: str, t2: str) -> bool:
    """Check if turn2 is sensible after turn1.

    - Identical models are excluded (no point retrying with same model).
    - Same-mode (nt->nt or t->t): turn2 size must be > turn1 size.
    - Cross-mode (nt->t or t->nt): always valid.
    """
    if t1 == t2:
        return False
    size1, mode1 = parse_model(t1)
    size2, mode2 = parse_model(t2)
    if mode1 == mode2:
        return SIZE_ORDER[size2] > SIZE_ORDER[size1]
    return True

# All 21 existing runs to skip (10 good + 11 already-rerun timing-bug runs)
EXISTING_RUNS = {
    ("0.6nt", "0.6t"),
    ("0.6nt", "1.7nt"),
    ("0.6nt", "1.7t"),
    ("0.6nt", "4t"),
    ("0.6nt", "8t"),
    ("1.7nt", "4t"),
    ("1.7nt", "8t"),
    ("4nt", "4t"),
    ("8nt", "4t"),
    ("8nt", "8t"),
    ("1.7nt", "1.7t"),
    ("1.7nt", "14t"),
    ("4nt", "1.7t"),
    ("4nt", "8t"),
    ("4nt", "14t"),
    ("8nt", "1.7t"),
    ("8nt", "14t"),
    ("14nt", "1.7t"),
    ("14nt", "4t"),
    ("14nt", "8t"),
    ("14nt", "14t"),
}


def generate_config(turn1_abbr: str, turn2_abbr: str, port1: int, port2: int) -> dict:
    """Generate a two-turn config dict."""
    model1, thinking1 = MODEL_MAP[turn1_abbr]
    model2, thinking2 = MODEL_MAP[turn2_abbr]

    # Extract short model size for benchmark name
    size1 = model1.split("-")[-1].lower()
    size2 = model2.split("-")[-1].lower()

    return {
        "servers": [
            {
                "model": model1,
                "port": port1,
                "tp": 1,
                "mem_fraction": 0.95,
                "device": 0,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": thinking1,
                    }
                },
            },
            {
                "model": model2,
                "port": port2,
                "tp": 1,
                "mem_fraction": 0.95,
                "device": 1,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": thinking2,
                    }
                },
            },
        ],
        "benchmarks": [
            {
                "name": f"humaneval_twoturn_{size1}_{size2}",
                "scenario_file": "benchmarks/HumanEval/Tasks/human_eval_AgentChatCopyNoCtx.jsonl",
                "server_port": port1,
                "config_template": "scripts/configs/config-twoturn-template.yaml",
                "server_port_map": {
                    "model_config_turn1": port1,
                    "model_config_turn2": port2,
                },
                "repeat": 2,
            }
        ],
        "server_startup_timeout": 600,
        "server_health_check_interval": 5,
        "shutdown_timeout": 30,
    }


def main():
    script_dir = Path(__file__).parent
    config_dir = script_dir / "configs"
    config_dir.mkdir(exist_ok=True)

    runs = []
    port_counter = 31001  # Start from 31001 to avoid conflicts with existing configs

    for t1 in TURN1_MODELS:
        for t2 in TURN2_MODELS:
            if not is_valid_pair(t1, t2):
                continue
            if (t1, t2) in EXISTING_RUNS:
                continue

            port1 = port_counter
            port2 = port_counter + 1
            port_counter += 2

            output_dir = f"outputs/two-turn-nc-{t1}-{t2}"

            config = generate_config(t1, t2, port1, port2)
            config_name = f"config-twoturn-nc-{t1}-{t2}.yaml"
            config_path = config_dir / config_name

            # Write config with comment header
            header = (
                f"\n# Two-turn benchmark configuration\n"
                f"# Turn 1: {t1} (port {port1})\n"
                f"# Turn 2: {t2} (port {port2})\n\n"
            )
            with open(config_path, "w") as f:
                f.write(header)
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            runs.append((config_name, output_dir, t1, t2))

    # Generate submission script — submit all jobs at once, let SLURM handle scheduling
    submit_path = script_dir / "submit_full_matrix.sh"
    with open(submit_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all two-turn-nc full matrix runs\n")
        f.write(f"# Total runs: {len(runs)}\n")
        f.write("# Generated by generate_twoturn_full_matrix.py\n\n")
        f.write("set -e\n\n")

        for idx, (config_name, output_dir, t1, t2) in enumerate(runs, 1):
            f.write(f'echo "[{idx}/{len(runs)}] Submitting {t1} -> {t2}"\n')
            f.write(f'sbatch scripts/start.sh "scripts/configs/{config_name}" "{output_dir}"\n')

        f.write(f'\necho "\\nAll {len(runs)} jobs submitted."\n')

    os.chmod(submit_path, 0o755)

    print(f"Generated {len(runs)} config files in {config_dir}/")
    print(f"Generated submission script: {submit_path}")
    # Count filtered pairs
    total_pairs = len(TURN1_MODELS) * len(TURN2_MODELS)
    invalid_pairs = sum(1 for t1 in TURN1_MODELS for t2 in TURN2_MODELS if not is_valid_pair(t1, t2))
    valid_existing = sum(1 for t1, t2 in EXISTING_RUNS if is_valid_pair(t1, t2))

    print(f"\nBreakdown:")
    print(f"  Total matrix: {total_pairs}")
    print(f"  Skipped (turn2 < turn1 in same mode): {invalid_pairs}")
    print(f"  Skipped (existing): {valid_existing}")
    print(f"  New runs: {len(runs)}")
    print(f"\nTo submit all jobs:")
    print(f"  cd python/packages/agbench && bash scripts/submit_full_matrix.sh")


if __name__ == "__main__":
    main()
