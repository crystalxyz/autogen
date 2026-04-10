import os
import json
import sys

from bleu import smoothed_bleu_4


def main(args):
    """Tabulate CodeXGlue code-to-text results using smoothed BLEU-4."""
    if len(args) < 2:
        print("Usage: python custom_tabulate.py <results_dir>")
        sys.exit(1)

    results_dir = args[1]
    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    references = []
    predictions = []
    failures = 0
    total = 0

    for task_dir in sorted(os.listdir(results_dir)):
        result_path = os.path.join(results_dir, task_dir, "result.json")
        if not os.path.isfile(result_path):
            continue

        total += 1
        with open(result_path, "r") as f:
            result = json.load(f)

        prediction = result.get("prediction", "").strip()
        reference = result.get("reference", "").strip()

        if not prediction:
            failures += 1
            continue

        references.append(reference)
        predictions.append(prediction)

    avg_bleu = smoothed_bleu_4(references, predictions)
    print(f"\n{'='*50}")
    print(f"CodeXGlue Code-to-Text Results")
    print(f"{'='*50}")
    print(f"Total tasks:      {total}")
    print(f"Scored:            {len(references)}")
    print(f"Failed/empty:      {failures}")
    print(f"Average BLEU-4:    {avg_bleu:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__" and __package__ is None:
    main(sys.argv)
