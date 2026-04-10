#
# Generates synthetic RULER benchmark tasks at various context lengths.
#

import json
import os
import random
import re
import string

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")

# Context lengths to generate (in approximate tokens; 1 token ~ 4 chars)
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768]
EXAMPLES_PER_LENGTH = 50

# Filler sentences for padding context
FILLER_TOPICS = [
    "The weather in various regions changes throughout the year depending on geographical factors.",
    "International trade agreements have shaped economic policies in many countries over the decades.",
    "Advances in renewable energy technology continue to drive changes in the global power sector.",
    "The history of architecture reveals how societies prioritize function, aesthetics, and durability.",
    "Marine biology research has uncovered many species previously unknown to the scientific community.",
    "Urbanization trends indicate that more people are moving to cities in search of opportunities.",
    "The development of transportation networks has been crucial for economic growth and connectivity.",
    "Agricultural innovations have significantly increased food production capacity worldwide.",
    "Digital communication technologies have transformed how people interact across distances.",
    "Geological surveys help scientists understand the composition and history of the Earth's crust.",
    "The study of ancient civilizations provides insights into human cultural and social development.",
    "Public health initiatives have played a key role in reducing the spread of infectious diseases.",
    "Space exploration missions continue to expand our understanding of the solar system and beyond.",
    "Philosophical debates about consciousness and free will have persisted for centuries.",
    "Music theory encompasses the study of harmony, rhythm, melody, and the structure of compositions.",
    "The principles of aerodynamics are fundamental to the design of aircraft and other vehicles.",
    "Biodiversity conservation efforts aim to protect ecosystems and the species that inhabit them.",
    "Financial markets operate based on complex interactions between supply, demand, and investor sentiment.",
    "Advances in materials science have led to the creation of stronger, lighter, and more versatile products.",
    "The evolution of programming languages reflects changing needs in software development and computing.",
]

NOUNS = [
    "apple", "bridge", "castle", "dolphin", "engine", "forest", "guitar", "harbor",
    "island", "jungle", "lantern", "mountain", "nebula", "ocean", "piano", "quartz",
    "river", "sunset", "tower", "umbrella", "valley", "window", "zenith", "basket",
    "candle", "dragon", "eagle", "falcon", "glacier", "helmet", "ivory", "jasmine",
    "kite", "lemon", "marble", "napkin", "orchid", "pebble", "ribbon", "saddle",
    "temple", "violet", "walnut", "anchor", "beacon", "cipher", "dagger", "emerald",
]

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nancy", "Oscar", "Patricia",
    "Quinn", "Rachel", "Steven", "Teresa", "Ulysses", "Victoria", "Walter", "Yvonne",
]


def generate_filler(target_chars: int) -> str:
    """Generate filler text of approximately target_chars length."""
    lines = []
    current = 0
    while current < target_chars:
        line = random.choice(FILLER_TOPICS)
        lines.append(line)
        current += len(line) + 1
    return "\n".join(lines)


def generate_niah_single(context_length: int, seed: int) -> dict:
    """Needle In A Haystack — single key-value."""
    rng = random.Random(seed)
    key = rng.choice(NOUNS)
    value = rng.choice([n for n in NOUNS if n != key])
    needle = f"The special key is '{key}' and its corresponding value is '{value}'."
    target_chars = context_length * 4
    filler_before = generate_filler(rng.randint(target_chars // 4, 3 * target_chars // 4))
    filler_after = generate_filler(target_chars - len(filler_before) - len(needle))
    context = filler_before + "\n" + needle + "\n" + filler_after
    prompt = f"{context}\n\nQuestion: What is the value corresponding to the key '{key}'? Answer with just the value, nothing else."
    return {"prompt": prompt, "answer": value, "task_type": "niah_single"}


def generate_niah_multikey(context_length: int, seed: int, num_keys: int = 4) -> dict:
    """Needle In A Haystack — multiple key-value pairs, query one."""
    rng = random.Random(seed)
    keys = rng.sample(NOUNS, num_keys)
    values = rng.sample([n for n in NOUNS if n not in keys], num_keys)
    pairs = list(zip(keys, values))
    needles = [f"The special key is '{k}' and its corresponding value is '{v}'." for k, v in pairs]

    target_chars = context_length * 4
    # Distribute needles throughout
    segments = []
    chars_per_segment = target_chars // (num_keys + 1)
    for i, needle in enumerate(needles):
        segments.append(generate_filler(chars_per_segment))
        segments.append(needle)
    segments.append(generate_filler(chars_per_segment))

    context = "\n".join(segments)
    query_idx = rng.randint(0, num_keys - 1)
    query_key = keys[query_idx]
    answer = values[query_idx]

    prompt = f"{context}\n\nQuestion: What is the value corresponding to the key '{query_key}'? Answer with just the value, nothing else."
    return {"prompt": prompt, "answer": answer, "task_type": "niah_multikey"}


def generate_niah_multivalue(context_length: int, seed: int, num_values: int = 4) -> dict:
    """Needle In A Haystack — one key with multiple values."""
    rng = random.Random(seed)
    key = rng.choice(NOUNS)
    values = rng.sample([n for n in NOUNS if n != key], num_values)
    needles = [f"One value for the key '{key}' is '{v}'." for v in values]

    target_chars = context_length * 4
    segments = []
    chars_per_segment = target_chars // (num_values + 1)
    for needle in needles:
        segments.append(generate_filler(chars_per_segment))
        segments.append(needle)
    segments.append(generate_filler(chars_per_segment))

    context = "\n".join(segments)
    answer = ", ".join(sorted(values))

    prompt = f"{context}\n\nQuestion: What are all the values for the key '{key}'? List them in alphabetical order, separated by commas. Answer with just the values, nothing else."
    return {"prompt": prompt, "answer": answer, "task_type": "niah_multivalue"}


def generate_variable_tracking(context_length: int, seed: int, chain_length: int = 5) -> dict:
    """Variable tracking through assignment chains."""
    rng = random.Random(seed)
    vars_used = rng.sample(NAMES, chain_length + 1)
    final_value = rng.choice(NOUNS)

    # Build chain: X1 = final_value, X2 = X1, X3 = X2, ...
    assignments = [f"{vars_used[0]} = {final_value}"]
    for i in range(1, len(vars_used)):
        assignments.append(f"{vars_used[i]} = {vars_used[i-1]}")

    # Shuffle and embed in filler
    rng.shuffle(assignments)
    target_chars = context_length * 4
    segments = []
    chars_per_segment = target_chars // (len(assignments) + 1)
    for assignment in assignments:
        segments.append(generate_filler(chars_per_segment))
        segments.append(f"Assignment: {assignment}")
    segments.append(generate_filler(chars_per_segment))

    context = "\n".join(segments)
    query_var = vars_used[-1]

    prompt = f"{context}\n\nQuestion: After all assignments are executed, what is the value of {query_var}? Answer with just the value, nothing else."
    return {"prompt": prompt, "answer": final_value, "task_type": "variable_tracking"}


def generate_common_words(context_length: int, seed: int, num_lists: int = 3, common_count: int = 2) -> dict:
    """Find words common to all embedded lists."""
    rng = random.Random(seed)
    common_words = rng.sample(NOUNS, common_count)
    lists = []
    for _ in range(num_lists):
        unique = rng.sample([n for n in NOUNS if n not in common_words], rng.randint(5, 8))
        word_list = common_words + unique
        rng.shuffle(word_list)
        lists.append(word_list)

    list_strings = [f"Word list: {', '.join(wl)}" for wl in lists]
    target_chars = context_length * 4
    segments = []
    chars_per_segment = target_chars // (num_lists + 1)
    for ls in list_strings:
        segments.append(generate_filler(chars_per_segment))
        segments.append(ls)
    segments.append(generate_filler(chars_per_segment))

    context = "\n".join(segments)
    answer = ", ".join(sorted(common_words))

    prompt = f"{context}\n\nQuestion: Which words appear in ALL of the word lists above? List them in alphabetical order, separated by commas. Answer with just the words, nothing else."
    return {"prompt": prompt, "answer": answer, "task_type": "common_words"}


def generate_frequent_words(context_length: int, seed: int) -> dict:
    """Find the most frequently mentioned word in text."""
    rng = random.Random(seed)
    target_word = rng.choice(NOUNS)
    other_words = rng.sample([n for n in NOUNS if n != target_word], 10)

    target_chars = context_length * 4
    mentions = []
    # Target word appears 10 times, others 1-3 times each
    for _ in range(10):
        mentions.append(f"The word '{target_word}' was mentioned in the report.")
    for w in other_words:
        for _ in range(rng.randint(1, 3)):
            mentions.append(f"The word '{w}' was mentioned in the report.")

    rng.shuffle(mentions)
    segments = []
    chars_per_segment = target_chars // (len(mentions) + 1)
    for m in mentions:
        segments.append(generate_filler(chars_per_segment))
        segments.append(m)
    segments.append(generate_filler(chars_per_segment))

    context = "\n".join(segments)
    prompt = f"{context}\n\nQuestion: Which single word was mentioned the most times in the text above? Answer with just the word, nothing else."
    return {"prompt": prompt, "answer": target_word, "task_type": "frequent_words"}


TASK_GENERATORS = {
    "niah_single": generate_niah_single,
    "niah_multikey": generate_niah_multikey,
    "niah_multivalue": generate_niah_multivalue,
    "variable_tracking": generate_variable_tracking,
    "common_words": generate_common_words,
    "frequent_words": generate_frequent_words,
}


def create_jsonl(name: str, tasks: list, template: str):
    """Creates a JSONL scenario file."""
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
        for task in tasks:
            print(f"Converting: [{name}] {task['id']}")
            record = {
                "id": task["id"],
                "template": template,
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": task["prompt"]},
                    "expected_answer.txt": {"__EXPECTED_ANSWER__": task["answer"]},
                    "scenario.py": {"__TASK_TYPE__": task["task_type"]},
                },
            }
            fh.write(json.dumps(record).strip() + "\n")


def main():
    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    if not templates:
        print("No templates found.")
        return

    for task_name, generator in TASK_GENERATORS.items():
        for ctx_len in CONTEXT_LENGTHS:
            ctx_label = f"{ctx_len // 1024}k"
            tasks = []
            for i in range(EXAMPLES_PER_LENGTH):
                seed = hash((task_name, ctx_len, i)) % (2**31)
                result = generator(ctx_len, seed)
                result["id"] = f"RULER_{task_name}_{ctx_label}_{i}"
                tasks.append(result)

            for template_name, template_path in templates.items():
                create_jsonl(f"ruler_{task_name}_{ctx_label}_{template_name}", tasks, template_path)

    print(f"\nDone! Task files created in: {TASKS_DIR}")


if __name__ == "__main__" and __package__ is None:
    main()
