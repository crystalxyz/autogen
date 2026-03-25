"""
Scenario expansion and utility functions for agbench.
"""

import errno
import json
import logging
import os
import pathlib
import random
import re
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import yaml
from typing_extensions import TypedDict

# Constants
DEFAULT_ENV_FILE_JSON = "ENV.json"
DEFAULT_ENV_FILE_YAML = "ENV.yaml"
DEFAULT_CONFIG_YAML = "config.yaml"


# Get the path to the template directory
BASE_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template")


class ScenarioInstance(TypedDict):
    id: str
    template: Union[str, List[Union[str, List[str]]]]
    substitutions: Dict[str, Dict[str, str]]
    values: Dict[str, Dict[str, str]]


def expand_scenario(
    scenario_dir: str, scenario: ScenarioInstance, output_dir: str, config_file: Union[str, None]
) -> None:
    """
    Expand a scenario into a folder.
    It is a series of copy commands (similar to `cp -R`), followed by a series of in-place fine and replace operations.
    """

    template = scenario["template"]

    # Either key works for finding the substiturions list. "values" may be deprecated in the future
    substitutions = scenario["substitutions"] if "substitutions" in scenario else scenario["values"]

    # Older versions are only one-level deep. Convert them,
    if len(substitutions) > 0 and isinstance(substitutions[next(iter(substitutions))], str):
        substitutions = {"scenario.py": cast(Dict[str, str], substitutions)}

    copy_operations: List[Tuple[str, str]] = []

    # Handle file (str), folder (str), or mapping (List) templates
    if isinstance(template, str):
        template_path = os.path.join(scenario_dir, template)
        if os.path.isdir(template_path):
            copy_operations.append((template, ""))
        else:
            copy_operations.append((template, "scenario.py"))
    elif isinstance(template, list):
        for elm in template:
            if isinstance(elm, list):
                copy_operations.append((elm[0], elm[1]))
            else:
                copy_operations.append((elm, ""))
    else:
        raise ValueError("expand_scenario expects an str or list for 'template'")

    # The global includes folder is always copied
    shutil.copytree(
        BASE_TEMPLATE_PATH,
        output_dir,
        ignore=shutil.ignore_patterns("*.example"),
        dirs_exist_ok=False,
    )

    # Expand other folders
    for items in copy_operations:
        src_path = pathlib.Path(os.path.join(scenario_dir, items[0])).absolute()
        dest_path = pathlib.Path(os.path.join(output_dir, items[1])).absolute()

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            if os.path.isdir(dest_path):
                # If the destination is a directory, use the same filename
                shutil.copyfile(src_path, os.path.join(dest_path, os.path.basename(src_path)))
            else:
                # Otherwuse use the filename provided
                shutil.copyfile(src_path, dest_path)

    # Expand templated files
    for templated_file in substitutions.keys():  # Keys are relative file paths
        # Read the templated file into memory
        template_contents: List[str] = list()
        with open(os.path.join(output_dir, templated_file), "rt") as fh:
            for line in fh:
                template_contents.append(line)

        # Rewrite the templated file with substitutions
        values = substitutions[templated_file]
        with open(os.path.join(output_dir, templated_file), "wt") as fh:
            for line in template_contents:
                for k, v in values.items():
                    line = line.replace(k, v)
                fh.write(line)

    # Copy the config
    if config_file is None:
        if os.path.isfile(DEFAULT_CONFIG_YAML):
            config_file = DEFAULT_CONFIG_YAML

    if config_file is not None:
        src_path = pathlib.Path(config_file).absolute()
        dest_path = pathlib.Path(os.path.join(output_dir, "config.yaml")).absolute()
        shutil.copyfile(src_path, dest_path)
    else:
        logging.warning(f"No {DEFAULT_CONFIG_YAML} file found.")


def get_scenario_env(token_provider: Optional[Callable[[], str]] = None, env_file: Optional[str] = None) -> Dict[str, str]:
    """
    Return a dictionary of environment variables needed to run a scenario.

    Args:
        token_provider: Token provider function for Azure auth
        env_file (str): The path to the env_file to read. (if None, default to DEFAULT_ENV_FILE)

    Returns: A dictionary of keys and values that need to be added to the system environment.
    """
    env: Dict[str, str] = dict()

    # Populate with commonly needed keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is not None and len(openai_api_key.strip()) > 0:
        env["OPENAI_API_KEY"] = openai_api_key

    ## Support Azure auth tokens
    azure_openai_ad_token = os.environ.get("AZURE_OPENAI_AD_TOKEN")
    if azure_openai_ad_token is None and token_provider is not None:
        azure_openai_ad_token = token_provider()
    if azure_openai_ad_token is not None and len(azure_openai_ad_token.strip()) > 0:
        env["AZURE_OPENAI_AD_TOKEN"] = azure_openai_ad_token

    # Update with any values from the ENV.json file
    env_file_contents: Dict[str, Any] = {}
    if env_file is None:
        # Env file was not specified, so read the default, or warn if the default file is missing.
        if os.path.isfile(DEFAULT_ENV_FILE_YAML):
            with open(DEFAULT_ENV_FILE_YAML, "r") as fh:
                env_file_contents = yaml.safe_load(fh)
        elif os.path.isfile(DEFAULT_ENV_FILE_JSON):
            with open(DEFAULT_ENV_FILE_JSON, "rt") as fh:
                env_file_contents = json.loads(fh.read())
            logging.warning(f"JSON environment files are deprecated. Migrate to '{DEFAULT_ENV_FILE_YAML}'")
        else:
            logging.warning(
                f"The environment file '{DEFAULT_ENV_FILE_YAML}' was not found. A default environment will be provided, containing the keys: {env.keys()}"
            )
    else:
        # Env file was specified. Throw an error if the file can't be read.
        with open(env_file, "rt") as fh:
            if env_file.endswith(".json"):
                logging.warning("JSON environment files are deprecated. Migrate to YAML")
                env_file_contents = json.loads(fh.read())
            else:
                env_file_contents = yaml.safe_load(fh)

    # Apply substitutions in-place
    substitute_env_variables(env_file_contents)

    # Flatten any structures
    for key, value in env_file_contents.items():
        if isinstance(value, dict) or isinstance(value, list):
            env_file_contents[key] = json.dumps(value)

    # Warn about carrying env variables
    if "OPENAI_API_KEY" in env and "OPENAI_API_KEY" not in env_file_contents:
        logging.warning(
            f"Implicit inclusion of OPENAI_API_KEY in the task environment is deprecated. Add it to {DEFAULT_ENV_FILE_YAML} instead. E.g.,\n"
            + """

OPENAI_API_KEY: ${OPENAI_API_KEY}

"""
        )

    # Apply the loaded variables
    env.update(cast(Dict[str, str], env_file_contents))

    return env


def substitute_env_variables(json_data: Any) -> None:
    """
    Recursively replaces any instance of "${ENV_VARIABLE}" with os.environ("ENV_VARIABLE") in a structure returned from json.loads()
    """

    def replace_env_var(match: Any) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    pattern = re.compile(r"\$\{(\w+)\}")

    def replace_in_dict(d: Dict[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, str):
                d[key] = pattern.sub(replace_env_var, value)
            elif isinstance(value, dict):
                replace_in_dict(cast(Dict[str, Any], value))
            elif isinstance(value, list):
                # Note: with the task mypy complains of a redundant cast
                # without the cast, pyright complains the type is unknown
                replace_in_list(cast(List[Any], value))  # type: ignore

    def replace_in_list(lst: List[Any]) -> None:
        for i, item in enumerate(lst):
            if isinstance(item, str):
                lst[i] = pattern.sub(replace_env_var, item)
            elif isinstance(item, dict):
                replace_in_dict(cast(Dict[str, Any], item))
            elif isinstance(item, list):
                replace_in_list(cast(List[Any], item))  # type: ignore

    if isinstance(json_data, dict):
        replace_in_dict(cast(Dict[str, Any], json_data))
    elif isinstance(json_data, list):
        replace_in_list(cast(List[Any], json_data))  # type: ignore


def find_autogen_repo(path: str) -> Optional[str]:
    """
    Utility for identifying if the path is a subdirectory of the autogen_core repo.

    Returns: the path to the root of the autogen_core repo if one is found, otherwise None
    """

    # Normalize the path (we expect a directory)
    path = os.path.abspath(path)
    if os.path.isfile(path):
        path = os.path.dirname(path)

    while True:
        test_path = os.path.join(path, "python", "packages", "autogen-core")  # We found autogen_core
        if os.path.isdir(test_path):
            return path

        # Stop if we hit the root
        parent_dir = os.path.abspath(os.path.join(path, os.pardir))
        if parent_dir == path:
            break

        # Keep searching
        path = parent_dir

    return None


def prepare_apptainer_binds(env: Dict[str, str]) -> List[str]:
    """
    Prepare bind mounts for Apptainer instance.

    Args:
        env: Environment variables dictionary (unused, kept for compatibility)

    Returns:
        List of bind mount strings in the format "host_path:container_path"
    """
    binds = []

    # Add the autogen repo if we can find it
    autogen_repo_base = os.environ.get("AUTOGEN_REPO_BASE")
    if autogen_repo_base is None:
        autogen_repo_base = find_autogen_repo(os.getcwd())
    elif not os.path.isdir(autogen_repo_base):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), autogen_repo_base)

    if autogen_repo_base is None:
        raise ValueError(
            "Could not find AutoGen repo base. Please set the environment variable AUTOGEN_REPO_BASE to the correct value."
        )

    autogen_repo_base = os.path.join(autogen_repo_base, "python")
    autogen_repo_abs = str(pathlib.Path(autogen_repo_base).absolute())
    binds.append(f"{autogen_repo_abs}:/autogen_python")

    # Also bind the home directory so Results and workspace are accessible
    home_dir = os.path.expanduser("~")
    binds.append(f"{home_dir}:{home_dir}")

    return binds


def split_jsonl(file_path: str, num_parts: int) -> List[List[Dict[str, Any]]]:
    """
    Split a JSONL file into num_parts approximately equal parts.
    """
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)  # Shuffle the data for better distribution
    chunk_size = len(data) // num_parts
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def mkdir_p(path: str) -> None:
    """
    Create a directory if it doesn't exist, handling race conditions.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def discover_scenario_files(scenario: str) -> List[str]:
    """
    Discover scenario files from a path (file, directory, or stdin).

    Args:
        scenario: Path to scenario file, directory, or "-" for stdin

    Returns:
        List of scenario file paths to process
    """
    files: List[str] = []

    if scenario == "-" or os.path.isfile(scenario):
        files.append(scenario)
    elif os.path.isdir(scenario):
        for f in os.listdir(scenario):
            scenario_file = os.path.join(scenario, f)

            if not os.path.isfile(scenario_file):
                continue

            if not scenario_file.lower().endswith(".jsonl"):
                continue

            files.append(scenario_file)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), scenario)

    return files


def get_scenario_name_and_dir(scenario_file: str) -> Tuple[str, str]:
    """
    Extract scenario name and directory from a scenario file path.

    Args:
        scenario_file: Path to scenario file or "-" for stdin

    Returns:
        Tuple of (scenario_name, scenario_dir)
    """
    if scenario_file == "-":
        return "stdin", "."
    else:
        scenario_name_parts = os.path.basename(scenario_file).split(".")
        scenario_name_parts.pop()
        scenario_name = ".".join(scenario_name_parts)
        scenario_dir = os.path.dirname(os.path.realpath(scenario_file))
        return scenario_name, scenario_dir
