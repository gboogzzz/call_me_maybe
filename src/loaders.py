import sys
import json
from json import JSONDecodeError
from pydantic import ValidationError
from .models import FunctionDefinition, TestPrompt


def _read_json_array(path: str) -> list[dict]:
    """Read and parse a JSON file containing a top-level array.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content as a list of dictionaries.
    """
    try:
        with open(path, encoding="utf-8") as f:
            all_data: list[dict] = json.load(f)
    except FileNotFoundError as e:
        print("File not found: ", e)
        sys.exit(1)
    except JSONDecodeError as e:
        print("JSON file malformed:", e)
        sys.exit(1)

    return all_data

def load_function_definitions(
        path: str ="data/input/functions_definition.json"
        ) -> list[FunctionDefinition]:
    """Load and validate the available function definitions.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        A list of validated FunctionDefinition objects.
    """

    fn_definitions: list[dict] = _read_json_array(path)
    fn_validated: list[FunctionDefinition] = []

    try:
        for f in fn_definitions:
            fn = FunctionDefinition.model_validate(f)
            fn_validated.append(fn)
    except ValidationError as e:
        print("Function not defined correctly: ", e)
        sys.exit(1)

    return fn_validated

def load_test_prompts(
        path: str = "data/input/function_calling_tests.json"
        ) -> list[TestPrompt]:
    """Load and validate the natural language test prompts.

    Args:
        path: Path to the test prompts JSON file.

    Returns:
        A list of validated TestPrompt objects.
    """

    t_prompts: list[dict] = _read_json_array(path)
    prompt_validated: list[TestPrompt] = []
    
    try:
        for t in t_prompts:
            prompt = TestPrompt.model_validate(t)
            prompt_validated.append(prompt)
    except ValidationError as e:
        print("Prompt not defined correctly: ", e)
        sys.exit(1)
    
    return prompt_validated



    
    



