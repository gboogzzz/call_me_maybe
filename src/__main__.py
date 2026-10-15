from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model
from .loaders import load_function_definitions, load_test_prompts
from .vocab import VocabIndex
from .decoding import generate_output, build_function_trie
import os
import json
import argparse


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")

    args = parser.parse_args()

    fn_definitions = load_function_definitions(args.functions_definition)
    test_prompts = load_test_prompts(args.input)

    fn_dict: Dict[str, Dict[str, Any]] = {}
    for fn in fn_definitions:
        fn_dict[fn.name] = fn.model_dump()

    tp_list: List[str] = []
    for tp in test_prompts:
        tp_list.append(tp.prompt)

    fn_list = list(fn_dict.keys())

    model = Small_LLM_Model()
    vocab = VocabIndex(model)
    fn_trie = build_function_trie(vocab, fn_list)
    list_outputs: List[Dict[str, Any]] = []

    for i, prompt_text in enumerate(tp_list):
        print(f"Prompt {i}:\n {prompt_text}\n")
        output, _ = generate_output(model, vocab, prompt_text, fn_trie, fn_dict)
        list_outputs.append(output)
        print(f"Output Prompt {i}:\n{output}\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(list_outputs, f, indent=2)


if __name__ == "__main__":
    main()
