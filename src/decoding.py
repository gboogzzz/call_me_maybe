from enum import Enum, auto
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np

from .vocab import VocabIndex
from llm_sdk import Small_LLM_Model  # type: ignore


TrieNode = Dict[Any, Any]
ParamValue = Union[int, float, str, bool]


class NumberState(Enum):
    START = auto()
    AFTER_MINUS = auto()
    AFTER_ZERO = auto()
    AFTER_DIGIT = auto()
    AFTER_DOT = auto()
    AFTER_DECIMAL_DIGIT = auto()
    NUMBER_DONE = auto()


class StringState(Enum):
    INSIDE = auto()
    CLOSED = auto()


def select_next_token(logits: List[float], valid_ids: Set[int]) -> int:
    """Pick the highest-scoring token id, restricted to valid_ids.

    Args:
        logits: The raw logits for every token in the vocabulary.
        valid_ids: The token ids allowed at this generation step.

    Returns:
        The id of the highest-scoring token among valid_ids.
    """
    masked_logits = np.empty(len(logits))
    masked_logits.fill(-np.inf)
    valid_ids_list = list(valid_ids)
    logits_array = np.array(logits)
    masked_logits[valid_ids_list] = logits_array[valid_ids_list]
    next_token = int(masked_logits.argmax())

    return next_token


def generate_number(
    model: Small_LLM_Model,
    vocab: VocabIndex,
    input_ids: List[int],
    terminator_id: int,
) -> Tuple[Union[int, float], List[int]]:
    """Generate a JSON-valid number via constrained decoding.

    Caps the number of digits generated (max_digits) to guarantee
    termination even if the model never favors stopping on its own.

    Args:
        model: The language model used to obtain logits.
        vocab: Precomputed vocabulary lookups.
        input_ids: The token ids generated so far.
        terminator_id: The single token id ("," or "}") that must follow
            this number, determined by the caller from the parameter
            schema position (not a choice left to the model).

    Returns:
        A tuple of the parsed value (int if no decimal point was
        generated, float otherwise) and the updated input_ids.
    """
    state = NumberState.START
    gen = ""
    max_digits = 15

    while state != NumberState.NUMBER_DONE:
        if state == NumberState.START:
            logits = model.get_logits_from_input_ids(input_ids)
            valid_ids = vocab.digits_id | {vocab.minus_id}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            gen += vocab.id_to_str[next_token]
            if next_token == vocab.minus_id:
                state = NumberState.AFTER_MINUS
            elif next_token == vocab.zero_id:
                state = NumberState.AFTER_ZERO
            else:
                state = NumberState.AFTER_DIGIT
        elif state == NumberState.AFTER_MINUS:
            logits = model.get_logits_from_input_ids(input_ids)
            valid_ids = vocab.digits_id
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            gen += vocab.id_to_str[next_token]
            if next_token == vocab.zero_id:
                state = NumberState.AFTER_ZERO
            else:
                state = NumberState.AFTER_DIGIT
        elif state == NumberState.AFTER_ZERO:
            logits = model.get_logits_from_input_ids(input_ids)
            valid_ids = {vocab.dot_id} | {terminator_id}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            if next_token == terminator_id:
                state = NumberState.NUMBER_DONE
            else:
                gen += vocab.id_to_str[next_token]
                state = NumberState.AFTER_DOT
        elif state == NumberState.AFTER_DIGIT:
            logits = model.get_logits_from_input_ids(input_ids)
            if len(gen) >= max_digits:
                valid_ids = {vocab.dot_id} | {terminator_id}
            else:
                valid_ids = vocab.digits_id | {vocab.dot_id} | {terminator_id}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            if next_token == terminator_id:
                state = NumberState.NUMBER_DONE
            elif next_token == vocab.dot_id:
                gen += vocab.id_to_str[next_token]
                state = NumberState.AFTER_DOT
            else:
                gen += vocab.id_to_str[next_token]
        elif state == NumberState.AFTER_DOT:
            logits = model.get_logits_from_input_ids(input_ids)
            valid_ids = vocab.digits_id
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            gen += vocab.id_to_str[next_token]
            state = NumberState.AFTER_DECIMAL_DIGIT
        elif state == NumberState.AFTER_DECIMAL_DIGIT:
            logits = model.get_logits_from_input_ids(input_ids)
            if len(gen) >= max_digits:
                valid_ids = {terminator_id}
            else:
                valid_ids = vocab.digits_id | {terminator_id}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            if next_token == terminator_id:
                state = NumberState.NUMBER_DONE
            else:
                gen += vocab.id_to_str[next_token]

    value: Union[int, float] = float(gen) if "." in gen else int(gen)
    result: Tuple[Union[int, float], List[int]] = (value, input_ids)

    return result


def generate_string(
    model: Small_LLM_Model,
    vocab: VocabIndex,
    input_ids: List[int],
    prompt_ids: List[int],
) -> Tuple[str, List[int]]:
    """Generate a JSON-valid string via constrained decoding.

    Caps the content length (max_length) to guarantee termination
    even if the model never favors closing the string on its own.

    Args:
        model: The language model used to obtain logits.
        vocab: Precomputed vocabulary lookups.
        input_ids: The token ids generated so far.
        prompt_ids: Clean, separate decoding context used only to
            decide this string's content (not polluted by the
            surrounding JSON structure being built in input_ids).

    Returns:
        A tuple of the generated string content (without the
        surrounding quotes) and the updated input_ids.
    """
    state = StringState.INSIDE
    gen = ""
    max_str_len = 80
    quote_id = vocab.structural_ids['"']

    while state != StringState.CLOSED:
        if state == StringState.INSIDE:
            logits = model.get_logits_from_input_ids(prompt_ids)
            if len(gen) >= max_str_len:
                valid_ids = {quote_id}
            else:
                valid_ids = vocab.string_content_ids | {quote_id}
            next_token = select_next_token(logits, valid_ids)
            prompt_ids.append(next_token)
            input_ids.append(next_token)
            if next_token == quote_id:
                state = StringState.CLOSED
            else:
                gen += vocab.id_to_str[next_token]

    result: Tuple[str, List[int]] = (gen, input_ids)

    return result


def generate_boolean(
    model: Small_LLM_Model,
    vocab: VocabIndex,
    input_ids: List[int],
) -> Tuple[bool, List[int]]:
    """Generate a JSON-valid boolean via constrained decoding.

    Args:
        model: The language model used to obtain logits.
        vocab: Precomputed vocabulary lookups.
        input_ids: The token ids generated so far.

    Returns:
        A tuple of the generated boolean value and the updated input_ids.
    """
    logits = model.get_logits_from_input_ids(input_ids)
    valid_ids = {vocab.true_id} | {vocab.false_id}
    next_token = select_next_token(logits, valid_ids)
    input_ids.append(next_token)
    if next_token == vocab.true_id:
        gen = True
    else:
        gen = False

    result: Tuple[bool, List[int]] = (gen, input_ids)

    return result


def generate_parameters(
    model: Small_LLM_Model,
    vocab: VocabIndex,
    input_ids: List[int],
    prompt: str,
    parameters_schema: Dict[str, Any],
) -> Tuple[Dict[str, ParamValue], List[int]]:
    """Generate a JSON-valid parameters object via constrained decoding.

    Iterates the function's parameter schema in order, forcing each
    known key literal and delegating each value to the generator that
    matches its declared type (number, string, or boolean). The comma
    or closing brace after each value is either produced internally
    (generate_number) or forced here, depending on the generator.

    Args:
        model: The language model used to obtain logits.
        vocab: Precomputed vocabulary lookups.
        input_ids: The token ids generated so far.
        prompt: The original natural-language request, used to build
            each string parameter's clean decoding context.
        parameters_schema: The "parameters" mapping from the chosen
            function's definition, e.g. {"a": {"type": "number"}}.

    Returns:
        A tuple of the generated parameters dict (native Python types)
        and the updated input_ids.
    """
    gen: Dict[str, ParamValue] = {}
    total = len(parameters_schema)

    input_ids.append(vocab.structural_ids["{"])
    for i, (key, spec) in enumerate(parameters_schema.items()):
        literal_key = f'"{key}":' if i == 0 else f' "{key}":'
        key_ids = vocab.encode_to_ids(literal_key)
        input_ids.extend(key_ids)
        is_last = (i == total - 1)
        if is_last:
            terminator_id = vocab.structural_ids["}"]
        else:
            terminator_id = vocab.structural_ids[","]

        value: ParamValue
        if spec["type"] == "number":
            value, input_ids = generate_number(
                model, vocab, input_ids, terminator_id
            )
        elif spec["type"] == "string":
            input_ids.append(vocab.structural_ids['"'])
            history = "".join(f'{k}: "{v}"\n' for k, v in gen.items())
            context = f"{prompt}\n" + history + f'{key}: "'
            prompt_ids = vocab.encode_to_ids(context)
            value, input_ids = generate_string(
                model, vocab, input_ids, prompt_ids
            )
            input_ids.append(terminator_id)
        elif spec["type"] == "boolean":
            value, input_ids = generate_boolean(model, vocab, input_ids)
            input_ids.append(terminator_id)
        gen[key] = value

    return (gen, input_ids)


def generate_output(
    model: Small_LLM_Model,
    vocab: VocabIndex,
    prompt: str,
    fn_trie: TrieNode,
    fn_definitions: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[int]]:
    """Generate the full JSON output entry for one prompt.

    Args:
        model: The language model used to obtain logits.
        vocab: Precomputed vocabulary lookups.
        prompt: The original natural-language request.
        fn_trie: The function-name trie from build_function_trie.
        fn_definitions: All function definitions, keyed by name.

    Returns:
        A tuple of the output dict and the updated input_ids.
    """

    input_ids: List[int] = vocab.encode_to_ids(prompt)

    input_ids.append(vocab.structural_ids["{"])
    input_ids.extend(vocab.forced_sequences_first['"prompt"'])
    input_ids.append(vocab.structural_ids[":"])
    prompt_str = f' "{prompt}"'
    prompt_str_ids = vocab.encode_to_ids(prompt_str)
    input_ids.extend(prompt_str_ids)
    input_ids.append(vocab.structural_ids[","])

    input_ids.extend(vocab.forced_sequences_next['"name"'])
    input_ids.append(vocab.structural_ids[":"])
    input_ids.extend(vocab.encode_to_ids(' "'))

    examples = "\n\n".join(
        f"Request: {definition['description']}\nFunction: {name}"
        for name, definition in fn_definitions.items()
    )
    context = f"{examples}\n\nRequest: {prompt}\nFunction:"

    prompt_ids = vocab.encode_to_ids(context)
    fn_name_str, input_ids = select_function_name(
        model, input_ids, prompt_ids, fn_trie
    )
    input_ids.append(vocab.structural_ids['"'])
    input_ids.append(vocab.structural_ids[","])

    input_ids.extend(vocab.forced_sequences_next['"parameters"'])
    input_ids.append(vocab.structural_ids[":"])
    value, input_ids = generate_parameters(
        model,
        vocab,
        input_ids,
        prompt,
        fn_definitions[fn_name_str]["parameters"],
    )

    output: Dict[str, Any] = {}
    output["prompt"] = prompt
    output["name"] = fn_name_str
    output["parameters"] = value

    return (output, input_ids)


def build_function_trie(
    vocab: VocabIndex,
    function_names: List[str],
) -> TrieNode:
    """Build a token-level trie over candidate function names.

    Args:
        vocab: Precomputed vocabulary lookups, for encoding names.
        function_names: The candidate function names to index.

    Returns:
        The trie's root node (token id -> child node), where a node
        holding "end" marks a complete function name.
    """

    trie: TrieNode = {}
    for name in function_names:
        actual_name = trie
        fn_token_ids = vocab.encode_to_ids(name)
        for token in fn_token_ids:
            if token not in actual_name:
                actual_name[token] = {}
            actual_name = actual_name[token]
        actual_name["end"] = name

    return trie


def select_function_name(
    model: Small_LLM_Model,
    input_ids: List[int],
    prompt_ids: List[int],
    trie: TrieNode,
) -> Tuple[str, List[int]]:
    """Select a function name via constrained decoding over a trie.

    Walks the trie one token at a time, restricting each choice to
    the current node's child tokens, until a node marks the end of a
    complete function name.

    Args:
        model: The language model used to obtain logits.
        input_ids: The full output's token ids generated so far.
        prompt_ids: Clean, growing prompt context used for the decision.
        trie: The function-name trie from build_function_trie.

    Returns:
        A tuple of the selected function name and the updated input_ids.
    """

    actual_node = trie

    while "end" not in actual_node:
        logits = model.get_logits_from_input_ids(prompt_ids)
        valid_ids = {
            token_id
            for token_id in actual_node.keys()
            if isinstance(token_id, int)
        }
        next_token = select_next_token(logits, valid_ids)
        prompt_ids.append(next_token)
        input_ids.append(next_token)
        actual_node = actual_node[next_token]

    return (actual_node["end"], input_ids)
