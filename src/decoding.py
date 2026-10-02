from enum import Enum, auto
import numpy as np
from .vocab import VocabIndex
from llm_sdk import Small_LLM_Model


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

def select_next_token(logits: list[float], valid_ids: set[int]) -> int:
    """Pick the highest-scoring token id, restricted to valid_ids.

    Args:
        logits: The raw logits for every token in the vocabulary.
        valid_ids: The token ids allowed at this generation step.

    Returns:
        The id of the highest-scoring token among valid_ids.
    """
    masked_logits = np.empty(len(logits))
    masked_logits.fill(-np.inf)
    valid_ids = list(valid_ids)
    logits_array = np.array(logits)
    masked_logits[valid_ids] = logits_array[valid_ids]
    next_token = int(masked_logits.argmax())

    return next_token

def generate_number(model: Small_LLM_Model, vocab: VocabIndex, input_ids: list[int], terminator_id: int) -> tuple[float, list[int]]:
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
        A tuple of the parsed float value and the updated input_ids.
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
            valid_ids = {vocab.dot_id}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            gen += vocab.id_to_str[next_token]
            state = NumberState.AFTER_DOT
        elif state == NumberState.AFTER_DIGIT:
            logits = model.get_logits_from_input_ids(input_ids)
            if len(gen) >= max_digits:
                valid_ids = {vocab.dot_id}
            else:
                valid_ids = vocab.digits_id | {vocab.dot_id}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            gen += vocab.id_to_str[next_token]
            if next_token == vocab.dot_id:
                state = NumberState.AFTER_DOT
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

    result: tuple[float, list[int]] = (float(gen), input_ids)

    return result


def generate_string(model: Small_LLM_Model, vocab: VocabIndex, input_ids: list[int]) -> tuple[str, list[int]]:
    """Generate a JSON-valid string via constrained decoding.

    Caps the content length (max_length) to guarantee termination
    even if the model never favors closing the string on its own.

    Args:
        model: The language model used to obtain logits.
        vocab: Precomputed vocabulary lookups.
        input_ids: The token ids generated so far.

    Returns:
        A tuple of the generated string content (without the
        surrounding quotes) and the updated input_ids.
    """
    state = StringState.INSIDE
    gen = ""
    max_str_len = 80

    while state != StringState.CLOSED:
        if state == StringState.INSIDE:
            logits = model.get_logits_from_input_ids(input_ids)
            if len(gen) >= max_str_len:
                valid_ids = {vocab.structural_ids['"']}
            else:
                valid_ids = vocab.string_content_ids | {vocab.structural_ids['"']}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            if next_token == vocab.structural_ids['"']:
                state = StringState.CLOSED
            else:
                gen += vocab.id_to_str[next_token]

    result: tuple[str, list[int]] = (gen, input_ids)

    return result

def generate_boolean(model: Small_LLM_Model, vocab: VocabIndex, input_ids: list[int]) -> tuple[bool, list[int]]:
    logits = model.get_logits_from_input_ids(input_ids)
    valid_ids = {vocab.true_id} | {vocab.false_id}
    next_token = select_next_token(logits, valid_ids)
    input_ids.append(next_token)
    if next_token == vocab.true_id:
        gen = True
    else:
        gen = False

    result: tuple[bool, list[int]] = (gen, input_ids)

    return result

def generate_parameters(model: Small_LLM_Model, vocab: VocabIndex, input_ids: list[int], parameters_schema: dict) -> tuple[dict, list[int]]:
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
        parameters_schema: The "parameters" mapping from the chosen
            function's definition, e.g. {"a": {"type": "number"}}.

    Returns:
        A tuple of the generated parameters dict (native Python types)
        and the updated input_ids.
    """
    gen = {}
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
        if spec["type"] == "number":
            value, input_ids = generate_number(model, vocab, input_ids, terminator_id)
        elif spec["type"] == "string":
            input_ids.append(vocab.structural_ids['"'])
            value, input_ids = generate_string(model, vocab, input_ids)
            input_ids.append(terminator_id)
        elif spec["type"] == "boolean":
            value, input_ids = generate_boolean(model, vocab, input_ids)
            input_ids.append(terminator_id)
        gen[key] = value

    return (gen, input_ids)

def generate_output(model: Small_LLM_Model, vocab: VocabIndex, prompt: str, fn_trie: dict[int, dict], fn_definitions: dict[str, dict]) -> tuple[dict, list[int]]:
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

    input_ids: list[int] = vocab.encode_to_ids(prompt)
    
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

    examples = "\n\n".join(f"Request: {definition['description']}\nFunction: {name}" for name, definition in fn_definitions.items())
    context = f"{examples}\n\nRequest: {prompt}\nFunction:"
    
    prompt_ids = vocab.encode_to_ids(context)
    fn_name_str, input_ids = select_function_name(model, input_ids, prompt_ids, fn_trie)
    input_ids.append(vocab.structural_ids['"'])
    input_ids.append(vocab.structural_ids[","])

    input_ids.extend(vocab.forced_sequences_next['"parameters"'])
    input_ids.append(vocab.structural_ids[":"])
    value, input_ids = generate_parameters(model, vocab, input_ids, fn_definitions[fn_name_str]["parameters"])

    output = {}
    output["prompt"] = prompt
    output["name"] = fn_name_str
    output["parameters"] = value

    return (output, input_ids)

def build_function_trie(vocab: VocabIndex, function_names: list[str]) -> dict[int, dict]:
    """Build a token-level trie over candidate function names.

    Args:
        vocab: Precomputed vocabulary lookups, for encoding names.
        function_names: The candidate function names to index.

    Returns:
        The trie's root node (token id -> child node), where a node
        holding "end" marks a complete function name.
    """

    trie = {}
    for name in function_names:
        actual_name = trie
        fn_token_ids = vocab.encode_to_ids(name)
        for token in fn_token_ids:
            if token not in actual_name:
                actual_name[token] = {}
            actual_name = actual_name[token]
        actual_name["end"] = name

    return trie

def select_function_name(model: Small_LLM_Model, input_ids: list[int], prompt_ids: list[int], trie:dict[int, dict]) -> tuple[str, list[int]]:
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
        valid_ids = {token_id for token_id in actual_node.keys() if isinstance(token_id, int) }
        next_token = select_next_token(logits, valid_ids)
        prompt_ids.append(next_token)
        input_ids.append(next_token)
        actual_node = actual_node[next_token]

    return (actual_node["end"], input_ids)













