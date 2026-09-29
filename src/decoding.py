from enum import Enum, auto
import numpy as np
from .vocab import VocabIndex
from llm_sdk import Small_LLM_Model


class StructuralState(Enum):
    EXPECT_OPEN_BRACE = auto()
    EXPECT_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    EXPECT_COMMA_OR_CLOSE = auto()
    DONE = auto()


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

def generate_number(model: Small_LLM_Model, vocab: VocabIndex, input_ids: list[int]) -> tuple[float, list[int]]:
    """Generate a JSON-valid number via constrained decoding.

    Caps the number of digits generated (max_digits) to guarantee
    termination even if the model never favors stopping on its own.

    Args:
        model: The language model used to obtain logits.
        vocab: Precomputed vocabulary lookups.
        input_ids: The token ids generated so far.

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
                valid_ids = {vocab.structural_ids[","]} | {vocab.structural_ids["}"]}
            else:
                valid_ids = vocab.digits_id | {vocab.structural_ids[","]} | {vocab.structural_ids["}"]}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            if next_token in (vocab.structural_ids["}"], vocab.structural_ids[","]):
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



