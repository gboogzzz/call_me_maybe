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
    AFTER_DIGIT = auto()
    AFTER_DOT = auto()
    NUMBER_DONE = auto()


class StringState(Enum):
    OPEN_QUOTE = auto()
    INSIDE = auto()
    CLOSED = auto()

def select_next_token(logits: list[float], valid_ids: set[int]) -> int:
    masked_logits = np.empty(len(logits))
    masked_logits.fill(-np.inf)
    valid_ids = list(valid_ids)
    logits_array = np.array(logits)
    masked_logits[valid_ids] = logits_array[valid_ids]
    next_token = int(masked_logits.argmax())

    return next_token

def generate_number(model: Small_LLM_Model, vocab: VocabIndex, input_ids: list[int]) -> tuple[float, list[int]]:
    state = NumberState.START
    gen = ""

    while state != NumberState.NUMBER_DONE:
        if state == NumberState.START:
            logits = model.get_logits_from_input_ids(input_ids)
            valid_ids = vocab.digits_id | {vocab.minus_id}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            gen += vocab.id_to_str[next_token]
            state = NumberState.AFTER_DIGIT
        elif state == NumberState.AFTER_DIGIT:
            logits = model.get_logits_from_input_ids(input_ids)
            valid_ids = vocab.digits_id | {vocab.dot_id}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            gen += vocab.id_to_str[next_token]
            if next_token == vocab.dot_id:
                state = NumberState.AFTER_DOT
        elif state == NumberState.AFTER_DOT:
            logits = model.get_logits_from_input_ids(input_ids)
            valid_ids = vocab.digits_id | {vocab.structural_ids[","]} | {vocab.structural_ids["}"]}
            next_token = select_next_token(logits, valid_ids)
            input_ids.append(next_token)
            if next_token in (vocab.structural_ids["}"], vocab.structural_ids[","]):
                state = NumberState.NUMBER_DONE
            else:
                gen += vocab.id_to_str[next_token]

    result: tuple[float, list[int]] = (float(gen), input_ids)

    return result
