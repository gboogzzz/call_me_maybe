"""Pydantic models for function definitions, test prompts and results."""

from typing import Literal, Union, Dict, TypeAlias
from pydantic import BaseModel


ParamType: TypeAlias = Literal["number", "integer", "string", "boolean", "null", "array", "object"]


class ParameterSpec(BaseModel):
    """Type description of a single function parameter.

    Attributes:
        type: JSON type expected for the parameter value
            ("number", "string" or "boolean").
    """

    type: ParamType


class ReturnSpec(BaseModel):
    """Type description of a function return value.

    Attributes:
        type: JSON type returned by the function
            ("number", "string" or "boolean").
    """

    type: ParamType


class FunctionDefinition(BaseModel):
    """A function the LLM is allowed to call.

    Mirrors one entry of ``functions_definition.json``.

    Attributes:
        name: Function name (e.g. "fn_add_numbers").
        description: Human-readable description of what the function does.
        parameters: Mapping from parameter name to its type specification.
        returns: Type specification of the function return value.
    """

    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    returns: ReturnSpec


class TestPrompt(BaseModel):
    """A natural-language prompt to translate into a function call.

    Mirrors one entry of ``function_calling_tests.json``.

    Attributes:
        prompt: The original natural-language request.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry of the output file ``function_calling_results.json``.

    Attributes:
        prompt: The original natural-language request.
        name: Name of the function selected by the LLM.
        parameters: Arguments extracted from the prompt, keyed by
            parameter name, with values typed as in the function definition.
    """

    prompt: str
    name: str
    parameters: Dict[str, Union[str, float, bool]]
