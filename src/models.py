from typing import Literal, Union, Dict, List, TypeAlias
from pydantic import BaseModel, field_validator, model_validator, ValidationError
import json


ParamType: TypeAlias = Literal["number", "string", "boolean"]


class ParameterSpec(BaseModel):
    type: ParamType


class ReturnSpec(BaseModel):
    type: ParamType


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    returns: ReturnSpec


class TestPrompt(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Union[str, float, bool]]

            
            
            






