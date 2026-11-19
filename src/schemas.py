from typing import Literal, Dict

from pydantic import BaseModel, Field


AllowedTypes = Literal["string", "number", "boolean", "integer"]


class ParamDefinition(BaseModel):
    type: AllowedTypes = Field(...)


class InputPrompts(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=99)


class InputFunctions(BaseModel):
    name: str = Field(..., min_length=5, max_length=80)
    description: str = Field(..., min_length=5, max_length=80)
    parameters: Dict[str, ParamDefinition] = Field(...)
    returns: ParamDefinition = Field(...)


class Response(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | float]
