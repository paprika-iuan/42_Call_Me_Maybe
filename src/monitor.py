import json
import os

from enum import Enum
from typing import List, Any, cast
from pydantic import TypeAdapter

from llm_sdk import Small_LLM_Model    # type: ignore
from .schemas import InputPrompts, InputFunctions
from .encoder import Encoder


class Response_Stage(Enum):
    EMPTY = 0
    ADD_NAME = 1
    ADD_PARAMETERS = 2


class Monitor:
    def __init__(self, encoder: Encoder):
        self.prompts: Any = {}
        self.functions: dict[tuple[int, ...], list[list[int]]] = {}
        self.e: Encoder = encoder

        self.stage = Response_Stage.EMPTY

        self.completeJSON: list[int] = self.e.encodeVar("[\n")

        self.current_func: list[int] = []
        self.func_dict: dict[str, Any] = {}
        self.func_prefix: list[int] = []
        self.inBracket: bool = False

    def _build_functon_dict(self, functions: List[InputFunctions]) -> None:
        for fn in functions:
            encoded_name = tuple(self.e.encodeVar(fn.name))
            encoded_params = [
                self.e.encodeVar(param_name) for param_name
                in fn.parameters.keys()
            ]
            self.functions[encoded_name] = encoded_params

    def validate_input_functions(self, args: Any) -> list[dict[str, Any]]:
        try:
            with open(args.functions_def, "r", encoding="utf-8") as file:
                function_data = json.load(file)
                adapter = TypeAdapter(List[InputFunctions])
                validated = adapter.validate_python(function_data)
                self._build_functon_dict(validated)
                return cast(list[dict[str, Any]], function_data)
        except Exception as e:
            print(f"Error: Missing file or invalid function file. Detail: {e}")
            exit(1)

    def validate_input_prompts(self, args: Any) -> str:
        try:
            with open(args.input, "r", encoding="utf-8") as file:
                prompt_data = json.load(file)
                adapter = TypeAdapter(List[InputPrompts])
                self.prompts = adapter.validate_python(prompt_data)
                return cast(str, prompt_data)
        except Exception as e:
            print(f"Error: Missing file or invalid prompt file. Detail: {e}")
            exit(1)

    def build_context(self,
                      model: Small_LLM_Model,
                      input: str | None,
                      is_last: bool) -> list[int]:
        match self.stage:
            case Response_Stage.EMPTY:
                self.stage = Response_Stage.ADD_NAME
                clean_input = input.replace('"', '\\"') if input else ""
                res = cast(list[int], self.e.stage_prompt[0]
                        + model.encode(clean_input).squeeze(0).tolist()
                        + self.e.stage_prompt[1])
                print(f"\033[1;35m{model.decode(res)}\033[0m",
                                      end="", flush=True)
                return (res)

            case Response_Stage.ADD_NAME:
                self.stage = Response_Stage.ADD_PARAMETERS
                print(
                    f"\033[1;35m{model.decode(self.e.stage_prompt[2])}\033[0m",
                    end="", flush=True)
                return self.e.stage_prompt[2]

            case Response_Stage.ADD_PARAMETERS:
                self.stage = Response_Stage.EMPTY
                return self.e.param_end if is_last else self.e.param_endl

    def is_stage_done(self, token_id: int, results: list[int]) -> bool:
        match self.stage:
            case Response_Stage.ADD_NAME:
                if token_id == self.e.delimiter_quote[0]:
                    return True

            case Response_Stage.ADD_PARAMETERS:
                if self.is_delimiter(token_id, self.e.delimiter_bracket):
                    format_numeric_param(self.e, token_id,  results)
                    return True

        return False

    def append_param_key(self, current_param: int, results: list[int]) -> None:
        param_tokens = self.functions[tuple(self.current_func)][current_param]
        results.extend(self.e.delimiter_quote
                       + param_tokens
                       + self.e.delimiter_quote
                       + self.e.param_del)

    def is_delimiter(self, token_id: int, delimiters: list[list[int]]) -> bool:
        return token_id in [
                delimiter[0]
                for delimiter in delimiters
                if delimiter
        ]

    def build_func_dict(self, sys_functs: list[dict[str, Any]]) -> None:
        self.func_dict = {
                function["name"]: function
                for function in sys_functs
            }

    def prefix_finder(self) -> None:
        prefix = os.path.commonprefix(
                        list(self.func_dict.keys())
                        )
        self.func_prefix = self.e.encodeVar(prefix) if prefix else []
        self.e.stage_prompt[1].extend(self.func_prefix[:-1])

    def is_bracket(self, token: int) -> None:
        self.e.model.decode()


def format_numeric_param(encoder: Encoder,
                         token_id: int,
                         results: list[int]) -> None:
    delimiter_str = encoder.model.decode([token_id])

    if '"' in delimiter_str or "'" in delimiter_str:
        return

    is_delim_appended = (results and results[-1] == token_id)

    val_idx = -2 if is_delim_appended else -1

    if len(results) >= abs(val_idx) and [results[val_idx]] in encoder.numbers:
        dot_idx = val_idx - 1
        if len(results) >= abs(dot_idx):
            if "." in encoder.model.decode([results[dot_idx]]):
                return

        if is_delim_appended:
            delim = results.pop()
            results.extend(encoder.encodeVar(".0"))
            results.append(delim)
        else:
            results.extend(encoder.encodeVar(".0"))
