from llm_sdk import Small_LLM_Model  # type: ignore
from typing import Any, cast
import json


class Encoder:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        self.sys_prompt_prev: list[int] = self.encodeVar(
            """System: You are a helpful assistant that translates
            user requests into structured function calls.
            You have access to the following functions:
            <tools>"""
            )

        self.sys_prompt_after: list[int] = self.encodeVar(
            """</tools>
            Given a user request, select the appropriate function and
            extract its parameters.
            Respond ONLY with a valid JSON object with keys "prompt",
            "name", and "parameters".
            """
            )

        self.delimiter_quote = self.encodeVar("\"")

        self.delimiter_param = [
            self.encodeVar("\","),
            # self.encodeVar(","),
            self.encodeVar("\'\","),
            self.encodeVar("}")
            ]

        self.delimiter_bracket = [
            self.encodeVar("}\n"),
            self.encodeVar(" }"),
            # self.encodeVar("} "),
            self.encodeVar(" }\n"),
            self.encodeVar("\"}\n"),
            self.encodeVar("}}\n"),
            self.encodeVar("\"}}"),
            self.encodeVar("}}"),
            self.encodeVar("\"}}\n"),
            self.encodeVar("\"}")
            # self.encodeVar("}")
            ]

        self.param_endl = self.encodeVar("}\n},\n")
        self.param_end = self.encodeVar("}\n}\n")

        self.param_del = self.encodeVar(":")

        self.delimiter_bad = [
            self.encodeVar("```"),
            self.encodeVar("```json"),
            self.encodeVar("```\n"),
            self.encodeVar("</")
            ]

        self.delimiter_space = self.encodeVar(" ")

        self.stage_prompt = [
            self.encodeVar("{\n\"prompt\": \""),
            self.encodeVar("\",\n\"name\": \""),
            self.encodeVar("\"parameters\": {")
            ]

        self.json_closer = self.encodeVar("]")

        self.numbers = [self.encodeVar(str(i)) for i in range(10)]

    def encodeVar(self, string: str) -> list[int]:
        tokens = self.model.encode(string).squeeze(0).tolist()
        return cast(list[int], tokens)

    def build_system_prompt(self, functions: Any) -> list[int]:
        functions_str = self.encodeVar(json.dumps(functions, indent=2))
        return self.sys_prompt_prev + functions_str + self.sys_prompt_after
