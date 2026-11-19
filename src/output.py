import argparse
import json
import re

from .monitor import Monitor
from .schemas import Response
from pydantic import TypeAdapter


def escape_match(match: re.Match[str]) -> str:
    slashes = match.group(1)
    nxt = match.group(2)
    if len(slashes) % 2 == 1:
        if nxt in '\\"/bfnrt' or (
            nxt == 'u' and re.match(
                r'u[0-9a-fA-F]{4}', match.group(0)[len(slashes):]
                )
                ):
            return slashes + nxt
        return slashes + '\\' + nxt
    return slashes + nxt


def generate_response(m: Monitor) -> str:
    full_response = m.e.model.decode(m.completeJSON)
    sanitized = re.sub(
        r'(\\+)(u[0-9a-fA-F]{4}|.)',
        escape_match,
        full_response,
        flags=re.DOTALL
        )
    sanitized = re.sub(
        r'(":\s*"(?:\\.|[^"\\])*)\n(\s*\})', r'\1"\n\2', sanitized
        )

    return sanitized


def generate_JSON(args: argparse.Namespace, response: str) -> None:
    try:
        json_output = TypeAdapter(list[Response]).validate_json(response)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(
                [item.model_dump() for item in json_output],
                file,
                indent=2,
                ensure_ascii=False
                )
        print(f"JSON saved in {args.output}")
    except Exception as e:
        print(f"Error: Cannot generate JSON. Reason: {e}")
        exit(1)
