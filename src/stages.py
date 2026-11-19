from .monitor import Monitor
from .encoder import Encoder


def stage_name(m: Monitor,
               len_before: int,
               sys_prompt_len: int,
               results: list[int]) -> int:
    m.current_func = m.func_prefix[:-1] + results[len_before:-1]
    f = m.e.model.decode(m.current_func)
    reduced_system_prompt = m.e.build_system_prompt(m.func_dict[f])
    reduced_prompt_len = len(reduced_system_prompt)
    results[:] = reduced_system_prompt + results[sys_prompt_len:]
    return reduced_prompt_len


def stage_param(e: Encoder, token_id: int, results: list[int]) -> None:
    if '"' in e.model.decode(token_id):
        results.extend(e.delimiter_quote)
