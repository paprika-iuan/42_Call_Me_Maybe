import argparse

from .monitor import Monitor, Response_Stage, format_numeric_param
from .stages import stage_name, stage_param
from llm_sdk import Small_LLM_Model    # type: ignore
from .encoder import Encoder
from .decoder import constrained_decoding
from .output import generate_response, generate_JSON

import numpy as np

import datetime


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='Call Me Maybe',
        description='A function calling tool that translates natural\
            language prompts into structured function calls.',
    )
    parser.add_argument('--functions_def',
                        type=str,
                        default='data/input/functions_definition.json')
    parser.add_argument('--input',
                        type=str,
                        default='data/input/function_calling_tests.json')
    parser.add_argument('--output',
                        type=str,
                        default='data/output/function_calls.json')

    args = parser.parse_args()
    model = Small_LLM_Model()
    print(
        f"\033[1;36mGeneration started at: "
        f"{datetime.datetime.now().strftime('%H:%M:%S')}\033[0m",
        flush=True,
    )
    encoder = Encoder(model)
    m = Monitor(encoder)
    m.validate_input_prompts(args)

    sys_functs = m.validate_input_functions(args)
    sys_prompt = encoder.build_system_prompt(sys_functs)
    sys_prompt_len = len(sys_prompt)
    m.build_func_dict(sys_functs)
    m.prefix_finder()

    results: list[int] = []

    for index, user_prompt in enumerate(m.prompts):
        print(
            f"\033[92m--- Generating Response {index} ---\033[0m",
            flush=True,
        )
        results.clear()
        results.extend(sys_prompt)
        is_last = (index == len(m.prompts) - 1)
        while True:
            results.extend(
                m.build_context(model, user_prompt.prompt, is_last))

            len_before = len(results)
            current_param = 0

            if m.stage == Response_Stage.ADD_PARAMETERS and current_param == 0:
                m.append_param_key(current_param, results)

            if m.stage == Response_Stage.EMPTY:
                break

            for _ in range(50):
                logits = np.array(
                    model.get_logits_from_input_ids(results), dtype=float)
                token_id = constrained_decoding(
                    logits, 0.8, 30, 0.9, encoder.delimiter_bad)
                if token_id == model._tokenizer.eos_token_id:
                    break
                print(f"\033[1;35m{model.decode([token_id])}\033[0m",
                      end="", flush=True)
                if m.is_stage_done(token_id, results):
                    if m.stage == Response_Stage.ADD_NAME:
                        reduced_prompt_len = stage_name(m,
                                                        len_before,
                                                        sys_prompt_len,
                                                        results)

                    elif m.stage == Response_Stage.ADD_PARAMETERS:
                        stage_param(encoder, token_id, results)
                    break

                results.append(token_id)

                if m.is_delimiter(token_id, encoder.delimiter_param):
                    format_numeric_param(encoder, token_id, results)
                    results.extend(encoder.delimiter_space)
                    current_param += 1
                    if current_param < len(m.functions[tuple(m.current_func)]):
                        m.append_param_key(current_param, results)

        m.completeJSON.extend(results[reduced_prompt_len:])
        if index == len(m.prompts) - 1:
            m.completeJSON.extend(encoder.json_closer)

    full_response = generate_response(m)
    print(
        f"\n\033[1;36mGeneration ended at: "
        f"{datetime.datetime.now().strftime('%H:%M:%S')}\033[0m",
        flush=True,
    )
    generate_JSON(args, full_response)


if __name__ == "__main__":
    main()
