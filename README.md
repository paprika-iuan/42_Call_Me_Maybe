*This project has been created as part of the 42 curriculum by jgirbau-.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling system that translates natural-language
requests into structured JSON function calls. Instead of answering a request
directly, the program selects the most appropriate function from a supplied
function definition file and extracts its arguments with the required types.

The project uses the `Qwen/Qwen3-0.6B` language model through the provided
`llm_sdk`. Because small language models do not reliably produce valid
machine-readable output on their own, this project uses constrained decoding
to guide generation token by token. The result is a JSON array containing the
original prompt, the selected function name, and its parameters.

For example, the prompt `What is the sum of 2 and 3?` should produce a call
similar to:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

## Instructions

### Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/)
- Enough disk space and memory for the Qwen3-0.6B model and its tokenizer

The project includes the required `llm_sdk/` package. Dependencies are
declared in `pyproject.toml` and locked in `uv.lock`.

### Installation

Clone the repository and install its dependencies:

```bash
git clone https://github.com/paprika-iuan/42_Call_Me_Maybe.git
cd 42_Call_Me_Maybe
make install
```

The equivalent direct command is:

```bash
uv sync
```

### Execution

Run the program with the demonstration files:

```bash
make run
```

This is equivalent to:

```bash
uv run python -m src
```

The default input files are:

- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

The default output is written to `data/output/function_calls.json`.
Custom paths can be supplied with the command-line options:

```bash
uv run python -m src \
  --functions_def data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Additional Makefile commands are available:

```bash
make debug        # Run through Python's pdb debugger
make lint         # Run flake8 and the configured mypy checks
make lint-strict  # Run flake8 and mypy --strict
make clean        # Remove generated caches, output, and local SDK caches
```

## Input Files

`data/input/function_calling_tests.json` is an array of objects containing
natural-language prompts:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet john"}
]
```

`data/input/functions_definition.json` describes the available functions.
Each function contains a `name`, `description`, `parameters`, and `returns`
field. The current schema supports `string` and `number` parameter and return
types:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

Both files are loaded as JSON and validated with Pydantic before generation.
Invalid JSON, missing files, or schema violations are reported as input
errors.

## Algorithm Explanation

The generation pipeline is implemented in `src/__main__.py`, `src/encoder.py`,
`src/monitor.py`, `src/stages.py`, and `src/decoder.py`.

1. `Monitor` loads and validates the input prompts and function definitions.
   Function names and parameter names are encoded with the model tokenizer so
   they can be used as token sequences during generation.
2. `Encoder` builds a system prompt containing the available tools and defines
   the fixed JSON fragments used by the generation stages.
3. For each user prompt, the model receives the system context and the prompt
   and is asked to produce a structured object with `prompt`, `name`, and
   `parameters`.
4. The response is generated in stages. The monitor first identifies the
   function name, then switches to the selected function's definition and
   generates its parameter values.
5. Before each token is selected, the model returns logits for the next token.
   `constrained_decoding` removes forbidden delimiters, applies temperature,
   top-k, and top-p sampling, and selects a remaining token.
6. Structural tokens such as quotes, commas, colons, and closing braces are
   inserted or recognized by the monitor. Numeric values receive `.0` when
   necessary so they remain valid JSON numbers.
7. The completed response is sanitized, parsed with `json.loads`, and written
   as formatted JSON. Invalid output is reported instead of being silently
   accepted.

The important constraint is applied before token selection: tokens that would
introduce forbidden delimiters or break the expected JSON construction are
assigned negative infinity logits. This prevents the model from escaping into
Markdown fences or unrelated prose while it is generating the call.

## Design Decisions

- **Pydantic validation:** Input prompts and function definitions are represented
  by Pydantic models, keeping file validation close to the data contract.
- **Stage-based generation:** Separating function-name and parameter generation
  lets the program rebuild a shorter system prompt for the selected function.
  This reduces context length during parameter extraction.
- **Tokenizer-level constraints:** Function and parameter names are compared as
  token sequences rather than as plain strings, which matches the model's
  actual generation unit.
- **Explicit JSON delimiters:** Reusable encoded delimiters centralize the JSON
  structure and avoid relying on the model to reproduce punctuation correctly.
- **Sampling controls:** Temperature, top-k, and top-p are kept in the decoder
  so the selection policy is separate from the state machine that constructs
  the response.
- **Local model execution:** The included SDK loads Qwen3 locally through the
  required project interface and automatically selects CPU, CUDA, or MPS when
  available.

## Performance Analysis

- **Reliability:** Constrained decoding rejects forbidden token paths before
  selection, and the final response is parsed as JSON before it is saved. This
  makes syntactic validity substantially more reliable than unconstrained
  prompting alone.
- **Accuracy:** The subject target is at least 90% correct function selection
  and argument extraction. Accuracy depends on the model, prompt wording,
  function definitions, and sampling settings; the included input set is
  intended as a reproducible demonstration rather than a benchmark claim.
- **Speed:** Generation performs a model forward pass for each selected token,
  so runtime is primarily determined by model inference, prompt length, and
  hardware. The implementation reduces repeated context by using the selected
  function's schema while generating parameters. The subject's target is to
  process the test set in under five minutes on standard hardware.
- **Memory:** Model and tokenizer files are cached by the Makefile under
  `~/goinfre/call-me-maybe` when the standard 42 environment is used.

## Challenges Faced

- **Small-model JSON reliability:** Free generation can add prose, Markdown, or
  malformed punctuation. The decoder filters unsafe delimiter tokens and the
  monitor controls the JSON stages.
- **Tokenizer behavior:** A visible word is not necessarily one token, and
  tokens may include leading spaces or punctuation. Function and parameter
  names are therefore encoded and compared using token sequences.
- **Variable parameter schemas:** Different functions have different parameter
  counts and types. Once a function is selected, the monitor uses its encoded
  parameter list to emit the expected keys in order.
- **Context length:** Repeating all tool definitions while generating every
  parameter is expensive. Rebuilding the system prompt for the selected
  function keeps later inference contexts smaller.

## Testing Strategy

The demonstration inputs cover:

- Multiple function selections
- Integer-like numeric values and square-root inputs
- String extraction and reversal
- Functions with one, two, and three parameters
- Regex-related replacement prompts
- Prompts containing quoted text and special characters

Before considering a run successful, verify that:

1. The program exits with a clear error for missing or malformed input files.
2. `data/output/function_calls.json` exists after a successful run.
3. The output parses with a standard JSON parser.
4. Every result contains exactly `prompt`, `name`, and `parameters`.
5. Each function name and parameter key exists in the supplied definitions.
6. Parameter values use the types declared by the selected function.

Useful checks include:

```bash
make lint
uv run python -m json.tool data/output/function_calls.json
```

The input JSON files can be replaced during review to test different function
sets and prompts; the implementation is not intended to hardcode the supplied
examples.

## Example Usage

Run the complete sample:

```bash
make install
make run
cat data/output/function_calls.json
```

Run with custom files:

```bash
uv run python -m src \
  --functions_def ./my_functions.json \
  --input ./my_prompts.json \
  --output ./results/function_calls.json
```

For the prompt `Reverse the string 'hello'`, a valid result has this shape:

```json
{
  "prompt": "Reverse the string 'hello'",
  "name": "fn_reverse_string",
  "parameters": {
    "s": "hello"
  }
}
```

## Resources

- [42 subject: Call Me Maybe](./en.subject.pdf)
- [Qwen documentation](https://qwen.readthedocs.io/)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html)
- [Transformers text-generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)

### AI Usage

AI tools were used as a documentation and review aid for this project. They
helped summarize the 42 subject requirements, organize the README sections,
explain constrained decoding and token-level logits, and compare the project
structure with similar public README examples. The implementation details,
commands, file names, and behavior documented here were checked against this
repository's source files and sample data.