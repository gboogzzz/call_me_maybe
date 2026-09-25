*This project has been created as part of the 42 curriculum by <your_login>.*

# call me maybe

## Description

**call me maybe** is a function calling tool for small language models. Given a natural language
prompt (e.g. *"What is the sum of 2 and 3?"*), the program does not answer the question directly.
Instead, it identifies which function should be called and with which arguments, producing a
structured, machine-readable result such as:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The project uses **Qwen/Qwen3-0.6B**, a small (0.6B parameter) language model that is notoriously
unreliable at producing valid structured output through prompting alone. To reach near-perfect
reliability, this project implements **constrained decoding from scratch**: at every generation
step, the model's output logits are masked so that only tokens leading to valid, schema-compliant
JSON can be selected. This guarantees 100% syntactically valid and schema-compliant JSON output,
regardless of the model's raw reliability.

The function to call is always chosen by the model itself (via its logits, restricted to the set
of valid function names) — never through keyword matching, regular expressions or other
heuristics.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- An internet connection on first run, to download the Qwen3-0.6B model and tokenizer files
  (~1.5GB) from the Hugging Face Hub. Subsequent runs use the local cache and require no network
  access.

### Setup

```bash
uv sync
```

### Running

```bash
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```

By default, the program reads:
- `data/input/functions_definition.json` — the available functions and their argument schemas
- `data/input/function_calling_tests.json` — the natural language prompts to process

and writes the results to `data/output/function_calling_results.json`.

Example:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile targets

```bash
make install   # install dependencies via uv
make run       # run the program
make debug     # run the program under pdb
make lint      # run flake8 and mypy
make clean     # remove __pycache__ / .mypy_cache
```

## Algorithm explanation

*(To be completed once the constrained decoding engine is implemented — Phase 4.)*

## Design decisions

- **`uv` workspace for `llm_sdk`.** The provided `llm_sdk` package is wired in as a local `uv`
  workspace member rather than copied loosely, so dependency resolution and imports stay
  consistent with the rest of the project.
- **Pydantic models for all structured data.** `FunctionDefinition`, `ParameterSpec`,
  `ReturnSpec`, `TestPrompt` and `FunctionCallResult` are all pydantic models, giving automatic
  validation of both the input files and the generated output, without hand-written structural
  checks.
- **Separation of file-loading concerns.** JSON/file-system errors (missing file, malformed JSON)
  are handled separately from schema validation errors (pydantic `ValidationError`), so failures
  are diagnosable at the layer where they actually occur.
- *(More to be added as later phases are implemented: vocabulary indexing, the constrained
  decoding state machine, function-name selection via a token trie, and the forced-token
  optimization for fixed JSON structure — see Performance analysis below.)*

## Performance analysis

*(To be completed once the full pipeline is implemented and benchmarked — Phases 4-6.)*

Planned optimization: token sequences that are fixed and known in advance (JSON structural
syntax, the literal output keys `"prompt"`, `"name"`, `"parameters"`, and parameter names once a
function has been selected) are appended directly to the generation context without invoking the
model, since there is no real choice to make at those positions. Model inference
(`get_logits_from_input_ids`) is only invoked at genuine decision points: selecting the function
name, and generating parameter values extracted from the prompt.

## Challenges faced

*(To be completed as the project progresses.)*

## Testing strategy

*(To be completed — Phase 7. Will cover: edge cases such as empty strings, large numbers, special
characters, wrong types and ambiguous prompts; malformed/missing input files; and validation of
JSON well-formedness and schema compliance for every generated output.)*

## Example usage

*(To be completed once the pipeline is runnable end to end.)*

## Resources

- [Constrained decoding / guided generation — general background]
- [Hugging Face `transformers` documentation — tokenizers and BPE]
- *(Add classic references here as the project progresses: articles, papers or documentation used
  for constrained decoding, BPE tokenization, and pydantic.)*

### AI usage disclosure

This project was developed with assistance from Claude (Anthropic), used specifically for:
- Explaining underlying concepts (tokenization, constrained decoding, pydantic validation) before
  implementation, without generating the implementation itself.
- Reviewing hand-written code for bugs, type-safety issues and adherence to the project's
  constraints (no heuristics for function selection, proper error handling, etc.).
- Helping structure the project setup (`uv` workspace configuration, `.gitignore`, `Makefile`).

All code was written and understood by the author(s); AI assistance was used as a learning aid and
reviewer, not as a code generator for the core algorithm.