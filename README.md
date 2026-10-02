*This project has been created as part of the 42 curriculum by gguia-ma.*

# call me maybe

## Description

A function-calling tool that turns a natural-language request (e.g. *"What is the sum of 2 and 3?"*) into a structured function call: a name and a JSON object of typed arguments (`fn_add_numbers`, `{"a": 2.0, "b": 3.0}`). The model is Qwen3-0.6B, accessed only through raw logits — all JSON validity and schema compliance come from constrained decoding, not from hoping the model behaves.

## Instructions

```bash
git clone <this-repository>
cd call_me_maybe
uv sync
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```

Defaults: `data/input/functions_definition.json`, `data/input/function_calling_tests.json`, `data/output/function_calling_results.json`. Paths are relative to the directory the command is run from.

> **Windows:** if `uv` isn't recognized after `pip install uv`, run `python -m uv sync` / `python -m uv run python -m src` instead — works regardless of `PATH`.

Missing or malformed input files are reported and exit cleanly, no raw traceback.

## Algorithm Explanation

At each token position, the program decides which tokens would keep the output valid JSON matching the schema, masks every other token to `-infinity` in the model's logits, and only then takes the `argmax`. Invalid tokens are mathematically impossible to pick, so the output is valid by construction.

Tokens known in advance — punctuation, JSON keys, the echoed prompt — are **forced** directly (no model call). Only real content — digits, string characters, a boolean, which function to call — is **restricted and asked**: the model only ever decides among the valid options for that exact position.

- `generate_number` / `generate_string` / `generate_boolean`: small state machines offering only valid next tokens, with a length cap to guarantee termination.
- `generate_parameters`: forces each parameter key, dispatches to the matching generator for its value.
- `generate_output`: assembles `{"prompt", "name", "parameters"}`, forcing the envelope and delegating the name to a trie-walk and the parameters to `generate_parameters`.
- `build_function_trie` / `select_function_name`: every candidate function name is encoded into token IDs and merged into a trie, so the model can only ever walk toward a complete, valid name — never a partial or invalid one.

The function name is decided from a **separate, clean context** (not the JSON-building one), built as a few-shot prompt generated at runtime from the function definitions:

```
Request: Add two numbers together and return their sum.
Function: fn_add_numbers

...

Request: Greet shrek
Function:
```

Each block pairs a function's own description with its own name (already in `functions_definition.json`, never guessed); only the last block, with the real prompt, is left open for the model to complete.

## Design Decisions

- Two separate token streams in `generate_output`: one builds the real JSON output, a disposable one is used only to decide the function name, so JSON punctuation never dilutes that decision.
- Pydantic models validate both input files before anything else runs.
- The terminator after a number (`,` or `}`) is passed in by the caller, not chosen by the model, since the model could terminate the parameters object too early.
- `id_to_str` replaces the tokenizer's internal `Ġ` space marker with a real space once, centrally, so generated strings never leak tokenizer artifacts.

## Performance Analysis

Function selection: 11/11 on the example prompt set, up from 6/11 before the few-shot context fix (the two newest functions were essentially never chosen correctly before it). Every output produced has been valid, schema-compliant JSON. One soft spot: generated `regex` parameter content is not always semantically perfect — a known limitation of free-form generation at 500M parameters, not a structural failure. Because most tokens are forced rather than generated, model calls per prompt stay low, comfortably within the 5-minute budget for a full test set.

## Challenges Faced

- BPE tokens don't compose predictably (`'"a":'` and `'"b":'` tokenize differently) — ruled out hardcoding any token sequence; everything dynamic is encoded at runtime.
- Generated strings occasionally contained a literal `Ġ` instead of a space — traced to the tokenizer's raw vocabulary and fixed centrally in `VocabIndex`.
- A function-selection bias that looked like a trie bug turned out to be context pollution: isolating the trie/selection logic with a clean context scored perfectly, proving the bug was in how much JSON structure and duplicated prompt text surrounded the decision. Inspecting raw per-token logits showed the model had a strong, prompt-independent preference for common subword tokens (e.g. `_get`). Plain descriptions and full-candidate log-likelihood scoring were both tried and failed; a dynamically-built few-shot context fixed it.

## Testing Strategy

Small disposable scripts isolated one variable at a time: per-generator checks against individual schemas, a trie-structure dump to confirm shared prefixes collapse correctly, raw logit inspection at each decision point, an end-to-end accuracy harness over the example prompts re-run after every relevant change, and input-file edge cases (missing file, malformed JSON, schema violations) to confirm graceful failure.

## Example Usage

```bash
uv run python -m src
```

Input `function_calling_tests.json`:
```json
[{ "prompt": "What is the sum of 2 and 3?" }, { "prompt": "Greet shrek" }]
```

Output `function_calling_results.json`:
```json
[
  { "prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": { "a": 2.0, "b": 3.0 } },
  { "prompt": "Greet shrek", "name": "fn_greet", "parameters": { "name": "shrek" } }
]
```

## Resources

Web research was used to look up the topics covered in this project (tokenization, constrained decoding, trie structures, and related concepts).

**How AI was used:** Claude was used only to explain concepts as they were asked about during development.