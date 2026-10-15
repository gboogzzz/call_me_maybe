.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 . --exclude=.venv,llm_sdk,.cache
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .hf_cache
