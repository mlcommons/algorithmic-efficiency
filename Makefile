.PHONY: pre-commit lint format

pre-commit:
	pre-commit run --all-files

lint:
	ruff check .

format:
	ruff format .
