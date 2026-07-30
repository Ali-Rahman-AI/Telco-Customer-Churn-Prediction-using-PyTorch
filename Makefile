.PHONY: help install test lint clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean generated files"

install:
	pip install -r requirements.txt

test:
	pytest tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
