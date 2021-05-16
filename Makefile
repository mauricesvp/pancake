all: format

format:
	python3 -m black pancake/ tests/
