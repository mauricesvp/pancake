all: format

format:
	python3 -m black pancake/ tests/

test:
	coverage run -m pytest
	coverage report -m --include "tests/*","pancake/*" --sort cover
