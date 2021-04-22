all: format

format:
	python3 -m black src/
	python3 -m isort src/
