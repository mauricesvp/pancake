all: format

format:
	python3 -m black pancake/main.py pancake/misc/detect_wrapper.py tests/
