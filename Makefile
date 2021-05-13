all: format

format:
	python3 -m black src/main.py
