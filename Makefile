.PHONY: check test public-hygiene

check: test public-hygiene

test:
	python3 -m unittest discover -s tests -p 'test_*.py'

public-hygiene:
	python3 tools/public_hygiene.py .
