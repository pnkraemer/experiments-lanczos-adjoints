format-and-lint:
	pre-commit run --all-files

test:
	pytest


clean-preview:
	git clean -xdn

clean:
	git clean -xdf
