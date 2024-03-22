format-and-lint:
	pre-commit run --all-files

test:
	pytest -x -v
	python -m doctest README.md


clean-preview:
	git clean -xdn

clean:
	git clean -xdf
