format-and-lint:
	pre-commit run --all-files

test:
	pytest -x -v -Werror


clean-preview:
	git clean -xdn

clean:
	git clean -xdf
