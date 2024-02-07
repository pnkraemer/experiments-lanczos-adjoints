format-and-lint:
	pre-commit run --all-files

test:
	pytest -x -v
	python -m doctest README.md


clean-preview:
	git clean -xdn

clean:
	git clean -xdf


doc-preview:
	python scripts/generate_api_docs.py
	python scripts/readme_to_dev_docs.py
	python scripts/tutorials_to_py_light.py
	mkdocs serve

doc-build:
	python scripts/generate_api_docs.py
	python scripts/readme_to_dev_docs.py
	python scripts/tutorials_to_py_light.py
	mkdocs build

run-benchmarks:
	python docs/benchmarks/control_variates.py
	python docs/benchmarks/jacobian_squared.py
