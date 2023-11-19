bloke_venv:
	python3 -m venv bloke_venv

.PHONY: install_requirements
install-requirements:
	python3 -m pip install < requirements.txt

.PHONY: bril-rs
brilirs:
	make -C vendor/bril/bril-rs install

.PHONY: brilirs
brilirs: bril-rs
	make -C vendor/bril/brilirs install

.PHONY: brilirs-python
brilirs-python:
	make -C vendor/bril/brilirs brilirs-python

.PHONY: venv-activate
venv-activate:
	source bloke_venv/bin/activate

.PHONY: venv-deactivate
venv-deactivate:
	deactivate
