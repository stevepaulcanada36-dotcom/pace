# Common tasks. Windows users: run the commands after each target by hand.
PY ?= python

.PHONY: install run test lint typecheck check reproduce docker docker-run

install:        ## install PaceSense with the web app and dev tools
	$(PY) -m pip install -e ".[studio,dashboard,dev]"

run:            ## start PaceSense Studio on http://127.0.0.1:8000
	pacesense studio

test:           ## the whole test suite
	$(PY) -m unittest discover -s tests -t .

lint:
	ruff check src/pacesense/studio src/pacesense/adaptive tests/test_studio.py tests/test_adaptive.py tests/asgi.py

typecheck:
	mypy src/pacesense/studio src/pacesense/adaptive

check: lint typecheck test   ## everything CI runs

reproduce:      ## rebuild every reported number and verify the manifest
	pacesense reproduce --quick --out results && pacesense reproduce --check results/manifest.json

docker:
	docker build -t pacesense .

docker-run:
	docker run --rm -p 8000:8000 -v pacesense-data:/data pacesense
