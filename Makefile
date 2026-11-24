.PHONY: setup data features eda all

setup:
	pip install -r requirements.txt && pip install -e .

data:
	./scripts/download_data.sh

features:
	python -m hcrisk.build_features

eda:
	python -m hcrisk.eda

all: data features eda
