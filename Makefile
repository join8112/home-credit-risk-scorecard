.PHONY: setup data features eda scorecard all

setup:
	pip install -r requirements.txt && pip install -e .

data:
	./scripts/download_data.sh

features:
	python -m hcrisk.build_features

eda:
	python -m hcrisk.eda

scorecard:
	python -m hcrisk.scorecard

all: data features eda scorecard
