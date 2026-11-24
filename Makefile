.PHONY: setup data features eda scorecard compare dashboard-data app all

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

compare:
	python -m hcrisk.compare

dashboard-data:
	python -m hcrisk.export_dashboard

app:
	streamlit run dashboard/app.py

all: data features eda scorecard compare dashboard-data
