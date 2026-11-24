#!/usr/bin/env bash
# Download the Kaggle "Home Credit Default Risk" data into data/raw/.
# Needs KAGGLE_USERNAME / KAGGLE_KEY (or ~/.kaggle/kaggle.json) and the
# competition rules accepted at https://www.kaggle.com/c/home-credit-default-risk/rules
set -euo pipefail
RAW_DIR="${1:-data/raw}"
mkdir -p "$RAW_DIR"
kaggle competitions download -c home-credit-default-risk -p "$RAW_DIR"
unzip -o -q "$RAW_DIR/home-credit-default-risk.zip" -d "$RAW_DIR"
rm -f "$RAW_DIR/home-credit-default-risk.zip"
ls -lh "$RAW_DIR"
