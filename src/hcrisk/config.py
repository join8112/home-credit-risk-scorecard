import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.getenv("HC_RAW_DIR", ROOT / "data" / "raw"))
DB_PATH = Path(os.getenv("HC_DB_PATH", ROOT / "data" / "home_credit.duckdb"))
PROCESSED_DIR = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# raw table name -> csv file
RAW_TABLES = {
    "raw_application_train": "application_train.csv",
    "raw_application_test": "application_test.csv",
    "raw_bureau": "bureau.csv",
    "raw_bureau_balance": "bureau_balance.csv",
    "raw_previous_application": "previous_application.csv",
    "raw_installments_payments": "installments_payments.csv",
    "raw_pos_cash_balance": "POS_CASH_balance.csv",
    "raw_credit_card_balance": "credit_card_balance.csv",
}
