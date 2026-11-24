"""Load the raw Kaggle CSVs into DuckDB and run the SQL feature layer.

    python -m hcrisk.build_features            # load + build
    python -m hcrisk.build_features --skip-load

Outputs
    data/home_credit.duckdb                 raw_*, stg_*, feat_*, applicant_features
    data/processed/applicant_features.parquet
"""
import argparse
import time

import duckdb

from .config import DB_PATH, PROCESSED_DIR, RAW_DIR, RAW_TABLES, SQL_DIR


def load_raw(con: duckdb.DuckDBPyConnection) -> None:
    for table, fname in RAW_TABLES.items():
        path = RAW_DIR / fname
        t0 = time.time()
        # sample_size=-1: scan the whole file for type inference, so a column
        # that is empty in the first rows is not mis-typed.
        con.execute(
            f"CREATE OR REPLACE TABLE {table} AS "
            f"SELECT * FROM read_csv_auto(?, header=true, sample_size=-1)",
            [str(path)],
        )
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<28} {n:>12,} rows  ({time.time() - t0:.1f}s)")


def run_sql_dir(con: duckdb.DuckDBPyConnection, sub: str) -> None:
    for f in sorted((SQL_DIR / sub).glob("*.sql")):
        t0 = time.time()
        con.execute(f.read_text())
        print(f"  {sub}/{f.name:<30} ({time.time() - t0:.1f}s)")


def check_grain(con: duckdb.DuckDBPyConnection) -> None:
    """Every feature table must be exactly one row per SK_ID_CURR."""
    for t in ["stg_application", "feat_bureau", "feat_prev", "feat_inst",
              "feat_pos", "feat_cc", "applicant_features"]:
        n, k = con.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT SK_ID_CURR) FROM {t}").fetchone()
        assert n == k, f"{t}: {n} rows but {k} distinct SK_ID_CURR"
        print(f"  grain ok: {t:<20} {n:>9,} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-load", action="store_true")
    args = ap.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    if not args.skip_load:
        print("Loading raw CSVs ...")
        load_raw(con)
    print("Building staging + features ...")
    run_sql_dir(con, "staging")
    run_sql_dir(con, "features")
    check_grain(con)
    out = PROCESSED_DIR / "applicant_features.parquet"
    con.execute(f"COPY applicant_features TO '{out}' (FORMAT PARQUET)")
    ncol = len(con.execute("DESCRIBE applicant_features").fetchall())
    print(f"Wrote {out} ({ncol} columns)")


if __name__ == "__main__":
    main()
