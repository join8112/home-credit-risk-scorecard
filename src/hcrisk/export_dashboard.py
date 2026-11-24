"""Export the aggregated data the Streamlit dashboard needs.

    python -m hcrisk.export_dashboard      (after scorecard and compare)

Only aggregates leave this script: score histograms, cut-off tables and
the published report tables. No applicant-level Kaggle rows are written,
so the dashboard can be deployed publicly without redistributing the data.
"""
import json
import pickle
import shutil
import types

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .compare import LOSS_BAD, MARGIN_GOOD, align_categories, prep
from .config import PROCESSED_DIR, REPORTS_DIR, ROOT
from .scorecard import MODEL_DIR, SEED, candidate_features, score_frame

OUT = ROOT / "dashboard" / "data"
TABLES = REPORTS_DIR / "tables"
COPY = ["scorecard_points.csv", "scorecard_coefficients.csv", "rating_grades.csv",
        "model_comparison.csv", "iv_ranking.csv", "shap_importance.csv",
        "scorecard_summary.json", "comparison_summary.json", "history_coverage.csv"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        shutil.copy(TABLES / f, OUT / f)

    full = pd.read_parquet(PROCESSED_DIR / "applicant_features.parquet")
    kt = full[~full.IS_TRAIN]
    data = full[full.IS_TRAIN]
    dev, hold = train_test_split(data, test_size=0.30, stratify=data.TARGET, random_state=SEED)
    fit, val = train_test_split(dev, test_size=0.20, stratify=dev.TARGET, random_state=SEED)
    y = hold.TARGET.astype(int).values
    credit = hold.AMT_CREDIT.values

    with open(MODEL_DIR / "scorecard.pkl", "rb") as fh:
        sc = pickle.load(fh)
    m = types.SimpleNamespace(params=sc["logit_params"])
    score = score_frame(sc["binning"], m, sc["features"], hold).sum(axis=1).values
    risk = {"Scorecard": -score}

    # rebuild the exact XGBoost inputs used in compare.py (same category codes)
    cands, _ = candidate_features(full)
    for name, feats, path in [("XGBoost monotone-12", sc["features"], "xgb_mono12.json"),
                              ("XGBoost full", cands, "xgb_full.json")]:
        frames = [prep(d, feats) for d in (fit, val, dev, hold, kt)]
        align_categories(frames)
        booster = xgb.Booster()
        booster.load_model(MODEL_DIR / path)
        risk[name] = booster.predict(xgb.DMatrix(frames[3], enable_categorical=True))

    # score histogram (scorecard)
    edges = np.arange(440, 680, 5)
    b = pd.cut(score, edges, right=False)
    hist = pd.DataFrame({"bin": b, "y": y}).groupby("bin", observed=False).y.agg(["size", "sum"])
    hist = pd.DataFrame({"score_from": edges[:-1], "n": hist["size"].values,
                         "bads": hist["sum"].values})
    hist.to_csv(OUT / "score_hist.csv", index=False)

    # cut-off tables: approve the safest x% for x = 50..100
    male = (hold.CODE_GENDER == "M").values
    female = (hold.CODE_GENDER == "F").values
    young = (hold.AGE_YEARS < 30).values
    older = (hold.AGE_YEARS >= 60).values
    rows = []
    for name, r in risk.items():
        order = np.argsort(r, kind="stable")
        for pct in range(50, 101):
            k = int(round(len(r) * pct / 100))
            appr = np.zeros(len(r), bool)
            appr[order[:k]] = True
            gain = np.where(y == 0, MARGIN_GOOD * credit, -LOSS_BAD * credit)
            rows.append({
                "model": name, "approval_pct": pct,
                "score_cutoff": float(score[order[k - 1]]) if name == "Scorecard" else None,
                "bad_rate_approved": y[appr].mean(),
                "bads_declined_share": y[~appr].sum() / y.sum(),
                "goods_declined_share": ((y == 0) & ~appr).sum() / (y == 0).sum(),
                "profit_bn": gain[appr].sum() / 1e9,
                "air_male_vs_female": appr[male].mean() / max(appr[female].mean(), 1e-9),
                "air_under30_vs_60plus": appr[young].mean() / max(appr[older].mean(), 1e-9),
            })
    pd.DataFrame(rows).to_csv(OUT / "cutoff_table.csv", index=False, float_format="%.5f")
    (OUT / "meta.json").write_text(json.dumps({
        "n_holdout": int(len(y)), "default_rate": float(y.mean()),
        "margin_good": MARGIN_GOOD, "loss_bad": LOSS_BAD,
        "factor": float(sc["factor"]), "offset": float(sc["offset"])}, indent=2))
    print("exported to", OUT)
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<32} {f.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
