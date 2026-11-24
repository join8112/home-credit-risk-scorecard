"""Stage 3: XGBoost + SHAP vs the WOE scorecard.

    python -m hcrisk.compare        (run python -m hcrisk.scorecard first)

Models, all trained on the same development sample and scored on the same
hold-out as the scorecard (seed 42, 70/30 stratified):
    A  Scorecard               12 WOE features, logistic, integer points (stage 2)
    B  XGBoost, monotone-12    same 12 raw features, monotone constraints
    C  XGBoost, full           all 186 stable candidate features, unconstrained
B vs A isolates the algorithm; C vs B shows the value of using every feature.
"""
import json
import pickle
import types
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import FIG_DIR, PROCESSED_DIR, REPORTS_DIR
from .eda import GRID, INK, INK2, SERIES, SURFACE, psi
from .scorecard import (MODEL_DIR, SEED, candidate_features, ks_stat,
                        metrics, score_frame)

warnings.filterwarnings("ignore")
TABLE_DIR = REPORTS_DIR / "tables"
MODEL_COLORS = {"Scorecard": SERIES[0], "XGBoost monotone-12": SERIES[2],
                "XGBoost full": SERIES[1]}

# Illustrative unit economics for the profit curve (stated in the report):
# an approved good loan earns 8% of the credit amount over its life; an
# approved bad loan loses 45% of it (Basel foundation-IRB LGD for senior
# unsecured exposures), so one bad wipes out the margin of ~5.6 goods.
MARGIN_GOOD, LOSS_BAD = 0.08, 0.45


def prep(X: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = X[feats].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):          # incl. nullable boolean
            X[c] = X[c].astype("Float64").astype(float)
        elif X[c].dropna().map(type).isin([bool, np.bool_]).all() and X[c].notna().any():
            # e.g. EMERGENCYSTATE_MODE: DuckDB parses Yes/No as BOOLEAN
            X[c] = X[c].map({True: 1.0, False: 0.0}).astype(float)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X


def align_categories(frames: list[pd.DataFrame]) -> None:
    """Same category codes in every frame (XGBoost requires it)."""
    for c in frames[0].columns:
        if isinstance(frames[0][c].dtype, pd.CategoricalDtype):
            cats = sorted(set().union(*[set(f[c].dropna().unique()) for f in frames]))
            for f in frames:
                f[c] = pd.Categorical(f[c].astype(object), categories=cats)


def fit_xgb(Xf, yf, Xv, yv, monotone=None):
    """Small grid on the validation fold, early stopping on AUC."""
    grid = [
        {"max_depth": 3, "min_child_weight": 50},
        {"max_depth": 4, "min_child_weight": 50},
        {"max_depth": 5, "min_child_weight": 100},
    ]
    best = None
    for g in grid:
        m = xgb.XGBClassifier(
            n_estimators=3000, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, reg_lambda=5.0, tree_method="hist",
            enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
            early_stopping_rounds=100, n_jobs=4, random_state=SEED,
            monotone_constraints=monotone, **g)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        auc = m.best_score
        print(f"    {g} -> val AUC {auc:.4f} at {m.best_iteration} trees")
        if best is None or auc > best[1]:
            best = (g, auc, m.best_iteration)
    return best


def refit(Xd, yd, params, n_trees, monotone=None):
    m = xgb.XGBClassifier(
        n_estimators=int(n_trees * 1.1), learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.5, reg_lambda=5.0, tree_method="hist",
        enable_categorical=True, max_cat_to_onehot=1, n_jobs=4,
        random_state=SEED, monotone_constraints=monotone, **params)
    return m.fit(Xd, yd, verbose=False)


def shap_values(model, X):
    """TreeSHAP via XGBoost's native pred_contribs (exact, fast)."""
    dm = xgb.DMatrix(X, enable_categorical=True)
    c = model.get_booster().predict(dm, pred_contribs=True)
    return pd.DataFrame(c[:, :-1], columns=X.columns, index=X.index), c[:, -1]


def profit_curve(score_risk, y, credit):
    """Profit as a function of approval rate, approving the safest first."""
    order = np.argsort(score_risk)            # low risk first
    y, credit = np.asarray(y)[order], np.asarray(credit)[order]
    gain = np.where(y == 0, MARGIN_GOOD * credit, -LOSS_BAD * credit)
    cum = np.cumsum(gain) / 1e9               # in billions of currency units
    rate = np.arange(1, len(y) + 1) / len(y)
    return rate, cum


# ---- charts ------------------------------------------------------------------
def fig_profit(curves):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, (rate, cum) in curves.items():
        ax.plot(rate * 100, cum, color=MODEL_COLORS[name], lw=2, label=name)
        i = int(np.argmax(cum))
        ax.plot(rate[i] * 100, cum[i], "o", color=MODEL_COLORS[name], ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Approval rate (%), safest applicants first")
    ax.set_ylabel("Portfolio profit (billions)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Profit curve on the hold-out (dot = optimal cut-off)", loc="left")
    fig.savefig(FIG_DIR / "11_profit_curve.png")
    plt.close(fig)


def fig_shap_global(sv: pd.DataFrame, top=20):
    imp = sv.abs().mean().sort_values(ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.barh(imp.index, imp.values, color=SERIES[1], height=0.72,
            edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"XGBoost full: top {top} features by mean |SHAP|", loc="left")
    fig.savefig(FIG_DIR / "12_shap_importance.png")
    plt.close(fig)
    return imp.iloc[::-1]


def fig_shap_dependence(X, sv, feats, labels):
    fig, axes = plt.subplots(1, len(feats), figsize=(4.2 * len(feats), 3.8), sharey=True)
    for ax, f in zip(axes, feats):
        x, s = X[f].astype(float), sv[f]
        ok = x.notna()
        idx = np.random.default_rng(0).choice(np.where(ok)[0], size=min(8000, ok.sum()),
                                              replace=False)
        ax.scatter(x.iloc[idx], s.iloc[idx], s=4, alpha=0.25, color=SERIES[1],
                   linewidths=0)
        # binned mean to show the shape
        q = pd.qcut(x[ok], 20, duplicates="drop")
        g = pd.DataFrame({"x": x[ok], "s": s[ok], "q": q}).groupby("q", observed=True).mean()
        ax.plot(g.x, g.s, color=INK, lw=1.8)
        ax.axhline(0, color=INK2, lw=0.8)
        lo, hi = x[ok].quantile([0.005, 0.995])
        ax.set_xlim(lo, hi)
        ax.set_title(labels.get(f, f), loc="left", fontsize=9.5)
        ax.set_xlabel(f, fontsize=8)
    axes[0].set_ylabel("SHAP value (→ higher risk)")
    fig.suptitle("How XGBoost uses a feature: SHAP dependence (black = binned mean)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_shap_dependence.png")
    plt.close(fig)


def fig_local(example, title):
    d = example.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [SERIES[1] if v > 0 else SERIES[0] for v in d.values]
    ax.barh(d.index, d.values, color=colors, height=0.7, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=INK2, lw=0.8)
    for i, v in enumerate(d.values):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    ax.set_xlabel("SHAP value, log-odds of default (orange = pushes towards decline)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10)
    fig.savefig(FIG_DIR / "14_shap_local_example.png")
    plt.close(fig)


# ---- main --------------------------------------------------------------------
def main() -> None:
    full = pd.read_parquet(PROCESSED_DIR / "applicant_features.parquet")
    kt = full[~full.IS_TRAIN].copy()
    data = full[full.IS_TRAIN].copy()
    dev, hold = train_test_split(data, test_size=0.30, stratify=data.TARGET, random_state=SEED)
    fit, val = train_test_split(dev, test_size=0.20, stratify=dev.TARGET, random_state=SEED)
    y_dev, y_h = dev.TARGET.astype(int), hold.TARGET.astype(int).values

    with open(MODEL_DIR / "scorecard.pkl", "rb") as fh:
        sc = pickle.load(fh)
    assert set(sc["hold_ids"]) == set(hold.SK_ID_CURR), "split differs from stage 2"
    sc_model = types.SimpleNamespace(params=sc["logit_params"])
    sc_feats = sc["features"]

    def sc_score(X):
        return score_frame(sc["binning"], sc_model, sc_feats, X).sum(axis=1).values

    def sc_pd(X):
        return 1 / (1 + np.exp((sc_score(X) - sc["offset"]) / sc["factor"]))

    cands, _ = candidate_features(full)
    print(f"{len(cands)} candidate features for XGBoost full")

    results, preds = {}, {}

    # A: scorecard
    preds["Scorecard"] = {"hold": sc_pd(hold), "kt": sc_pd(kt), "dev": sc_pd(dev)}

    # B: XGBoost on the same 12 raw features, monotone in the scorecard's direction
    print("B. XGBoost monotone-12")
    # direction from the scorecard: correlation of raw value with points
    mono = []
    for f in sc_feats:
        if pd.api.types.is_numeric_dtype(dev[f]):
            pts = score_frame(sc["binning"], sc_model, [f], dev.head(20000))
            r = np.corrcoef(dev[f].head(20000).fillna(dev[f].median()), pts[f])[0, 1]
            mono.append(-1 if r > 0 else 1)    # more points = lower risk
        else:
            mono.append(0)
    mono_t = tuple(mono)
    Xs = [prep(d, sc_feats) for d in (fit, val, dev, hold, kt)]
    align_categories(Xs)
    p, auc, n = fit_xgb(Xs[0], fit.TARGET, Xs[1], val.TARGET, mono_t)
    mB = refit(Xs[2], y_dev, p, n, mono_t)
    preds["XGBoost monotone-12"] = {"hold": mB.predict_proba(Xs[3])[:, 1],
                                    "kt": mB.predict_proba(Xs[4])[:, 1],
                                    "dev": mB.predict_proba(Xs[2])[:, 1]}
    results["XGBoost monotone-12"] = {"params": p, "trees": int(n * 1.1)}

    # C: XGBoost on everything stable
    print("C. XGBoost full")
    Xc = [prep(d, cands) for d in (fit, val, dev, hold, kt)]
    align_categories(Xc)
    p, auc, n = fit_xgb(Xc[0], fit.TARGET, Xc[1], val.TARGET)
    mC = refit(Xc[2], y_dev, p, n)
    preds["XGBoost full"] = {"hold": mC.predict_proba(Xc[3])[:, 1],
                             "kt": mC.predict_proba(Xc[4])[:, 1],
                             "dev": mC.predict_proba(Xc[2])[:, 1]}
    results["XGBoost full"] = {"params": p, "trees": int(n * 1.1)}

    # ---- comparison table ----
    rows = []
    credit = hold.AMT_CREDIT.values
    curves = {}
    for name, pr in preds.items():
        m = metrics(y_h, pr["hold"])
        rate, cum = profit_curve(pr["hold"], y_h, credit)
        curves[name] = (rate, cum)
        i = int(np.argmax(cum))
        thr = np.quantile(pr["hold"], 0.85)          # same policy: decline riskiest 15%
        declined = pr["hold"] > thr
        rows.append({
            "model": name, **m,
            "dev_AUC": roc_auc_score(y_dev, pr["dev"]),
            "brier": brier_score_loss(y_h, pr["hold"]),
            "mean_pd_vs_actual": pr["hold"].mean() / y_h.mean(),
            "psi_kaggle_test": psi(pd.Series(pr["dev"]), pd.Series(pr["kt"])),
            "bads_caught_at_15pct_decline": y_h[declined].sum() / y_h.sum(),
            "bad_rate_approved_at_15pct": y_h[~declined].mean(),
            "optimal_approval_rate": rate[i],
            "max_profit_bn": cum[i],
            "profit_at_85pct_bn": cum[int(0.85 * len(cum)) - 1],
        })
    comp = pd.DataFrame(rows).set_index("model")
    comp.to_csv(TABLE_DIR / "model_comparison.csv", float_format="%.4f")
    print(comp.T.to_string())
    fig_profit(curves)

    # ---- fairness at 15% decline ----
    fair = []
    age_band = pd.cut(hold.AGE_YEARS, [0, 30, 40, 50, 60, 100],
                      labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str).values
    for name, pr in preds.items():
        approved = pr["hold"] <= np.quantile(pr["hold"], 0.85)
        for col, grp in [("gender", hold.CODE_GENDER.fillna("NA").values), ("age", age_band)]:
            rates = pd.Series(approved).groupby(grp).mean()
            rates = rates[rates.index != "NA"]
            for g, r in rates.items():
                fair.append({"model": name, "attribute": col, "group": g,
                             "approval_rate": r, "air": r / rates.max()})
    fair = pd.DataFrame(fair)
    fair.to_csv(TABLE_DIR / "fairness_by_model.csv", index=False, float_format="%.4f")
    print(fair.pivot_table(index=["attribute", "group"], columns="model", values="air"))

    # ---- SHAP on the full model ----
    print("SHAP")
    sv, base = shap_values(mC, Xc[3])
    imp = fig_shap_global(sv)
    imp.rename("mean_abs_shap").to_csv(TABLE_DIR / "shap_importance.csv", float_format="%.5f")
    share_top = sv.abs().mean()
    share_top = share_top.sort_values(ascending=False)
    top_share = {"top1": share_top.iloc[0] / share_top.sum(),
                 "top10": share_top.iloc[:10].sum() / share_top.sum(),
                 "n_features_for_80pct": int((share_top.cumsum() / share_top.sum() < 0.8).sum() + 1)}
    print(top_share)

    labels = {"EXT_SOURCE_MEAN": "External score (mean)",
              "AGE_YEARS": "Age", "AMT_ANNUITY": "Annuity amount",
              "AMT_GOODS_PRICE": "Goods price", "INST_LATE_RATE": "Share of installments late",
              "EMPLOYED_YEARS": "Years employed", "AMT_CREDIT": "Credit amount",
              "DAYS_ID_PUBLISH": "Days since ID issued"}
    dep = [f for f in imp.index if pd.api.types.is_numeric_dtype(Xc[3][f])][:4]
    fig_shap_dependence(Xc[3], sv, dep, labels)

    # monotonicity violations in the full model: share of applicants for whom
    # raising a "bad" feature lowers the predicted risk is hard to compute
    # directly; instead compare the binned SHAP mean with the scorecard direction
    viol = {}
    for f in [f for f in sc_feats if pd.api.types.is_numeric_dtype(dev[f])]:
        if f not in sv.columns:
            continue
        x, s = Xc[3][f].astype(float), sv[f]
        ok = x.notna()
        g = pd.DataFrame({"s": s[ok], "q": pd.qcut(x[ok], 10, duplicates="drop")}) \
            .groupby("q", observed=True).s.mean().values
        d = np.sign(np.diff(g))
        main_dir = np.sign(d.sum()) if d.sum() != 0 else 1
        viol[f] = int((d == -main_dir).sum())
    print("direction reversals across deciles:", viol)

    # local explanation: same applicant as the stage-2 report
    ex_id = 249287
    ex_idx = hold.index[hold.SK_ID_CURR == ex_id][0]
    ex = sv.loc[ex_idx]
    top = ex.reindex(ex.abs().sort_values(ascending=False).index).head(10)
    p_ex = float(preds["XGBoost full"]["hold"][list(hold.index).index(ex_idx)])
    fig_local(top, f"Applicant {ex_id}: top 10 SHAP contributions "
                   f"(XGBoost PD {p_ex:.1%}, scorecard score 522)")
    ex_tab = pd.DataFrame({"value": [Xc[3].loc[ex_idx, f] for f in top.index],
                           "shap": top.values}, index=top.index)
    ex_tab.to_csv(TABLE_DIR / "shap_local_example.csv", float_format="%.4f")
    print(ex_tab)

    # reason-code agreement: how often is the scorecard's #1 reason among
    # XGBoost's top-3 SHAP reasons, for applicants both models decline?
    pts = score_frame(sc["binning"], sc_model, sc_feats, hold)
    max_pts = sc["scorecard"].groupby("feature")["points"].max()
    sc_reason = (max_pts[pts.columns] - pts).idxmax(axis=1)
    both = (preds["Scorecard"]["hold"] > np.quantile(preds["Scorecard"]["hold"], 0.85)) & \
           (preds["XGBoost full"]["hold"] > np.quantile(preds["XGBoost full"]["hold"], 0.85))
    idx_both = np.where(both)[0]
    top3 = np.argsort(-sv.values[idx_both], axis=1)[:, :3]
    cols = np.array(sv.columns)
    agree = np.mean([sc_reason.iloc[i] in cols[t] for i, t in zip(idx_both, top3)])
    overlap_decl = both.sum() / (preds["Scorecard"]["hold"] > np.quantile(preds["Scorecard"]["hold"], 0.85)).sum()
    print(f"declined by both: {overlap_decl:.1%}; reason agreement {agree:.1%}")

    summary = {"models": results, "shap_concentration": top_share,
               "direction_reversals": viol, "example_xgb_pd": p_ex,
               "decline_overlap": float(overlap_decl), "reason_agreement": float(agree),
               "n_candidates": len(cands), "monotone_constraints": dict(zip(sc_feats, mono)),
               "assumptions": {"margin_good": MARGIN_GOOD, "loss_bad": LOSS_BAD}}
    (TABLE_DIR / "comparison_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    mC.save_model(MODEL_DIR / "xgb_full.json")
    mB.save_model(MODEL_DIR / "xgb_mono12.json")
    print("done")


if __name__ == "__main__":
    main()
