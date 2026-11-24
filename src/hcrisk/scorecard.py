"""Stage 2: WOE/IV logistic-regression scorecard.

    python -m hcrisk.scorecard

Pipeline (each step is explained in reports/02_scorecard.md)
    1. candidate features: stage-1 exclusions + stability filter (PSI)
    2. supervised, monotonic binning with optbinning -> WOE
    3. IV filter, WOE-correlation filter
    4. forward stepwise selection on the development sample
    5. logistic regression (statsmodels for p-values, VIF)
    6. scaling to points (PDO), integer scorecard table
    7. validation on a hold-out: AUC / Gini / KS, calibration, rating grades,
       score PSI, reason codes, fairness check
"""
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import FIG_DIR, PROCESSED_DIR, REPORTS_DIR, ROOT
from .eda import INK, INK2, SERIES, SURFACE, psi  # noqa: F401  (also sets rcParams)

warnings.filterwarnings("ignore")
TABLE_DIR = REPORTS_DIR / "tables"
MODEL_DIR = ROOT / "models"
SEED = 42

# ---- scaling convention -----------------------------------------------------
# 600 points = odds of 50:1 (good:bad); every 20 points doubles the odds.
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

# ---- stage-1 decisions --------------------------------------------------------
ID_COLS = ["SK_ID_CURR", "TARGET", "IS_TRAIN"]
EXCLUDE = {
    "CODE_GENDER",                       # protected attribute (fair lending)
    "DAYS_BIRTH", "DAYS_EMPLOYED",       # duplicates of AGE_YEARS / EMPLOYED_YEARS
}
EXCLUDE_PREFIX = ("BUREAU_BB_", "BUREAU_N_LOANS_WITH_BB")  # PSI > 1.3 (stage 1)
PSI_MAX = 0.25        # industry rule: > 0.25 = significant population shift
IV_MIN = 0.02         # Siddiqi: below 0.02 not predictive
CORR_MAX = 0.6        # max |Pearson| between WOE-transformed features
MAX_FEATURES = 15     # a scorecard a credit committee can read
MIN_AUC_GAIN = 0.0015 # stop stepwise selection below this gain


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_stat(y, p)}


# ---- 1. candidates ------------------------------------------------------------
def candidate_features(full: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    train, test = full[full.IS_TRAIN], full[~full.IS_TRAIN]
    rows = []
    for c in full.columns:
        if c in ID_COLS:
            continue
        reason = None
        if c in EXCLUDE:
            reason = "excluded (stage 1 decision)"
        elif c.startswith(EXCLUDE_PREFIX):
            reason = "excluded (bureau_balance coverage shift)"
        else:
            p = psi(train[c], test[c])
            if p > PSI_MAX:
                reason = f"unstable (PSI {p:.2f} > {PSI_MAX})"
        rows.append({"feature": c, "dropped_reason": reason})
    log = pd.DataFrame(rows)
    return log.loc[log.dropped_reason.isna(), "feature"].tolist(), log


# ---- 2-3. binning, IV and correlation filters ---------------------------------
def fit_binning(X: pd.DataFrame, y: np.ndarray, feats: list[str]) -> BinningProcess:
    cats = [f for f in feats if not pd.api.types.is_numeric_dtype(X[f])]
    params = {f: {"monotonic_trend": "auto_asc_desc"} for f in feats if f not in cats}
    # no cat_cutoff: a rare level (e.g. "Academic degree", 0.05%) would become
    # its own tiny "others" bin with an extreme WOE. Without it, optbinning
    # merges rare levels into the neighbour with the closest bad rate, so
    # min_bin_size holds for every bin.
    bp = BinningProcess(
        feats, categorical_variables=cats, binning_fit_params=params,
        min_bin_size=0.05,         # every bin holds >= 5% of applicants
        max_n_prebins=20,
        selection_criteria={"iv": {"min": IV_MIN}},
        n_jobs=4,
    )
    X = X[feats].copy()
    for c in cats:
        X[c] = X[c].astype(object)
    bp.fit(X, y)
    return bp


def to_woe(bp: BinningProcess, X: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = X[bp.variable_names].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object)
    W = bp.transform(X, metric="woe")
    W.index = X.index
    return W[feats]


def corr_filter(W: pd.DataFrame, iv: pd.Series) -> tuple[list[str], list[dict]]:
    keep, dropped = [], []
    for f in iv.sort_values(ascending=False).index:
        hit = next((k for k in keep if abs(np.corrcoef(W[f], W[k])[0, 1]) > CORR_MAX), None)
        if hit is None:
            keep.append(f)
        else:
            dropped.append({"feature": f, "dropped_for": hit,
                            "corr": float(np.corrcoef(W[f], W[hit])[0, 1])})
    return keep, dropped


# ---- 4. forward stepwise -----------------------------------------------------
def forward_selection(W, y, W_val, y_val, pool):
    """Greedy forward selection on AUC of a validation fold.
    Only accepts a feature whose coefficient keeps the expected sign
    (negative on WOE: higher WOE = safer bin = lower PD)."""
    chosen, path, best = [], [], 0.5
    while len(chosen) < MAX_FEATURES:
        scores = {}
        for f in pool:
            if f in chosen:
                continue
            cols = chosen + [f]
            m = LogisticRegression(C=1e6, max_iter=500).fit(W[cols], y)
            if (m.coef_[0] >= 0).any():
                continue
            scores[f] = roc_auc_score(y_val, m.predict_proba(W_val[cols])[:, 1])
        if not scores:
            break
        f, s = max(scores.items(), key=lambda kv: kv[1])
        if s - best < MIN_AUC_GAIN:
            break
        chosen.append(f)
        path.append({"step": len(chosen), "added": f, "val_auc": s, "gain": s - best})
        best = s
        print(f"  step {len(chosen):>2}: +{f:<32} val AUC {s:.4f}")
    return chosen, pd.DataFrame(path)


# ---- 6. scaling --------------------------------------------------------------
def build_scorecard(bp, model, feats):
    b0 = model.params["const"]
    n = len(feats)
    rows = []
    for f in feats:
        bt = bp.get_binned_variable(f).binning_table.build()
        bt = bt[~bt.index.isin(["Totals"])]
        bt = bt[bt["Count"] > 0]
        beta = model.params[f]
        for _, r in bt.iterrows():
            woe = float(r["WoE"])
            pts = -(beta * woe + b0 / n) * FACTOR + OFFSET / n
            # transform() maps WoE sign so that good bins are positive; the
            # binning table reports the same sign
            b = r["Bin"]
            label = ", ".join(map(str, b)) if isinstance(b, (list, tuple, np.ndarray)) \
                or hasattr(b, "tolist") else str(b)
            rows.append({"feature": f, "bin": label, "count": int(r["Count"]),
                         "share": float(r["Count (%)"]), "bad_rate": float(r["Event rate"]),
                         "woe": woe, "points": int(round(pts))})
    sc = pd.DataFrame(rows)
    return sc


def score_frame(bp, model, feats, X) -> pd.DataFrame:
    W = to_woe(bp, X, feats)
    b0, n = model.params["const"], len(feats)
    pts = pd.DataFrame({f: (-(model.params[f] * W[f] + b0 / n) * FACTOR + OFFSET / n).round()
                        for f in feats}, index=X.index)
    return pts


# ---- 7. validation pieces ------------------------------------------------------
def rating_grades(score, y, n=8):
    """Equal-population score bands, the way a bank presents a rating scale."""
    q = pd.qcut(score, n, labels=False, duplicates="drop")
    g = pd.DataFrame({"score": score, "y": y, "band": q}).groupby("band").agg(
        score_min=("score", "min"), score_max=("score", "max"),
        n=("y", "size"), bads=("y", "sum"), bad_rate=("y", "mean"))
    g = g.iloc[::-1].reset_index(drop=True)
    g.index = [chr(ord("A") + i) for i in range(len(g))]
    g.index.name = "grade"
    g["cum_bad_capture"] = g["bads"][::-1].cumsum()[::-1] / g["bads"].sum()
    return g


def reason_codes(points: pd.DataFrame, max_pts: pd.Series, k=3) -> pd.DataFrame:
    """Adverse-action reasons: features where the applicant lost the most
    points against the best attainable bin (the 'points below max' method)."""
    lost = max_pts[points.columns] - points
    top = np.argsort(-lost.values, axis=1)[:, :k]
    names = np.array(points.columns)[top]
    return pd.DataFrame(names, index=points.index,
                        columns=[f"reason_{i + 1}" for i in range(k)])


# ---- charts ------------------------------------------------------------------
def fig_score_dist(score, y, cutoff):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(score.min() // 10 * 10, score.max() + 10, 8)
    ax.hist(score[y == 0], bins=bins, density=True, histtype="step", color=SERIES[0],
            lw=2, label="Good (repaid)")
    ax.hist(score[y == 1], bins=bins, density=True, histtype="step", color=SERIES[1],
            lw=2, label="Bad (defaulted)")
    ax.axvline(cutoff, color=INK, lw=1, ls=(0, (3, 3)))
    ax.text(cutoff - 2, ax.get_ylim()[1] * 0.97,
            f"cut-off {cutoff:.0f}\n(decline lowest 15%)", fontsize=8, color=INK,
            va="top", ha="right")
    ax.set_xlabel("Score (higher = safer)")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Score distribution on the hold-out sample", loc="left")
    fig.savefig(FIG_DIR / "07_score_distribution.png")
    plt.close(fig)


def fig_roc_ks(y, p, m):
    fpr, tpr, thr = roc_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.plot(fpr, tpr, color=SERIES[0], lw=2)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("False positive rate (goods declined)")
    ax.set_ylabel("True positive rate (bads caught)")
    ax.set_title(f"ROC  ·  AUC {m['AUC']:.3f}, Gini {m['Gini']:.3f}", loc="left")
    ax.set_aspect("equal")
    ax = axes[1]
    order = np.argsort(-p)
    ys = np.asarray(y)[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    x = np.arange(1, len(ys) + 1) / len(ys)
    ax.plot(x, cum_bad, color=SERIES[1], lw=2, label="Bads")
    ax.plot(x, cum_good, color=SERIES[0], lw=2, label="Goods")
    i = np.argmax(cum_bad - cum_good)
    ax.vlines(x[i], cum_good[i], cum_bad[i], color=INK, lw=1)
    ax.text(x[i] + 0.02, (cum_good[i] + cum_bad[i]) / 2, f"KS = {m['KS']:.3f}",
            fontsize=9, color=INK)
    ax.set_xlabel("Share of applicants, riskiest first")
    ax.set_ylabel("Cumulative share")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("KS: max gap between bad and good capture", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_roc_ks.png")
    plt.close(fig)


def fig_calibration(p, y):
    d = pd.DataFrame({"p": p, "y": y})
    d["bin"] = pd.qcut(d.p, 10, labels=False)
    g = d.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = max(g.max()) * 1.1 * 100
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.plot(g.pred * 100, g.obs * 100, "o-", color=SERIES[0], lw=2, ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted PD (%)")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Calibration by decile (hold-out)", loc="left")
    fig.savefig(FIG_DIR / "09_calibration.png")
    plt.close(fig)


def fig_points(sc, iv):
    feats = list(iv.index)
    n = len(feats)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(13, 2.3 * ((n + 2) // 3)))
    for ax, f in zip(axes.ravel(), feats):
        d = sc[sc.feature == f]
        lab = [b if len(b) < 22 else b[:20] + "…" for b in d["bin"]]
        ax.barh(range(len(d)), d["points"], color=SERIES[0], height=0.7,
                edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(range(len(d)), lab, fontsize=6.5)
        ax.invert_yaxis()
        for i, v in enumerate(d["points"]):
            ax.text(v, i, f" {v}", va="center", fontsize=6.5, color=INK2)
        ax.set_title(f"{f}  (IV {iv[f]:.3f})", loc="left", fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Scorecard points by bin (more points = lower risk)", x=0.02,
                 ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_scorecard_points.png")
    plt.close(fig)


# ---- main --------------------------------------------------------------------
def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    full = pd.read_parquet(PROCESSED_DIR / "applicant_features.parquet")
    kaggle_test = full[~full.IS_TRAIN].copy()
    data = full[full.IS_TRAIN].copy()
    y_all = data.TARGET.astype(int)

    # 70% development, 30% hold-out; development further split 80/20 so the
    # stepwise selection is judged on data the coefficients were not fit on
    dev, hold = train_test_split(data, test_size=0.30, stratify=y_all, random_state=SEED)
    fit, val = train_test_split(dev, test_size=0.20, stratify=dev.TARGET, random_state=SEED)
    print(f"dev {len(dev):,} (fit {len(fit):,} / val {len(val):,}), hold-out {len(hold):,}")

    print("1. candidates")
    cands, droplog = candidate_features(full)
    print(f"  {len(cands)} candidates; dropped:\n{droplog.dropped_reason.value_counts()}")

    print("2. binning")
    bp = fit_binning(fit, fit.TARGET.values, cands)
    summ = bp.summary().set_index("name")
    selected = [v for v in cands if summ.loc[v, "selected"]]
    iv = summ.loc[selected, "iv"].astype(float)
    print(f"  {len(selected)} features pass IV >= {IV_MIN}")

    W_fit = to_woe(bp, fit, selected)
    W_val = to_woe(bp, val, selected)
    pool, corr_drops = corr_filter(W_fit, iv)
    print(f"  {len(pool)} after WOE-correlation filter (|r| <= {CORR_MAX})")

    print("3. forward stepwise selection")
    feats, path = forward_selection(W_fit, fit.TARGET.values, W_val, val.TARGET.values, pool)

    # refit on the full development sample with the final binning
    print("4. final fit on development sample")
    bp = fit_binning(dev, dev.TARGET.values, feats)
    W_dev = to_woe(bp, dev, feats)
    logit = sm.Logit(dev.TARGET.values, sm.add_constant(W_dev)).fit(disp=0)
    print(logit.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]])
    vif = pd.Series([variance_inflation_factor(W_dev.values, i) for i in range(len(feats))],
                    index=feats)
    iv_final = pd.Series({f: bp.get_binned_variable(f).binning_table.iv for f in feats})

    coef = pd.DataFrame({"coef": logit.params[feats], "std_err": logit.bse[feats],
                         "p_value": logit.pvalues[feats], "vif": vif, "iv": iv_final})
    coef.to_csv(TABLE_DIR / "scorecard_coefficients.csv", float_format="%.4f")

    print("5. scorecard")
    sc = build_scorecard(bp, logit, feats)
    sc.to_csv(TABLE_DIR / "scorecard_points.csv", index=False, float_format="%.4f")
    max_pts = sc.groupby("feature")["points"].max()
    min_pts = sc.groupby("feature")["points"].min()

    def score(X):
        pts = score_frame(bp, logit, feats, X)
        return pts, pts.sum(axis=1)

    def pd_from_score(s):
        return 1 / (1 + np.exp((s - OFFSET) / FACTOR))

    print("6. validation")
    res = {}
    for name, X in [("development", dev), ("hold-out", hold)]:
        _, s = score(X)
        res[name] = metrics(X.TARGET.values, -s.values)
    # what does an unrestricted logistic on the same WOE give? (sanity)
    print(pd.DataFrame(res).T)

    # how much AUC did the stability rule cost? refit with the unstable
    # CREDIT_TERM_MONTHS added (it was dropped by the PSI filter)
    alt = feats + ["CREDIT_TERM_MONTHS"]
    bp_alt = fit_binning(dev, dev.TARGET.values, alt)
    m_alt = LogisticRegression(C=1e6, max_iter=1000).fit(to_woe(bp_alt, dev, alt), dev.TARGET)
    auc_alt = roc_auc_score(hold.TARGET, m_alt.predict_proba(to_woe(bp_alt, hold, alt))[:, 1])

    # governance check: how dependent is the card on the external vendor scores?
    # rerun the same selection without EXT_SOURCE_* and score the hold-out
    print("  challenger without external scores")
    pool_int = [f for f in pool if not f.startswith("EXT_SOURCE")]
    pool_int += [f for f in selected if f not in pool and not f.startswith("EXT_SOURCE")
                 and all(abs(np.corrcoef(W_fit[f], W_fit[k])[0, 1]) <= CORR_MAX
                         for k in pool_int)]
    feats_int, _ = forward_selection(W_fit, fit.TARGET.values, W_val, val.TARGET.values,
                                     pool_int)
    bp_int = fit_binning(dev, dev.TARGET.values, feats_int)
    m_int = LogisticRegression(C=1e6, max_iter=1000).fit(to_woe(bp_int, dev, feats_int),
                                                          dev.TARGET)
    p_int = m_int.predict_proba(to_woe(bp_int, hold, feats_int))[:, 1]
    res_internal = {"features": feats_int, **metrics(hold.TARGET.values, p_int)}
    print(res_internal)

    pts_h, s_h = score(hold)
    y_h = hold.TARGET.values
    pd_h = pd_from_score(s_h.values)
    cutoff = float(np.quantile(s_h, 0.15))
    fig_score_dist(s_h.values, y_h, cutoff)
    fig_roc_ks(y_h, -s_h.values, res["hold-out"])
    fig_calibration(pd_h, y_h)
    fig_points(sc, iv_final.sort_values(ascending=False))

    grades = rating_grades(s_h.values, y_h)
    grades.to_csv(TABLE_DIR / "rating_grades.csv", float_format="%.4f")
    print(grades)

    # score stability: development vs hold-out vs Kaggle test (later population)
    _, s_dev = score(dev)
    _, s_kt = score(kaggle_test)
    score_psi = {"hold-out vs development": psi(s_dev, s_h),
                 "Kaggle test vs development": psi(s_dev, s_kt)}
    print(score_psi)

    # decisions at the cut-off
    declined = s_h < cutoff
    approved = ~declined
    decision = {
        "cutoff": cutoff,
        "approval_rate": float(approved.mean()),
        "bad_rate_all": float(y_h.mean()),
        "bad_rate_approved": float(y_h[approved].mean()),
        "bad_rate_declined": float(y_h[declined].mean()),
        "share_of_bads_declined": float(y_h[declined].sum() / y_h.sum()),
    }
    print(decision)

    # reason codes
    rc = reason_codes(pts_h[declined.values], max_pts)
    top_reasons = rc["reason_1"].value_counts(normalize=True).rename("share_of_declines")
    top_reasons.to_csv(TABLE_DIR / "top_decline_reasons.csv", float_format="%.4f")
    examples = []
    for idx in hold.index[declined.values][:3]:
        row = pts_h.loc[idx]
        r = rc.loc[idx]
        examples.append({
            "SK_ID_CURR": int(hold.loc[idx, "SK_ID_CURR"]),
            "score": int(row.sum()),
            "reasons": [{"feature": r[c], "value": None if pd.isna(hold.loc[idx, r[c]])
                         else (hold.loc[idx, r[c]] if isinstance(hold.loc[idx, r[c]], str)
                               else round(float(hold.loc[idx, r[c]]), 3)),
                         "points": int(row[r[c]]), "max_points": int(max_pts[r[c]])}
                        for c in rc.columns]})
    print(json.dumps(examples, indent=1, default=str))

    # fairness check (gender is NOT a model input)
    fair = (pd.DataFrame({"gender": hold.CODE_GENDER.fillna("Unknown"), "approved": approved.values,
                          "bad": y_h})
            .query("gender != 'Unknown'")
            .groupby("gender").agg(n=("bad", "size"), approval_rate=("approved", "mean"),
                                   bad_rate=("bad", "mean")))
    fair["bad_rate_if_approved"] = [
        y_h[(hold.CODE_GENDER.values == g) & approved.values].mean() for g in fair.index]
    fair["adverse_impact_ratio"] = fair.approval_rate / fair.approval_rate.max()
    fair.to_csv(TABLE_DIR / "fairness_gender.csv", float_format="%.4f")
    age_band = pd.cut(hold.AGE_YEARS, [0, 30, 40, 50, 60, 100],
                      labels=["<30", "30-39", "40-49", "50-59", "60+"])
    fair_age = (pd.DataFrame({"age": age_band.values, "approved": approved.values, "bad": y_h})
                .groupby("age", observed=True).agg(n=("bad", "size"),
                                                   approval_rate=("approved", "mean"),
                                                   bad_rate=("bad", "mean")))
    fair_age["adverse_impact_ratio"] = fair_age.approval_rate / fair_age.approval_rate.max()
    fair_age.to_csv(TABLE_DIR / "fairness_age.csv", float_format="%.4f")
    print(fair, "\n", fair_age)

    summary = {
        "n_candidates": len(cands), "n_iv_pass": len(selected), "n_after_corr": len(pool),
        "features": feats, "metrics": res, "auc_with_unstable_term": auc_alt,
        "challenger_without_ext_source": res_internal,
        "score_psi": score_psi, "decision": decision,
        "scaling": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO},
        "points_range": [int(min_pts.sum()), int(max_pts.sum())],
        "examples": examples,
    }
    (TABLE_DIR / "scorecard_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    droplog.to_csv(TABLE_DIR / "candidate_log.csv", index=False)
    pd.DataFrame(corr_drops).to_csv(TABLE_DIR / "corr_drops.csv", index=False,
                                    float_format="%.3f")
    path.to_csv(TABLE_DIR / "stepwise_path.csv", index=False, float_format="%.4f")
    with open(MODEL_DIR / "scorecard.pkl", "wb") as fh:
        pickle.dump({"binning": bp, "logit_params": logit.params, "features": feats,
                     "scorecard": sc, "factor": FACTOR, "offset": OFFSET,
                     "hold_ids": hold.SK_ID_CURR.values}, fh)
    print("done")


if __name__ == "__main__":
    main()
