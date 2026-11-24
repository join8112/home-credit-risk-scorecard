"""Exploratory data analysis on the SQL feature layer.

    python -m hcrisk.eda

Reads data/processed/applicant_features.parquet and writes
    reports/figures/*.png         charts used in reports/01_data_and_eda.md
    reports/tables/*.csv          IV ranking, PSI, coverage, category tables
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .config import FIG_DIR, PROCESSED_DIR, REPORTS_DIR

TABLE_DIR = REPORTS_DIR / "tables"

# ---- palette (validated reference palette, light mode) ----------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SOURCE_ORDER = ["Application", "Bureau", "Previous app", "Installments",
                "POS cash", "Credit card"]
SOURCE_COLOR = dict(zip(SOURCE_ORDER, SERIES))
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a283", "#c4471c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})

ID_COLS = {"SK_ID_CURR", "TARGET", "IS_TRAIN"}


def source_of(col: str) -> str:
    for prefix, name in [("BUREAU_", "Bureau"), ("PREV_", "Previous app"),
                         ("INST_", "Installments"), ("POS_", "POS cash"),
                         ("CC_", "Credit card")]:
        if col.startswith(prefix):
            return name
    if col in ("HAS_BUREAU",):
        return "Bureau"
    return "Application"


# ---- binning, WOE / IV, PSI -------------------------------------------------
def make_bins(s: pd.Series, n: int = 10) -> pd.Series:
    """Quantile bins for numerics, levels for categoricals; missing = own bin.
    A deliberately simple, unsupervised binning: good enough to rank
    variables before the supervised (monotonic) binning in step 2."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        s = s.astype("string")
        freq = s.value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        out = s.where(~s.isin(rare), "Other (<1%)")
        return out.fillna("Missing").astype(str)
    if s.nunique(dropna=True) <= n:
        out = s.astype("string")
    else:
        try:
            out = pd.qcut(s, q=n, duplicates="drop").astype("string")
        except ValueError:
            out = s.astype("string")
    return out.fillna("Missing").astype(str)


def iv_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.crosstab(bins, y)
    t = t.reindex(columns=[0, 1], fill_value=0)
    good = (t[0] + 0.5) / (t[0].sum() + 0.5)
    bad = (t[1] + 0.5) / (t[1].sum() + 0.5)
    woe = np.log(good / bad)
    return pd.DataFrame({"n": t.sum(axis=1), "bad_rate": t[1] / t.sum(axis=1),
                         "woe": woe, "iv": (good - bad) * woe})


def psi(train: pd.Series, test: pd.Series) -> float:
    """Population Stability Index of test vs train, using train's bins."""
    both = pd.concat([train, test])
    b = make_bins(both)
    p = b.iloc[: len(train)].value_counts(normalize=True)
    q = b.iloc[len(train):].value_counts(normalize=True)
    p, q = p.align(q, fill_value=0)
    p, q = p + 1e-4, q + 1e-4
    return float(((q - p) * np.log(q / p)).sum())


def iv_strength(iv: float) -> str:
    # Siddiqi (2006) rule of thumb, the convention used in bank scorecard teams
    if iv < 0.02:
        return "not predictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious (check leakage)"


# ---- charts -----------------------------------------------------------------
def fig_iv(ivdf: pd.DataFrame, top: int = 25) -> None:
    d = ivdf.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7.2))
    ax.barh(d["feature"], d["iv"], color=[SOURCE_COLOR[s] for s in d["source"]],
            height=0.72, edgecolor=SURFACE, linewidth=1)
    for thr, lab in [(0.02, "weak"), (0.1, "medium"), (0.3, "strong")]:
        ax.axvline(thr, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.text(thr, top - 0.2, f" {lab} ≥{thr}", color=INK2, fontsize=7,
                va="bottom")
    ax.set_xlabel("Information Value (10 quantile bins + missing bin)")
    ax.set_title(f"Top {top} features by Information Value", loc="left")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SOURCE_COLOR[s])
               for s in SOURCE_ORDER if s in set(d["source"])]
    ax.legend(handles, [s for s in SOURCE_ORDER if s in set(d["source"])],
              title="Source table", fontsize=8, title_fontsize=8,
              frameon=False, loc="lower right")
    fig.savefig(FIG_DIR / "01_iv_top25.png")
    plt.close(fig)


def fig_deciles(df: pd.DataFrame, feats: list[str], labels: dict) -> None:
    base = df["TARGET"].mean()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True)
    for ax, f in zip(axes.ravel(), feats):
        s = df[f]
        q = pd.qcut(s.rank(method="first"), 10, labels=False)
        g = df.groupby(q)["TARGET"].mean()
        x = np.arange(1, len(g) + 1)
        ax.bar(x, g.values * 100, color=SERIES[0], width=0.78,
               edgecolor=SURFACE, linewidth=1)
        miss = s.isna().mean()
        if miss > 0.001:
            ax.bar([12], [df.loc[s.isna(), "TARGET"].mean() * 100],
                   color="#b5b4ad", width=0.78)
            ax.set_xticks(list(x) + [12], [str(i) for i in x] + ["NA"])
        else:
            ax.set_xticks(x)
        ax.axhline(base * 100, color=SERIES[1], lw=1.5)
        ax.set_title(labels.get(f, f), loc="left", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Default rate (%)")
    for ax in axes[1]:
        ax.set_xlabel("Decile (1 = lowest value)")
    fig.suptitle("Default rate by decile: monotonic patterns a scorecard can use",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.06, 0.925, f"Orange line = portfolio default rate ({base:.1%}). "
             "Grey bar = applicants with no record in that table.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "02_default_rate_by_decile.png")
    plt.close(fig)


def fig_categories(df: pd.DataFrame) -> pd.DataFrame:
    base = df["TARGET"].mean()
    cols = [("NAME_INCOME_TYPE", "Income type"),
            ("NAME_EDUCATION_TYPE", "Education"),
            ("OCCUPATION_TYPE", "Occupation (top 12 by volume)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    rows = []
    for ax, (c, title) in zip(axes, cols):
        g = (df.assign(**{c: df[c].fillna("Missing")})
               .groupby(c)["TARGET"].agg(["mean", "size"]))
        g = g[g["size"] >= 500].sort_values("size", ascending=False).head(12)
        g = g.sort_values("mean")
        for k, r in g.iterrows():
            rows.append({"variable": c, "level": k, "n": int(r["size"]),
                         "default_rate": r["mean"]})
        ylabels = [f"{k}  (n={int(n):,})" for k, n in zip(g.index, g["size"])]
        ax.barh(ylabels, g["mean"] * 100, color=SERIES[0], height=0.7,
                edgecolor=SURFACE, linewidth=1)
        for i, v in enumerate(g["mean"]):
            ax.text(v * 100 + 0.2, i, f"{v:.1%}", va="center", fontsize=7,
                    color=INK2)
        ax.axvline(base * 100, color=SERIES[1], lw=1.5)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Default rate (%)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, g["mean"].max() * 100 * 1.2)
    fig.suptitle("Default rate by applicant profile (orange line = portfolio average)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_default_rate_by_category.png")
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_coverage(df: pd.DataFrame) -> pd.DataFrame:
    flags = [("HAS_BUREAU", "Credit Bureau"), ("HAS_PREV_APP", "Previous application"),
             ("HAS_INSTALLMENTS", "Installment history"),
             ("HAS_CREDIT_CARD", "Home Credit card")]
    rows = []
    for f, name in flags:
        for v in (1, 0):
            sub = df[df[f] == v]
            rows.append({"table": name, "has_history": bool(v),
                         "share_of_applicants": len(sub) / len(df),
                         "default_rate": sub["TARGET"].mean()})
    cov = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(flags))
    w = 0.36
    has = cov[cov.has_history]
    no = cov[~cov.has_history]
    ax.barh(y + w / 2, has["default_rate"] * 100, height=w, color=SERIES[0],
            label="Has history", edgecolor=SURFACE, linewidth=1)
    ax.barh(y - w / 2, no["default_rate"] * 100, height=w, color=SERIES[1],
            label="No history (thin file)", edgecolor=SURFACE, linewidth=1)
    for i, (h, n) in enumerate(zip(has.itertuples(), no.itertuples())):
        ax.text(h.default_rate * 100 + 0.15, i + w / 2,
                f"{h.default_rate:.1%}  ({h.share_of_applicants:.0%} of applicants)",
                va="center", fontsize=7.5, color=INK2)
        ax.text(n.default_rate * 100 + 0.15, i - w / 2,
                f"{n.default_rate:.1%}  ({n.share_of_applicants:.0%})",
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [n for _, n in flags])
    ax.invert_yaxis()
    ax.set_xlim(0, 15)
    ax.set_xlabel("Default rate (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.45, -0.18), ncol=2)
    ax.set_title("Does having a credit history change risk?", loc="left")
    fig.savefig(FIG_DIR / "04_history_coverage.png")
    plt.close(fig)
    return cov


def fig_corr(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    c = df[feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.6))
    im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(feats)), feats, fontsize=7)
    ax.grid(False)
    for i in range(len(feats)):
        for j in range(len(feats)):
            v = c.values[i, j]
            if i != j and abs(v) >= 0.6:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.75 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    ax.set_title("Spearman correlation among the top-IV features (|ρ| ≥ 0.6 labelled)",
                 loc="left")
    fig.savefig(FIG_DIR / "05_correlation_top_iv.png")
    plt.close(fig)
    return c


def fig_psi(ivdf: pd.DataFrame) -> None:
    d = (ivdf[ivdf.psi_train_vs_test > 0.05]
         .sort_values("psi_train_vs_test").copy())
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(d) + 1.3))
    colors = [SERIES[1] if v >= 0.25 else (SERIES[3] if v >= 0.1 else SERIES[0])
              for v in d["psi_train_vs_test"]]
    ax.barh(d["feature"], d["psi_train_vs_test"], color=colors, height=0.7,
            edgecolor=SURFACE, linewidth=1)
    for thr, lab, ha in [(0.1, "monitor ≥0.1 ", "right"),
                         (0.25, " significant shift ≥0.25", "left")]:
        ax.axvline(thr, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.text(thr, len(d) - 0.3, lab, color=INK2, fontsize=7, va="bottom",
                ha=ha)
    for i, v in enumerate(d["psi_train_vs_test"]):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("PSI, Kaggle test vs train applicants")
    ax.grid(axis="y", visible=False)
    ax.set_title("Population stability: which features drift between samples?",
                 loc="left")
    fig.savefig(FIG_DIR / "06_psi_train_vs_test.png")
    plt.close(fig)


# ---- main ------------------------------------------------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(PROCESSED_DIR / "applicant_features.parquet")
    train = full[full["IS_TRAIN"]].copy()
    test = full[~full["IS_TRAIN"]].copy()
    y = train["TARGET"].astype(int)
    print(f"train {len(train):,} rows, default rate {y.mean():.2%}; "
          f"test {len(test):,} rows")

    feats = [c for c in full.columns if c not in ID_COLS]

    # IV + PSI for every feature
    rows = []
    for f in feats:
        t = iv_table(make_bins(train[f]), y)
        rows.append({"feature": f, "source": source_of(f),
                     "iv": t["iv"].sum(), "missing_rate": train[f].isna().mean(),
                     "n_bins": len(t),
                     "psi_train_vs_test": psi(train[f], test[f])})
    ivdf = pd.DataFrame(rows).sort_values("iv", ascending=False)
    ivdf["strength"] = ivdf["iv"].map(iv_strength)
    ivdf.to_csv(TABLE_DIR / "iv_ranking.csv", index=False, float_format="%.4f")
    print(ivdf.head(30).to_string(index=False))
    print(ivdf["strength"].value_counts())
    print("PSI > 0.1:\n", ivdf[ivdf.psi_train_vs_test > 0.1][["feature", "psi_train_vs_test"]])

    fig_iv(ivdf)
    fig_psi(ivdf)
    numeric_top = [f for f in ivdf["feature"]
                   if pd.api.types.is_numeric_dtype(train[f])
                   and train[f].nunique() > 10][:20]
    labels = {
        "EXT_SOURCE_MEAN": "External credit score (mean of 3)",
        "AGE_YEARS": "Applicant age",
        "CREDIT_TERM_MONTHS": "Loan term (credit / annuity)",
        "INST_LATE_RATE": "Share of past installments paid late",
        "PREV_REFUSED_RATE": "Share of past applications refused",
        "BUREAU_DAYS_SINCE_LAST_OPEN": "Days since last bureau loan opened",
        "EMPLOYED_YEARS": "Years employed",
        "DAYS_ID_PUBLISH": "Days since ID document issued",
    }
    decile_feats = ["EXT_SOURCE_MEAN", "AGE_YEARS", "CREDIT_TERM_MONTHS",
                    "INST_LATE_RATE", "PREV_REFUSED_RATE",
                    "BUREAU_DAYS_SINCE_LAST_OPEN"]
    fig_deciles(train, decile_feats, labels)
    fig_categories(train).to_csv(TABLE_DIR / "default_rate_by_category.csv",
                                 index=False, float_format="%.4f")
    fig_coverage(train).to_csv(TABLE_DIR / "history_coverage.csv", index=False,
                               float_format="%.4f")
    corr = fig_corr(train, numeric_top)
    pairs = (corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
             .rename("rho").reset_index())
    pairs = pairs[pairs.rho.abs() >= 0.6].sort_values("rho", key=abs, ascending=False)
    pairs.to_csv(TABLE_DIR / "high_correlation_pairs.csv", index=False,
                 float_format="%.3f")
    print(pairs.to_string(index=False))

    # extra facts quoted in the report
    facts = {
        "n_train": len(train), "n_test": len(test), "n_features": len(feats),
        "default_rate": y.mean(),
        "employed_anomaly_share": train["FLAG_EMPLOYED_ANOMALY"].mean(),
        "default_rate_employed_anomaly": train.loc[train.FLAG_EMPLOYED_ANOMALY == 1, "TARGET"].mean(),
        "default_rate_employed_normal": train.loc[train.FLAG_EMPLOYED_ANOMALY == 0, "TARGET"].mean(),
        "default_rate_male": train.loc[train.CODE_GENDER == "M", "TARGET"].mean(),
        "default_rate_female": train.loc[train.CODE_GENDER == "F", "TARGET"].mean(),
        "train_bb_coverage_among_bureau": (train.loc[train.HAS_BUREAU == 1, "BUREAU_N_LOANS_WITH_BB"] > 0).mean(),
        "test_bb_coverage_among_bureau": (test.loc[test.HAS_BUREAU == 1, "BUREAU_N_LOANS_WITH_BB"] > 0).mean(),
        "train_revolving_share": (train.NAME_CONTRACT_TYPE == "Revolving loans").mean(),
        "test_revolving_share": (test.NAME_CONTRACT_TYPE == "Revolving loans").mean(),
        "train_median_term": train.CREDIT_TERM_MONTHS.median(),
        "test_median_term": test.CREDIT_TERM_MONTHS.median(),
        "share_app_cols_over_50pct_missing": (train[[f for f in feats if source_of(f) == "Application"]].isna().mean() > 0.5).mean(),
    }
    pd.Series(facts).to_csv(TABLE_DIR / "facts.csv", header=["value"],
                            float_format="%.4f")
    print(pd.Series(facts))


if __name__ == "__main__":
    main()
