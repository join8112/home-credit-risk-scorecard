"""Home Credit risk scorecard: interactive demo.

    streamlit run dashboard/app.py

Runs only on the aggregated files in dashboard/data (exported by
`python -m hcrisk.export_dashboard`), so it needs no Kaggle data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK2, GRID = "#52514e", "#e4e3df"
MODEL_COLORS = {"Scorecard": BLUE, "XGBoost monotone-12": AQUA, "XGBoost full": ORANGE}

LABELS = {
    "EXT_SOURCE_MEAN": "External credit score (mean of 3)",
    "EXT_SOURCE_3": "External credit score #3",
    "EMPLOYED_YEARS": "Years in current job",
    "CREDIT_TO_GOODS": "Loan amount ÷ price of goods",
    "PREV_REFUSED_RATE": "Share of past applications refused",
    "CC_AVG_UTILISATION_LAST_6M": "Credit-card utilisation, last 6 months",
    "INST_LATE_RATE": "Share of past installments paid late",
    "PREV_AVG_GRANTED_TO_ASKED": "Amount granted ÷ asked, past loans",
    "NAME_EDUCATION_TYPE": "Education",
    "FLAG_DOCUMENT_3": "Provided document 3",
    "POS_N_LOANS": "Number of past POS / cash loans",
    "PREV_N_HIGH_YIELD": "Number of past high-interest loans",
}

st.set_page_config(page_title="Credit Risk Scorecard", page_icon="📊", layout="wide")


@st.cache_data
def load():
    d = {f.stem: pd.read_csv(f) for f in DATA.glob("*.csv")}
    for f in DATA.glob("*.json"):
        d[f.stem] = json.loads(f.read_text())
    return d


D = load()
meta = D["meta"]
points = D["scorecard_points"]
grades = D["rating_grades"]
cut = D["cutoff_table"]
comp = D["model_comparison"].set_index("model")
FEATURES = list(dict.fromkeys(points.feature))


def pd_from_score(s):
    return 1 / (1 + np.exp((s - meta["offset"]) / meta["factor"]))


def grade_of(s):
    for _, g in grades.iterrows():
        if s >= g.score_min:
            return g.grade, g.bad_rate
    return grades.iloc[-1].grade, grades.iloc[-1].bad_rate


def style(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2), font=dict(size=12))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ---- sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Credit Risk Scorecard")
    st.caption(
        "Kaggle Home Credit Default Risk · WOE/IV logistic scorecard vs XGBoost + SHAP. "
        f"All numbers are on a hold-out of {meta['n_holdout']:,} applicants.")
    st.markdown(
        "**中文說明**：本儀表板展示銀行常用的 WOE 評分卡，"
        "可以模擬單一申請人的分數與拒貸理由、調整核准門檻看違約率／利潤／公平性的變化，"
        "並比較評分卡與 XGBoost 的取捨。")
    st.markdown("[Code & reports on GitHub](https://github.com/join8112/home-credit-risk-scorecard)")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Score an applicant", "Cut-off & fairness", "Model insights"])

# ---- 1. overview --------------------------------------------------------------
with tab1:
    st.subheader("A 12-variable scorecard vs XGBoost")
    c = st.columns(4)
    c[0].metric("Scorecard hold-out AUC", f"{comp.loc['Scorecard', 'AUC']:.3f}",
                help="Gini 0.50, KS 0.38")
    c[1].metric("XGBoost (186 features) AUC", f"{comp.loc['XGBoost full', 'AUC']:.3f}",
                f"+{comp.loc['XGBoost full', 'AUC'] - comp.loc['Scorecard', 'AUC']:.3f}")
    c[2].metric("Same 12 features, XGBoost", f"{comp.loc['XGBoost monotone-12', 'AUC']:.3f}",
                f"+{comp.loc['XGBoost monotone-12', 'AUC'] - comp.loc['Scorecard', 'AUC']:.3f}",
                help="Monotone-constrained: isolates the gain from the algorithm")
    c[3].metric("Portfolio default rate", f"{meta['default_rate']:.1%}")

    show = comp[["AUC", "Gini", "KS", "brier", "psi_kaggle_test",
                 "bads_caught_at_15pct_decline", "max_profit_bn"]].copy()
    show.columns = ["AUC", "Gini", "KS", "Brier", "Score PSI", "Defaulters caught (decline 15%)",
                    "Max profit (bn)"]
    st.dataframe(show.style.format({"AUC": "{:.3f}", "Gini": "{:.3f}", "KS": "{:.3f}",
                                    "Brier": "{:.4f}", "Score PSI": "{:.3f}",
                                    "Defaulters caught (decline 15%)": "{:.1%}",
                                    "Max profit (bn)": "{:.3f}"}),
                 use_container_width=True)
    st.markdown("""
**What this shows**
- **The algorithm alone adds little.** On the scorecard's own 12 variables, XGBoost gains only
  0.004 AUC. The full model's +0.034 comes from using 186 features.
- **Calibration, stability and fairness are similar** across all three models.
- **Explainability is the cost**: the full model needs 46 features for 80% of its SHAP
  attribution, and would flip about 30% of the scorecard's declines.
- **Recommendation:** scorecard as the decision model (champion), XGBoost as a challenger.
""")

# ---- 2. score an applicant ----------------------------------------------------
with tab2:
    st.subheader("Build an applicant and see the score")
    st.caption("Pick a bin for each of the 12 scorecard variables. Defaults are the most common bins.")
    left, right = st.columns([1.1, 1])
    chosen = {}
    with left:
        cols = st.columns(2)
        for i, f in enumerate(FEATURES):
            d = points[points.feature == f].reset_index(drop=True)
            default = int(d["count"].idxmax())
            opts = [f"{b}  ({p} pts)" for b, p in zip(d["bin"], d["points"])]
            pick = cols[i % 2].selectbox(LABELS.get(f, f), opts, index=default, key=f)
            chosen[f] = d.iloc[opts.index(pick)]
    total = int(sum(r.points for r in chosen.values()))
    pd_ = pd_from_score(total)
    g, g_bad = grade_of(total)
    cutoff = int(D["scorecard_summary"]["decision"]["cutoff"])
    with right:
        k = st.columns(3)
        k[0].metric("Score", total)
        k[1].metric("Model PD", f"{pd_:.1%}")
        k[2].metric("Grade", g, help=f"Observed default rate in grade {g}: {g_bad:.1%}")
        if total >= cutoff:
            st.success(f"Approve: score {total} ≥ cut-off {cutoff} (declines the lowest 15%).")
        else:
            st.error(f"Decline: score {total} < cut-off {cutoff}.")
        maxp = points.groupby("feature").points.max()
        lost = pd.Series({f: maxp[f] - r.points for f, r in chosen.items()}).sort_values(ascending=False)
        st.markdown("**Top reasons (points below the best bin)**")
        for f, v in lost.head(3).items():
            if v > 0:
                st.markdown(f"- {LABELS.get(f, f)}: **−{v} points** (bin `{chosen[f].bin}`)")
        if lost.max() == 0:
            st.markdown("- None: every variable is in its best bin.")
        order = lost.index[::-1]
        fig = go.Figure()
        fig.add_bar(y=[LABELS.get(f, f) for f in order], x=[chosen[f].points for f in order],
                    orientation="h", marker_color=BLUE, name="Points earned",
                    hovertemplate="%{y}: %{x} points<extra></extra>")
        fig.add_bar(y=[LABELS.get(f, f) for f in order], x=[lost[f] for f in order],
                    orientation="h", marker_color=GRID, name="Points missed",
                    hovertemplate="%{y}: %{x} points below best<extra></extra>")
        fig.update_layout(barmode="stack", title="Points by variable")
        st.plotly_chart(style(fig, 420), use_container_width=True)
    st.caption("Scaling: 600 points = 50:1 good:bad odds; every 20 points doubles the odds.")

# ---- 3. cut-off & fairness ----------------------------------------------------
with tab3:
    st.subheader("Choose an approval rate")
    pct = st.slider("Approve the safest … % of applicants", 50, 100, 85, 1)
    sel = cut[cut.approval_pct == pct].set_index("model")
    cols = st.columns(3)
    for col, m in zip(cols, MODEL_COLORS):
        r = sel.loc[m]
        col.markdown(f"**{m}**")
        col.metric("Default rate of approved", f"{r.bad_rate_approved:.2%}")
        col.metric("Defaulters declined", f"{r.bads_declined_share:.1%}")
        col.metric("Profit (bn)", f"{r.profit_bn:.3f}")
    if pct < 100:
        sc_cut = sel.loc["Scorecard", "score_cutoff"]
        st.caption(f"Scorecard cut-off at this rate: score ≥ {sc_cut:.0f} (applicants tied at the "
                   "cut-off score are split by rank so every model approves exactly the same share). "
                   f"Profit assumes "
                   f"+{meta['margin_good']:.0%} of the loan on a good and "
                   f"−{meta['loss_bad']:.0%} on a bad (illustrative).")

    a, b = st.columns(2)
    fig = go.Figure()
    for m, colr in MODEL_COLORS.items():
        d = cut[cut.model == m]
        fig.add_scatter(x=d.approval_pct, y=d.profit_bn, name=m, line=dict(color=colr, width=2),
                        hovertemplate="%{x}% approved: %{y:.3f} bn<extra>" + m + "</extra>")
    fig.add_vline(x=pct, line=dict(color=INK2, dash="dot", width=1))
    fig.update_layout(title="Profit vs approval rate", xaxis_title="Approval rate (%)",
                      yaxis_title="Profit (bn)")
    a.plotly_chart(style(fig), use_container_width=True)

    fig = go.Figure()
    for m, colr in MODEL_COLORS.items():
        d = cut[cut.model == m]
        fig.add_scatter(x=d.approval_pct, y=d.air_under30_vs_60plus, name=m,
                        line=dict(color=colr, width=2),
                        hovertemplate="%{x}% approved: ratio %{y:.2f}<extra>" + m + "</extra>")
    fig.add_hline(y=0.8, line=dict(color=INK2, dash="dash", width=1),
                  annotation_text="four-fifths rule (0.8)", annotation_position="bottom right")
    fig.add_vline(x=pct, line=dict(color=INK2, dash="dot", width=1))
    fig.update_layout(title="Approval-rate ratio: under 30 vs 60+", xaxis_title="Approval rate (%)",
                      yaxis_title="Adverse impact ratio")
    b.plotly_chart(style(fig), use_container_width=True)

    gm = sel["air_male_vs_female"]
    st.markdown(
        "**Fairness at this approval rate.** Male ÷ female approval ratio: "
        + ", ".join(f"{m} **{v:.2f}**" for m, v in gm.items())
        + ". Gender is not a model input in any model. The under-30 gap appears in all three "
        "models because job tenure and credit-history length act as proxies for age.")

    h = D["score_hist"]
    fig = go.Figure()
    fig.add_bar(x=h.score_from, y=h.n - h.bads, name="Good", marker_color=BLUE,
                hovertemplate="score %{x}: %{y} goods<extra></extra>")
    fig.add_bar(x=h.score_from, y=h.bads, name="Bad", marker_color=ORANGE,
                hovertemplate="score %{x}: %{y} bads<extra></extra>")
    if pct < 100:
        fig.add_vline(x=sel.loc["Scorecard", "score_cutoff"], line=dict(color=INK2, dash="dot"))
    fig.update_layout(barmode="stack", title="Scorecard score distribution (hold-out)",
                      xaxis_title="Score", yaxis_title="Applicants", bargap=0.05)
    st.plotly_chart(style(fig, 320), use_container_width=True)

# ---- 4. model insights --------------------------------------------------------
with tab4:
    a, b = st.columns(2)
    iv = D["iv_ranking"].head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=iv.iv, y=iv.feature, orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: IV %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Top 20 features by Information Value")
    a.plotly_chart(style(fig, 520), use_container_width=True)
    sh = D["shap_importance"].head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=sh.iloc[:, 1], y=sh.iloc[:, 0], orientation="h",
                           marker_color=ORANGE,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title="XGBoost full: mean |SHAP| (top 20)")
    b.plotly_chart(style(fig, 520), use_container_width=True)

    st.markdown("**Rating grades (scorecard, hold-out)**")
    gshow = grades[["grade", "score_min", "score_max", "n", "bad_rate"]].copy()
    gshow.columns = ["Grade", "Score from", "Score to", "Applicants", "Default rate"]
    st.dataframe(gshow.style.format({"Score from": "{:.0f}", "Score to": "{:.0f}",
                                     "Applicants": "{:,.0f}", "Default rate": "{:.1%}"}),
                 hide_index=True, use_container_width=True)
    with st.expander("Full scorecard points table"):
        st.dataframe(points[["feature", "bin", "share", "bad_rate", "woe", "points"]]
                     .style.format({"share": "{:.1%}", "bad_rate": "{:.1%}", "woe": "{:.3f}"}),
                     hide_index=True, use_container_width=True)
