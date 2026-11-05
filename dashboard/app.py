"""Streamlit review dashboard.

Run:  python -m streamlit run dashboard/app.py
Settings (environment): FD_DB_PATH (default data/decisions.db), FD_DRIFT_REPORT
(data/drift_report.json), FD_TEST_RESULTS (reports/test_results.json).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `api` and `ml` import

import altair as alt  # noqa: E402
import streamlit as st  # noqa: E402

from api.store import DecisionStore  # noqa: E402
from dashboard.data import (  # noqa: E402
    activity_frame,
    alerts_frame,
    queue_frame,
    read_json,
    reviewer_precision,
)

DB_PATH = Path(os.environ.get("FD_DB_PATH", "data/decisions.db"))
DRIFT_REPORT = Path(os.environ.get("FD_DRIFT_REPORT", "data/drift_report.json"))
TEST_RESULTS = Path(os.environ.get("FD_TEST_RESULTS", "reports/test_results.json"))

st.set_page_config(page_title="Fraud review", layout="wide")
st.title("Fraud review")
st.caption("Synthetic data only. A learning project, not a real fraud system.")

if not DB_PATH.exists():
    st.warning(f"No decisions yet: {DB_PATH} not found. Run `python -m api.load_queue` first.")
    st.stop()

store = DecisionStore(DB_PATH)
drift = read_json(DRIFT_REPORT)
if drift and drift.get("warning"):
    st.error("Model drift warning: " + "; ".join(drift["warnings"]))

queue_tab, overview_tab = st.tabs(["Review queue", "Overview"])

with queue_tab:
    pending_only = st.toggle("Only cases without a verdict", value=True)
    queue = queue_frame(store, pending_only=pending_only)
    if queue.empty:
        st.success("Nothing to review.")
    else:
        st.dataframe(queue[["id", "score", "decision", "step", "type", "amount", "name_orig",
                            "name_dest", "review_label"]],
                     hide_index=True, width="stretch",
                     column_config={"score": st.column_config.NumberColumn(format="%.3f"),
                                    "amount": st.column_config.NumberColumn(format="%.2f")})
        case_id = st.selectbox("Case", queue["id"].tolist(),
                               format_func=lambda i: f"#{i}  score "
                               f"{float(queue.loc[queue['id'] == i, 'score'].iloc[0]):.3f}")
        case = store.get(int(case_id))
        left, right = st.columns([2, 3])
        with left:
            st.subheader(f"Case #{case['id']}: {case['decision']}")
            st.metric("Fraud score", f"{case['score']:.3f}")
            st.write(f"**{case['type']}** of **{case['amount']:,.2f}** at step {case['step']}  \n"
                     f"from `{case['name_orig']}` to `{case['name_dest']}`")
            st.markdown("**Why it was flagged**")
            for reason in case["reasons"] or ["(no reasons recorded)"]:
                st.markdown(f"- {reason}")
            c1, c2 = st.columns(2)
            if c1.button("Confirm fraud", type="primary", width="stretch"):
                store.add_review(case["id"], "fraud", "dashboard")
                st.rerun()
            if c2.button("False alarm", width="stretch"):
                store.add_review(case["id"], "false_alarm", "dashboard")
                st.rerun()
        with right:
            st.markdown(f"**Sender's recent activity** (`{case['name_orig']}`)")
            act = activity_frame(store, case["name_orig"], case["step"])
            st.dataframe(act, hide_index=True, width="stretch")

with overview_tab:
    stats = store.stats()
    d = stats["decisions"]
    precision = reviewer_precision(stats)
    c = st.columns(5)
    c[0].metric("Scored", f"{sum(d.values()):,}")
    c[1].metric("Sent for review", f"{d['review']:,}")
    c[2].metric("Blocked", f"{d['block']:,}")
    p50, p99 = stats["latency_us_p50"], stats["latency_us_p99"]
    c[3].metric("Scoring latency p50 / p99",
                f"{p50:.0f} / {p99:.0f} µs" if p50 is not None else "n/a")
    c[4].metric("Reviewer-confirmed precision",
                f"{precision:.2f}" if precision is not None else "no verdicts yet",
                help=f"{stats['reviews']['fraud']} confirmed fraud, "
                     f"{stats['reviews']['false_alarm']} false alarms")
    st.markdown("**Alerts over time** (1 step = 1 hour)")
    st.altair_chart(
        alt.Chart(alerts_frame(store)).mark_bar().encode(
            x=alt.X("step:Q", title="step"), y=alt.Y("alerts:Q", stack=True),
            color=alt.Color("decision:N", scale=alt.Scale(domain=["review", "block"]))),
        width="stretch")
    if drift:
        st.markdown("**Feature drift** (PSI vs training data; above 0.25 = shifted)")
        st.dataframe([{"feature": k, "PSI": v} for k, v in drift["psi"].items()],
                     hide_index=True)
    results = read_json(TEST_RESULTS)
    if results:
        st.markdown(f"**Test-set results** ({results['test_rows']:,} transactions, "
                    f"{results['test_fraud']} fraud)")
        st.dataframe([{"method": k, **v} for k, v in results["methods"].items()],
                     hide_index=True)
