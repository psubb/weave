"""
Streamlit dashboard for PostHog engineering impact analysis.
"""

import streamlit as st
import pandas as pd

from github_client import fetch_all_data
from analysis import compute_impact_scores, get_scoring_explanation


st.set_page_config(
    page_title="Engineering Impact Dashboard",
    page_icon="📊",
    layout="centered",
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load GitHub data with caching. Returns (prs_df, error_message)."""
    try:
        prs_df, _, _ = fetch_all_data(days=90, refresh=False)
        return prs_df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


def render_summary_metrics(prs_df: pd.DataFrame, impact_df: pd.DataFrame):
    """Render the summary metrics section."""
    col1, col2 = st.columns(2)
    
    total_prs = len(prs_df) if not prs_df.empty else 0
    active_engineers = len(impact_df) if not impact_df.empty else 0
    
    with col1:
        st.metric("Merged PRs", total_prs)
    with col2:
        st.metric("Active Engineers", active_engineers)


def render_top_engineers_table(impact_df: pd.DataFrame):
    """Render the top 5 engineers table."""
    st.subheader("Top 5 Impactful Engineers")
    
    if impact_df.empty:
        st.info("No impact data available yet. GitHub data may still be loading or rate-limited.")
        return
    
    top5 = impact_df.head(5).copy()
    
    display_cols = {
        "engineer": "Engineer",
        "impact_score": "Impact Score",
        "merged_pr_count": "Merged PRs",
        "delivery_score": "Delivery",
        "collaboration_score": "Collaboration",
    }
    
    available_cols = [c for c in display_cols.keys() if c in top5.columns]
    display_df = top5[available_cols].copy()
    
    for col in ["impact_score", "delivery_score", "collaboration_score"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(1)
    
    display_df.columns = [display_cols.get(c, c) for c in display_df.columns]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def render_score_breakdown_chart(impact_df: pd.DataFrame):
    """Render a bar chart comparing delivery vs collaboration for top engineers."""
    st.subheader("Impact Score Breakdown")
    
    if impact_df.empty:
        st.info("No data available for chart.")
        return
    
    required_cols = ["engineer", "delivery_score", "collaboration_score"]
    if not all(col in impact_df.columns for col in required_cols):
        st.warning("Score breakdown data is incomplete.")
        return
    
    top5 = impact_df.head(5).copy()
    
    if len(top5) == 0:
        st.info("No engineers to display.")
        return
    
    chart_data = pd.DataFrame({
        "Engineer": top5["engineer"],
        "Delivery": top5["delivery_score"].round(1),
        "Collaboration": top5["collaboration_score"].round(1),
    })
    chart_data = chart_data.set_index("Engineer")
    
    st.bar_chart(chart_data, height=250)
    
    st.caption("Delivery = code authored & merged | Collaboration = comment activity")


def render_scoring_explanation():
    """Render the scoring model explanation."""
    with st.expander("How is the Impact Score calculated?", expanded=False):
        st.markdown(get_scoring_explanation())


def main():
    st.title("Engineering Impact Dashboard")
    st.markdown("**PostHog/posthog** — Analysis of last 90 days of GitHub activity")
    st.caption("By Pranav Subbiah")
    
    with st.spinner("Loading GitHub data..."):
        prs_df, error = load_data()
    
    if error:
        st.error(f"Failed to load GitHub data: {error}")
        st.info("The dashboard will display with empty data. Try refreshing later.")
        prs_df = pd.DataFrame()
    
    if prs_df.empty and not error:
        st.warning("No data found. The local JSON may be missing or GitHub API may be rate-limited.")
    
    impact_df = compute_impact_scores(prs_df, pd.DataFrame())
    
    render_summary_metrics(prs_df, impact_df)
    render_scoring_explanation()
    render_top_engineers_table(impact_df)
    render_score_breakdown_chart(impact_df)


if __name__ == "__main__":
    main()
