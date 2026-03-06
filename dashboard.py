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
    layout="wide",
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load GitHub data with caching. Returns (prs_df, reviews_df, error_message)."""
    try:
        prs_df, reviews_df = fetch_all_data(days=90, use_cache=True)
        return prs_df, reviews_df, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), str(e)


def safe_get_column(df: pd.DataFrame, col: str, default=0):
    """Safely get a column from a DataFrame, returning default if missing."""
    if df.empty or col not in df.columns:
        return pd.Series([default] * len(df)) if not df.empty else pd.Series([])
    return df[col].fillna(default)


def render_summary_metrics(prs_df: pd.DataFrame, reviews_df: pd.DataFrame, impact_df: pd.DataFrame):
    """Render the summary metrics section."""
    col1, col2, col3 = st.columns(3)
    
    total_prs = len(prs_df) if not prs_df.empty else 0
    total_reviews = len(reviews_df) if not reviews_df.empty else 0
    active_engineers = len(impact_df) if not impact_df.empty else 0
    
    with col1:
        st.metric("Merged PRs", total_prs)
    with col2:
        st.metric("Review Events", total_reviews)
    with col3:
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
        "reviews_count": "Reviews",
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
    
    st.bar_chart(chart_data, height=300)
    
    st.caption("Delivery = code authored & merged | Collaboration = code reviews given")


def render_engineer_drilldown(prs_df: pd.DataFrame, reviews_df: pd.DataFrame, impact_df: pd.DataFrame):
    """Render the engineer drilldown section."""
    st.subheader("Engineer Drilldown")
    
    if impact_df.empty:
        st.info("No engineers available for drilldown.")
        return
    
    engineers = impact_df["engineer"].tolist() if "engineer" in impact_df.columns else []
    
    if not engineers:
        st.info("No engineers found.")
        return
    
    selected = st.selectbox("Select an engineer", engineers, index=0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Recent Merged PRs**")
        if prs_df.empty or "author" not in prs_df.columns:
            st.caption("No PR data available.")
        else:
            engineer_prs = prs_df[prs_df["author"] == selected].copy()
            if engineer_prs.empty:
                st.caption(f"No merged PRs found for {selected}.")
            else:
                engineer_prs = engineer_prs.head(5)
                pr_display = pd.DataFrame({
                    "PR": safe_get_column(engineer_prs, "pr_number", ""),
                    "Merged": pd.to_datetime(safe_get_column(engineer_prs, "merged_at", "")).dt.strftime("%Y-%m-%d"),
                    "+/-": safe_get_column(engineer_prs, "additions", 0).astype(int).astype(str) + " / " + safe_get_column(engineer_prs, "deletions", 0).astype(int).astype(str),
                    "Files": safe_get_column(engineer_prs, "changed_files", 0).astype(int),
                })
                st.dataframe(pr_display, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Recent Reviews**")
        if reviews_df.empty or "reviewer" not in reviews_df.columns:
            st.caption("No review data available.")
        else:
            engineer_reviews = reviews_df[reviews_df["reviewer"] == selected].copy()
            if engineer_reviews.empty:
                st.caption(f"No reviews found for {selected}.")
            else:
                engineer_reviews = engineer_reviews.head(5)
                review_display = pd.DataFrame({
                    "PR": safe_get_column(engineer_reviews, "pr_number", ""),
                    "State": safe_get_column(engineer_reviews, "state", ""),
                    "Date": pd.to_datetime(safe_get_column(engineer_reviews, "submitted_at", "")).dt.strftime("%Y-%m-%d"),
                })
                st.dataframe(review_display, use_container_width=True, hide_index=True)


def render_scoring_explanation():
    """Render the scoring model explanation."""
    with st.expander("How is the Impact Score calculated?", expanded=False):
        st.markdown(get_scoring_explanation())


def main():
    st.title("Engineering Impact Dashboard")
    st.markdown("**PostHog/posthog** — Analysis of last 90 days of GitHub activity")
    
    st.divider()
    
    with st.spinner("Loading GitHub data..."):
        prs_df, reviews_df, error = load_data()
    
    if error:
        st.error(f"Failed to load GitHub data: {error}")
        st.info("The dashboard will display with empty data. Try refreshing later.")
        prs_df = pd.DataFrame()
        reviews_df = pd.DataFrame()
    
    if prs_df.empty and reviews_df.empty and not error:
        st.warning("No data found. The cache may be empty or GitHub API may be rate-limited.")
    
    impact_df = compute_impact_scores(prs_df, reviews_df)
    
    render_summary_metrics(prs_df, reviews_df, impact_df)
    
    st.divider()
    
    col_left, col_right = st.columns([1.2, 0.8])
    
    with col_left:
        render_top_engineers_table(impact_df)
    
    with col_right:
        render_score_breakdown_chart(impact_df)
    
    st.divider()
    
    render_engineer_drilldown(prs_df, reviews_df, impact_df)
    
    st.divider()
    
    render_scoring_explanation()
    
    st.caption("Data refreshes hourly. Scores are relative to the 90-day window.")


if __name__ == "__main__":
    main()
