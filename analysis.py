"""
Impact score computation for GitHub engineers based on PR and review activity.
"""

import pandas as pd


def get_pr_size_weight(changes: int) -> float:
    """Map PR size (additions + deletions) to a weight."""
    if changes < 50:
        return 1.0    # XS
    elif changes < 200:
        return 2.0    # S
    elif changes < 500:
        return 3.5    # M
    elif changes < 1000:
        return 5.5    # L
    else:
        return 7.0    # XL


def get_review_weight(state: str) -> float:
    """Map review state to a weight."""
    weights = {
        "APPROVED": 1.0,
        "CHANGES_REQUESTED": 1.75,
        "COMMENTED": 0.5,
    }
    return weights.get(state, 0.25)


def compute_delivery_scores(prs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delivery scores per engineer from authored PRs.
    
    Returns DataFrame with columns: engineer, delivery_score, merged_pr_count
    """
    if prs_df.empty:
        return pd.DataFrame(columns=["engineer", "delivery_score", "merged_pr_count"])
    
    df = prs_df.copy()
    df["additions"] = df["additions"].fillna(0)
    df["deletions"] = df["deletions"].fillna(0)
    df["changed_files"] = df["changed_files"].fillna(0)
    
    df["pr_size"] = df["additions"] + df["deletions"]
    df["size_weight"] = df["pr_size"].apply(get_pr_size_weight)
    df["breadth_bonus"] = 0.25 * df["changed_files"].clip(upper=10)
    df["pr_score"] = df["size_weight"] + df["breadth_bonus"]
    
    delivery = df.groupby("author").agg(
        delivery_score=("pr_score", "sum"),
        merged_pr_count=("pr_number", "count"),
    ).reset_index()
    delivery.rename(columns={"author": "engineer"}, inplace=True)
    
    return delivery


def compute_collaboration_scores_from_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute collaboration scores per engineer from review activity.
    
    Returns DataFrame with columns: engineer, collaboration_score, reviews_count
    """
    if reviews_df.empty:
        return pd.DataFrame(columns=["engineer", "collaboration_score", "reviews_count"])
    
    df = reviews_df.copy()
    df["review_weight"] = df["state"].apply(get_review_weight)
    
    collab = df.groupby("reviewer").agg(
        collaboration_score=("review_weight", "sum"),
        reviews_count=("pr_number", "count"),
    ).reset_index()
    collab.rename(columns={"reviewer": "engineer"}, inplace=True)
    
    return collab


def compute_collaboration_scores_fallback(prs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute collaboration scores using comment activity as a fallback
    when review data is unavailable.
    
    Returns DataFrame with columns: engineer, collaboration_score, reviews_count
    """
    if prs_df.empty:
        return pd.DataFrame(columns=["engineer", "collaboration_score", "reviews_count"])
    
    df = prs_df.copy()
    df["comments"] = df["comments"].fillna(0)
    df["review_comments"] = df["review_comments"].fillna(0)
    
    df["collab_proxy"] = 0.2 * df["comments"] + 0.35 * df["review_comments"]
    
    collab = df.groupby("author").agg(
        collaboration_score=("collab_proxy", "sum"),
    ).reset_index()
    collab.rename(columns={"author": "engineer"}, inplace=True)
    collab["reviews_count"] = 0  # No actual reviews
    
    return collab


def compute_consistency_scores(prs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute consistency scores based on merged PR count (capped at 10).
    
    Returns DataFrame with columns: engineer, consistency_score
    """
    if prs_df.empty:
        return pd.DataFrame(columns=["engineer", "consistency_score"])
    
    pr_counts = prs_df.groupby("author")["pr_number"].count().reset_index()
    pr_counts.columns = ["engineer", "pr_count"]
    pr_counts["consistency_score"] = pr_counts["pr_count"].clip(upper=10) * 0.4
    
    return pr_counts[["engineer", "consistency_score"]]


def compute_impact_scores(
    prs_df: pd.DataFrame,
    reviews_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute engineer impact scores from PR and review data.
    
    Scoring weights:
    - 55% delivery (PR authoring)
    - 30% collaboration (reviews or comment activity)
    - 15% consistency (regular contributions)
    
    Args:
        prs_df: DataFrame with PR data
        reviews_df: DataFrame with review data (can be None or empty)
    
    Returns:
        DataFrame with columns:
        - engineer
        - delivery_score
        - collaboration_score
        - consistency_score
        - impact_score
        - merged_pr_count
        - reviews_count
    """
    if prs_df.empty:
        return pd.DataFrame(columns=[
            "engineer", "delivery_score", "collaboration_score",
            "consistency_score", "impact_score", "merged_pr_count", "reviews_count"
        ])
    
    # Compute component scores
    delivery_df = compute_delivery_scores(prs_df)
    consistency_df = compute_consistency_scores(prs_df)
    
    # Use review data if available, otherwise fall back to comment-based proxy
    if reviews_df is not None and not reviews_df.empty:
        collab_df = compute_collaboration_scores_from_reviews(reviews_df)
    else:
        collab_df = compute_collaboration_scores_fallback(prs_df)
    
    # Merge all scores
    scores = delivery_df.merge(consistency_df, on="engineer", how="outer")
    scores = scores.merge(collab_df, on="engineer", how="outer")
    
    # Fill missing values
    scores["delivery_score"] = scores["delivery_score"].fillna(0)
    scores["collaboration_score"] = scores["collaboration_score"].fillna(0)
    scores["consistency_score"] = scores["consistency_score"].fillna(0)
    scores["merged_pr_count"] = scores["merged_pr_count"].fillna(0).astype(int)
    scores["reviews_count"] = scores["reviews_count"].fillna(0).astype(int)
    
    # Compute final impact score
    scores["impact_score"] = (
        0.55 * scores["delivery_score"] +
        0.30 * scores["collaboration_score"] +
        0.15 * scores["consistency_score"]
    )
    
    # Sort by impact score descending
    scores = scores.sort_values("impact_score", ascending=False).reset_index(drop=True)
    
    # Reorder columns
    scores = scores[[
        "engineer",
        "delivery_score",
        "collaboration_score",
        "consistency_score",
        "impact_score",
        "merged_pr_count",
        "reviews_count",
    ]]
    
    return scores


def get_scoring_explanation() -> str:
    """Return a plain-English explanation of the scoring model."""
    return """
**Impact Score Model**

The impact score measures engineer contributions across three dimensions:

1. **Delivery (55%)** - Code authored and merged
   - PR size weighted by complexity (XS=1, S=2, M=3.5, L=5.5, XL=7)
   - Small bonus for touching multiple files (breadth of changes)

2. **Collaboration (30%)** - Reviewing others' code
   - Approvals: 1 point each
   - Change requests: 1.75 points (thorough review)
   - Comments: 0.5 points
   - *Fallback: If review data unavailable, uses comment activity on authored PRs*

3. **Consistency (15%)** - Regular contributions
   - Based on number of merged PRs (capped at 10)
   - Rewards steady, sustained output

**Score Interpretation**
- Higher scores indicate more impactful contributions
- A balanced engineer contributes across all three dimensions
- Scores are relative—compare within the same time period
""".strip()


def get_scoring_explanation_short() -> str:
    """Return a short one-liner about the scoring model."""
    return "Impact = 55% Delivery + 30% Collaboration + 15% Consistency"


if __name__ == "__main__":
    from github_client import fetch_all_data
    
    print("=" * 60)
    print("Analysis Module Sanity Check")
    print("=" * 60)
    
    print("\nLoading data from GitHub client...")
    prs_df, reviews_df, _ = fetch_all_data(days=90, refresh=False)
    
    print(f"\nInput data:")
    print(f"  PRs: {len(prs_df)} rows")
    print(f"  Reviews: {len(reviews_df)} rows")
    
    print("\nComputing impact scores...")
    scores_df = compute_impact_scores(prs_df, reviews_df)
    
    print(f"\nOutput: {len(scores_df)} engineers scored")
    print(f"Columns: {list(scores_df.columns)}")
    
    print("\n" + "-" * 60)
    print("TOP 10 ENGINEERS BY IMPACT SCORE")
    print("-" * 60)
    
    top10 = scores_df.head(10).copy()
    top10["impact_score"] = top10["impact_score"].round(2)
    top10["delivery_score"] = top10["delivery_score"].round(2)
    top10["collaboration_score"] = top10["collaboration_score"].round(2)
    top10["consistency_score"] = top10["consistency_score"].round(2)
    
    print(top10.to_string(index=False))
    
    print("\n" + "-" * 60)
    print("SCORING MODEL")
    print("-" * 60)
    print(get_scoring_explanation())
    
    print("\n" + "=" * 60)
    print("Sanity check complete!")
    print("=" * 60)
