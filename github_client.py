"""
GitHub GraphQL client for fetching PR data from PostHog/posthog.

JSON-first design: saves PR data to a local JSON file (data/merged_prs_posthog_90d.json).
Uses the JSON file by default, only fetches from GitHub API when file is missing or --refresh.
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "PostHog"
REPO_NAME = "posthog"
GRAPHQL_URL = "https://api.github.com/graphql"
DATA_DIR = Path("data")
PAGE_SIZE = 100
MAX_PAGES = 100
MAX_PRS_WARNING = 500
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0
REQUEST_TIMEOUT = 30
CHECKPOINT_INTERVAL = 5
EARLY_STOP_THRESHOLD = 2


# =============================================================================
# JSON Dataset Storage
# =============================================================================

def _get_data_file_path(days: int) -> Path:
    """Get the JSON file path for PR data."""
    return DATA_DIR / f"merged_prs_posthog_{days}d.json"


def _load_prs_from_json(days: int) -> list | None:
    """Load PR data from JSON file. Returns None if file doesn't exist."""
    file_path = _get_data_file_path(days)
    if not file_path.exists():
        return None
    with open(file_path) as f:
        return json.load(f)


def _save_prs_to_json(days: int, data: list) -> Path:
    """Save PR data to JSON file with pretty formatting. Returns the file path."""
    DATA_DIR.mkdir(exist_ok=True)
    file_path = _get_data_file_path(days)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    return file_path


# =============================================================================
# GitHub API
# =============================================================================

def _build_session() -> requests.Session:
    """Build a requests session with auth headers pre-configured."""
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable is required for GraphQL API")
    
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    })
    return session


def _graphql_request(session: requests.Session, query: str, variables: dict = None) -> dict:
    """
    Make a GraphQL request with retry/backoff for rate limits and server errors.
    
    Retries on: 403, 429 (rate limit), 500, 502, 503, 504 (server errors), timeouts.
    Returns the 'data' portion of the response.
    Raises an exception after all retries are exhausted.
    """
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(GRAPHQL_URL, json=payload, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                result = response.json()
                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown GraphQL error")
                    if "rate limit" in error_msg.lower():
                        wait_time = BACKOFF_FACTOR * (2 ** attempt)
                        print(f"  [Rate Limited] Retry {attempt + 1}/{MAX_RETRIES} in {wait_time:.0f}s...")
                        time.sleep(wait_time)
                        continue
                    raise Exception(f"GraphQL error: {error_msg}")
                return result.get("data", {})
            
            if response.status_code in (403, 429):
                retry_after = response.headers.get("Retry-After")
                reset_time = response.headers.get("X-RateLimit-Reset")
                
                if retry_after:
                    wait_time = float(retry_after)
                elif reset_time:
                    wait_time = max(0, int(reset_time) - time.time()) + 1
                else:
                    wait_time = BACKOFF_FACTOR * (2 ** attempt)
                
                wait_time = min(wait_time, 120)
                
                if attempt < MAX_RETRIES - 1:
                    print(f"  [Rate Limited] HTTP {response.status_code}, retry in {wait_time:.0f}s...")
                    time.sleep(wait_time)
                    continue
            
            if response.status_code in (500, 502, 503, 504):
                wait_time = BACKOFF_FACTOR * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    print(f"  [Server Error] HTTP {response.status_code}, retry {attempt + 1}/{MAX_RETRIES} in {wait_time:.0f}s...")
                    time.sleep(wait_time)
                    continue
                last_error = f"HTTP {response.status_code} after {MAX_RETRIES} retries"
            else:
                response.raise_for_status()
                
        except requests.exceptions.Timeout:
            wait_time = BACKOFF_FACTOR * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                print(f"  [Timeout] Retry {attempt + 1}/{MAX_RETRIES} in {wait_time:.0f}s...")
                time.sleep(wait_time)
                continue
            last_error = f"Timeout after {MAX_RETRIES} retries"
            
        except requests.exceptions.ConnectionError as e:
            wait_time = BACKOFF_FACTOR * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                print(f"  [Connection Error] Retry {attempt + 1}/{MAX_RETRIES} in {wait_time:.0f}s...")
                time.sleep(wait_time)
                continue
            last_error = f"Connection error: {e}"
    
    raise Exception(last_error or "Max retries exceeded for GraphQL request")


PULL_REQUESTS_QUERY = """
query($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      first: $first
      after: $after
      states: [MERGED]
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        createdAt
        mergedAt
        additions
        deletions
        changedFiles
        comments { totalCount }
        reviews { totalCount }
        author { login }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""


def _fetch_merged_prs_graphql(
    days: int,
    session: requests.Session,
) -> tuple[list[dict], bool, bool]:
    """
    Fetch merged PRs from the last N days using GraphQL.
    
    Returns:
        (list of PR dicts, early_stopped: bool, is_partial: bool)
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_ts = cutoff_date.timestamp()
    
    filtered_prs = []
    cursor = None
    page = 1
    consecutive_empty_pages = 0
    early_stopped = False
    is_partial = False
    
    print(f"  Fetching from GitHub API (cutoff: {cutoff_date.strftime('%Y-%m-%d')})...")
    
    while page <= MAX_PAGES:
        variables = {
            "owner": REPO_OWNER,
            "name": REPO_NAME,
            "first": PAGE_SIZE,
            "after": cursor,
        }
        
        try:
            data = _graphql_request(session, PULL_REQUESTS_QUERY, variables)
        except Exception as e:
            print(f"  [Error] Page {page} failed: {e}")
            if filtered_prs:
                print(f"  Returning partial data ({len(filtered_prs)} PRs collected so far)")
                is_partial = True
                break
            raise
        
        repo = data.get("repository")
        if not repo:
            print(f"  Page {page}: no repository data, stopping")
            break
        
        pull_requests = repo.get("pullRequests", {})
        nodes = pull_requests.get("nodes", [])
        page_info = pull_requests.get("pageInfo", {})
        
        if not nodes:
            print(f"  Page {page}: empty")
            break
        
        in_range_count = 0
        for pr in nodes:
            merged_at_str = pr.get("mergedAt")
            if not merged_at_str:
                continue
            
            merged_at = datetime.fromisoformat(merged_at_str.replace("Z", "+00:00"))
            if merged_at.timestamp() < cutoff_ts:
                continue
            
            in_range_count += 1
            author = pr.get("author")
            
            filtered_prs.append({
                "pr_number": pr.get("number"),
                "author": author.get("login") if author else "unknown",
                "created_at": pr.get("createdAt"),
                "merged_at": merged_at_str,
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "changed_files": pr.get("changedFiles", 0),
                "comments": pr.get("comments", {}).get("totalCount", 0),
                "review_comments": pr.get("reviews", {}).get("totalCount", 0),
            })
        
        should_log = (page % 10 == 0) or (in_range_count < PAGE_SIZE) or (page == 1)
        if should_log:
            rate_limit = data.get("rateLimit", {})
            remaining = rate_limit.get("remaining", "?")
            print(f"  Page {page}: {in_range_count} in range (total: {len(filtered_prs)}, API remaining: {remaining})")
        
        if in_range_count == 0:
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= EARLY_STOP_THRESHOLD:
                print(f"  Stopping early: {EARLY_STOP_THRESHOLD} consecutive pages with 0 PRs in range")
                early_stopped = True
                break
        else:
            consecutive_empty_pages = 0
        
        if not page_info.get("hasNextPage"):
            print(f"  No more pages available")
            break
        
        if page % CHECKPOINT_INTERVAL == 0 and filtered_prs:
            _save_prs_to_json(days, filtered_prs)
        
        cursor = page_info.get("endCursor")
        page += 1
    
    if page > MAX_PAGES:
        print(f"  Reached max page limit ({MAX_PAGES})")
    
    return filtered_prs, early_stopped, is_partial


# =============================================================================
# Main API
# =============================================================================

def fetch_all_data(
    days: int = 90,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fetch merged PRs from the last N days.
    
    JSON-first behavior:
    - If refresh=False and JSON file exists, loads from file immediately.
    - If refresh=True or no JSON file, fetches from GitHub API and saves to JSON.
    
    Args:
        days: Number of days to look back
        refresh: If True, force fetch from API even if JSON file exists
    
    Returns:
        (prs_df, reviews_df, metadata)
        - reviews_df is always empty
        - metadata contains: source, count, file_path, is_partial, early_stopped, runtime
    """
    start_time = time.time()
    file_path = _get_data_file_path(days)
    metadata = {
        "source": None,
        "count": 0,
        "file_path": str(file_path),
        "is_partial": False,
        "early_stopped": False,
    }
    
    # JSON-first: check for existing file unless refresh requested
    if not refresh:
        existing_data = _load_prs_from_json(days)
        if existing_data is not None:
            elapsed = time.time() - start_time
            print(f"  Loaded {len(existing_data)} PRs from {file_path}")
            metadata.update(source="json", count=len(existing_data), runtime=elapsed)
            return pd.DataFrame(existing_data), pd.DataFrame(), metadata
        else:
            print(f"  No local JSON found, fetching from GitHub...")
    else:
        print(f"  Refresh requested, fetching fresh PR data from GitHub...")
    
    # Fetch from API
    try:
        session = _build_session()
        pr_data, early_stopped, is_partial = _fetch_merged_prs_graphql(days, session)
    except ValueError as e:
        print(f"  [Error] {e}")
        return pd.DataFrame(), pd.DataFrame(), metadata
    except Exception as e:
        print(f"  [Error] Failed to fetch PRs: {e}")
        # Try to return existing JSON as fallback
        existing_data = _load_prs_from_json(days)
        if existing_data:
            print(f"  Returning existing JSON as fallback ({len(existing_data)} PRs)")
            elapsed = time.time() - start_time
            metadata.update(source="json (fallback)", count=len(existing_data), runtime=elapsed)
            return pd.DataFrame(existing_data), pd.DataFrame(), metadata
        return pd.DataFrame(), pd.DataFrame(), metadata
    
    if not pr_data:
        print("  No merged PRs found in the specified time range.")
        elapsed = time.time() - start_time
        metadata.update(source="api", count=0, runtime=elapsed)
        return pd.DataFrame(), pd.DataFrame(), metadata
    
    # Save to JSON
    saved_path = _save_prs_to_json(days, pr_data)
    print(f"  Saved {len(pr_data)} PRs to {saved_path}")
    
    elapsed = time.time() - start_time
    metadata.update(
        source="api",
        count=len(pr_data),
        is_partial=is_partial,
        early_stopped=early_stopped,
        runtime=elapsed,
    )
    
    if len(pr_data) > MAX_PRS_WARNING:
        print(f"  [Note] High PR count ({len(pr_data)}), which is unusual but OK.")
    
    return pd.DataFrame(pr_data), pd.DataFrame(), metadata


def fetch_merged_prs(days: int = 90, refresh: bool = False) -> pd.DataFrame:
    """Fetch merged PRs. Convenience wrapper around fetch_all_data."""
    prs_df, _, _ = fetch_all_data(days=days, refresh=refresh)
    return prs_df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch GitHub PR data for analysis")
    parser.add_argument("--refresh", action="store_true", help="Force refresh from GitHub API")
    parser.add_argument("--days", type=int, default=90, help="Number of days to look back")
    args = parser.parse_args()
    
    overall_start = time.time()
    
    print("=" * 60)
    print("GitHub PR Data Fetcher")
    print("=" * 60)
    
    if not GITHUB_TOKEN:
        print("\nWARNING: GITHUB_TOKEN not set.")
        print("JSON-only mode: will fail if no local data exists.")
    
    print(f"\nRepository: {REPO_OWNER}/{REPO_NAME}")
    print(f"Time window: {args.days} days")
    print(f"Refresh: {args.refresh}")
    print(f"Data file: {_get_data_file_path(args.days)}")
    
    print("\n" + "-" * 60)
    
    prs_df, reviews_df, metadata = fetch_all_data(days=args.days, refresh=args.refresh)
    
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Source: {metadata.get('source', 'unknown')}")
    print(f"Total PRs: {metadata.get('count', 0)}")
    print(f"File: {metadata.get('file_path', 'N/A')}")
    print(f"Partial: {metadata.get('is_partial', False)}")
    print(f"Early stopped: {metadata.get('early_stopped', False)}")
    print(f"Runtime: {metadata.get('runtime', 0):.2f}s")
    
    if not prs_df.empty:
        print(f"\nColumns: {list(prs_df.columns)}")
        print(f"\nFirst 5 rows:")
        print(prs_df.head().to_string())
    
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"Done in {overall_elapsed:.2f}s")
    print("=" * 60)
