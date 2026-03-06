"""
GitHub API client for fetching PR and review data from PostHog/posthog.
"""

import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "PostHog"
REPO_NAME = "posthog"
BASE_URL = "https://api.github.com"
CACHE_DIR = Path(".cache")
MAX_WORKERS = 4
REQUEST_DELAY = 0.1  # Small delay between requests to avoid burst rate limiting
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.0  # Exponential backoff: 1s, 2s, 4s, 8s, 16s

# Thread-safe progress counter and rate limit state
_progress_lock = Lock()
_progress_count = 0
_rate_limit_lock = Lock()
_rate_limit_reset_time = 0

# Global session for connection reuse
_session: requests.Session | None = None
_session_lock = Lock()


def _get_session() -> requests.Session:
    """Get or create a thread-safe requests session with connection pooling."""
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
            # Configure retry strategy for non-rate-limit errors
            retry_strategy = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=MAX_WORKERS + 2)
            _session.mount("https://", adapter)
            _session.mount("http://", adapter)
        return _session


def _get_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _check_rate_limit(response: requests.Response) -> None:
    """Check rate limit headers and sleep if we're close to the limit."""
    global _rate_limit_reset_time
    
    remaining = response.headers.get("X-RateLimit-Remaining")
    reset_time = response.headers.get("X-RateLimit-Reset")
    
    if remaining is not None and reset_time is not None:
        remaining = int(remaining)
        reset_timestamp = int(reset_time)
        
        if remaining < 10:
            with _rate_limit_lock:
                _rate_limit_reset_time = reset_timestamp
            wait_time = max(0, reset_timestamp - time.time()) + 1
            print(f"\n  [Rate Limit] Only {remaining} requests remaining. Waiting {wait_time:.0f}s until reset...")
            time.sleep(wait_time)


def _handle_rate_limit_error(response: requests.Response) -> float:
    """Handle 403/429 rate limit errors. Returns seconds to wait."""
    reset_time = response.headers.get("X-RateLimit-Reset")
    retry_after = response.headers.get("Retry-After")
    
    if retry_after:
        return float(retry_after)
    elif reset_time:
        wait_time = max(0, int(reset_time) - time.time()) + 1
        return wait_time
    else:
        return 60  # Default wait time if no header info


def _request_with_retry(method: str, url: str, params: dict = None) -> requests.Response:
    """Make a request with exponential backoff retry for rate limits."""
    session = _get_session()
    headers = _get_headers()
    
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY)  # Small delay to prevent burst
        
        response = session.request(method, url, headers=headers, params=params)
        
        if response.status_code == 200:
            _check_rate_limit(response)
            return response
        
        if response.status_code in (403, 429):
            # Rate limited
            if attempt < MAX_RETRIES - 1:
                wait_time = _handle_rate_limit_error(response)
                # Cap wait time at 5 minutes for sanity
                wait_time = min(wait_time, 300)
                # Add exponential backoff on top
                backoff = BACKOFF_FACTOR * (2 ** attempt)
                total_wait = max(wait_time, backoff)
                
                print(f"\n  [Rate Limited] HTTP {response.status_code} on {url}")
                print(f"  [Rate Limited] Retry {attempt + 1}/{MAX_RETRIES} in {total_wait:.0f}s...")
                time.sleep(total_wait)
                continue
            else:
                print(f"\n  [Rate Limited] Max retries exceeded for {url}")
                response.raise_for_status()
        
        # Non-rate-limit error
        response.raise_for_status()
    
    return response


def _get_cache_path(cache_key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    filename = hashlib.md5(cache_key.encode()).hexdigest() + ".json"
    return CACHE_DIR / filename


def _load_from_cache(cache_key: str, max_age_hours: int = 1) -> list | None:
    cache_path = _get_cache_path(cache_key)
    if not cache_path.exists():
        return None
    
    modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    if datetime.now(timezone.utc) - modified_time > timedelta(hours=max_age_hours):
        return None
    
    with open(cache_path) as f:
        return json.load(f)


def _save_to_cache(cache_key: str, data: list) -> None:
    cache_path = _get_cache_path(cache_key)
    with open(cache_path, "w") as f:
        json.dump(data, f)


def _paginate_request(url: str, params: dict = None) -> list:
    """Fetch all pages from a paginated GitHub API endpoint."""
    results = []
    params = params or {}
    params["per_page"] = 100
    
    while url:
        response = _request_with_retry("GET", url, params=params)
        results.extend(response.json())
        
        url = response.links.get("next", {}).get("url")
        params = {}  # params are included in the next URL
    
    return results


def _fetch_pr_detail(pr_number: int) -> dict:
    """Fetch detailed PR data including additions, deletions, changed_files."""
    url = f"{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}"
    response = _request_with_retry("GET", url)
    return response.json()


def _fetch_pr_reviews_single(pr_number: int) -> list[dict]:
    """Fetch reviews for a single PR."""
    url = f"{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/reviews"
    reviews = _paginate_request(url)
    result = []
    for review in reviews:
        if review["user"] is None:
            continue
        result.append({
            "pr_number": pr_number,
            "reviewer": review["user"]["login"],
            "state": review["state"],
            "submitted_at": review["submitted_at"],
        })
    return result


def _fetch_pr_detail_and_reviews(pr_number: int, total: int) -> tuple[dict, list[dict]]:
    """Fetch both PR detail and reviews for a single PR."""
    global _progress_count
    
    detail = _fetch_pr_detail(pr_number)
    reviews = _fetch_pr_reviews_single(pr_number)
    
    with _progress_lock:
        _progress_count += 1
        if _progress_count % 25 == 0 or _progress_count == total:
            print(f"  Progress: {_progress_count}/{total} PRs processed")
    
    return detail, reviews


def _fetch_merged_pr_numbers(days: int) -> list[dict]:
    """
    Fetch basic info for merged PRs from the last N days.
    Returns list of dicts with pr_number, author, created_at, merged_at.
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    pulls_url = f"{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
    params = {
        "state": "closed",
        "sort": "updated",
        "direction": "desc",
        "per_page": 100,
    }
    
    merged_prs = []
    page = 1
    stop_pagination = False
    
    print(f"  Scanning closed PRs (page {page})...", end="", flush=True)
    
    while not stop_pagination:
        params["page"] = page
        response = _request_with_retry("GET", pulls_url, params=params)
        prs = response.json()
        
        if not prs:
            break
        
        for pr in prs:
            if pr["merged_at"] is None:
                continue
            
            merged_at = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
            
            if merged_at < cutoff_date:
                stop_pagination = True
                break
            
            merged_prs.append({
                "pr_number": pr["number"],
                "author": pr["user"]["login"] if pr["user"] else "unknown",
                "created_at": pr["created_at"],
                "merged_at": pr["merged_at"],
            })
        
        if prs:
            last_updated = prs[-1].get("updated_at")
            if last_updated:
                last_updated_dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                if last_updated_dt < cutoff_date:
                    stop_pagination = True
        
        page += 1
        if not stop_pagination:
            print(f" {page}...", end="", flush=True)
    
    print(f" done ({len(merged_prs)} merged PRs found)")
    
    return merged_prs


def fetch_all_data(days: int = 90, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch merged PRs and their reviews from the last N days.
    Uses concurrent fetching for speed with rate limit handling.
    
    Returns:
        (prs_df, reviews_df)
    """
    global _progress_count
    
    start_time = time.time()
    
    pr_cache_key = f"merged_prs_{REPO_OWNER}_{REPO_NAME}_{days}"
    review_cache_key = f"pr_reviews_{REPO_OWNER}_{REPO_NAME}_{days}"
    
    if use_cache:
        cached_prs = _load_from_cache(pr_cache_key)
        cached_reviews = _load_from_cache(review_cache_key)
        if cached_prs is not None and cached_reviews is not None:
            elapsed = time.time() - start_time
            print(f"  Loaded from cache in {elapsed:.1f}s")
            return pd.DataFrame(cached_prs), pd.DataFrame(cached_reviews)
    
    # Step 1: Get list of merged PRs (basic info only)
    print("\nStep 1: Identifying merged PRs...")
    merged_prs_basic = _fetch_merged_pr_numbers(days)
    
    if not merged_prs_basic:
        print("  No merged PRs found in the specified time range.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Step 2: Fetch PR details and reviews concurrently
    print(f"\nStep 2: Fetching details and reviews for {len(merged_prs_basic)} PRs...")
    print(f"  Using {MAX_WORKERS} concurrent workers with rate limit handling")
    _progress_count = 0
    
    pr_data = []
    review_data = []
    failed_prs = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _fetch_pr_detail_and_reviews, 
                pr["pr_number"], 
                len(merged_prs_basic)
            ): pr
            for pr in merged_prs_basic
        }
        
        for future in as_completed(futures):
            basic_pr = futures[future]
            try:
                detail, reviews = future.result()
                
                pr_data.append({
                    "pr_number": basic_pr["pr_number"],
                    "author": basic_pr["author"],
                    "created_at": basic_pr["created_at"],
                    "merged_at": basic_pr["merged_at"],
                    "additions": detail.get("additions", 0),
                    "deletions": detail.get("deletions", 0),
                    "changed_files": detail.get("changed_files", 0),
                    "comments": detail.get("comments", 0),
                    "review_comments": detail.get("review_comments", 0),
                })
                
                review_data.extend(reviews)
                
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (403, 429):
                    print(f"\n  [Rate Limit Error] PR #{basic_pr['pr_number']}: {e}")
                else:
                    print(f"\n  [HTTP Error] PR #{basic_pr['pr_number']}: {e}")
                failed_prs.append(basic_pr["pr_number"])
            except Exception as e:
                print(f"\n  [Error] PR #{basic_pr['pr_number']}: {e}")
                failed_prs.append(basic_pr["pr_number"])
    
    if failed_prs:
        print(f"\n  Warning: Failed to fetch {len(failed_prs)} PRs: {failed_prs[:10]}{'...' if len(failed_prs) > 10 else ''}")
    
    # Cache results
    if use_cache:
        if pr_data:
            _save_to_cache(pr_cache_key, pr_data)
        if review_data:
            _save_to_cache(review_cache_key, review_data)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({len(pr_data)} PRs, {len(review_data)} reviews)")
    
    return pd.DataFrame(pr_data), pd.DataFrame(review_data)


# Keep these for backward compatibility
def fetch_merged_prs(days: int = 90, use_cache: bool = True) -> pd.DataFrame:
    """Fetch merged PRs. Delegates to fetch_all_data for efficiency."""
    prs_df, _ = fetch_all_data(days=days, use_cache=use_cache)
    return prs_df


def fetch_pr_reviews(pr_numbers: list[int], use_cache: bool = True) -> pd.DataFrame:
    """Fetch PR reviews. For new code, prefer fetch_all_data instead."""
    cache_key = f"pr_reviews_{REPO_OWNER}_{REPO_NAME}_{hash(tuple(sorted(pr_numbers)))}"
    
    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            return pd.DataFrame(cached)
    
    review_data = []
    total = len(pr_numbers)
    failed = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_pr_reviews_single, pr_num): pr_num
            for pr_num in pr_numbers
        }
        
        completed = 0
        for future in as_completed(futures):
            pr_num = futures[future]
            try:
                reviews = future.result()
                review_data.extend(reviews)
                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(f"  Progress: {completed}/{total} PRs processed")
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (403, 429):
                    print(f"\n  [Rate Limit Error] PR #{pr_num}: {e}")
                else:
                    print(f"\n  [HTTP Error] PR #{pr_num}: {e}")
                failed.append(pr_num)
            except Exception as e:
                print(f"\n  [Error] PR #{pr_num}: {e}")
                failed.append(pr_num)
    
    if failed:
        print(f"\n  Warning: Failed to fetch reviews for {len(failed)} PRs")
    
    if use_cache and review_data:
        _save_to_cache(cache_key, review_data)
    
    return pd.DataFrame(review_data)


if __name__ == "__main__":
    overall_start = time.time()
    
    print("=" * 60)
    print("GitHub Client Sanity Check")
    print("=" * 60)
    
    if not GITHUB_TOKEN:
        print("\nWARNING: GITHUB_TOKEN not set. API rate limits will be very low.")
        print("Set it with: export GITHUB_TOKEN=your_token\n")
    else:
        print("\nGITHUB_TOKEN detected.")
    
    print(f"\nFetching data from {REPO_OWNER}/{REPO_NAME}...")
    print(f"Configuration: MAX_WORKERS={MAX_WORKERS}, REQUEST_DELAY={REQUEST_DELAY}s")
    
    prs_df, reviews_df = fetch_all_data(days=90, use_cache=True)
    
    print("\n" + "-" * 60)
    print("MERGED PRs")
    print("-" * 60)
    print(f"Total PRs fetched: {len(prs_df)}")
    if not prs_df.empty:
        print(f"\nColumns: {list(prs_df.columns)}")
        print(f"\nFirst 5 rows:")
        print(prs_df.head().to_string())
    
    print("\n" + "-" * 60)
    print("PR REVIEWS")
    print("-" * 60)
    print(f"Total review events fetched: {len(reviews_df)}")
    if not reviews_df.empty:
        print(f"\nColumns: {list(reviews_df.columns)}")
        print(f"\nFirst 5 rows:")
        print(reviews_df.head().to_string())
        print(f"\nReview states distribution:")
        print(reviews_df["state"].value_counts().to_string())
    
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"Sanity check complete! Total runtime: {overall_elapsed:.1f}s")
    print("=" * 60)
