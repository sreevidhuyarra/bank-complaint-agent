"""Pull and cache consumer complaints from the public CFPB Complaint Database.

The CFPB search API is public domain and needs no key or account:
https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/

Usage
-----
    python -m data.fetch_cfpb                     # default scope from config.py
    python -m data.fetch_cfpb --list-companies schwab
    python -m data.fetch_cfpb --company "WELLS FARGO & COMPANY" --max 500
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from typing import Iterable, Iterator

import pandas as pd
import requests

from config import (
    MAX_COMPLAINTS_PER_COMPANY,
    RAW_COMPLAINTS_CSV,
    TARGET_COMPANIES,
    default_date_range,
)

API_ROOT = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
SUGGEST_COMPANY = API_ROOT + "_suggest_company/"

# The search API ignores the documented `frm` offset — every offset returns the
# same first page — so results are paged by *date window* instead. Each month is
# requested whole, with `size` set high enough to exhaust it in one call. The API
# caps a single response at 10,000 hits.
MAX_RESPONSE_SIZE = 10_000
REQUEST_TIMEOUT = 180
RETRIES = 3

# Fields kept from the API's `_source`. `complaint_what_happened` is the API's
# name for what the public CSV export calls `consumer_complaint_narrative`.
FIELDS = [
    "complaint_id",
    "date_received",
    "company",
    "state",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "complaint_what_happened",
    "company_response",
    "timely",
    "submitted_via",
]


def suggest_companies(text: str) -> list[str]:
    """Resolve a partial company name to the exact strings the API filters on."""
    resp = requests.get(SUGGEST_COMPANY, params={"text": text}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return list(resp.json())


def _get(params: dict) -> dict:
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            resp = requests.get(API_ROOT, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"CFPB API request failed after {RETRIES} attempts: {last_error}")


def month_windows(date_min: str, date_max: str) -> list[tuple[str, str]]:
    """Split a date range into [start, end] month windows, newest first."""
    start = datetime.strptime(date_min, "%Y-%m-%d").date().replace(day=1)
    end = datetime.strptime(date_max, "%Y-%m-%d").date()
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        windows.append((cursor.isoformat(), min(next_month - timedelta(days=1), end).isoformat()))
        cursor = next_month
    return list(reversed(windows))


def fetch_company(
    company: str,
    date_min: str,
    date_max: str,
    max_records: int = MAX_COMPLAINTS_PER_COMPANY,
    products: Iterable[str] | None = None,
) -> Iterator[dict]:
    """Yield complaint records for one company, newest month first.

    Only complaints that carry a consumer narrative are requested — the rest are
    useless to both the retrieval and the linguistic-risk agents.
    """
    fetched = 0
    for window_min, window_max in month_windows(date_min, date_max):
        if fetched >= max_records:
            return
        params = {
            "company": company,
            "date_received_min": window_min,
            "date_received_max": window_max,
            "has_narrative": "true",
            "size": min(MAX_RESPONSE_SIZE, max_records - fetched),
            "sort": "created_date_desc",
            "no_aggs": "true",
        }
        if products:
            params["product"] = list(products)

        hits = _get(params).get("hits", {}).get("hits", [])
        for hit in hits:
            source = hit.get("_source", {})
            yield {field: source.get(field) for field in FIELDS}
        fetched += len(hits)


def fetch_all(
    companies: Iterable[str],
    date_min: str,
    date_max: str,
    max_per_company: int = MAX_COMPLAINTS_PER_COMPANY,
) -> pd.DataFrame:
    rows: list[dict] = []
    for company in companies:
        print(f"  fetching {company} ...", end="", flush=True)
        before = len(rows)
        rows.extend(fetch_company(company, date_min, date_max, max_per_company))
        print(f" {len(rows) - before} complaints")
    return pd.DataFrame(rows, columns=FIELDS)


def main(argv: list[str] | None = None) -> int:
    date_min, date_max = default_date_range()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-companies", metavar="TEXT",
                        help="resolve a partial company name and exit")
    parser.add_argument("--company", action="append", dest="companies",
                        help="exact CFPB company name (repeatable); defaults to config.TARGET_COMPANIES")
    parser.add_argument("--date-min", default=date_min)
    parser.add_argument("--date-max", default=date_max)
    parser.add_argument("--max", type=int, default=MAX_COMPLAINTS_PER_COMPANY,
                        help="max complaints per company")
    parser.add_argument("--out", default=str(RAW_COMPLAINTS_CSV))
    args = parser.parse_args(argv)

    if args.list_companies:
        matches = suggest_companies(args.list_companies)
        print("\n".join(matches) if matches else "(no matching companies)")
        return 0

    companies = args.companies or TARGET_COMPANIES
    print(f"CFPB pull · {args.date_min} → {args.date_max} · {len(companies)} companies")
    frame = fetch_all(companies, args.date_min, args.date_max, args.max)

    if frame.empty:
        print("No complaints returned — check the company names with --list-companies.")
        return 1

    frame = frame.drop_duplicates(subset="complaint_id")
    frame.to_csv(args.out, index=False)
    print(f"\nWrote {len(frame):,} complaints to {args.out}")
    print(frame["company"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
