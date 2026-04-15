"""
Boston 311 ETL Pipeline
Pulls 2020-2026 data from CKAN Datastore API (paginated), normalizes columns,
and saves to a single parquet file.

Target: 500K+ records for robust statistical analysis.

Usage:
    python etl_311.py                  # default: 2020-2026
    python etl_311.py --start 2015     # from 2015
    python etl_311.py --limit 1000     # quick test run
"""

import argparse
import time
import pandas as pd
import requests

# Resource IDs per year from Boston Open Data CKAN
RESOURCES = {
    2026: "1a0b420d-99f1-4887-9851-990b2a5a6e17",
    2025: "9d7c2214-4709-478a-a2e8-fb2020a5bb94",
    2024: "dff4d804-5031-443a-8409-8344efd0e5c8",
    2023: "e6013a93-1321-4f2a-bf91-8d8a02f1e62f",
    2022: "81a7b022-f8fc-4da5-80e4-b160058ca207",
    2021: "f53ebccd-bc61-49f9-83db-625f209c95f5",
    2020: "6ff6a6fd-3141-4440-a880-6f60a37fe789",
    2019: "ea2e4696-4a2d-429c-9807-d02eb92e0222",
    2018: "2be28d90-3a90-4af1-a3f6-f28c1e25880a",
    2017: "30022137-709d-465e-baae-ca155b51927d",
    2016: "b7ea6b1b-3ca4-4c5b-9713-6dc1db52379a",
    2015: "c9509ab4-6f6d-4b97-979a-0cf2a10c922b",
}

API_BASE = "https://data.boston.gov/api/3/action/datastore_search"
PAGE_SIZE = 5000  # CKAN default max per request
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Columns to keep (unified across all years)
KEEP_COLS = [
    "case_enquiry_id", "open_dt", "closed_dt", "sla_target_dt",
    "on_time", "case_status", "closure_reason", "case_title",
    "subject", "reason", "type", "queue", "department",
    "source", "location", "location_street_name", "location_zipcode",
    "neighborhood", "ward", "precinct",
    "fire_district", "pwd_district", "city_council_district", "police_district",
    "latitude", "longitude",
]


def fetch_resource(resource_id: str, year: int, limit: int | None = None) -> pd.DataFrame:
    """Fetch all records from a single CKAN datastore resource with pagination."""
    records = []
    offset = 0
    total = None

    while True:
        params = {"resource_id": resource_id, "limit": PAGE_SIZE, "offset": offset}

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(API_BASE, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, ValueError) as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retry {attempt + 1}/{MAX_RETRIES} for {year} offset={offset}: {e}")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to fetch {year} at offset={offset} after {MAX_RETRIES} retries") from e

        result = data["result"]
        if total is None:
            total = result["total"]
            print(f"  {year}: {total:,} records to fetch")

        batch = result["records"]
        if not batch:
            break

        records.extend(batch)
        offset += len(batch)

        if offset % 50000 == 0 or offset >= total:
            print(f"  {year}: {offset:,}/{total:,} ({offset / total:.0%})")

        if limit and offset >= limit:
            print(f"  {year}: hit --limit={limit}, stopping early")
            break

        # polite delay to avoid hammering the API
        time.sleep(0.3)

    df = pd.DataFrame(records)
    df["data_year"] = year
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the raw dataframe."""
    # keep only columns that exist
    cols_present = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols_present + ["data_year"]].copy()

    # parse dates
    for col in ["open_dt", "closed_dt", "sla_target_dt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # numeric coords
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # drop rows with no coordinates (can't do spatial join later)
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows missing coordinates ({dropped / before:.1%})")

    # ensure consistent string types for mixed-type columns
    for col in ["case_enquiry_id", "ward", "precinct", "fire_district",
                "pwd_district", "city_council_district", "police_district"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", pd.NA)

    # standardize zipcode
    df["location_zipcode"] = (
        df["location_zipcode"]
        .fillna("")
        .astype(str)
        .str.extract(r"(\d{5})", expand=False)
    )
    df["location_zipcode"] = df["location_zipcode"].replace("", pd.NA)

    # resolution time
    df["resolution_days"] = (df["closed_dt"] - df["open_dt"]).dt.total_seconds() / 86400
    # negative or zero-second closes are likely test/auto-close
    df.loc[df["resolution_days"] < 0, "resolution_days"] = pd.NA

    return df


def main():
    parser = argparse.ArgumentParser(description="Boston 311 ETL Pipeline")
    parser.add_argument("--start", type=int, default=2020, help="Start year (default: 2020)")
    parser.add_argument("--end", type=int, default=2026, help="End year inclusive (default: 2026)")
    parser.add_argument("--limit", type=int, default=None, help="Max records per year (for testing)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: same dir)")
    args = parser.parse_args()

    output_dir = "/Users/rayxu/CS506_Final_Project/problemstic"
    output_path = args.output or f"{output_dir}/boston_311_{args.start}_{args.end}.parquet"

    years = sorted([y for y in RESOURCES if args.start <= y <= args.end])
    print(f"Fetching years: {years}")

    frames = []
    for year in years:
        print(f"\n{'='*50}")
        print(f"Fetching {year}...")
        df = fetch_resource(RESOURCES[year], year, limit=args.limit)
        frames.append(df)

    print(f"\n{'='*50}")
    print("Concatenating and normalizing...")
    raw = pd.concat(frames, ignore_index=True)
    print(f"Raw records: {len(raw):,}")

    df = normalize(raw)
    print(f"Final records: {len(df):,}")
    print(f"\nRecords by year:\n{df['data_year'].value_counts().sort_index()}")
    print(f"\nCase status distribution:\n{df['case_status'].value_counts()}")

    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    import os
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
