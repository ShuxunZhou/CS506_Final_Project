"""
Download SVI component variables at Census Tract level for Massachusetts
directly from the Census Bureau ACS 5-Year API (2022).

No API key needed for small requests. Produces the same EP_ percentage
features used in SVI, computed from raw ACS numerators/denominators.

Output: data/svi_2022_ma_tract.csv
"""

import json
import urllib.request
import pandas as pd

BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

# ACS variables needed to compute SVI percentage features
# Each tuple: (numerator_var, denominator_var, output_name, description)
SVI_VARS = [
    # EP_POV150: % below 150% poverty level
    # C17002: ratio of income to poverty level in past 12 months
    # 002=<0.50, 003=0.50-0.99, 004=1.00-1.24, 005=1.25-1.49
    ("C17002_002E,C17002_003E,C17002_004E,C17002_005E", "C17002_001E", "EP_POV150",
     "% Below 150% Poverty"),
    # EP_UNEMP: % unemployed (civilian labor force 16+)
    ("B23025_005E", "B23025_003E", "EP_UNEMP",
     "% Unemployed"),
    # EP_NOHSDP: % age 25+ with no high school diploma
    ("B15003_002E,B15003_003E,B15003_004E,B15003_005E,B15003_006E,"
     "B15003_007E,B15003_008E,B15003_009E,B15003_010E,B15003_011E,"
     "B15003_012E,B15003_013E,B15003_014E,B15003_015E,B15003_016E",
     "B15003_001E", "EP_NOHSDP",
     "% No High School Diploma"),
    # EP_NOVEH: % occupied housing units with no vehicle
    ("B25044_003E,B25044_010E", "B25044_001E", "EP_NOVEH",
     "% No Vehicle"),
    # EP_LIMENG: % age 5+ who speak English less than 'well'
    ("B16005_007E,B16005_008E,B16005_012E,B16005_013E,B16005_017E,"
     "B16005_018E,B16005_022E,B16005_023E,B16005_029E,B16005_030E,"
     "B16005_034E,B16005_035E,B16005_039E,B16005_040E,B16005_044E,B16005_045E",
     "B16005_001E", "EP_LIMENG",
     "% Limited English"),
]

# EP_MINRTY: needs a different approach (total - white non-hispanic) / total
MINRTY_VARS = ("B01001_001E", "B01001H_001E")  # total pop, white non-hispanic


def fetch_acs(variables: str) -> list:
    """Fetch variables from ACS API for all MA tracts."""
    url = f"{BASE_URL}?get={variables}&for=tract:*&in=state:25&in=county:*"
    req = urllib.request.Request(url, headers={"User-Agent": "CS506-Project/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main():
    print("Fetching ACS 2022 data for Massachusetts Census Tracts...\n")

    # Collect all unique variables we need
    all_vars = set()
    for nums, denom, _, _ in SVI_VARS:
        all_vars.update(nums.split(","))
        all_vars.add(denom)
    all_vars.update(MINRTY_VARS)

    # Split into chunks of 48 vars (Census API limit is 50 including geo)
    var_list = sorted(all_vars)
    chunk_size = 48
    chunks = [var_list[i:i + chunk_size] for i in range(0, len(var_list), chunk_size)]

    # Fetch each chunk and merge
    merged = None
    for i, chunk in enumerate(chunks):
        print(f"  Fetching chunk {i + 1}/{len(chunks)} ({len(chunk)} variables)...")
        data = fetch_acs(",".join(chunk))
        df = pd.DataFrame(data[1:], columns=data[0])
        # Build GEOID (state + county + tract)
        df["FIPS"] = df["state"] + df["county"] + df["tract"]
        df = df.drop(columns=["state", "county", "tract"])
        # Convert numeric columns
        for col in chunk:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="FIPS", how="outer")

    print(f"\n  Total tracts fetched: {len(merged)}")

    # Compute EP_ percentages
    for nums, denom, name, desc in SVI_VARS:
        num_cols = nums.split(",")
        numerator = merged[num_cols].sum(axis=1)
        denominator = merged[denom]
        merged[name] = (numerator / denominator * 100).round(1)
        # Where denominator is 0, set to NaN
        merged.loc[denominator == 0, name] = pd.NA
        print(f"  Computed {name}: {desc}")

    # EP_MINRTY: (total - white_nh) / total * 100
    total_pop = merged["B01001_001E"]
    white_nh = merged["B01001H_001E"]
    merged["EP_MINRTY"] = ((total_pop - white_nh) / total_pop * 100).round(1)
    merged.loc[total_pop == 0, "EP_MINRTY"] = pd.NA
    print("  Computed EP_MINRTY: % Minority")

    # Select output columns
    ep_cols = [v[2] for v in SVI_VARS] + ["EP_MINRTY"]
    output = merged[["FIPS"] + ep_cols].copy()

    # Save
    out_path = "/Users/rayxu/CS506_Final_Project/problemstic/data/svi_2022_ma_tract.csv"
    output.to_csv(out_path, index=False)
    print(f"\nSaved {len(output)} tracts to {out_path}")
    print(f"\nSample:\n{output.head()}")
    print(f"\nNull counts:\n{output.isnull().sum()}")
    print(f"\nDescriptive stats:\n{output[ep_cols].describe().round(1)}")


if __name__ == "__main__":
    main()
