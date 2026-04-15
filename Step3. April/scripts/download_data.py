"""
Download CDC SVI 2022 data and Census Tract shapefiles for Massachusetts.
Then validate both datasets.
"""

import os
import urllib.request
import zipfile
import ssl

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHAPEFILE_DIR = os.path.join(DATA_DIR, "shapefiles")
SVI_CSV_PATH = os.path.join(DATA_DIR, "svi_2022_ma_tract.csv")
SHAPEFILE_ZIP = os.path.join(SHAPEFILE_DIR, "tl_2022_25_tract.zip")

# URLs - multiple patterns to try since CDC changes URL structure
SVI_URLS = [
    # State-level files - various URL patterns
    "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI_2022_MASSACHUSETTS.csv",
    "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI2022_MASSACHUSETTS.csv",
    "https://svi.cdc.gov/Documents/Data/2022/csv/states/Massachusetts.csv",
    # Alternate path structures
    "https://svi.cdc.gov/Documents/Data/2022_SVI_Data/csv/states/SVI_2022_MASSACHUSETTS.csv",
    "https://svi.cdc.gov/Documents/Data/2022_SVI_Data/csv/states/SVI2022_MASSACHUSETTS.csv",
    # US-level files
    "https://svi.cdc.gov/Documents/Data/2022/csv/SVI_2022_US.csv",
    "https://svi.cdc.gov/Documents/Data/2022/csv/SVI2022_US.csv",
    "https://svi.cdc.gov/Documents/Data/2022_SVI_Data/csv/SVI_2022_US.csv",
    "https://svi.cdc.gov/Documents/Data/2022_SVI_Data/csv/SVI2022_US.csv",
    # ATSDR hosted
    "https://www.atsdr.cdc.gov/placeandhealth/svi/data/SVI_2022_US.csv",
]
TRACT_SHAPEFILE_URL = "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_25_tract.zip"


def make_ssl_context():
    ctx = ssl.create_default_context()
    return ctx


def download_file(url, dest, description=""):
    print(f"Downloading {description or url} ...")
    try:
        ctx = make_ssl_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        size = os.path.getsize(dest)
        print(f"  -> Saved to {dest} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return False


def filter_us_to_ma(us_path):
    """Filter a US-level SVI CSV to Massachusetts rows only."""
    import csv
    with open(us_path, "r") as fin, open(SVI_CSV_PATH, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        # Find ST (state FIPS) column
        st_idx = None
        for i, col in enumerate(header):
            if col.upper() in ("ST", "STATE_FIPS", "STCNTY", "ST_ABBR"):
                st_idx = i
                break
        # Also try FIPS column - MA tracts start with 25
        fips_idx = None
        for i, col in enumerate(header):
            if col.upper() == "FIPS":
                fips_idx = i
                break
        count = 0
        for row in reader:
            is_ma = False
            if st_idx is not None and row[st_idx].strip() in ("25", "MA", "Massachusetts"):
                is_ma = True
            elif fips_idx is not None and row[fips_idx].strip().startswith("25"):
                is_ma = True
            if is_ma:
                writer.writerow(row)
                count += 1
        print(f"  -> Filtered {count} MA rows to {SVI_CSV_PATH}")
    os.remove(us_path)
    return count > 0


def download_svi():
    os.makedirs(DATA_DIR, exist_ok=True)

    for url in SVI_URLS:
        is_us_file = "US" in url.upper()
        if is_us_file:
            dest = os.path.join(DATA_DIR, "SVI_2022_US_temp.csv")
        else:
            dest = SVI_CSV_PATH

        if download_file(url, dest, f"SVI from {url}"):
            if is_us_file:
                print("Filtering US file to MA (FIPS 25)...")
                return filter_us_to_ma(dest)
            return True

    print("\nAll SVI download URLs failed.")
    print("Please manually download from: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html")
    print(f"Save the file as: {SVI_CSV_PATH}")
    return False


def download_shapefiles():
    os.makedirs(SHAPEFILE_DIR, exist_ok=True)
    # Skip download if shapefile already exists
    existing_shp = [f for f in os.listdir(SHAPEFILE_DIR) if f.endswith(".shp")]
    if existing_shp:
        print(f"Shapefile already exists: {existing_shp} - skipping download.")
        return True
    if not download_file(TRACT_SHAPEFILE_URL, SHAPEFILE_ZIP, "Census Tract shapefiles for MA"):
        return False
    # Unzip
    print("Unzipping shapefiles...")
    with zipfile.ZipFile(SHAPEFILE_ZIP, "r") as zf:
        zf.extractall(SHAPEFILE_DIR)
        print(f"  -> Extracted: {zf.namelist()}")
    # Check for .shp
    shp_files = [f for f in os.listdir(SHAPEFILE_DIR) if f.endswith(".shp")]
    if shp_files:
        print(f"  -> .shp file(s) found: {shp_files}")
        return True
    else:
        print("  -> WARNING: No .shp file found after unzip!")
        return False


def validate():
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # Validate SVI CSV
    print("\n--- SVI CSV ---")
    import csv
    with open(SVI_CSV_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    print(f"Row count: {len(rows)}")
    ep_cols = [c for c in header if "EP_" in c.upper()]
    print(f"EP_ columns ({len(ep_cols)}): {ep_cols}")
    # Find FIPS column
    fips_col = None
    fips_idx = None
    for i, c in enumerate(header):
        if c.upper() == "FIPS":
            fips_col = c
            fips_idx = i
            break
    if fips_idx is not None:
        first_fips = [rows[i][fips_idx] for i in range(min(3, len(rows)))]
        print(f"FIPS column: '{fips_col}', first 3 values: {first_fips}")
    else:
        print(f"No FIPS column found. Columns: {header[:15]}...")

    # Validate shapefile
    print("\n--- Shapefile ---")
    try:
        import geopandas as gpd
        shp_files = [f for f in os.listdir(SHAPEFILE_DIR) if f.endswith(".shp")]
        shp_path = os.path.join(SHAPEFILE_DIR, shp_files[0])
        gdf = gpd.read_file(shp_path)
        print(f"Row count: {len(gdf)}")
        print(f"CRS: {gdf.crs}")
        print(f"Columns: {list(gdf.columns)}")
        # Check for GEOID
        geoid_col = None
        for c in gdf.columns:
            if c.upper() in ("GEOID", "GEOID20", "GEOID10"):
                geoid_col = c
                break
        if geoid_col:
            print(f"GEOID column: '{geoid_col}', first 3 values: {list(gdf[geoid_col].head(3))}")
        else:
            print("No GEOID column found.")
    except ImportError:
        print("geopandas not installed - skipping shapefile validation.")
        print("Install with: pip install geopandas")

    # Check join compatibility
    print("\n--- Join Compatibility ---")
    if fips_idx is not None and geoid_col is not None:
        svi_fips_sample = first_fips[0] if first_fips else ""
        shp_geoid_sample = str(gdf[geoid_col].iloc[0])
        print(f"SVI FIPS sample: '{svi_fips_sample}' (len={len(svi_fips_sample)})")
        print(f"Shapefile GEOID sample: '{shp_geoid_sample}' (len={len(shp_geoid_sample)})")
        if len(svi_fips_sample) == len(shp_geoid_sample):
            print("Lengths match - should be directly joinable.")
        else:
            print("Lengths differ - may need padding/trimming for join.")
    print("\nDone!")


if __name__ == "__main__":
    svi_ok = download_svi()
    shp_ok = download_shapefiles()
    if svi_ok and shp_ok:
        validate()
    else:
        if not svi_ok:
            print("ERROR: Failed to download SVI data.")
        if not shp_ok:
            print("ERROR: Failed to download shapefiles.")
