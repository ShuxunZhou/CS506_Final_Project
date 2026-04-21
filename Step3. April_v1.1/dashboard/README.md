# Boston 311 Dashboard

Single-page HTML dashboard combining CS506 equity analysis with PRD dedup/routing demo.

## Open the dashboard

**Recommended** — serve locally (works in all browsers):

```bash
cd "Step3. April_v1.1/dashboard"
python -m http.server 8000
```

Then visit http://localhost:8000/ in your browser.

**Double-click `index.html`** — works in Firefox. Chrome blocks local `fetch()` on `file://` URLs by default, so charts will not load without the local server above.

## Build from source

Prerequisites:
- `Step3. April_v1.1/data/boston_311_with_svi.parquet` (produced by `notebooks/spatial_join.ipynb`)
- `Step3. April_v1.1/models/*.pkl` (produced by `notebooks/prd_dedup_routing.ipynb`)
- `Step3. April_v1.1/scripts/data/shapefiles/tl_2022_25_tract.shp` (produced by `scripts/download_data.py`)

If any input is missing the export script writes a placeholder section and logs a warning — the HTML still builds.

From repo root:

```bash
make dashboard
```

Or manually:

```bash
cd "Step3. April_v1.1"
pip install -r requirements.txt
python scripts/export_dashboard_data.py   # → dashboard/data/*.json + tracts.geojson
python scripts/build_dashboard.py          # → dashboard/index.html
```

## What's inside

| Section | Content |
|---|---|
| `#equity` | Odds ratios with CI, PR/ROC curves, confusion matrices (baseline ↔ optimized), RF feature importance |
| `#map` | Tract-level choropleth (Plotly): toggle overdue rate / EP_MINRTY / archetype |
| `#prd` | Pre-computed router demo (20 samples) + dedup pair table + KPI cards |
| `#bias` | Missing-coord χ² diagnostic + K-Means archetype sunburst |

## Known limitations

- **Online required**: Plotly loads from CDN (`cdn.plot.ly`). Offline open = blank charts.
- **Router/dedup are pre-computed**: the HTML ships with 20 curated router examples and 10 dedup pairs. Live inference requires loading the pickled models in a notebook.
- **Data not bundled**: build-time parquet (~197 MB) is gitignored; the dashboard itself ships only aggregated JSON (~3 MB total).
