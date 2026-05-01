# Boston 311 Service Latency & Equity Analysis

[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/Raaaaay-x/CS506_Final_Project)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Course](https://img.shields.io/badge/Course-CS506-red)](https://www.bu.edu/cs)

**CS506 Final Project** — An equity-focused audit of Boston's 311 service request system, combining multivariate logistic regression with machine learning classifiers to quantify socioeconomic disparities in municipal service delivery.
**Display Video** - (https://youtu.be/OqvFAl3WLiQ)

---

## Table of Contents

- [1. Project Motivation](#1-project-motivation)
- [2. Repository Structure](#2-repository-structure)
- [3. Quick Start (Reproducibility)](#3-quick-start-reproducibility)
- [4. Data Collection](#4-data-collection)
- [5. Data Cleaning](#5-data-cleaning)
- [6. Feature Extraction](#6-feature-extraction)
- [7. Model Training & Evaluation](#7-model-training--evaluation)
- [8. Data Visualization & Results](#8-data-visualization--results)
- [9. Limitations & Discussion](#9-limitations--discussion)

---

## 1. Project Motivation

Boston's 311 system processes over 250,000 service requests annually. Aggregate metrics like "Average Time to Close" mask systemic inefficiencies: a handful of chronically delayed tickets inflate the mean, while neighborhood-level disparities remain hidden.

**Research Questions:**

1. **Equity:** Are socioeconomic factors (poverty, minority status, vehicle access, English proficiency) statistically significant predictors of service resolution time, even after controlling for request type?
2. **Efficiency:** Can we build an early-warning classifier that identifies tickets likely to become overdue (>30 days) using only information available at submission time?
3. **Systemic Bias:** Does the dataset itself contain structural gaps — for example, are certain types of complaints systematically missing geolocation data?

---

## 2. Repository Structure

```
CS506_Final_Project/
├── Makefile                          # One-command build system
├── README.md                         # This file
├── .gitignore
│
├── notebooks/                        # Analysis notebooks (numbered, executable)
│   ├── 00a_proposal_EDA.ipynb        # Initial exploratory data analysis
│   ├── 00b_march_checkin_demo.ipynb  # March 2026 progress checkpoint
│   ├── 01_spatial_join.ipynb         # Spatial join: 311 tickets → Census Tracts → SVI features
│   ├── 02_equity_regression.ipynb    # Logistic regression: SVI → overdue probability
│   ├── 03_overdue_classifier.ipynb   # ML classifier: predict overdue tickets
│   └── 04_prd_dedup_routing.ipynb    # NLP duplicate detection & department router
│
├── scripts/                          # ETL & data pipeline scripts
│   ├── download_data.py              # Download CDC SVI CSV + Census Tract shapefiles
│   ├── download_svi_tract.py         # Alternative SVI download with validation
│   ├── etl_311.py                    # CKAN API → clean parquet (paginated, 500K+ rows)
│   ├── export_dashboard_data.py      # Aggregate model outputs → JSON for HTML dashboard
│   └── build_dashboard.py            # Jinja2 template → static index.html
│
├── models/                           # Serialized model artifacts
│   ├── lgbm_router.pkl               # LightGBM department router (v1)
│   ├── lgbm_router_v2.pkl            # LightGBM department router (v2)
│   ├── tfidf_router.pkl              # TF-IDF vectorizer for NLP routing
│   ├── label_encoder.pkl             # Department label encoder
│   └── model_usage.ipynb             # Demo notebook for loading & using saved models
│
├── exports/                          # Model output summaries (CSV/TXT)
│   ├── logistic_regression_summary.txt
│   ├── odds_ratios.csv
│   └── kmeans_cluster_summary.csv
│
├── Step3. April_v1.1/
│   ├── dashboard/
│   │   ├── index.html                # Built dashboard (git-tracked for portability)
│   │   ├── template.html.j2          # Jinja2 template
│   │   └── data/                     # Aggregated JSON payloads + GeoJSON
│   ├── requirements.txt              # Python dependencies
│   └── scripts/
│       ├── export_models_data.py     # Train & serialize models for dashboard
│       └── run_dashboard.sh          # Convenience launcher
│
└── figures/                          # Generated plots
    ├── eda_equity.png                # Neighborhood disparity bar chart
    ├── eda_variance.png              # Resolution time boxplots by service type
    ├── odds_ratios_clustered.png     # Odds ratios with confidence intervals
    ├── confusion_matrix_optimized.png
    ├── feature_importance.png
    ├── pr_roc_curves.png
    └── ... (additional diagnostic plots)
```

**Key design decisions:**
- Notebooks are numbered in execution order and self-contained (each produces a verifiable output)
- Model artifacts and data files are excluded from git (`.gitignore`) but reproducible via `make build`
- Dashboard data is small (~150 KB) and tracked in git so the HTML works immediately after clone

---

## 3. Quick Start (Reproducibility)

### One-Command Setup

```bash
git clone https://github.com/Raaaaay-x/CS506_Final_Project.git
cd CS506_Final_Project
make all              # installs venv → downloads data → trains models → renders dashboard → serves at :8000
```

### Step-by-Step

```bash
# 1. Create virtual environment + install all dependencies
make install

# 2. Build the full pipeline (data → spatial join → models → dashboard)
make build

# 3. Serve the dashboard
make serve            # opens http://localhost:8000/

# 4. Force a full rebuild (ignoring cached artifacts)
make rebuild
```

### Individual Targets

| Target | Description |
|--------|-------------|
| `make install` | Create venv, install `requirements.txt` + Jupyter |
| `make build` | End-to-end: download → spatial join → train models → export JSON → render HTML |
| `make rebuild` | Force full rebuild (ignores cached artifacts) |
| `make serve` | Serve dashboard at `http://localhost:8000` |
| `make all` | `install` + `build` + `serve` |
| `make check` | Verify venv and core dependencies |
| `make clean` | Remove generated artifacts (JSON, HTML, model pkl files) |
| `make clean-venv` | Remove the venv directory entirely |

### Requirements

- **Python 3.9+** (tested on 3.11)
- **Dependencies** (auto-installed via `make install`):
  - Core: `pandas`, `numpy`, `scikit-learn`, `scipy`
  - Statistics: `statsmodels` (logistic regression with clustered standard errors)
  - Spatial: `geopandas`, `shapely`, `fiona`
  - ML: `lightgbm` (gradient-boosted department router)
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Web: `jinja2` (dashboard template rendering)
  - Jupyter: `jupyter`, `nbconvert`, `ipykernel`

### Reproducibility Guarantees

- All random seeds are fixed (`random_state=42` across notebooks and scripts)
- The Makefile pipeline is incremental (each stage checks for existing artifacts before re-running)
- Notebook execution is automated via `nbconvert --execute` with a 30-minute timeout
- Dashboard HTML is committed to git and renders immediately after clone

---

## 4. Data Collection

### Data Sources

| Source | Description | Access Method | Records |
|--------|-------------|---------------|---------|
| **Boston 311 CKAN API** | Service requests (2015–2026) | Paginated JSON via `requests` | 500,000+ |
| **CDC/ATSDR SVI 2022** | Social Vulnerability Index at Census Tract level | Direct CSV download | ~1,500 MA tracts |
| **US Census Bureau TIGER** | Census Tract shapefiles for spatial join | ZIP download from census.gov | ~1,500 MA tract polygons |

### Why These Sources?

- **Boston CKAN API** is the official open data portal for the City of Boston, providing structured, machine-readable access to all 311 records with consistent field definitions across years.
- **CDC SVI** is the gold standard for socioeconomic vulnerability measurement, used by FEMA, HHS, and state health departments. We **decompose the composite index** into its constituent ACS variables rather than using the aggregate score, which would mask which specific drivers correlate with service delays.
- **Census TIGER shapefiles** enable spatial joins at the Census Tract level (approximately 4,000 residents per tract), providing much finer geographic resolution than zip-code-level analysis.

### Collection Implementation

- **`scripts/etl_311.py`** — Paginated CKAN Datastore API client. Handles cross-year deduplication (a case opened in December may appear in both years' exports), column name normalization (different years use different casing), and coordinate validation against Boston's bounding box.
- **`scripts/download_data.py`** — Robust downloader for CDC SVI CSV and Census TIGER shapefiles. Implements multiple fallback URLs (CDC periodically changes their URL structure) and automatic filtering of US-level files to Massachusetts (FIPS prefix `25`).
- **`notebooks/01_spatial_join.ipynb`** — Point-in-polygon spatial join using GeoPandas `sjoin(predicate="within")`, mapping each 311 ticket's lat/lon to its containing Census Tract polygon, then attaching the decomposed SVI variables.

---

## 5. Data Cleaning

All cleaning logic is implemented in **`scripts/etl_311.py`** (the `normalize()` function) and **`notebooks/01_spatial_join.ipynb`**.

### Steps Applied

| Step | Method | Rationale |
|------|--------|-----------|
| **Date parsing** | `pd.to_datetime(..., errors='coerce')` | Handle inconsistent date formats across years; coerce unparseable values to NaT |
| **Coordinate validation** | Drop rows outside Boston bounding box (lat 42.2–42.45, lon -71.2 – -70.9) and `(0, 0)` sentinel values | Eliminates GPS errors and test records |
| **Cross-year deduplication** | `drop_duplicates(subset='case_enquiry_id', keep='last')` | Cases opened in December appear in multiple annual exports |
| **Column normalization** | Map variant names (`ontime` → `on_time`, `closeddt` → `closed_dt`) | Different CKAN resource years use different column naming conventions |
| **Case status standardization** | `.str.strip().str.title()` | Unify `"open"`, `"Open"`, `"OPEN"` to a single representation |
| **Zip code extraction** | Regex `(\d{5})` from free-text zip field | Some entries contain full ZIP+4 or text annotations |
| **Negative resolution time** | Set to NaN | Auto-closed or test tickets with `closed_dt < open_dt` |
| **Right-censoring** | Exclude tickets opened after `REFERENCE_DATE - 30d` from overdue classification | Prevents labeling open tickets as "not overdue" when their outcome is unknown |
| **Coordinate bias assessment** | Chi-square test on missing-coordinate distribution by request type and hour | Quantifies whether missing lat/lon is random or systematically correlated with certain complaint types (0.76% missing rate, but non-random: χ² p ≈ 0) |

### Handling Missing Data

- **Coordinates:** 0.76% of 2024 records lack lat/lon. These are dropped from spatial analysis, but we formally test for systematic bias in `04_prd_dedup_routing.ipynb` and report findings in the dashboard's "Missing Bias" section.
- **SVI features:** Tracts with missing SVI values (rare, < 0.1%) are excluded from regression. We verified no systematic pattern in which tracts have missing SVI.
- **Resolution time:** Tickets still open have undefined resolution time. These are excluded from duration analysis but included in the missing-coord bias analysis.

---

## 6. Feature Extraction

### Target Variable

**`is_overdue`** (binary): `1` if `resolution_days > 30`, else `0`. The 30-day threshold is the standard definition of a "long-tail" ticket in Boston's 311 system (top decile of resolution times).

### Features (all available at ticket creation time — no future leakage)

#### Primary Features: SVI Socioeconomic Variables

Decomposed from the CDC Social Vulnerability Index at the Census Tract level:

| Variable | Definition | Hypothesis |
|----------|------------|------------|
| `EP_POV150` | % households below 150% poverty line | Higher poverty → longer resolution |
| `EP_UNEMP` | % civilian labor force unemployed | Higher unemployment → longer resolution |
| `EP_NOHSDP` | % population (25+) with no high school diploma | Lower education → longer resolution |
| `EP_LIMENG` | % households with limited English proficiency | Language barrier → longer resolution |
| `EP_MINRTY` | % population identifying as racial/ethnic minority | Minority concentration → longer resolution |
| `EP_NOVEH` | % households with no vehicle access | Lower mobility → longer resolution |

These features are attached via spatial join (`notebooks/01_spatial_join.ipynb`), mapping each ticket's lat/lon to its Census Tract GEOID, then joining with the SVI table.

#### Derived Features: Operational Archetypes

Rather than using 175+ service type dummies (which would create a high-dimensional, sparse feature space), we cluster service types into **4 operational archetypes** using K-Means (k=4) on:

- `volume` (log-transformed): how many tickets of this type
- `mean_res`: average resolution time
- `var_res`: variance in resolution time
- `overdue_rate`: proportion exceeding 30 days

This produces four interpretable clusters:

| Archetype | Overdue Rate | Description |
|-----------|-------------|-------------|
| **Fast Resolution** (0) | ~2% | High-volume, quickly resolved (e.g., parking enforcement) |
| **Routine** (1) | ~14% | Moderate volume and resolution time |
| **Slow Queue** (2) | ~37% | Elevated delay rates, infrastructure-related |
| **Chronic Backlog** (3) | ~84% | Consistently delayed (e.g., street light outages, tree maintenance) |

#### Supplementary Features

- **Report Source:** `Citizens Connect App`, `Constituent Call`, `Employee Generated`, `City Worker App`, `Self Service` — different channels may correlate with different urgency levels
- **Temporal Features:** `open_hour`, `open_dayofweek` — off-hours submissions may route differently
- **Geographic Fixed Effects:** Census Tract GEOID as clustering variable in regression standard errors

### Feature Appropriateness Rationale

- **SVI variables** are the CDC's standard operationalization of socioeconomic vulnerability — they are well-validated and widely used in public health and urban policy research
- **K-Means archetypes** reduce dimensionality while preserving interpretability (unlike PCA or autoencoders), which is essential for policy recommendations
- **All features are pre-outcome**: nothing requires knowing the ticket's eventual resolution to compute, ensuring the classifier can operate in a real-time prediction setting

---

## 7. Model Training & Evaluation

### Model 1: Equity Regression (Logistic Regression)

**Notebook:** `notebooks/02_equity_regression.ipynb`

**Research question:** Are socioeconomic factors statistically significant predictors of overdue probability after controlling for service type and report source?

**Methodology:**
- **Model:** Logistic regression (`statsmodels.Logit`) with clustered standard errors by Census Tract GEOID
- **Formula:** `is_overdue ~ EP_POV150 + EP_UNEMP + EP_NOHSDP + EP_LIMENG + EP_MINRTY + EP_NOVEH + C(service_type) + C(source)`
- **Sample:** Tickets opened before `2026-03-09` (30-day right-censoring cutoff), N ≈ 1.59M after filtering
- **Covariance:** Clustered by Census Tract to account for within-neighborhood correlation of errors

**Evaluation:**
- **Variance Inflation Factor (VIF):** All 6 SVI features retained; max VIF = 4.6 (well below the VIF > 10 exclusion threshold), confirming multicollinearity is not inflating standard errors
- **Pseudo R²:** 0.088 (McFadden) — modest, expected for binary outcomes with many unobserved factors
- **Odds Ratios:** Reported with 95% confidence intervals

**Key Results:**

| Variable | Odds Ratio | 95% CI | p-value | Significant (p < 0.05)? |
|----------|-----------|--------|---------|--------------------------|
| `EP_MINRTY` | 1.00280 | [1.00048, 1.00512] | 0.018 | **Yes** |
| `EP_NOVEH` | 1.00112 | [0.99761, 1.00465] | 0.532 | No |
| `EP_POV150` | 1.00042 | [0.99731, 1.00354] | 0.790 | No |
| `EP_LIMENG` | 1.00040 | [0.99569, 1.00514] | 0.869 | No |
| `EP_NOHSDP` | 1.00097 | [0.99681, 1.00515] | 0.650 | No |
| `EP_UNEMP` | 0.99810 | [0.99260, 1.00363] | 0.500 | No |

**Interpretation:** `EP_MINRTY` (minority concentration) is the only SVI variable with a statistically significant positive association with overdue probability, after controlling for service type and report source. A one-percentage-point increase in minority share corresponds to a ~0.28% increase in the odds of a ticket being overdue (OR = 1.00280). While the effect size is small per percentage point, it compounds across the full range of minority concentration (0% to ~95%).

### Model 2: Overdue Ticket Classifier (Early Warning System)

**Notebook:** `notebooks/03_overdue_classifier.ipynb`

**Objective:** Predict at submission time whether a ticket will become overdue (>30 days).

**Success criterion:** Recall > 75% on the "overdue" class.

**Models compared:**

| Model | ROC-AUC | PR-AUC | Recall (at optimized threshold) | Precision |
|-------|---------|--------|--------------------------------|-----------|
| Logistic Regression | 0.690 | 0.360 | — | — |
| Decision Tree | 0.721 | 0.411 | — | — |
| **Random Forest** | **0.723** | **0.420** | **0.750** | **0.269** |

**Model selection rationale:**
- **Random Forest** was chosen as the best model (highest AUC across both ROC and PR curves). It captures non-linear interactions between SVI features and operational archetypes without requiring manual interaction terms. It also provides feature importance scores for interpretation.
- **Logistic Regression** serves as an interpretable baseline (coefficients are directly interpretable as odds ratios)
- **Decision Tree** provides a middle ground — more flexible than logistic regression but more interpretable than random forest

**Training procedure:**
- **Train/Test Split:** 80/20 split, stratified by `is_overdue` to preserve class distribution
- **Class imbalance:** ~18.3% prevalence of overdue class — handled via `class_weight='balanced'`
- **Threshold optimization:** The default 0.5 threshold is suboptimal for imbalanced classification. We optimize the decision threshold to maximize F1-score on the validation set, yielding a threshold of 0.430.

**Evaluation strategy:**
- **Primary metric:** Recall (we prioritize finding overdue tickets over precision — a false positive means an extra inspection, but a false negative means a neglected case)
- **Secondary metrics:** PR-AUC (more informative than ROC-AUC for imbalanced classification), confusion matrix, feature importance
- **PR-AUC:** 0.420 (Random Forest), **2.3× the baseline** of 0.183 (which would be achieved by random guessing at the prevalence rate)

**Confusion Matrix (at optimized threshold):**

|                | Predicted Negative | Predicted Positive |
|----------------|-------------------|--------------------|
| **Actual Negative** | 467,000 | 173,000 |
| **Actual Positive** | 35,000  | 108,000 |

- Recall = 108,000 / (35,000 + 108,000) = **75.5%**
- Precision = 108,000 / (173,000 + 108,000) = **38.4%**

### Model 3: NLP Department Router (Supplementary)

**Notebook:** `notebooks/04_prd_dedup_routing.ipynb`

**Objective:** Predict which department should handle a ticket based solely on its free-text title.

**Methodology:**
- TF-IDF vectorization of case titles → LightGBM classifier for 12 departments
- 48-hour temporal window for duplicate detection: two tickets within 50m, within 48 hours, with TF-IDF cosine similarity > 0.85 are flagged as duplicates

**Results:**
- Overall accuracy: ~44% (12-way classification, substantially better than random baseline of 8%)
- High-confidence categories: "Abandoned Vehicle" → BTDT (93.9%), "Water Leak" → BWSC (60%)

**Limitations discussed in notebook:**
- TF-IDF cannot capture semantic similarity ("gas leak" vs "natural gas emergency" are different inputs)
- Case titles often contain insufficient information for unambiguous routing
- Historical labels contain noise (same issue may have been routed to different departments over time)

---

## 8. Data Visualization & Results

### Dashboard

A single-page HTML dashboard is served at `http://localhost:8000/` (see [Quick Start](#3-quick-start-reproducibility)). It displays:

1. **Equity Regression** — Odds ratios with 95% confidence intervals for all 6 SVI variables, VIF diagnostics table
2. **Overdue Classifier** — PR and ROC curves, confusion matrices (baseline vs. optimized threshold), feature importance bar chart
3. **Tract Choropleth Map** — Interactive maps showing overdue rate, EP_MINRTY, EP_POV150, and EP_NOVEH across Boston Census Tracts
4. **PRD Router Demo** — Live text input → department prediction with confidence score and TF-IDF keywords
5. **Missing Coordinate Bias** — Bar charts showing systematic gaps by complaint type, hour of day, and day of week
6. **Archetype Sunburst** — Hierarchical view of the 4 operational archetypes and their constituent service types

### Key Figures

All figures are generated in the notebooks and saved to `figures/`:

| Figure | Notebook | Insight |
|--------|----------|---------|
| `eda_equity.png` | `00a_proposal_EDA` | Neighborhood disparities in median days to fix street light outages — some neighborhoods wait 3× longer |
| `eda_variance.png` | `00a_proposal_EDA` | Boxplots reveal massive within-type variance — median is low but long tail extends for months |
| `odds_ratios_clustered.png` | `02_equity_regression` | EP_MINRTY is the only SVI variable with OR significantly > 1.0 after controls |
| `pr_roc_curves.png` | `03_overdue_classifier` | Random Forest achieves PR-AUC = 0.420 (2.3× baseline); ROC-AUC = 0.723 |
| `confusion_matrix_optimized.png` | `03_overdue_classifier` | 75% recall at optimized threshold (0.430), with manageable false positive rate |
| `feature_importance.png` | `03_overdue_classifier` | Operational archetypes dominate; EP_NOVEH is the most important SVI feature |
| `svi_distributions.png` | `01_spatial_join` | SVI features show meaningful variation across tracts (not all clustered in a few tracts) |
| `archetype_boxplot.png` | `02_equity_regression` | 4 archetype clusters separate clearly on overdue rate and resolution time variance |

**Design principles for visualizations:**
- All plots use labeled axes, titles, and legends
- Color palettes are colorblind-friendly (matplotlib `tab10`, seaborn `colorblind`)
- Confidence intervals are always shown when reporting statistical estimates
- Interactive dashboard uses Plotly for zoom/pan/hover tooltips

---

## 9. Limitations & Discussion

### Statistical Limitations

1. **Right-censoring:** Tickets opened within 30 days of the data snapshot may not have reached the overdue threshold yet. We address this by excluding tickets opened after `REFERENCE_DATE - 30 days` (hard cutoff at 2026-03-09). This preserves label integrity at the cost of discarding recent data.
2. **Small effect sizes:** The odds ratios for SVI variables are close to 1.0 (e.g., EP_MINRTY OR = 1.0028 per percentage point). While statistically significant, the practical magnitude is modest. Across a 90-percentage-point range in minority concentration, this corresponds to ~29% higher odds of overdue — meaningful in aggregate but small at the individual ticket level.
3. **Observational data:** We cannot make causal claims. SVI variables may proxy for unobserved factors (infrastructure quality, political representation, department staffing) rather than directly causing delays.
4. **Pseudo-R² (0.088):** The logistic regression explains only ~9% of the variation in overdue status (McFadden's R²). Most variation is driven by unobserved factors — this is typical for individual-level binary outcomes and does not invalidate the statistically significant SVI coefficients.

### Data Limitations

5. **Missing coordinates (0.76%):** Tickets missing lat/lon are systematically biased toward "General Comments," "Programs," and "Request for Information" categories. These are excluded from spatial analysis, but their exclusion likely undercounts certain types of feedback — particularly policy comments submitted without a specific location.
6. **SVI vintage (2022):** The CDC SVI uses 2018–2022 ACS 5-year estimates. Neighborhood demographics may have shifted since then, especially post-COVID.
7. **Service type evolution:** Boston's 311 taxonomy has changed over time (categories added, merged, or renamed). The 175+ types in the raw data were manually reviewed and normalized.

### Model Limitations

8. **NLP Router (44% accuracy):** TF-IDF is inherently limited for semantic understanding. Future work could replace it with sentence-transformers (e.g., `all-MiniLM-L6-v2`) for better generalization across paraphrases.
9. **Generalizability:** Models trained on Boston data may not transfer to other cities with different 311 taxonomies, demographic patterns, and service delivery structures.
10. **Class imbalance:** At 18.3% prevalence, the overdue class is a minority. While `class_weight='balanced'` mitigates this, the precision-recall tradeoff means the system will generate many false positives — acceptable for a screening tool, but would require human triage in production.

---

## License

This project is developed for Boston University CS506 (Spring 2026). Data sourced from the City of Boston Open Data Portal and CDC/ATSDR.
