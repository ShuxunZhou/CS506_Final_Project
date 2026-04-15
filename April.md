# Boston 311 Intelligent Ticketing & Equity Analysis

## Project Overview

This project has two deliverables sharing the same data foundation:

1. **CS506 Final Project (Must Ship)**: Equity-focused audit of Boston 311 service delays — regression analysis proving socioeconomic factors predict resolution time, plus an overdue ticket classifier.
2. **PRD Extension (Stretch Goal)**: An AI-powered dispatch system with spatio-temporal deduplication, NLP department routing, and a Streamlit dashboard.

---

## Architecture

```
Data Layer (Shared)CKAN) → 500K+ records (2015-Present)
├── CDC SVI (2022) → Census Tract level, decomposed features
│   EP_POV150, EP_UNEMP, EP_NOHSDP, EP_LIMENG, EP_MINRTY, EP_NOVEH
└── Spatial Join via GeoPandas (lat/lon → Census Tract polygons)

CS506 Analysis Pipeline
├── Target Variable: resolution_time_days (right-censored for open tickets)
├── Binary Label: is_overdue (>30 days)
├── Logistic Regression (statsmodels, clustered SE by tract)
├── Overdue Classifier (train/test split, optimize Recall >75%)
└── K-Means on operational metrics [Volume, Mean_Res_Time, Variance, Overdue_Rate]

PRD: Intelligent Dispatch System
├── Module A: Spatio-Temporal Dedup Engine
│   Haversine distance (<50m) + TF-IDF cosine similarity (>85%) → merge
├── Module B: NLP Smart Router
│   TF-IDF + LightGBM/NaiveBayes → predict assigned_department + SHAP
└── Module C: Streamlit Dashboard (MVP)
    Text input → Dedup result → Routing result → Folium heatmap
```

---

## Completed

### March 2026 check-in (old baseline)
- [x] Data ingestion: 5,096 records loaded, date parsing, right-censoring logic
- [x] SVI merge: 7 decomposed SVI features joined by zip_code (ZCTA level)
- [x] Baseline logistic regression (statsmodels Logit) — runs but has known issues
- [x] K-Means clustering: 4 operational archetypes from service metrics
- [x] SVD text mining on closure_comments (5 latent topics extracted)
- [x] Preliminary EDA visualizations (resolution time distribution, log-scale)

### April 2026 rebuild
- [x] ETL: 1,694,448 records (2020-2026) via CKAN API → `boston_311_2020_2026.parquet` (183MB) — `etl_311.py`
- [x] SVI: 1,620 MA Census Tracts from ACS 2022 API → `data/svi_2022_ma_tract.csv` — `download_svi_tract.py`
- [x] Shapefile: TIGER 2022 MA tracts → `data/shapefiles/tl_2022_25_tract.shp`
- [x] Spatial join: 99.99% match, 1,605,853 records × 327 tracts → `boston_311_with_svi.parquet` (197MB) — `spatial_join.ipynb`
- [x] Dept mapping validation: `queue`→`department` is 100% 1:1; `type`→`department` 94.2% via lookup
- [x] Equity regression with clustered SE + train/test split — `equity_regression.ipynb`
  - Model converged, Pseudo R²=0.0882, 1.59M records × 314 tracts
  - Only EP_MINRTY significant (OR=1.0028, p=0.018) — 1pp increase → 0.28% more overdue odds
  - Other SVI features NOT significant after controlling for archetype + source
  - K-Means archetypes: Fast Resolution, Routine, Slow Queue, Chronic Backlog (84% overdue rate)
  - Baseline classifier: 83% accuracy but only 10% Recall on Overdue class at default threshold
- [x] Overdue classifier with threshold tuning — `overdue_classifier.ipynb`
  - Best model: Random Forest (AUC=0.7231) > Decision Tree (0.7207) > Logistic Regression (0.6904)
  - Optimal threshold=0.4304 → **Recall=75.00%** (target met), Precision=26.94%
  - Top features: Arch_Routine, Arch_Slow Queue, Src_Employee Generated, Arch_Chronic Backlog
  - SVI features ranked lower but EP_NOVEH and EP_MINRTY in top 10
- [x] PRD Module A+B combined — `prd_dedup_routing.ipynb`
  - **Module A (Dedup)**: 13,719 duplicate pairs in Q1 2024 (50m/85% thresholds), avg distance 16.4m, similarity 0.999. Top types: Parking Enforcement, Tree Maintenance. Est. annual savings $8.2M
  - **Module B (Routing)**: TF-IDF + LightGBM on case_title → 44% accuracy (12 departments). Note: low accuracy expected — case_title is heavily dominated by PWDx (48%). Demo routing works well for clear-cut cases. Model artifacts saved to `models/`
  - Top routing keywords: bprd, printed, park, conditions, equipment, light, graffiti

### Known Issues (March baseline — ALL resolved in April rebuild)

1. ~~**Data volume too small**~~: Resolved — 1.6M records, model converges
2. ~~**Ecological fallacy**~~: Resolved — Census Tract level + clustered SE
3. ~~**No train/test split**~~: Resolved — 80/20 stratified split with confusion matrix
4. ~~**Dismissal Index not validated**~~: Dropped — replaced with K-Means archetypes (data-driven, no annotation needed)
5. ~~**K-Means archetype names hardcoded**~~: Resolved — archetypes now named by operational profile (Fast Resolution, Routine, Slow Queue, Chronic Backlog)

---

## Key Conclusions (April 8, 2026)

### The Equity Story Is More Nuanced Than Expected

The headline finding is **not** "poor neighborhoods get worse 311 service." Instead:

1. **Service type is the dominant driver of delays**, not demographics. The K-Means archetypes (Chronic Backlog at 84% overdue rate vs Fast Resolution at 2%) explain far more variance than any SVI feature.
2. **Only EP_MINRTY is statistically significant** (OR=1.0028, p=0.018) after controlling for archetype + source channel. A 1pp increase in minority population → 0.28% higher overdue odds — real but tiny.
3. **Other SVI features (poverty, unemployment, education, vehicle access, English proficiency) are NOT significant** once you account for what type of service request it is and how it was filed.
4. **The policy implication**: Fixing equity isn't about treating neighborhoods differently — it's about fixing the chronically backlogged service types (which may disproportionately affect certain communities).

### Classifier Meets Target but Reveals Data Limitations

- Random Forest AUC=0.7231, Recall=75% at threshold 0.4304 — meets the >75% target
- But Precision is only 26.94% — the model flags many false positives
- Top predictive features are operational (archetype, source channel), not demographic
- This confirms: overdue prediction is fundamentally a service-type routing problem

### PRD Modules Show Real Business Value

- **Dedup Engine**: 13,719 duplicate pairs in Q1 2024 alone → est. $8.2M annual savings
- **NLP Routing**: 44% accuracy on 12 departments from free-text `case_title` — limited by PWDx dominance (48% of tickets). Structured fields (`type`→`department`) give 94%+ via lookup. NLP adds value only for free-text/ambiguous submissions.

---

## April 8 Tutor Review — 5 Issues & Optimization Plan

Source: `demo_April_8.pdf` (tutor feedback on current pipeline)

| # | Issue | Severity | Our Status | Fix |
|---|-------|----------|------------|-----|
| 1 | **Right-Censoring Fallacy** — open tickets <30 days labeled not-overdue, but outcome unknown | 致命 | **命中** — 当前逻辑确实把 15天未关的票标为 not overdue | 硬截断：只取 `open_dt` > 30天前的样本，确保标签确定性 |
| 2 | **VIF 多重共线性** — SVI 特征间高相关，p值不可信 | 致命 | **命中** — 未做 VIF 诊断 | 计算 VIF，VIF>5 的做 PCA 或剔除，重跑回归 |
| 3 | **去重缺时间窗口 + TF-IDF 语义盲区** | 致命 | **部分命中** — Q1 2024 内无时间约束 | 加 `abs(time_diff) < 48h` 硬规则；报告承认 TF-IDF 语义局限 |
| 4 | **虚荣指标** — 只看 Recall 不看 Precision/PR-AUC | 重要 | **部分命中** — 已报告 Precision=26.94% 但未讨论 PR-AUC | 补充 PR-AUC，讨论 precision-recall tradeoff 业务含义 |
| 5 | **Spatial Join CRS 不对齐 + 缺经纬度偏差** | 致命 | **已解决** — `to_crs()` + 99.99% 匹配率 | 检查缺经纬度票的数量/分布，报告说明是否存在系统性偏差 |

### Optimization Priority

1. **[P0]** 硬截断 right-censoring → 重跑 equity_regression + overdue_classifier
2. **[P0]** VIF 诊断 → PCA/剔除 → 重跑回归（可能改变 "只有 EP_MINRTY 显著" 的结论）
3. **[P1]** 去重加 48h 时间窗口 → 重跑 dedup，更新 duplicate pair 数量
4. **[P1]** 补充 PR-AUC + precision-recall 讨论
5. **[P2]** 缺经纬度偏差分析（补充说明即可）

---

## Remaining Work

### Optimization Pass (April 8 Tutor Feedback)

- [ ] Right-censoring 硬截断：只取 open_dt > 30天前样本，重跑 regression + classifier
- [ ] VIF 诊断 + PCA/剔除高共线性 SVI 特征，重跑 equity regression
- [ ] 去重加 48h 时间窗口，报告承认 TF-IDF 语义局限
- [ ] 补充 PR-AUC 指标，讨论 precision-recall tradeoff
- [ ] 检查缺经纬度票的系统性偏差

### Week 3 — Analysis Report & Visualizations

- [ ] Finalize CS506 equity regression visualizations (coefficient plots, archetype maps)
- [ ] Write analysis report / final notebook synthesizing all findings
- [ ] Equity choropleth maps: overdue rate by Census Tract with SVI overlay

### Week 4 — Dashboard + Polish

- [ ] PRD Module C: minimal Streamlit app (text input → dedup → routing → Folium map)
- [ ] Final presentation prep
- [ ] PRD document finalization with real metrics

---

## Key Decisions & Rationale

| Decision | Rationale |
|---|---|
| Right-censoring for open tickets | Dropping open tickets hides the worst delays in vulnerable communities |
| Decompose SVI instead of using composite score | Composite score masks which specific factors drive delays |
| Census Tract over zip_code for spatial join | Finer granularity, reduces ecological fallacy risk |
| Drop Dismissal Index unless annotated | SVD topic assumption is unvalidated; closure_reason distribution analysis is more defensible |
| Haversine + threshold over DBSCAN for dedup | Simpler, sufficient for the use case, avoids over-engineering |
| Folium over Kepler.gl | Lower learning curve, sufficient for MVP demo |
| Node/LightGBM over deep learning for routing | Small data, interpretability matters (SHAP), fast iteration |

---

## Tech Stack

- **Data**: Pandas, NumPy, GeoPandas, haversine
- **ML**: scikit-learn (TF-IDF, KMeans, LogisticRegression), statsmodels (OLS, Logit), LightGBM, SHAP
- **NLP**: TF-IDF + TruncatedSVD, cosine similarity
- **Viz**: Matplotlib, Seaborn, Folium
- **App**: Streamlit
- **Data Sources**: Boston 311 CKAN API, CDC SVI 2022, Census Bureau shapefiles

---

## Success Metrics

| Metric | Target | Scope |
|---|---|---|
| Overdue classifier Recall | >75% | CS506 |
| Equity regression significance | p<0.05 with clustered SE | CS506 |
| Routing accuracy | >85% on test set | PRD |
| Duplicate catch precision | >90% on manual review | PRD |
| Dashboard end-to-end latency | <2s | PRD |
