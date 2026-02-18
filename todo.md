# CS506 Final Project - To-Do List

> Project: Boston 311 Service Latency & Equity Analysis  
> Timeline: 8 Weeks

---

## Phase 1: Data Engineering

- [ ] **Environment setup** - Project structure, dependencies, gitignore
- [ ] **Data collection** - Fetch 311 data via CKAN API, download CDC SVI data
- [ ] **Data cleaning & spatial join** - Clean datasets, calculate resolution time, map coordinates to neighborhoods

---

## Phase 2: EDA & Feature Engineering

- [ ] **EDA report** - Distribution analysis, equity gaps visualization, long-tail identification
- [ ] **Feature engineering** - Create labels (>30 days), decompose SVI variables, handle missing values

---

## Phase 3: Unsupervised Learning

- [ ] **Request type clustering** - Cluster 175+ request types into critical vs routine categories

---

## Phase 4: Text Mining

- [ ] **Topic extraction** - SVD on case descriptions, extract latent topics as features

---

## Phase 5: Regression Modeling

- [ ] **Equity regression** - OLS regression on SVI variables, test statistical significance (p < 0.05)
- [ ] **Early warning classifier** - Logistic regression to predict long-tail tickets (Goal: Recall > 75%)

---

## Phase 6: Visualization & Dashboard

- [ ] **Streamlit dashboard** - Interactive map, equity analysis charts, prediction interface

---

## Phase 7: Final Delivery

- [ ] **Final report** - Background, methodology, results, policy implications
- [ ] **Code & presentation** - Documentation, demo notebook/ppt, reproducibility check
