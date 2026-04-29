"""
Produce the three JSON payloads that ``export_dashboard_data.py`` leaves at
fallback when the notebooks never dumped their intermediate state:

    dashboard/data/_classifier_cache.json   ← PR/ROC + CM + feat imp (retrained)
    dashboard/data/router_samples.json      ← LightGBM predictions on 20 titles
    dashboard/data/dedup_samples.json       ← Q1-2024 duplicate pair scan

Run this BEFORE export_dashboard_data.py so the cache + notebook-authored
JSONs are picked up:

    python scripts/export_models_data.py
    python scripts/export_dashboard_data.py
    python scripts/build_dashboard.py

Feature engineering mirrors overdue_classifier.ipynb cell[3] exactly so the
numbers match the report. Router + dedup reuse the pkl/parameters from
prd_dedup_routing.ipynb.
"""
from __future__ import annotations

import json
import sys
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
STEP_DIR = SCRIPT_DIR.parent
REPO_ROOT = STEP_DIR.parent
DATA_DIR = STEP_DIR / "data"
# Notebooks were moved to repo-root in the April rewrite; pkl artifacts now
# live at <repo>/models/ instead of Step3/models/.
MODELS_DIR = REPO_ROOT / "models"
OUT_DIR = STEP_DIR / "dashboard" / "data"
PARQUET = DATA_DIR / "boston_311_with_svi.parquet"

SVI_FEATURES = ["EP_POV150", "EP_UNEMP", "EP_NOHSDP", "EP_LIMENG", "EP_MINRTY", "EP_NOVEH"]
REFERENCE_DATE = pd.Timestamp("2026-04-08", tz="UTC")
TARGET_RECALL = 0.75
CURVE_POINTS = 60  # downsample PR/ROC to this many points for the dashboard


def log(msg: str, *, level: str = "INFO") -> None:
    print(f"[{level}] {msg}", file=sys.stderr)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = path.stat().st_size / 1024
    log(f"wrote {path.relative_to(STEP_DIR)} ({size_kb:,.1f} KB)")


def load_parquet() -> pd.DataFrame:
    if not PARQUET.exists():
        raise FileNotFoundError(
            f"{PARQUET} missing — run spatial_join.ipynb first."
        )
    log(f"loading {PARQUET.relative_to(STEP_DIR)}")
    df = pd.read_parquet(PARQUET)
    df["open_dt"] = pd.to_datetime(df["open_dt"], utc=True)
    cutoff = REFERENCE_DATE - pd.Timedelta(days=30)
    return df[df["open_dt"] < cutoff].copy()


# ---------------------------------------------------------------------------
# classifier — retrain 3 models, dump PR/ROC/CM/feat-imp
# ---------------------------------------------------------------------------
def _downsample_curve(xs: np.ndarray, ys: np.ndarray, n: int) -> list[list[float]]:
    if len(xs) <= n:
        return [[float(x), float(y)] for x, y in zip(xs, ys)]
    # Pick evenly-spaced indices from the raw (recall, precision) array so the
    # shape is preserved even when threshold grid is non-uniform.
    idx = np.linspace(0, len(xs) - 1, n).astype(int)
    return [[float(xs[i]), float(ys[i])] for i in idx]


def export_classifier(df: pd.DataFrame) -> None:
    log("classifier: building feature matrix (mirrors overdue_classifier.ipynb)")
    df = df.copy()
    df["hour"] = df["open_dt"].dt.hour
    df["dayofweek"] = df["open_dt"].dt.dayofweek
    df["month"] = df["open_dt"].dt.month

    # Archetype via K-Means on operational stats (same seed as notebook)
    ops = df.groupby("type").agg(
        volume=("is_overdue", "count"),
        mean_res=("resolution_time_days", "mean"),
        var_res=("resolution_time_days", "var"),
        overdue_rate=("is_overdue", "mean"),
    ).fillna(0)
    ops = ops[ops["volume"] >= 50]
    ops["log_volume"] = np.log1p(ops["volume"])
    X_km = StandardScaler().fit_transform(
        ops[["log_volume", "mean_res", "var_res", "overdue_rate"]]
    )
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_km)
    ops["cluster"] = km.labels_
    rank = ops.groupby("cluster")["overdue_rate"].mean().sort_values()
    archetype_map = dict(zip(rank.index, ["Fast Resolution", "Routine", "Slow Queue", "Chronic Backlog"]))
    ops["archetype"] = ops["cluster"].map(archetype_map)
    df["archetype"] = df["type"].map(ops["archetype"]).fillna("Rare")

    top_sources = df["source"].value_counts().nlargest(4).index
    df["source_group"] = np.where(df["source"].isin(top_sources), df["source"], "Other")

    df_ml = df.dropna(subset=SVI_FEATURES + ["is_overdue"]).copy()
    arch_dum = pd.get_dummies(df_ml["archetype"], prefix="Arch")
    src_dum = pd.get_dummies(df_ml["source_group"], prefix="Src")
    feat_cols = SVI_FEATURES + ["hour", "dayofweek", "month"]
    X = pd.concat(
        [df_ml[feat_cols].reset_index(drop=True),
         arch_dum.reset_index(drop=True),
         src_dum.reset_index(drop=True)],
        axis=1,
    ).astype(float)
    y = df_ml["is_overdue"].reset_index(drop=True).astype(int)
    log(f"classifier: X={X.shape} prevalence={y.mean():.4f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    specs = {
        "LogisticRegression": (
            LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"),
            True,
        ),
        "DecisionTree": (
            DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced"),
            False,
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42,
                class_weight="balanced", n_jobs=-1,
            ),
            False,
        ),
    }

    models: dict[str, dict] = {}
    best_name = None
    best_pr_auc = -1.0
    for name, (model, scale) in specs.items():
        log(f"classifier: fitting {name}")
        model.fit(X_tr_s if scale else X_tr.values, y_tr)
        y_prob = model.predict_proba(X_te_s if scale else X_te.values)[:, 1]
        auc = float(roc_auc_score(y_te, y_prob))
        pr_auc = float(average_precision_score(y_te, y_prob))
        models[name] = {"auc": auc, "pr_auc": pr_auc}
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
            best_prob = y_prob
            best_model = model
            best_scaled = scale

    log(f"classifier: best={best_name} PR-AUC={best_pr_auc:.4f}")

    prec, rec, thr = precision_recall_curve(y_te, best_prob)
    fpr, tpr, _ = roc_curve(y_te, best_prob)

    # Threshold tuned for recall >= 75% with max precision
    mask = rec[:-1] >= TARGET_RECALL
    if mask.any():
        candidates = np.where(mask)[0]
        idx = candidates[np.argmax(prec[:-1][candidates])]
        opt_thr = float(thr[idx])
    else:
        opt_thr = 0.5

    y_pred_base = (best_prob >= 0.5).astype(int)
    y_pred_opt = (best_prob >= opt_thr).astype(int)
    cm_base = confusion_matrix(y_te, y_pred_base).tolist()
    cm_opt = confusion_matrix(y_te, y_pred_opt).tolist()
    tp = int(cm_opt[1][1]); fn = int(cm_opt[1][0])
    fp = int(cm_opt[0][1]); tn = int(cm_opt[0][0])
    opt_recall = tp / max(tp + fn, 1)
    opt_precision = tp / max(tp + fp, 1)

    # Feature importance from RF (most interpretable)
    rf = specs["RandomForest"][0]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    feat_imp = [{"name": k, "imp": float(v)} for k, v in imp.head(15).items()]

    payload = {
        "source": "computed",
        "models": models,
        "best": best_name,
        "threshold": opt_thr,
        "prevalence": float(y_te.mean()),
        "metrics": {
            "recall": float(opt_recall),
            "precision": float(opt_precision),
            "auc": models[best_name]["auc"],
            "pr_auc": models[best_name]["pr_auc"],
        },
        "pr_curve": _downsample_curve(rec, prec, CURVE_POINTS),
        "roc_curve": _downsample_curve(fpr, tpr, CURVE_POINTS),
        "cm_base": cm_base,
        "cm_opt": cm_opt,
        "feat_imp": feat_imp,
    }
    # export_dashboard_data.py looks for _classifier_cache.json and stamps source="cached"
    write_json(OUT_DIR / "_classifier_cache.json", payload)


# ---------------------------------------------------------------------------
# router — 20 title predictions using lgbm + tfidf + label_encoder
# ---------------------------------------------------------------------------
CURATED_TITLES = [
    "Large pothole on Washington Street causing traffic issues",
    "Street light out on Boylston for 3 weeks",
    "Rats in the alley behind my building need exterminator",
    "Abandoned car blocking my driveway for a month",
    "Tree fell on power line after storm last night",
    "Graffiti on MBTA station wall near Kenmore",
    "Broken traffic signal at Mass Ave intersection",
    "Missed trash pickup on Harrison Ave yesterday",
    "Overflowing recycling bin in the park",
    "Illegal parking blocking the fire hydrant",
    "Water main break flooding the basement",
    "Snow not cleared from sidewalk in front of school",
    "Needle found on playground equipment",
    "Loud construction noise after 10pm in Allston",
    "Damaged street sign at school crossing zone",
    "Sewer backup onto residential street",
    "Dead raccoon removal needed on sidewalk",
    "Park bench vandalized with spray paint",
    "Request for new streetlight on dark corner",
    "Building permit inspection for basement construction",
]


def export_router() -> None:
    pkls = {
        "lgbm": MODELS_DIR / "lgbm_router.pkl",
        "tfidf": MODELS_DIR / "tfidf_router.pkl",
        "le": MODELS_DIR / "label_encoder.pkl",
    }
    missing = [p for p in pkls.values() if not p.exists()]
    if missing:
        log(f"router: missing {missing} — skip (run prd_dedup_routing.ipynb first)", level="WARN")
        return

    log("router: loading pkls")
    lgbm = joblib.load(pkls["lgbm"])
    tfidf = joblib.load(pkls["tfidf"])
    le = joblib.load(pkls["le"])

    # Build the same training subset notebook-04 used so we can (a) refit tfidf
    # if its idf_ vector got dropped on save, and (b) score held-out accuracy.
    df_full = pd.read_parquet(PARQUET, columns=["case_title", "department"])
    df_route = df_full.dropna(subset=["case_title", "department"]).copy()
    dept_counts = df_route["department"].value_counts()
    valid_depts = dept_counts[dept_counts >= 500].index
    df_route = df_route[df_route["department"].isin(valid_depts)]

    # The April rewrite re-pickled tfidf without the fitted idf_ vector
    # (sklearn cross-version save artefact). Vocab is deterministic given the
    # parquet + notebook-04 config, so refit in-place — feature count still
    # aligns with lgbm.n_features_in_.
    if not hasattr(tfidf, "idf_"):
        log("router: tfidf missing idf_ — refitting from parquet", level="WARN")
        tfidf.fit(df_route["case_title"])
        n_feat = tfidf.transform([""]).shape[1]
        if n_feat != lgbm.n_features_in_:
            log(
                f"router: refit produced {n_feat} features, expected "
                f"{lgbm.n_features_in_} — skipping router export",
                level="WARN",
            )
            return
    feat_names = tfidf.get_feature_names_out()

    # Held-out accuracy on the same 80/20 split notebook-04 uses
    # (random_state=42, stratify=y). Mirrors the value the dashboard expects.
    from sklearn.metrics import accuracy_score
    try:
        X_all = tfidf.transform(df_route["case_title"])
        y_all = le.transform(df_route["department"])
        _, X_test, _, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        test_accuracy = float(accuracy_score(y_test, lgbm.predict(X_test)))
        log(f"router: test_accuracy={test_accuracy:.4f} on {len(y_test):,} held-out tickets")
    except Exception as exc:  # noqa: BLE001
        log(f"router: test_accuracy calc failed ({exc})", level="WARN")
        test_accuracy = None

    samples = []
    for title in CURATED_TITLES:
        X = tfidf.transform([title])
        proba = lgbm.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        dept = str(le.classes_[pred_idx])
        conf = float(proba[pred_idx])

        nz = X.nonzero()[1]
        keywords = sorted(
            ((str(feat_names[j]), float(X[0, j])) for j in nz),
            key=lambda p: -p[1],
        )[:5]

        samples.append({
            "title": title,
            "predicted_dept": dept,
            "proba": round(conf, 4),
            "top_keywords": [[w, round(s, 4)] for w, s in keywords],
        })

    payload = {
        "source": "notebook",  # preserved by export_dashboard_data.py
        "n_departments": int(len(le.classes_)),
        "test_accuracy": test_accuracy,
        "samples": samples,
    }
    write_json(OUT_DIR / "router_samples.json", payload)


# ---------------------------------------------------------------------------
# dedup — Q1 2024 spatio-temporal scan (matches prd_dedup_routing.ipynb)
# ---------------------------------------------------------------------------
EARTH_R = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, lo1, la2, lo2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = sin(dlat / 2) ** 2 + cos(la1) * cos(la2) * sin(dlon / 2) ** 2
    return EARTH_R * 2 * atan2(sqrt(a), sqrt(1 - a))


def find_duplicates(
    df: pd.DataFrame,
    radius_m: float = 50.0,
    sim_threshold: float = 0.85,
    time_window_hours: float = 48.0,
) -> pd.DataFrame:
    work = df.dropna(subset=["latitude", "longitude", "case_title", "open_dt"]).copy()
    work = work.sort_values("open_dt").reset_index(drop=True)

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    M = tfidf.fit_transform(work["case_title"].fillna(""))

    pairs = []
    window = pd.Timedelta(hours=time_window_hours)
    for _, group in work.groupby("type"):
        if len(group) < 2:
            continue
        idxs = group.index.to_numpy()
        lats = group["latitude"].to_numpy()
        lons = group["longitude"].to_numpy()
        times = group["open_dt"].to_numpy()
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                dt = pd.Timestamp(times[j]) - pd.Timestamp(times[i])
                if dt > window:
                    break
                dist = haversine_m(lats[i], lons[i], lats[j], lons[j])
                if dist > radius_m:
                    continue
                sim = float(cosine_similarity(M[idxs[i]], M[idxs[j]])[0, 0])
                if sim < sim_threshold:
                    continue
                pairs.append({
                    "ticket_a": idxs[i],
                    "ticket_b": idxs[j],
                    "distance_m": dist,
                    "time_diff_h": dt.total_seconds() / 3600.0,
                    "text_sim": sim,
                })
    return pd.DataFrame(pairs), work


def export_dedup(df: pd.DataFrame) -> None:
    log("dedup: scanning Q1 2024")
    sample = df[
        (df["open_dt"] >= pd.Timestamp("2024-01-01", tz="UTC")) &
        (df["open_dt"] < pd.Timestamp("2024-04-01", tz="UTC"))
    ]
    log(f"dedup: sample size {len(sample):,}")
    pairs, work = find_duplicates(sample, radius_m=50, sim_threshold=0.85, time_window_hours=48)
    log(f"dedup: pairs found {len(pairs):,}")

    if pairs.empty:
        log("dedup: 0 pairs at 50m/85%/48h — leaving fallback in place", level="WARN")
        return

    # Top 10 example pairs ranked by text similarity × proximity. Drop pairs
    # with dist <= 5m before ranking — those are exact-coord re-submissions
    # (same address, GPS-noise duplicates) that dominate the score but read
    # as trivial in the dashboard. We want the story "near-by addresses
    # caught as same incident", which lives in the 5–50m band.
    GPS_NOISE_M = 5.0
    spatial = pairs[pairs["distance_m"] > GPS_NOISE_M].copy()
    if spatial.empty:
        spatial = pairs.copy()  # fall back if all pairs are co-located
    spatial["score"] = spatial["text_sim"] * (1 - spatial["distance_m"] / 50.0)
    top = spatial.sort_values("score", ascending=False).head(10)

    examples = []
    for _, p in top.iterrows():
        a = work.loc[p["ticket_a"]]
        b = work.loc[p["ticket_b"]]
        examples.append({
            "id1": str(a["case_enquiry_id"]),
            "id2": str(b["case_enquiry_id"]),
            "dist_m": round(float(p["distance_m"]), 1),
            "hours_apart": round(float(p["time_diff_h"]), 2),
            "type": str(a["type"]),
            "title1": str(a["case_title"]),
            "title2": str(b["case_title"]),
        })

    COST_PER_DISPATCH = 150
    annual_pairs = int(len(pairs)) * 4
    payload = {
        "source": "notebook",  # preserved by export_dashboard_data.py
        "summary": {
            "total_pairs": int(len(pairs)),
            "quarter": "Q1 2024",
            "avg_distance_m": round(float(pairs["distance_m"].mean()), 2),
            "avg_similarity": round(float(pairs["text_sim"].mean()), 4),
            "est_annual_savings_usd": int(annual_pairs * COST_PER_DISPATCH),
            "time_window_hours": 48,
            "distance_threshold_m": 50,
            "similarity_threshold": 0.85,
        },
        "pairs": examples,
    }
    write_json(OUT_DIR / "dedup_samples.json", payload)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_parquet()
    export_classifier(df)
    export_router()
    export_dedup(df)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
