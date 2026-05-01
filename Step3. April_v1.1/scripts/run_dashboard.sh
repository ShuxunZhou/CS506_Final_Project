#!/usr/bin/env bash
# End-to-end: 从现有 data/boston_311_2020_2026.parquet 出发，补齐 shapefile +
# spatial join + 模型，再重跑 export / build，最后可选起 HTTP server。
#
# 幂等：已存在的产物默认跳过；--force 强制重跑所有步骤。
# 不重跑 etl_311.py（耗时 ~20min，用户已有 parquet）。
#
# Usage:
#   ./scripts/run_dashboard.sh                # 增量跑缺的步骤
#   ./scripts/run_dashboard.sh --force        # 全部重跑
#   ./scripts/run_dashboard.sh --serve        # 结束后起 :8000
#   ./scripts/run_dashboard.sh --force --serve

set -euo pipefail

FORCE=0
SERVE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --serve) SERVE=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# 定位到 Step3. April_v1.1/ —— 脚本无论从哪里调用都能工作
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$STEP_DIR"

PY="${PYTHON:-python}"
log() { printf '\033[1;36m[run_dashboard]\033[0m %s\n' "$*"; }
skip() { printf '\033[0;33m[skip]\033[0m %s\n' "$*"; }

need() {
  # need <path>  —— 返回 0 表示需要重做（不存在 或 --force）
  [[ $FORCE -eq 1 ]] && return 0
  [[ ! -e "$1" ]]
}

run_nb() {
  local nb="$1"
  log "executing notebook: $nb"
  "$PY" -m jupyter nbconvert --to notebook --execute "$nb" --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=python3
}

# ---------- 0. 依赖 ----------
log "python = $(command -v "$PY")  ($("$PY" --version 2>&1))"
if ! "$PY" -c "import nbconvert" 2>/dev/null; then
  log "installing jupyter + nbconvert (一次性)"
  "$PY" -m pip install --quiet jupyter nbconvert ipykernel
fi
if ! "$PY" -c "import geopandas, statsmodels, lightgbm, plotly, jinja2, matplotlib, seaborn, shap" 2>/dev/null; then
  log "requirements.txt 里的包没装全，安装中"
  "$PY" -m pip install --quiet -r requirements.txt
fi

# ---------- 1. Shapefile ----------
SHP="data/shapefiles/tl_2022_25_tract.shp"
if need "$SHP"; then
  log "① 下载 TIGER 2022 MA tract shapefile"
  "$PY" scripts/download_data.py
else
  skip "① shapefile 已存在 ($SHP)"
fi

# ---------- 2. Spatial join ----------
SVI_PARQUET="data/boston_311_with_svi.parquet"
if need "$SVI_PARQUET"; then
  log "② Spatial join (311 points → census tracts → SVI)"
  run_nb notebooks/spatial_join.ipynb
else
  skip "② $SVI_PARQUET 已存在"
fi

# ---------- 3. 模型 ----------
#  export_dashboard_data.py 会检查 models/ 下的 pkl；缺则 fallback。
#  overdue_classifier 产出分类器，prd_dedup_routing 产出 TF-IDF router + dedup。
mkdir -p models
CLASSIFIER_PKL=$(ls models/*classifier*.pkl models/*overdue*.pkl 2>/dev/null | head -1 || true)
ROUTER_PKL=$(ls models/*router*.pkl models/*tfidf*.pkl 2>/dev/null | head -1 || true)

if [[ -z "$CLASSIFIER_PKL" || $FORCE -eq 1 ]]; then
  log "③a 训练 overdue classifier"
  run_nb notebooks/overdue_classifier.ipynb
else
  skip "③a classifier pkl 已存在 ($CLASSIFIER_PKL)"
fi

if [[ -z "$ROUTER_PKL" || $FORCE -eq 1 ]]; then
  log "③b 训练 router + dedup (PRD)"
  run_nb notebooks/prd_dedup_routing.ipynb
else
  skip "③b router pkl 已存在 ($ROUTER_PKL)"
fi

# equity_regression 的系数 export 脚本会从 parquet 直接重跑 statsmodels 得到，
# 所以这本默认跳过。--force 时一并跑，便于对数。
if [[ $FORCE -eq 1 ]]; then
  log "③c (--force) 重跑 equity regression notebook"
  run_nb notebooks/equity_regression.ipynb
fi

# ---------- 4. Export + build ----------
log "④a 重跑 classifier + router + dedup (真数据 → dashboard/data/)"
"$PY" scripts/export_models_data.py

log "④b 重导 regression / tracts / missing bias JSON"
"$PY" scripts/export_dashboard_data.py

log "④ 渲染 dashboard/index.html"
"$PY" scripts/build_dashboard.py

# ---------- 5. 验证 source ----------
log "JSON source 标签："
for f in dashboard/data/*.json; do
  src=$(grep -oE '"source":"[^"]+"' "$f" | head -1 || echo '"source":"(none)"')
  printf '  %-40s %s\n' "$(basename "$f")" "$src"
done

# ---------- 6. (可选) 起服务器 ----------
if [[ $SERVE -eq 1 ]]; then
  log "启动 http://localhost:8000  (Ctrl-C 停)"
  cd dashboard && exec "$PY" -m http.server 8000
fi

log "完成 ✓"
