# =============================================================================
# Boston 311 Service Latency & Equity Analysis -- Makefile
# =============================================================================
# Quick start:
#   make install       # create venv and install all dependencies
#   make build         # build data + train models + render dashboard
#   make serve         # serve dashboard at http://localhost:8000
#   make all           # install + build + serve  (one-shot)
#
# Incremental build (skip existing artifacts): default behavior
# Full rebuild (ignore existing artifacts):    make rebuild
# Remove generated artifacts:                  make clean
# =============================================================================

SHELL          := /bin/bash
.SHELLFLAGS    := -eu -o pipefail -c

# ----- Paths ----------------------------------------------------------------
ROOT_DIR       := $(CURDIR)
STEP_DIR       := $(ROOT_DIR)/Step3. April_v1.1
DASH_DIR       := $(STEP_DIR)/dashboard
VENV_DIR       := $(ROOT_DIR)/venv
REQUIREMENTS   := $(STEP_DIR)/requirements.txt
RUN_SCRIPT     := $(STEP_DIR)/scripts/run_dashboard.sh

# Pipeline artifacts (used as guards so re-runs skip work)
SHAPEFILE      := $(STEP_DIR)/data/shapefiles/tl_2022_25_tract.shp
SVI_PARQUET    := $(STEP_DIR)/data/boston_311_with_svi.parquet
SPATIAL_JOIN_NB:= $(ROOT_DIR)/notebooks/01_spatial_join.ipynb

# Python / pip inside the venv
PY             := $(VENV_DIR)/bin/python
PIP            := $(VENV_DIR)/bin/pip

# Host python used to bootstrap the venv. Override with `make PYTHON=python3.11 install`.
PYTHON         ?= python3
PORT           ?= 8000

# ----- Default goal ---------------------------------------------------------
.DEFAULT_GOAL  := help

.PHONY: help install venv deps build rebuild data spatial-join dashboard serve clean clean-venv all check

help:
	@echo "Boston 311 Equity Analysis -- Make targets"
	@echo ""
	@echo "  make install     create venv and install requirements.txt + jupyter"
	@echo "  make build       end-to-end: download data -> train models -> export JSON -> render HTML"
	@echo "  make rebuild     force full rebuild (ignore existing artifacts)"
	@echo "  make serve       start local HTTP server at http://localhost:$(PORT)"
	@echo "  make all         install + build + serve"
	@echo "  make check       verify venv and core dependencies are ready"
	@echo "  make clean       remove generated dashboard JSON / index.html / model pkl files"
	@echo "  make clean-venv  also remove the venv directory (use with care)"
	@echo ""
	@echo "Override variables: PYTHON=python3.11   PORT=8080"

# ----- Install --------------------------------------------------------------
install: venv deps

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	@echo ">>> Creating venv with $(PYTHON)"
	$(PYTHON) -m venv "$(VENV_DIR)"
	$(PIP) install --upgrade pip setuptools wheel

deps: venv
	@echo ">>> Installing requirements.txt"
	@test -f "$(REQUIREMENTS)" || (echo "ERROR: $(REQUIREMENTS) not found" && exit 1)
	$(PIP) install -r "$(REQUIREMENTS)"
	@echo ">>> Installing notebook execution deps (jupyter / nbconvert / ipykernel)"
	$(PIP) install jupyter nbconvert ipykernel

check: venv
	@echo ">>> Checking core dependencies"
	$(PY) -c "import pandas, numpy, geopandas, sklearn, statsmodels, lightgbm, plotly, jinja2, matplotlib, seaborn; print('OK -- core dependencies are ready')"

# ----- Build ----------------------------------------------------------------
# End-to-end pipeline (incremental: each step is guarded by its own check).
#   1. shapefile + SVI csv  -- download if absent
#   2. spatial-join parquet -- execute notebook if absent
#   3. export model JSON    -- export_models_data.py retrains internally
#   4. export aggregate JSON + tracts geojson
#   5. render index.html from Jinja template
build: venv data spatial-join dashboard
	@echo ">>> Build complete"

# Force-rerun every stage by removing artifact guards first.
rebuild: venv
	@echo ">>> Full rebuild (removing guard artifacts)"
	-rm -f "$(SVI_PARQUET)"
	-rm -f "$(DASH_DIR)/index.html"
	-rm -f "$(DASH_DIR)/data/"*.json
	$(MAKE) build

# Stage 1: shapefile + SVI csv (download_data.py is idempotent).
data: venv
	@echo ">>> Stage 1: shapefile + SVI csv"
	@if [ -f "$(SHAPEFILE)" ]; then \
		echo "[skip] shapefile already present"; \
	else \
		cd "$(STEP_DIR)" && "$(PY)" "$(ROOT_DIR)/scripts/download_data.py"; \
	fi

# Stage 2: spatial join (run notebook only if parquet missing).
spatial-join: venv data
	@echo ">>> Stage 2: spatial join"
	@if [ -f "$(SVI_PARQUET)" ]; then \
		echo "[skip] $(SVI_PARQUET) already present"; \
	elif [ -f "$(SPATIAL_JOIN_NB)" ]; then \
		"$(PY)" -m jupyter nbconvert --to notebook --execute "$(SPATIAL_JOIN_NB)" \
			--inplace --ExecutePreprocessor.timeout=1800 \
			--ExecutePreprocessor.kernel_name=python3; \
	else \
		echo "ERROR: $(SPATIAL_JOIN_NB) not found" && exit 1; \
	fi

# Stage 3-5: export JSON payloads + render dashboard (model retrain is internal).
dashboard: venv
	@echo ">>> Stage 3: export model payloads"
	cd "$(STEP_DIR)" && "$(PY)" scripts/export_models_data.py
	@echo ">>> Stage 4: export aggregate JSON + tracts geojson"
	cd "$(ROOT_DIR)" && "$(PY)" scripts/export_dashboard_data.py
	@echo ">>> Stage 5: render index.html"
	cd "$(ROOT_DIR)" && "$(PY)" scripts/build_dashboard.py

# ----- Run ------------------------------------------------------------------
serve: venv
	@echo ">>> Serving dashboard at http://localhost:$(PORT)/  (Ctrl-C to stop)"
	cd "$(DASH_DIR)" && "$(PY)" -m http.server $(PORT)

all: install build serve

# ----- Clean ----------------------------------------------------------------
clean:
	@echo ">>> Removing generated artifacts"
	-rm -f "$(DASH_DIR)/index.html"
	-rm -f "$(DASH_DIR)/data/"*.json
	-rm -f "$(DASH_DIR)/data/"*.geojson
	-rm -rf "$(STEP_DIR)/models"/*.pkl
	-rm -rf "$(ROOT_DIR)/exports"/*.csv "$(ROOT_DIR)/exports"/*.txt

clean-venv:
	@echo ">>> Removing venv"
	-rm -rf "$(VENV_DIR)"
