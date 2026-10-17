# ==============================================================================
# Makefile: Automation Pipeline for Sentinel-2 Crop Classification (TSViT-DINOv3)
# ==============================================================================

SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
STREAMLIT := $(VENV)/bin/streamlit
PIP := $(VENV)/bin/pip

.PHONY: help venv install test train export ui docker-build docker-run clean all

help: ## Show this help message
	@echo "=========================================================================="
	@echo "🌱 Sentinel-2 Crop Classification (TSViT-DINOv3) Automation Pipeline"
	@echo "=========================================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment using uv or python
	@if command -v uv >/dev/null 2>&1; then \
		echo "[VENV] Creating virtual environment with uv..."; \
		uv venv --python /usr/bin/python3.12 --system-site-packages $(VENV) --allow-existing; \
	else \
		echo "[VENV] Creating virtual environment with python3..."; \
		python3 -m venv --system-site-packages $(VENV); \
	fi

install: venv ## Install project dependencies
	@echo "[INSTALL] Installing dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(VENV) -r requirements.txt; \
	else \
		$(PIP) install -r requirements.txt; \
	fi

test: ## Run full unit test suite

	@echo "[TEST] Running unit tests..."
	@PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -p "test_*.py"

train: ## Train TSViT-DINOv3 model and export checkpoints
	@echo "[TRAIN] Starting TSViT-DINOv3 model training..."
	@PYTHONPATH=. $(PYTHON) train.py --epochs 3 --batch-size 8 --output-dir checkpoints

export: ## Export PyTorch checkpoint to ONNX format
	@echo "[EXPORT] Exporting model to ONNX..."
	@PYTHONPATH=. $(PYTHON) -c "from src.models.tsvit_dinov3 import TSViT_DINOv3; from src.utils.export import export_onnx; m = TSViT_DINOv3(); export_onnx(m, 'checkpoints/tsvit_dinov3_crop.onnx', num_timesteps=20, img_res=24)"

ui: ## Launch the interactive Streamlit Web Application
	@echo "[UI] Launching Streamlit interface on port 8501..."
	@PYTHONPATH=. $(STREAMLIT) run app/app.py --server.port 8501

docker-build: ## Build production Docker container image
	@echo "[DOCKER] Building container image: crop-tsvit-dinov3..."
	@docker build -t crop-tsvit-dinov3:latest .

docker-run: ## Run Docker container on port 7860
	@echo "[DOCKER] Running container on http://localhost:7860..."
	@docker run --rm -p 7860:7860 crop-tsvit-dinov3:latest

clean: ## Clean cache, bytecode, and temporary test artifacts
	@echo "[CLEAN] Removing pycache and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.py[cod]" -delete
	@rm -rf .pytest_cache /tmp/test_s2_data_pipeline /tmp/test_tsvit_model
	@echo "[CLEAN] Done."

all: install test train export ## Execute complete pipeline (install, test, train, export)
