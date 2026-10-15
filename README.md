# FedAssess: Federated Viability Index (FVI) Platform

An end-to-end **Federated Learning Feasibility & Benchmarking Platform** structured in standard Python `src/` layout (`src/fvi/`) with a minimalist, modern **FastAPI Web UI** (`app/server.py` + `app/static/index.html`) and a headless CLI runner (`scripts/run_fvi_cli.py`).

Designed for seamless development on your **local Ubuntu machine**, reproducible sharing with collaborators on **GitHub**, and remote GPU execution over **SSH on `rama` (NVIDIA L40S)**.

---

## 1. Environment Setup (Local Ubuntu, Collaborators & `rama` L40S)

You can create a dedicated environment (`fvi-env`) using either **Option A (1-Command Automated Script)**, **Option B (Conda `environment.yml`)**, or **Option C (Standard `conda create` / `venv` + `pip`)**.

### Option A: One-Command Automated Setup (Recommended)
Run the included setup script, which automatically creates the `fvi-env` Conda environment (or `.venv` if Conda is not installed), installs all required packages from `requirements.txt`, installs `fvi` in editable mode (`pip install -e .`), and runs the hardware/pipeline smoke test:
```bash
bash scripts/setup_env.sh fvi-env
conda activate fvi-env
```

### Option B: Conda `environment.yml`
```bash
conda env create -f environment.yml
conda activate fvi-env
python scripts/00_verify_env.py
```

### Option C: Step-by-Step Conda or `venv` + `pip`
```bash
# 1. Create and activate a fresh Python 3.11 environment
conda create -n fvi-env python=3.11 -y
conda activate fvi-env

# 2. Install all required libraries (PyTorch, Torchvision, FastAPI, Uvicorn, NumPy, Matplotlib, Pillow, Pytest)
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install the `fvi` package in editable mode (src/ layout)
pip install -e .

# 4. Verify environment, GPU/BF16 support, and pipeline smoke test
python scripts/00_verify_env.py
```

### Required Libraries (`requirements.txt`)
* `torch>=2.1.0` & `torchvision>=0.16.0` — Deep learning & L40S CUDA/BF16 execution
* `fastapi>=0.104.0`, `uvicorn[standard]>=0.24.0`, `pydantic>=2.4.0`, `python-multipart>=0.0.6` — REST API & Web UI server
* `numpy>=1.24.0`, `matplotlib>=3.7.0`, `Pillow>=10.0.0` — Data preprocessing, image loading, and diagnostic plots
* `pytest>=7.4.0` — Unit and integration test suite

---

## 2. Running the FastAPI Web UI on `rama` (L40S) via SSH

Because `rama` is accessed over SSH, use **SSH Local Port Forwarding** so the FastAPI web server running on `rama` (with direct access to the NVIDIA L40S GPU) opens in your local laptop's browser:

### Step 1: Connect to `rama` with Port Forwarding (from your local laptop terminal)
```bash
ssh -L 8080:127.0.0.1:8080 <your_username>@rama
```
*(If you use VS Code Remote-SSH, VS Code automatically forwards port `8080` to your local browser.)*

### Step 2: Activate `fvi-env` and Start the FastAPI Server on `rama`
```bash
conda activate fvi-env
cd fvi-platform

# Launch the FastAPI + Uvicorn server on port 8080:
python app/server.py --port 8080
# (Or equivalently via Uvicorn CLI or installed entry point):
# uvicorn app.server:fastapi_app --host 127.0.0.1 --port 8080
# fvi-server --port 8080
```

### Step 3: Open in Your Local Browser
Open **`http://localhost:8080`** in Chrome or Firefox on your laptop.

---

## 3. Running Headless CLI Benchmarks & Unit Tests

```bash
conda activate fvi-env

# Run the unit and integration test suite:
pytest -v
# (or: python -m unittest discover -s tests -v)

# Run a headless Federated Viability Index study from the terminal:
python scripts/run_fvi_cli.py \
    --data-path data/pe_malware_features.csv \
    --target malware_family \
    --clients 10 \
    --strategy dirichlet \
    --alpha 0.05
```

---

## 4. Repository Structure (`src/` Layout)

```text
fvi-platform/
├── environment.yml                  # Conda environment specification (`fvi-env`)
├── requirements.txt                 # Pip dependencies (PyTorch, FastAPI, Uvicorn, etc.)
├── pyproject.toml                   # PEP-621 package configuration (`src/` layout)
├── README.md                        # Setup and execution guide
├── app/
│   ├── server.py                    # FastAPI + Uvicorn REST backend (with HTTP fallback)
│   └── static/
│       └── index.html               # Minimalist, modern 4-step single-page web UI
├── scripts/
│   ├── setup_env.sh                 # 1-command environment creation & setup script
│   ├── 00_verify_env.py             # Hardware (L40S), library & smoke-test verifier
│   └── run_fvi_cli.py               # Headless CLI runner for batch benchmarks
├── src/
│   └── fvi/
│       ├── config.py                # Dataclasses for Dataset, Partition, Model & Benchmark
│       ├── cli_entry.py             # Console script entry points (`fvi-server`, `fvi-bench`)
│       ├── ingestion/               # Tabular CSV & Image (Folder/Manifest) loaders + Malware generators
│       ├── partition/               # IID, Dirichlet Dir_M(alpha), Quantity Skew, Generalist-Specialist
│       ├── models/                  # Mode 1 Unified TabularMLP, VisionResNet8, SimpleCNN
│       ├── engines/                 # Centralized, Isolated Silos, Iterative FL, and FENS engines
│       ├── metrics/                 # Evaluator, FVI Calculator (CPR, CG, HR), and Matplotlib plots
│       └── pipeline.py              # Master end-to-end orchestrator
└── tests/
    └── test_fvi_platform.py         # Automated unit & integration test suite
```
