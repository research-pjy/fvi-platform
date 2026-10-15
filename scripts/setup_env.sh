#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="${1:-fvi-env}"
PYTHON_VER="3.11"

echo "=========================================================================="
echo "  Setting up FedAssess (Federated Viability Index) Environment: ${ENV_NAME}"
echo "=========================================================================="

if command -v conda >/dev/null 2>&1; then
    echo "[1/4] Detected Conda. Creating/updating Conda environment ${ENV_NAME} (Python ${PYTHON_VER})..."
    eval "$(conda shell.bash hook)"
    if conda info --envs | awk "{print \$1}" | grep -qx "${ENV_NAME}"; then
        echo "  -> Conda environment ${ENV_NAME} already exists. Activating..."
    else
        conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y
    fi
    conda activate "${ENV_NAME}"
else
    echo "[1/4] Conda not found. Creating Python virtual environment .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

echo "[2/4] Upgrading pip and installing required libraries from requirements.txt..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[3/4] Installing fvi package in editable mode (src/ layout)..."
python -m pip install -e . --no-deps

echo "[4/4] Running Environment & Hardware Smoke Test..."
python scripts/00_verify_env.py

echo "=========================================================================="
echo "  Setup Complete! Activate with: conda activate ${ENV_NAME}"
echo "  Start FastAPI Server with    : python app/server.py --port 8080"
echo "=========================================================================="
