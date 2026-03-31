#!/bin/bash
# =============================================================================
# MEGA-RAG: Kaggle Setup & Fine-Tuning Push Script
# =============================================================================
#
# One-time setup:
#   1. Install Kaggle CLI
#   2. Get API token from https://www.kaggle.com/settings → "Create New API Token"
#   3. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
#   4. Run this script
#
# Usage:
#   chmod +x scripts/kaggle_setup.sh
#   ./scripts/kaggle_setup.sh setup     # First time setup
#   ./scripts/kaggle_setup.sh push      # Upload dataset + notebook
#   ./scripts/kaggle_setup.sh status    # Check notebook run status
#   ./scripts/kaggle_setup.sh pull      # Download results
# =============================================================================

set -e

# ---- CONFIG (change these) ----
KAGGLE_USERNAME="${KAGGLE_USERNAME:-your-kaggle-username}"   # Set this!
DATASET_SLUG="${KAGGLE_USERNAME}/pubmedqa-splits"
KERNEL_SLUG="${KAGGLE_USERNAME}/mega-rag-finetune-llama31"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ---- FUNCTIONS ----

setup() {
    echo "=== KAGGLE SETUP ==="

    # Install kaggle CLI
    if ! command -v kaggle &> /dev/null; then
        echo "Installing kaggle CLI..."
        pip3 install --user kaggle
    fi

    # Check API token
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo ""
        echo "ERROR: Kaggle API token not found!"
        echo ""
        echo "Steps:"
        echo "  1. Go to https://www.kaggle.com/settings"
        echo "  2. Scroll to 'API' section"
        echo "  3. Click 'Create New API Token'"
        echo "  4. Save the downloaded file to ~/.kaggle/kaggle.json"
        echo "  5. Run: chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        exit 1
    fi

    chmod 600 ~/.kaggle/kaggle.json
    echo "Kaggle CLI ready!"
    kaggle --version
}

push_dataset() {
    echo "=== PUSHING DATASET ==="

    # Create dataset metadata
    DATASET_DIR="$PROJECT_DIR/kaggle_dataset_tmp"
    rm -rf "$DATASET_DIR"
    mkdir -p "$DATASET_DIR"

    # Copy split files
    cp "$PROJECT_DIR/pubmedQA/splits/train_oversampled.json" "$DATASET_DIR/"
    cp "$PROJECT_DIR/pubmedQA/splits/train_balanced.json" "$DATASET_DIR/"
    cp "$PROJECT_DIR/pubmedQA/splits/dev.json" "$DATASET_DIR/"
    cp "$PROJECT_DIR/pubmedQA/splits/test.json" "$DATASET_DIR/"

    # Create dataset-metadata.json
    cat > "$DATASET_DIR/dataset-metadata.json" << EOF
{
    "title": "PubMedQA Splits (Official Protocol)",
    "id": "${DATASET_SLUG}",
    "licenses": [{"name": "CC0-1.0"}]
}
EOF

    echo "Uploading dataset..."
    cd "$DATASET_DIR"
    kaggle datasets create -p . --dir-mode zip 2>/dev/null || \
    kaggle datasets version -p . -m "Updated splits" --dir-mode zip

    rm -rf "$DATASET_DIR"
    echo "Dataset uploaded: https://www.kaggle.com/datasets/${DATASET_SLUG}"
}

push_notebook() {
    echo "=== PUSHING NOTEBOOK ==="

    KERNEL_DIR="$PROJECT_DIR/kaggle_kernel_tmp"
    rm -rf "$KERNEL_DIR"
    mkdir -p "$KERNEL_DIR"

    # Copy notebook
    cp "$PROJECT_DIR/notebooks/finetune_llama31_pubmedqa.ipynb" "$KERNEL_DIR/"

    # Create kernel-metadata.json
    cat > "$KERNEL_DIR/kernel-metadata.json" << EOF
{
    "id": "${KERNEL_SLUG}",
    "title": "MEGA-RAG: Fine-tune Llama 3.1 8B on PubMedQA (QLoRA)",
    "code_file": "finetune_llama31_pubmedqa.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": true,
    "enable_gpu": true,
    "enable_internet": true,
    "dataset_sources": ["${DATASET_SLUG}"],
    "competition_sources": [],
    "kernel_sources": []
}
EOF

    echo "Pushing notebook..."
    cd "$KERNEL_DIR"
    kaggle kernels push -p .

    rm -rf "$KERNEL_DIR"
    echo ""
    echo "Notebook pushed! It will start running automatically."
    echo "Check status: ./scripts/kaggle_setup.sh status"
    echo "View at: https://www.kaggle.com/code/${KERNEL_SLUG}"
}

check_status() {
    echo "=== CHECKING STATUS ==="
    kaggle kernels status "${KERNEL_SLUG}"
}

pull_results() {
    echo "=== PULLING RESULTS ==="

    OUTPUT_DIR="$PROJECT_DIR/kaggle_output"
    mkdir -p "$OUTPUT_DIR"

    kaggle kernels output "${KERNEL_SLUG}" -p "$OUTPUT_DIR"

    echo ""
    echo "Results downloaded to: $OUTPUT_DIR/"
    ls -la "$OUTPUT_DIR/"

    # Check for adapter zip
    if [ -f "$OUTPUT_DIR/llama31-pubmedqa-lora-adapter.zip" ]; then
        echo ""
        echo "LoRA adapter found! To use with Ollama:"
        echo "  1. unzip $OUTPUT_DIR/llama31-pubmedqa-lora-adapter.zip -d ./lora-adapter"
        echo "  2. Create a Modelfile (see notebook for template)"
        echo "  3. ollama create llama31-medical -f Modelfile"
        echo "  4. OLLAMA_MODEL=llama31-medical LLM_PROVIDER=ollama python3 run.py --interactive"
    fi

    # Check for training summary
    if [ -f "$OUTPUT_DIR/training_summary.json" ]; then
        echo ""
        echo "Training summary:"
        cat "$OUTPUT_DIR/training_summary.json"
    fi
}

# ---- MAIN ----

case "${1:-help}" in
    setup)
        setup
        ;;
    push)
        push_dataset
        push_notebook
        ;;
    dataset)
        push_dataset
        ;;
    notebook)
        push_notebook
        ;;
    status)
        check_status
        ;;
    pull)
        pull_results
        ;;
    help|*)
        echo "Usage: $0 {setup|push|dataset|notebook|status|pull}"
        echo ""
        echo "  setup     - Install kaggle CLI, verify API token"
        echo "  push      - Upload dataset + notebook to Kaggle (starts training)"
        echo "  dataset   - Upload only the dataset"
        echo "  notebook  - Upload only the notebook"
        echo "  status    - Check if training is running/complete"
        echo "  pull      - Download training results (adapter + summary)"
        echo ""
        echo "First-time setup:"
        echo "  1. Get API token: https://www.kaggle.com/settings → Create New API Token"
        echo "  2. Save to: ~/.kaggle/kaggle.json"
        echo "  3. export KAGGLE_USERNAME=your-username"
        echo "  4. $0 setup"
        echo "  5. $0 push"
        ;;
esac
