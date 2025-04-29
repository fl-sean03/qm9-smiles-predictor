Below is a proposed **README.md** for the project. It lives at the repo root (`README.md`) and explains data layout, installation, and the end-to-end workflows for each of the four methods (Baseline, Multitask, Autoencoder, Hybrid).

```markdown
# CHEM555Project: QM9 Property Prediction Workflows

This repository implements and compares four distinct feed-forward neural-network (FNN) pipelines for predicting nine QM9 molecular properties:
1. **Baseline**: single-task FNN on Mordred descriptors  
2. **Multitask**: shared-encoder FNN on Mordred descriptors  
3. **Autoencoder**: FNNs on autoencoder embeddings of Mordred descriptors  
4. **Hybrid**: single-task FNN on concatenated ECFP4 fingerprints + Mordred descriptors  

---

## 📁 Repository Layout

```
.
├── .gitignore
├── README.md
├── data/
│   ├── mordred_features/
│   │   ├── mordred_cleaned.csv    # Cleaned Mordred descriptors
│   │   ├── orig_idx.txt           # List of orig_index
│   │   ├── X_autoencoder.npy      # AE embeddings
│   │   └── X_hybrid.npy           # Hybrid features
│   └── qm9/
│       ├── qm9_clean.csv          # Raw QM9 SMILES + properties
│       └── qm9_targets.csv        # Extracted 9-targets
├── outputs/                     # Generated output files (excluding large CSVs and run directories)
│   ├── baseline/
│   │   └── evaluation/          # Evaluation results (metrics, plots, smaller CSVs)
│   ├── multitask/
│   │   └── eval_multitask/      # Evaluation results (metrics, plots, smaller CSVs)
│   ├── autoencoder/
│   │   └── eval_ae_run/         # Evaluation results (metrics, plots, smaller CSVs)
│   └── hybrid/
│       └── hybrid_eval/         # Evaluation results (metrics, plots, smaller CSVs)
└── src/                         # Source code for workflows
    ├── __init__.py
    ├── gen_qm9_targets.py
    ├── autoencoder/
    │   ├── __init__.py
    │   ├── eval_qm9_with_ae.py
    │   ├── featurize_autoencoder.py
    │   ├── model.py
    │   ├── train_autoencoder.py
    │   ├── train_qm9_with_ae.py
    │   └── viz_qm9_with_ae.py
    ├── baseline/
    │   ├── __init__.py
    │   ├── eval_qm9.py
    │   ├── train_baseline_qm9.py
    │   └── viz_qm9.py
    ├── hybrid/
    │   ├── __init__.py
    │   ├── eval_hybrid.py
    │   ├── featurize_hybrid.py
    │   ├── train_hybrid.py
    │   ├── utils.py
    │   └── viz_hybrid.py
    ├── mordred/
    │   ├── __init__.py
    │   ├── clean_mordred.py
    │   ├── extract_mordred.py
    │   └── verify_mordred.py
    └── multitask/
        ├── __init__.py
        ├── eval_multitask_qm9.py
        ├── train_multitask_qm9.py
        └── viz_multitask_qm9.py
```

---

## 🚀 Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-org/5555CHEM-proj.git
   cd 5555CHEM-proj
   ```
2. **Create** a Python 3.9+ virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   *(Make sure `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `rdkit`, `mordred`, etc. are included.)*

3. **Prepare Data**:
   - Place `qm9_clean.csv` and `qm9_targets.csv` under `data/qm9/`.
   - Place cleaned Mordred CSV under `data/mordred_features/mordred_cleaned.csv`.

---

## 📊 Workflows

### 1. Baseline (Mordred-only)

```bash
# 1) Train single-task FNNs on Mordred
python src/train_baseline_qm9.py \
  --descriptors_csv data/mordred_features/mordred_cleaned.csv \
  --targets_csv     data/qm9/qm9_targets.csv \
  --output_dir      outputs/baseline/model_run

# 2) Evaluate vs. paper MAEs
python src/eval_qm9.py \
  --descriptors_csv data/mordred_features/mordred_cleaned.csv \
  --targets_csv     data/qm9/qm9_targets.csv \
  --run_dir         outputs/baseline/model_run \
  --out_dir         outputs/baseline/evaluation

# 3) Visualize parity, residuals, comparison
python src/viz_qm9.py \
  --eval_dir       outputs/baseline/evaluation \
  --comparison_dir outputs/baseline/evaluation/comparison
```

---

### 2. Multitask (Mordred shared FNN)

```bash
# Train multitask FNN
python src/train_multitask_qm9.py \
  --descriptors_csv data/mordred_features/mordred_cleaned.csv \
  --targets_csv     data/qm9/qm9_targets.csv \
  --output_dir      outputs/multitask/multitask_run

# Evaluate multitask
python src/eval_multitask_qm9.py \
  --descriptors_csv data/mordred_features/mordred_cleaned.csv \
  --targets_csv     data/qm9/qm9_targets.csv \
  --out_dir         outputs/multitask/eval_multitask

# Visualize multitask
python src/viz_multitask_qm9.py \
  --eval_dir       outputs/multitask/eval_multitask \
  --comparison_dir outputs/multitask/eval_multitask/comparison
```

---

### 3. Autoencoder (AE → FNN)

```bash
# 1) Train autoencoder on Mordred features
python src/autoencoder/train_autoencoder.py \
  --input-desc   data/mordred_features/mordred_cleaned.csv \
  --output-dir   outputs/autoencoder/autoencoder_run \
  --bottleneck-dim 100

# 2) Generate AE embeddings
python src/autoencoder/featurize_autoencoder.py \
  --input-desc       data/mordred_features/mordred_cleaned.csv \
  --encoder-weights  outputs/autoencoder/autoencoder_run/encoder_best.weights.h5 \
  --scaler           outputs/autoencoder/autoencoder_run/scaler.pkl \
  --output-emb       data/mordred_features/X_autoencoder.npy

# 3) Train FNNs on AE embeddings
python src/autoencoder/train_qm9_with_ae.py \
  --embeddings        data/mordred_features/X_autoencoder.npy \
  --descriptors_csv   data/mordred_features/mordred_cleaned.csv \
  --targets_csv       data/qm9/qm9_targets.csv \
  --output_dir        outputs/autoencoder/fnn_ae_run

# 4) Evaluate AE-run FNNs
python src/autoencoder/eval_qm9_with_ae.py \
  --embeddings      data/mordred_features/X_autoencoder.npy \
  --descriptors_csv data/mordred_features/mordred_cleaned.csv \
  --targets_csv     data/qm9/qm9_targets.csv \
  --run_dir         outputs/autoencoder/fnn_ae_run \
  --out_dir         outputs/autoencoder/eval_ae_run

# 5) Visualize AE-run
python src/autoencoder/viz_qm9_with_ae.py \
  --eval_dir        outputs/autoencoder/eval_ae_run \
  --comparison_dir  outputs/autoencoder/eval_ae_run/comparison
```

---

### 4. Hybrid (ECFP4 + Mordred)

```bash
# 1) Extract orig_index list from cleaned Mordred
cut -d, -f1 data/mordred_features/mordred_cleaned.csv > data/mordred_features/orig_idx.txt

# 2) Featurize: build hybrid fingerprint+Mordred matrix
python src/hybrid/featurize_hybrid.py \
  --smiles_csv  data/qm9/qm9_clean.csv \
  --desc_csv    data/mordred_features/mordred_cleaned.csv \
  --output_npy  data/mordred_features/X_hybrid.npy \
  --radius      2 \
  --nBits       1024

# 3) Train FNNs on hybrid features
python src/hybrid/train_hybrid.py \
  --embeddings     data/mordred_features/X_hybrid.npy \
  --orig_index     data/mordred_features/orig_idx.txt \
  --targets_csv    data/qm9/qm9_targets.csv \
  --output_dir     outputs/hybrid/hybrid_run \
  --batch_size     256 \
  --epochs         200 \
  --learning_rate  1e-3 \
  --hidden_layers  256,256 \
  --patience       10

# 4) Evaluate hybrid models
python src/hybrid/eval_hybrid.py \
  --embeddings   data/mordred_features/X_hybrid.npy \
  --orig_index   data/mordred_features/orig_idx.txt \
  --targets_csv  data/qm9/qm9_targets.csv \
  --run_dir      outputs/hybrid/hybrid_run \
  --out_dir      outputs/hybrid/hybrid_eval

# 5) Visualize hybrid
python src/hybrid/viz_hybrid.py \
  --eval_dir       outputs/hybrid/hybrid_eval \
  --comparison_dir outputs/hybrid/hybrid_eval/comparison
```

---

## ⚙️ Configuration

- All scripts accept command-line flags for batch size, learning rate, hidden layers, patience, etc.
- Default FNN architecture is two hidden layers of size `256,256` with ReLU activations and Adam optimizer.

---

## 📄 License & Citation

Please cite the original Pinheiro et al. (2020) **J. Phys. Chem. A** paper when using or building on this work.

---

Happy modeling! 🚀