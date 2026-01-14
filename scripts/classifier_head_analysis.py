from pathlib import Path
import sys
import os
import time
import torch
import numpy as np

from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut

# --- Paths / Imports ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GENOMIC_FM_DIR = REPO_ROOT / "external" / "genomic-FM"
DATA_DIR = GENOMIC_FM_DIR / "root" / "data"  # where verified_real_clinvar.csv likely lives

print("\n[1/6] Setting up import paths...")
print(f"  - REPO_ROOT:     {REPO_ROOT}")
print(f"  - GENOMIC_FM_DIR:{GENOMIC_FM_DIR}")
print(f"  - DATA_DIR:      {DATA_DIR}")

sys.path.insert(0, str(GENOMIC_FM_DIR))
os.chdir(GENOMIC_FM_DIR)

# Load Embeddings
path = "root/data/clinvar_pooled_embeddings__n155__bp3000__tok505__layers10.pt"
payload = torch.load(path, map_location="cpu")

print("Loading embeddings from:", path)

layers = payload["layers"]      # store all layer information
labels = np.array(payload["labels"])        # store labels
n = len(labels)  # number of variants 

# Construct Logistic Regression Classifier Pipeline
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("PCA", PCA(n_components=50)),  # optional: reduce dimensionality
    ("lr", LogisticRegression(max_iter=5000, class_weight="balanced"))
])

# Analyze Each Layer and Conduct Stratified K-Fold Cross-Validation
print("\nPerforming Stratified K-Fold Cross-Validation for each layer...")
layers = [1, 3, 5, 9, 12, 15, 18, 22, 25, 28]  # 0..28 indexing
for layer in layers:
    E = payload["embeddings_by_layer"][layer]  # shape (2N, 1024), torch.Tensor

    ref = E[:n].numpy()      # (N, 1024)
    alt = E[n:].numpy()      # (N, 1024)

    X = alt - ref            # (N, 1024)

    kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

    for i, (train_index, test_index) in enumerate(kfolds.split(X, labels)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
