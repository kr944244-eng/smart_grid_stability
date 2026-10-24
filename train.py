"""
Smart Grid Stability Prediction - Model Training
-------------------------------------------------
Run:  python train.py
Needs: smart_grid_stability_augmented.csv in the same folder.
Creates: models/model.pkl and models/metrics.json (used by app.py)
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = "smart_grid_stability_augmented.csv"
FEATURES = ["tau1", "tau2", "tau3", "tau4",
            "p2", "p3", "p4",
            "g1", "g2", "g3", "g4"]
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------------ 1. Load
print("Loading data ...")
df = pd.read_csv(DATA_FILE)
print(f"  rows={len(df)}, columns={df.shape[1]}, missing values={df.isna().sum().sum()}")
print(df["stabf"].value_counts().to_string())

# ------------------------------------------------------------------ 2. Prepare
# 'stab' is the raw stability value that 'stabf' is computed from -> leakage, drop it.
# 'p1' = -(p2 + p3 + p4) (producer power = total consumption) -> redundant, drop it.
X = df[FEATURES]
y = (df["stabf"] == "stable").astype(int)   # 1 = stable, 0 = unstable

# The augmented dataset = the original 10,000 simulations x 6 orderings of the
# 3 consumer nodes. Row i, i+10000, i+20000 ... are the SAME grid. A random split
# would put copies of one grid in both train and test (data leakage), so we
# split by original simulation ID instead.
groups = np.arange(len(df)) % 10000
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"  train={len(X_train)}  test={len(X_test)} (no grid appears in both)")

# ------------------------------------------------------------------ 3. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08,
                                                        max_leaf_nodes=48, random_state=42),
    "Neural Network (ANN)": MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-4,
                                          learning_rate_init=1e-3, max_iter=400,
                                          early_stopping=True, n_iter_no_change=20,
                                          random_state=42),
}

results = []
fitted = {}
for name, clf in models.items():
    t0 = time.time()
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    res = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "train_time_s": round(time.time() - t0, 1),
    }
    results.append(res)
    fitted[name] = pipe
    print(f"  {name:<22} acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  "
          f"auc={res['roc_auc']:.4f}  ({res['train_time_s']}s)")

# ------------------------------------------------------------------ 4. Best model
best = max(results, key=lambda r: r["f1"])
best_name = best["model"]
best_model = fitted[best_name]
print(f"\nBest model: {best_name}")

cm = confusion_matrix(y_test, best_model.predict(X_test)).tolist()

# Which inputs matter most? (permutation importance on a test sample)
print("Computing feature importance ...")
sample = X_test.sample(4000, random_state=0)
pi = permutation_importance(best_model, sample, y_test.loc[sample.index],
                            scoring="f1", n_repeats=5, random_state=0, n_jobs=-1)
importance = sorted(
    [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(FEATURES, pi.importances_mean)],
    key=lambda d: d["importance"], reverse=True)

# Example grids for the website's preset buttons
ex_stable = df[df["stabf"] == "stable"].sort_values("stab").iloc[50][FEATURES].round(3).to_dict()
ex_unstable = df[df["stabf"] == "unstable"].sort_values("stab").iloc[-50][FEATURES].round(3).to_dict()

# ------------------------------------------------------------------ 5. Save
joblib.dump(best_model, "models/model.pkl")
meta = {
    "best_model": best_name,
    "features": FEATURES,
    "results": results,
    "confusion_matrix": cm,
    "importance": importance,
    "dataset": {"rows": int(len(df)), "stable": int(y.sum()), "unstable": int((1 - y).sum()),
                "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
    "ranges": {f: [round(float(df[f].min()), 3), round(float(df[f].max()), 3)] for f in FEATURES},
    "examples": {"stable": ex_stable, "unstable": ex_unstable},
}
with open("models/metrics.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print("Saved models/model.pkl and models/metrics.json")
print("Now run:  python app.py")
