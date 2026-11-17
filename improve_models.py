import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------- Load the same data and the same split as before ----------
df = pd.read_csv("data/features.csv")
X = df.drop(columns=["URL", "label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# class_weight="balanced" makes missing a phishing URL cost more during training
models = {
    "Decision Tree (balanced)": DecisionTreeClassifier(
        max_depth=15, min_samples_leaf=5, class_weight="balanced", random_state=42
    ),
    "Random Forest (balanced)": RandomForestClassifier(
        n_estimators=100, max_depth=25, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "Gradient Boosting (balanced)": HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=42
    ),
}

# ---------- 1. Train and test on the held-out 20% ----------
rows = []
trained = {}
for name, model in models.items():
    print(f"Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "Test accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "Training time (s)": seconds,
    })
    trained[name] = model

results = pd.DataFrame(rows).set_index("Model")
results.to_csv("models/results_improved.csv")
print("\n=== Test results with class_weight='balanced' ===")
print(results.round(4).T)

joblib.dump(trained["Random Forest (balanced)"],
            "models/random_forest_balanced.joblib", compress=3)
joblib.dump(trained["Gradient Boosting (balanced)"],
            "models/gradient_boosting_balanced.joblib", compress=3)

# ---------- 2. 5-fold cross-validation on all the data ----------
print("\nRunning 5-fold cross-validation (takes a few minutes)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]
cv_rows = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
    row = {"Model": name}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric} mean"] = values.mean()
        row[f"{metric} std"] = values.std()
    cv_rows.append(row)
    print(f"  done: {name}")

cv_table = pd.DataFrame(cv_rows).set_index("Model")
cv_table.to_csv("models/cross_validation.csv")
print("\n=== 5-fold cross-validation (mean and spread) ===")
print(cv_table.round(4).T)

# ---------- 3. Precision-recall curve ----------
fig, ax = plt.subplots(figsize=(7, 5))
for name, model in trained.items():
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ax.plot(recall, precision, label=name)
ax.set_xlabel("Recall (share of phishing URLs caught)")
ax.set_ylabel("Precision (share of flags that are correct)")
ax.set_title("Precision-Recall Curve")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("static/precision_recall_curve.png", dpi=150)
print("\nSaved: models/results_improved.csv, models/cross_validation.csv,")
print("       static/precision_recall_curve.png")