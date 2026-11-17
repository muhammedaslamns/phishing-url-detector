import time

import joblib
import matplotlib
matplotlib.use("Agg")  # save charts as image files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------- 1. Load the data ----------
df = pd.read_csv("data/features.csv")
X = df.drop(columns=["URL", "label"])
y = df["label"]
feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.joblib")

# ---------- 2. Split: 80% training, 20% testing ----------
# stratify=y keeps the good/phishing ratio the same in both parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

# ---------- 3. Define the two models ----------
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15, min_samples_leaf=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=25, min_samples_leaf=3,
        n_jobs=-1, random_state=42
    ),
}

# ---------- 4. Train, evaluate, save ----------
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start

    pred = model.predict(X_test)
    row = {
        "Model": name,
        "Train accuracy": accuracy_score(y_train, model.predict(X_train)),
        "Test accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "Training time (s)": seconds,
    }
    results.append(row)

    tag = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{tag}.joblib", compress=3)

    # Confusion matrix picture
    cm = confusion_matrix(y_test, pred)
    print(f"{name} confusion matrix (rows = actual, columns = predicted):")
    print(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Good", "Phishing"]).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title(f"{name} - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(f"static/confusion_matrix_{tag}.png", dpi=150)
    plt.close(fig)

    # Feature importance picture (top 12)
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{name} - Top 12 Feature Importances")
    fig.tight_layout()
    fig.savefig(f"static/feature_importance_{tag}.png", dpi=150)
    plt.close(fig)

# ---------- 5. Picture of the top of the Decision Tree ----------
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(
    models["Decision Tree"], max_depth=3, feature_names=feature_names,
    class_names=["Good", "Phishing"], filled=True, fontsize=8, ax=ax,
)
fig.tight_layout()
fig.savefig("static/decision_tree_top.png", dpi=120)
plt.close(fig)

# ---------- 6. Results table ----------
table = pd.DataFrame(results).set_index("Model")
table.to_csv("models/results.csv")
print("\n================ FINAL RESULTS ================")
print(table.round(4).T)
print("\nSaved: models/, and charts in static/")