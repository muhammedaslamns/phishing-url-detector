import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

from feature_extractor import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


def model_path(name):
    return os.path.join(BASE_DIR, "models", name)


# ---------- Load the trained models once, when the app starts ----------
feature_names = joblib.load(model_path("feature_names.joblib"))
MODELS = {
    "Decision Tree": joblib.load(model_path("decision_tree.joblib")),
    "Random Forest": joblib.load(model_path("random_forest.joblib")),
    "Random Forest (balanced)": joblib.load(model_path("random_forest_balanced.joblib")),
}


def load_table(filename):
    """Read a saved results CSV and format it for the page."""
    path = model_path(filename)
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "model": r["Model"],
            "test": f"{r['Test accuracy'] * 100:.2f}%",
            "precision": f"{r['Precision'] * 100:.2f}%",
            "recall": f"{r['Recall'] * 100:.2f}%",
            "f1": f"{r['F1-score']:.3f}",
            "time": f"{r['Training time (s)']:.1f} s",
        })
    return rows


RESULTS_STANDARD = load_table("results.csv")
RESULTS_BALANCED = load_table("results_improved.csv")


def warning_signs(f):
    """Plain-language signs found in the URL (descriptive, not the model's exact logic)."""
    signs = []
    if f["has_ip_address"]:
        signs.append("Uses a raw IP address instead of a domain name")
    if f["has_at_symbol"]:
        signs.append("Contains an '@' symbol, which can hide the real destination")
    if f["suspicious_word_count"] >= 2:
        signs.append(f"Contains {f['suspicious_word_count']} sensitive words such as login, verify, secure or paypal")
    elif f["suspicious_word_count"] == 1:
        signs.append("Contains a sensitive word such as login, verify or secure")
    if f["is_shortener"]:
        signs.append("Uses a URL shortener that hides the real address")
    if f["num_subdomains"] >= 3:
        signs.append(f"Has {f['num_subdomains']} subdomains, which is unusually many")
    if f["host_num_hyphens"] >= 2:
        signs.append("Domain name contains several hyphens")
    if f["url_length"] > 75:
        signs.append(f"Long URL ({f['url_length']} characters)")
    if f["digit_ratio"] > 0.25:
        signs.append("Unusually high share of digits in the URL")
    if f["double_slash_in_path"]:
        signs.append("Contains a '//' inside the path, a redirection trick")
    return signs


def run_models(url, threshold):
    feats = extract_features(url)
    row = pd.DataFrame([feats])[feature_names]   # same column order as training
    predictions = []
    for name, model in MODELS.items():
        prob = float(model.predict_proba(row)[0][1])   # probability of phishing
        is_phishing = prob >= threshold / 100
        predictions.append({
            "model": name,
            "verdict": "Phishing" if is_phishing else "Looks Legitimate",
            "css": "danger" if is_phishing else "safe",
            "probability": round(prob * 100, 1),
        })
    return predictions, warning_signs(feats)


@app.route("/", methods=["GET", "POST"])
def index():
    url, error, predictions, signs = "", None, None, []
    threshold = 50
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        try:
            threshold = int(float(request.form.get("threshold", 50)))
        except ValueError:
            threshold = 50
        threshold = max(10, min(90, threshold))   # keep it between 10 and 90
        if not url:
            error = "Please enter a URL."
        elif len(url) > 2000:
            error = "That URL is too long (limit is 2000 characters)."
        else:
            predictions, signs = run_models(url, threshold)
    return render_template(
        "index.html", url=url, error=error, threshold=threshold,
        predictions=predictions, signs=signs,
        results_standard=RESULTS_STANDARD, results_balanced=RESULTS_BALANCED,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)