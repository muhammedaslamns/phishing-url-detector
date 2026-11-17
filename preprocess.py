import pandas as pd
from feature_extractor import extract_features

RAW_FILE = "data/archive/phishing_site_urls.csv"
OUT_FILE = "data/features.csv"

df = pd.read_csv(RAW_FILE)
print("Rows loaded:", len(df))

# 1. Basic cleaning
df["URL"] = df["URL"].astype(str).str.strip()
df = df.dropna()

# 2. Remove exact duplicate rows
df = df.drop_duplicates()
print("After removing duplicate rows:", len(df))

# 3. Remove URLs that appear with conflicting labels
df = df.drop_duplicates(subset="URL", keep=False)
print("After removing conflicting URLs:", len(df))

# 4. Convert labels to numbers: bad = 1 (phishing), good = 0 (legitimate)
df["label"] = (df["Label"] == "bad").astype(int)
df = df.reset_index(drop=True)

# 5. Extract features (this takes 1 to 3 minutes)
print("Extracting features, please wait...")
features = pd.DataFrame([extract_features(u) for u in df["URL"]])

# 6. Combine and save
final = pd.concat([df[["URL", "label"]], features], axis=1)
final.to_csv(OUT_FILE, index=False)

print("\nSaved to:", OUT_FILE)
print("Final shape:", final.shape)
print("\nClass balance (0 = good, 1 = bad):")
print(final["label"].value_counts())
print("\nFirst 3 rows:")
print(final.head(3).T)