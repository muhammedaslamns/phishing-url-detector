import pandas as pd

# Load the dataset
df = pd.read_csv("data/archive/phishing_site_urls.csv")

print("Shape (rows, columns):", df.shape)
print("\nColumn names:", list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate URLs:", df["URL"].duplicated().sum())

print("\nClass balance:")
print(df["Label"].value_counts())

print("\nURL length (characters):")
print(df["URL"].str.len().describe())