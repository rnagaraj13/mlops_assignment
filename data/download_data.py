"""Download the Cleveland file and write data/heart.csv."""

import os
import ssl
import urllib.request

import certifi
import pandas as pd

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "heart.csv")


def download_and_clean():
    print("Downloading dataset from UCI repository...")
    raw_path = OUTPUT_PATH.replace(".csv", "_raw.csv")

    # certifi keeps the download working on lab machines whose Python
    # install does not have the system certificate store wired up.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(DATASET_URL, context=ssl_context) as response:
        payload = response.read()
    with open(raw_path, "wb") as raw_file:
        raw_file.write(payload)
    print(f"Raw file saved to {raw_path}")

    # Load and assign column names
    df = pd.read_csv(raw_path, header=None, names=COLUMNS, na_values="?")

    # Binarise target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)

    # Drop rows with missing values
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Dropped {before - after} rows with missing values. {after} rows remaining.")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Clean dataset saved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    df = download_and_clean()
    print("\nDataset info:")
    print(df.dtypes)
    print(f"\nClass distribution:\n{df['target'].value_counts()}")
