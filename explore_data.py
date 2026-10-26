import io
import ssl
import zipfile
import urllib.request

import certifi
import pandas as pd

# dataset is downloaded from UCI every time, so it is not stored in the repo
DATA_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
XLSX_NAME = "DryBeanDataset/Dry_Bean_Dataset.xlsx"
TARGET = "Class"


def load_data(url=DATA_URL):
    # certifi gives a fresh certificate list (the Windows one failed for me)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context, timeout=60) as resp:
        zip_bytes = resp.read()
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    df = pd.read_excel(io.BytesIO(z.read(XLSX_NAME)))
    return df


def summarise(df):
    print("Shape:", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nClass counts:")
    counts = df[TARGET].value_counts()
    for name in counts.index:
        pct = counts[name] / len(df) * 100
        print(f"{name:10s} {counts[name]:5d}  {pct:.1f}%")

    print("\nFeature summary:")
    x = df.drop(columns=[TARGET])
    print(x.describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    df = load_data()
    summarise(df)
