"""
Anomaly Detector

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def load_data(filepath_or_url):
    df = pd.read_csv(filepath_or_url)
    print(f"Loaded data — shape: {df.shape}")
    print(df.head())
    return df


def check_data(df):
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nStatistics:")
    print(df.describe())
    return df


def clean_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df


def detect_anomalies(df, feature_columns, contamination=0.01):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(df[feature_columns])
    df["anomaly"] = model.predict(df[feature_columns])

    counts = df["anomaly"].value_counts()
    print(f"\nNormal points: {counts.get(1, 0)}")
    print(f"Anomalies found: {counts.get(-1, 0)}")
    return df, model


def plot_anomalies(df, x_column, y_column):
    normal = df[df["anomaly"] == 1]
    anomaly = df[df["anomaly"] == -1]

    plt.figure(figsize=(12, 5))
    plt.plot(df[x_column], df[y_column], color="lightblue", zorder=1, label="Data")
    plt.scatter(anomaly[x_column], anomaly[y_column], color="red", s=20, zorder=2, label="Anomaly")
    plt.title("Detected Anomalies")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("=== Anomaly Detection Tool ===\n")

    source = input("Enter CSV file path or URL: ").strip()
    df = load_data(source)

    print(f"\nAvailable columns: {list(df.columns)}")

    feature_input = input("\nEnter column(s) to check for anomalies (comma separated if more than one): ")
    feature_columns = [col.strip() for col in feature_input.split(",")]

    # Ask expected anomaly percentage (optional, defaults to 1%)
    contamination_input = input("Expected anomaly percentage, e.g. 1 for 1% (press Enter for default 1%): ").strip()
    contamination = float(contamination_input) / 100 if contamination_input else 0.01

    df = check_data(df)
    df = clean_data(df)

    df, model = detect_anomalies(df, feature_columns=feature_columns, contamination=contamination)

    x_col = input("\nEnter column for X-axis, e.g. a timestamp column (press Enter to use row number): ").strip()
    if x_col == "":
        df["row_number"] = df.index
        x_col = "row_number"
    elif "time" in x_col.lower() or "date" in x_col.lower():
        df[x_col] = pd.to_datetime(df[x_col])

    plot_anomalies(df, x_column=x_col, y_column=feature_columns[0])
