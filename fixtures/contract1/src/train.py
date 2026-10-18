"""Train a tiny classifier and write metrics.json + model.pkl."""

import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    df = pd.read_csv("data/train.csv")
    X, y = df[["x1", "x2"]], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    model = LogisticRegression().fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": round(accuracy, 4)}, f)
    print(f"accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
