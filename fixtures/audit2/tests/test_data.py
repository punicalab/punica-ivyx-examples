"""Sanity tests over the training data."""

import pandas as pd


def test_data_has_rows():
    df = pd.read_csv("data/train.csv")
    assert len(df) > 100


def test_labels_are_binary():
    df = pd.read_csv("data/train.csv")
    assert set(df["label"].unique()) <= {0, 1}
