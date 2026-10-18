"""Pandera contract for data/train.csv."""

import pandera.pandas as pa

schema = pa.DataFrameSchema(
    {
        "x1": pa.Column(float),
        "x2": pa.Column(float),
        "label": pa.Column(int, pa.Check.isin([0, 1])),
    }
)
