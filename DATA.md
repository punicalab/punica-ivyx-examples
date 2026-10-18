# Where the data comes from

- **Titanic** (`data/train.csv`, `data/titanic/train.csv`): the passenger list
  of the RMS Titanic as compiled from Encyclopedia Titanica and distributed
  by Vanderbilt University's biostatistics department, in the 891-row
  training split that Kaggle's "Titanic: Machine Learning from Disaster"
  made familiar. The records are public domain. `data/test.csv` in the
  data-first-look example is not Kaggle's test set: it is 200 rows cut
  from the same training split, 40 of them copied unchanged from
  `train.csv` and 3 duplicated, so that a leakage check has something to
  find.
- **`data/trips.parquet`**: three synthetic trips (`trip_id`, `started`,
  `km`), there so that a first look has a Parquet file beside a CSV.
- **`data/train.csv` in the MLOps pipeline** (`x1`, `x2`, `label`): 400
  synthetic rows, two Gaussian blobs (180 rows labelled `0` around
  (-0.7, -0.4), 210 labelled `1` around (0.7, 0.35)) and ten rows labelled
  `2` near the origin that the data contract rejects.
- **`data/sales.csv`**: six synthetic rows (three cities, two months, one
  missing value).
- **Kestrel Outdoor** (`docs/`): an invented shop, its policies and its
  products, written for the examples. Any resemblance to a real shop is
  accidental.

Everything in this repository is under the MIT licence in `LICENSE`.
