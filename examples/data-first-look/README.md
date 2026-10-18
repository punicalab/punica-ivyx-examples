# A new data file

The Titanic passenger list as a training split (`data/train.csv`, 891 rows)
and a test split that is not clean (`data/test.csv`, 200 rows: 40 of them
copied from the training split, 3 duplicated), plus a small Parquet file
(`data/trips.parquet`). Nothing to install: DuckDB runs inside Studio.

## Try

| Skill | Type into the assistant | A good answer |
|---|---|---|
| `dataset-first-look` | *What is in data/train.csv, and what is wrong with it? Missing values, odd columns, class balance.* | Profiles the file before judging it: 891 rows, `Cabin` 77.1 % missing, `Embarked` 2 rows, `PassengerId` named an identifier, survival 38 % against 62 %. No number is guessed. |
| `train-test-leakage-check` | *Is my test set leaking from my training set? Check data/train.csv against data/test.csv: overlapping rows, duplicates, label balance.* | Registers both files, runs the intersect and the distinct-count queries, and reports 40 leaked rows, 3 duplicate rows and the label balance of each split. |

Needs `extension-duckdb` and `extension-profiler` (both ship with Studio).
