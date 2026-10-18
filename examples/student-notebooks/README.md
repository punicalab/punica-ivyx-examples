# A student's notebooks

Three small notebooks over `data/sales.csv` (three cities, two months, one
missing value): `practice.ipynb`, where the student is studying `groupby`;
`homework2.ipynb`, whose last cell fails with a `KeyError`; and
`assignment3.ipynb`, empty but for its title, with the assignment in
`assignment.md`. A Python kernel with pandas runs them
(`pip install -r requirements.txt`).

## Try

| Skill | Type into the assistant | A good answer |
|---|---|---|
| `concept-practice` | *I'm practising groupby in my notebook practice.ipynb and don't really get it. Can you explain it and give me an exercise?* | Five sentences with placeholder names, then two cells at the end of the notebook: an exercise and an empty code cell. Never the solving line. |
| `notebook-error-hint` | *The last cell in homework2.ipynb fails. Why?* | Reads the cell's error from the notebook, names the cause in one hint (the column is `city`, not `City`) and lets the student fix it. |
| `assignment-starter` | *Help me start assignment.md in assignment3.ipynb.* | Reads the assignment, looks at the data once, and adds one `# TODO` code cell per requirement with a plan; it does not solve the assignment. |

Needs `extension-ipynb-viewer` and `extension-duckdb`.
