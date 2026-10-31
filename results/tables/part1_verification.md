# Part 1: our exact search vs the shipped ground truth

`naive` is the plain set-intersection recall; `tie-aware` accepts a returned
vector whose exact distance equals the k-th ground-truth distance within
`rtol=1e-5`. Where the two differ, the disagreement is a tie, not an error.

| dataset | queries | naive recall@10 | tie-aware recall@10 | queries with a differing top-10 set | queries still below 1.0 |
|---|---|---|---|---|---|
| sift | 5000 | 0.999500 | 1.000000 | 25 | 0 |
| glove | 5000 | 1.000000 | 1.000000 | 0 | 0 |
