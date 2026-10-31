# Part 1: exact search, one query at a time vs one batch

Both measured on a single thread, median of repeated runs.

| dataset | n | d | single-query QPS | batched QPS | speedup | queries timed |
|---|---|---|---|---|---|---|
| sift | 1,000,000 | 128 | 36.8 | 80.7 | 2.2x | 512 |
| glove | 1,183,514 | 100 | 41.2 | 95.7 | 2.3x | 512 |
