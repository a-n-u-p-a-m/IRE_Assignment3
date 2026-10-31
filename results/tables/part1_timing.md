# Part 1: exact search, one query at a time vs one batch

Both measured on a single thread, median of repeated runs.

| dataset | n | d | single-query QPS | batched QPS | speedup | queries timed |
|---|---|---|---|---|---|---|
| sift | 1,000,000 | 128 | 38.2 | 82.3 | 2.2x | 512 |
| glove | 1,183,514 | 100 | 42.4 | 75.8 | 1.8x | 512 |
