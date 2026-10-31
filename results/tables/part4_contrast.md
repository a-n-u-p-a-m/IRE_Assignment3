# Part 4: relative contrast per condition

Relative contrast is the mean distance from a query to a random sample of the
database divided by its distance to the true nearest neighbour. Values near 1
mean nothing is distinctly closest, which is the regime where partition-based
pruning stops working.

`NN / noise floor` is the mean true nearest-neighbour distance divided by
the expected distance between two noise vectors. Below ~1 the nearest
neighbour is determined by the noise rather than by the subspace, which
happens at the lowest k because many points packed into few effective
dimensions sit closer together than any noise floor that keeps the data
full-rank.

| k_intrinsic | control | median RC | p10 | p90 | NN dist | NN / noise floor |
|---|---|---|---|---|---|---|
| 2 |  | 18.816 | 15.005 | 26.152 | 0.0637 | 0.90 |
| 8 |  | 5.258 | 4.292 | 6.471 | 0.2681 | 3.79 |
| 32 |  | 1.853 | 1.752 | 1.995 | 0.7568 | 10.70 |
| 128 | yes | 1.313 | 1.289 | 1.348 | 1.0711 | 15.15 |
