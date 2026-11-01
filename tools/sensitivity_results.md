# TI-HOTA sensitivity (training GT as predictions)

Mean over 8 games, tau=100cm, fast metric. Corruptions applied in 30-frame blocks, seed 0.

| scenario | TI-HOTA | DetA | AssA |
|---|---|---|---|
| gaussian noise sigma=5cm | 0.9965 | 0.9965 | 0.9965 |
| gaussian noise sigma=10cm | 0.9428 | 0.9428 | 0.9428 |
| gaussian noise sigma=20cm | 0.7545 | 0.7545 | 0.7545 |
| gaussian noise sigma=30cm | 0.5543 | 0.5543 | 0.5543 |
| gaussian noise sigma=50cm | 0.2801 | 0.2801 | 0.2801 |
| gaussian noise sigma=75cm | 0.1351 | 0.1351 | 0.1351 |
| constant bias 10cm (x) | 1.0000 | 1.0000 | 1.0000 |
| constant bias 20cm (x) | 0.8949 | 0.8949 | 0.8949 |
| constant bias 30cm (x) | 0.7897 | 0.7897 | 0.7897 |
| constant bias 50cm (x) | 0.4743 | 0.4743 | 0.4743 |
| team labels swapped in 5% frames | 0.9243 | 0.9047 | 0.9444 |
| team labels swapped in 10% frames | 0.8532 | 0.8182 | 0.8899 |
| team labels swapped in 25% frames | 0.6633 | 0.6012 | 0.7323 |
| team labels swapped in 50% frames | 0.4009 | 0.3349 | 0.4808 |
| 2 same-team ids swapped in 5% frames | 0.9678 | 0.9673 | 0.9684 |
| 2 same-team ids swapped in 10% frames | 0.9375 | 0.9355 | 0.9394 |
| 2 same-team ids swapped in 25% frames | 0.8565 | 0.8463 | 0.8668 |
| 2 same-team ids swapped in 50% frames | 0.7460 | 0.7150 | 0.7782 |
| player dropped in 5% frames | 0.9917 | 0.9917 | 0.9917 |
| player stale (held) in 5% frames | 0.9920 | 0.9919 | 0.9921 |
| player dropped in 10% frames | 0.9834 | 0.9834 | 0.9834 |
| player stale (held) in 10% frames | 0.9820 | 0.9816 | 0.9825 |
| player dropped in 25% frames | 0.9584 | 0.9584 | 0.9584 |
| player stale (held) in 25% frames | 0.9533 | 0.9504 | 0.9561 |
| player dropped in 50% frames | 0.9169 | 0.9169 | 0.9168 |
| player stale (held) in 50% frames | 0.9022 | 0.8903 | 0.9143 |
| 7th spurious row/frame (unknown id) | 0.9260 | 0.8574 | 1.0000 |
| 7th row/frame = duplicate id 300cm away | 0.8866 | 0.8574 | 0.9168 |
