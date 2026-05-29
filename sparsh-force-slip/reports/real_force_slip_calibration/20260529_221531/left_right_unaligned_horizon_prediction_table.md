# Left/Right Unaligned Real Tactile Prediction Table

- Source: object_holdout/window_level_predictions.csv
- Left/right sensors are shown independently; no temporal alignment or side fusion is applied.
- Current future head outputs H1/H3/H5 only; H2/H4 are not available.
- Optimized deployment prediction uses window-level pSlip_max threshold 0.97 from object-holdout validation.

| object | seq | side | window | frames | target | H1 | H3 | H5 | pSlip_mean | pSlip_max | Calib_mean | opt_pred |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| banana | banana_1 | left | slip 27-37 | 11 | instability | 0.798 | 1.000 | 1.000 | 0.251 | 0.753 | 0.617 | stable |
| banana | banana_1 | left | stable 40-70 | 31 | stable | 0.473 | 0.988 | 1.000 | 0.001 | 0.002 | 0.320 | stable |
| banana | banana_1 | right | slip 27-37 | 11 | instability | 0.879 | 0.987 | 0.996 | 0.541 | 1.000 | 0.672 | instability |
| banana | banana_1 | right | stable 40-70 | 31 | stable | 0.627 | 0.777 | 0.801 | 0.089 | 0.555 | 0.335 | stable |
| banana | banana_2 | left | slip 26-40 | 15 | instability | 0.955 | 0.986 | 0.994 | 0.960 | 0.997 | 0.993 | instability |
| banana | banana_2 | left | stable 60-90 | 31 | stable | 0.889 | 0.979 | 0.990 | 0.720 | 0.969 | 0.896 | stable |
| banana | banana_2 | right | slip 26-40 | 15 | instability | 0.814 | 0.921 | 0.930 | 0.711 | 1.000 | 0.774 | instability |
| banana | banana_2 | right | stable 60-90 | 31 | stable | 0.782 | 0.987 | 0.997 | 0.294 | 0.776 | 0.657 | stable |
| cucumber | cucumber_1 | left | slip 30-60 | 31 | instability | 0.920 | 0.999 | 1.000 | 0.683 | 1.000 | 0.836 | instability |
| cucumber | cucumber_1 | right | slip 30-60 | 31 | instability | 0.895 | 0.988 | 0.996 | 0.686 | 1.000 | 0.837 | instability |
| cucumber | cucumber_2 | left | slip 38-54 | 17 | instability | 0.961 | 0.999 | 1.000 | 0.830 | 1.000 | 0.883 | instability |
| cucumber | cucumber_2 | right | slip 38-54 | 17 | instability | 0.998 | 0.998 | 0.999 | 0.997 | 1.000 | 0.993 | instability |
| cucumber | cucumber_3 | left | slip 40-64 | 25 | instability | 0.988 | 1.000 | 1.000 | 0.618 | 1.000 | 0.779 | instability |
| cucumber | cucumber_3 | left | stable 80-110 | 31 | stable | 0.915 | 1.000 | 1.000 | 0.025 | 0.241 | 0.505 | stable |
| cucumber | cucumber_3 | right | slip 40-64 | 25 | instability | 0.825 | 0.992 | 0.999 | 0.932 | 1.000 | 0.960 | instability |
| cucumber | cucumber_3 | right | stable 80-110 | 31 | stable | 0.305 | 0.817 | 0.929 | 0.039 | 0.135 | 0.235 | stable |
| hammer | hammer_1 | left | slip 50-80 | 31 | instability | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 | instability |
| hammer | hammer_1 | right | slip 50-80 | 31 | instability | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 | instability |
| hammer | hammer_2 | left | slip 34-54 | 21 | instability | 1.000 | 1.000 | 1.000 | 0.892 | 1.000 | 0.979 | instability |
| hammer | hammer_2 | left | stable 90-116 | 27 | stable | 1.000 | 1.000 | 1.000 | 0.723 | 0.758 | 0.960 | stable |
| hammer | hammer_2 | right | slip 34-54 | 21 | instability | 1.000 | 1.000 | 1.000 | 0.789 | 1.000 | 0.934 | instability |
| hammer | hammer_2 | right | stable 90-116 | 27 | stable | 1.000 | 1.000 | 1.000 | 0.419 | 0.738 | 0.758 | stable |
| hammer | hammer_3 | left | slip 40-70 | 31 | instability | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.991 | instability |
| hammer | hammer_3 | right | slip 40-70 | 31 | instability | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 | instability |
| lemon | lemon_1 | left | slip 28-39 | 12 | instability | 0.906 | 0.996 | 1.000 | 0.356 | 0.956 | 0.656 | stable |
| lemon | lemon_1 | left | stable 50-80 | 31 | stable | 0.589 | 0.877 | 0.921 | 0.001 | 0.002 | 0.306 | stable |
| lemon | lemon_1 | right | slip 28-39 | 12 | instability | 0.748 | 0.995 | 1.000 | 0.417 | 1.000 | 0.622 | instability |
| lemon | lemon_1 | right | stable 50-80 | 31 | stable | 0.686 | 0.997 | 1.000 | 0.007 | 0.076 | 0.385 | stable |
| lemon | lemon_2 | left | slip 28-40 | 13 | instability | 1.000 | 1.000 | 1.000 | 0.539 | 0.996 | 0.714 | instability |
| lemon | lemon_2 | left | stable 60-90 | 31 | stable | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.255 | stable |
| lemon | lemon_2 | right | slip 28-40 | 13 | instability | 1.000 | 1.000 | 1.000 | 0.588 | 1.000 | 0.731 | instability |
| lemon | lemon_2 | right | stable 60-90 | 31 | stable | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.274 | stable |
| lemon | lemon_3 | left | slip 27-37 | 11 | instability | 1.000 | 1.000 | 1.000 | 0.663 | 1.000 | 0.788 | instability |
| lemon | lemon_3 | left | stable 70-100 | 31 | stable | 1.000 | 1.000 | 1.000 | 0.000 | 0.001 | 0.261 | stable |
| lemon | lemon_3 | right | slip 27-37 | 11 | instability | 1.000 | 1.000 | 1.000 | 0.543 | 1.000 | 0.681 | instability |
| lemon | lemon_3 | right | stable 70-100 | 31 | stable | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.281 | stable |
| lemon | lemon_4 | left | slip 31-41 | 11 | instability | 1.000 | 1.000 | 1.000 | 0.519 | 1.000 | 0.657 | instability |
| lemon | lemon_4 | left | stable 70-100 | 31 | stable | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.260 | stable |
| lemon | lemon_4 | right | slip 31-41 | 11 | instability | 1.000 | 1.000 | 1.000 | 0.710 | 1.000 | 0.816 | instability |
| lemon | lemon_4 | right | stable 70-100 | 31 | stable | 1.000 | 1.000 | 1.000 | 0.001 | 0.002 | 0.253 | stable |
| lemon | lemon_5 | left | slip 31-42 | 12 | instability | 1.000 | 1.000 | 1.000 | 0.515 | 1.000 | 0.685 | instability |
| lemon | lemon_5 | left | stable 70-100 | 31 | stable | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.247 | stable |
| lemon | lemon_5 | right | slip 31-42 | 12 | instability | 1.000 | 1.000 | 1.000 | 0.861 | 1.000 | 0.924 | instability |
| lemon | lemon_5 | right | stable 70-100 | 31 | stable | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.264 | stable |
