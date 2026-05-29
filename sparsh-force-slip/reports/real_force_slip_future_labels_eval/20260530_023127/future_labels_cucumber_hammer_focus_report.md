# Future-label Real Dataset Re-test Focus Report

- source_report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_future_labels_eval/20260530_023127`
- label_source: `future_labels.safe_windows` as stable/future-safe and `future_labels.slip_onset/slip_end` as unstable/slip.
- stage-II checkpoint: horizon-fix best head `sharp_sphere_to_flat_balanced_bce_smooth/checkpoints/best.pth`.
- focus objects: cucumber and hammer, because their slip windows are longer and visually clearer.

## Overall metrics

| score | thr | n | stable mean | slip mean | gap | Acc | F1 | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p_slip_current@0.5 | 0.500 | 931 | 0.122 | 0.877 | 0.755 | 0.879 | 0.857 | 71 | 479 |
| p_instability_H1@0.5 | 0.500 | 931 | 0.915 | 0.987 | 0.072 | 0.409 | 0.581 | 550 | 0 |
| p_instability_H3@0.5 | 0.500 | 931 | 0.944 | 0.991 | 0.048 | 0.409 | 0.581 | 550 | 0 |
| p_instability_H5@0.5 | 0.500 | 931 | 0.956 | 0.993 | 0.037 | 0.409 | 0.581 | 550 | 0 |
| p_instability_H1@orig-val-thr | 0.848 | 931 | 0.915 | 0.987 | 0.072 | 0.516 | 0.622 | 441 | 109 |
| p_instability_H3@orig-val-thr | 0.728 | 931 | 0.944 | 0.991 | 0.048 | 0.415 | 0.583 | 545 | 5 |
| p_instability_H5@orig-val-thr | 0.784 | 931 | 0.956 | 0.993 | 0.037 | 0.415 | 0.583 | 545 | 5 |

## Cucumber + hammer focus metrics

| score | thr | n | stable mean | slip mean | gap | Acc | F1 | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p_slip_current@0.5 | 0.500 | 401 | 0.283 | 0.930 | 0.647 | 0.863 | 0.907 | 37 | 79 |
| p_instability_H1@0.5 | 0.500 | 401 | 0.962 | 0.991 | 0.029 | 0.711 | 0.831 | 116 | 0 |
| p_instability_H3@0.5 | 0.500 | 401 | 0.977 | 0.994 | 0.016 | 0.711 | 0.831 | 116 | 0 |
| p_instability_H5@0.5 | 0.500 | 401 | 0.984 | 0.995 | 0.011 | 0.711 | 0.831 | 116 | 0 |
| p_instability_H1@orig-val-thr | 0.848 | 401 | 0.962 | 0.991 | 0.029 | 0.696 | 0.821 | 116 | 0 |
| p_instability_H3@orig-val-thr | 0.728 | 401 | 0.977 | 0.994 | 0.016 | 0.711 | 0.831 | 116 | 0 |
| p_instability_H5@orig-val-thr | 0.784 | 401 | 0.984 | 0.995 | 0.011 | 0.711 | 0.831 | 116 | 0 |

## Object/window means

| object | window | n | pSlip | H1 | H3 | H5 |
|---|---|---:|---:|---:|---:|---:|
| cucumber | slip | 142 | 0.859 | 0.983 | 0.987 | 0.990 |
| cucumber | stable | 62 | 0.032 | 0.932 | 0.959 | 0.970 |
| hammer | slip | 143 | 1.000 | 1.000 | 1.000 | 1.000 |
| hammer | stable | 54 | 0.571 | 0.997 | 0.999 | 0.999 |

## Sequence-side/window detail for cucumber and hammer

| sequence | side | window | range | n | pSlip | H1 | H3 | H5 |
|---|---|---|---|---:|---:|---:|---:|---:|
| cucumber_1 | left | slip | 35-75 | 41 | 0.838 | 0.975 | 0.982 | 0.986 |
| cucumber_1 | right | slip | 48-73 | 26 | 0.984 | 0.998 | 0.998 | 0.998 |
| cucumber_2 | left | slip | 39-52 | 14 | 0.862 | 0.956 | 0.963 | 0.970 |
| cucumber_2 | right | slip | 38-51 | 14 | 0.996 | 0.995 | 0.995 | 0.996 |
| cucumber_3 | left | slip | 40-62 | 23 | 0.591 | 0.978 | 0.983 | 0.986 |
| cucumber_3 | left | stable | 80-110 | 31 | 0.025 | 0.939 | 0.961 | 0.969 |
| cucumber_3 | right | slip | 41-64 | 24 | 0.936 | 0.995 | 0.997 | 0.998 |
| cucumber_3 | right | stable | 80-110 | 31 | 0.039 | 0.925 | 0.956 | 0.972 |
| hammer_1 | left | slip | 50-80 | 31 | 1.000 | 1.000 | 1.000 | 1.000 |
| hammer_1 | right | slip | 50-80 | 31 | 1.000 | 1.000 | 1.000 | 1.000 |
| hammer_2 | left | slip | 34-44 | 11 | 0.995 | 0.999 | 1.000 | 1.000 |
| hammer_2 | left | stable | 90-116 | 27 | 0.723 | 0.998 | 0.999 | 0.999 |
| hammer_2 | right | slip | 34-41 | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| hammer_2 | right | stable | 90-116 | 27 | 0.419 | 0.996 | 0.998 | 0.999 |
| hammer_3 | left | slip | 40-70 | 31 | 1.000 | 1.000 | 1.000 | 1.000 |
| hammer_3 | right | slip | 40-70 | 31 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

- Stage-I `p_slip_current` still separates real stable/slip windows best: overall F1@0.5 = 0.857. Cucumber+hammer focus F1@0.5 is higher because slip windows are clearer.
- Stage-II future-instability scores have high ranking separation after relabeling (stable mean < slip mean), but raw probabilities remain saturated high on real data. Therefore threshold 0.5 is not usable for stable-window rejection.
- Cucumber/hammer are more appropriate for qualitative future-instability discussion because their slip intervals are longer; however hammer stable windows still receive high H1-H5 scores, showing a remaining real-domain calibration gap.
- For paper reporting, use this as zero-shot real-world evidence: slip detection transfers well; future instability has correct ordering but needs calibration/thresholding before deployment.
