# Phase7 Step4 Force Per-Axis / Physical Decomposition

| condition | Fx | Fy | Fz | Fn | Ft | Fmag | mean | slip F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| separate_baseline | 0.0369 | 0.0345 | 0.0253 | 0.0253 | 0.0380 | 0.0346 | 0.0322 | 0.9731 |
| naive_shared_multitask | 0.0402 | 0.0333 | 0.0357 | 0.0357 | 0.0359 | 0.0432 | 0.0364 | 0.9784 |
| partially_shared_lambda_0.25 | 0.0331 | 0.0323 | 0.0278 | 0.0278 | 0.0365 | 0.0378 | 0.0311 | 0.9771 |
| consistency_decoder | 0.0346 | 0.0324 | 0.0336 | 0.0336 | 0.0335 | 0.0405 | 0.0335 | 0.9819 |
| decoupled_multitask | 0.0293 | 0.0324 | 0.0295 | 0.0295 | 0.0363 | 0.0397 | 0.0304 | 0.9652 |
| joint_lightweight_force_slip_future | 0.0707 | 0.0828 | 0.0665 | 0.0665 | n/a | n/a | 0.0734 | 0.9816 |
