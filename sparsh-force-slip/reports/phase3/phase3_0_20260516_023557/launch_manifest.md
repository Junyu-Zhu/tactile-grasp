# Phase3-0 launch manifest

{
  "stamp": "20260516_023557",
  "phase": "P3-0 sanity + DINO ABC launch",
  "run_id_forceonly": "phase3_0_forceonly_gsmini_20260516_023557",
  "dino_a_prefix": "phase3_0_dino_a_gsmini_20260516_023557",
  "dino_b_run_ids": {
    "0.25": "phase3_0_dino_b_lam025_gsmini_20260516_023557",
    "0.50": "phase3_0_dino_b_lam050_gsmini_20260516_023557"
  },
  "derived_dataset": "/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini",
  "runbook_dir": "/home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/phase3_0_20260516_023557",
  "log_dir": "/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase3/phase3_0_20260516_023557",
  "queues": {
    "gpu0": [
      "forceonly_mae",
      "dino_a_force"
    ],
    "gpu1": [
      "forceonly_dinov2",
      "dino_a_slip"
    ],
    "gpu2": [
      "forceonly_ijepa",
      "dino_b_lambda025"
    ],
    "gpu3": [
      "forceonly_vjepa",
      "dino_b_lambda050",
      "forceonly_dino"
    ]
  },
  "single_process_single_gpu": true,
  "wandb_mode": "online",
  "auto_push_pull": false
}
