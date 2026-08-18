# Results Audit

This audit records what can be cited publicly from the repository today. It does not invent success rates or claim model superiority.

## Public-Ready Results

| Result | Source file | Experiment setting | Model | Dataset | Metric | Suitable for README |
| --- | --- | --- | --- | --- | --- | --- |
| Real robot pipeline exists | Collector, training wrappers, deployment wrappers | ARX5 X5 real-robot workflow | DP / CFG / ACT branches | Local ARX5 datasets | Code path presence | Yes, as implementation status |
| Dataset alignment tooling exists | `check_dataset_alignment.py`, `inspect_training_dataset.py`, `analyze_recordings.py` | Offline dataset checks | N/A | Local ARX5 datasets | Shape/timestamp/video checks | Yes, as tooling |
| Checkpoint inspection tooling exists | `load_arx5_ckpt.py`, ACT/CFG inspectors | Offline checkpoint loading | DP / CFG / ACT | Local checkpoints | Shape/import sanity | Yes, as tooling |

## Results Not Yet Public-Ready

| Result type | Why not ready | Needed before README table |
| --- | --- | --- |
| Success rate | No curated, consistent evaluation table was found. | Define task splits, number of trials, object poses, and failure criteria. |
| DP vs ACT comparison | Experiments exist, but settings are not consolidated into a fair comparison table. | Same dataset, same test poses, same camera order, same robot reset protocol. |
| DP-EEF vs DP-Joint | Qualitative notes exist, but no standardized metric file was found. | Run matched trials and record success/failure. |
| DP-CFG smoothness | Timing diagnostics exist, but no final metric table was found. | Log boundary jumps, latency, tracking error, and success rate. |
| Training curves | W&B/offline logs exist locally, but are not curated public figures. | Export selected curves with run metadata and dataset names. |
| Inference latency | Deployment logs print latency fields, but no summarized report was found. | Aggregate JSONL logs into mean/median/p95. |

## README Guidance

The README should use a `Current Status` checklist instead of a numeric results table until the evaluation records are curated.

