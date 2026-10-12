#!/usr/bin/env bash
# Convenience script: behaviour-cloning warm start → PPO fine-tune, stage 0.
#
# Usage (from the project root):
#   ./training/train_bc.sh                    # default 80 PPO iterations
#   ./training/train_bc.sh --iterations 200   # more RL after cloning
#   ./training/train_bc.sh --stage 1 --init models/policy_stage0.json
#
# The BC warm start collects 30 000 expert transitions from the rule-based
# controller, pre-trains the actor with cross-entropy, then hands control to
# PPO.  The policy that beats the heuristic baseline is published to models/.

set -euo pipefail
cd "$(dirname "$0")/.."

python3 training/run_training.py \
    --stage 0 \
    --bc 1 \
    --bcSteps 30000 \
    --bcEpochs 12 \
    --bcLr 1e-3 \
    --iterations 80 \
    "$@"
