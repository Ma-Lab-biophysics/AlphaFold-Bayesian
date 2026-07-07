#!/usr/bin/env bash
set -euo pipefail
bayes-infer multi-d run \
  --data-dir output_multi_d/multi_d_input \
  --theta 3000 \
  --out-dir output_multi_d/theta_3000
