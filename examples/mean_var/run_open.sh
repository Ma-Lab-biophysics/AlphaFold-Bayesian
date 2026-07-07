#!/usr/bin/env bash
set -euo pipefail
bayes-infer mean-var run open.input \
  -o output_open.dat \
  --theta-log theta_open.log
