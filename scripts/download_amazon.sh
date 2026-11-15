#!/usr/bin/env bash
set -euo pipefail

bucket="s3://amazon-last-mile-challenges/almrrc2021/"
destination="data/raw/amazon/"

if [[ "${1:-}" == "--dry-run" ]]; then
  aws s3 ls --no-sign-request "${bucket}" --recursive
  exit 0
fi

command -v aws >/dev/null 2>&1 || {
  echo "AWS CLI is required. Install it, then rerun this script." >&2
  exit 1
}

mkdir -p "${destination}"
aws s3 sync --no-sign-request "${bucket}" "${destination}"
