#!/usr/bin/env bash
set -euo pipefail

revision="e1f85a3339848b9eb606ce543fcaaba3f4be1893"
base="https://huggingface.co/datasets/Cainiao-AI/LaDe-D/resolve/${revision}/data"
destination="data/raw/lade_d"
mkdir -p "${destination}"

files=(
  delivery_cq-00000-of-00001-465887add76aeabc.parquet
  delivery_hz-00000-of-00001-8090c86f64781f71.parquet
  delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet
  delivery_sh-00000-of-00001-ad9a4b1d79823540.parquet
  delivery_yt-00000-of-00001-cc85c1fcb1d10955.parquet
)

for file in "${files[@]}"; do
  curl -L --fail --show-error --continue-at - --output "${destination}/${file}" "${base}/${file}"
done
