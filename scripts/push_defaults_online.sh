#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 VERSION [CCSNE_MODEL] [BLIP_MODEL]" >&2
  exit 2
fi

version=$1
repo_root=$(git rev-parse --show-toplevel)
ccsne_model=${2:-"$repo_root/out_ccsne/model.h5"}
blip_model=${3:-"$repo_root/out_blip/model.h5"}

for model in "$ccsne_model" "$blip_model"; do
  if [[ ! -f "$model" ]]; then
    echo "missing trained model: $model" >&2
    exit 1
  fi
done

clone_dir=$(mktemp -d /tmp/starccato-data-weights.XXXXXX)
trap 'rm -rf "$clone_dir"' EXIT
git clone git@github.com:starccato/data.git "$clone_dir"

destination="$clone_dir/weights/starcatto_jax/$version"
if [[ -e "$destination" ]]; then
  echo "refusing to overwrite immutable weight version: $version" >&2
  exit 1
fi
mkdir -p "$destination"
cp "$ccsne_model" "$destination/ccsne_vae.h5"
cp "$blip_model" "$destination/blip_vae.h5"

git -C "$clone_dir" add \
  "weights/starcatto_jax/$version/ccsne_vae.h5" \
  "weights/starcatto_jax/$version/blip_vae.h5"
git -C "$clone_dir" commit -m \
  "feat: add starccato_jax $version VAE weights"
git -C "$clone_dir" push origin main
