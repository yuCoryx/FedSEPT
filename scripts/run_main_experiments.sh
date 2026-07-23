#!/usr/bin/env bash
set -euo pipefail

group="${1:-}"
if [[ "$group" != "pathological" && "$group" != "practical" && "$group" != "domain" ]]; then
  echo "Usage: $0 {pathological|practical|domain}" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data_root="${DATA_ROOT:-$project_root/data}"
device="${DEVICE:-cuda:0}"

trainers=(promptfl fedpgp fedotp fedpha pfedmoap dpfpl fedsept)
datasets=()
betas=(0.0)
num_users=10
fraction=1.0

case "$group" in
  pathological)
    datasets=(food101 caltech101 oxford_flowers dtd oxford_pets)
    ;;
  practical)
    datasets=(cifar10 cifar100)
    betas=(0.1 0.3 0.5)
    num_users=100
    fraction=0.1
    ;;
  domain)
    datasets=(pacs office31 officehome domainnet)
    betas=(0.1 0.3 0.5)
    ;;
esac

cd "$project_root"
for dataset in "${datasets[@]}"; do
  for beta in "${betas[@]}"; do
    for trainer in "${trainers[@]}"; do
      "$python_bin" federated_main.py \
        --trainer "$trainer" \
        --dataset "$dataset" \
        --root "$data_root" \
        --device "$device" \
        --num_users "$num_users" \
        --frac "$fraction" \
        --beta "$beta" \
        --round 50 \
        --epoch 1 \
        --train_batch_size 32 \
        --test_batch_size 128 \
        --lr 0.001 \
        --optimizer sgd \
        --num_shots 16 \
        --dp_mode local \
        --dp_epsilon 1.0 \
        --dp_delta 1e-5 \
        --dp_clip 1.0 \
        --dp_microbatch_size 8
    done
  done
done
