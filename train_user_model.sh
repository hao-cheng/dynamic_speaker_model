#!/usr/bin/env bash
set -o nounset
set -e

user_model_dir="misc/user_model_dir"

if ! [[ -d ${user_model_dir} ]]; then
  mkdir -p ${user_model_dir}
fi

python -m src.train_user_model \
  -c configs/train_user_model.config  \
  --model_outdir ${user_model_dir}
