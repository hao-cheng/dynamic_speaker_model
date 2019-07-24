#!/usr/bin/env bash
set -o nounset
set -e

# Defined in train_user_model.sh
user_model_dir="misc/user_model_dir"

eval_user_model_dir="misc/eval_user_model_dir"

if ! [[ -d ${eval_user_model_dir} ]]; then
  mkdir -p ${eval_user_model_dir}
fi

# If is_infer==0, there is no output.
python -m src.eval_user_model \
  -c configs/eval_user_model.config  \
  --model_indir ${user_model_dir} \
  --is_infer 0 \
  --outbase "${eval_user_model_dir}/infer_output"
