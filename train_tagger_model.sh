#!/usr/bin/env bash
# Training script for tagger model.
set -o nounset
set -e

# Defined in train_user_model.sh.
user_model_dir="misc/user_model_dir"

tagger_model_dir="misc/tagger_model_dir"

if ! [[ -d ${tagger_model_dir} ]]; then
  mkdir -p ${tagger_model_dir}
fi

pretrain_model_dir=${user_model_dir}
pretrain_model_config="${user_model_dir}/model.config"

python -m src.train_tagger_model \
  -c configs/train_tagger.config  \
  --pretrain_model_dir ${pretrain_model_dir} \
  --pretrain_model_config ${pretrain_model_config} \
  --model_outdir ${tagger_model_dir}
