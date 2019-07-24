#!/usr/bin/env bash
# Evaluation script for tagger model.
set -o nounset
set -e

# Defined in train_tagger_model.sh
tagger_model_dir="./pretrained_model"
eval_tagger_model_dir="misc/eval_tagger_model_dir"

if ! [[ -d ${eval_tagger_model_dir} ]]; then
  mkdir -p ${eval_tagger_model_dir}
fi

python -m src.eval_tagger_model \
  -c configs/eval_tagger.config  \
  --model_indir ${tagger_model_dir} \
  --outbase "${eval_tagger_model_dir}/infer_output"
