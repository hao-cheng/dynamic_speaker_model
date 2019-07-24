#!/usr/bin/env bash
# Processing script for Switchboard dialog act data.

set -o nounset
set -e

# Defines the output base directory.
outbase_dir="data"

if ! [[ -d ${outbase_dir} ]]; then
  mkdir ${outbase_dir}
fi

if [[ -d "./data_script/py_data_lib" ]]; then
  echo "Removes the old py_data_lib under folder data_script"
  rm -rf "./data_script/py_data_lib"
fi

echo "Copies py_data_lib"
cp -r "./src/model/py_data_lib" "./data_script/"

# Please make sure all required data is properly specified in process.config.
python -m data_script.process_predictor_data \
  -c configs/process_swda.config \
  --out_basedir  ${outbase_dir}
