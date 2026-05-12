#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

models=$base/models
configs=$base/configs

mkdir -p $models

num_threads=8
#device=0

# measure time

SECONDS=0

logs=$base/logs

for model_name in deen_transformer_prenorm deen_transformer_postnorm; do

    mkdir -p $logs/$model_name

    OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/$model_name.yaml > $logs/$model_name/out 2> $logs/$model_name/err

done

echo "time taken:"
echo "$SECONDS seconds"
