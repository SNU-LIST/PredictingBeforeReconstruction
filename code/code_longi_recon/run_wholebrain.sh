#!/bin/bash

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

PYTHON_PATH=~/.conda/envs/python312/bin/python

export LOG_DATE=""
export RUN_DIR=$LOG_DATE


export DATA_ROOT=""
GPU=0
TRAINED_CHECKPOINTS=""
nohup $PYTHON_PATH test_wholebrain.py \
  --gpu $GPU \
  --trained_checkpoints $TRAINED_CHECKPOINTS \
  > /dev/null 2>&1 &
sleep 90



find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +


