#!/bin/bash

LOG_DATE=""

# echo "[INFO] Removing previous log directory: $LOG_DATE"
# rm -rf $LOG_DATE

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

PYTHON_PATH=~/.conda/envs/python312/bin/python
export RUN_DIR=$LOG_DATE

CHECKPOINT=""

export DATA_ROOT=""
GPU=5
nohup $PYTHON_PATH test_wholebrain.py \
  --gpu $GPU \
  --trained_checkpoints $CHECKPOINT \
  > /dev/null 2>&1 &
sleep 30




unset DATA_ROOT
unset RUN_DIR


echo "[INFO] Tail logs in: $LOG_DATE"
find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +
