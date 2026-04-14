#!/bin/bash

LOG_DATE=""

# echo "[INFO] Removing previous log directory: $LOG_DATE"
# rm -rf $LOG_DATE

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

PYTHON_PATH=~/.conda/envs/python312/bin/python

export DATA_ROOT=""
export RUN_DIR=$LOG_DATE
GPU="0,1,2,3,4,5,6,7"
TRAIN_BATCH=32
nohup $PYTHON_PATH train.py \
  --gpu $GPU \
  --train_batch $TRAIN_BATCH \
  --valid_batch $TRAIN_BATCH \
  --recon_net_chan 64 \
  --valid_tol 50 \
  > /dev/null 2>&1 &


unset DATA_ROOT
unset RUN_DIR

sleep 20
echo "[INFO] Tail logs in: $LOG_DATE"
find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +

