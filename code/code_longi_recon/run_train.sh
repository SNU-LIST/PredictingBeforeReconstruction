#!/bin/bash

MODE=$1  # base | pred

echo "========================================"
echo "[INFO] Script started at: $(date)"
echo "========================================"
sleep 1

if [ -z "$MODE" ]; then
  echo "[ERROR] Please provide a mode: base | pred "
  exit 1
fi
sleep 1

LOG_DATE=""
PYTHON_PATH=~/.conda/envs/python312/bin/python
  
echo "[INFO] Python path: $PYTHON_PATH"
echo "[INFO] Log directory: $LOG_DATE"
sleep 1

echo "[INFO] Removing nohup.out"
rm -rf nohup.out
sleep 1

export DATA_ROOT=""
export RUN_DIR=$LOG_DATE

GPU="0,1,2,3,4,5,6,7"

TRAIN_BATCH=32

ACS_NUM=24
PARALLEL_FACTOR=8

echo "[INFO] Environment setup:"
echo "  DATA_ROOT       : $DATA_ROOT"
echo "  RUN_DIR         : $RUN_DIR"
echo "  GPU             : $GPU"
echo "  TRAIN_BATCH     : $TRAIN_BATCH"
echo "  ACS_NUM         : $ACS_NUM"
echo "  PARALLEL_FACTOR : $PARALLEL_FACTOR"
sleep 1

COMMON_ARGS="\
  --debugmode False \
  --gpu $GPU \
  --acs_num $ACS_NUM \
  --parallel_factor $PARALLEL_FACTOR \
  --valid_batch $TRAIN_BATCH \
  --train_batch $TRAIN_BATCH"

case $MODE in
  base)
    export DATASET_MODE="base"
    echo "[MODE] Running Visit1 PRIOR"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key visit1 \
      --target_key visit2 \
      --target_mask_key visit2_mask \
      > /dev/null 2>&1 &
    ;;

  pred)
    export DATASET_MODE="pred"
    echo "[MODE] Running Prediction PRIOR"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key pred \
      --target_key img \
      --target_mask_key mask \
      > /dev/null 2>&1 &
    ;;
  *)
    echo "[ERROR] Invalid mode: $MODE"
    exit 1
    ;;
esac

# Clean up
unset DATA_ROOT
unset TRAIN_ITER
unset RUN_DIR
unset DATASET_MODE

# Log follow-up
sleep 20
echo "[INFO] Tail logs in: $LOG_DATE"
find "$LOG_DATE" -type f -name "*.log" -exec tail -n 100 -f {} +
