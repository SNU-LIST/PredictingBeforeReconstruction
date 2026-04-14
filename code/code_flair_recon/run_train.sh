#!/bin/bash

MODE=$1  # base | t1prior | t2prior | t1pred | t2pred | t1t2pred

echo "========================================"
echo "[INFO] Script started at: $(date)"
echo "========================================"
sleep 1

if [ -z "$MODE" ]; then
  echo "[ERROR] Please provide a mode: base | t1prior | t2prior | t1pred | t2pred | t1t2pred"
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
PARALLEL_FACTOR=12

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
    echo "[MODE] Running NO PRIOR"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key none \
      --target_key fl \
      --target_mask_key fl_mask \
      > /dev/null 2>&1 &
    ;;

  t1prior)
    export DATASET_MODE="t1prior"
    echo "[MODE] Running T1 PRIOR"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key t1_reg \
      --target_key fl \
      --target_mask_key fl_mask \
      > /dev/null 2>&1 &
    ;;

  t2prior)
    export DATASET_MODE="t2prior"
    echo "[MODE] Running T2 PRIOR"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key t2_reg \
      --target_key fl \
      --target_mask_key fl_mask \
      > /dev/null 2>&1 &
    ;;

    
  t1pred)
    export DATASET_MODE="t1pred"
    echo "[MODE] Running T1 PREDICTION"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key t1pred \
      --target_key fl \
      --target_mask_key fl_mask \
      > /dev/null 2>&1 &
    ;;


  t2pred)
    export DATASET_MODE="t2pred"
    echo "[MODE] Running T2 PREDICTION"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key t2pred \
      --target_key fl \
      --target_mask_key fl_mask \
      > /dev/null 2>&1 &
    ;;

  t1t2pred)
    export DATASET_MODE="t1t2pred"
    echo "[MODE] Running T1T2 PREDICTION"
    nohup $PYTHON_PATH train.py \
      $COMMON_ARGS \
      --prior_key t1t2pred \
      --target_key fl \
      --target_mask_key fl_mask \
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
