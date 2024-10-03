#!/bin/bash
DATASET=$1
ADAPTER=$2
CONFIG=$3

if [ -z $3 ]
then
      echo Using default configs
      CONFIG=${ADAPTER}_${DATASET}_default
else
      echo Using configs from file $CONFIG
      CONFIG=$3
fi

python main.py \
      -acfg configs/adapter/$ADAPTER/$DATASET/$CONFIG.yaml \
      -dcfg configs/dataset/${DATASET}_recur=20.yaml \
      OUTPUT_DIR outputs/${ADAPTER}/${DATASET}_recur=20