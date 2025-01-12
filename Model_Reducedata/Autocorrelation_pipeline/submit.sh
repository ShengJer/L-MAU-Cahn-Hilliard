#!/bin/bash
#PBS -N Autocorrelation_data_model
#PBS -l select=1:vnode=cftlab-c1:ncpus=1
#PBS -q workq
#PBS -o Autocorrelation_data_model.out
#PBS -e Autocorrelation_data_model.err

set -e  # Exit script on any command failure

DATA_DIR=../../High_Dimension_data/autocorrelation_data # the directory of high dimensional autocorrelation data
PCA_PATH=PCA_model_PC=50 # the directory for storing PCA model in autocorrelation pipeline
RESULT_PATH=PCA_data_PC=50 # the directory for storing reduced data

echo "$(date): Create PCA model from Autocorrelation data"

python3 gPCA.py \
-train_filepath $DATA_DIR/train \
-PCA_components 50 \
-time 80 \
-width 256 \
-height 256 \
-PCA_path $PCA_PATH

echo "$(date): Create training, validation, testing data for Autocorrelation pipeline"

python3 create_PCAdata.py \
-train_filepath $DATA_DIR/train \
-valid_filepath $DATA_DIR/valid \
-test_filepath $DATA_DIR/test \
-time 80 \
-PCA_components 50 \
-PCA_path $PCA_PATH \
-result_path $RESULT_PATH
