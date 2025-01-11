#!/bin/bash

echo "Create PCA model from Autocorrelation data"

python3 gPCA.py \
-train_filepath ../../High_Dimension_data/autocorrelation_data/train \
-PCA_components 500 \
-time 80 \
-width 256 \
-height 256 \
-PCA_path PCA_model_PC=50 \


echo "Create training, validation, testing data for Autocorrelation pipeline"

python3 create_PCAdata.py \
-train_filepath ../../High_Dimension_data/autocorrelation_data/train \
-valid_filepath ../../High_Dimension_data/autocorrelation_data/valid \
-test_filepath ../../High_Dimension_data/autocorrelation_data/test \
-time 80 \
-PCA_components 50 \
-PCA_path PCA_model_PC=50 \
-result_path PCA_data_PC=50 \