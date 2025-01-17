#!/bin/bash

set -e  # Exit on any command failure


# Define reusable variables
DATA_DIR=../../High_Dimension_data/microstructure_data
AUTOENCODER_DIR=./LCA_model
AUTOENCODER_NAME=EncoderDecoder.pt.tar
PCA_PATH=Autoencoder_PCA_model
RESULT_PATH=C-LCA_PCA_data
DEVICE=1

echo "$(date): Create PCA model for (C-)LCA+PCA+LMAU pipeline"

python3 gLCA_PCA.py \
-train_filepath $DATA_DIR/train \
-test_filepath $DATA_DIR/test \
-PCA_components 300 \
-Autoencoder_dir $AUTOENCODER_DIR \
-Autoencoder_name $AUTOENCODER_NAME \
-device cuda:$DEVICE \
-time 80 \
-width 256 \
-height 256 \
-channels 1 \
-latent_width 8 \
-latent_height 8 \
-latent_channel 128 \
-batch_size 10 \
-num_workers 2 \
-PCA_path $PCA_PATH \
-graph_path Reconstruction

echo "$(date): Create training, validation, testing data for (C-)LCA+PCA+LMAU pipeline"

python3 create_LCAencoderdata.py \
-train_filepath $DATA_DIR/train \
-valid_filepath $DATA_DIR/valid \
-test_filepath $DATA_DIR/test \
-PCA_path $PCA_PATH \
-PCA_components 300 \
-Autoencoder_dir $AUTOENCODER_DIR \
-Autoencoder_name $AUTOENCODER_NAME \
-device cuda:$DEVICE \
-time 80 \
-width 256 \
-height 256 \
-channels 1 \
-batch_size 10 \
-num_workers 2 \
-result_path $RESULT_PATH

echo "$(date): Script completed successfully!"
