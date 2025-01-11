#!/bin/bash

echo "Create PCA model for (C-)LCA+PCA+LMAU pipeline"

python3 gLCA_PCA.py \
-train_filepath ../../High_Dimension_data/microstructure_data/train \
-test_filepath ../../High_Dimension_data/microstructure_data/test \
-PCA_components 500 \
-Autoencoder_dir ./LCA_model \
-Autoencoder_name EncoderDecoder.pt.tar \
-device 0 \
-time 80 \
-width 256 \
-height 256 \
-channels 1 \
-latent_width 8 \
-latent_height 8 \
-latent_channel 128 \
-batch_size 10 \
-num_workers 4 \
-PCA_path Autoencoder_PCA_model \
-graph_path Reconstruction \


echo "Create training, validation, testing data for (C-)LCA+PCA+LMAU pipeline"

python3 create_encoderdata.py \
-train_filepath ../../High_Dimension_data/microstructure_data/train \
-valid_filepath ../../High_Dimension_data/microstructure_data/valid \
-test_filepath ../../High_Dimension_data/microstructure_data/test \
-PCA_path Autoencoder_PCA_model \
-PCA_components 500 \
-Autoencoder_dir ./LCA_model \
-Autoencoder_name EncoderDecoder.pt.tar \
-device 0 \
-time 80 \
-width 256 \
-height 256 \
-channels 1 \
-batch_size 10 \
-num_workers 4 \
-result_path C-LCA_PCA_data \