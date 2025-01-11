#!/bin/bash

echo "Create training, validation, testing data for HCA pipeline"

python3 create_encoderdata.py \
-train_filepath ../../High_Dimension_data/microstructure_data/train \
-valid_filepath ../../High_Dimension_data/microstructure_data/valid \
-test_filepath ../../High_Dimension_data/microstructure_data/test \
-Autoencoder_dir ./HCA_model \
-Autoencoder_name EncoderDecoder256x1x1.pt.tar \
-device 0 \
-time 80 \
-width 256 \
-height 256 \
-channels 1 \
-latent_width 1 \
-latent_height 1 \
-latent_channel 256 \
-batch_size 10 \
-num_workers 4 \
-result_path HCA_data \