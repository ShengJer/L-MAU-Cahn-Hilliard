#!/bin/bash

echo "start Autoencoder model"

python3 run.py \
-train_filepath ./train_data \
-ckpt_path ckpt_path \
-graph_path graph_path \
-load_model 0 \
-time 80 \
-width 256 \
-height 256 \
-channels 1 \
-batch_size 10 \
-num_workers 4 \
-model_name LCA \
-num_epoch 1000 \
-lr 0.001 \
-step_size 50 \
-gamma 0.8 \
-device cuda:0 \
-display_epoch 20 \
-valid_epoch 100 \
