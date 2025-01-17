#!/bin/bash

set -e  # Exit on any command failure

DEVICE=0

echo "$(date): start Autocorrelation+LMAU pipeline !! "


python3 main_body.py \
-train_data_paths ./data/PCA_data/PCA_train_data.npz \
-valid_data_paths ./data/PCA_data/PCA_valid_data.npz \
-test_data_paths ./data/PCA_data/PCA_test_data.npz \
-PCA_dir ./data/PCA_model \
-PCA_name pca_50.pkl \
-gen_frm_dir results \
-test_frm_dir test_results \
-save_dir checkpoints \
-cplot_dir cplot \
-Graph_dir Graph \
-dataset_name phase_field \
-save_modelname model.pt.tar-58000 \
-batch_size 10 \
-in_features 50 \
-img_width 256 \
-img_height 256 \
-img_channel 1 \
-total_length 80 \
-input_length 10 \
-output_length 70 \
-display_interval 100 \
-max_iterations 80000 \
-plt_num_PCs 20 \
-model_name lmau \
-num_layers 4 \
-num_hidden 128 \
-tau 40 \
-cell_mode residual \
-model_mode recall \
-lr 0.001 \
-lr_decay 0.85 \
-step_size 3000 \
-test_interval 4000 \
-num_save_samples 5 \
-is_training 1 \
-load_model 0 \
-device cuda:$DEVICE \
-scheduled_sampling 1 \
-sampling_stop_iter 50000 \
-sampling_start_value 1.0 \
-sampling_changing_rate 0.00002 \
