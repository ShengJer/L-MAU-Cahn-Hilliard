#!/bin/bash


set -e  # Exit on any command failure


DEVICE=1
# if load_model=0, save_modelname would not be activated. save_modelname would be activated only when is_training=0

echo "$(date): start HCA+LMAU pipeline !! "

python3 main_body.py \
-train_data_paths ./data/encoder_data/train_data_256.npz \
-valid_data_paths ./data/encoder_data/valid_data_256.npz \
-test_data_paths ./data/encoder_data/test_data_256.npz \
-test_ms_data_paths ../../High_Dimension_data/microstructure_data/test \
-gen_frm_dir results \
-test_frm_dir test_results \
-save_dir checkpoints \
-cplot_dir cplot \
-Graph_dir Graph \
-dataset_name phase_field \
-save_modelname model.pt.tar-36000 \
-Autoencoder_dir ./data/HCA_model \
-save_Autoencodername EncoderDecoder256x1x1.pt.tar \
-batch_size 10 \
-in_features 256 \
-latent_width 1 \
-latent_height 1 \
-latent_channel 256 \
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
-num_hidden 256 \
-tau 20 \
-cell_mode residual \
-model_mode recall \
-lr 0.001 \
-lr_decay 0.85 \
-step_size 3000 \
-loss_type L1+L2 \
-test_interval 4000 \
-num_save_samples 3 \
-is_training 1 \
-load_model 0 \
-device cuda:$DEVICE \
-scheduled_sampling 1 \
-sampling_stop_iter 50000 \
-sampling_start_value 1.0 \
-sampling_changing_rate 0.00002 \
