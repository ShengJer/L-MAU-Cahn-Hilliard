```
project: Autocorrelation_pipeline
│   run.py
│   submit.sh
│   main_body.py
│   trainer.py  
└───model 
│   │   LMAU.py
│   │   model_factory.py
│   └───layers
│       │   LMAUcell.py   
└───data_provider
│   │   datasets_factory.py
│   │   phase_field.py
└───data
|   └───PCA_data
│       │   (put the data reduced from PCA here, including train, validation and testing data)
|   └───PCA_model
│       │   (put the PCA model here)
```

# Parameters Description
**main_body.py :** <br> 
This file is the main program

User can either submit **.sh**  file or execute the **run.py** program in their IDE by setting parameters in submit.sh file.
<br>

#### `argparse.ArgumentParser`

Parameters in the program : <br>

- `-train_data_paths`: **.npz** file (the **training** dataset already been reduced by PCA)
- `-valid_data_paths`: **.npz** file (the **validation** dataset already been reduced by PCA)
- `-test_data_paths`: **.npz** file (the **testing** dataset already been reduced by PCA)
- `-PCA_dir`: directory of PCA model
- `-PCA_name`: the name of PCA in **.pkl** file 
- `-gen_frm_dir`: name of output directory that would be created for training result (**is_training** == 1)
- `-test_frm_dir`: name of output directory that would be created for testing result (**is_training** == 0)
- `-save_dir`: name of output directory that would be created for storing checkpoint model
- `-Graph_dir`: output the training loss and validation loss during training stage 
- `-dataset_name`: dataset provider  default : **data_provider/phase_field.py**
- `-save_modelname`: name of checkpoint model
- `-batch_size`: the batch size of training data
- `-in_features`: the reduced features of low dimensional data (i.e. 50, 200, 300 ... smaller than original high dimension)
- `-img_width`: the width of original high dimensional data
- `-img_height`: the height of original high dimensional data
- `-img_channel`: must set to **1** for use in pytorch convolution
- `-total_length`: total time length of data **(total_length = input_length + output_length)**
- `-input_length`: time length for initialized the recurrent-type model 
- `-output_length`: time length for prediction from the model
- `-display_interval`: how many training iteration does the information print on the terminal
- `-max_iterations`: the maximum iterations in training stage
- `-plt_num_PCs`: number of principle components been plot at validation stage to compare with ground truth
- `-model_name`: set to **lmau**
- `-num_layers`: number of layers in the model
- `-num_hidden`: number of hidden size of linear layers in the model
- `-tau`: length of previous history been considered in L-MAU model
- `-cell_mode`: **normal** or **residual** model. If **residual** is activated, residual mechanism is switched on in the L-MAU unit
- `-model_mode`: **normal** or **recall** mode. If **recall** model is activated, the final output would be the combination of last layer's output and input flow
- `-lr`: initial learning rate for adam optimizer 
- `-lr_decay`: parameter gamma in **torch.optim.lr_scheduler.StepLR**
- `-step_size`: parameter **step_size** in **torch.optim.lr_scheduler.StepLR**
- `-loss_type`: **L1+L2** or **L1** or **L2** // **L1=torch.nn.L1Loss()**, **L2=torch.nn.MSELoss()**
- `-test_interval`: every number of iterations for doing validation
- `-num_save_samples`: save the first **num_save_samples** batch result in validation
- `-is_training`: **1** for training stage, **0** for testing stage
- `-load_model`: **1** to load previous saved checkpoint model into training stage
- `-device`: **GPU id** for training the model
- `-scheduled_sampling`: **1** or **0** // **1** to apply scheduled sampling inside training stage
- `-sampling_stop_iter`: end iteration in scheduled sampling
- `-sampling_start_value`: startinig probability in scheduled sampling technique 
- `-sampling_changing_rate`: probability decaying with sampling changing_rate

# Training, Validation Stage
## Reminder
* `-is_training` should be set to **1**
* `-load_model` would be **0** for completely new training but **1** for training from previous checkpoint. In this case,  `-save_modelname` should be pointed to previous stored model.
* the `-batch_size` cannot be specified larger than total number of validation samples or the validation stage would cause error.
## output
* Three directories would be created during this stage from `-save_dir`, `-Graph_dir`, and `-gen_frm_dir` <br>
    * `-save_dir` is the directory for saving checkpoint model by early stopping.
    * `-Graph_dir` is the directory for saving history of training and validation loss.
    * `-gen_frm_dir` is the directory for saving validation result at each `-test_interval` iteration. each folder have `-num_save_samples` subfolders which is the first few batch results in validation. Inside each subfolder, the prediction and ground truth result of first `-plt_num_PCs` is compared. Also, the reconstructed autocorrelation is compared with the ground truth one in two different time frame as well.

# Testing Stage
## Reminder
* `-is_training` should be set to **0**
* If `-is_training` set to **0**, `-load_model` will not function anymore since testing stage should always read the checkpoint model.
* `-save_modelname` must be pointed to previous stored model.
* the `-batch_size` should be set to **1** for applying trained model on every testing samples.
## output
* One directory would be created during this stage from `-test_frm_dir`