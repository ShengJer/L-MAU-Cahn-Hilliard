# Parameters Description
**submit.sh** <br> 
All the parameters are defined in this file.
User can either submit **.sh**  file by terminal or execute the **run.py** program
<br>

**main_body.py :** 
### `argparse.ArgumentParser`
- `-train_filepath`: **.npz** file (the original high dimensional microstructure data for **training** )
- `-ckpt_path`: the directory for storing trained autoencoder data such as HCA, LCA or C-LCA 
- `-graph_path`: the directory for storing training and validation loss history
- `-load_model`: **1** to load previous saved checkpoint model into training stage
- `-time`: total simulation time of the high dimensional microstructure data
- `-width`: the width of original high dimensional microstructure data
- `-height`: the height of original high dimensional microstructure data
- `-channels`:  must set to **1** for use in pytorch convolution
- `-batch_size`: batch size for **torch.utils.data.DataLoader**
- `-num_workers`: num_workers in **torch.utils.data.DataLoader**
- `-model_name`: **LCA**, **HCA**, or **C-LCA** for different autoencoder model using in different pipelines.
- `-num_epoch`: the number of epoch for training the autoencoder model
- `-step_size`: parameter **step_size** in **torch.optim.lr_scheduler.StepLR**
- `-gamma`: parameter **gamma** in **torch.optim.lr_scheduler.StepLR**
- `-alpha`: If **C-LCA** been used, it is the weighting parameters in front of the mse loss for mass conservation.
- `-device`: **GPU id** for training the model
- `-display_epoch`: the training info would be displayed every # of epoch
- `-valid_epoch`: the validation would be done every # of epoch

<br>

**model_autoencoder.py** :<br>
the file define the model of LCA and HCA

