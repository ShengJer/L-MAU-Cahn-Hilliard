# Parameters Description
**submit.sh** <br> 
All the parameters are defined in this file.
User can either submit **.sh**  file by terminal or execute the **run.py** program
<br>

**Create_HCAencoderdata.py :** 
### `argparse.ArgumentParser`
- `-train_filepath`: **.npz** file (the original high dimensional microstructure data for **training** )
- `-valid_filepath`: **.npz** file (the original high dimensional microstructure data for **validation**)
- `-test_filepath`: **.npz** file (the original high dimensional microstructure data for **testing**)
- `-Autoencoder_dir`: the directory for HCA
- `-Autoencoder_name`: the filename of the stored parameters of tained HCA model
- `-device`: **GPU id** for training the model
- `-time`: total simulation time of the high dimensional microstructure data
- `-width`: the width of original high dimensional microstructure data
- `-height`: the height of original high dimensional microstructure data
- `-channels`:  must set to **1** for use in pytorch convolution
- `-latent_width`: the width of latent space after HCA
- `-latent_height`: the height of latent space after HCA
- `-latent_channel`: the channel of latent space after HCA
- `-batch_size`: batch size for **torch.utils.data.DataLoader**
- `-num_workers`: num_workers in **torch.utils.data.DataLoader**
- `-result_path`: the training, validation, testing data reduced by HCA model