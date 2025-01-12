# Parameters Description
**submit.sh** <br> 
All the parameters are defined in this file.
User can either submit **.sh**  file by terminal or execute the **run.py** program
<br>

**gLCA_PCA.py :** 
### `argparse.ArgumentParser`
- `-train_filepath`: **.npz** file (the original high dimensional microstructure data for **training** )
- `-valid_filepath`: **.npz** file (the original high dimensional microstructure data for **validation**)
- `-test_filepath`: **.npz** file (the original high dimensional microstructure data for **testing**)
- `-PCA_components`: number of principle components been preserved
- `-Autoencoder_dir`: the directory for LCA or C-LCA model
- `-Autoencoder_name`: the filename of the stored parameters of trained LCA or C-LCA 
- `-device`: GPU id
- `-time`: total simulation time of the high dimensional microstructure data
- `-width`: the width of original high dimensional microstructure data
- `-height`: the height of original high dimensional microstructure data
- `-channels`:  must set to **1** for use in pytorch convolution
- `-latent_width`: the width of latent space after (C-)LCA
- `-latent_height`: the height of latent space after (C-)LCA
- `-latent_channel`: the channel of latent space after (C-)LCA
- `-batch_size`: batch size for **torch.utils.data.DataLoader**
- `-num_workers`: num_workers in **torch.utils.data.DataLoader**
- `-PCA_path`: the directory for storing PCA model of (C-)LCA pipeline
- `-graph_path`: Some result of testing PCA reconstruction, PCA+(C-)LCA reconstruction would be stored here

**Create_LCAencoderdata.py :** 
### `argparse.ArgumentParser`
- `-train_filepath`: **.npz** file (the original high dimensional microstructure data for **training** )
- `-valid_filepath`: **.npz** file (the original high dimensional microstructure data for **validation**)
- `-test_filepath`: **.npz** file (the original high dimensional microstructure data for **testing**)
- `-PCA_path`: the directory PCA model of (C-)LCA pipeline been created by the program **gLCA_PCA.py** 
- `-PCA_components`: number of principle components been preserved
- `-Autoencoder_dir`: the directory for LCA or C-LCA model
- `-Autoencoder_name`: the filename of the stored parameters of trained LCA or C-LCA 
- `-device`: **GPU id** for training the model
- `-time`: total simulation time of the high dimensional microstructure data
- `-width`: the width of original high dimensional microstructure data
- `-height`: the height of original high dimensional microstructure data
- `-channels`:  must set to **1** for use in pytorch convolution
- `-batch_size`: batch size for **torch.utils.data.DataLoader**
- `-num_workers`: num_workers in **torch.utils.data.DataLoader**
- `-result_path`: the training, validation, testing data reduced by (C-)LCA+PCA model
