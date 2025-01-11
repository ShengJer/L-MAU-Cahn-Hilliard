# Parameters Description
**submit.sh** <br> 
All the parameters are defined in this file.
User can either submit **.sh**  file by terminal or execute the **run.py** program
<br>

**gPCA.py :** 
### `argparse.ArgumentParser`
- `-train_filepath`: **.npz** file (the original high dimensional autocorrelation data)
- `-PCA_components`: number of principle components been preserved from autocorrelation data
- `-time`: total simulation time of the high dimensional autocorrelation data
- `-width`: the width of original high dimensional autocorrelation data
- `-height`: the height of original high dimensional autocorrelation data
- `-PCA_path`: the directory for storing PCA model of Autocorrelation pipeline


**Create_PCAdata.py :** 
### `argparse.ArgumentParser`
- `-train_filepath`: **.npz** file (the original high dimensional autocorrelation data for **training** )
- `-valid_filepath`: **.npz** file (the original high dimensional autocorrelation data for **validation**)
- `-test_filepath`: **.npz** file (the original high dimensional autocorrelation data for **testing**)
- `-time`: total simulation time of the high dimensional autocorrelation data
- `-PCA_components`: number of principle components been preserved from autocorrelation data
- `-PCA_path`: the PCA model created by **gPCA.py** program 
- `-result_path`: the training, validation, testing data reduced by PCA model
