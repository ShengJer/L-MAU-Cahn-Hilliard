#  MPI Cahn-Hilliard solver and L-MAU model

The source code of the work : <br>
[Chen SJ, Yu HY. L-MAU: A multivariate time-series network for predicting the Cahn-Hilliard microstructure evolutions via low-dimensional approaches. Computer Physics Communications. 2024 Dec 1;305:109342.](https://www.sciencedirect.com/science/article/pii/S0010465524002650)

## MPI_CH_solver folder
This code is modified from Ref.[1], a semi-implicit spectral method, to include simple parallelism with MPI.
The Cahn-Hilliard model includes Flory-Huggins free energy and fast mobility model [2].
1. Requirement
   * mpicc (OpenMPI or other MPI library)
   * FFTW3
     * please ensure FFTW3 MPI libraries and header files are successfully configured and compiled
     * https://www.fftw.org/
   * gnu/gsl https://www.gnu.org/software/gsl/ <br>

2. Compiling code
   * Please follow the makefile comment to change the vpath directory of gsl, fftw3, and mpi based on the machine
   * After successful compilation, two object files: VariableCH.o evolution.o and one executable file CH_variable_mobility.out would be created <br>
   
3. Run 
```
mpirun -np <number_of_processors> CH_variable_mobility.out

``` 
Examples: <br>
![CH_animation!](CH_evol/CH_animation.gif)
# Machine Learning Pipeline

## A. &nbsp; Data preparation
1. &nbsp; Please generate microstructure and autocorrelation data by the MPI_CH_solver. The data should be stored in numpy format and be read by `np.load(filename)['data']`. The shape of the stored microstructure array and autocorrelation is of shape (TL, 1, Ny, Nx) and (TL, Ny, Nx), respectively, where TL is the total simulation time steps, and Nx, Ny are the 2D spatial dimension. <br>

2. &nbsp; Separate all data into three datasets: training, validation, and testing, and place them into the corresponding folder in **./High_Dimension_data**. An example data format is placed in **./High_Dimension_data/autocorrelation_data/train**, and **./High_Dimension_data/microstructure_data/train**.


## B. &nbsp; PCA, Autocoencoder model, data creation

### I. &nbsp; Autocorrelation pipeline
1.  &nbsp; In **./Model_Reducedata/Autocorrelation_pipeline**, either run the script file **submit.sh** by terminal or execute **run.py** file in python IDE.
(All the parameters specified by user are listed in submit.sh)

2. &nbsp; Two directories would be created with the name from parameters `-PCA_path` and `-result_path`. <br>
   2.1 PCA model from autocorrelation function will be stored in `-PCA_path` with two other files named **PCA_variance.png** and **cum_var.npy**. The first file is the cumulative explained variance to evaluate the quality of PCA model, and the second file is the variance data been stored.
    
   2.2 The low dimensional data been reduced from PCA is stored in `-result_path` with name **PCA_train_data.npz**, **PCA_valid_data.npz**, and **PCA_test_data.npz**

### II. &nbsp; (C-)LCA+PCA pipeline
1. &nbsp; Train LCA or C-LCA model in **./Autoencoder** by submitting the script file **submit.sh** in terminal or execute run.py file in python IDE  <br> **(All the parameters specified by user are listed in submit.sh).

2. &nbsp; Copy trained LCA or C-LCA model to the directory **./Model_Reducedata/LCA+PCA_pipeline/LCA_model**

3.  &nbsp; In **./Model_Reducedata/LCA_pipeline**, either run the script file **submit.sh** by terminal or execute **run.py** file in python IDE.

4. &nbsp; Three directories would be created with the name specified from parameters `-PCA_path`, `-graph_path`, and `-result_path`. <br>
   4.1 &nbsp; PCA model with its cumulative explained variance built from the latent space of LCA or C-LCA would be stored in `-PCA_path`, named `pca_{# principle component}.pkl` and `cum_var_ms.npy`, respectively. 3D plots of first three principal components would be created in `-PCA_path` as well.

   4.2 &nbsp; the testing result of reconstruction is stored in `-graph_path`. Two conditions are compared with ground truth (with filename **gtxx.png**). The first one is to test the capability of LCA or C-LCA by input ground truth data into encoder and reconstruct them from decoder directly (filename **Dir-decodexx.png**). The second one is to test the reconstructed capability of LCA or C-LCA + PCA by doing the same process again (filename **PCA+decodexx.png**). (**xx=** time)

   4.3 &nbsp; the low dimensional data reduced from C-LCA + PCA is stored in `-result_path` with name **train_data_{xx}**, **valid_data_{xx}**, and **test_data_{xx}** (**xx=** `-PCA_components`)
### III. &nbsp; HCA pipeline
1. &nbsp; Train HCA model in **./Autoencoder** by submitting the script file **submit.sh** in terminal or execute run.py file in python IDE 
(All the parameters specified by user are listed in submit.sh).

2. &nbsp; Copy trained HCA model to **./Model_Reducedata/HCA_pipeline/HCA_model**

3. &nbsp; In **./Model_Reducedata/HCA_pipeline**, either run the script file **submit.sh** by terminal or execute **run.py** file in python IDE.

4. &nbsp; the low dimensional data reduced from HCA is stored in `-result_path` with name **train_data_256**, **valid_data_256**, and **test_data_256**

## C. &nbsp; LMAU model training
**LMAU_model** :<br>
The code is modified from Ref.[3], the original MAU model, to inherit the capability of MAU model for predicting low dimensional data evolution.
### I. &nbsp; Autocorrelation pipeline
1. &nbsp;  copy the PCA model and PCA data been created from the file **gPCA.py** and **create_PCAdata.py** in **./Model_Reducedata/Autocorrelation_pipeline** to the directory in **./LMAU_model/Autocorrelation_pipeline/data/PCA_model** and **./LMAU_model/Autocorrelation_pipeline/data/PCA_data**

2. &nbsp; submit the **submit.sh** script file or execute the run.py file in python IDE for training, validation and testing.

### II. &nbsp; (C-)LCA+PCA pipeline
1. &nbsp;  copy the PCA model and reduced data been created from the file **gLCA_PCA.py** and **create_LCAencoderdata.py** in **./Model_Reducedata/LCA+PCA_pipeline** to the directory in **./LMAU_model/LCA+PCA_pipeline/data/PCA_model** and **./LMAU_model/LCA+PCA_pipeline/data/encoder_data**

2. &nbsp; copy the LCA or C-LCA been trained from **./Autoencoder/main_body.py** to the **./LMAU_model/LCA+PCA_pipeline/data/LCA_model**

3. &nbsp; submit the **submit.sh** script file or execute the run.py file in python IDE for training, validation and testing.

### III. &nbsp; HCA pipeline
1. &nbsp;  copy the reduced data been created from the file **create_HCAencoderdata.py** in **./Model_Reducedata/HCA_pipeline** to the directory in **./LMAU_model/HCA_pipeline/data/encoder_data**

2. &nbsp; copy the HCA model been trained from **./Autoencoder/main_body.py** to the **./LMAU_model/HCA_pipeline/data/HCA_model**

3. &nbsp; submit the **submit.sh** script file or execute the run.py file in python IDE for training, validation and testing.

## D. &nbsp; Analysis folder
* This folder includes three files: Autocorrelation.py Structure_factor.py for statistical analysis and GS_algorithm for phase recovery.


## Reference

[1] Variable Mobility Cahn-Hilliard sequential code repository https://github.com/abhinavroy1999/variable-mobility-Cahn-Hilliard-code

[2] Manzanarez, H., Mericq, J. P., Guenoun, P., Chikina, J., & Bouyer, D. (2017). Modeling phase inversion using Cahn-Hilliard equations–Influence of the mobility on the pattern formation dynamics. Chemical Engineering Science, 173, 411-427.

[3] Motion Aware Unit (MAU) repository https://github.com/ZhengChang467/MAU

[4] Zhu, J., Chen, L. Q., Shen, J., & Tikare, V. (1999). Coarsening kinetics from a variable-mobility Cahn-Hilliard equation: Application of a semi-implicit Fourier spectral method. Physical Review E, 60(4), 3564.

[5] Chen, B., Huang, K., Raghupathi, S., Chandratreya, I., Du, Q., & Lipson, H. (2022). Automated discovery of fundamental variables hidden in experimental data. Nature Computational Science, 2(7), 433-442.


## Citation
If you use this code for academic research, you are encouraged to cite the following paper: <br>
```
@article{chen2024LMAU,
author = {Chen SJ and Yu HY},
title = {L-MAU: A multivariate time-series network for predicting the Cahn-Hilliard microstructure evolutions via low-dimensional approaches}, journal = {Computer Physics Communications},
volume = {305},
year = {2024},
pages = {109342},
doi = {10.1016/j.cpc.2024.109342}
}
```

