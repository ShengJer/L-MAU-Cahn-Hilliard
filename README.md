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

2. &nbsp; PCA model and PCA data from autocorrelation data would be created.

### II. &nbsp; (C-)LCA+PCA pipeline
1. &nbsp; Train LCA or C-LCA model in **./Autoencoder** by submitting the script file **submit.sh** in terminal or execute run.py file in python IDE 
(All the parameters specified by user are listed in submit.sh).

2. Copy trained LCA or C-LCA model to **./Model_Reducedata/LCA+PCA_pipeline/LCA_model**

3.  &nbsp; In **./Model_Reducedata/(C)LCA_pipeline**, either run the script file **submit.sh** by terminal or execute **run.py** file in python IDE.

4. &nbsp; PCA model and reduced data from (C)LCA+PCA would be created.

### III. &nbsp; HCA pipeline
1. &nbsp; Train HCA model in **./Autoencoder** by submitting the script file **submit.sh** in terminal or execute run.py file in python IDE 
(All the parameters specified by user are listed in submit.sh).

2. Copy trained HCA model to **./Model_Reducedata/HCA_pipeline/HCA_model**

3.  &nbsp; In **./Model_Reducedata/HCA_pipeline**, either run the script file **submit.sh** by terminal or execute **run.py** file in python IDE.

4. &nbsp; PCA model and reduced data from HCA would be created.

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

