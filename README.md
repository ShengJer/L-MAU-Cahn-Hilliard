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
## L-MAU_model folder
The code is modified from Ref.[3], the original MAU model, to inherit the capability of MAU model for predicting low dimensional data evolution.

### Autocorrelation pipeline
1. Please generate the evolution by MPI solver and store it into a (TL, Ny, Nx) numpy array which TL is the total simulation time steps and Nx, Ny is the 2D spatial dimension.
  
2. Place the training data into ./data/auto_data/auto_train_data, the validating data into ./data/auto_data/auto_valid_data, and the testing data into ./data/auto_data/auto_test_data.
	
3. modify and run the ./data/gPCA.py file to generate PCA model in ./data/PCA_model (some parameters need to be changed according to the size of training set, and the PCA components you want).

4. run the ./data/create_PCAdata to transform autocorrelation data into PCA data with reduced dimension by the PCA model been created early (some parameters still need to be change according to the user at the start of the .py file).

5. To run the L-MAU model, please change the parameters inside the submit.sh file and execute the bashfile by ` bash submit.sh` or write a script file for the cluster scheduler.


## Analysis folder
* This folder includes two files: Autocorrelation.py and Structure_factor.py for statistical analysis

## Autoencoder folder
1. This folder includes the model of LCA (low compression ratio) and HCA (high compression ratio) in model_autoencoder.py and the training process in run.py

2. the submit.sh file show an example of the reqired parameters for execute run.py (user must specify the training data directory: train_filepath)

3. the training data directory must has several training data (.npz file) having shape (TL(total time), Ny, Nx) that can be loaded by numpy command ` np.load(filename)['data']` (an example file is placed in directory train_data)

### LCA pipelines
  a. train the low compression autoencoder and transform high dimensional training and validating dataset into latent space features.
  
  b. generate PCA model and data by fitting the latent space features.
  
  c. transform latent space features into PCA data with reduced dimension by the PCA model been created.

  d. run the L-MAU model to perform prediction on low dimensional evolution. 



## Reference

[1] Variable Mobility Cahn-Hilliard sequential code repository https://github.com/abhinavroy1999/variable-mobility-Cahn-Hilliard-code

[2] Manzanarez, H., Mericq, J. P., Guenoun, P., Chikina, J., & Bouyer, D. (2017). Modeling phase inversion using Cahn-Hilliard equations–Influence of the mobility on the pattern formation dynamics. Chemical Engineering Science, 173, 411-427.

[3] Motion Aware Unit (MAU) repository https://github.com/ZhengChang467/MAU

[4] Zhu, J., Chen, L. Q., Shen, J., & Tikare, V. (1999). Coarsening kinetics from a variable-mobility Cahn-Hilliard equation: Application of a semi-implicit Fourier spectral method. Physical Review E, 60(4), 3564.

[5] Chen, B., Huang, K., Raghupathi, S., Chandratreya, I., Du, Q., & Lipson, H. (2022). Automated discovery of fundamental variables hidden in experimental data. Nature Computational Science, 2(7), 433-442.


## Citation
If you use this code for academic research, you are encouraged to cite the following paper: <br>
```
@article{chen2024LMAU, author = {Chen SJ and Yu HY},
title = {L-MAU: A multivariate time-series network for predicting the Cahn-Hilliard microstructure evolutions via low-dimensional approaches}, journal = {Computer Physics Communications},
volume = {305},
year = {2024},
pages = {109342},
doi = {10.1016/j.cpc.2024.109342}
}
```

