#  MPI Cahn-Hilliard solver and L-MAU model

## MPI_CH_solver folder
This code is modified from Ref.[2] to include simple parallelism with MPI.
The Cahn-Hilliard model includes Flory-Huggins free energy and fast mobility model[4].
1. Requirement
   * mpicc (OpenMPI or other MPI library)
   * fftw3 https://www.fftw.org/
   * gnu/gsl https://www.gnu.org/software/gsl/ <br>

2. Compiling code
   * Please follow the makefile comment to change the vpath directory of gsl, fftw3, and mpi based on the machine
   * After successful compilation, two object files: VariableCH.o evolution.o and one executable file CH_variable_mobility.out would be created <br>
   
3. Run 
```
mpirun -np <number_of_processors> CH_variable_mobility.out

``` 
## L-MAU_model folder

### Autocorrelation pipeline
1. Please generate the evolution by MPI solver and store it into a (TL, Ny, Nx) numpy array which TL is the total simulation time steps and Nx, Ny is the 2D spatial dimension 
  
2. Place the training data into ./data/auto_data/auto_train_data, the validating data into ./data/auto_data/auto_valid_data, and the testing data into ./data/auto_data/auto_test_data  
	
3. modify and run the ./data/gPCA.py file to generate PCA model in ./data/PCA_model (some parameters need to be changed according to the size of training set, and the PCA components you want)

4. run the ./data/create_PCAdata to transform autocorrelation data into PCA data with reduced dimension by the PCA model created early (some parameters still need to be change according to the user at the start of the .py file)

5. 





















## Reference
[1] Motion Aware Unit (MAU) repository https://github.com/ZhengChang467/MAU

[2] Variable Mobility Cahn-Hilliard sequential code repository https://github.com/abhinavroy1999/variable-mobility-Cahn-Hilliard-code

[3] Zhu, J., Chen, L. Q., Shen, J., & Tikare, V. (1999). Coarsening kinetics from a variable-mobility Cahn-Hilliard equation: Application of a semi-implicit Fourier spectral method. Physical Review E, 60(4), 3564.

[4] Manzanarez, H., Mericq, J. P., Guenoun, P., Chikina, J., & Bouyer, D. (2017). Modeling phase inversion using Cahn-Hilliard equations–Influence of the mobility on the pattern formation dynamics. Chemical Engineering Science, 173, 411-427.
