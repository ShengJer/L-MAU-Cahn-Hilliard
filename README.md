#  MPI Cahn-Hilliard solver and L-MAU model

## MPI_CH_solver folder
This code is modified from Ref.[2] to include simple parallelism with MPI.
The Cahn-Hilliard model includes Flory-Huggins free energy and fast mobility model.
1. Requirement
   * mpicc (OpenMPI or other MPI library)
   * fftw3 https://www.fftw.org/
   * gnu/gsl https://www.gnu.org/software/gsl/ <br>

2. Compiling
   * Please follow the makefile comment to change the vpath directory of gsl, fftw3, and mpi based on the machine
   * After successful compilation, two object files: VariableCH.o evolution.o and one executable file CH_variable_mobility.out would be created
   
3. Run
   * You can either run it with terminal or by submitting a script file to the schedulers 
```
mpirun -np <number_of_processors> CH_variable_mobility.out

```   
## L-MAU_model folder

   
	

























## Reference
[1] Motion Aware Unit (MAU) repository https://github.com/ZhengChang467/MAU

[2] Variable Mobility Cahn-Hilliard sequential code repository https://github.com/abhinavroy1999/variable-mobility-Cahn-Hilliard-code

[3] Zhu, J., Chen, L. Q., Shen, J., & Tikare, V. (1999). Coarsening kinetics from a variable-mobility Cahn-Hilliard equation: Application of a semi-implicit Fourier spectral method. Physical Review E, 60(4), 3564.