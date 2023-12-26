#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>
#include<fftw3-mpi.h>
#include<gsl/gsl_rng.h>
#include<mpi.h>
#include<time.h>
#include "../header_file/headers.h"


int main(int argc, char *argv[])
{
	
	ptrdiff_t Nx, Ny; 
	double dx,dy;
	double chi,kappa, DA, DB;
	int time_step, record_step;
	double dt;
	double c_zero, c_noise;
	double *c;
	gsl_rng * ran_num;
	const gsl_rng_type * Taus;

	
	int i1,i2;
	int id, p;
	int i;
	ptrdiff_t alloc_local, local_n0, local_0_start;
	ptrdiff_t highidx;
	ptrdiff_t *LOW_IDX;
	ptrdiff_t *HIGH_IDX;
	ptrdiff_t *SIZE;
	int *intsize;
	int *intlowidx;
	int *inthighidx;
	
	MPI_Status status;
	fftw_complex *comp;
	
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	LOW_IDX=(ptrdiff_t *)malloc(1*p*sizeof(ptrdiff_t));
	HIGH_IDX=(ptrdiff_t *)malloc(1*p*sizeof(ptrdiff_t));
	SIZE=(ptrdiff_t *)malloc(1*p*sizeof(ptrdiff_t));
	
	intsize=(int *)malloc(1*p*sizeof(int));
	intlowidx=(int *)malloc(1*p*sizeof(int));
	inthighidx=(int *)malloc(1*p*sizeof(int));
	srand(id);
	if (id == p-1)
	{
		FILE *fr, *fw;
		//Clearing the output directory in order to print fresh result from every new simulation run.
		(void) system("rm -rf output/*");


		//Opening the file simulation_data where all the physical constants and system information related to the simulation is stored.
		if ((fw = fopen("output/simulation_data","w")) == NULL)
		{
			printf("Unable to open output/simulation_data. Exiting.");
			exit(0);
		}
		else
		{
			fw = fopen("output/simulation_data","w");
		}	
		
		if ((fr = fopen("input/constants_interface","r")) == NULL)
		{
			printf("Unable to open input/constants_interface. Exiting.");
			exit(0);
		}
		else
		{
			fr = fopen("input/constants_interface","r");
		}
		(void)fscanf(fr,"%le%le%le%le",&kappa,&chi,&DA,&DB);
		fclose(fr);
		(void)fprintf(fw,"kappa = %le\nchi = %le\nDA = %le\nDB = %le\n",kappa,chi,DA,DB);
		

		if ((fr = fopen("input/time_info","r")) == NULL)
		{
			printf("Unable to open input/time_info. Exiting.");
			exit(0);
		}
		else
		{
			fr = fopen("input/time_info","r");
		}
		(void)fscanf(fr,"%d%d%le",&time_step,&record_step,&dt);
		(void)fclose(fr);
		fprintf(fw,"time_step = %d\nrecord_step = %d\ndt = %le\n",time_step, record_step, dt);

		if ((fr = fopen("input/system_info","r")) == NULL)
		{
			printf("Unable to open input/system_info. Exiting.");
			exit(0);
		}
		else
		{
			fr = fopen("input/system_info","r");
		}
		(void)fscanf(fr,"%ld%ld%le%le",&Nx,&Ny,&dx,&dy);
		fclose(fr);
		(void)fprintf(fw,"Nx = %ld\nNy = %ld\n", Nx, Ny);
		(void)fprintf(fw,"dx = %le\ndy = %le\n", dx, dy);
		

		if ((fr = fopen("input/composition_profile","r")) == NULL)
		{
			printf("Unable to open input/composition_profile. Exiting.");
			exit(0);
		}
		else
		{
			fr = fopen("input/composition_profile","r");
		}
		(void)fscanf(fr,"%le%le", &c_zero, &c_noise);
		(void)fclose(fr);
		fprintf(fw,"c_zero = %le\nc_noise = %le\n", c_zero, c_noise);
		
		 //Closing the file pointer for writing the simulation data.
		(void)fclose(fw);
		fflush(fw);
	}
	
	// read from last processors and broadcast to all the processors
	MPI_Bcast(&Nx, 1, MPI_AINT, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&Ny, 1, MPI_AINT, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&kappa, 1, MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&chi, 1,  MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&DA, 1,  MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&DB, 1,  MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&time_step, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&record_step, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&dt, 1,  MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&dx, 1,  MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&dy, 1,  MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&c_noise, 1,  MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&c_zero, 1,  MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	
	// initialized the fftw3 buffer and get the index token
	fftw_mpi_init();
	alloc_local = fftw_mpi_local_size_2d(Nx, Ny, MPI_COMM_WORLD, &local_n0, &local_0_start);
	highidx = local_0_start + local_n0 - 1;
	
	// store the information local_n0, local_0_start, highidx on all the processors
	MPI_Allgather(&local_n0, 1, MPI_AINT, SIZE, 1, MPI_AINT, MPI_COMM_WORLD);
	MPI_Allgather(&local_0_start, 1, MPI_AINT, LOW_IDX, 1, MPI_AINT, MPI_COMM_WORLD);
	MPI_Allgather(&highidx, 1, MPI_AINT, HIGH_IDX, 1, MPI_AINT, MPI_COMM_WORLD);
	
	// change the type from ptrdiff_t to integer (this may not be necessary)
	for(i=0; i<p;i++)
	{
		intsize[i] = (int) SIZE[i];
		intlowidx[i] = (int) LOW_IDX[i];
		inthighidx[i] = (int) HIGH_IDX[i];
	}
	
	free(LOW_IDX);
	free(HIGH_IDX);
	free(SIZE);
	
	// create memory for each comp(store complex number) and c (store real double)
	comp = fftw_alloc_complex(alloc_local);
	c = (double *)malloc((size_t) Nx*local_n0* sizeof(double));
	
	(void) gsl_rng_env_setup();
	Taus = gsl_rng_taus;
	ran_num = gsl_rng_alloc (Taus);
	gsl_rng_set(ran_num, rand()*id);
	//Setting the initial composition profile.
	
	
	
	for(i1=0; i1 < local_n0; ++i1)
	{
		for(i2=0; i2 < Nx; ++i2)
		{
		  __real__(comp[i2+Nx*i1]) = c_zero + c_noise*(0.5 - gsl_rng_uniform_pos(ran_num));
		  __imag__(comp[i2+Nx*i1]) = 0.0;
		  c[i2+Nx*i1] = __real__(comp[i2+Nx*i1]);

		}
	}
	
	
	gsl_rng_free(ran_num);
	
	cahn_hilliard_evolution(kappa, chi, DA, DB, time_step, record_step, dt, (int)Nx, (int)Ny, dx, dy, &alloc_local, intsize, intlowidx, inthighidx, comp, c);
	
	
	free(intsize);
	free(intlowidx);
	free(inthighidx);
	free(c);
	fftw_free(comp);
	MPI_Finalize();
	return 0;
}
/*------------------------------------------------End of CODE---------------------------------------------------------*/
