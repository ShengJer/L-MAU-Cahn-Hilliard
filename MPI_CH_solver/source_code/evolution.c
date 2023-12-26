#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <mpi.h>
#include <gsl/gsl_math.h>
#include "../header_file/headers.h"

void cahn_hilliard_evolution(
	double kappa,
	double chi,
	double DA,
	double DB,
	int time_step,
	int record_step,
	double dt,
	int Nx,
	int Ny,
	double dx,
	double dy,
	ptrdiff_t *alloc_local,
	int *intsize,
	int *intlowidx,
	int *inthighidx,
	fftw_complex *comp,
	double *c)
{
	MPI_File fh;
	MPI_Status status;
	char file_name[50];
	int halfNx, halfNy;
	float kx, ky, delkx, delky;
	double k2, k4;
	double denominator;
	int i1, i2;
	int i11;
	int temp = 0;
	int id, p;
	int local_n0, local_0_start, highidx;
	
	fftw_complex *g;
	fftw_complex *r;
	fftw_complex *y1, *y2;
	fftw_complex *qx, *qy;
	fftw_complex *comp2, *s;
	
	// create local size:
	g = fftw_alloc_complex(*alloc_local);
	y1 = fftw_alloc_complex(*alloc_local);
	y2 = fftw_alloc_complex(*alloc_local);
	r = fftw_alloc_complex(*alloc_local);
	qx = fftw_alloc_complex(*alloc_local);
	qy = fftw_alloc_complex(*alloc_local);
	comp2 = fftw_alloc_complex(*alloc_local);
	s = fftw_alloc_complex(*alloc_local);
	
	// Stability factor
	double alpha = 0.5;
	
	
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	// create low, high index
	local_n0 = intsize[id];
	local_0_start = intlowidx[id];
	highidx = inthighidx[id];

	for(i1=0; i1<p;i1++)
	{
		intsize[i1] = intsize[i1]*Nx;
		intlowidx[i1] = intlowidx[i1]*Nx;
		
	}

	//Defining the plans for fourier transforms.
	fftw_plan plan1, plan2, plan3, plan4, plan5, plan6, plan7;

	plan1 = fftw_mpi_plan_dft_2d((ptrdiff_t)Nx, (ptrdiff_t)Ny, comp, comp, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
	plan2 = fftw_mpi_plan_dft_2d((ptrdiff_t)Nx, (ptrdiff_t)Ny, g, g, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
	plan3 = fftw_mpi_plan_dft_2d((ptrdiff_t)Nx, (ptrdiff_t)Ny, y1, y1, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
	plan4 = fftw_mpi_plan_dft_2d((ptrdiff_t)Nx, (ptrdiff_t)Ny, y2, y2, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
	plan5 = fftw_mpi_plan_dft_2d((ptrdiff_t)Nx, (ptrdiff_t)Ny, qx, qx, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
	plan6 = fftw_mpi_plan_dft_2d((ptrdiff_t)Nx, (ptrdiff_t)Ny, qy, qy, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
	plan7 = fftw_mpi_plan_dft_2d((ptrdiff_t)Nx, (ptrdiff_t)Ny, comp, comp, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

	halfNx = Nx/2;
	halfNy = Ny/2;

	//delta kx and delta ky - for defining the Fourier space vectors.
	delkx = (2*M_PI)/((double)Nx*dx);
	delky = (2*M_PI)/((double)Ny*dy);
	
	
	// gather and initialize composition by MPI/IO
	sprintf(file_name, "output/time%d.dat", temp);
	MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_write_ordered(fh, c, (int)local_n0*Nx, MPI_DOUBLE, &status);
	MPI_File_close(&fh);
	
	//Loop for temporal evolution.
	for (temp = 1; temp <time_step+1; ++temp)
	{
		for (i1 = 0; i1<local_n0; ++i1)
		{
			for (i2 = 0; i2<Nx; ++i2)
			{
				__real__(comp2[i2 + i1*Nx]) = __real__(comp[i2 + i1*Nx]);
				__imag__(comp2[i2 + i1*Nx]) = 0.0;
			}
		}
		for(i1=0; i1<local_n0; ++i1)
		{
			for(i2=0; i2<Nx; ++i2)
			{
				g[i2+Ny*i1] = log(comp2[i2+Ny*i1]) - log(1.0-comp2[i2+Ny*i1]) + chi*(1.0-2.0*comp2[i2+Ny*i1]);
			}
		}
		
		fftw_execute(plan1);
		fftw_execute(plan2);
		
		for(i1=local_0_start; i1 <= highidx; ++i1)
		{
			if(i1 <= halfNy)
			{ ky = i1*delky;}
			else
			{ ky = (i1-Ny)*delky;}
		
			i11 = i1 - local_0_start;
			for(i2=0; i2<Nx; ++i2)
			{
				if(i2 <= halfNx)
				{ kx = i2*delkx;}
				else
				{ kx = (i2-Nx)*delkx;}
			
				k2 = kx*kx + ky*ky;
				r[i2 + i11*Nx] = g[i2 + i11*Nx] + kappa*k2*comp[i2 + i11*Nx];
			}
		}

		for(i1=local_0_start; i1 <= highidx; ++i1)
		{
			if(i1 <= halfNy)
			{ ky = i1*delky;}
			else
			{ ky = (i1-Ny)*delky;}
			i11 = i1 - local_0_start;
			for(i2=0; i2<Nx; ++i2)
			{
				if(i2 <= halfNx)
				{ kx = i2*delkx;}
				else
				{ kx = (i2-Nx)*delkx;}
				y1[i2 + i11*Nx] = (_Complex_I)*kx*r[i2 + i11*Nx];
				y2[i2 + i11*Nx] = (_Complex_I)*ky*r[i2 + i11*Nx];
			}
		}

		fftw_execute(plan3);	// inverse fourier transform of y1
		fftw_execute(plan4);	// inverse fourier transform of y2
		
		//Normalizing the r complex array after the inverse fourier transform (as required by the fftw algorithm).
		for(i1=0; i1<local_n0; ++i1)
		{
			for(i2=0; i2<Nx; ++i2)
			{
				y1[i2 + i1*Nx] = y1[i2 + i1*Nx]/(Nx*Ny);
			}
		}
		for(i1=0; i1<local_n0; ++i1)
		{
			for(i2=0; i2<Nx; ++i2)
			{
				y2[i2 + i1*Nx] = y2[i2 + i1*Nx]/(Nx*Ny);
			}
		}
		for (i1=0; i1 < local_n0; ++i1)
		{
			for (i2=0; i2 < Nx; ++i2)
			{
				s[i2 + Ny*i1] = (DA*(1-comp2[i2 + Ny*i1])+DB*comp2[i2 + Ny*i1])*comp2[i2 + Ny*i1]*(1-comp2[i2 + Ny*i1]);
				qx[i2 + Ny*i1] = s[i2 + Ny*i1]*y1[i2 + Ny*i1];
				qy[i2 + Ny*i1] = s[i2 + Ny*i1]*y2[i2 + Ny*i1];
				
			}
		}
		
		fftw_execute(plan5);
		fftw_execute(plan6);

		//evolving the composition profile.
		for(i1=local_0_start; i1 <= highidx; ++i1)
		{
			if(i1 <= halfNy)
			{ ky = i1*delky;}
			else
			{ ky = (i1-Ny)*delky;}
			i11 = i1 - local_0_start;
			for(i2=0; i2<Nx; ++i2)
			{
				if(i2 <= halfNx)
				{ kx = i2*delkx;}
				else
				{ kx = (i2-Nx)*delkx;}
				k2 = kx*kx + ky*ky;
				k4 = k2*k2;
				denominator = (1.0 + alpha*dt*kappa*k4);
				comp[i2 + i11*Nx] = comp[i2 + i11*Nx] + (dt*((_Complex_I)*kx*qx[i2 + i11*Nx] + (_Complex_I)*ky*qy[i2 + i11*Nx]))/denominator;
			}
		}
		fftw_execute(plan7);
		
		for(i1=0; i1<local_n0; ++i1)
		{
			for(i2=0; i2<Nx; ++i2)
			{
				comp[i2 + i1*Nx] = comp[i2 + i1*Nx]/(Nx*Ny);

			}
		}
		
		//Taking the output every record_step interval of time.
		if (temp%record_step == 0)
		{	
			
			for (i1 = 0; i1<local_n0; ++i1)
			{
				for (i2 = 0; i2<Nx; ++i2)
				{
					c[i2 + i1*Nx] = __real__(comp[i2 + i1*Nx]);
				}
			}
			
			//write out the file by MPI/IO
			sprintf(file_name, "output/time%d.dat", temp);
			MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
			MPI_File_write_ordered(fh, c, (int)local_n0*Nx, MPI_DOUBLE, &status);
			MPI_File_close(&fh);
			
			
		}
		
	}

	//free the memory allocated for all the variables.
	fftw_free(g);
	fftw_free(r);
	fftw_free(y1);
	fftw_free(y2);
	fftw_free(qx);
	fftw_free(qy);
	fftw_free(comp2);
	fftw_free(s);
	fftw_destroy_plan(plan1);
	fftw_destroy_plan(plan2);
	fftw_destroy_plan(plan3);
	fftw_destroy_plan(plan4);
	fftw_destroy_plan(plan5);
	fftw_destroy_plan(plan6);
	fftw_destroy_plan(plan7);
}
