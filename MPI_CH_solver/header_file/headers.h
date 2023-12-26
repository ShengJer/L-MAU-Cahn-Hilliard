#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>
#include<fftw3-mpi.h>

#ifndef _HEADERS_H
#define _HEADERS_H


#define DATA_MSG 0
#define TERMINATE_MSG 1
#define MALLOC_ERROR -2


extern void cahn_hilliard_evolution(
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
	double *c);

#endif

