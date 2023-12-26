#suffixes used
CC = mpicc

.SUFFIXES: .o .h .c 

#compilation options

COMPOPS = -g -Wall -O3 -ldl

#linking options

LINKOPS = -lgsl -lgslcblas -lfftw3_mpi -lfftw3 -lm

#list of object files

objects = VariableCH.o evolution.o

#list of header files

headers = stdio.h stdlib.h math.h complex.h gsl_rng.h gsl_math.h fftw3-mpi.h mpi.h headers.h time.h

#list of source codes

sources = VariableCH.c evolution.c

#directory paths for the source and header files

INCLUDE_FFTW3 = -I/usr/local/include

LIB_FFTW3 = -L/usr/local/lib

vpath %.c source_code
vpath %.h ./header_file/
vpath %.h /usr/lib/gcc/x86_64-linux-gnu/11/include/ #gcc compiler directory. To be changed accordingly based on the machine.
vpath %.h /usr/local/include/    #directory path for fftw3 fftw3-mpi header function. To be changed accordingly based on the directory path on the machine.
vpath %.h /usr/lib/               #lsgl and lsglcblas file directory. To be changed accordingly based on the machine.
vpath %.h /usr/include/gsl/                      #gsl header functions directory. To be changed accordingly based on the machine.
vpath %.h /home/sheng/local/opt/openmpi/include/ #mpi.h file directory
vpath %.h /usr/include/
#actions on the source files	

VariableCH: $(objects) $(headers)
	$(CC) -o CH_variable_mobility.out $(objects) $(INCLUDE_FFTW3) $(LIB_FFTW3) $(LINKOPS) 
VariableCH.o: $(sources) $(headers)
	$(CC) -o $@ -c ./source_code/VariableCH.c $(COMPOPS) $(INCLUDE_FFTW3) $(LIB_FFTW3)
evolution.o: evolution.c stdio.h stdlib.h math.h complex.h gsl_math.h fftw3-mpi.h headers.h mpi.h
	$(CC) -c $(COMPOPS) $< $(INCLUDE_FFTW3) $(LIB_FFTW3)


.PHONY : clean CLEAN
clean:
	rm -rf *.o
CLEAN:
	rm -rf CH_variable_mobility.out


#end of the Makefile
