#!/bin/sh
export OMP_NUM_THREADS=4
mpirun -np 1 ./pde_hybrid $1 $2 > out.txt
