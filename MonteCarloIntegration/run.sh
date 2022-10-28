#! /bin/sh

# $1 -- MPI process count
# $2 -- eps

mpirun -np $1 ./pmci $2
