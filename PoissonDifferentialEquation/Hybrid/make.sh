#!/bin/sh
mpicc pde_hybrid.c -fopenmp -lm -O3 -o pde_hybrid
