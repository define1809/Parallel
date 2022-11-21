#!/bin/sh

M=$1
N=$2

./pde_seq $M $N > res.txt
python3 vis.py
