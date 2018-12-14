#!/usr/bin/env bash
echo "starting timing"
cd build/
module load cmake/3.9.1 gcc openmpi

mpirun -np 8 ./p30 forward ../TimeTest128.txt ../Output.txt
mpirun -np 8 ./p30 forward ../TimeTest256.txt ../Output.txt
mpirun -np 8 ./p30 forward ../TimeTest512.txt ../Output.txt
mpirun -np 8 ./p30 forward ../TimeTest1024.txt ../Output.txt
mpirun -np 8 ./p30 forward ../TimeTest2048.txt ../Output.txt
