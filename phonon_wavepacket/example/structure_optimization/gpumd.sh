#!/bin/sh
#PBS -q DA_002
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:10:00
#PBS -N gpumd

cd ${PBS_O_WORKDIR}
 
module load cuda

/home/sunq/Peng-HuDu/code/gpumd/GPUMD-3.9.1/src/gpumd < input_gpumd.txt
