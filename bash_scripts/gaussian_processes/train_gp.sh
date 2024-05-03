#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J train_gp
### do we need more than one core?
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
###BSUB -R "rusage[mem=12GB]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -B
#BSUB -N
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

pwd
### Load the cuda module
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source ../adjoints/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python experiments/applications/gaussian_process/error_metrics/gp_train_and_runtimes.py -gpm adjoints -kry 1 -sb 1_000 -e 5 -data airquality
