#!/bin/bash
#BSUB -J callibrate_imagenet
#BSUB -q p1
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 20:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load cuda/12.4 
source mfe/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u experiments/applications/linearised_laplace/imagenet_callibration.py
