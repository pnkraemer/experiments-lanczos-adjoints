#!/bin/bash
#BSUB -q gpuv100
#BSUB -J gp_jax
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o gp_jax-%J.out
#BSUB -e gp_jax-%J.err

### Load the cuda module
module load cuda/12.4
source ../adjoints/bin/activate

python experiments/applications/gaussian_process/error_metrics/gp_jax_test.py
