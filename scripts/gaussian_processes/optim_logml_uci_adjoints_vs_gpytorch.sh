#!/bin/bash
#BSUB -q gpuv100
#BSUB -J gp_jax
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o gp_jax-400x-%J.out
#BSUB -e gp_jax-400x-%J.err

### Load the cuda module
module load cuda/12.4
source ../penv/bin/activate

time python experiments/applications/gaussian_process/train/optim_logml_adjoints_adaptive.py --name adjoints_small --seed 1 --num_data 4000 --rank_precon 50 --num_matvecs 10 --num_samples 1 --num_epochs 50 --cg_tol 1. --num_partitions 10;
