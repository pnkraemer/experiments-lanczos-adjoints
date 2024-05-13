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

time python experiments/applications/gaussian_process/error_metrics/gp_jax_test.py --name adjoints_small --seed 1 --num_data 4000 --rank_precon 50 --num_matvecs 10 --num_samples 1 --num_epochs 50 --cg_tol 1. --num_partitions 500;
time python experiments/applications/gaussian_process/error_metrics/gp_keops_test.py --name gpytorch_small --seed 1 --num_data 400000 --rank_precon 500 --num_matvecs 10 --num_samples 5 --num_epochs 50 --cg_tol 1.;
time python experiments/applications/gaussian_process/error_metrics/gp_jax_test.py --name adjoints_big --seed 1 --num_data 400000 --rank_precon 500 --num_matvecs 15 --num_samples 15 --num_epochs 50 --cg_tol 1e-2 --num_partitions 288;
time python experiments/applications/gaussian_process/error_metrics/gp_keops_test.py --name gpytorch_big --seed 1 --num_data 400000 --rank_precon 500 --num_matvecs 15 --num_samples 15 --num_epochs 50 --cg_tol 1e-2;
