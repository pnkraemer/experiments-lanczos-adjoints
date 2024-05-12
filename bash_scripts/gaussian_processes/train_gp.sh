#!/bin/bash
#BSUB -q gpuv100
#BSUB -J gp_jax
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o gp_jax-40x-%J.out
#BSUB -e gp_jax-40x-%J.err

### Load the cuda module
module load cuda/12.4
source ../penv/bin/activate

time python experiments/applications/gaussian_process/error_metrics/gp_jax_test.py --name adjoints --seed 2 --num_data 40000 --rank_precon 500 --num_matvecs 10 --num_samples 3 --num_epochs 50 --num_partitions 20;
time python experiments/applications/gaussian_process/error_metrics/gp_keops_test.py --name gpytorch --seed 2 --num_data 40000 --rank_precon 500 --num_matvecs 10 --num_samples 3 --num_epochs 50 --cg_tol 1.;
time python experiments/applications/gaussian_process/error_metrics/gp_keops_test.py --name gpytorch3x --seed 2 --num_data 40000 --rank_precon 500 --num_matvecs 30 --num_samples 9 --num_epochs 50 --cg_tol 1e-2;
