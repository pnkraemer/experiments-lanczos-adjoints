#!/bin/bash

#BSUB -q gpuv100
#BSUB -J gp-train-again
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 14:00
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/gp-train-again-%J.out
#BSUB -e logs/gp-train-again-%J.err

### Load the cuda module
module load cuda/12.4
source ../penv/bin/activate

# Optimisation with Adam(0.1) is built in.
# (Even though Pytorch's and Optax's implementations differ slightly)
#
# Eval-CG tolerance 1e-4 is built in.
# 80/20 Train/test splits are built in.

# No point setting below 10, gpytorch seems to override anyway
# CG usually converged within 10-20 iterations, so 20 seems fine
num_matvecs=10

# Like in GPyTorch
cg_tol=1e0
rank_precon=15

# Similar to GPyTorch (slightly less, but checkpoints...)
num_samples=10

# Slightly more than in most papers
# (because 50 aren't enough for convergence)
num_epochs=100


for seed in 1 2 3 4 5
do
  for dataset in kin40k elevators kegg_directed kegg_undirected protein elevators
  do
    python experiments/applications/gaussian_process/train/optim_logml_gpytorch_adaptive.py \
      --name gpytorch-next-run-nll --seed $seed --num_matvecs $num_matvecs \
      --num_samples $num_samples --num_epochs $num_epochs --cg_tol $cg_tol \
      --dataset $dataset --rank_precon $rank_precon --num_partitions 10;
    python experiments/applications/gaussian_process/train/optim_logml_adjoints_adaptive.py \
      --name adjoints-next-run-nll --seed $seed --num_matvecs $num_matvecs \
      --num_samples $num_samples --num_epochs $num_epochs --cg_tol $cg_tol \
      --dataset $dataset --rank_precon $rank_precon  --num_partitions 10;
  done
done
