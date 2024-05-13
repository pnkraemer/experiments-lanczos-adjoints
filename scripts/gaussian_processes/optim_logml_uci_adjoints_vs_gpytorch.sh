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

name=adjoints_small
seed=1
rank_precon=50
num_partitions=10
num_matvecs=10
num_epochs=50
num_samples=1
cg_tol=1.0

for dataset in concrete power_plant
do
  for seed in 1 2 3
  do
  python experiments/applications/gaussian_process/train/optim_logml_adjoints_adaptive.py \
    --name $name --dataset $dataset --seed $seed --rank_precon $rank_precon --num_matvecs $num_matvecs \
    --num_samples $num_samples --num_epochs $num_epochs --cg_tol $cg_tol --num_partitions $num_partitions;
  done
done

time python experiments/applications/gaussian_process/train/optim_logml_adjoints_adaptive.py --name $name --dataset $dataset --seed 1 --rank_precon 50 --num_matvecs 10 --num_samples 1 --num_epochs 50 --cg_tol 1. --num_partitions 10;
