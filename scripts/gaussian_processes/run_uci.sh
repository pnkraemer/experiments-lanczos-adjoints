#!/bin/bash
#BSUB -q gpuv100
#BSUB -J gp_jax
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/final_run_uci-%J.out
#BSUB -e logs/final_run_uci-%J.err

### Load the cuda module
module load cuda/12.4
source ../penv/bin/activate

# Optimisation with Adam(0.1) is built in.
# Eval-CG tolerance 1e-4 is built in.
# 80/20 Train/test splits are built in.

# No point setting below 10, gpytorch seems to override anyway
num_matvecs=20  

# Like in GPyTorch
cg_tol=1e0      
rank_precon=15 

# Similar to GPyTorch (slightly less, but hey)
num_samples=10  

# Like in most papers
num_epochs=50    

for seed in 1
do
  for dataset in concrete power_plant elevators protein kin40k kegg_directed kegg_undirected
  do
    python experiments/applications/gaussian_process/train/optim_logml_gpytorch_adaptive.py \
      --name final-gpytorch --seed $seed --num_matvecs $num_matvecs \
      --num_samples $num_samples --num_epochs $num_epochs --cg_tol $cg_tol \
      --dataset $dataset --rank_precon $rank_precon --num_partitions 10;
    python experiments/applications/gaussian_process/train/optim_logml_adjoints_adaptive.py \
      --name final-adjoints --seed $seed --num_matvecs $num_matvecs \
      --num_samples $num_samples --num_epochs $num_epochs --cg_tol $cg_tol \
      --dataset $dataset --rank_precon $rank_precon  --num_partitions 10;
  done
done
