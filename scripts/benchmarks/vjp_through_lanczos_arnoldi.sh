#!/bin/bash

#BSUB -q gpuv100
#BSUB -J runtime-run
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 6:00
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/runtime-run-%J.out
#BSUB -e logs/runtime-run-%J.err

### Load the cuda module
module load cuda/12.4
source ../penv/bin/activate

# Stop backprop early (bc inefficient)
backprop_until=100

# Go up to K=500
max_krylov_depth=50

# Test different matrices, methods, reorthogonalisation
# and measure compile- and run-time

for matrix in "1138_bus" "gyro"
do
    for method in "arnoldi" "lanczos" 
    do
        for reortho in "full" "none" 
        do 
            time python experiments/benchmarks/wall_times_vjp_through_lanczos_arnoldi/suite_sparse/benchmark.py  --which_matrix $matrix --lanczos_or_arnoldi $method --reortho $reortho --max_krylov_depth $max_krylov_depth --backprop_until $backprop_until  --num_runs 1;

            time python experiments/benchmarks/wall_times_vjp_through_lanczos_arnoldi/suite_sparse/benchmark.py --which_matrix $matrix --lanczos_or_arnoldi $method --reortho $reortho --max_krylov_depth $max_krylov_depth --backprop_until $backprop_until  --num_runs 5 --precompile;
        done 
    done
done
