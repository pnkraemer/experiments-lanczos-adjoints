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
source ../adjoints/bin/activate

python experiments/benchmarks/gram_matvec_versus_keops/matvec/benchmark_size_toy.py
