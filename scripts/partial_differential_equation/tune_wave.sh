#!/bin/bash

#BSUB -q gpuv100
#BSUB -J pde-final-run
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 6:00
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/pde-final-run-%J.out
#BSUB -e logs/pde-final-run-%J.err

### Load the cuda module
module load cuda/12.4
source ../penv/bin/activate

# For all problems
resolution=128
num_data=256

# For the work-precision scripts
num_runs=5
num_steps_max=30

# For the training script
num_epochs=3000


printf "\nGenerating data...\n"
time python experiments/applications/partial_differential_equation/make_data.py  \
    --resolution $resolution --num_data $num_data --seed 1;


printf "\nCollecting results for the workprecision diagram\n"
for method in \
    arnoldi \
    diffrax:euler+backsolve \
    diffrax:heun+recursive_checkpoint \
    diffrax:dopri5+backsolve \
    diffrax:tsit5+recursive_checkpoint
do 
    time python experiments/applications/partial_differential_equation/workprecision.py  \
        --resolution $resolution --num_runs $num_runs \ 
        --num_steps_max $num_steps_max --method $method;
done


printf "\nTraining the network...\n"
for seed in 1 2 3 4 5
do
    for method in \
        arnoldi \
        diffrax:tsit5+recursive_checkpoint \
        diffrax:dopri5+backsolve
    do
        time python experiments/applications/partial_differential_equation/train.py \
            --num_epochs $num_epochs --resolution $resolution --method $method --seed $seed;
    done
done 
