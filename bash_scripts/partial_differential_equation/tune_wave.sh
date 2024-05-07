printf "\nGenerating data... TODO\n"
time python experiments/applications/partial_differential_equation/make_data.py  --seed 1 --resolution 128 --num_data 128;


printf "\nCollecting data for the workprecision diagram\n"
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 128 --num_runs 3 --num_steps_max 30 --method arnoldi;
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 128 --num_runs 3 --num_steps_max 30 --method diffrax:euler+backsolve;
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 128 --num_runs 3 --num_steps_max 30 --method diffrax:heun+recursive_checkpoint;
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 128 --num_runs 3 --num_steps_max 30 --method diffrax:dopri5+backsolve;
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 128 --num_runs 3 --num_steps_max 30 --method diffrax:tsit5+recursive_checkpoint;

printf "\nTraining the network...\n"
time python experiments/applications/partial_differential_equation/train.py --num_epochs 2000 --resolution 128 --method arnoldi;
time python experiments/applications/partial_differential_equation/train.py --num_epochs 2000 --resolution 128 --method diffrax:euler+backsolve;
time python experiments/applications/partial_differential_equation/train.py --num_epochs 2000 --resolution 128 --method diffrax:heun+recursive_checkpoint;
time python experiments/applications/partial_differential_equation/train.py --num_epochs 2000 --resolution 128 --method diffrax:dopri5+backsolve;
time python experiments/applications/partial_differential_equation/train.py --num_epochs 2000 --resolution 128 --method diffrax:tsit5+recursive_checkpoint;
