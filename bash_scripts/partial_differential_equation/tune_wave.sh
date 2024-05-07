printf "\nGenerating data... TODO\n"
time python experiments/applications/partial_differential_equation/make_data.py  --seed 1 --resolution 16 --num_data 100;


printf "\nCollecting data for the workprecision diagram\n"
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 16 --num_runs 1 --method arnoldi;
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 16 --num_runs 1 --method diffrax:euler+backsolve;
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 16 --num_runs 1 --method diffrax:heun+recursive_checkpoint;
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 16 --num_runs 1 --method diffrax:dopri5+backsolve;
time python experiments/applications/partial_differential_equation/workprecision.py  --resolution 16 --num_runs 1 --method diffrax:tsit5+recursive_checkpoint;

printf "\nTraining the network...\n"
time python experiments/applications/partial_differential_equation/train.py  --resolution 16 --method arnoldi;
time python experiments/applications/partial_differential_equation/train.py  --resolution 16 --method diffrax:euler+backsolve;
time python experiments/applications/partial_differential_equation/train.py  --resolution 16 --method diffrax:heun+recursive_checkpoint;
time python experiments/applications/partial_differential_equation/train.py  --resolution 16 --method diffrax:dopri5+backsolve;
time python experiments/applications/partial_differential_equation/train.py  --resolution 16 --method diffrax:tsit5+recursive_checkpoint;
