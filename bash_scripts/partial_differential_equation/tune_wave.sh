printf "\nGenerating data... TODO\n"


printf "\nCollecting data for the workprecision diagram\n"
time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 64 --num_runs 1 --method arnoldi;
time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 64 --num_runs 1 --method diffrax:euler+backsolve;
time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 64 --num_runs 1 --method diffrax:heun+recursive_checkpoint;
time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 64 --num_runs 1 --method diffrax:dopri5+backsolve;
time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 64 --num_runs 1 --method diffrax:tsit5+recursive_checkpoint;

printf "\nTraining the network... TODO\n"
