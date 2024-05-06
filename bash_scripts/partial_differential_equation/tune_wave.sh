
# python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 120 --num_matvecs 100 --method expm-pade --num_epochs 50;
# python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 128 --num_matvecs 100 --method arnoldi --num_epochs 250;
# python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 128 --num_matvecs 100 --method diffrax-euler --num_epochs 250;
# python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 128 --num_matvecs 100 --method diffrax-heun --num_epochs 250;
# python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 128 --num_matvecs 100 --method diffrax-tsit5 --num_epochs 250;


time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 128 --num_runs 1 --method arnoldi --log2_num_matvec_min 1;
time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 128 --num_runs 1 --method diffrax:euler+backsolve --log2_num_matvec_min 1;
time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 128 --num_runs 1 --method diffrax:heun+recursive_checkpoint --log2_num_matvec_min 2;
time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 128 --num_runs 1 --method diffrax:tsit5+backsolve --log2_num_matvec_min 3;
time python experiments/applications/partial_differential_equation/tune_wave_equation/workprecision.py  --resolution 128 --num_runs 1 --method diffrax:dopri5+recursive_checkpoint --log2_num_matvec_min 3;
