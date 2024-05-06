
# python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 120 --num_matvecs 100 --method expm-pade --num_epochs 50;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 128 --num_matvecs 100 --method arnoldi --num_epochs 250;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 128 --num_matvecs 100 --method diffrax-euler --num_epochs 250;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 128 --num_matvecs 100 --method diffrax-heun --num_epochs 250;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 128 --num_matvecs 100 --method diffrax-tsit5 --num_epochs 250;
