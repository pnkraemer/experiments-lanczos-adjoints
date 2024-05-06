
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 100 --num_matvecs 100 --method expm-pade --num_epochs 50;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 100 --num_matvecs 50 --method euler --num_epochs 25;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 100 --num_matvecs 50 --method arnoldi --num_epochs 25;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 100 --num_matvecs 50 --method diffrax-euler --num_epochs 25;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 100 --num_matvecs 50 --method diffrax-tsit5 --num_epochs 25;
