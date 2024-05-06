
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 10 --num_matvecs 10 --method expm-pade;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 10 --num_matvecs 10 --method arnoldi;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 10 --num_matvecs 10 --method euler;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 10 --num_matvecs 10 --method diffrax-euler;
python experiments/applications/partial_differential_equation/tune_wave_equation/tune.py --num_dx_points 10 --num_matvecs 10 --method diffrax-tsit5;
