python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "1138_bus" --max_krylov_depth 500 --backprop_until 100 --lanczos_or_arnoldi arnoldi --reortho full --num_runs 1 
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "1138_bus" --max_krylov_depth 500 --backprop_until 100 --lanczos_or_arnoldi arnoldi --reortho full --num_runs 5 --precompile
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "1138_bus" --max_krylov_depth 500 --backprop_until 100 --lanczos_or_arnoldi arnoldi --reortho none --num_runs 1
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "1138_bus" --max_krylov_depth 500 --backprop_until 100 --lanczos_or_arnoldi arnoldi --reortho none --num_runs 5 --precompile

python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "1138_bus" --max_krylov_depth 500 --backprop_until 100 --lanczos_or_arnoldi lanczos --reortho full --num_runs 1
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "1138_bus" --max_krylov_depth 500 --backprop_until 100 --lanczos_or_arnoldi lanczos --reortho full --num_runs 5 --precompile
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "1138_bus" --max_krylov_depth 500 --backprop_until 100 --lanczos_or_arnoldi lanczos --reortho none --num_runs 1
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "1138_bus" --max_krylov_depth 500 --backprop_until 100 --lanczos_or_arnoldi lanczos --reortho none --num_runs 5 --precompile

python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "gyro" --max_krylov_depth 1000 --backprop_until 100 --lanczos_or_arnoldi arnoldi --reortho full --num_runs 1
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "gyro" --max_krylov_depth 1000 --backprop_until 100 --lanczos_or_arnoldi arnoldi --reortho full --num_runs 5 --precompile
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "gyro" --max_krylov_depth 1000 --backprop_until 100 --lanczos_or_arnoldi arnoldi --reortho none --num_runs 1
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "gyro" --max_krylov_depth 1000 --backprop_until 100 --lanczos_or_arnoldi arnoldi --reortho none --num_runs 5 --precompile

python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "gyro" --max_krylov_depth 1000 --backprop_until 100 --lanczos_or_arnoldi lanczos --reortho full --num_runs 1
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "gyro" --max_krylov_depth 1000 --backprop_until 100 --lanczos_or_arnoldi lanczos --reortho full --num_runs 5 --precompile
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "gyro" --max_krylov_depth 1000 --backprop_until 100 --lanczos_or_arnoldi lanczos --reortho none --num_runs 1
python experiments/benchmark_vjp_wall_times/suite_sparse/benchmark.py  --which_matrix "gyro" --max_krylov_depth 1000 --backprop_until 100 --lanczos_or_arnoldi lanczos --reortho none --num_runs 5 --precompile