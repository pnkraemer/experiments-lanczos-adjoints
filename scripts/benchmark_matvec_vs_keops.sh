
# Small-scale runs

for i in {1..8}; do \
    python experiments/applications/gaussian_process/benchmark_matvecs_versus_keops/benchmark_matvec.py --num_runs 3 --log_data_size 6 --data_dim $i ;
done

for i in {1..10}; do \
    python experiments/applications/gaussian_process/benchmark_matvecs_versus_keops/benchmark_matvec.py --num_runs 3 --log_data_size $i --data_dim 1 ;
done



# Large-scale runs

for i in {1..15}; do \
    python experiments/applications/gaussian_process/benchmark_matvecs_versus_keops/benchmark_matvec.py --num_runs 3 --log_data_size 16 --data_dim $i ;
done

for i in {10..19}; do \
    python experiments/applications/gaussian_process/benchmark_matvecs_versus_keops/benchmark_matvec.py --num_runs 3 --log_data_size $i --data_dim 1 ;
    python experiments/applications/gaussian_process/benchmark_matvecs_versus_keops/benchmark_matvec.py --num_runs 3 --log_data_size $i --data_dim 2 ;
    python experiments/applications/gaussian_process/benchmark_matvecs_versus_keops/benchmark_matvec.py --num_runs 3 --log_data_size $i --data_dim 4 ;
done
