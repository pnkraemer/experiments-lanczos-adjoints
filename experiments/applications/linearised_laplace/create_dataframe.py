import os
import pickle

import pandas as pd
from matfree_extensions import exp_util

# Make directories
directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)

directory_results = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory_results, exist_ok=True)

# Load dictionary results
results = {}
for GGN_TYPE in ["full", "diag"]:
    with open(f"{directory_results}/results_{GGN_TYPE}.pkl", "rb") as f:
        results[GGN_TYPE] = pickle.load(f)

print(results)

results_dataframe = pd.DataFrame(results).T
print(results_dataframe)
print(results_dataframe.to_latex())
