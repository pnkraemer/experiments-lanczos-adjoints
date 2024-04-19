import pickle

import jax
import pandas as pd
from matfree_extensions import exp_util

# Read the corresponding directory
directory_results = exp_util.matching_directory(__file__, "results/")

# Load the dictionary results
results = {}
for ggn in ["full", "diag"]:
    for numerics in ["cholesky", "lanczos"]:
        file_path = f"{directory_results}/results_{ggn}_{numerics}.pkl"
        with open(file_path, "rb") as f:
            label = f"GGN ({ggn} + {numerics})"
            results[label] = pickle.load(f)

# Turn shape=() arrays into floats (so the formatter below applies)
results = jax.tree_util.tree_map(float, results)

# Create a data frame
results_dataframe = pd.DataFrame(results).T

# Create a latex-table
num_keys = len(results[label].keys())
column_format = f"l{'c'*num_keys}"
latex = results_dataframe.to_latex(column_format=column_format, float_format="%.2f")
print(latex)
