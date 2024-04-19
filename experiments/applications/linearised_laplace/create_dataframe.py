import pickle

import jax
import pandas as pd
from matfree_extensions import exp_util

# Read the corresponding directory
directory_results = exp_util.matching_directory(__file__, "results/")

# Load the dictionary results
results = {}
for GGN_TYPE in ["full", "diag"]:
    with open(f"{directory_results}/results_{GGN_TYPE}.pkl", "rb") as f:
        label = f"GGN ({GGN_TYPE})"
        results[label] = pickle.load(f)

# Turn shape=() arrays into floats (so the formatter below applies)
results = jax.tree_util.tree_map(float, results)

# Create a data frame
results_dataframe = pd.DataFrame(results).T

# Create a latex-table
latex = results_dataframe.to_latex(column_format="lccccc", float_format="%.2f")
print(latex)
