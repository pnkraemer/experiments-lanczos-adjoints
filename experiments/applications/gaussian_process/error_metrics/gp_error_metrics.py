"""
Script for error metrics of GP models on UCI (test) datasets.

Currently, use either of the following datasets:
* concrete_compressive_strength  (small)
* combined_cycle_power_plant  (medium)
* ___________________________   (large)
* ___________________________   (very large)

These are the GP methods/solvers available:
* naive
* gpytorch
* adjoints
* ___________________________
"""

import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util

font = {"family": "serif", "size": 35}

plt.rc("text", usetex=True)
plt.rc("font", **font)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

# COOLORS.CO palettes
color_palette_1 = ["#335c67", "#fff3b0", "#e09f3e", "#9e2a2b", "#540b0e"]
color_palette_2 = ["#177e89", "#084c61", "#db3a34", "#ef8354", "#323031"]
color_palette_3 = ["#bce784", "#5dd39e", "#348aa7", "#525274", "#513b56"]
color_palette_4 = ["#002642", "#840032", "#e59500", "#e5dada", "#02040e"]
color_palette_5 = ["#202c39", "#283845", "#b8b08d", "#f2d449", "#f29559"]
color_palette_6 = ["#21295c", "#1b3b6f", "#065a82", "#1c7293", "#9eb3c2"]
color_palette_7 = ["#f7b267", "#f79d65", "#f4845f", "#f27059", "#f25c54"]
color_palette_10 = [
    "#001219",
    "#005F73",
    "#0A9396",
    "#94D2BD",
    "#E9D8A6",
    "#EE9B00",
    "#CA6702",
    "#BB3E03",
    "#AE2012",
    "#9B2226",
]

palette_red = [
    "#03071e",
    "#370617",
    "#6a040f",
    "#9d0208",
    "#d00000",
    "#dc2f02",
    "#e85d04",
    "#f48c06",
    "#faa307",
    "#ffba08",
]
palette_blue = [
    "#012a4a",
    "#013a63",
    "#01497c",
    "#014f86",
    "#2a6f97",
    "#2c7da0",
    "#468faf",
    "#61a5c2",
    "#89c2d9",
    "#a9d6e5",
]
palette_green = [
    "#99e2b4",
    "#88d4ab",
    "#78c6a3",
    "#67b99a",
    "#56ab91",
    "#469d89",
    "#358f80",
    "#248277",
    "#14746f",
    "#036666",
]
palette_pink = [
    "#ea698b",
    "#d55d92",
    "#c05299",
    "#ac46a1",
    "#973aa8",
    "#822faf",
    "#6d23b6",
    "#6411ad",
    "#571089",
    "#47126b",
]
palette_super_red = [
    "#641220",
    "#6e1423",
    "#85182a",
    "#a11d33",
    "#a71e34",
    "#b21e35",
    "#bd1f36",
    "#c71f37",
    "#da1e37",
    "#e01e37",
]

palettes = [palette_red, palette_pink, palette_blue, palette_green]

palette_9_colors = [
    "#54478C",
    "#2C699A",
    "#048BA8",
    "#0DB39E",
    "#16DB93",
    "#83E377",
    "#B9E769",
    "#EFEA5A",
    "#F1C453",
    "#F29E4C",
]
palette_20_colors = [
    "#001219",
    "#005f73",
    "#0a9396",
    "#94d2bd",
    "#e9d8a6",
    "#ee9b00",
    "#ca6702",
    "#bb3e03",
    "#ae2012",
    "#9b2226",
    "#f72585",
    "#b5179e",
    "#7209b7",
    "#560bad",
    "#480ca8",
    "#3a0ca3",
    "#3f37c9",
    "#4361ee",
    "#4895ef",
    "#4cc9f0",
]

palette = palette_20_colors
##### This is a draft version for quickly reading the data

GP_METHODS_ARGS = ["adjoints", "naive", "gpytorch"]
ADJOINTS_KRY_ARGS = [1, 5, 10, 50, 100]


def load_params_and_curves(args):
    dir_local = exp_util.matching_directory(__file__, "results/")
    os.makedirs(dir_local, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    do_plot = False
    do_naive = True
    i = 0

    for gpm in GP_METHODS_ARGS:
        for kry in ADJOINTS_KRY_ARGS:
            if args.dataset == "concrete_compressive_strength":
                subdir_data = "concrete/"
            elif args.dataset == "combined_cycle_power_plant":
                subdir_data = "combined/"

            if gpm == "adjoints":
                filename = "adjoints"
                if kry == 1:
                    filename += "_kry1"
                    do_plot = True
                elif kry == 5:
                    filename += "_kry5"
                    do_plot = True
                elif kry == 10:
                    filename += "_kry10"
                    do_plot = True
                elif kry == 50:
                    filename += "_kry50"
                    do_plot = True
                elif kry == 100:
                    filename += "_kry100"
                    do_plot = True

            elif gpm == "naive":
                filename = "naive"
                if i < 7 and do_naive:
                    do_plot = True
                    do_naive = False
            elif gpm == "gpytorch":
                filename = "gpytorch"
                if i < 7:
                    do_plot = True

            if do_plot:
                dir_files = dir_local + subdir_data
                loss_curve = jnp.load(dir_files + "convergence_" + filename + ".npy")
                time_stamps = jnp.load(dir_files + "time_" + filename + ".npy")
                # params = jnp.load(dir_files + "params_" + filename + ".npy")

                plt.plot(
                    time_stamps,
                    loss_curve,
                    color=palette[i],
                    lw=3.0,
                    alpha=0.8,
                    label=filename,
                )
                do_plot = False
                i += 1

    plt.title(r"" + args.dataset)
    plt.ylabel(r"Loss")
    plt.xlabel(r"Run Time (seconds)")
    plt.legend()
    plt.savefig(
        fname=f"{dir_local}/{args.dataset}" + ".pdf", format="pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gp_method", "-gpm", type=str, default="gpytorch")
    parser.add_argument("--slq_krylov_depth", "-kry", type=int, default=5)
    parser.add_argument(
        "--dataset", "-data", type=str, default="concrete_compressive_strength"
    )
    args = parser.parse_args()
    print(args, "\n")

    load_params_and_curves(args)

    # Loading {convergence_curve, convergence time_stamps, optimized parameters}
    # jnp.save(f"{dir_local}/convergence_{args.gp_method}.npy", jnp.array(conv))
    # jnp.save(f"{dir_local}/time_{args.gp_method}.npy", jnp.array(tstamp))
    # jnp.save(f"{dir_local}/params_{args.gp_method}.npy", jnp.array(params))

    # Saving {argparse configuration}
    # save_parser(dir_local, args)
