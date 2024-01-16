import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import hutchinson, lanczos

from matfree_extensions import exp_util

PATH = "./data/matrices/"

# nzbounds = (5_000, 15)
nzbounds = (None, 10_000)
sizebounds = (1_000, None)
exp_util.suite_sparse_download(
    path=PATH,
    limit=15,
    isspd=True,
    nzbounds=nzbounds,
    rowbounds=sizebounds,
    colbounds=sizebounds,
)

# matrices  = ["bcsstm09", "bcsstm21", "t2dal_e", "1138_bus"]
matrices = ["t2dal_e"]
fig, axes = plt.subplot_mosaic(
    [matrices], constrained_layout=True, figsize=(len(matrices) * 3, 3)
)


# with jax.ensure_compile_time_eval():

for matrix in matrices:
    M = exp_util.suite_sparse_load(matrix, path=PATH)
    print(M)
    print(M)

    nrows, ncols = M.shape
    v_like = jnp.ones((ncols,), dtype=float)

    sampler = hutchinson.sampler_normal(v_like, num=30_000)
    with jax.profiler.trace("/tmp/tensorboard"):
        x = sampler(jax.random.PRNGKey(1))
        x.block_until_ready()

    problem = hutchinson.integrand_trace(lambda v: v)
    sample = hutchinson.hutchinson(problem, sampler)
    result = jax.jit(sample)(jax.random.PRNGKey(1)).block_until_ready()

    print(result)
