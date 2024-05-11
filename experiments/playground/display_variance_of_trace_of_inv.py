import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import hutchinson

ndata = 5
xs = jnp.linspace(0, 1, num=ndata)
K = jnp.exp(-0.5 * jnp.abs(xs[:, None] - xs[None, :]) ** 2)


def matvec_inv(v):
    return jnp.linalg.solve(K, v)


def matvec_log(v):
    return jax.scipy.linalg.funm(K, jnp.log) @ v


def matvec(v):
    return K @ v


labels = {matvec_inv: "Inverse", matvec_log: "Log", matvec: "Matrix"}

nums = 3 ** jnp.arange(10.0)
plt.subplots(figsize=(5, 3))
plt.title(f"Trace of a {ndata}x{ndata} RBF kernel matrix")
for mv in [matvec, matvec_log, matvec_inv]:
    means = []
    stds = []
    print(mv)
    for num in nums:
        sampler = hutchinson.sampler_rademacher(xs, num=int(num))
        integrand = hutchinson.integrand_trace(mv)
        estimate = jax.jit(hutchinson.hutchinson(integrand, sampler))

        key = jax.random.PRNGKey(1)
        keys = jax.random.split(key, num=1_000)
        all_ = jax.lax.map(estimate, keys)
        means.append(jnp.mean(all_))
        stds.append(jnp.std(all_))
        print(num, stds)
    print()
    plt.loglog(nums, stds, ".-", label=labels[mv])

plt.grid(axis="both", which="major", linestyle="dotted")
plt.xlabel("No. samples")
plt.ylabel("Std: trace-estimation")
plt.legend()
plt.tight_layout()
plt.show()
