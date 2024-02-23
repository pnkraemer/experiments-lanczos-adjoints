# todo: make multi-dimensional
# todo: use a proper optimizer
# todo: plot initial guess and final guess
# todo: write a plotting function (learned from the PDE mess...)
# todo: use a different kernel
# todo: make matrix-free
# todo: make the target a sample from a the GP with known parameters


import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions import gp

# Parameters
params_obs = {"noise": 1e-7}
params_kernel = {"scale_in": 1e1, "scale_out": 1e0}
params_true = {"observation": params_obs, "kernel": params_kernel}
kernel = gp.kernel_quadratic_rational(gram_matrix=True)


# Training data
key = jax.random.PRNGKey(2)
xs = jnp.linspace(0, 1, num=100, endpoint=True)[..., None]
ys = gp.process_sample(key, **params_obs, inputs=xs, kernel=kernel(**params_kernel))


# Parametrize and condition
params_init1 = {"noise": 1.0}
params_init2 = {"scale_in": 1e1, "scale_out": 1e-1}
params = {"observation": params_init1, "kernel": params_init2}
params_flat, unflatten_p = jax.flatten_util.ravel_pytree(params)


@jax.jit
@jax.value_and_grad
def loss_value_and_grad(p_flat):
    params_ = unflatten_p(p_flat)
    kernel_p = kernel(**params_["kernel"])
    obs_p = params_["observation"]
    score, _coeff = gp.log_likelihood(xs, ys, kernel=kernel_p, **obs_p)
    return -score


# Train
num_steps = 110
for i in range(num_steps):
    score, grad = loss_value_and_grad(params_flat)
    params_flat += 0.001 * grad

    if i % (num_steps // 10) == 0:
        print("index =", i)
        print("\tscore =", score)
        print("\tparams =", unflatten_p(params_flat))
        print()


# Condition
# Read the solution
parameters = unflatten_p(params_flat)
params_obs = parameters["observation"]
params_kernel = parameters["kernel"]
kernel_calibrated = kernel(**params_kernel)
cond = gp.process_condition(xs, ys, kernel=kernel_calibrated, noise=params_obs["noise"])
mean_cond, cov_cond = cond

# Evaluate

xs_new = jnp.linspace(0, 1, num=200, endpoint=True)[..., None]
ys_new = mean_cond(xs_new)


# Plot
plt.plot(xs, ys, "x", label="Truth")
plt.plot(xs_new, ys_new, label="Estimate")
plt.xlim((jnp.amin(xs), jnp.amax(xs)))
plt.legend()
plt.show()
