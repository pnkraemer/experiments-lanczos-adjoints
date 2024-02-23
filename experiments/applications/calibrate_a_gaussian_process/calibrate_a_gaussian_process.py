# todo: make multi-dimensional
# todo: make matrix-free


import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matfree import hutchinson, lanczos
from matfree_extensions import gp


def solve(xs_, p_flat):
    parameters = unflatten_p(p_flat)
    p_obs = parameters["observation"]
    p_kernel = parameters["kernel"]
    kernel_calibrated = kernel(**p_kernel)
    cond = gp.process_condition(xs, ys, kernel=kernel_calibrated, noise=p_obs["noise"])
    mean_cond, cov_cond = cond
    return mean_cond(xs_)


# Parameters
params_obs = {"noise": 1e-5}
params_kernel = {"scale_in": 1e1, "scale_out": 1e0}
params_true = {"observation": params_obs, "kernel": params_kernel}
kernel = gp.kernel_quadratic_rational(gram_matrix=True)


# Training data
key_ = jax.random.PRNGKey(2)
key_data, key_slq = jax.random.split(key_, num=2)
xs = jnp.linspace(0, 1, num=100, endpoint=True)[..., None]
ys = gp.process_sample(
    key_data, **params_obs, inputs=xs, kernel=kernel(**params_kernel)
)


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

    def solve_fun(M, v):
        result, _info = jax.scipy.sparse.linalg.cg(lambda s: M @ s, v)
        return result

    def slogdet_fun(M):
        x_like = jnp.ones((len(M),))
        krylov_depth = 2
        integrand = lanczos.integrand_spd(jnp.log, krylov_depth, lambda s: M @ s)
        sampler = hutchinson.sampler_rademacher(x_like, num=1)
        estimate = hutchinson.hutchinson(integrand, sampler)
        logdet = estimate(key_slq)
        return jnp.sign(logdet), jnp.abs(logdet)

    score, _coeff = gp.log_likelihood(
        xs, ys, kernel=kernel_p, solve_fun=solve_fun, slogdet_fun=slogdet_fun, **obs_p
    )
    return score


# Initial guess:
xs_new = jnp.linspace(0, 1, num=200, endpoint=True)[..., None]
ys_new_init = solve(xs_new, params_flat)

# Train
learning_rate = 1e-3
optimizer = optax.sgd(learning_rate)
opt_state = optimizer.init(params_flat)

num_steps = 100
for i in range(num_steps):
    score, grad = loss_value_and_grad(params_flat)

    updates, opt_state = optimizer.update(grad, opt_state)
    params_flat = optax.apply_updates(params_flat, updates)

    if i % (num_steps // 10) == 0:
        print("index =", i)
        print("\tscore =", score)
        print("\tparams =", unflatten_p(params_flat))
        print()


# Read the final solution
ys_new_final = solve(xs_new, params_flat)

# Plot
plt.plot(xs, ys, "x", label="Truth")
plt.plot(xs_new, ys_new_init, label="Initial estimate")
plt.plot(xs_new, ys_new_final, label="Final estimate")
plt.xlim((jnp.amin(xs), jnp.amax(xs)))
plt.ylim((jnp.amin(ys), jnp.amax(ys)))
plt.legend()
plt.show()
