# todo: make multi-dimensional
# todo: make matrix-free
# todo: use a proper adjoint


import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matfree import hutchinson
from matfree_extensions import arnoldi, gp


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
xs_1d = jnp.linspace(0, 1, num=10, endpoint=True)
xs = jnp.stack(jnp.meshgrid(xs_1d, xs_1d)).reshape((2, -1)).T


key_ = jax.random.PRNGKey(2)
key_data, key_slq = jax.random.split(key_, num=2)
ys = gp.process_sample(
    key_data, **params_obs, inputs=xs, kernel=kernel(**params_kernel)
)


# Parametrize and condition
params_init1 = {"noise": 1.0}
params_init2 = {"scale_in": 1e1, "scale_out": 1e-1}
params = {"observation": params_init1, "kernel": params_init2}
params_flat, unflatten_p = jax.flatten_util.ravel_pytree(params)


krylov_depth = 2


def slq_integrand(v, p):
    alg = arnoldi.arnoldi(
        lambda s, x: s @ x, krylov_depth, reortho="full", custom_vjp=True
    )
    Q, H, _, c = alg(v, p)
    eigvals, eigvecs = jnp.linalg.eigh((H + H.T) / 2)

    return c * eigvecs[0, :] @ jnp.diag(jnp.log(eigvals)) @ eigvecs[0, :].T


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
        # integrand = lanczos.integrand_spd(jnp.log, krylov_depth, lambda s: M @ s)
        sampler = hutchinson.sampler_rademacher(x_like, num=10)
        estimate = hutchinson.hutchinson(slq_integrand, sampler)
        logdet = estimate(key_slq, M)
        return jnp.sign(logdet), jnp.abs(logdet)

    score, _coeff = gp.log_likelihood(
        xs, ys, kernel=kernel_p, solve_fun=solve_fun, slogdet_fun=slogdet_fun, **obs_p
    )
    return score


# Initial guess:
xs_new_1d = jnp.linspace(0, 1, num=11, endpoint=True)
xs_new = jnp.stack(jnp.meshgrid(xs_new_1d, xs_new_1d)).reshape((2, -1)).T
ys_new_init = solve(xs_new, params_flat)

# Train
learning_rate = 1e-3
optimizer = optax.sgd(learning_rate)
opt_state = optimizer.init(params_flat)

num_steps = 3
for i in range(num_steps):
    score, grad = loss_value_and_grad(params_flat)

    updates, opt_state = optimizer.update(grad, opt_state)
    params_flat = optax.apply_updates(params_flat, updates)

    print("index =", i)
    print("\tscore =", score)
    print("\tparams =", unflatten_p(params_flat))
    print()


# Read the final solution
ys_new_final = solve(xs_new, params_flat)


def fl(s, i):
    return s.reshape((i, i))


# Plot
fig, axes = plt.subplots(
    ncols=3, figsize=(9, 2), dpi=150, constrained_layout=True, sharex=True, sharey=True
)

fig.suptitle(f"N={len(xs)}, Krylov-depth={krylov_depth}")

axes[0].set_title("Initial estimate")
axes[0].contourf(
    fl(xs_new[:, 0], len(xs_new_1d)),
    fl(xs_new[:, 1], len(xs_new_1d)),
    fl(ys_new_init, len(xs_new_1d)),
)

axes[1].set_title("Final estimate")
axes[1].contourf(
    fl(xs_new[:, 0], len(xs_new_1d)),
    fl(xs_new[:, 1], len(xs_new_1d)),
    fl(ys_new_final, len(xs_new_1d)),
)

axes[2].set_title("Truth")
axes[2].contourf(fl(xs[:, 0], len(xs_1d)), fl(xs[:, 1], len(xs_1d)), fl(ys, len(xs_1d)))

plt.show()
