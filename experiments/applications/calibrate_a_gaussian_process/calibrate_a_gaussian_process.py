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

#
#
#
#
# def sample(key_, p, *, kfun, inputs):
#     k_partial = parametrize(kfun, p["scale_in"], p["scale_out"])
#     k_vectorised = vect(k_partial)
#
#     K = k_vectorised(inputs[..., None], inputs[None, ...])
#     K += observation_noise * jnp.eye(len(K))
#
#     xi = jax.random.normal(key_, shape=inputs.shape)
#     return jnp.linalg.solve(K, xi)
#     pass


def create_loss(inputs, targets, kfun, unflatten_p):
    def loss_fun(p_flat):
        params_ = unflatten_p(p_flat)
        return parameter_to_solution(**params_)

    def parameter_to_solution(observation_noise, **p):
        k_partial = kfun(**p)
        k_vectorised = vect(k_partial)
        return log_likelihood(kernel=k_vectorised, observation_noise=observation_noise)

    def log_likelihood(kernel, observation_noise):
        K = kernel(inputs[..., None], inputs[None, ...])
        shift = observation_noise * jnp.eye(len(K))

        coeffs = jnp.linalg.solve(K + shift, targets)
        mahalanobis = jnp.dot(targets, coeffs)

        _sign, entropy = jnp.linalg.slogdet(K)
        return -(mahalanobis + entropy), (coeffs, jnp.linalg.cond(K))

    return loss_fun


def vect(fun):
    tmp = jax.vmap(fun, in_axes=(0, None), out_axes=0)
    return jax.vmap(tmp, in_axes=(None, 1), out_axes=1)


# Parameters
params_true = {"observation_noise": 1e-3, "scale_in": 1e1, "scale_out": 1e-1}
params_true_flat, unflatten_p = jax.flatten_util.ravel_pytree(params_true)
kernel = gp.kernel_quadratic_rational()


# Training data
key = jax.random.PRNGKey(1)
xs = jnp.linspace(0, 1, num=100, endpoint=True)

ys = xs**2
# ys = sample(key, params_true, kfun=k, inputs=xs)


# Parametrize and condition
params = {"observation_noise": 1.0, "scale_in": 1e0, "scale_out": 1e0}
params_flat, _ = jax.flatten_util.ravel_pytree(params)
loss = create_loss(xs, ys, kernel, unflatten_p)
loss_value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

# Train
num_steps = 1_000
for i in range(num_steps):
    (score, (_coeff, cond)), grad = loss_value_and_grad(params_flat)
    params_flat += 0.01 * grad

    if i % (num_steps // 10) == 0:
        print("index =", i)
        print("\tscore =", score)
        print("\tcond =", jnp.log10(cond))
        print("\tparams =", unflatten_p(params_flat))
        print()

# Read the solution
_score, (coeff, _cond) = loss(params_flat)
parameters = unflatten_p(params_flat)
scale_in, scale_out = parameters["scale_in"], parameters["scale_out"]


# Evaluate the solution
k_p = kernel(scale_in=scale_in, scale_out=scale_out)
k_vect = vect(k_p)
xs_new = jnp.linspace(0, 1, num=33, endpoint=True)
ys_new = k_vect(xs_new[:, None], xs[None, :]) @ coeff

# Plot
plt.title(f"Score={jnp.round(score, 2):2f}")
plt.plot(xs, ys, label="Truth")
plt.plot(xs_new, ys_new, label="Estimate")
plt.xlim((jnp.amin(xs), jnp.amax(xs)))
plt.legend()
plt.show()
