import functools

import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt


def f(x, *, p=2):
    return jnp.sin(p * x**2)


def k(x, y, *, scale_in, scale_out):
    diff = x - y
    log_k = scale_in * jnp.dot(diff, diff)
    return scale_out * jnp.exp(log_k)


def create_loss(inputs, targets, kfun, unflatten):
    def loss(params_flat):
        params_ = unflatten(params_flat)
        return parameter_to_solution(**params_)

    def parameter_to_solution(observation_noise, **parameters):
        k_p = parametrize(kfun, **parameters)
        k_vect = vect(k_p)
        return log_likelihood(kernel=k_vect, observation_noise=observation_noise)

    def log_likelihood(kernel, observation_noise):
        K = kernel(inputs[..., None], inputs[None, ...])
        shift = observation_noise * jnp.eye(len(K))

        coeffs = jnp.linalg.solve(K + shift, targets)
        mahalanobis = jnp.dot(targets, coeffs)

        _sign, entropy = jnp.linalg.slogdet(K)
        return mahalanobis + entropy, coeffs

    return loss


def parametrize(fun, **parameters):
    return functools.partial(fun, **parameters)


def vect(fun):
    tmp = jax.vmap(fun, in_axes=(0, None), out_axes=0)
    return jax.vmap(tmp, in_axes=(None, 1), out_axes=1)


# Training data
xs = jnp.linspace(0, 1, num=100, endpoint=True)
ys = f(xs)

# Parametrize and condition

params = {"observation_noise": 1.0, "scale_in": 1e0, "scale_out": 1e0}
params_flat, unflatten = jax.flatten_util.ravel_pytree(params)
loss = create_loss(xs, ys, k, unflatten)
loss_value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

for _ in range(100):
    (score, _coeff), grad = loss_value_and_grad(params_flat)
    params_flat += 0.01 * grad
    print(score)

_score, coeff = loss(params_flat)

parameters = unflatten(params_flat)
scale_in, scale_out = parameters["scale_in"], parameters["scale_out"]


# Evaluate
k_p = parametrize(k, scale_in=scale_in, scale_out=scale_out)
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
