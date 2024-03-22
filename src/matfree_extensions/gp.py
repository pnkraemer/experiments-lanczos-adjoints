import jax
import jax.numpy as jnp


def kernel_periodic():
    def parametrize(*, scale_sqrt_in, scale_sqrt_out, period):
        def k(x, y):
            diff = x - y
            scaled = period**2 * jnp.pi * jnp.sqrt(jnp.dot(diff, diff))

            inner_squared = scale_sqrt_in**2 * jnp.sin(scaled) ** 2
            return scale_sqrt_out**2 * jnp.exp(-inner_squared)

        return _vmap_gram(k)

    empty = jnp.empty(())
    params_like = {"scale_sqrt_in": empty, "scale_sqrt_out": empty, "period": empty}
    return parametrize, params_like


def kernel_matern_32():
    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            diff = x - y
            scaled = scale_sqrt_in**2 * jnp.dot(diff, diff)

            # Shift by epsilon to guarantee differentiable sqrts
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return scale_sqrt_out**2 * (1 + sqrt) * jnp.exp(-sqrt)

        return _vmap_gram(k)

    empty = jnp.empty(())
    params_like = {"scale_sqrt_in": empty, "scale_sqrt_out": empty}
    return parametrize, params_like


def kernel_matern_12():
    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            diff = x - y
            scaled = jnp.dot(diff, diff) / (scale_sqrt_in**2)

            # Shift by epsilon to guarantee differentiable sqrts
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return scale_sqrt_out**2 * jnp.exp(-sqrt)

        return _vmap_gram(k)

    empty = jnp.empty(())
    params_like = {"scale_sqrt_in": empty, "scale_sqrt_out": empty}
    return parametrize, params_like


def kernel_quadratic_exponential():
    """Construct a square exponential kernel."""

    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            diff = x - y
            log_k = scale_sqrt_in**2 * jnp.dot(diff, diff)
            return scale_sqrt_out**2 * jnp.exp(-log_k)

        return _vmap_gram(k)

    empty = jnp.empty(())
    params_like = {"scale_sqrt_in": empty, "scale_sqrt_out": empty}
    return parametrize, params_like


def kernel_quadratic_rational():
    """Construct a rational quadratic kernel."""

    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            diff = x - y
            tmp = scale_sqrt_in**2 * jnp.dot(diff, diff)
            return scale_sqrt_out**2 / (1 + tmp)

        return _vmap_gram(k)

    empty = jnp.empty(())
    params_like = {"scale_sqrt_in": empty, "scale_sqrt_out": empty}
    return parametrize, params_like


def _vmap_gram(fun):
    tmp = jax.vmap(fun, in_axes=(0, None), out_axes=0)
    return jax.vmap(tmp, in_axes=(None, 1), out_axes=1)


#
# def condition(parameters, noise_std, /, *, kernel_fun, data, inputs_eval):
#     kernel_fun_p = kernel_fun(**parameters)
#
#     K = kernel_fun_p(data.inputs, data.inputs.T)
#     eye = jnp.eye(len(K))
#     coeffs = jnp.linalg.solve(K + noise_std**2 * eye, data.targets)
#
#     K_eval = kernel_fun_p(inputs_eval, data.inputs.T)
#     means = K_eval @ coeffs
#
#     K_xy = kernel_fun_p(inputs_eval, data.inputs.T)
#     K_xx = kernel_fun_p(inputs_eval, inputs_eval.T)
#     coeffs = jnp.linalg.solve(K + noise_std**2 * eye, K_xy.T)
#
#     covs = K_xx - K_xy @ coeffs
#     return means, covs
#
#
# def condition_mean(parameters, noise_std, /, *, kernel_fun, data, inputs_eval):
#     kernel_fun_p = kernel_fun(**parameters.unravelled)
#
#     K = kernel_fun_p(data.inputs, data.inputs.T)
#     eye = jnp.eye(len(K))
#     coeffs = jnp.linalg.solve(K + noise_std**2 * eye, data.targets)
#
#     K_eval = kernel_fun_p(inputs_eval, data.inputs.T)
#     mean = K_eval @ coeffs
#     return TimeSeriesData(inputs_eval, mean)
#
#
# def condition_std(parameters, noise_std, /, *, kernel_fun, data, inputs_eval):
#     kernel_fun_p = kernel_fun(**parameters.unravelled)
#
#     K = kernel_fun_p(data.inputs, data.inputs.T)
#     eye = jnp.eye(len(K))
#
#     K_xy = kernel_fun_p(inputs_eval, data.inputs.T)
#     K_xx = kernel_fun_p(inputs_eval, inputs_eval.T)
#
#     coeffs = jnp.linalg.solve(K + noise_std**2 * eye, K_xy.T)
#     stds = jnp.sqrt(jnp.diag(K_xx - K_xy @ coeffs))
#
#     return TimeSeriesData(inputs_eval, stds)
