import jax
import jax.numpy as jnp
import jax.scipy.linalg

dx = 0.01
dt = 0.1
ts = jnp.arange(0.0, 1.0 + dt, step=dt)
xs = jnp.arange(0.0, 1.0 + dx, step=dx)
# coeff = -0.1

stencil = jnp.asarray([-1, 2.0, -1]) / dx**2


def matvec(x):
    x_padded = jnp.pad(x, 1, mode="constant", constant_values=0.0)
    return jnp.convolve(stencil, x_padded, mode="valid")


y0 = jnp.exp(-100 * (xs - 0.5) ** 2)


def sol(t, coeff):
    A = jax.jacfwd(lambda s: matvec(s))(y0)
    return jax.scipy.linalg.expm(t * A * coeff) @ y0


coeff_true = -0.1
data = sol(1.0, coeff_true) + 0.01 * jax.random.normal(
    jax.random.PRNGKey(1), shape=y0.shape
)


@jax.jit
@jax.value_and_grad
def parameter_to_error(c):
    solution = sol(1.0, c)
    diff = solution - data
    return jnp.dot(diff, diff)


c0 = -1.0
val, grad = parameter_to_error(c0)

while jnp.linalg.norm(grad) > jnp.sqrt(jnp.finfo(val.dtype).eps):
    val, grad = parameter_to_error(c0)
    c0 -= 0.1 * grad
    print(c0)
