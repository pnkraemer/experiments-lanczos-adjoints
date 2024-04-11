import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matfree_extensions import exp_util
import os
import flax.linen 
import tree_math as tm


class MLP(flax.linen.Module):                    # create a Flax Module dataclass
  out_dims: int

  @flax.linen.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))
    x = flax.linen.Dense(8)(x)   
    x = flax.linen.tanh(x)
    x = flax.linen.Dense(self.out_dims)(x)
    return x

def f(x):
    return jnp.sin(x)

if __name__=="__main__":
    # Make directories
    directory = exp_util.matching_directory(__file__, "figures/")
    os.makedirs(directory, exist_ok=True)

    # Create data
    num_data = 100
    x_train = jnp.linspace(0, 2*jnp.pi, num_data)
    eps = jax.random.normal(jax.random.PRNGKey(0), (num_data,)) * 0.05
    y_train = f(x_train) + eps

    # Plot data
    plt.plot(x_train, y_train, 'o')
    plt.savefig(f"{directory}/figure.pdf")

    # Create model
    model = MLP(out_dims=1)
    variables = model.init(jax.random.PRNGKey(42), x_train)
    y = model.apply(variables, x_train)

    # Train model via optax
    def loss_fn(params, x_, y_):
        y_pred = model.apply(params, x_).reshape((-1,))
        return jnp.mean((y_pred - y_)**2, axis=0)
    

    optimizer = optax.sgd(1e-1)
    optimizer_state = optimizer.init(variables)
    n_epochs = 200
    key = jax.random.PRNGKey(412576)
    batch_size = num_data
    for epoch in range(n_epochs):
        # Subsample data
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, x_train.shape[0], (batch_size,), replace=False)

        loss, grad = jax.value_and_grad(loss_fn)(variables, x_train[idx], y_train[idx])
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        variables = optax.apply_updates(variables, updates)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss {loss:.3f}")
            y = model.apply(variables, x_train)

    # Plot predictions
    x_val = jnp.linspace(0, 2*jnp.pi, 1000)
    y_pred = model.apply(variables, x_val)
    plt.plot(x_val, f(x_val))
    plt.plot(x_val, y_pred)
    plt.savefig(f"{directory}/predixcyions.pdf")
    

