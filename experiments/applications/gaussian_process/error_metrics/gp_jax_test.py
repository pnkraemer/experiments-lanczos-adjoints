import os.path
import urllib.request

import linear_operator
import jax.numpy as jnp 
import jax 
from scipy.io import loadmat

if not os.path.isfile("../3droad.mat"):
    print("Downloading '3droad' UCI dataset...")
    urllib.request.urlretrieve(
        "https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1", "../3droad.mat"
    )

data = jnp.asarray(loadmat("../3droad.mat")["data"])


assert False 
