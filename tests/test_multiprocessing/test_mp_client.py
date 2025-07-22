from multiprocessing.connection import Client
import jax.numpy as jnp

def f():
    return 1

address = ('localhost', 6000)
conn = Client(address)
# uses pickle
# conn.send(f)
conn.send(jnp.arange(10,dtype=float))
conn.send('close')
conn.close()