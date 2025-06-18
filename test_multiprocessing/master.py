from io import BufferedReader, BufferedWriter
import subprocess
import sys
import jax
import jax.export
import jax.numpy as jnp
import pickle

@jax.jit
def step(carry, any):
  val, key = carry
  key1, key2 = jax.random.split(key)
  val = val + jax.random.normal(key1)
  return (val,key2), None
  
@jax.jit
def f(seed):
  return jax.lax.scan(step, (jnp.array(0.,float), seed), length=10**7)[0][0]


def read_transport_layer(reader: BufferedReader):
    msg = reader.readline().decode("utf8").rstrip()
    print(msg)
    if msg != "OK":
        raise Exception
    obj = pickle.load(reader)
    return obj

def write_task_transport_layer(writer: BufferedWriter, obj):
    writer.write("task\r\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()

def write_close_transport_layer(writer: BufferedWriter):
    writer.write("close\r\n".encode("utf8"))
    writer.flush()
    

p = subprocess.Popen(
    [sys.executable,  "test_multiprocessing/worker.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
assert p.stdin is not None
assert p.stdout is not None


exported_fn = jax.export.export(f)(jax.random.PRNGKey(0))

write_task_transport_layer(p.stdin, (exported_fn.serialize(), (jax.random.PRNGKey(1),))) # type: ignore
response = read_transport_layer(p.stdout) # type: ignore
print(response)
write_close_transport_layer(p.stdin) # type: ignore
# p.stdin.close()
p.wait()
