import subprocess
import sys
import pickle
import jax

s = """
import sys
import pickle

def read_transport_layer(reader):
    obj = pickle.load(reader)
    return obj

def write_transport_layer(writer, obj):
    pickle.dump(obj, writer)
    writer.flush()

while True:
    obj = read_transport_layer(sys.stdin.buffer)
    if obj is None:
        break
    write_transport_layer(sys.stdout.buffer, repr(sys.stdin.buffer))
    del obj
"""

def read_transport_layer(reader):
    obj = pickle.load(reader)
    return obj

def write_transport_layer(writer, obj):
    pickle.dump(obj, writer)
    writer.flush()

p = subprocess.Popen(
    [sys.executable,  "-c", s],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
assert p.stdin is not None
assert p.stdout is not None
print(p.stdin)

write_transport_layer(p.stdin, jax.random.normal(jax.random.key(0), (1000,1000,250)))
response = read_transport_layer(p.stdout)
print(response)
# p.stdin.close()
p.wait()
