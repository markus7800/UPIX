import sys
import pickle
from io import BufferedReader, BufferedWriter
import jax
import jax.export

def read_transport_layer(reader: BufferedReader):
    msg = reader.readline().decode("utf8").rstrip()
    print(f"got msg: {msg}", file=sys.stderr)
    if msg == "":
        return None
    if msg == "close":
        return None
    obj = pickle.load(reader)
    return obj

def write_transport_layer(writer: BufferedWriter, obj):
    writer.write("OK\r\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()

print(f"Starting worker.", file=sys.stderr)
while True:
    obj = read_transport_layer(sys.stdin.buffer) # type: ignore
    if obj is None:
        print(f"Terminating worker.", file=sys.stderr)
        break
    jax_serialised_fn, args = obj 
    jax_fn = jax.export.deserialize(jax_serialised_fn)
    out = jax_fn.call(*args)
    write_transport_layer(sys.stdout.buffer, out) # type: ignore
    del obj
    del out
    del jax_fn