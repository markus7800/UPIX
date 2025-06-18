import sys
import pickle
import jax
import jax.export
from typing import IO

def read_transport_layer(reader: IO[bytes]):
    msg = reader.readline().decode("utf8").rstrip()
    print(f"got msg: {msg}", file=sys.stderr)
    if msg == "":
        return None
    if msg == "close":
        return None
    obj = pickle.load(reader)
    return obj

def write_transport_layer(writer: IO[bytes], obj):
    writer.write("OK\r\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()
    
def write_error_transport_layer(writer: IO[bytes], obj):
    writer.write("ERROR\r\n".encode("utf8"))
    writer.flush()
    pickle.dump(obj, writer)
    writer.flush()

print(f"Starting worker.", file=sys.stderr)
while True:
    obj = read_transport_layer(sys.stdin.buffer)
    if obj is None:
        print(f"Terminating worker.", file=sys.stderr)
        break
    try:
        jax_serialised_fn, args = obj 
        jax_fn = jax.export.deserialize(jax_serialised_fn)
        # assert False, "Oopsie"
        out = jax_fn.call(*args) # this will always compile
        write_transport_layer(sys.stdout.buffer, out)
        del obj
        del out
        del jax_fn
    except Exception as e:
        write_error_transport_layer(sys.stdout.buffer, str(e))