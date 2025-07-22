import subprocess
import sys
from multiprocessing.reduction import ForkingPickler
import jax

s = """
import sys
from multiprocessing.reduction import ForkingPickler
# import jax

def read_transport_layer(reader):
    line = reader.readline().decode('utf8')
    if line == '':
        return None
    key, _, val = line.partition(':')
    assert key == 'Content-Length'
    message_length = int(val)
    obj = ForkingPickler.loads(reader.read(message_length))
    return obj 

def write_transport_layer(writer, obj):
    obj_bytes = ForkingPickler.dumps(obj)
    writer.write(f'Content-Length: {len(obj_bytes)}\\r\\n'.encode('utf8'))
    writer.write(obj_bytes)
    writer.flush()

while True:
    obj = read_transport_layer(sys.stdin.buffer)
    if obj is None:
        break
    write_transport_layer(sys.stdout.buffer, repr(sys.stdin.buffer))
    del obj
"""

def read_transport_layer(reader):
    line = reader.readline().decode('utf8')
    if line == '':
        return None
    key, _, val = line.partition(':')
    assert key == 'Content-Length'
    message_length = int(val)
    obj = ForkingPickler.loads(reader.read(message_length))
    return obj 

def write_transport_layer(writer, obj):
    obj_bytes = ForkingPickler.dumps(obj)
    writer.write(f'Content-Length: {len(obj_bytes)}\r\n'.encode('utf8'))
    writer.write(obj_bytes)
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
p.stdin.close()
p.wait()
