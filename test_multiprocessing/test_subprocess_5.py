import subprocess
import sys
import time
import threading


s = """
      
import sys

def read_transport_layer(reader):
    line = reader.readline().decode('utf8')
    if line == '':
        return None
    key, _, val = line.partition(':')
    assert key == 'Content-Length'
    message_length = int(val)
    message_str = reader.read(message_length).decode('utf8')
    return message_str 

def write_transport_layer(writer, response):
    response_utf8 = response.encode('utf8')
    writer.write(f'Content-Length: {len(response_utf8)}\\r\\n'.encode('utf8'))
    writer.write(response_utf8)
    writer.flush()

while True:
    msg = read_transport_layer(sys.stdin.buffer)
    if msg is None:
        break
    write_transport_layer(sys.stdout.buffer, f"{msg} read")
"""

def read_transport_layer(reader):
    line = reader.readline().decode('utf8')
    if line == '':
        return None
    key, _, val = line.partition(':')
    assert key == 'Content-Length'
    message_length = int(val)
    message_str = reader.read(message_length).decode('utf8')
    return message_str 

def write_transport_layer(writer, response):
    response_utf8 = response.encode('utf8')
    writer.write(f'Content-Length: {len(response_utf8)}\r\n'.encode('utf8'))
    writer.write(response_utf8)
    writer.flush()

p = subprocess.Popen(
    [sys.executable,  "-c", s],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
assert p.stdin is not None
assert p.stdout is not None

# msg = b"matthew"
# l = len(msg)
# p.stdin.write(f"Content-Length: {l}\r\n".encode("utf8"))
# p.stdin.write(msg)
write_transport_layer(p.stdin, "matthew")
response = read_transport_layer(p.stdout)
print(response)
# p.stdin.close()
p.wait()
