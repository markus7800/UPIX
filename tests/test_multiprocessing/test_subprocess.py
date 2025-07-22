import subprocess
import sys
import time
import threading

# s = """
# import sys
# for line in sys.stdin.readlines():
#     print(line, end="")
#     # sys.stdout.write(line)
#     # sys.stdout.flush()
# """

# s = """
# import sys
# while True:
#     try:
#         s = input()
#         print(s)
#     except EOFError:
#         break
# """

# s = """
# import sys
# s = sys.stdin.buffer.read()
# print(s.decode(), end="")
# """

# s = """
# import sys
# for s in sys.stdin.buffer.read():
#     print(s, end="")
# """

s = """
def read_exactly(stream, n):
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            break
        data += chunk
    return data
import sys
while True:
    s = read_exactly(sys.stdin.buffer, 4)
    if len(s) < 4:
        break
    print(s)
"""

p = subprocess.Popen(
    [sys.executable,  "-c", s],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
assert p.stdin is not None
assert p.stdout is not None

def read_out(p: subprocess.Popen):
    assert p.stdout is not None
    for line in p.stdout.readlines():
        print("OUT", line)
        
t = threading.Thread(target=read_out, args=(p,))
t.start()

p.stdin.write(b"matthew\n")
p.stdin.flush()
p.stdin.write(b"mark\n")
p.stdin.write(b"luke\n")
p.stdin.close()
p.wait()
t.join()