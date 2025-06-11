from multiprocessing.connection import Listener
import jax

# family is the type of socket (or named pipe) to use.
# This can be one of the strings 'AF_INET' (for a TCP socket), 'AF_UNIX' (for a Unix domain socket) or 'AF_PIPE' (for a Windows named pipe).
# Of these only the first is guaranteed to be available.
# If family is None then the family is inferred from the format of address.
# If address is also None then a default is chosen.
# This default is the family which is assumed to be the fastest available.
# See Address Formats. Note that if family is 'AF_UNIX' and address is None then the socket will be created in a private temporary directory created using tempfile.mkstemp().

# Address Formats
# An 'AF_INET' address is a tuple of the form (hostname, port) where hostname is a string and port is an integer.
# An 'AF_UNIX' address is a string representing a filename on the filesystem.
# An 'AF_PIPE' address is a string of the form r'\\.\pipe\PipeName'. To use Client() to connect to a named pipe on a remote computer called ServerName one should use an address of the form r'\\ServerName\pipe\PipeName' instead.
# Note that any string beginning with two backslashes is assumed by default to be an 'AF_PIPE' address rather than an 'AF_UNIX' address.

address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
listener = Listener(address, authkey=b'secret password')
conn = listener.accept()
print('connection accepted from', listener.last_accepted)
while True:
    msg = conn.recv()
    print(type(msg), msg)
    if isinstance(msg, jax.Array):
        print(jax.lax.exp(msg))
    # do something with msg
    if msg == 'close':
        conn.close()
        break
listener.close()