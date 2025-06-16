import multiprocessing as mp

def foo(q):
    q.put('hello')

# popen_spawn_posix.py
# cmd = spawn.get_command_line(tracker_fd=tracker_fd,
#                                 pipe_handle=child_r)
# [b'/Users/markus/Documents/DCCxJAX/venv/bin/python3', '-c', 'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=7, pipe_handle=11)', '--multiprocessing-fork']

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()