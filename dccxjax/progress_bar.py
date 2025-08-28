import jax
import jax.experimental
from typing import Optional, Callable, TypeVar, Tuple
from tqdm.auto import tqdm
import threading
from dccxjax.types import IntArray
from time import time_ns

__all__ = [
    
]
class ProgressbarManager:
    def __init__(self, desc: str, shared_tqdm_bar: tqdm | None, thread_locked: bool = False) -> None:
        self.desc = desc
        self.tqdm_bar: Optional[tqdm] = shared_tqdm_bar
        self.share_bar: bool = shared_tqdm_bar is not None
        self.num_samples = 0
        self.thread_locked = thread_locked
        self.close_msg_counter = 0
        self.msgs_per_update = 1 # pmap sends msg for each thread of pmap
        self.iternums = [0]
        if self.thread_locked:
            self._lock = threading.Lock()

    def set_num_samples(self, num_samples: int):
        self.num_samples = num_samples # affects only tqdm bar length, not print rate

    def set_msgs_per_update(self, msgs_per_update: int):
        self.msgs_per_update = msgs_per_update
        self.close_msg_counter = 0
        self.iternums = [0] * self.msgs_per_update

    def start_progress(self):
        if self.tqdm_bar is None:
            self.tqdm_bar = tqdm(total=self.num_samples, position=0)
        self.tqdm_bar.set_description(f"Compiling {self.desc}", refresh=False)
        self.tqdm_bar.reset(total=self.num_samples)
        
    def _maybe_with_lock(self, func: Callable):
        if self.thread_locked:
            with self._lock:
                func()
        else:
            func()
        
    def _update_bar(self, iternum):
        def _update_fn():
            if self.tqdm_bar is not None:
                # keep track of minimum iternum (slowest thread of pmap)
                min_iter_num = float("inf")
                did_update = False
                for i in range(len(self.iternums)):
                    if not did_update and self.iternums[i] < iternum:
                        did_update = True
                        self.iternums[i] = iternum
                    if self.iternums[i] < min_iter_num:
                        min_iter_num = self.iternums[i]                    
                if self.tqdm_bar.n < min_iter_num:
                    self.tqdm_bar.update(min_iter_num - self.tqdm_bar.n)
        self._maybe_with_lock(_update_fn)
    
    def _close_bar(self):
        def _close_fn():
            self.close_msg_counter += 1
            # tqdm.write(f"_close_fn {self.msg_counter}/{self.msgs_per_update}")
            if self.close_msg_counter == self.msgs_per_update:
                self.close_msg_counter = 0
                self.iternums = [0] * self.msgs_per_update
                if self.tqdm_bar is not None:
                    if not self.share_bar:
                        self.tqdm_bar.close()
                        self.tqdm_bar = None
                    else:
                        pass
        self._maybe_with_lock(_close_fn)
            
    def _init_tqdm(self, iternum):
        iternum = int(iternum)
        def _init_fn():
            if self.tqdm_bar is None: 
                self.tqdm_bar = tqdm(total=self.num_samples, position=0)
            else:
                self.tqdm_bar.reset(total=self.num_samples)
            self.tqdm_bar.set_description(f"  Running {self.desc}", refresh=True)
        self._maybe_with_lock(_init_fn)
        # self._update_bar(iternum, 0)

    def _update_tqdm(self, iternum):
        if self.tqdm_bar is not None:
            iternum = int(iternum)
            if iternum == self.num_samples:
                self._update_bar(iternum)
                # tqdm.write(f"Close pbar {self} {hex(id(self.tqdm_bar))}")
                self._close_bar()
            else:
                self._update_bar(iternum)


# adapted form numpyro/util.py

SCAN_DATA_TYPE = TypeVar("SCAN_DATA_TYPE")
SCAN_RETURN_TYPE = TypeVar("SCAN_RETURN_TYPE")
SCAN_CARRY_TYPE = TypeVar("SCAN_CARRY_TYPE")

def _add_progress_bar(
    step: Callable[[SCAN_CARRY_TYPE,SCAN_DATA_TYPE],Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]],
    get_iternum_fn: Callable[[SCAN_CARRY_TYPE], IntArray],
    progressbar_mngr: ProgressbarManager,
    num_samples: int) -> Callable[[SCAN_CARRY_TYPE,SCAN_DATA_TYPE],Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]]:
    
    print_rate = max(int(num_samples / 100), 1)

    # remainder = num_samples % print_rate

    def _update_progress_bar(iter_num: jax.Array):
        iter_num = iter_num + 1 # all chains are at the same iteration, init iteration=0
        _ = jax.lax.cond(
            (iter_num % print_rate == 0) | (iter_num == num_samples),
            lambda _: jax.experimental.io_callback(progressbar_mngr._update_tqdm, None, iter_num),
            lambda _: None,
            operand=None,
        )
    
    def wrapped_step(carry: SCAN_CARRY_TYPE, data: SCAN_DATA_TYPE) -> Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]:
        result = step(carry, data)
        _update_progress_bar(get_iternum_fn(carry))
        return result
    
    return wrapped_step