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
        if self.thread_locked:
            self._lock = threading.Lock()

    def set_num_samples(self, num_samples: int):
        self.num_samples = num_samples # affects only tqdm bar length, not print rate

    def start_progress(self):
        if self.tqdm_bar is None:
            self.tqdm_bar = tqdm(total=self.num_samples, position=0)
        # tqdm.write(f"start_progress {self} {time_ns()/10**9}")
        self.tqdm_bar.set_description(f"Compiling {self.desc}... ", refresh=False)
        self.tqdm_bar.reset(total=self.num_samples)
        
    def _update_bar(self, iternum, n):
        if self.tqdm_bar is not None:
            if self.thread_locked:
                with self._lock:
                    if self.tqdm_bar.n < iternum:
                        self.tqdm_bar.update(n)
            else:
                if self.tqdm_bar.n < iternum:
                    self.tqdm_bar.update(n)

    def _init_tqdm(self, iternum):
        if self.tqdm_bar is None: 
            self.tqdm_bar = tqdm(total=self.num_samples, position=0)
        else:
            self.tqdm_bar.reset(total=self.num_samples)
        # tqdm.write(f"init_tqdm {time_ns()/10**9}")
        iternum = int(iternum)
        self.tqdm_bar.set_description(f"  Running {self.desc}", refresh=True)
        self._update_bar(iternum, 0)

    def _update_tqdm(self, iternum, increment, remainder):
        if self.tqdm_bar is not None:
            iternum = int(iternum)
            increment = int(increment)
            remainder = int(remainder)
            if iternum == self.num_samples:
                if remainder == 0:
                    # update and close event happen at same time
                    self._update_bar(iternum, increment)
                else:
                    self._update_bar(iternum, remainder)
                # tqdm.write(f"Close pbar {self}")
                if not self.share_bar:
                    self.tqdm_bar.close()
                    self.tqdm_bar = None
                else:
                    pass
                    # self.tqdm_bar.clear()
                    # tqdm.clear(self.tqdm_bar)
            else:
                self._update_bar(iternum, increment)


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

    remainder = num_samples % print_rate

    def _update_progress_bar(iter_num: jax.Array):
        iter_num = iter_num + 1 # all chains are at the same iteration, init iteration=0
        _ = jax.lax.cond(
            (iter_num % print_rate == 0) | (iter_num == num_samples),
            lambda _: jax.experimental.io_callback(progressbar_mngr._update_tqdm, None, iter_num, print_rate, remainder),
            lambda _: None,
            operand=None,
        )
    
    def wrapped_step(carry: SCAN_CARRY_TYPE, data: SCAN_DATA_TYPE) -> Tuple[SCAN_CARRY_TYPE,SCAN_RETURN_TYPE]:
        result = step(carry, data)
        _update_progress_bar(get_iternum_fn(carry))
        return result
    
    return wrapped_step