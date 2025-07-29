import jax
import jax.experimental
from typing import Optional, Callable
from tqdm.auto import tqdm
import threading

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
        self.num_samples = num_samples # affects for tqdm bar lenght, not print rate

    def start_progress(self):
        if self.tqdm_bar is None:
            self.tqdm_bar = tqdm(total=self.num_samples, position=0)
        else:
            self.tqdm_bar.reset(total=self.num_samples)
        self.tqdm_bar.set_description(f"Compiling {self.desc}... ", refresh=True)

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
                # tqdm_auto.write(f"Close pbar {self}")
                if not self.share_bar:
                    self.tqdm_bar.close()
                    self.tqdm_bar = None
            else:
                self._update_bar(iternum, increment)


# adapted form numpyro/util.py
def _add_progress_bar(kernel: Callable, progressbar_mngr: ProgressbarManager, num_samples: int) -> Callable:
    print_rate = max(int(num_samples / 100), 1)

    remainder = num_samples % print_rate

    def _update_progress_bar(iter_num: jax.Array):
        # nonlocal t0
        # t0 = time()
        
        iter_num = iter_num + 1 # all chains are at the same iteration, init iteration=0
        _ = jax.lax.cond(
            (iter_num % print_rate == 0) | (iter_num == num_samples),
            lambda _: jax.experimental.io_callback(progressbar_mngr._update_tqdm, None, iter_num, print_rate, remainder),
            lambda _: None,
            operand=None,
        )
    
    def wrapped_kernel(state, data):
        result = kernel(state, data)
        _update_progress_bar(state.iteration) # NOTE: we don't have to return something for this to work?
        return result
    
    return wrapped_kernel