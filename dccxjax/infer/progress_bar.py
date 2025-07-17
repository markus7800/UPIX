import jax
import jax.experimental
from typing import Optional, Callable
from tqdm.auto import tqdm

__all__ = [
    
]

class ProgressbarManager:
    def __init__(self, desc: str, shared_tqdm_bar: tqdm | None) -> None:
        self.desc = desc
        self.tqdm_bar: Optional[tqdm] = shared_tqdm_bar
        self.share_bar: bool = shared_tqdm_bar is not None
        self.num_samples = 0

    def set_num_samples(self, num_samples: int):
        self.num_samples = num_samples # affects for tqdm bar lenght, not print rate

    def start_progress(self):
        if self.tqdm_bar is None:
            self.tqdm_bar = tqdm(total=self.num_samples, position=0)
        else:
            self.tqdm_bar.reset(total=self.num_samples)
        self.tqdm_bar.set_description(f"Compiling {self.desc}... ", refresh=True)

    def _init_tqdm(self, increment):
        if self.tqdm_bar is None: 
            self.tqdm_bar = tqdm(total=self.num_samples, position=0)
        else:
            self.tqdm_bar.reset(total=self.num_samples)
        increment = int(increment)
        self.tqdm_bar.set_description(f"  Running {self.desc}", refresh=True)
        self.tqdm_bar.update(increment)

    def _update_tqdm(self, iternum, increment, remainder):
        if self.tqdm_bar is not None:
            iternum = int(iternum)
            increment = int(increment)
            remainder = int(remainder)
            if iternum == self.num_samples:
                if remainder == 0:
                    # update and close event happen at same time
                    self.tqdm_bar.update(increment)
                else:
                    self.tqdm_bar.update(remainder)
                # tqdm_auto.write(f"Close pbar {self}")
                if not self.share_bar:
                    self.tqdm_bar.close()
                    self.tqdm_bar = None
            else:
                self.tqdm_bar.update(increment)


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