# from tqdm.auto import tqdm
# from time import sleep

# for i in tqdm(range(5), desc="outer"):
#     # inner_tqdm = tqdm()
#     for _ in tqdm(range(10), desc="inner", leave=True):
#         sleep(0.2)

# from fastprogress.fastprogress import master_bar, progress_bar
# from time import sleep
# mb = master_bar(range(10))
# for i in mb:
#     for j in progress_bar(range(100), parent=mb):
#         sleep(0.01)
#         mb.child.comment = f'second bar stat'
#     mb.main_bar.comment = f'first bar stat'
#     mb.write(f'Finished loop {i}.')

from tqdm import tqdm
from time import sleep

outer_loop = 3
inner_loop = 3

pbar_outer = tqdm(total=outer_loop, position=1, leave=True, ncols=80)
pbar_inner = tqdm(total=inner_loop, position=0, leave=False, ncols=80)

for n in range(outer_loop):
    pbar_outer.set_description(f"outer")
    pbar_outer.update()
    sleep(1)

    pbar_inner.reset()
    pbar_inner.set_description(f"inner {n}")
    for m in range(inner_loop):
        pbar_inner.update()
        # tqdm.write(str(m))
        sleep(1)
    tqdm.write(f"done {n}")

pbar_outer.close()