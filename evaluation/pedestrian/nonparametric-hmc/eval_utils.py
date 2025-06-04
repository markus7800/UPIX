import matplotlib
import seaborn as sns

# Use type 42 (TrueType) fonts instead of the default Type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42

# Make plots more accessible:
sns.set_palette("colorblind")

palette = {
    "ours": "C0",
    "nonparametric-hmc": "C0",
    "LMH": "C2",
    "PGibbs": "C1",
    "RMH": "C3",
    "IPMCMC": "C5",
    "ground truth": "C4",
    "Pyro HMC": "C8",
    "Pyro NUTS": "C9",
}

anglican_methods = ["lmh", "pgibbs", "rmh", "ipmcmc"]  # replace with [] to disable
all_methods = ["hmc", "is"] + anglican_methods
compared_methods = ["hmc", "lmh", "pgibbs", "rmh"] if anglican_methods else ["hmc"]

method_name = {
    "hmc": "ours",
    "is": "IS",
    "lmh": "LMH",
    "ipmcmc": "IPMCMC",
    "pgibbs": "PGibbs",
    "rmh": "RMH",
}


def thin_list(l: list, target_size: int) -> list:
    size = len(l)
    assert size >= target_size
    result = []
    for i in range(target_size):
        result.append(l[i * size // target_size])
    return result


def thin_runs(all_methods: list, runs: list) -> list:
    thinned_runs = []
    for run in runs:
        thinned_runs.append({})
        N = len(run["hmc"]["samples"])
        for method in all_methods:
            thinned_runs[-1][method] = thin_list(run[method]["samples"], N)
    return thinned_runs

def collect_values(all_methods: list, thinned_runs: list) -> dict:
    values = {m: [] for m in all_methods}
    for run in thinned_runs:
        N = len(run["hmc"])
        for method in all_methods:
            values[method] += run[method]
    return values


def collect_chains(all_methods: list, thinned_runs: list) -> dict:
    chains = {m: [] for m in all_methods}
    for run in thinned_runs:
        for method in all_methods:
            chains[method].append(run[method])
    return chains


def print_running_time(all_methods: list, runs: list, thinned_runs: list):
    print("\nRunning times:")
    for method in all_methods:
        running_time = sum(run[method]["time"] for run in runs)
        count = sum(len(run[method]) for run in thinned_runs)
        per_sample = running_time / count
        print(
            f"{method}: {running_time:.2f}s    {per_sample:.4f}s per sample (after thinning)"
        )



from collections import defaultdict
def thin_runs_2(runs: list, burnin: int = 0) -> list:
    thinned_runs = []
    for run in runs:
        thinned_runs.append(defaultdict(list))
        N = min(len(r["samples"][burnin:]) for r in run.values())
        for method in run.keys():
            thinned_runs[-1][method] = thin_list(run[method]["samples"][burnin:], N)
    return thinned_runs

def collect_values_2(thinned_runs: list, config=None) -> dict:
    values = defaultdict(list)
    for run in thinned_runs:
        for method in run.keys():
            if config is None:
                values[method] += run[method]
            else:
                values[(method, config)] += run[method]
    return values


def collect_chains_2(thinned_runs: list, config=None) -> dict:
    chains = defaultdict(list)
    for run in thinned_runs:
        for method in run.keys():
            if config is None:
                chains[method].append(run[method])
            else:
                chains[(method, config)].append(run[method])
    return chains


method_name_2 = {
    "npdhmc": "NP-DHMC",
    "is": "IS",
    "npdhmc-persistent": "NP-DHMC pers.",
    "np-la-dhmc": "NP-Lookahead-DHMC",
    "npladhmc": "NP-Lookahead-DHMC",
    "npladhmc-persistent": "NP-Lookahead-DHMC pers.",
}


def legend_str(config) -> str:
    if type(config) == tuple:
        assert len(config) == 2
        if config[1]:
            return f"{method_name_2[config[0]]} ({config[1]})"
        else:
            return f"{method_name_2[config[0]]}"
    elif config:
        return str(config)
    else:
        return ""
    

def toconfigstr(L, alpha, K):
    alphastr = [] if alpha == 1.0 else [f"α={alpha}"]
    Kstr = [] if K == 0 else [f"K={K}"]
    if L is not None:
        return ", ".join([f"L={L}"] + alphastr + Kstr)
    else:
        return ", ".join(alphastr + Kstr)
    
def print_running_time_2(runs: list, thinned_runs: list):
    for method in runs[0].keys():
        running_time = sum(run[method]["time"] for run in runs)
        count = sum(len(run[method]) for run in thinned_runs)
        per_sample = running_time / count
        print(
            f"{legend_str(method)}: {running_time:.2f}s    {per_sample:.4f}s per sample (after thinning)"
        )
