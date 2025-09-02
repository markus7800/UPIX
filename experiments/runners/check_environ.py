from dccxjax.utils import get_environment_info
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("platform", help="cpu | gpu")
parser.add_argument("ndevices", type=int)
args = parser.parse_args()

info = get_environment_info()

assert args.platform == info["platform"], f"Configured platform = {args.platform} does not match environment platform = {info['platform']}"
assert args.ndevices == info["n_available_devices"], f"Configured device count = {args.ndevices} does not match environment device count = {info['n_available_devices']}"