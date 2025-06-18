import os
import re

all = [
    "set_platform",
    "set_host_device_count",
]

# adapted from numpyro

def set_platform(platform: str) -> None:
    assert platform.lower() in ("cpu", "gpu", "tpu")
    os.environ["JAX_PLATFORMS"] = platform.lower()

def set_host_device_count(n: int) -> None:
    xla_flags_str = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags_str
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags
    )