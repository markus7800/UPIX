import pathlib
import os
pathlib.Path("experiments", "data", "logs").mkdir(exist_ok=True)
for file in os.listdir(pathlib.Path(".")):
    if file.endswith(".out"):
        os.rename(file, pathlib.Path("experiments", "data", "logs", file))