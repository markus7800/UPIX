import pathlib
import os
for file in os.listdir(pathlib.Path(".")):
    if file.endswith(".out"):
        os.rename(file, pathlib.Path("experiments", "logs", file))