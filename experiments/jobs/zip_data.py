import shutil
import pathlib
from datetime import datetime
from uuid import uuid4
now = datetime.now().strftime("%Y-%m-%d")
uid = str(uuid4())[:8]
shutil.make_archive(
    f"results_data_{now}_{uid}",
    "zip",
    pathlib.Path("experiments", "data")
)