from job_gen import sbatch
import sys

jobstr = """
python3 experiments/runners/run_pedestrian_scale.py cuda $1 20 sequential pmap --no_progress --no_colors
"""

sbatch("CPU", "ped_gpu", int(sys.argv[1]), jobstr)