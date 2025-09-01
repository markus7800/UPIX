from job_gen import sbatch
import sys

jobstr = """
python3 experiments/runners/run_pedestrian_scale.py cpu $1 20 sequential pmap --no_progress --no_colors
"""

sbatch("CPU", "ped_cpu", int(sys.argv[1]), jobstr)