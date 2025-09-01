from job_gen import sbatch
import sys

jobstr = """
python3 experiments/runners/run_gp_vi_scale.py cpu $1 12 sequential smap_local --no_progress --no_colors
"""

sbatch("CPU", "gp_vi_cpu", int(sys.argv[1]), jobstr)