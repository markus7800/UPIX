from job_gen import sbatch
import sys

ndevices = int(sys.argv[1])

jobstr = """
python3 experiments/runners/run_gp_vi_scale.py cuda %d 12 sequential smap_local --no_progress
""" % ndevices

sbatch("GPU", "gp_vi_gpu", ndevices, jobstr)