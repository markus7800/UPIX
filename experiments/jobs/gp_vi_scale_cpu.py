from job_gen import sbatch
import sys

ndevices = int(sys.argv[1])

jobstr = """
python3 experiments/runners/run_gp_vi_scale.py cpu %d 13 sequential smap_local --no_progress
""" % ndevices

sbatch("CPU", "gp_vi_cpu", ndevices, jobstr)