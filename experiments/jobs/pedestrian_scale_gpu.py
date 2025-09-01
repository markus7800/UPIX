from job_gen import sbatch
import sys

ndevices = int(sys.argv[1])

jobstr = """
python3 experiments/runners/run_pedestrian_scale.py cuda %d 20 sequential pmap --no_progress
""" % ndevices

sbatch("GPU", "ped_gpu", ndevices, jobstr)