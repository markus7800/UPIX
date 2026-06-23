#! /bin/bash
# ncpu ngpu

time bash experiments/runners/run_scale_all_references.sh $1 cpu $3 $4 $5 $6
time bash experiments/runners/run_scale_all_upix.sh cpu $1 cpu $3 $4 $5 $6
time bash experiments/runners/run_scale_all_upix.sh cuda $2 gpu $3 $4 $5 $6
time bash experiments/runners/run_scale_all_accuracy.sh $1 $3 $4 $5 $6