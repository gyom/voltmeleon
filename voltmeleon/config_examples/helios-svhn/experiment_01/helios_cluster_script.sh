#!/bin/bash

#PBS -l nodes=4:gpus=2
#PBS -l wallclock=12:00:00
#PBS -A jvb-000-ag
#PBS -M guillaume.alain.umontreal@gmail.com
#PBS -m bea

cd ${HOME}/NIPS/experiments/01/voltmeleon/voltmeleon
# Assuming that we've previously performed a
# git clone https://github.com/gyom/voltmeleon.git voltmeleon
# into that directory in preparation.

# Make sure you use a different device for those two calls.
PYTHONPATH=${PYTHONPATH}:${HOME}/NIPS/distdrop THEANO_FLAGS=device=gpu0,floatX=float32 python voltmeleon_run.py --experiment_dir=config_examples/helios-svhn/experiment_01 --helios &

PYTHONPATH=${PYTHONPATH}:${HOME}/NIPS/distdrop THEANO_FLAGS=device=gpu1,floatX=float32 python voltmeleon_run.py --experiment_dir=config_examples/helios-svhn/experiment_01 --helios &

# taken from
# http://stackoverflow.com/questions/356100/how-to-wait-in-bash-for-several-subprocesses-to-finish-and-return-exit-code-0
# because we want to wait for the two jobs running
for job in `jobs -p`
do
echo $job
    wait $job
done

