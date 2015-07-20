#!/bin/bash

#PBS -l nodes=1:gpus=2
#PBS -l walltime=1:00:00
#PBS -A jvb-000-ag
#PBS -M guillaume.alain.umontreal@gmail.com
#PBS -m bea
#PBS -t [0-3]%4

# This whole script assumes that you've started the server
# on `helios1` the entry node for the Helios cluster.
# You should have picked a port that nobody else was using.


export EXPERIMENT_DIR=${HOME}/NIPS/experiments/01
# Assuming that we've previously performed a
# git clone https://github.com/gyom/voltmeleon.git ${EXPERIMENT_DIR}/voltmeleon
# into that directory in preparation.
cd ${EXPERIMENT_DIR}/voltmeleon/voltmeleon

# Make sure you use a different device for those two calls.
# DEBUG : Use only one GPU. This is wasteful, but let's try this for now
#         and tomorrow we'll figure out a way to hash this thing out.
#         (Maybe use the gpu number as part of the internal index.)
export PYTHONPATH=${PYTHONPATH}:${HOME}/NIPS/distdrop
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/fuel
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/theano
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/picklable_itertools
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/blocks


THEANO_FLAGS=device=gpu0,floatX=float32 python voltmeleon_run.py --experiment_dir=config_examples/helios-svhn/experiment_01 --helios &

#PYTHONPATH=${PYTHONPATH}:${HOME}/NIPS/distdrop THEANO_FLAGS=device=gpu1,floatX=float32 python voltmeleon_run.py --experiment_dir=config_examples/helios-svhn/experiment_01 --helios &

# taken from
# http://stackoverflow.com/questions/356100/how-to-wait-in-bash-for-several-subprocesses-to-finish-and-return-exit-code-0
# because we want to wait for the two jobs running
for job in `jobs -p`
do
echo $job
    wait $job
done

