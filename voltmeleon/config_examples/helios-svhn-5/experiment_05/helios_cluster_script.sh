#!/bin/bash

#PBS -l nodes=1:gpus=2
#PBS -l walltime=6:00:00
#PBS -A jvb-000-ag
#PBS -m bea
#PBS -t [0-1]%2

"""#PBS -M guillaume.alain.umontreal@gmail.com"""

# This whole script assumes that you've started the server
# on `helios1` the entry node for the Helios cluster.
# You should have picked a port that nobody else was using.


# To run this manually at home (dpln@lambda) :
#
#/home/dpln/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/voltmeleon/voltmeleon/config_examples/helios-svhn/scratch/server_params_desc.json --port=7000
#
#cd /home/dpln/NIPS/distdrop/bin/
#python auto_init_server_params.py --server=127.0.0.1 --port=7000 --W_range=0.1 --b_range=0.1 --want_zero_momentum
#
#cd ${HOME}/NIPS/voltmeleon/voltmeleon
#THEANO_FLAGS=device=gpu0,floatX=float32 python voltmeleon_run.py --experiment_dir=config_examples/helios-svhn/scratch


export EXPID=05


export EXPERIMENT_DIR=${HOME}/NIPS/experiments-svhn-5/${EXPID}
# Assuming that we've previously performed a
# git clone https://github.com/gyom/voltmeleon.git ${EXPERIMENT_DIR}/voltmeleon
# into that directory in preparation.
cd ${EXPERIMENT_DIR}/voltmeleon/voltmeleon

export RELATIVE_CONFIG_DIR=config_examples/helios-svhn-5/experiment_${EXPID}

# Make sure you use a different device for those two calls.
# DEBUG : Use only one GPU. This is wasteful, but let's try this for now
#         and tomorrow we'll figure out a way to hash this thing out.
#         (Maybe use the gpu number as part of the internal index.)
export PYTHONPATH=${PYTHONPATH}:${HOME}/NIPS/distdrop
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/fuel
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/theano
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/picklable_itertools
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/blocks

# transfer the stdout to stderr to it gets logged
THEANO_FLAGS=device=gpu0,floatX=float32 stdbuf -i0 -o0 -e0 python voltmeleon_run.py --experiment_dir=${RELATIVE_CONFIG_DIR} --helios >> ${MOAB_JOBARRAYINDEX}_0_out &

THEANO_FLAGS=device=gpu1,floatX=float32 stdbuf -i0 -o0 -e0 python voltmeleon_run.py --experiment_dir=${RELATIVE_CONFIG_DIR} --helios >> ${MOAB_JOBARRAYINDEX}_1_out &

# taken from
# http://stackoverflow.com/questions/356100/how-to-wait-in-bash-for-several-subprocesses-to-finish-and-return-exit-code-0
# because we want to wait for the two jobs running
for job in `jobs -p`
do
echo $job
    wait $job
done

cat ${MOAB_JOBARRAYINDEX}_0_out
cat ${MOAB_JOBARRAYINDEX}_1_out

#rm ${MOAB_JOBARRAYINDEX}_0_out
#rm ${MOAB_JOBARRAYINDEX}_1_out
