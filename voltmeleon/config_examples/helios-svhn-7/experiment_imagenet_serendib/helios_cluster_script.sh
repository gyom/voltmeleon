#!/bin/bash

#PBS -l nodes=1:gpus=1
#PBS -l walltime=12:00:00
#PBS -A jvb-000-ag
#PBS -m bea
#PBS -t [0-8]%9

"""#PBS -M guillaume.alain.umontreal@gmail.com"""

# This whole script assumes that you've started the server
# on `helios1` the entry node for the Helios cluster.
# You should have picked a port that nobody else was using.


#
# /home/dpln/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/voltmeleon/voltmeleon/config_examples/helios-svhn-7/experiment_imagenet_serendib/server_params_desc.json --port=7000
#
# python /home/dpln/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7000 --W_range=0.1 --b_range=0.1 --want_zero_momentum
#

export EXPID=imagenet_serendib
export VOLTMELEON_MODEL_DESC_EXO_DROP=0.0
export VOLTMELEON_MODEL_DESC_ENDO_DROP=0.5

export EXPERIMENT_DIR=${HOME}/NIPS/voltmeleon_experiments/experiments-svhn-7/${EXPID}
mkdir ${HOME}/NIPS/voltmeleon_experiments/experiments-svhn-7
mkdir $EXPERIMENT_DIR
# Assuming that we've previously performed a
# git clone https://github.com/gyom/voltmeleon.git ${EXPERIMENT_DIR}/voltmeleon
# into that directory in preparation.
cd ${EXPERIMENT_DIR}/voltmeleon/voltmeleon

export RELATIVE_CONFIG_DIR=config_examples/helios-svhn-7/experiment_${EXPID}

export FORCE_QUIT_AFTER_TOTAL_DURATION=36000

export PYTHONPATH=${PYTHONPATH}:${HOME}/NIPS/distdrop
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/fuel
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/theano
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/picklable_itertools
export PYTHONPATH=${PYTHONPATH}:${HOME}/deep-learning-suite/blocks

# transfer the stdout to stderr to it gets logged
THEANO_FLAGS=device=gpu0,floatX=float32 stdbuf -i0 -o0 -e0 python voltmeleon_run.py --experiment_dir=${RELATIVE_CONFIG_DIR} --helios --jobid=${MOAB_JOBARRAYINDEX} --force_quit_after_total_duration=${FORCE_QUIT_AFTER_TOTAL_DURATION}

# 
# THEANO_FLAGS=device=gpu0,floatX=float32 stdbuf -i0 -o0 -e0 python voltmeleon_run.py --experiment_dir=${RELATIVE_CONFIG_DIR} --jobid=1 --force_quit_after_total_duration=${FORCE_QUIT_AFTER_TOTAL_DURATION} --output_server_params_desc_path=${RELATIVE_CONFIG_DIR}/server_params_desc.json


# THEANO_FLAGS=device=gpu0,floatX=float32 stdbuf -i0 -o0 -e0 python voltmeleon_run.py --experiment_dir=${RELATIVE_CONFIG_DIR} --jobid=1 --force_quit_after_total_duration=${FORCE_QUIT_AFTER_TOTAL_DURATION}

