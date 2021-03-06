


cd ${HOME}/NIPS/voltmeleon/voltmeleon
THEANO_FLAGS=device=gpu0,floatX=float32 python voltmeleon_run.py --experiment_dir=config_examples/helios-svhn-5/experiment_08 --output_server_params_desc_path=config_examples/helios-svhn-5/experiment_08/server_params_desc.json






for EXPID in 08 09 10 11 12 13 14 15 ;
do
    mkdir ${HOME}/NIPS/experiments-svhn-5
    mkdir ${HOME}/NIPS/experiments-svhn-5/${EXPID}
    cd ${HOME}/NIPS/experiments-svhn-5/${EXPID}
    git clone https://github.com/gyom/voltmeleon.git voltmeleon
done


for EXPID in 08 09 10 11 12 13 14 15 ;
do
    cd ${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon
    git pull
done






export EXPID=08
export EXPPORT=7408
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=09
export EXPPORT=7409
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=10
export EXPPORT=7410
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=11
export EXPPORT=7411
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}


export EXPID=12
export EXPPORT=7412
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=13
export EXPPORT=7413
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=14
export EXPPORT=7414
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=15
export EXPPORT=7415
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}





python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7408 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7409 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7410 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7411 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7412 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7413 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7414 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7415 --W_range=0.1 --b_range=0.1 --want_zero_momentum


# loading from the previous experiments-svhn-4
#python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=7400 --load_path=${HOME}/NIPS/experiments-svhn-4/experiment_05/final_params.pkl



for EXPID in 08 09 10 11 12 13 14 15 ;
do
    # optional initialization here.
    # loading from the previous experiments-svhn-4.
    python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=74${EXPID} --load_path=${HOME}/NIPS/experiments-svhn-4/12/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_12/final_params.pkl

    cd ${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}
    msub helios_cluster_script.sh
done








export EXPID=08
export EXPPORT=7408
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=09
export EXPPORT=7409
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=10
export EXPPORT=7410
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=11
export EXPPORT=7411
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl


export EXPID=12
export EXPPORT=7412
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=13
export EXPPORT=7413
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=14
export EXPPORT=7414
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=15
export EXPPORT=7415
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl







for EXPID in 08 09 10 11 12 13 14 15 ;
do
    rsync -av --delete helios:/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID} /home/dpln/NIPS/experiments-svhn-5
    rsync -av helios:/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/'*_*_out' /home/dpln/NIPS/experiments-svhn-5/experiment_${EXPID}
done



for EXPID in 08 09 10 11 12 13 14 15 ;
do
    cd ${HOME}/NIPS/experiments-svhn-5/experiment_${EXPID}
    python plot.py
done





# run as gyomalin@lambda
mkdir /home/gyomalin/Dropbox/umontreal_extra/NIPS2015/only_plots_5
for EXPID in 08 09 10 11 12 13 14 15 ;
do
    cp /home/dpln/NIPS/experiments-svhn-5/experiment_${EXPID}/error_rate.png /home/gyomalin/Dropbox/umontreal_extra/NIPS2015/only_plots_5/${EXPID}_error_rate.png
    cp /home/dpln/NIPS/experiments-svhn-5/experiment_${EXPID}/cost.png /home/gyomalin/Dropbox/umontreal_extra/NIPS2015/only_plots_5/${EXPID}_cost.png
done



