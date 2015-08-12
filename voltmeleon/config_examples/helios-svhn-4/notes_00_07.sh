
for EXPID in 00 01 02 03 04 05 06 07 ;
do
    mkdir ${HOME}/NIPS/experiments-svhn-4
    mkdir ${HOME}/NIPS/experiments-svhn-4/${EXPID}
    cd ${HOME}/NIPS/experiments-svhn-4/${EXPID}
    git clone https://github.com/gyom/voltmeleon.git voltmeleon
done


for EXPID in 00 01 02 03 04 05 06 07 ;
do
    cd ${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon
    git pull
done






export EXPID=00
export EXPPORT=7400
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=01
export EXPPORT=7401
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=02
export EXPPORT=7402
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=03
export EXPPORT=7403
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}


export EXPID=04
export EXPPORT=7404
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=05
export EXPPORT=7405
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=06
export EXPPORT=7406
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=07
export EXPPORT=7407
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}





python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7400 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7401 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7402 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7403 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7404 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7405 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7406 --W_range=0.1 --b_range=0.1 --want_zero_momentum
python ${HOME}/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7407 --W_range=0.1 --b_range=0.1 --want_zero_momentum





for EXPID in 00 01 02 03 04 05 06 07 ;
do
    cd ${HOME}/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}
    msub helios_cluster_script.sh
done






export EXPID=00
export EXPPORT=7400
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/final_params.pkl

export EXPID=01
export EXPPORT=7401
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/final_params.pkl

export EXPID=02
export EXPPORT=7402
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/final_params.pkl

export EXPID=03
export EXPPORT=7403
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/final_params.pkl


export EXPID=04
export EXPPORT=7404
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/final_params.pkl

export EXPID=05
export EXPPORT=7405
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/final_params.pkl

export EXPID=06
export EXPPORT=7406
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/final_params.pkl

export EXPID=07
export EXPPORT=7407
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID}/final_params.pkl






for EXPID in 00 01 02 03 04 05 06 07 ;
do
    rsync -av --delete helios:/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_${EXPID} /home/dpln/NIPS/experiments-svhn-4
    rsync -av helios:/home/alaingui/NIPS/experiments-svhn-4/${EXPID}/voltmeleon/voltmeleon/'*_*_out' ${HOME}/NIPS/experiments-svhn-4/experiment_${EXPID}
done



for EXPID in 00 01 02 03 04 05 06 07 ;
do
    cd ${HOME}/NIPS/experiments-svhn-4/experiment_${EXPID}
    python plot.py
done


