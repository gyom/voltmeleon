
for EXPID in 00 01 02 03 04 05 06 07 ;
do
    mkdir ${HOME}/NIPS/experiments-svhn-5
    mkdir ${HOME}/NIPS/experiments-svhn-5/${EXPID}
    cd ${HOME}/NIPS/experiments-svhn-5/${EXPID}
    git clone https://github.com/gyom/voltmeleon.git voltmeleon
done


for EXPID in 00 01 02 03 04 05 06 07 ;
do
    cd ${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon
    git pull
done






export EXPID=00
export EXPPORT=7400
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=01
export EXPPORT=7401
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=02
export EXPPORT=7402
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=03
export EXPPORT=7403
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}


export EXPID=04
export EXPPORT=7404
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=05
export EXPPORT=7405
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=06
export EXPPORT=7406
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}

export EXPID=07
export EXPPORT=7407
/home/alaingui/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/server_params_desc.json --port=${EXPPORT}




python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=7400 --load_path=${HOME}/NIPS/experiments-svhn-5/experiment_05/final_params.pkl
python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=7401 --load_path=${HOME}/NIPS/experiments-svhn-5/experiment_05/final_params.pkl
python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=7402 --load_path=${HOME}/NIPS/experiments-svhn-5/experiment_05/final_params.pkl
python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=7403 --load_path=${HOME}/NIPS/experiments-svhn-5/experiment_05/final_params.pkl
python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=7404 --load_path=${HOME}/NIPS/experiments-svhn-5/experiment_05/final_params.pkl
python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=7405 --load_path=${HOME}/NIPS/experiments-svhn-5/experiment_05/final_params.pkl
python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=7406 --load_path=${HOME}/NIPS/experiments-svhn-5/experiment_05/final_params.pkl
python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=7407 --load_path=${HOME}/NIPS/experiments-svhn-5/experiment_05/final_params.pkl





for EXPID in 00 01 02 03 04 05 06 07 ;
do
    # optional initialization here.
    # loading from the previous experiments-svhn-4.
    python ${HOME}/NIPS/distdrop/bin/load_server_params.py --server=127.0.0.1 --port=74${EXPID} --load_path=${HOME}/NIPS/experiments-svhn-4/04/voltmeleon/voltmeleon/config_examples/helios-svhn-4/experiment_04/final_params.pkl

    cd ${HOME}/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}
    msub helios_cluster_script.sh
done






export EXPID=00
export EXPPORT=7400
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=01
export EXPPORT=7401
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=02
export EXPPORT=7402
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=03
export EXPPORT=7403
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl


export EXPID=04
export EXPPORT=7404
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=05
export EXPPORT=7405
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=06
export EXPPORT=7406
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl

export EXPID=07
export EXPPORT=7407
python ${HOME}/NIPS/distdrop/bin/save_server_params.py --server=127.0.0.1 --port=${EXPPORT} --save_path=/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID}/final_params.pkl






for EXPID in 00 01 02 03 04 05 06 07 ;
do
    rsync -av --delete helios:/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/config_examples/helios-svhn-5/experiment_${EXPID} /home/dpln/NIPS/experiments-svhn-5
    rsync -av helios:/home/alaingui/NIPS/experiments-svhn-5/${EXPID}/voltmeleon/voltmeleon/'*_*_out' ${HOME}/NIPS/experiments-svhn-5/experiment_${EXPID}
done



for EXPID in 00 01 02 03 04 05 06 07 ;
do
    cd ${HOME}/NIPS/experiments-svhn-5/experiment_${EXPID}
    python plot.py
done




# run as gyomalin@lambda
mkdir /home/gyomalin/Dropbox/umontreal_extra/NIPS2015/only_plots_5
for EXPID in 00 01 02 03 04 05 06 07 ;
do
    cp /home/dpln/NIPS/experiments-svhn-5/experiment_${EXPID}/error_rate.png /home/gyomalin/Dropbox/umontreal_extra/NIPS2015/only_plots_5/${EXPID}_error_rate.png
    cp /home/dpln/NIPS/experiments-svhn-5/experiment_${EXPID}/cost.png /home/gyomalin/Dropbox/umontreal_extra/NIPS2015/only_plots_5/${EXPID}_cost.png
done


