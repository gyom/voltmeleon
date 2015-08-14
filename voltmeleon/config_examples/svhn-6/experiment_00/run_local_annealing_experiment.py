

import time

import os
import signal
import subprocess

import numpy as np


def init_server_params():
    cmd = "python /home/dpln/NIPS/distdrop/bin/auto_init_server_params.py --server=127.0.0.1 --port=7000 --W_range=0.1 --b_range=0.1 --want_zero_momentum"
    print subprocess.check_output(cmd, shell=True)


def run_clients(duration_in_secs, nbr_clients, voltmeleon_root_dir, config_dir, jobid_offset, exo_drop=None, endo_drop=None, want_dry_run=False):

    L_client_processes = []

    for jobid in range(jobid_offset, jobid_offset + nbr_clients):

        model_desc_override_flags = ""
        if exo_drop is not None:
            model_desc_override_flags = model_desc_override_flags + "VOLTMELEON_MODEL_DESC_EXO_DROP=%0.3f" % exo_drop
        if endo_drop is not None:
            model_desc_override_flags = model_desc_override_flags + "  VOLTMELEON_MODEL_DESC_ENDO_DROP=%0.3f" % endo_drop

        theano_flags = "THEANO_FLAGS=device=gpu0,floatX=float32"

        script_path = os.path.join(voltmeleon_root_dir, "voltmeleon/voltmeleon_run.py")

        cmd = "%s %s python %s --experiment_dir=%s --jobid=%d" % (model_desc_override_flags, theano_flags, script_path, config_dir, jobid)
        print "Running subprocess: \n\t%s" % cmd

        if want_dry_run:
            print "\t(not really calling subprocess because this is a dry run)"
        else:
            # The os.setsid() is passed in the argument preexec_fn so
            # it's run after the fork() and before  exec() to run the shell.
            pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                   shell=True, preexec_fn=os.setsid) 
            L_client_processes.append(pro)


    print "Now running %d clients." % nbr_clients

    timestamp_start = time.time()
    while time.time() - timestamp_start < duration_in_secs:
        # sleep for 5 seconds until time has elapsed
        time.sleep(5)
    
    print "Time has elapsed. Going to kill the processes."

    for pro in L_client_processes:
        print "Killing pid %d with SIGTERM." % pro.pid
        os.killpg(pro.pid, signal.SIGTERM)  # Send the signal to all the process groups
        time.sleep(5)
        print "Killing pid %d with SIGTERM again (in case Blocks caught it)." % pro.pid
        os.killpg(pro.pid, signal.SIGTERM)  # Send the signal to all the process groups
        time.sleep(5)
        os.killpg(pro.pid, signal.SIGTERM)  # Send the signal to all the process groups
        pro.wait()

def run():

    init_server_params()

    #duration_in_secs = 60*60
    duration_in_secs = 5*60

    nbr_clients = 4
    want_dry_run = False

    voltmeleon_root_dir = "/home/dpln/NIPS/voltmeleon"

    config_dir = os.path.join(voltmeleon_root_dir, "voltmeleon/config_examples/svhn-6/experiment_00")

    endo_drop = 0.5

    L_exo_drop = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    L_jobid_offset = 10 * np.arange(0, len(L_exo_drop))

    for exo_drop, jobid_offset in zip(L_exo_drop, L_jobid_offset):
        run_clients(duration_in_secs, nbr_clients, voltmeleon_root_dir, config_dir, jobid_offset, exo_drop=exo_drop, endo_drop=endo_drop, want_dry_run=want_dry_run)

if __name__ == "__main__":
    run()



"""

/home/dpln/NIPS/distdrop/bin/server --model_params_desc=${HOME}/NIPS/voltmeleon/voltmeleon/config_examples/svhn-6/experiment_00/server_params_desc.json --port=7000

"""