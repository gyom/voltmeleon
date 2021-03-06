

import os, sys
import getopt


import client_runner


def usage():
    print "THEANO_FLAGS=device=gpu0,floatX=float32 python voltmeleon_run.py --experiment_dir=/home/gyomalin/NIPS/experiments/experiment_dir_1 --output_server_params_desc_path=server_params_desc.json"
    print "python voltmeleon_run.py --experiment_dir=config_examples/experiment_04 --obs"


def override_special_case_environment_variable(model_desc):

    L_subs = [  ('L_exo_dropout_conv_layers',  "VOLTMELEON_MODEL_DESC_EXO_DROP"),
                ('L_exo_dropout_full_layers',  "VOLTMELEON_MODEL_DESC_EXO_DROP"),
                ('L_endo_dropout_conv_layers', "VOLTMELEON_MODEL_DESC_ENDO_DROP"),
                ('L_endo_dropout_full_layers', "VOLTMELEON_MODEL_DESC_ENDO_DROP")]

    for (model_desc_key, environ_key) in L_subs:

        if os.environ.has_key(environ_key):
            v = float(os.environ[environ_key])
            # Leave the value alone if it's not equal to `environ_key`.
            # If it is equal to `environ_key`, then just replace it with
            # the floating point variable contained in the environment.
            model_desc[model_desc_key] = [(e if e != environ_key else v) for e in model_desc[model_desc_key] ]

            print "model_desc[%s] is now %s" % (model_desc_key, str(model_desc[model_desc_key]))

    # unnecessary since we mutate the argument
    return model_desc


def run(experiment_dir, output_server_params_desc_path=None, want_observer_mode=False, running_on_helios=False, jobid=None, force_quit_after_total_duration=None, server_params_desc=None):

    if running_on_helios and jobid is not None:

        assert type(jobid) == int
        want_observer_mode = (jobid == 0)
        if want_observer_mode:
            print "We are running on helios in OBSERVER mode."
        else:
            print "We are running on helios."

    elif running_on_helios and jobid is None:
        # We have to do something special here because all the jobs
        # are launched with the same command-line, yet we want to run
        # an "observer" with the job zero.
        import helios
        helios.print_properties()
        print os.environ
        #helios_props = helios.get_properties()
        # we override the specifications to have `want_observer_mode` to True
        # automatically when we run on helios AND we're "job zero"
        want_observer_mode = helios.is_job_zero()
        if want_observer_mode:
            print "We are running on helios in OBSERVER mode."
        else:
            print "We are running on helios."
        jobid = helios.get_id()

    elif jobid is not None:
        assert type(jobid) == int
    else:
        import numpy as np
        jobid = np.random.randint(low=0, high=100000)

    print "jobid : %d" % jobid

    assert os.path.exists(experiment_dir), "Cannot find experiment_dir : %s" % experiment_dir

    model_desc_file = os.path.join(experiment_dir, "model_desc.json")

    # The "oserver mode" is a special case in which we use the file "observer_desc.json"
    # instead of the usual "train_desc.json". This should save us a lot of trouble in
    # managing configuration files.
    if want_observer_mode:
        train_desc_file = os.path.join(experiment_dir, "observer_desc.json")
    else:
        train_desc_file = os.path.join(experiment_dir, "train_desc.json")
    print "train_desc_file : %s" % train_desc_file

    import json
    model_desc = json.load(open(model_desc_file, "r"))
    model_desc = override_special_case_environment_variable(model_desc)

    print model_desc

    train_desc = json.load(open(train_desc_file, "r"))
    # hardcoded path for blocks saving a zip file

    # This will never be available on the first time that we run the script,
    # but it will be available in later runs (while the server is running).
    server_params_desc_file = os.path.join(experiment_dir, "server_params_desc.json")
    if os.path.exists(server_params_desc_file):
        server_params_desc = json.load(open(server_params_desc_file, "r"))
    else:
        server_params_desc = None
    

    if want_observer_mode:
        saving_path = os.path.join(experiment_dir, "log_%0.2d_obs" % (jobid ,))
    else:
        saving_path = os.path.join(experiment_dir, "log_%0.2d" % (jobid ,))

    print "saving_path : %s" % saving_path

    if os.path.exists(saving_path):
        print "saving path already exists. you have setup something wrong in your experiment."
        exit()

    client_runner.run(model_desc, train_desc, experiment_dir, saving_path, output_server_params_desc_path=output_server_params_desc_path, force_quit_after_total_duration=force_quit_after_total_duration, server_params_desc=server_params_desc)


def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["experiment_dir=", "output_server_params_desc_path=", "obs", "want_observer_mode", "helios", "jobid=", "force_quit_after_total_duration="])

    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    experiment_dir = None
    output_server_params_desc_path = None
    want_observer_mode = False
    running_on_helios = False
    jobid = None
    force_quit_after_total_duration = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--experiment_dir"):
            experiment_dir = a
        elif o in ("--output_server_params_desc_path"):
            output_server_params_desc_path = a
        elif o in ("--obs", "--want_observer_mode"):
            want_observer_mode = True
        elif o in ("--helios"):
            running_on_helios = True
        elif o in ("--jobid"):
            jobid = int(a)
        elif o in ("--force_quit_after_total_duration"):
            force_quit_after_total_duration = float(a)
        else:
            assert False, "unhandled option"


    #print "output_server_params_desc_path is %s" % output_server_params_desc_path

    run(experiment_dir, output_server_params_desc_path=output_server_params_desc_path, want_observer_mode=want_observer_mode, running_on_helios=running_on_helios, jobid=jobid, force_quit_after_total_duration=force_quit_after_total_duration)



if __name__ == "__main__":
    main(sys.argv)
