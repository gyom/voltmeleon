

import os, sys
import getopt


import client_runner


def usage():
    print "THEANO_FLAGS=device=gpu0,floatX=float32 python voltmeleon_run.py --experiment_dir=/home/gyomalin/NIPS/experiments/experiment_dir_1 --output_server_params_desc_path=server_params_desc.json"
    print "python voltmeleon_run.py --experiment_dir=config_examples/experiment_01"


def run(experiment_dir, output_server_params_desc_path=None):

    assert os.path.exists(experiment_dir), "Cannot find experiment_dir : %s" % experiment_dir

    model_desc_file = os.path.join(experiment_dir, "model_desc.json")
    train_desc_file = os.path.join(experiment_dir, "train_desc.json")

    import json
    model_desc = json.load(open(model_desc_file, "r"))
    train_desc = json.load(open(train_desc_file, "r"))
    # hardcoded path for blocks saving a zip file

    assert os.path.exists(experiment_dir)
    n = 0
    while True:
        saving_path = os.path.join(experiment_dir, "log_%0.2d.zip" % n)
        n += 1
        if not os.path.exists(saving_path):
            break
            


    client_runner.run(model_desc, train_desc, experiment_dir, saving_path, output_server_params_desc_path=output_server_params_desc_path)


def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["experiment_dir=", "output_server_params_desc_path="])

    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    experiment_dir = None
    output_server_params_desc_path = None

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
        else:
            assert False, "unhandled option"


    run(experiment_dir, output_server_params_desc_path=output_server_params_desc_path)



if __name__ == "__main__":
    main(sys.argv)
