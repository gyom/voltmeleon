

import os, sys
import getopt


import client_runner


def usage():
    print "THEANO_FLAGS=device=gpu0,floatX=float32 python voltmeleon_run.py --experiment_dir=/home/gyomalin/NIPS/experiments/experiment_dir_1"
    print "python voltmeleon_run.py --experiment_dir=config_examples/experiment_01"


def run(experiment_dir):

    assert os.path.exists(experiment_dir), "Cannot find experiment_dir : %s" % experiment_dir

    model_desc_file = os.path.join(experiment_dir, "model_desc.json")
    train_desc_file = os.path.join(experiment_dir, "train_desc.json")

    import json
    model_desc = json.load(open(model_desc_file, "r"))
    train_desc = json.load(open(train_desc_file, "r"))
    # hardcoded path for blocks saving a zip file
    saving_path = os.path.join(experiment_dir, "log.zip")

    client_runner.run(model_desc, train_desc, experiment_dir, saving_path)


def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["experiment_dir="])

    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    experiment_dir = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--experiment_dir"):
            experiment_dir = a
        else:
            assert False, "unhandled option"


    run(experiment_dir)



if __name__ == "__main__":
    main(sys.argv)
