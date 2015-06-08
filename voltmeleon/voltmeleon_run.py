
# TODO : import the needed modules

def usage():
    print "THEANO_FLAGS=device=gpu0,floatX=float32 python client_standalone_svhn_ninjite.py --experiment_dir=/home/gyomalin/Documents/ddt14/ImageNet/ninjite/experiment_dir_1"


def run(experiment_dir):

    assert os.path.exists(experiment_dir), "Cannot find experiment_dir : %s" % experiment_dir

    experiment_desc_file = os.path.join(experiment_dir, "experiment_desc.json")
    import json
    experiment_desc = json.load(open(experiment_desc_file, "r"))

    batch_size = experiment_desc['batch_size']
    #learning_rate = experiment_desc['learning_rate']
    #momentum = experiment_desc['momentum']
    step_flavor = experiment_desc['step_flavor']    
    integral_drop_rate = experiment_desc['integral_drop_rate']
    flecked_drop_rate = experiment_desc['flecked_drop_rate']
    L_nbr_filters = experiment_desc['L_nbr_filters']
    L_nbr_hidden_units = experiment_desc['L_nbr_hidden_units']
    weight_decay_factor = experiment_desc['weight_decay_factor'] if experiment_desc.has_key('weight_decay_factor') else 0.0
    dataset_hdf5_file = experiment_desc['dataset_hdf5_file']
    saving_path = experiment_desc['saving_path']

    server_desc = experiment_desc['server'] if experiment_desc.has_key('server') else None
    sync_desc = experiment_desc['sync'] if experiment_desc.has_key('sync') else None


    optional_args = {}
    for key in ['nbr_epochs']:
        if experiment_desc.has_key(key):
            optional_args[key] = experiment_desc[key]

    assert 1 <= batch_size
    assert os.path.exists(dataset_hdf5_file)

    print "L_nbr_filters : " + str(L_nbr_filters) 
    print "L_nbr_hidden_units : " + str(L_nbr_hidden_units) 

    run_training(   batch_size, step_flavor,
                    integral_drop_rate, flecked_drop_rate,
                    L_nbr_filters, L_nbr_hidden_units,
                    weight_decay_factor=weight_decay_factor,
                    dataset_hdf5_file=dataset_hdf5_file,
                    saving_path=saving_path,
                    server_desc=server_desc,
                    sync_desc=sync_desc,
                    **optional_args)


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