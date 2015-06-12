
import sys, os
import getopt

import numpy as np

####################################################
# because "fuck you" .blocksrc with default_seed !
#s0 = np.random.randint(low=0, high=100000)
#s1 = np.random.randint(low=0, high=100000)
#import blocks
#import blocks.config
#import fuel
#blocks.config.config.default_seed = s0
#fuel.config.default_seed = s1
####################################################



# Reminder : You probably need to modify your PYTHONPATH to have the
#            `distdrop` library accessible.
#    export PYTHONPATH=${PYTHONPATH}:/home/gyomalin/ML/deep-learning-suite/distdrop
#    export PYTHONPATH=${PYTHONPATH}:/u/alaingui/Documents/distdrop
#



from distdrop.client.client_api import ClientCNNAutoSplitter


from server_sync_extensions import ServerSyncAutoAdjustTiming

import build_model
import build_training


"""

ef build_submodel(input_shape,
                   output_dim,
                   L_dim_conv_layers,
                   L_filter_size,
                   L_pool_size,
                   L_activation_conv,
                   L_dim_full_layers,
                   L_activation_full,
                   L_exo_dropout_conv_layers,
                   L_exo_dropout_full_layers,
                   L_endo_dropout_conv_layers,
                   L_endo_dropout_full_layers):





    #learning_rate = experiment_desc['learning_rate']
    #momentum = experiment_desc['momentum']
    step_flavor = experiment_desc['step_flavor']    
    integral_drop_rate = experiment_desc['integral_drop_rate']
    flecked_drop_rate = experiment_desc['flecked_drop_rate']
    L_nbr_filters = experiment_desc['L_nbr_filters']
    L_nbr_hidden_units = experiment_desc['L_nbr_hidden_units']
    weight_decay_factor = experiment_desc['weight_decay_factor'] if experiment_desc.has_key('weight_decay_factor') else 0.0
    dataset_hdf5_file = experiment_desc['dataset_hdf5_file']

    # TODO : Figure out what to do with `saving_path`.
    # "saving_path" : "/home/dpln/NIPS/experiments/debug",
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


batch_size = train_desc['batch_size']


   batch_size, step_flavor,
                    integral_drop_rate, flecked_drop_rate,
                    L_nbr_filters, L_nbr_hidden_units,
                    weight_decay_factor,
                    dataset_hdf5_file,
                    saving_path,
                    nbr_epochs,
                    server_desc,
                    sync_desc):

"""


def run(model_desc, train_desc, experiment_dir, saving_path):

    # it's okay to not use the `experiment_dir` argument directly, for now

    (cg, error_rate, cost, D_params, D_kind) = build_model.build_submodel(**model_desc)



    build_model.build_step_rule_parameters(train_desc['step_flavor'], D_params, D_kind)

    (step_rule, D_additional_params, D_additional_kind) = build_model.build_step_rule_parameters(train_desc['step_flavor'], D_params, D_kind)

    # merge the two dicts of parameters
    D_params = dict(D_params.items() + D_additional_params.items())
    D_kind = dict(D_kind.items() + D_additional_kind.items())


    server_desc = train_desc['server']
    client = None
    if server_desc is not None:

        if not server_desc.has_key('server'):
            server_desc['server'] = "127.0.0.1"
            assert server_desc.has_key('port')

            assert server_desc.has_key('alpha')
            if not server_desc.has_key('beta'):
                server_desc['beta'] = 1.0 - server_desc['alpha']


                print "(server, port, alpha, beta)"
                print (server_desc['server'], server_desc['port'], server_desc['alpha'], server_desc['beta'])
                client = ClientCNNAutoSplitter.new_basic_alpha_beta(server_desc['server'],
                    server_desc['port'],
                    server_desc['alpha'],
                    server_desc['beta'])

                client.connect()
                print client.read_param_desc_from_server()

        # Note that we don't need to get the parameters here.
        # We use the `server_sync_initial_read_extension` to do this job.
    
    
    for key in sync_desc:
        assert key in ['want_read_only', 'r', 'momentum_weights_scaling']

    # Run extension at every iteration, but that doesn't mean that we're updating at every iteration.
    # It just means that we'll consider updating if the timing is good (in order to respect the
    # ratio `r` of time spend synching vs total).



    # TODO : These two next sections are really part of the code. They need to be uncommented.

    # TODO : Find out how to manage `D_dropout_probs`.
    #server_sync_extension_auto_timing = ServerSyncAutoAdjustTiming( client, D_dropout_probs, names,
    #                                                                params_dict,
    #                                                                every_n_batches=1, verbose=True,
    #                                                                **sync_desc)
    server_sync_extension_auto_timing = None

    #import copy
    #sync_desc_override_with_read_only = copy.copy(sync_desc)
    #sync_desc_override_with_read_only['want_read_only'] = True
    #server_sync_initial_read_extension = ServerSyncAutoAdjustTiming(client, D_dropout_probs, names,
    #                                                                params_dict,
    #                                                                before_training=True, verbose=True,
    #                                                                **sync_desc_override_with_read_only)
    server_sync_initial_read_extension = None


    main_loop = build_training(cg, error_rate, cost, step_rule,
                               weight_decay_factor=train_desc['weight_decay_factor'],
                               dataset_hdf5_file=train_desc['dataset_hdf5_file'],
                               batch_size=train_desc['batch_size'],
                               nbr_epochs=train_desc['nbr_epochs'],
                               saving_path=saving_path,
                               server_sync_extension=server_sync_extension,
                               server_sync_initial_read_extension=server_sync_initial_read_extension,
                               checkpoint_interval_nbr_batches=train_desc['server_sync_initial_read_extension'])

    main_loop.run()















