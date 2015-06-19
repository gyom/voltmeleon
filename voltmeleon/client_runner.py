
import sys, os
import getopt

import json

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


def run(model_desc, train_desc, experiment_dir, saving_path, output_server_params_desc_path=None):

    # it's okay to not use the `experiment_dir` argument directly, for now

    # If `output_server_params_desc_path` is used, then this function will terminate early
    # after writing out the json file that the server will need.
    # Conceptually, one can run this before the experiment, in order to obtain the
    # file to be used for the server. Then we launch the server and we run the thing for real.

    (cg, error_rate, cost, D_params, D_kind) = build_model.build_submodel(**model_desc)



    build_model.build_step_rule_parameters(train_desc['step_flavor'], D_params, D_kind)

    (step_rule, D_additional_params, D_additional_kind) = build_model.build_step_rule_parameters(train_desc['step_flavor'], D_params, D_kind)

    # merge the two dicts of parameters
    D_params = dict(D_params.items() + D_additional_params.items())
    D_kind = dict(D_kind.items() + D_additional_kind.items())

    print "======================"
    for (name, param_var) in sorted(D_params.items(), key=lambda e:e[0]):
        print "    %s has shape %s" % (name, param_var.get_value(borrow=True, return_internal_type=True).shape)
    print ""

    if output_server_params_desc_path is not None:
        L_server_params_desc = build_model.get_model_desc_for_server(D_params, D_kind)
        json.dump(L_server_params_desc, open(output_server_params_desc_path, "w"))
        print "Wrote the json file for the server parameter description in %s. Now exiting." % output_server_params_desc_path
        return


    if train_desc.has_key('server'):
        server_desc = train_desc['server']
    else:
        server_desc = None

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
    
    sync_desc = train_desc['sync']
    for key in sync_desc:
        assert key in ['want_read_only', 'r', 'momentum_weights_scaling']

    # Run extension at every iteration, but that doesn't mean that we're updating at every iteration.
    # It just means that we'll consider updating if the timing is good (in order to respect the
    # ratio `r` of time spend synching vs total).


    # add a 0.0 at the end because we keep all the outputs, and that's not a number
    # that's provided by the L_endo_dropout_full_layers variable
    L_exo_dropout = model_desc['L_exo_dropout_conv_layers'] + model_desc['L_exo_dropout_full_layers'] + [0.0]

    D_dropout_probs = dict( ("layer_%d" % layer_number, e) for (layer_number, e) in enumerate(zip(L_exo_dropout, L_exo_dropout[1:])) )


    server_sync_extension_auto_timing = ServerSyncAutoAdjustTiming( client, D_dropout_probs,
                                                                    D_params,
                                                                    every_n_batches=1, verbose=True,
                                                                    **sync_desc)

    import copy
    sync_desc_override_with_read_only = copy.copy(sync_desc)
    sync_desc_override_with_read_only['want_read_only'] = True
    server_sync_initial_read_extension = ServerSyncAutoAdjustTiming(client, D_dropout_probs,
                                                                    D_params,
                                                                    before_training=True, verbose=True,
                                                                    **sync_desc_override_with_read_only)


    main_loop = build_training.build_training(cg, error_rate, cost, step_rule,
                                              weight_decay_factor=train_desc['weight_decay_factor'],
                                              dataset_hdf5_file=train_desc['dataset_hdf5_file'],
                                              batch_size=train_desc['batch_size'],
                                              nbr_epochs=train_desc['nbr_epochs'],
                                              saving_path=saving_path,
                                              server_sync_extension=server_sync_extension_auto_timing,
                                              server_sync_initial_read_extension=server_sync_initial_read_extension,
                                              checkpoint_interval_nbr_batches=train_desc['checkpoint_interval_nbr_batches'])

    main_loop.run()















