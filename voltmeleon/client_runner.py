
import sys, os
import getopt

import json

import numpy as np

####################################################
# because "fuck you" .blocksrc with default_seed !
s0 = np.random.randint(low=0, high=100000)
s1 = np.random.randint(low=0, high=100000)
import blocks
import blocks.config
import fuel
blocks.config.config.default_seed = s0
fuel.config.default_seed = s1
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

def set_all_dropout_in_model_desc_to_zero(model_desc):
    # note that this destroys the information in model_desc
    for key in ["L_exo_dropout_conv_layers", "L_exo_dropout_full_layers"]:
        if model_desc.has_key(key):
            model_desc[key] = [0.0] * len(model_desc[key])


def run(model_desc, train_desc, experiment_dir, saving_path, output_server_params_desc_path=None):

    # it's okay to not use the `experiment_dir` argument directly, for now

    # If `output_server_params_desc_path` is used, then this function will terminate early
    # after writing out the json file that the server will need.
    # Conceptually, one can run this before the experiment, in order to obtain the
    # file to be used for the server. Then we launch the server and we run the thing for real.


    if output_server_params_desc_path is not None:
        # we need to replace all the exo dropout values in order to generate the json file for the server config
        set_all_dropout_in_model_desc_to_zero(model_desc)
        print "Setting all the exo dropout values in order to generate the json file for the server config."


    def exo_dropout_helper(model_desc):
        # Add a 0.0 at the end because we keep all the outputs, and that's not a number
        # that's provided by the L_endo_dropout_full_layers variable.
        L_exo_dropout = model_desc['L_exo_dropout_conv_layers'] + model_desc['L_exo_dropout_full_layers'] + [0.0]

        D_dropout_probs = dict( ("layer_%d" % layer_number, e) for (layer_number, e) in enumerate(zip(L_exo_dropout, L_exo_dropout[1:])) )
        return L_exo_dropout, D_dropout_probs


    want_ignore_endo_dropout = (train_desc.has_key('sync') and
                                train_desc['sync'].has_key('want_ignore_endo_dropout') and
                                train_desc['sync']['want_ignore_endo_dropout'] == True)

    want_undo_exo_dropout = (train_desc.has_key('sync') and
                             train_desc['sync'].has_key('want_undo_exo_dropout') and
                             train_desc['sync']['want_undo_exo_dropout'] == True)

    # When we're running the client in "observer" mode,
    # we want to get rid of the exo and the endo dropout.
    #
    # There are multiple ways to go about doing this, but
    # for the endo dropout we'll just check if the 'sync'
    # component of `train_desc` has a 'want_ignore_endo_dropout'
    # key that is set to `True`. If such is the case, we'll just
    # mutate the values found in `model_desc` to set them to 0.0
    # to achieve the desired effect.
    if want_ignore_endo_dropout:
        print "Overriding ENDO dropout as requested by the train_desc."
        for k in ["L_endo_dropout_conv_layers", "L_endo_dropout_full_layers"]:
            if model_desc.has_key(k):
                model_desc[k] = [0.0] * len(model_desc[k])

    if want_undo_exo_dropout:
        print "Overriding EXO dropout as requested by the train_desc."

        # building a model with the exo dropout active just to see what the shapes will be
        (_, _, _, D_params_dropped, _) = build_model.build_submodel(**model_desc)

        for k in ["L_exo_dropout_conv_layers", "L_exo_dropout_full_layers"]:
            if model_desc.has_key(k):
                model_desc[k] = [0.0] * len(model_desc[k])
    
        # this is the actual built model that we are going to use
        (cg, error_rate, cost, D_params, D_kind) = build_model.build_submodel(**model_desc)

        # refresh the values now that we have overridden the config to contain zeros
        L_exo_dropout, D_dropout_probs = exo_dropout_helper(model_desc)

        D_rescale_factor_exo_dropout = {}
        for name, param_dropped in D_params_dropped.items():
            param = D_params[name]
            
            if D_kind[name] == "FULLY_CONNECTED_WEIGHTS":
                # the input dimension is the 0
                s = param_dropped.get_value().shape[0] / param.get_value().shape[0]
                D_rescale_factor_exo_dropout[name] = s
                print "Rescaling parameter %s by %f to compensate for exo dropout." % (name, s)
            elif D_kind[name] == "CONV_FILTER_WEIGHTS":
                # the input dimension is the 1
                s = param_dropped.get_value().shape[1] / param.get_value().shape[1]
                D_rescale_factor_exo_dropout[name] = s
                print "Rescaling parameter %s by %f to compensate for exo dropout." % (name, s)
            else:
                print "No need to rescale parameter %s to compensate for exo dropout." % name

        del _
        del D_params_dropped

    else:
        # This means that we are NOT compensating for the removal of the exo dropout.
        # This is the most common branch taken. All the clients, except the "observers",
        # will take this branch.
        L_exo_dropout, D_dropout_probs = exo_dropout_helper(model_desc)
        D_rescale_factor_exo_dropout = {}

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

        if not server_desc.has_key('hostname'):
            server_desc['hostname'] = "127.0.0.1"

        assert server_desc.has_key('port')

        assert server_desc.has_key('alpha')
        if not server_desc.has_key('beta'):
            server_desc['beta'] = 1.0 - server_desc['alpha']

            print "(server, port, alpha, beta)"
            print (server_desc['hostname'], server_desc['port'], server_desc['alpha'], server_desc['beta'])
            client = ClientCNNAutoSplitter.new_basic_alpha_beta(server_desc['hostname'],
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

    if client is None:
        server_sync_initial_read_extension = None
        server_sync_extension_auto_timing = None
        print "WARNING : No client. Setting the sync extensions to be None."

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















