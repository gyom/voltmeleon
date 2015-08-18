
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

def set_all_exo_dropout_in_model_desc_to_zero(model_desc):
    # note that this destroys the information in model_desc
    for key in ["L_exo_dropout_conv_layers", "L_exo_dropout_full_layers"]:
        if model_desc.has_key(key):
            model_desc[key] = [0.0] * len(model_desc[key])



def build_model_with_endo_adjustments(model_desc, server_params_desc=None, want_undo_exo_dropout=False):

    if want_undo_exo_dropout:
        set_all_exo_dropout_in_model_desc_to_zero(model_desc)

    # That `server_params_desc` argument is something that we can only have
    # if we've run the code before. It comes from the "server_params_desc.json" file
    # that describes all the parameters in the full model.
    #
    # Instead of constructing the model twice, we use that description of
    # the parameters to simply evaluate the ratio between certain shapes.

    (cg, error_rate, cost, D_params, D_kind) = build_model.build_submodel(**model_desc)

    #import pdb; pdb.set_trace()

    D_rescale_factor_exo_dropout = {}
    for param_desc_full in server_params_desc:
        name = str(param_desc_full['name'])

        if not D_params.has_key(name):
            # Sometimes the parameter just isn't here yet.
            # For example, in the case of "decay" versions of
            # parameters, to be used with momentum, they are found
            # in the `server_params_desc` but they are not yet
            # instantiated in our model.
            continue

        param = D_params[name]
        
        if D_kind[name] == "FULLY_CONNECTED_WEIGHTS":
            # the input dimension is the 0
            s = 1.0 * param_desc_full['shape'][0] / param.get_value().shape[0]
            D_rescale_factor_exo_dropout[name] = s
            print "Rescaling parameter %s by %f when read from server, to compensate for exo dropout." % (name, s)
        elif D_kind[name] == "CONV_FILTER_WEIGHTS":
            # the input dimension is the 1
            s = 1.0 * param_desc_full['shape'][1] / param.get_value().shape[1]
            D_rescale_factor_exo_dropout[name] = s
            print "Rescaling parameter %s by %f when read from server, to compensate for exo dropout." % (name, s)
        else:
            print "No need to rescale parameter %s to compensate for exo dropout." % name

    # Add a 0.0 at the end because we keep all the outputs, and that's not a number
    # that's provided by the L_endo_dropout_full_layers variable.
    L_exo_dropout = model_desc['L_exo_dropout_conv_layers'] + model_desc['L_exo_dropout_full_layers'] + [0.0]
    D_dropout_probs = dict( ("layer_%d" % layer_number, e) for (layer_number, e) in enumerate(zip(L_exo_dropout, L_exo_dropout[1:])) )

    return (cg, error_rate, cost,
            D_params, D_kind,
            L_exo_dropout, D_dropout_probs,
            D_rescale_factor_exo_dropout)




def run(model_desc, train_desc, experiment_dir, saving_path, output_server_params_desc_path=None, force_quit_after_total_duration=None, server_params_desc=None):

    # it's okay to not use the `experiment_dir` argument directly, for now

    # If `output_server_params_desc_path` is used, then this function will terminate early
    # after writing out the json file that the server will need.
    # Conceptually, one can run this before the experiment, in order to obtain the
    # file to be used for the server. Then we launch the server and we run the thing for real.


    if output_server_params_desc_path is not None:
        # we need to replace all the exo dropout values in order to generate the json file for the server config
        set_all_exo_dropout_in_model_desc_to_zero(model_desc)
        print "Setting all the exo dropout values in order to generate the json file for the server config."


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
    #
    # We don't have to scale the parameters to compensate for anything
    # due to the way that dropout is implemented in Blocks.
    # The implementation is such that the parameters are set to their
    # proper values that they would have if we ignored the endo dropout,
    # so we don't have to compute the equivalent of the `D_rescale_factor_exo_dropout`.
    if want_ignore_endo_dropout:
        print "Overriding ENDO dropout as requested by the train_desc."
        for k in ["L_endo_dropout_conv_layers", "L_endo_dropout_full_layers"]:
            if model_desc.has_key(k):
                model_desc[k] = [0.0] * len(model_desc[k])

    (cg, error_rate, cost,
     D_params, D_kind,
     L_exo_dropout,
     D_dropout_probs, D_rescale_factor_exo_dropout) = build_model_with_endo_adjustments(model_desc, server_params_desc)

    # This `D_rescale_factor_exo_dropout` will be used for the blocks extensions.
    # The rest of the returned arguments will be used to setup the other parts of the training.


    build_model.build_step_rule_parameters(train_desc['step_flavor'], D_params, D_kind)

    (step_rule, D_additional_params, D_additional_kind) = build_model.build_step_rule_parameters(train_desc['step_flavor'], D_params, D_kind)

    # merge the two dicts of parameters
    D_params = dict(D_params.items() + D_additional_params.items())
    D_kind = dict(D_kind.items() + D_additional_kind.items())

    print "======================"
    for (name, param_var) in sorted(D_params.items(), key=lambda e:e[0]):
        print "    %s has shape %s" % (name, param_var.get_value(borrow=True, return_internal_type=True).shape)
    print "======================"
    print ""

    # We need to add the corresponding entries in `D_rescale_factor_exo_dropout`
    # for all those additional variables.
    for (name, param) in D_params.items():

        if D_rescale_factor_exo_dropout.has_key(name):
            print "Already a dropout entry for %s." % name
            continue

        for k in D_rescale_factor_exo_dropout.keys():
            # check if `k` could be a prefix of `name`
            if len(k) < len(name) and name[0:len(k)] == k:
                print "D_rescale_factor_exo_dropout[%s] = D_rescale_factor_exo_dropout[%s]" % (name, k)
                D_rescale_factor_exo_dropout[name] = D_rescale_factor_exo_dropout[k]
        
        if D_rescale_factor_exo_dropout.has_key(name):
            print "Failed to find the dropout entry for %s." % name
            continue




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
            E = client.read_param_desc_from_server()
            print ""
            print "==== read_param_desc_from_server() ===="
            for e in sorted(E, key=lambda e: e['name']):
                print e
            print "==== ===="
            print ""

        # Note that we don't need to get the parameters here.
        # We use the `server_sync_initial_read_extension` to do this job.
    

               

    sync_desc = train_desc['sync']
    # At this point we strip away the keys in `sync_desc` that
    # are not used by the extensions.
    # This is not a great practice to do, but
    # we're already aware that the is a bit of a conceptual
    # mismatch in our choice of having those keys in `sync_desc`
    # alongside the arguments to the extensions themselves.
    for key in ['want_undo_exo_dropout',
                'want_ignore_endo_dropout']:
        if sync_desc.has_key(key):
            del sync_desc[key]

    for key in sync_desc:
        assert key in [ 'want_read_only',
                        'max_time_ratio_spent',
                        'momentum_weights_scaling'], "Unrecognized key : %s" % key

    if sync_desc.has_key('r'):
        print "The 'r' value in the 'sync' dictionary is now called 'max_time_ratio_spent'."
        print "Change your configuration file to reflect this."
        print "Exiting."
        exit()


    dataset_desc = train_desc['dataset']
    for key in dataset_desc:
        assert key in ['hdf5_file', 'want_subset_valid', 'want_eval_on_valid',
                       'want_eval_on_test', 'want_subset_test']


    # Run extension at every iteration, but that doesn't mean that we're updating at every iteration.
    # It just means that we'll consider updating if the timing is good (in order to respect the
    # ratio `max_time_ratio_spent` of time spend synching vs total).

    server_sync_extension_auto_timing = ServerSyncAutoAdjustTiming( client, D_dropout_probs,
                                                                    D_params,
                                                                    every_n_batches=1, verbose=True,
                                                                    D_rescale_factor_exo_dropout=D_rescale_factor_exo_dropout,
                                                                    **sync_desc)

    import copy
    sync_desc_override_with_read_only = copy.copy(sync_desc)
    sync_desc_override_with_read_only['want_read_only'] = True
    server_sync_initial_read_extension = ServerSyncAutoAdjustTiming(client, D_dropout_probs,
                                                                    D_params,
                                                                    before_training=True, verbose=True,
                                                                    D_rescale_factor_exo_dropout=D_rescale_factor_exo_dropout,
                                                                    **sync_desc_override_with_read_only)

    if client is None:
        server_sync_initial_read_extension = None
        server_sync_extension_auto_timing = None
        print "WARNING : No client. Setting the sync extensions to be None."

    print "Asked to run for force_quit_after_total_duration = %d seconds." % force_quit_after_total_duration

    main_loop = build_training.build_training(cg, error_rate, cost, step_rule,
                                              weight_decay_factor=train_desc['weight_decay_factor'],
                                              hdf5_file=dataset_desc['hdf5_file'],
                                              want_subset_valid=dataset_desc['want_subset_valid'],
                                              want_eval_on_valid=dataset_desc['want_eval_on_valid'],
                                              want_eval_on_test=dataset_desc['want_eval_on_test'],
                                              want_subset_test=dataset_desc['want_subset_test'],
                                              batch_size=train_desc['batch_size'],
                                              nbr_epochs=train_desc['nbr_epochs'],
                                              saving_path=saving_path,
                                              server_sync_extension=server_sync_extension_auto_timing,
                                              server_sync_initial_read_extension=server_sync_initial_read_extension,
                                              monitor_interval_nbr_batches=train_desc['monitor_interval_nbr_batches'],
                                              force_quit_after_total_duration=force_quit_after_total_duration)

    main_loop.run()















