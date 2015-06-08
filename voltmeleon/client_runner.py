
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




sys.path.append("../../client")
#import client_api
from client_api import ClientCNNAutoSplitter


from server_update_extensions import ServerSyncAutoAdjustTiming

from architecture import (init_param, build_training, return_param, build_architecture)

def run_training(   batch_size, step_flavor,
                    integral_drop_rate, flecked_drop_rate,
                    L_nbr_filters, L_nbr_hidden_units,
                    weight_decay_factor,
                    dataset_hdf5_file,
                    saving_path,
                    nbr_epochs,
                    server_desc,
                    sync_desc):


    # TODO : start a client here based on `server_desc` if there is anything available

    D_dropout_probs = {'layer_0' : [integral_drop_rate, 0.0],
                       'layer_1' : [integral_drop_rate, integral_drop_rate],
                       'layer_2' : [integral_drop_rate, integral_drop_rate],
                       'layer_3' : [integral_drop_rate, integral_drop_rate],
                       'layer_4' : [integral_drop_rate, integral_drop_rate],
                       'layer_5' : [integral_drop_rate, integral_drop_rate],
                       'layer_6' : [integral_drop_rate, integral_drop_rate],
                       'layer_7' : [integral_drop_rate, 0.0]}

    drop_conv = [integral_drop_rate] * 4
    drop_mlp = [integral_drop_rate] * 4
    
    # note that these rates are applied on inputs,
    # hence the fact that the final value is not
    # required to be 0.0    
    D_dropout_mouchete = {'layer_0' : 0.0,
                          'layer_1' : flecked_drop_rate,
                          'layer_2' : flecked_drop_rate,
                          'layer_3' : flecked_drop_rate,
                          'layer_4' : flecked_drop_rate,
                          'layer_5' : flecked_drop_rate,
                          'layer_6' : flecked_drop_rate,
                          'layer_7' : flecked_drop_rate}


    cg, error_rate, cost, step_rule, names, params_dict, diagnostic_output = build_architecture(step_flavor=step_flavor,
                                                                                                drop_conv=drop_conv,
                                                                                                drop_mlp=drop_mlp,
                                                                                                L_nbr_filters=L_nbr_filters,
                                                                                                L_nbr_hidden_units=L_nbr_hidden_units,
                                                                                                weight_decay_factor=weight_decay_factor,
                                                                                                dataset_hdf5_file=dataset_hdf5_file)

    #print "params_dict"
    #print params_dict
    #print "names"
    #print names

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

        # This is now obsolete. We use the `server_sync_initial_read_extension`
        # to do this job.
        #client.perform_split(D_dropout_probs)
        #for name in names:
        #    param_value = client.pull_split_param(name)
        #    init_param(params_dict, name, param_value)
    
    
    for key in sync_desc:
        assert key in ['want_read_only', 'r', 'momentum_weights_scaling']

    # Run this at every iteration, but that doesn't mean that we're updating at every iteration.
    # It just means that we'll consider updating if the timing is good (in order to respect the
    # ratio `r` of time spend synching vs total).
    print sync_desc
    
    server_sync_extension_auto_timing = ServerSyncAutoAdjustTiming( client, D_dropout_probs, names,
                                                                    params_dict,
                                                                    every_n_batches=1, verbose=True,
                                                                    **sync_desc)

    import copy
    sync_desc_override_with_read_only = copy.copy(sync_desc)
    sync_desc_override_with_read_only['want_read_only'] = True
    server_sync_initial_read_extension = ServerSyncAutoAdjustTiming(client, D_dropout_probs, names,
                                                                    params_dict,
                                                                    before_training=True, verbose=True,
                                                                    **sync_desc_override_with_read_only)


    #
    # TODO : Implement the `velocities` for the step_rule, whether it's RMSProp or whatever other step rule.
    #



    checkpoint_interval_nbr_batches = 100
    # this launches the main loop internally
    build_training(cg, error_rate, cost, step_rule,
        batch_size=batch_size, dropout_bis = D_dropout_mouchete,
        dataset_hdf5_file=dataset_hdf5_file,
        checkpoint_interval_nbr_batches=checkpoint_interval_nbr_batches,
        diagnostic_output=diagnostic_output,
        saving_path=saving_path,
        nbr_epochs=nbr_epochs,
        server_sync_extension=server_sync_extension_auto_timing,
        server_sync_initial_read_extension=server_sync_initial_read_extension)


