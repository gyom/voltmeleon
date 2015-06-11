
import time

from common_tools import (set_param_value_shared_var, get_param_value_shared_var)
#from architecture import (set_param_value_shared_var, return_param)

import blocks
from blocks.extensions import SimpleExtension

# this is an old implementation that will be replaced
class ServerUpdateAfterTBatches_Full(SimpleExtension):

    def __init__(self, client, D_dropout_probs, names, params_dict, T, **kwargs):
        # `params_dict` contains both the parameters and the momentums
        super(ServerUpdateAfterTBatches, self).__init__(after_n_batches=T, **kwargs)

        self.client = client
        self.D_dropout_probs = D_dropout_probs
        self.names = names
        self.params_dict = params_dict
        self.T = T

    def do(self, which_callback, *args):

        if which_callback in ['after_batch', 'after_n_batches']:

            self.client.perform_split(self.D_dropout_probs)

            for name in self.names:
                param_value = self.client.pull_split_param(name)
                set_param_value_shared_var(self.params_dict, name, param_value)

# this is the current implementation
class ServerUpdateAfterTBatches(SimpleExtension):

    def __init__(self, client, D_dropout_probs, names, params_dict, T, **kwargs):
        # `params_dict` contains both the parameters and the momentums
        super(ServerUpdateAfterTBatches, self).__init__(every_n_batches=T, **kwargs)

        self.client = client
        self.D_dropout_probs = D_dropout_probs
        self.names = names
        self.params_dict = params_dict
        self.T = T

    def do(self, which_callback, *args):

        if which_callback:
            for name in self.names:
                param_value = return_param(self.params_dict, name)
                self.client.push_split_param(name, param_value)
            self.client.perform_split(self.D_dropout_probs)
            for name in self.names:
                param_value = self.client.pull_split_param(name)
                set_param_value_shared_var(self.params_dict, name, param_value)


class ServerSyncAutoAdjustTiming(SimpleExtension):

    # make sure to specify the argument "every_n_batches=T" when you instantiate this extension,
    # or something to that effect to determine how often we want to call it

    def __init__(self, client, D_dropout_probs, names, params_dict,
        want_read_only=False, r=0.25, momentum_weights_scaling=1.0,
        verbose=False, **kwargs):
        # `params_dict` contains both the parameters and the momentums
        super(ServerSyncAutoAdjustTiming, self).__init__(**kwargs)

        # When `client` is None, you get a kind of "dry_run" version
        # that just waits one second instead of performing the actual updates.
        # This is to be used for debugging.
        self.client = client

        self.D_dropout_probs = D_dropout_probs
        self.names = names
        self.params_dict = params_dict

        # `want_read_only` is True if we want to skip sending updates to the server
        self.want_read_only = want_read_only

        # `r` is the "target_max_ratio_time_spent_synching", and it is set to something less than 1.0
        # if we want to automatically skip updates when those updates would bring the
        # ratio of {time spent communicating with the server / total time} too high.
        # We can't spend more than 100% of our time communicating, so 1.0 is the default
        # value to turn this option off.
        self.r = r


        # TODO : Add argument to scale weights to compensate for dropout.


        ### Private members that are not arguments ###

        self.rolling_estimate_sync_cost = None
        # We'll keep a rolling average so we need to know how
        # to manage the estimate. Use a weighted update.
        self.rolling_estimate_sync_cost_decay = 0.9
        self.timestamp_previous_update = None

        self.verbose = verbose


    def do(self, which_callback, *args):

        if which_callback:

            # use False as default value, but this isn't even read
            want_sync = False

            if self.timestamp_previous_update is None:
                # on the first time ever, you always update
                want_sync = True
            else:
                time_elapsed = time.time() - self.timestamp_previous_update
                assert self.rolling_estimate_sync_cost is not None

                if self.rolling_estimate_sync_cost < self.r * time_elapsed:
                    want_sync = True
                else:
                    want_sync = False

            # update only if needed
            if want_sync:

                tic = time.time()

                ## Do all the sync stuff while the tic-toc timer is running. ##

                if not self.want_read_only:
                    # Write all the parameters.
                    if self.client is not None:
                        for name in self.names:
                            param_value = get_param_value_shared_var(self.params_dict, name)
                            self.client.push_split_param(name, param_value)
                        if self.verbose:
                            print "Client pushing parameters to server."
                    else:
                        # used for debugging, sleep one second
                        time.sleep(1)
                else:
                    if self.verbose:
                        print "Read-only client skips pushing parameters to server."


                if self.client is not None:
                    self.client.perform_split(self.D_dropout_probs)
                    
                    for name in self.names:
                        param_value = self.client.pull_split_param(name)
                        set_param_value_shared_var(self.params_dict, name, param_value)
                    if self.verbose:
                        print "Client pulling parameters to server."
                else:
                    # used for debugging, sleep one second
                    time.sleep(1)


                ## Stop the timer and update the estimate for the rolling_estimate_sync_cost.

                toc = time.time()

                s = self.rolling_estimate_sync_cost_decay
                if self.rolling_estimate_sync_cost is not None:
                    self.rolling_estimate_sync_cost = s * self.rolling_estimate_sync_cost + (1-s) * (toc-tic)
                else:
                    self.rolling_estimate_sync_cost = (toc-tic)

                # One alternative is to use
                #    self.timestamp_previous_update = time.time()
                # but it's even better if we use as the mark the start
                # of this whole method. In that case, having r=0.5 really
                # means that at most half of the time should be spent during updates.
                self.timestamp_previous_update = tic

            # DEBUG
            #if want_sync:
            #    print "Synched with server."
            #else:
            #    print "Skipped sync with server. rolling_estimate_sync_cost is %0.3fs." % self.rolling_estimate_sync_cost
            #    print "%0.3f < %0.3f * %0.3f" % (self.rolling_estimate_sync_cost, self.r, time_elapsed)



