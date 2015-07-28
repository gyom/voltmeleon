
import time
import numpy as np

from common_tools import (set_param_value_shared_var, get_param_value_shared_var)
#from architecture import (set_param_value_shared_var, return_param)

import blocks
from blocks.extensions import SimpleExtension

# this is an old implementation that will be replaced
class ServerUpdateAfterTBatches_Full(SimpleExtension):

    def __init__(self, client, D_dropout_probs, names, D_params, T, **kwargs):
        # `D_params` contains both the parameters and the momentums
        super(ServerUpdateAfterTBatches, self).__init__(after_n_batches=T, **kwargs)

        self.client = client
        self.D_dropout_probs = D_dropout_probs
        self.names = names
        self.D_params = D_params
        self.T = T

    def do(self, which_callback, *args):

        if which_callback in ['after_batch', 'after_n_batches']:

            self.client.perform_split(self.D_dropout_probs)

            for name in self.names:
                param_value = self.client.pull_split_param(name)
                set_param_value_shared_var(self.D_params, name, param_value)

# This is the current implementation, but it's not used
# because we use the `ServerSyncAutoAdjustTiming` instead.
# It's still there because it provides an implementation that's
# easier to read for the purposes of understanding.
class ServerUpdateAfterTBatches(SimpleExtension):

    def __init__(self, client, D_dropout_probs, D_params, T, D_rescale_factor_exo_dropout = {}, **kwargs):
        # `D_params` contains both the parameters and the momentums
        super(ServerUpdateAfterTBatches, self).__init__(every_n_batches=T, **kwargs)

        self.client = client
        self.D_dropout_probs = D_dropout_probs
        self.D_params = D_params
        self.T = T

        # Refer to comment down below in the implementation of `ServerSyncAutoAdjustTiming`.
        self.D_rescale_factor_exo_dropout = D_rescale_factor_exo_dropout

    def do(self, which_callback, *args):
        if which_callback:

            for (name, param_var) in D_params.items():
                param_value = param_var.get_value() / np.float32(self.D_rescale_factor_exo_dropout.get(name, 1.0))
                self.client.push_split_param(name, param_value)

            self.client.perform_split(self.D_dropout_probs)

            for (name, param_var) in D_params.items():
                param_value = self.client.pull_split_param(name) * np.float32(self.D_rescale_factor_exo_dropout.get(name, 1.0))
                shape = param_var.get_value(borrow=True, return_internal_type=True).shape
                param_var.set_value(param_value.reshape(shape))



class ServerSyncAutoAdjustTiming(SimpleExtension):

    # make sure to specify the argument "every_n_batches=T" when you instantiate this extension,
    # or something to that effect to determine how often we want to call it

    def __init__(self, client, D_dropout_probs, D_params,
        want_read_only=False, max_time_ratio_spent=0.25,
        D_rescale_factor_exo_dropout = {},
        verbose=False, **kwargs):
        # `D_params` contains both the parameters and the momentums
        super(ServerSyncAutoAdjustTiming, self).__init__(**kwargs)

        # When `client` is None, you get a kind of "dry_run" version
        # that just waits one second instead of performing the actual updates.
        # This is to be used for debugging.
        self.client = client

        self.D_dropout_probs = D_dropout_probs
        self.D_params = D_params

        # `want_read_only` is True if we want to skip sending updates to the server
        self.want_read_only = want_read_only

        # `max_time_ratio_spent` is the target max ratio time spent synching,
        # and it is set to something less than 1.0
        # if we want to automatically skip updates when those updates would bring the
        # ratio of {time spent communicating with the server / total time} too high.
        # We can't spend more than 100% of our time communicating, so 1.0 is the default
        # value to turn this option off.
        self.max_time_ratio_spent = max_time_ratio_spent


        # Note that `D_rescale_factor_exo_dropout` contains factors
        # by which we will rescale the corresponding parameters after
        # they are read. When committing the values back to the server,
        # we divide by those factors, but in general the clients that
        # use the `D_rescale_factor_exo_dropout` are "observers" that
        # don't write back anything to the server.
        self.D_rescale_factor_exo_dropout = D_rescale_factor_exo_dropout



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

                if self.rolling_estimate_sync_cost < self.max_time_ratio_spent * time_elapsed:
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
                        for (name, param_var) in self.D_params.items():
                            param_value = param_var.get_value() / np.float32(self.D_rescale_factor_exo_dropout.get(name, 1.0))
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

                    for (name, param_var) in self.D_params.items():
                        param_value = self.client.pull_split_param(name) * np.float32(self.D_rescale_factor_exo_dropout.get(name, 1.0))
                        # Works.
                        shape = param_var.get_value(borrow=True, return_internal_type=True).shape
                        # More expensive. Has to be correct.
                        #    shape = param_var.get_value().shape
                        # Also works.
                        #    shape = param_var.shape.eval()

                        if False:
                            print "Reading split parameter %s from server." % name
                            print "The parameter read has shape : %s" % str(param_value.shape)
                            print "The variable on the GPU has shape : %s" % str(param_var.get_value(borrow=True, return_internal_type=True).shape)
                            print "The variable read from the GPU has shape : %s" % str(param_var.get_value().shape)
                            print "param_var.shape.eval() is : %s" % str(param_var.shape.eval())
                            print ""
                            print ""
                            print "indices = client.splits_indices[name]"
                            indices = self.client.splits_indices[name]
                            print indices
                            print "indices[0].shape : %s" % str(indices[0].shape)
                            print "indices[1].shape : %s" % str(indices[1].shape)
                            print "param_desc = client.get_param_desc(name)"
                            print self.client.get_param_desc(name)


                        param_var.set_value(param_value.reshape(shape))
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



