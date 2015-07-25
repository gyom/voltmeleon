# Image net architecture
import numpy as np
import theano
import theano.tensor as T
from blocks.algorithms import GradientDescent
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
#from blocks.extensions.training import SharedVariableModifier
from blocks.extensions import SimpleExtension
from blocks.extensions.saveload import Checkpoint
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.main_loop import MainLoop
from blocks.roles import WEIGHT
from fuel.datasets.hdf5 import H5PYDataset
from blocks.utils import shared_floatx
from schemes import LimitedScheme
floatX = theano.config.floatX
import os
import time

def build_training(cg, error_rate, cost, step_rule,
                   weight_decay_factor=0.0,
                   hdf5_file=None,
                   want_eval_on_valid=False,
                   want_eval_on_test=False,
                   want_subset_valid=False,
                   want_subset_test=False,
                   batch_size=256,
                   nbr_epochs=1,
                   saving_path=None,
                   server_sync_extension=None,
                   server_sync_initial_read_extension=None,
                   want_save_model_best_valid_model=False,
                   monitor_interval_nbr_batches=10):

    if 1e-8 < weight_decay_factor:
        weight_decay_factor = sum([(W**2).mean() for W in VariableFilter(roles=[WEIGHT])(cg.variables)])
        cost = cost + weight_decay_factor
        cost.name = "cost"
        cg = ComputationGraph(cost)


    extra_variables_to_monitor = []
    for W in VariableFilter(roles=[WEIGHT])(cg.variables):
        e = T.abs_(W).mean()
        e.name = W.name + "_absmean"
        extra_variables_to_monitor.append(e)
    
    
    # ugly hack
    import inspect
    class SharedVariableModifier(SimpleExtension):
        def __init__(self, parameter, function, **kwargs):
            kwargs.setdefault("after_batch", True)
            super(SharedVariableModifier, self).__init__(**kwargs)
            self.parameter = parameter
            self.function = function
            self.num_args = len(inspect.getargspec(function).args)

        def __getstate__(self):
            return {}
            #return False

        def __setstate__(self, state):
            pass

        def do(self, which_callback, *args):
            iterations_done = self.main_loop.log.status['iterations_done']
            if self.num_args == 1:
                new_value = self.function(iterations_done)
            else:
                old_value = self.parameter.get_value()
                new_value = self.function(iterations_done, old_value)
            self.parameter.set_value(new_value)

    
    timestamp_start_of_experiment = time.time()
    minibatch_timestamp = shared_floatx(np.array([0.0, timestamp_start_of_experiment], dtype=floatX))
    minibatch_timestamp.name = "minibatch_timestamp"
    def update_minibatch_timestamp(_, old_value):
        now = time.time()
        return np.array([now-timestamp_start_of_experiment, now], dtype=floatX)
    SharedVariableModifier.__getstate__
    minibatch_timestamp_extension = SharedVariableModifier(minibatch_timestamp, update_minibatch_timestamp)
    #extra_variables_to_monitor.append(minibatch_timestamp)




    train_set = H5PYDataset(hdf5_file, which_sets=('train',))
    data_stream_train = DataStream.default_stream(train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size))
    #data_stream_train = DataStream.default_stream(train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size))
    
    monitor_train = TrainingDataMonitoring(
        variables=[cost, error_rate] + extra_variables_to_monitor, prefix="train", every_n_batches=monitor_interval_nbr_batches)

    if want_eval_on_valid:
        valid_set = H5PYDataset(hdf5_file, which_sets=('valid',))
        if want_subset_valid:
            # DEBUG : This probably causes a problem because it's not serializable or something like that.
            data_stream_valid = DataStream.default_stream(
                valid_set, iteration_scheme=LimitedScheme(ShuffledScheme(valid_set.num_examples, batch_size), 2000))
        else:
            data_stream_valid = DataStream.default_stream(
                #valid_set, iteration_scheme=ShuffledScheme(valid_set.num_examples, batch_size))
                valid_set, iteration_scheme=SequentialScheme(valid_set.num_examples, batch_size))

        monitor_valid = DataStreamMonitoring(
            variables=[cost, error_rate], data_stream=data_stream_valid, prefix="valid", every_n_batches=monitor_interval_nbr_batches)
    else:
        monitor_valid = None

    if want_eval_on_test:
        test_set = H5PYDataset(hdf5_file, which_sets=('test',))
        if want_subset_test:
            data_stream_test = DataStream.default_stream(
                test_set, iteration_scheme=LimitedScheme(ShuffledScheme(test_set.num_examples, batch_size), 2000))
        else:
            data_stream_test = DataStream.default_stream(
                test_set, iteration_scheme=ShuffledScheme(test_set.num_examples, batch_size))
        monitor_test = DataStreamMonitoring(
            variables=[cost, error_rate], data_stream=data_stream_test, prefix="test", every_n_batches=monitor_interval_nbr_batches)
    else:
        monitor_test = None



    extensions = (  [monitor_train] +
                    [e for e in (monitor_valid, monitor_test) if e is not None] +
                    [FinishAfter(after_n_epochs=nbr_epochs),
                     Printing(every_n_batches=monitor_interval_nbr_batches)] )
                     #minibatch_timestamp_extension] )



    if server_sync_extension is not None:
        extensions.append(server_sync_extension)

    if server_sync_initial_read_extension is not None:
        extensions.append(server_sync_initial_read_extension)
    else:
        print "WARNING : You are not using an extension to read the parameters from the server."

    if saving_path is not None:
        #print "WARNING : Checkpoint not supported yet."
        assert isinstance(saving_path, str)
        assert os.path.isdir(os.path.dirname(saving_path)), "The directory for saving_path (%s) does not exist." % saving_path
        if want_save_model_best_valid_model:
            #extensions.append(Checkpoint_observer(path=saving_path, use_cpickle=True, save_separately=['log', 'model'],
            #                             every_n_batches=monitor_interval_nbr_batches))
            extensions.append(Checkpoint_observer(path=saving_path, save_separately=['log', 'model'],
                                         every_n_batches=monitor_interval_nbr_batches))
        else:
            #extensions.append(Checkpoint(path=saving_path, use_cpickle=True, save_separately=['log'],
            #                             every_n_batches=monitor_interval_nbr_batches))
            extensions.append(Checkpoint(path=saving_path, save_separately=['log'],
                                         every_n_batches=monitor_interval_nbr_batches))

    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=step_rule)

    main_loop = MainLoop(data_stream=data_stream_train,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)

    # TODO : maybe we'll return other things that make sense to have
    return main_loop

