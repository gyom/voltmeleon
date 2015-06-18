# Image net architecture
import numpy as np
import theano
import theano.tensor as T
from blocks.algorithms import GradientDescent
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.training import SharedVariableModifier
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.main_loop import MainLoop
from blocks.roles import WEIGHT
from fuel.datasets.hdf5 import H5PYDataset
from blocks.utils import shared_floatx
floatX = theano.config.floatX
import os
import time

def build_training(cg, error_rate, cost, step_rule,
                   weight_decay_factor=0.0,
                   dataset_hdf5_file=None,
                   batch_size=256,
                   nbr_epochs=1,
                   saving_path=None,
                   server_sync_extension=None,
                   server_sync_initial_read_extension=None,
                   checkpoint_interval_nbr_batches=10):

    if 1e-8 < weight_decay_factor:
        weight_decay_factor = sum([(W**2).mean() for W in VariableFilter(roles=[WEIGHT])(cg.variables)])
        cost = cost + weight_decay_factor
        cost.name = "cost"
        cg = ComputationGraph(cost)

    train_set = H5PYDataset(dataset_hdf5_file, which_sets=('train',))
    valid_set = H5PYDataset(dataset_hdf5_file, which_sets=('valid',))
    data_stream_train = DataStream.default_stream(
            train_set, iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size))   
    data_stream_valid = DataStream.default_stream(
            train_set, iteration_scheme=SequentialScheme(valid_set.num_examples, batch_size))

    timestamp_start_of_experiment = time.time()
    minibatch_timestamp = shared_floatx(np.array([0.0, timestamp_start_of_experiment], dtype=floatX))
    minibatch_timestamp.name = "minibatch_timestamp"
    def update_minibatch_timestamp(_, old_value):
        now = time.time()
        return np.array([now-timestamp_start_of_experiment, now], dtype=floatX)
    minibatch_timestamp_extension = SharedVariableModifier(minibatch_timestamp, update_minibatch_timestamp)

    monitor_valid = DataStreamMonitoring(
        variables=[cost], data_stream=data_stream_valid, prefix="valid", every_n_batches=checkpoint_interval_nbr_batches)

    monitor_train = TrainingDataMonitoring(
        variables=[cost, minibatch_timestamp], prefix="train", every_n_batches=checkpoint_interval_nbr_batches)

    extensions = [  monitor_valid,
                    monitor_train, 
                    FinishAfter(after_n_epochs=nbr_epochs),
                    Printing(every_n_batches=checkpoint_interval_nbr_batches),
                    minibatch_timestamp_extension
                  ]

    if server_sync_extension is not None:
        extensions.append(server_sync_extension)

    if server_sync_initial_read_extension is not None:
        extensions.append(server_sync_initial_read_extension)
    else:
        print "WARNING : You are not using an extension to read the parameters from the server."

    if saving_path is not None:
        print "WARNING : no saving is available right now"

    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=step_rule)

    main_loop = MainLoop(data_stream=data_stream_train,
                        algorithm=algorithm, model = Model(cost),
                        extensions=extensions)

    # TODO : maybe we'll return other things that make sense to have
    return main_loop

