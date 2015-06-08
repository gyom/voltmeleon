# Image net architecture
import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import Rectifier, Softmax, MLP, Identity
from blocks.initialization import Constant, Uniform, IsotropicGaussian
from blocks.algorithms import GradientDescent
from blocks.roles import WEIGHT, BIAS, INPUT
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Dump
from blocks.extensions.training import SharedVariableModifier
from blocks.model import Model
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.main_loop import MainLoop
import operator
from blocks.roles import PARAMETER
#from batch_normalize import ConvolutionalLayer, ConvolutionalActivation, Linear
# change for cpu tests
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence
from blocks.bricks.conv import MaxPooling, ConvolutionalActivation, Flattener
from blocks.bricks import Linear

from blocks.bricks.cost import MisclassificationRate, CategoricalCrossEntropy
from blocks.algorithms import Momentum, RMSProp, AdaDelta, AdaGrad, Adam
from fuel.datasets.hdf5 import H5PYDataset
from fuel.transformers import Flatten
from blocks.utils import shared_floatx

floatX = theano.config.floatX
import h5py
from contextlib import closing

#from momentum import Momentum_dict
import momentum

import os
import time

def init_param(params, name, value):

    if name in params:
        param_i = params[name]
        shape = param_i.get_value().shape
        #print (name, shape, value.shape)
        param_i.set_value((value.reshape(shape)).astype(floatX))
    else:
        raise Exception("unknown parameter")


def return_param(params, name):
    #if name in momentum :
    #    return momentum[name].get_value()
    #else :
    if name in params:
        return params[name].get_value()
    else:
        raise Exception("unknown parameter")


def errors(p_y_given_x, y):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero one
    loss over the size of the minibatch

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    """
    y = T.cast(y, 'int8')
    y_pred = T.argmax(p_y_given_x, axis=1)
    y_pred = y_pred.dimshuffle((0, 'x'))
    y_pred = T.cast(y_pred, 'int8')
    # check if y has same dimension of y_pred
    if y.ndim != y_pred.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('y', y.type, 'y_pred', y_pred.type)
        )
    # check if y is of the correct datatype
    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(y_pred, y), dtype=floatX)
    else:
        raise NotImplementedError()


def build_params(input, x, cnn_layer, mlp_layer):

    params = []
    names = []
    for i, layer in zip(range(len(cnn_layer)), cnn_layer):
        param_layer = VariableFilter(roles=[WEIGHT, BIAS])(ComputationGraph(layer.apply(input).sum()).variables)
        for p in param_layer:
            p.name = "layer_"+str(i)+"_"+p.name
            names.append(p.name)
            params.append(p)
            print (p.name, p.get_value().shape)
        # do the same for the input -> easier for the second level of dropout
        #param_layer = VariableFilter(roles=[INPUT])(ComputationGraph(layer.apply(input).sum()).variables)
        #for p in param_layer:
        #    p.name = "layer_"+str(i)+"_input"     

    for i, layer in zip(range(len(mlp_layer)), mlp_layer):
        param_layer= VariableFilter(roles=[WEIGHT, BIAS])(ComputationGraph(layer.apply(x).sum()).variables)
        for p in param_layer:
            p.name = "layer_"+str(i+len(cnn_layer))+"_"+p.name
            names.append(p.name)
            params.append(p)
            print (p.name, p.get_value().shape)
        #param_layer= VariableFilter(roles=[INPUT])(ComputationGraph(layer.apply(x).sum()).variables)
        #for p in param_layer:
        #    p.name = "layer_"+str(i+len(cnn_layer))+"_input"

    return params, names


        
def build_architecture( step_flavor,
                        drop_conv, drop_mlp,
                        L_nbr_filters, L_nbr_hidden_units,
                        weight_decay_factor=0.0,
                        dataset_hdf5_file=None):

    # dataset_hdf5_file is like "/rap/jvb-000-aa/data/ImageNet_ILSVRC2010/pylearn2_h5/imagenet_2010_train.h5"

    assert len(L_nbr_filters) == 4
    assert len(L_nbr_hidden_units) == 4
    # Note that L_nb_hidden_units[0] is something constrained
    # by the junction between the filters and the fully-connected section.
    # Find this value in the JSON file that describes the model.

    x=T.tensor4('x')
    
    if dataset_hdf5_file is not None:
        with closing(h5py.File(dataset_hdf5_file, 'r')) as f:
            x_mean = (f['x_mean']).value
            x_mean = x_mean.reshape((1, 3, 32, 32)) # equivalent to a dimshuffle on the number of elem in the batch
        x = (x - x_mean)
        # TODO : maybe normalize ?
    
    y = T.imatrix('y')
    
    num_channels = 3
    filter_size = (3, 3)
    activation = Identity().apply
    #activation = Rectifier().apply
    ########
    num_filters = L_nbr_filters[0] - int(drop_conv[0]*L_nbr_filters[0])
    layer0 = ConvolutionalActivation(activation, filter_size, num_filters,
                              num_channels,
                              weights_init=IsotropicGaussian(0.1),
                              biases_init=Uniform(width=0.1), name="layer_0")

    
    num_channels = num_filters
    num_filters = L_nbr_filters[1] - int(drop_conv[1]*L_nbr_filters[1])
    filter_size = (3,3)
    layer1 = ConvolutionalActivation(activation, filter_size, num_filters,
                              num_channels,
                              weights_init=IsotropicGaussian(0.1),
                              biases_init=Uniform(width=0.1), name="layer_1")

    
    num_channels = num_filters
    num_filters = L_nbr_filters[2] - int(drop_conv[2]*L_nbr_filters[2])
    filter_size=(3,3)
    pooling_size = 2   
    layer2 = ConvolutionalLayer(activation, filter_size, num_filters,
                              (pooling_size, pooling_size),
                              num_channels,
                              weights_init=IsotropicGaussian(0.1),
                              biases_init=Uniform(width=0.1), name="layer_2")

    
    num_channels = num_filters
    num_filters = L_nbr_filters[3] - int(drop_conv[3]*L_nbr_filters[3])
    filter_size=(2,2)
    pooling_size = 2
    layer3 = ConvolutionalLayer(activation, filter_size, num_filters,
                              (pooling_size, pooling_size),
                              num_channels,
                              weights_init=IsotropicGaussian(0.1),
                              biases_init=Uniform(width=0.1), name="layer_3")


    ####################################################


    conv_layers = [layer0, layer1, layer2, layer3]
    convnet = ConvolutionalSequence(conv_layers, num_channels= 3,
                                    image_size=(32, 32))
    convnet.initialize()
    output_dim = np.prod(convnet.get_dim('output'))
    # Fully connected layers

    output_conv = Flattener().apply(convnet.apply(x))

    nbr_classes_to_predict = 10 # because it's SVHN


    #nbr_hidden_units = [output_dim, 1024, 512, 512]
    padded_drop_mlp = drop_mlp

    L_nbr_hidden_units_left = [ n-int(p*n) for (n,p) in zip(L_nbr_hidden_units, padded_drop_mlp) ]
    # add afterwards the final number of hidden units to the number of classes to predict
    mlp_dim_pairs = zip(L_nbr_hidden_units_left, L_nbr_hidden_units_left[1:]) + [(L_nbr_hidden_units_left[-1], nbr_classes_to_predict)]


    assert L_nbr_hidden_units_left[0] == output_dim, "%d is not %d" % (L_nbr_hidden_units_left[0], output_dim)

    print "mlp_dim_pairs"
    print mlp_dim_pairs

    # MLP
    sequences_mlp = []
    mlp_layer = []

    mlp_layer0 = Linear(mlp_dim_pairs[0][0], mlp_dim_pairs[0][1],
                        weights_init=IsotropicGaussian(0.1),
                        biases_init=Uniform(width=0.1), name="layer_4")
    mlp_layer0.initialize()
    sequences_mlp += [mlp_layer0, Rectifier(name="layer4_1")]
    mlp_layer.append(mlp_layer0)




    mlp_layer1 = Linear(mlp_dim_pairs[1][0], mlp_dim_pairs[1][1],
                        weights_init=IsotropicGaussian(0.1),
                        biases_init=Uniform(width=0.1), name="layer_5")
    mlp_layer1.initialize()
    mlp_layer.append(mlp_layer1)
    sequences_mlp += [mlp_layer1, Rectifier(name="layer5_1")]

    mlp_layer2 = Linear(mlp_dim_pairs[2][0], mlp_dim_pairs[2][1],
                        weights_init=IsotropicGaussian(0.1),
                        biases_init=Uniform(width=0.1), name="layer_6")
    mlp_layer2.initialize()
    mlp_layer.append(mlp_layer2)
    sequences_mlp += [mlp_layer2, Rectifier(name="layer6_1")]




    # having Identity() there is a trick from Eloi and Bart
    # to get something more stable numerically by having the
    # software (or cross-entropy) come later
    
    #mlp_layer3 = MLP([Identity()],
    #                 [mlp_dim_pairs[3][0], mlp_dim_pairs[3][1]],
    #                 weights_init=IsotropicGaussian(0.1),
    #                 biases_init=Uniform(width=0.1), name="layer7")

    mlp_layer3 = Linear(mlp_dim_pairs[3][0], mlp_dim_pairs[3][1],
                        weights_init=IsotropicGaussian(0.1),
                        biases_init=Uniform(width=0.1), name="layer_7")
    mlp_layer.append(mlp_layer3)
    mlp_layer3.initialize()
    sequences_mlp += [mlp_layer3]
    
    #print "mlp_dim_pairs[3] is "
    #print mlp_dim_pairs[3]

    output_test_0 = Rectifier(name="toto1").apply(mlp_layer0.apply(output_conv))
    output_test_1 = Rectifier(name="toto2").apply(mlp_layer1.apply(output_test_0))
    output_test_2 = Rectifier(name="toto3").apply(mlp_layer2.apply(output_test_1))
    output_test_3 = mlp_layer3.apply(output_test_2)
    output_full = output_test_3

    #output_full = output_conv
    #for i in xrange(len(sequences_mlp)):
    #    output_full = sequences_mlp[i].apply(output_full)

    

    # sanity check (optional)
    """
    arbitrary_tmp_batch_size = 834
    f = theano.function([x], output_full)
    value_x = np.random.ranf((arbitrary_tmp_batch_size, 3, 32, 32)).astype(np.float32)
    A = f(value_x)
    print A.shape
    assert A.shape == (arbitrary_tmp_batch_size, nbr_classes_to_predict)
    """

    # Numerical stable softmax
    output_full = Softmax().apply(output_full)
    cost = CategoricalCrossEntropy().apply( y.flatten(), output_full )

    #cost = Softmax().categorical_cross_entropy(y.flatten(), output_full)
    #cost = (output_full-1).norm(2) + 0.001*y.norm(2)


    if 1e-8 < weight_decay_factor:
        cg = ComputationGraph(cost)
        weight_decay_factor = sum([(W**2).mean() for W in VariableFilter(roles=[WEIGHT])(cg.variables)])
        cost = cost + weight_decay_factor

    cost.name = 'cost'
    cg = ComputationGraph(cost)

    #DEBUG : amplitude de la sortie
    #tmp = Flattener().apply(convnet.apply(x))
    #output_test_0 = mlp_layer0.apply(tmp)
    #output_test_1 = mlp_layer1.apply(output_test_0)
    #output_test_2 = mlp_layer2.apply(output_test_1)
    #output_test_3 = mlp_layer3.apply(output_test_2)
    diagnostic_output = output_full[0,:]
    diagnostic_output.name = "diagnostic_output"


    # put names
    params, names = build_params(x, T.matrix(), conv_layers, mlp_layer)
    # test computation graph
    error_rate_brick = MisclassificationRate()
    error_rate = error_rate_brick.apply(y.flatten(), output_full)

    #error_rate = errors(output_full, y)
    error_rate.name = 'error'
    ###step_rule = Momentum_dict(learning_rate, momentum, params=params)
    

    print "params right out of blocks"
    print params
    for p in params:
        print "%s" % p.name
        print p.get_value().shape



    if step_flavor['method'].lower() == "rmsprop":
        assert 0.0 <= step_flavor['learning_rate']
        assert 0.0 <= step_flavor['decay_rate']
        assert step_flavor['decay_rate'] <= 1.0

        step_rule = momentum.RMSProp(learning_rate=step_flavor['learning_rate'],
                            decay_rate=step_flavor['decay_rate'], params=params)

        #step_rule = RMSProp(learning_rate=step_flavor['learning_rate'],
        #                    decay_rate=step_flavor['decay_rate'])

    elif step_flavor['method'].lower() == "adadelta":
        # maybe this should be a composite rule with a learning rate also
        assert 0.0 <= step_flavor['decay_rate']
        assert step_flavor['decay_rate'] <= 1.0

        step_rule = AdaDelta(decay_rate=step_flavor['decay_rate'])

    elif step_flavor['method'].lower() == "adam":
        assert 0.0 <= step_flavor['learning_rate']

        optional = {}
        for key in ['beta1', 'beta2', 'epsilon', 'decay_factor']:
            if step_flavor.has_key(key):
                optional[key] = step_flavor[key]

        step_rule = Adam(learning_rate=step_flavor['learning_rate'], **optional)

    elif step_flavor['method'].lower() == "momentum":
        assert 0.0 <= step_flavor['learning_rate']
        assert 0.0 <= step_flavor['momentum']
        assert step_flavor['momentum'] <= 1.0

        step_rule = Momentum(   learning_rate=step_flavor['learning_rate'],
                                momentum=step_flavor['momentum'])

        #step_rule = momentum.Momentum_dict( learning_rate=step_flavor['learning_rate'],
        #                                    momentum=step_flavor['momentum'])

    else:
        raise Error("Unrecognized step flavor method : " + step_flavor['method'])



    # TODO : After you fix the whole velocities thing, you need to
    #        add the stuff here so that we keep track of them also.

    ## DEBUG : The role of this part of the code is questioned.
    dict_params = step_rule.velocities
    #dict_params = {}
    for param_m in dict_params :
        print param_m
        names.append(param_m)
    for param in params:
        dict_params[param.name] = param
    ####################
    
    return (cg, error_rate, cost, step_rule, names, dict_params, diagnostic_output)
