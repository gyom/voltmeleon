import os
import time
import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import Rectifier, Tanh, Logistic, Softmax, MLP, Linear
from blocks.initialization import Constant, Uniform
from blocks.roles import WEIGHT, BIAS, INPUT
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.bricks.conv import (ConvolutionalLayer, ConvolutionalSequence,
                                ConvolutionalActivation, Flattener)
from blocks.bricks.cost import MisclassificationRate, CategoricalCrossEntropy
import h5py
from contextlib import closing

#from momentum import Momentum_dict
import modified_blocks_algorithms as optimisation_rule
import re
from theano.compat import OrderedDict
floatX = theano.config.floatX

# to debug our model that just won't train on MNIST !
import blocks.algorithms

def build_params(input, x, cnn_layer, mlp_layer):

    D_params = {}
    D_kind = {}
    for i, layer in zip(range(len(cnn_layer)), cnn_layer):
        param_layer = VariableFilter(roles=[WEIGHT, BIAS])(ComputationGraph(layer.apply(input).sum()).variables)
        for p in param_layer:
            if p.name =="W":
                kind = "WEIGHTS"
            elif p.name =="b":
                kind = "BIASES"
            else:
                raise Exception("unhandled type of parameters : "
                                "build_params expects only weights or biases (W,b) but received %s", p.name)
            p.name = "layer_%d_%s" % (i, p.name)
            D_params[p.name] = p
            D_kind[p.name] = "CONV_FILTER_"+kind

    for i, layer in zip(range(len(mlp_layer)), mlp_layer):
        param_layer= VariableFilter(roles=[WEIGHT, BIAS])(ComputationGraph(layer.apply(x).sum()).variables)
        for p in param_layer:
            if p.name =="W":
                kind = "WEIGHTS"
            elif p.name =="b":
                kind = "BIASES"
            else:
                raise Exception("unhandled type of parameters : "
                                "build_params expects only weights or biases (W,b) but received %s", p.name)
            p.name = "layer_%d_%s" % (i+len(cnn_layer), p.name)
            D_params[p.name] = p
            D_kind[p.name] = "FULLY_CONNECTED_"+kind

    return D_params, D_kind


def build_submodel(input_shape,
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

    # TO DO : target size and name of the features
    # TO DO : border_mode

    x = T.tensor4('features')
    y = T.imatrix('targets')

    assert len(input_shape) == 3, "input_shape must be a 3d tensor"

    num_channels = input_shape[0]
    image_size = tuple(input_shape[1:])
    print image_size
    print num_channels
    prediction = output_dim

    # CONVOLUTION
    output_conv = x
    output_dim = num_channels*np.prod(image_size)
    conv_layers = []
    assert len(L_dim_conv_layers) == len(L_filter_size)
    assert len(L_dim_conv_layers) == len(L_pool_size)
    assert len(L_dim_conv_layers) == len(L_activation_conv)
    assert len(L_dim_conv_layers) == len(L_endo_dropout_conv_layers)
    assert len(L_dim_conv_layers) == len(L_exo_dropout_conv_layers)

    # regarding the batch dropout : the dropout is applied on the filter
    # which is equivalent to the output dimension
    # you have to look at the dropout_rate of the next layer
    # that is why we need to have the first dropout value of L_exo_dropout_full_layers
    
    # the first value has to be 0.0 in this context, and we'll
    # assume that it is, but let's have an assert
    assert L_exo_dropout_conv_layers[0] == 0.0, "L_exo_dropout_conv_layers[0] has to be 0.0 in this context. There are ways to make it work, of course, but we don't support this with this scripts."

    # here modifitication of L_exo_dropout_conv_layers
    L_exo_dropout_conv_layers = L_exo_dropout_conv_layers[1:] + [L_exo_dropout_full_layers[0]]

    if len(L_dim_conv_layers):
        for (num_filters, filter_size,
            pool_size, activation_str,
            dropout, index) in zip(L_dim_conv_layers,
                                  L_filter_size,
                                  L_pool_size,
                                  L_activation_conv,
                                  L_exo_dropout_conv_layers,
                                  xrange(len(L_dim_conv_layers))
                                  ):

            # convert filter_size and pool_size in tuple
            filter_size = tuple(filter_size)
            pool_size = tuple(pool_size)

            # TO DO : leaky relu
            if activation_str.lower() == 'rectifier':
                activation = Rectifier().apply
            elif activation_str.lower() == 'tanh':
                activation = Tanh().apply
            elif activation_str.lower() in ['sigmoid', 'logistic']:
                activation = Logistic().apply
            else:
                raise Exception("unknown activation function : %s", activation_str)

            assert 0.0 <= dropout and dropout < 1.0
            num_filters = num_filters - int(num_filters*dropout)

            if pool_size[0] == 0 and pool_size[1] == 0:
                layer_conv = ConvolutionalActivation(activation=activation,
                                                filter_size=filter_size,
                                                num_filters=num_filters,
                                                name="layer_%d" % index)
            else:
                layer_conv = ConvolutionalLayer(activation=activation,
                                                filter_size=filter_size,
                                                num_filters=num_filters,
                                                pooling_size=pool_size,
                                                name="layer_%d" % index)

            conv_layers.append(layer_conv)

        convnet = ConvolutionalSequence(conv_layers, num_channels=num_channels,
                                    image_size=image_size,
                                    weights_init=Uniform(width=0.1),
                                    biases_init=Constant(0.0),
                                    name="conv_section")
        convnet.push_allocation_config()
        convnet.initialize()
        output_dim = np.prod(convnet.get_dim('output'))
        output_conv = convnet.apply(output_conv)
        


    output_conv = Flattener().apply(output_conv)

    # FULLY CONNECTED
    output_mlp = output_conv
    full_layers = []
    assert len(L_dim_full_layers) == len(L_activation_full)
    assert len(L_dim_full_layers) + 1 == len(L_endo_dropout_full_layers)
    assert len(L_dim_full_layers) + 1 == len(L_exo_dropout_full_layers)

    # reguarding the batch dropout : the dropout is applied on the filter
    # which is equivalent to the output dimension
    # you have to look at the dropout_rate of the next layer
    # that is why we throw away the first value of L_exo_dropout_full_layers
    L_exo_dropout_full_layers = L_exo_dropout_full_layers[1:]
    pre_dim = output_dim
    if len(L_dim_full_layers):
        for (dim, activation_str,
            dropout, index) in zip(L_dim_full_layers,
                                  L_activation_full,
                                  L_exo_dropout_full_layers,
                                  range(len(L_dim_conv_layers),
                                        len(L_dim_conv_layers)+ 
                                        len(L_dim_full_layers))
                                   ):
                                          
                # TO DO : leaky relu
                if activation_str.lower() == 'rectifier':
                    activation = Rectifier().apply
                elif activation_str.lower() == 'tanh':
                    activation = Tanh().apply
                elif activation_str.lower() in ['sigmoid', 'logistic']:
                    activation = Logistic().apply
                else:
                    raise Exception("unknown activation function : %s", activation_str)

                assert 0.0 <= dropout and dropout < 1.0
                dim = dim - int(dim*dropout)

                layer_full = MLP(activations=[activation], dims=[pre_dim, dim],
                                 weights_init=Uniform(width=0.1),
                                 biases_init=Constant(0.0),
                                name="layer_%d" % index)
                layer_full.initialize()
                full_layers.append(layer_full)
                pre_dim = dim

        for layer in full_layers:
            output_mlp = layer.apply(output_mlp)

        output_dim = L_dim_full_layers[-1] - int(L_dim_full_layers[-1]*L_exo_dropout_full_layers[-1])

    # COST FUNCTION
    output_layer = Linear(output_dim, prediction,
                          weights_init=Uniform(width=0.1),
                          biases_init=Constant(0.0),
                          name="layer_"+str(len(L_dim_conv_layers)+ 
                                            len(L_dim_full_layers))
                          )
    output_layer.initialize()
    full_layers.append(output_layer)
    y_pred = output_layer.apply(output_mlp)
    y_hat = Softmax().apply(y_pred)
    # SOFTMAX and log likelihood
    y_pred = Softmax().apply(y_pred)
    # be careful. one version expects the output of a softmax; the other expects just the
    # output of the network
    cost = CategoricalCrossEntropy().apply(y.flatten(), y_pred)
    #cost = Softmax().categorical_cross_entropy(y.flatten(), y_pred)
    cost.name = "cost"

    # Misclassification
    error_rate_brick = MisclassificationRate()
    error_rate = error_rate_brick.apply(y.flatten(), y_hat)
    error_rate.name = "error_rate"

    # put names

    D_params, D_kind = build_params(x, T.matrix(), conv_layers, full_layers)
    # test computation graph
    

    # TO DO : weight decay factor in the second json file !!!

    cg = ComputationGraph(cost)


    # DROPOUT
    L_endo_dropout = L_endo_dropout_conv_layers + L_endo_dropout_full_layers

    cg_dropout = cg
    inputs = VariableFilter(roles=[INPUT])(cg.variables)

    print inputs

    # TODO : print inputs to check
    for (index, drop_rate) in enumerate(L_endo_dropout):
        
        for input_ in inputs:
            m = re.match(r"layer_(\d+)_apply.*", input_.name)
            if m and index == int(m.group(1)):
                cg_dropout = apply_dropout(cg, [input_], drop_rate)
                print "Applied dropout %f on %s." % (drop_rate, input_.name)
                break


    cg = cg_dropout

    return (cg, error_rate, cost, D_params, D_kind)



def build_step_rule_parameters(step_flavor, D_params, D_kind):

    if step_flavor['method'].lower() == "rmsprop":
        assert 0.0 <= step_flavor['learning_rate']
        assert 0.0 <= step_flavor['decay_rate']
        assert step_flavor['decay_rate'] <= 1.0

        step_rule = optimisation_rule.RMSProp(D_params=D_params, D_kind=D_kind,
                                              learning_rate=step_flavor['learning_rate'],
                                              decay_rate=step_flavor['decay_rate'])

    elif step_flavor['method'].lower() == "adadelta":
        # maybe this should be a composite rule with a learning rate also
        assert 0.0 <= step_flavor['decay_rate']
        assert step_flavor['decay_rate'] <= 1.0

        step_rule = optimisation_rule.AdaDelta(D_params=D_params, D_kind=D_kind,
                                               decay_rate=step_flavor['decay_rate'])
                                               

    elif step_flavor['method'].lower() == "adam":
        assert 0.0 <= step_flavor['learning_rate']

        optional = {}
        for key in ['beta1', 'beta2', 'epsilon', 'decay_factor']:
            if step_flavor.has_key(key):
                optional[key] = step_flavor[key]

        step_rule = optimisation_rule.Adam(D_params=D_params, D_kind=D_kind,
                                           learning_rate=step_flavor['learning_rate'],
                                           **optional)


    elif step_flavor['method'].lower() == "adagrad":
        assert 0.0 <= step_flavor['learning_rate']

        optional = {}
        for key in ['epislon']:
            if step_flavor.has_key(key):
                optional[key] = step_flavor[key]

        step_rule = optimisation_rule.AdaGrad(D_params=D_params, D_kind=D_kind,
                                              learning_rate=step_flavor['learning_rate'],
                                              **optional)

    elif step_flavor['method'].lower() == "momentum":
        assert 0.0 <= step_flavor['learning_rate']
        assert 0.0 <= step_flavor['momentum']
        assert step_flavor['momentum'] <= 1.0

        step_rule = optimisation_rule.Momentum(D_params=D_params, D_kind=D_kind,
                                               learning_rate=step_flavor['learning_rate'],
                                               momentum=step_flavor['momentum'])
        #step_rule = blocks.algorithms.Momentum(learning_rate=step_flavor['learning_rate'], momentum=step_flavor['momentum'])
        
    elif step_flavor['method'].lower() == "noupdates":
        step_rule = optimisation_rule.NoUpdates()
    else:
        raise Error("Unrecognized step flavor method : " + step_flavor['method'])

    # This is because we debug things by using the blocks.algorithms implementations
    # and they don't have these attributes.
    if hasattr(step_rule, "velocities"):
        D_additional_params = step_rule.velocities
    else:
        D_additional_params = OrderedDict()
    if hasattr(step_rule, "D_kind"):
        D_additional_kind = step_rule.D_kind
    else:
        D_additional_kind = {}

    # TODO : Make sure that these variables are indeed dictionaries and not lists.
    #        Remove this afterwards.

    assert isinstance(D_additional_params, OrderedDict)
    assert type(D_additional_kind) == dict

    return (step_rule, D_additional_params, D_additional_kind)




def get_model_desc_for_server(D_params, D_kind):

    """
    Takes a dictionary of the parameters (shared variables) and generates a JSON structure
    like model_params_desc.json to be used by the server executable.

    Note that this function generates a structures and the act of saving it
    as a json file is going to be done elsewhere.
    """

    # TODO : know CONV or FULLY CONNECTED layer
    # it will help to do the reshape too
    L_server_params_desc=[]

    for (param_name, param) in D_params.items():
        params_dict={}
        params_dict["name"] = param_name
        params_dict["kind"] = D_kind[param_name]

        # voir les tailles pour les reshape
        if D_kind[param.name]=="CONV_FILTER_WEIGHTS":
            params_dict["shape"] = list(param.get_value().shape)
        elif D_kind[param.name]=="CONV_FILTER_BIASES":
            E = list(param.get_value().shape)
            params_dict["shape"] = [E[0], 1, E[1], E[2]]
        elif D_kind[param.name]=="FULLY_CONNECTED_WEIGHTS":
            params_dict["shape"] = list(param.get_value().shape) + [1, 1]
        elif D_kind[param.name]=="FULLY_CONNECTED_BIASES":
            params_dict["shape"] = [1] + list(param.get_value().shape) + [1, 1]
        else:
            raise Exception("unknow kind of parameters : %s for param %s",
                            D_kind[param_name],
                            param_name)

        assert len(params_dict["shape"])==4
        L_server_params_desc.append(params_dict)

    return L_server_params_desc



