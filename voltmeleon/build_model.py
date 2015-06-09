import os
import time
import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import Rectifier, Tanh, Sigmoid, Softmax, MLP, Linear
from blocks.initialization import Constant
from blocks.roles import WEIGHT, BIAS, INPUT
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence, ConvolutionalActivation
from blocks.bricks.cost import MisclassificationRate
import h5py
from contextlib import closing

#from momentum import Momentum_dict
import modified_blocks_algorithms as optimisation_rule
floatX = theano.config.floatX


def build_params(input, x, cnn_layer, mlp_layer):

    D_params = {}
    D_kind = {}
    for i, layer in zip(range(len(cnn_layer)), cnn_layer):
        param_layer = VariableFilter(roles=[WEIGHT, BIAS])(ComputationGraph(layer.apply(input).sum()).variables)
        for p in param_layer:
            if p.name =="W":
                kind = "WEIGHTS"
            elif p.name =="b":
                kind = "BIASES":
            else
                raise Exception("unhandled type of parameters : "
                                "build_params expects only weights or biases (W,b) but received %s", p.name)
            p.name = "layer_"+str(i)+"_"+p.name
            D_params[p.name] = p
            D_kind[p.name] = "CONV_FILTER_"+kind

    for i, layer in zip(range(len(mlp_layer)), mlp_layer):
        param_layer= VariableFilter(roles=[WEIGHT, BIAS])(ComputationGraph(layer.apply(x).sum()).variables)
        for p in param_layer:
            if p.name =="W":
                kind = "WEIGHTS"
            elif p.name =="b":
                kind = "BIASES":
            else
                raise Exception("unhandled type of parameters : "
                                "build_params expects only weights or biases (W,b) but received %s", p.name)
            p.name = "layer_"+str(i+len(cnn_layer))+"_"+p.name
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

    x = T.tensor4('x')
    y = T.imatrix('y')

    assert len(input_shape) == 3, "input_shape must be a 3d tensor"

    num_channels = input_shape[0]
    image_size = input_shape[1:]
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
    if len(L_dim_conv_layers):
        for num_filters, filter_size,
            pool_size, activation_str,
            dropout, index in zip(L_dim_conv_layers,
                                  L_filter_size,
                                  L_pool_size,
                                  L_activation_conv,
                                  L_exo_dropout_conv_layers,
                                  xrange(len(L_dim_conv_layers))
                                  )

            # TO DO : leaky relu
            if activation_str.lower() == 'rectifier':
                activation = Rectifier()
            elif activation_str.lower() == 'tanh':
                activation = Tanh()
            elif activation_str.lower() == 'sigmoid':
                activation = Sigmoid()
            else
                raise Exception("unknown activation function : %s", activation_str)

            assert dropout >= 0. and dropout < 1.
            num_filters = num_filters - int(num_filters*dropout)

            if pool_size[0] == 0 and pool_size[1] == 0:
                layer_conv = ConvolutionalActivation(activation=activation,
                                                filter_size=filter_size,
                                                num_filters=num_filters,
                                                name="layer_"+str(index))
            else
                layer_conv = ConvolutionalLayer(activation=activation,
                                                filter_size=filter_size,
                                                num_filters=num_filters,
                                                pooling_size=poolin_size,
                                                name="layer_"+str(index))

            conv_layers.append(layer_conv)
    
        convnet = ConvolutionalSequence(conv_layers, num_channels=num_channels,
                                    image_size=image_size,
                                    weights_init=Constant(0.),
                                    biases_init=Constant(0.))
        convnet.initialize()
        output_dim = np.prod(convnet.get_dim('output'))
        output_conv = convnet.apply(output_conv)


    output_conv = Flattener().apply(output_conv)

    # FULLY CONNECTED
    output_mlp = output_conv
    full_layers = []
    assert len(L_dim_full_layers) == len(L_activation_full)
    assert len(L_dim_full_layers) == len(L_endo_dropout_full_layers)
    assert len(L_dim_full_layers) == len(L_exo_dropout_full_layers)

    if len(L_dim_full_layers):
        for dim, activation_str,
            dropout, index in zip(L_dim_full_layers,
                                  L_activation_full,
                                  L_exo_dropout_full_layers
                                  range(len(L_dim_conv_layers),
                                             len(L_dim_conv_layers)+ 
                                             len(L_dim_full_layers))
                                         ):
                                          
                # TO DO : leaky relu
                if activation_str.lower() == 'rectifier':
                    activation = Rectifier()
                elif activation_str.lower() == 'tanh':
                    activation = Tanh()
                elif activation_str.lower() == 'sigmoid':
                    activation = Sigmoid()
                else
                    raise Exception("unknown activation function : %s", activation_str)

                assert dropout >= 0. and dropout < 1.
                dim = dim - int(dim*dropout)

                layer_full = MLP(activations=[activation], dims=[dim],
                                weights_init=Constant(0.),
                                biases_init=Constant(0.),
                                name="layer_"+str(index))
                layer_full.initialize()
                full_layers.append(layer_full)

        for layer in full_layers:
            output_mlp = layer.apply(output_mlp)

        output_dim = L_dim_full_layers[-1] - int(L_dim_full_layers[-1]*L_exo_dropout_full_layers[-1])

    # COST FUNCTION
    output_layer = Linear(output_dim, prediction,
                          weights_init=Constant(0.),
                          biases_init=Constant(0.),
                          name="layer_"+str(len(L_dim_conv_layers)+ 
                                            len(L_dim_full_layers))
                          )
    output_layer.initialize()
    y_pred = output_layer.apply(output_mlp)
    y_hat = Softmax().apply(y_pred)
    # SOFTMAX and log likelihood
    cost = Softmax().categorical_cross_entropy(y.flatten(), y_pred)
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
    L_endo_dropout = L_endo_dropout_conv_layers +
                     L_endo_dropout_full_layers

    cg_dropout = cg
    inputs = VariableFilter(roles=[INPUT])(cg.variables)
    for drop_rate, index in zip(L_endo_dropout, range(len(L_endo_dropout))):
        
        for input_ in inputs:
            if re.match("layer_"+str(index)+"_apply_input_", input_):
                cg_dropout = apply_dropout(cg, [input_], drop_rate)
                break

    cg = cg_dropout


    return (cg, error_rate, cost, D_params, D_kind)


def build_submodel_old(drop_conv, drop_mlp,
                   L_nbr_filters, L_nbr_hidden_units,
                   weight_decay_factor=0.0):

    # dataset_hdf5_file is like "/rap/jvb-000-aa/data/ImageNet_ILSVRC2010/pylearn2_h5/imagenet_2010_train.h5"

    # We can change the architecture now
    # assert len(L_nbr_filters) == 4
    # assert len(L_nbr_hidden_units) == 4


    # Note that L_nb_hidden_units[0] is something constrained
    # by the junction between the filters and the fully-connected section.
    # Find this value in the JSON file that describes the model.

    x=T.tensor4('x')
    
    # preprocessing should be applied in the hdf5 file
    """
    if dataset_hdf5_file is not None:
        with closing(h5py.File(dataset_hdf5_file, 'r')) as f:
            x_mean = (f['x_mean']).value
            x_mean = x_mean.reshape((1, 3, 32, 32)) # equivalent to a dimshuffle on the number of elem in the batch
        x = (x - x_mean)
        # TODO : maybe normalize ?
    
    y = T.imatrix('y')
    """

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


    # put names
    L_params, names = build_params(x, T.matrix(), conv_layers, mlp_layer)
    # test computation graph
    error_rate_brick = MisclassificationRate()
    error_rate = error_rate_brick.apply(y.flatten(), output_full)

    #error_rate = errors(output_full, y)
    error_rate.name = 'error'
    ###step_rule = Momentum_dict(learning_rate, momentum, params=params)
    

    print "params right out of blocks"
    print L_params
    for p in L_params:
        print "%s" % p.name
        print p.get_value().shape

    D_params D_kind = {}
    for param in L_params:
        D_params[param.name] = param
    ####################
    
    return (cg, error_rate, cost, names, D_params)


def build_step_rule_parameters(step_flavor, D_params, D_kind):

    # TO DO
    if step_flavor['method'].lower() == "rmsprop":
        assert 0.0 <= step_flavor['learning_rate']
        assert 0.0 <= step_flavor['decay_rate']
        assert step_flavor['decay_rate'] <= 1.0

        # TODO : change momentum.RMSProp to take D_params instead of L_params
        step_rule = optimisation_rule.RMSProp(learning_rate=step_flavor['learning_rate'],
                            decay_rate=step_flavor['decay_rate'], D_params=D_params, D_kind = D_kind)

    # TO DO : implement this method
    """
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
    """
    elif step_flavor['method'].lower() == "momentum":
        assert 0.0 <= step_flavor['learning_rate']
        assert 0.0 <= step_flavor['momentum']
        assert step_flavor['momentum'] <= 1.0

        step_rule = optimisation_rule.Momentum(   learning_rate=step_flavor['learning_rate'],
                                momentum=step_flavor['momentum'],
                                D_params=D_params,
                                D_kind = D_kind)
    else:
        raise Error("Unrecognized step flavor method : " + step_flavor['method'])

    D_additional_params = step_rule.velocities
    D_additional_kind = setp_rule.D_kind

    # TODO : Make sure that these variables are indeed dictionaries and not lists.
    #        Remove this afterwards.
    assert type(D_additional_params) == dict
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
    server_dict=[]

    for param in D_params:
        param_dict={}
        param_dict["name"] = param.name
        param_dict["kind"] = D_kind[param.name]
        # voir les tailles pour les reshape
        if D_kind[param.name]=="CONV_FILTER_WEIGHTS":
            params_dict["shape"] = list(param.get_value().shape)
        elif D_kind[param.name]=="CONV_FILTER_BIASES":
            params_dict["shape"] = [1] + list(param.get_value().shape)
        elif D_kind[param.name]=="FULLY_CONNECTED_WEIGHTS":
            params_dict["shape"] = list(param.get_value().shape) + [1, 1]
        elif D_kind[param.name]=="FULLY_CONNECTED_BIASES":
            params_dict["shape"] = [1] + list(param.get_value().shape) + [1, 1]
        else:
            raise Exception("unknow kind of parameters : %s for param %s",
                            D_kind[param.name],
                            param.name)

        assert len(params_dict["shape"])==4
        server_dict.append(params_dict)

    return server_dict



