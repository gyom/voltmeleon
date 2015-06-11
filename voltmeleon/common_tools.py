
import theano
floatX = theano.config.floatX


def set_param_value_shared_var(D_params, name, value):
    if name in D_params:
        param_i = D_params[name]
        shape = param_i.get_value().shape
        #print (name, shape, value.shape)
        param_i.set_value((value.reshape(shape)).astype(floatX))
    else:
        raise Exception("unknown parameter")


def get_param_value_shared_var(D_params, name):
    if name in D_params:
        return D_params[name].get_value()
    else:
        raise Exception("unknown parameter")
