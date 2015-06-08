
"""
Momentum where you can retrieve the velocities
"""
import theano
import theano.tensor as T
from blocks.algorithms import StepRule, CompositeRule, Scale
from theano.compat import OrderedDict
from blocks.utils import shared_floatx


class BasicRMSProp(StepRule):
    """Scales the step size by a running average of the recent step norms.
    Parameters
    ----------
    decay_rate : float, optional
        How fast the running average decays, value in [0, 1]
        (lower is faster).  Defaults to 0.9.
    max_scaling : float, optional
        Maximum scaling of the step size, in case the running average is
        really small. Needs to be greater than 0. Defaults to 1e5.
    Notes
    -----
    This step rule is intended to be used in conjunction with another
    step rule, _e.g._ :class:`Scale`. For an all-batteries-included
    experience, look at :class:`RMSProp`.
    In general, this step rule should be used _before_ other step rules,
    because it has normalization properties that may undo their work.
    For instance, it should be applied first when used in conjunction
    with :class:`Scale`.
    For more information, see [Hint2014]_.
    """
    def __init__(self, decay_rate=0.9, max_scaling=1e5, params=None):
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay rate needs to be in [0, 1]")
        if max_scaling <= 0:
            raise ValueError("max. scaling needs to be greater than 0")
        self.decay_rate = shared_floatx(decay_rate)
        self.epsilon = 1. / max_scaling
        self.velocities = OrderedDict()
        for param_i in params:
            velocity = shared_floatx(param_i.get_value() * 0.)
            velocity.name = param_i.name+ "_momentum"
            self.velocities[velocity.name] = velocity


    def compute_step(self, param, previous_step):
        if param.name+"_momentum" in self.velocities:
            mean_square_step_tm1 = self.velocities[param.name+"_momentum"]
        else:
            raise Error('unknow parameter %s', param.name)

        mean_square_step_t = (
            self.decay_rate * mean_square_step_tm1 +
            (1 - self.decay_rate) * T.sqr(previous_step))
        rms_step_t = T.maximum(
            T.sqrt(mean_square_step_t), self.epsilon)
        step = previous_step / rms_step_t
        updates = [(mean_square_step_tm1, mean_square_step_t)]
        return step, updates


class RMSProp(CompositeRule):
    """Scales the step size by a running average of the recent step norms.
    Combines :class:`BasicRMSProp` and :class:`Scale` to form the step rule
    described in [Hint2014]_.
    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    Parameters
    ----------
    learning_rate : float, optional
        The learning rate by which the previous step scaled. Defaults to 1.
    decay_rate : float, optional
        How fast the running average decays (lower is faster).
        Defaults to 0.9.
    max_scaling : float, optional
        Maximum scaling of the step size, in case the running average is
        really small. Defaults to 1e5.
    Attributes
    ----------
    learning_rate : :class:`~tensor.SharedVariable`
        A variable for learning rate.
    decay_rate : :class:`~tensor.SharedVariable`
        A variable for decay rate.
    See Also
    --------
    :class:`SharedVariableModifier`
    """
    def __init__(self, learning_rate=1.0, decay_rate=0.9, max_scaling=1e5, params=None):
        basic_rms_prop = BasicRMSProp(decay_rate=decay_rate,
                                      max_scaling=max_scaling,
                                      params=params)
        scale = Scale(learning_rate=learning_rate)
        self.learning_rate = scale.learning_rate
        self.decay_rate = basic_rms_prop.decay_rate
        self.components = [basic_rms_prop, scale]
        self.velocities = basic_rms_prop.velocities


class BasicMomentum_dict(StepRule):
    """Accumulates step with exponential discount.
    Parameters
    ----------
    momentum : float, optional
        The momentum coefficient. Defaults to 0.
    params : the set of params on which we apply the momentum

    Notes
    -----
    This step rule is intended to be used in conjunction with another
    step rule, _e.g._ :class:`Scale`. For an all-batteries-included
    experience, look at :class:`Momentum`.
    """
    def __init__(self, params=None, momentum=0.):
        self.momentum = shared_floatx(momentum)
        # dictionary of velocities
        self.velocities = OrderedDict()
        for param_i in params:
            velocity = shared_floatx(param_i.get_value() * 0.)
            velocity.name = param_i.name+ "_momentum"
            self.velocities[velocity.name] = velocity

    def compute_step(self, param, previous_step):
        if param.name+"_momentum" in self.velocities:
            velocity = self.velocities[param.name+"_momentum"]
        else:
            raise Error('unknow parameter %s', param.name)
                
        step = self.momentum * velocity + previous_step
        updates = [(velocity, step)]
        return step, updates


class Momentum_dict(CompositeRule):
    """Accumulates step with exponential discount.
    Combines :class:`BasicMomentum` and :class:`Scale` to form the
    usual momentum step rule.
    Parameters
    ----------
    learning_rate : float, optional
        The learning rate by which the previous step scaled. Defaults to 1.
    momentum : float, optional
        The momentum coefficient. Defaults to 0.
    Attributes
    ----------
    learning_rate : :class:`~tensor.SharedVariable`
        A variable for learning rate.
    momentum : :class:`~tensor.SharedVariable`
        A variable for momentum.
    See Also
    --------
    :class:`SharedVariableModifier`
    """
    def __init__(self, learning_rate=1.0, momentum=0., params=None):
        scale = Scale(learning_rate=learning_rate)
        basic_momentum = BasicMomentum_dict(momentum=momentum, params=params)
        self.learning_rate = scale.learning_rate
        self.momentum = basic_momentum.momentum
        self.components = [scale, basic_momentum]
        self.velocities = basic_momentum.velocities
