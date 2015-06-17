
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
    def __init__(self, D_params, D_kind, decay_rate=0.9, max_scaling=1e5):
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay rate needs to be in [0, 1]")
        if max_scaling <= 0:
            raise ValueError("max. scaling needs to be greater than 0")
        self.decay_rate = shared_floatx(decay_rate)
        self.epsilon = 1. / max_scaling
        self.velocities = OrderedDict()
        self.D_kind = {}
        for p_name in D_params:
            param_i = D_params[p_name]
            velocity = shared_floatx(param_i.get_value() * 0.)
            velocity.name = p_name+ "_decay"
            self.velocities[velocity.name] = velocity
            self.D_kind[velocity.name] = D_kind[p_name]


    def compute_step(self, param, previous_step):
        if param.name+"_decay" in self.velocities:
            mean_square_step_tm1 = self.velocities[param.name+"_decay"]
        else:
            raise Exception('unknow parameter %s', param.name)

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
    def __init__(self, D_params, D_kind, learning_rate=1.0, decay_rate=0.9, max_scaling=1e5):
        basic_rms_prop = BasicRMSProp(decay_rate=decay_rate,
                                      max_scaling=max_scaling,
                                      D_params=D_params,
                                      D_kind=D_kind)
        scale = Scale(learning_rate=learning_rate)
        self.learning_rate = scale.learning_rate
        self.decay_rate = basic_rms_prop.decay_rate
        self.components = [basic_rms_prop, scale]
        self.velocities = basic_rms_prop.velocities
        self.D_kind = basic_rms_prop.D_kind


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
    def __init__(self, D_params, D_kind, momentum=0.):
        self.momentum = shared_floatx(momentum)
        # dictionary of velocities
        self.velocities = OrderedDict()
        self.D_kind = {}
        for p_name in D_params:
            param_i = D_params[p_name]
            velocity = shared_floatx(param_i.get_value() * 0.)
            velocity.name = p_name+ "_momentum"
            self.velocities[velocity.name] = velocity
            self.D_kind[velocity.name] = D_kind[p_name]

    def compute_step(self, param, previous_step):
        if param.name+"_momentum" in self.velocities:
            velocity = self.velocities[param.name+"_momentum"]
        else:
            raise Exception('unknow parameter %s', param.name)
                
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
    def __init__(self, D_params, D_kind, learning_rate=1.0, momentum=0.0):
        scale = Scale(learning_rate=learning_rate)
        basic_momentum = BasicMomentum_dict(momentum=momentum, D_params=D_params, D_kind=D_kind)
        self.learning_rate = scale.learning_rate
        self.momentum = basic_momentum.momentum
        self.components = [scale, basic_momentum]
        self.velocities = basic_momentum.velocities
        self.D_kind = basic_momentum.D_kind


class AdaGrad(StepRule):
    """Implements the AdaGrad learning rule.
    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.0002.
    epsilon : float, optional
        Stabilizing constant for one over root of sum of squares.
        Defaults to 1e-6.
    Notes
    -----
    For more information, see [ADAGRAD]_.
    .. [ADADGRAD] Duchi J, Hazan E, Singer Y.,
       *Adaptive subgradient methods for online learning and
        stochastic optimization*,
       http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    def __init__(self, D_params, D_kind, learning_rate=0.002, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.velocities = OrderedDict()
        self.D_kind = {}
        for p_name in D_params:
            param_i = D_params[p_name]
            velocity = shared_floatx(param_i.get_value() * 0.)
            velocity.name = p_name + "_ssq"
            self.velocities[velocity.name] = velocity
            self.D_kind[velocity.name] = D_kind[p_name]
            
    def compute_step(self, param, previous_step):
        ssq = self.velocities[param.name+"_ssq"]
        ssq_t = (tensor.sqr(previous_step) + ssq)
        step = (self.learning_rate * previous_step /
                (tensor.sqrt(ssq_t) + self.epsilon))

        updates = [(ssq, ssq_t)]

        return step, updates


class Adam(StepRule):
    """Adam optimizer as described in [King2014]_.
    .. [King2014] Diederik Kingma, Jimmy Ba,
       *Adam: A Method for Stochastic Optimization*,
       http://arxiv.org/abs/1412.6980
    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.0002.
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
        Default value is set to 0.1.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
        Default value is set to 0.001.
    epsilon : float, optional
        Default value is set to 1e-8.
    decay_factor : float, optional
        Default value is set to 1 - 1e-8.
    """
    def __init__(self, D_params, D_kind, learning_rate=0.002,
                 beta1=0.1, beta2=0.001, epsilon=1e-8,
                 decay_factor=(1 - 1e-8)):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.velocities = OrderedDict()
        self.D_kind = {}
        for p_name in D_params:
            param_i = D_params[p_name]
            velocity_mean = shared_floatx(param_i.get_value() * 0.)
            velocity_mean.name = p_name + "_mean"
            self.velocities[velocity_mean.name] = velocity_mean
            self.D_kind[velocity_mean.name] = D_kind[p_name]

            velocity_var = shared_floatx(param_i.get_value() * 0.)
            velocity_var.name = p_name + "_var"
            self.velocities[velocity_var.name] = velocity_var
            self.D_kind[velocity_var.name] = D_kind[p_name]

    def compute_step(self, param, previous_step):
        mean = self.velocities[param.name+"_mean"]
        variance = self.velocities[param.name+"_var"]
        # TODO check time
        time = shared_floatx(0., 'time')

        t1 = time + 1
        learning_rate = (self.learning_rate *
                         tensor.sqrt((1. - (1. - self.beta2)**t1)) /
                         (1. - (1. - self.beta1)**t1))
        beta_1t = 1 - (1 - self.beta1) * self.decay_factor ** (t1 - 1)
        mean_t = beta_1t * previous_step + (1. - beta_1t) * mean
        variance_t = (self.beta2 * tensor.sqr(previous_step) +
                      (1. - self.beta2) * variance)
        step = (learning_rate * mean_t /
                (tensor.sqrt(variance_t) + self.epsilon))

        updates = [(mean, mean_t),
                   (variance, variance_t),
                   (time, t1)]

        return step, updates


class AdaDelta(StepRule):
    """Adapts the step size over time using only first order information.
    Parameters
    ----------
    decay_rate : float, optional
        Decay rate in [0, 1]. Defaults to 0.95.
    epsilon : float, optional
        Stabilizing constant for RMS. Defaults to 1e-6.
    Notes
    -----
    For more information, see [ADADELTA]_.
    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    def __init__(self, D_params, D_kind, decay_rate=0.95, epsilon=1e-6):
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay rate needs to be in [0, 1]")
        self.decay_rate = shared_floatx(decay_rate)
        self.epsilon = shared_floatx(epsilon)
        self.velocities = OrderedDict()
        self.D_kind = {}
        for p_name in D_params:
            param_i = D_params[p_name]
            velocity_tm1 = shared_floatx(param_i.get_value() * 0.)
            velocity_tm1.name = p_name + "_tm1"
            self.velocities[velocity_tm1.name] = velocity_tm1
            self.D_kind[velocity_tm1.name] = D_kind[p_name]

            velocity_x_tm1 = shared_floatx(param_i.get_value() * 0.)
            velocity_tm1.name = p_name + "_xtm1"
            self.velocities[velocity_x_tm1.name] = velocity_x_tm1
            self.D_kind[velocity_x_tm1.name] = D_kind[p_name]

    def compute_step(self, param, previous_step):
        mean_square_step_tm1 = self.velocities[param.name+"_tm1"]
        mean_square_delta_x_tm1 = self.velocities[param.name+"_xtm1"]

        mean_square_step_t = (
            self.decay_rate * mean_square_step_tm1 +
            (1 - self.decay_rate) * tensor.sqr(previous_step)
        )

        rms_delta_x_tm1 = tensor.sqrt(mean_square_delta_x_tm1 + self.epsilon)
        rms_step_t = tensor.sqrt(mean_square_step_t + self.epsilon)
        delta_x_t = rms_delta_x_tm1 / rms_step_t * previous_step

        mean_square_delta_x_t = (
            self.decay_rate * mean_square_delta_x_tm1 +
            (1 - self.decay_rate) * tensor.sqr(delta_x_t)
        )

        step = delta_x_t
        updates = [(mean_square_step_tm1, mean_square_step_t),
                   (mean_square_delta_x_tm1, mean_square_delta_x_t)]
        return step, updates

