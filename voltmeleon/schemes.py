from abc import ABCMeta, abstractmethod
import numpy
from picklable_itertools import iter_
from six import add_metaclass
from fuel.schemes import IterationScheme

@add_metaclass(ABCMeta)
class LimitedScheme(IterationScheme):
    """Iterate over the sequence given an iterationScheme object
        but in a limited number of excursions defined by times
    """
    def __init__(self, iteration_scheme, times, *args, **kwargs):
        self.iteration_scheme = iteration_scheme
        self.times = times
        assert self.times >=1, "Error : you need a positiv number of mini batches"

    def get_request_iterator(self):
        it = self.iteration_scheme.get_request_iterator()
        return iter_([it.next() for n in range(self.times)]) 
