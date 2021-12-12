from typing import Tuple
import numpy as np

class Discretizer:

    def __init__(self, bounds: Tuple[Tuple], steps: Tuple[int]):
        self.bounds = bounds
        self.steps = steps
    
    def index(self, value, low, high, steps):
        lim_norm = high - low
        val_norm = value - low
        l = int(val_norm * steps // lim_norm)
        return l

    def map(self, values: Tuple) -> Tuple[int]:
        point = [self.index(values[i],*self.bounds[i],self.steps[i]) for i in range(len(values))]
        return tuple(point)

    def get_zeros(self, dim: tuple):
        return np.zeros(self.steps + dim)