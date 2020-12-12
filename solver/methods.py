import numpy as np

import solver


class SolverMethod(object):
    """SolverMethod Class:

    Apply numerical methods to solve given initial value problem

    Parameters
    ----------
    model: solver.ForwardModel class
    start: simulation start time
    end: simulation end time
    """

    def __init__(self, func, x_min, x_max, initial_value, mesh_points): # noqa
        super(SolverMethod, self).__init__()

        self.func = func
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.initial_value = float(initial_value)
        self.mesh_points = int(mesh_points)

    def Euler_explicit(self):

        y_n = [self.initial_value]

        mesh = np.linspace(self.x_min, self.x_max, num=self.mesh_points) # noqa
        mesh_size = (self.x_max - self.x_min) / self.mesh_points
        for n in range(self.mesh_points):
            x_n = self.x_min + n * mesh_size
            y_n.append(y_n[-1] + mesh_size * func(x_n, y_n[-1]))

        return y_n
