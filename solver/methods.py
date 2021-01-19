#
# This file is part of numerical-solver
# (https://github.com/FarmHJ/numerical-solver/) which is released under the BSD
# 3-clause license. See accompanying LICENSE.md for copyright notice and full
# license details.
#


class SolverMethods(object):
    """SolverMethods Class:

    Apply numerical methods to solve given initial value problem

    Parameters
    ----------
    model: solver.ForwardModel class
    start: simulation start time
    end: simulation end time
    """

    def __init__(self, func, x_min, x_max, initial_value, mesh_points): # noqa
        super(SolverMethods, self).__init__()

        self.func = func
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.initial_value = float(initial_value)
        self.mesh_points = int(mesh_points)

        self.mesh_size = (self.x_max - self.x_min) / self.mesh_points

    def Euler_explicit(self):

        y_n = [self.initial_value]
        x_n = [self.x_min]

        for n in range(1, self.mesh_points + 1):
            x_n.append(self.x_min + n * self.mesh_size)
            y_n.append(y_n[-1] + self.mesh_size * self.func(x_n[-1], y_n[-1]))

        return x_n, y_n

    def fixed_pt_iteration(self, init_pred, x):

        y_0 = init_pred
        y_1 = init_pred + self.mesh_size * self.func(x, y_0)
        iteration_counts = 0

        while abs(y_1 - y_0) >= 0.01 and iteration_counts <= 100:
            y_0 = y_1
            y_1 = init_pred + self.mesh_size * self.func(x, y_0)
            iteration_counts += 1

        if abs(y_1 - y_0) < 0.01:
            return y_1
        else:
            raise RuntimeError('Fixed point iteration does not converge')

    def Euler_implicit(self):

        y_n = [self.initial_value]
        x_n = [self.x_min]

        for n in range(1, self.mesh_points + 1):
            x_n.append(self.x_min + n * self.mesh_size)
            est_y = self.fixed_pt_iteration(y_n[-1], x_n[-1])
            y_n.append(est_y)

        return x_n, y_n
