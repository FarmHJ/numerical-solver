#
# This file is part of numerical-solver
# (https://github.com/FarmHJ/numerical-solver/) which is released under the BSD
# 3-clause license. See accompanying LICENSE.md for copyright notice and full
# license details.
#


class OneStepMethods(object):
    """OneStepMethods Class:

    One-step numerical methods to solve given initial value problem

    Parameters
    ----------
    func: callable function
        ODE function to be solved numerically
    x_min
        Starting value of mesh
    x_max
        Final value of mesh
    initial_value
        Value of solution at starting point of mesh.
    mesh_points
        Total number of mesh points within the range
        ``x_min`` to ``x_max``.
    """

    def __init__(self, func, x_min, x_max, initial_value, mesh_points): # noqa
        super(OneStepMethods, self).__init__()

        if not callable(func):
            raise TypeError('Input func is not a callable function')

        self.func = func
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.initial_value = float(initial_value)
        self.mesh_points = int(mesh_points)

        # Calculate the size of mesh
        self.mesh_size = (self.x_max - self.x_min) / self.mesh_points

    def Euler_explicit(self):
        r"""
        Runs the Euler's explicit numerical method to approximate
        the solution to the initial value problem.

        .. math::
            y_{n+1} = y_n + hf(x_n, y_n)

        where :math:`h` is the mesh size and function :math:
        `f` is the ODE.
        """

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Calculate approximated solution for each mesh point
        for n in range(1, self.mesh_points + 1):
            x_n.append(self.x_min + n * self.mesh_size)
            y_n.append(y_n[-1] + self.mesh_size * self.func(x_n[-1], y_n[-1]))

        return x_n, y_n

    def fixed_pt_iteration(self, init_pred, x):
        r"""
        To approximate the solution :math:`y_{n+1}`for Euler's implicit method
        at mesh point :math:`x_{n+1}`

        Parameters
        ----------
        init_pred
            Predicted value for fixed point iteration algorithm.
            Conventionally taken as approximated solution as mesh
            point :math:`n`, :math:`y_n`
        x
            The mesh point :math:`x_{n+1}` where :math:`y_{n+1}`
            is approximated under the Euler's implicit method.
        """

        y_0 = init_pred
        y_1 = init_pred + self.mesh_size * self.func(x, y_0)
        iteration_counts = 0

        # Approximate solution with fixed point iteration
        while abs(y_1 - y_0) >= 0.01 and iteration_counts <= 100:
            y_0 = y_1
            y_1 = init_pred + self.mesh_size * self.func(x, y_0)
            iteration_counts += 1

        # Raise error if algorithm doesn't converge
        if abs(y_1 - y_0) < 0.01:
            return y_1
        else:
            raise RuntimeError('Fixed point iteration does not converge')

    def Euler_implicit(self):
        r"""
        Runs the Euler's implicit numerical method to approximate
        the solution to the initial value problem.

        .. math::
            y_{n+1} = y_n + hf(x_{n+1}, y_{n+1})

        where :math:`h` is the mesh size and function :math:
        `f` is the ODE.
        """

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Calculate approximated solution for each mesh point
        for n in range(1, self.mesh_points + 1):
            x_n.append(self.x_min + n * self.mesh_size)
            est_y = self.fixed_pt_iteration(y_n[-1], x_n[-1])
            y_n.append(est_y)

        return x_n, y_n

    def RungeKutta4(self):
        r"""
        Runs the 4-stage Runge-Kutta numerical method to approximate
        the solution to the initial value problem.

        .. math::
            y_{n+1} = y_n + \frac{1}{6}h(k_1 + 2k_2 + 2k_3 + k4)
        where
        .. math::
            k_1 = f(x_n, y_n),
        .. math::
            k_2 = f(x_n + \frac{1}{2}h, y_n + \frac{1}{2}hk_1),
        .. math::
            k_3 = f(x_n + \frac{1}{2}h, y_n + \frac{1}{2}hk_2),
        .. math::
            k_4 = f(x_n + h, y_n + hk_3),

        and :math:`h` is the mesh size and function :math:
        `f` is the ODE.
        """

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Calculate approximated solution for each mesh point
        for n in range(1, self.mesh_points + 1):
            x_n.append(self.x_min + n * self.mesh_size)
            k1 = self.func(x_n[-1], y_n[-1])
            k2 = self.func(
                x_n[-1] + self.mesh_size / 2,
                y_n[-1] + self.mesh_size / 2 * k1)
            k3 = self.func(
                x_n[-1] + self.mesh_size / 2,
                y_n[-1] + self.mesh_size / 2 * k2)
            k4 = self.func(
                x_n[-1] + self.mesh_size, y_n[-1] + self.mesh_size * k3)
            funcs_value = k1 + 2 * k2 + 2 * k3 + k4
            y_n.append(y_n[-1] + self.mesh_size / 6 * funcs_value)

        return x_n, y_n
