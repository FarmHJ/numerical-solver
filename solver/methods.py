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

        where :math:`h` is the mesh size and function :math:`f` is the ODE.
        """

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Calculate approximated solution for each mesh point
        for n in range(1, self.mesh_points + 1):
            y_n.append(y_n[-1] + self.mesh_size * self.func(x_n[-1], y_n[-1]))
            x_n.append(self.x_min + n * self.mesh_size)

        return x_n, y_n

    def fixed_pt_iteration(self, prediction, numerical_method):
        r"""
        To approximate the solution :math:`y_{n+1}`for any implicit numerical
        method at mesh point :math:`x_{n+1}`

        Parameters
        ----------
        prediction
            Predicted value for fixed point iteration algorithm.
            Conventionally taken as approximated solution at mesh
            point :math:`n`, :math:`y_n`
        numerical_method
            The numerical method or equation where the solution
            cannot be obtained explicitly.
        """

        next_prediction = numerical_method(prediction)
        iteration_counts = 0
        threshold = 0.001

        # Approximate solution with fixed point iteration
        while abs(prediction - next_prediction) >= threshold and iteration_counts <= 1000: # noqa
            prediction = next_prediction
            next_prediction = numerical_method(prediction)
            iteration_counts += 1

        # Raise error if algorithm doesn't converge
        if abs(prediction - next_prediction) < threshold:
            return next_prediction
        else:
            raise RuntimeError('Fixed point iteration does not converge')

    def Euler_implicit(self, prediction=None):
        r"""
        Runs the Euler's implicit numerical method to approximate
        the solution to the initial value problem.

        .. math::
            y_{n+1} = y_n + hf(x_{n+1}, y_{n+1})

        where :math:`h` is the mesh size and function :math:`f` is the ODE.
        """

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Calculate approximated solution for each mesh point
        if prediction is None:
            prediction = y_n[-1]

        for n in range(1, self.mesh_points + 1):

            def num_method(prediction):
                return y_n[-1] + self.mesh_size * (
                    self.func(x_n[-1] + self.mesh_size, prediction))

            est_y = self.fixed_pt_iteration(prediction, num_method)
            y_n.append(est_y)
            x_n.append(self.x_min + n * self.mesh_size)

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

        and :math:`h` is the mesh size and function :math:`f` is the ODE.
        """

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Calculate approximated solution for each mesh point
        for n in range(1, self.mesh_points + 1):
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
            x_n.append(self.x_min + n * self.mesh_size)

        return x_n, y_n

    def trapezium_rule(self, prediction=None):

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Calculate approximated solution for each mesh point
        if prediction is None:
            prediction = y_n[-1]

        for n in range(1, self.mesh_points + 1):

            def num_method(prediction):
                return y_n[-1] + self.mesh_size / 2 * (
                    self.func(x_n[-1], y_n[-1]) + self.func(
                        x_n[-1] + self.mesh_size, prediction))

            est_y = self.fixed_pt_iteration(prediction, num_method)
            y_n.append(est_y)
            x_n.append(self.x_min + n * self.mesh_size)

        return x_n, y_n


class PredictorCorrector(object):
    """PredictorCorrector Class:

    Predictor corrector method to solve given initial value problem
    when using implicit linear multistep method as solution cannot be
    solved numerically.

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
        super(PredictorCorrector, self).__init__()

        if not callable(func):
            raise TypeError('Input func is not a callable function')

        self.func = func
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.initial_value = float(initial_value)
        self.mesh_points = int(mesh_points)

        # Calculate the size of mesh
        self.mesh_size = (self.x_max - self.x_min) / self.mesh_points

    def Euler_trapezium(self):

        y_n = [self.initial_value]
        x_n = [self.x_min]

        def trapezium(x_point, y_n, prediction):
            return y_n[-1] + self.mesh_size / 2 * (
                self.func(x_point, y_n[-1]) +
                self.func(x_point + self.mesh_size, prediction))

        for n in range(1, self.mesh_points + 1):

            predictor = OneStepMethods(
                self.func, self.x_min + (n - 1) * self.mesh_size,
                self.x_min + n * self.mesh_size,
                y_n[-1], 1)
            _, prediction = predictor.Euler_explicit()

            next_correction = trapezium(
                x_n[-1], y_n, prediction[-1])
            correction = prediction[-1]
            iteration_counts = 0

            while abs(correction - next_correction) > 0.001 and iteration_counts <= 1000: # noqa
                correction = next_correction
                next_correction = trapezium(
                    x_n[-1], y_n, correction)
                iteration_counts += 1

            y_n.append(next_correction)
            x_n.append(self.x_min + n * self.mesh_size)

        return x_n, y_n


class AdaptiveMethod(object):
    """AdaptiveMethod Class:

    Adaptive numerical method is used to solve a given
    initial value problem to control errors.

    Parameters
    ----------
    func: callable function
        ODE function to be solved numerically
    x_min
        Starting value of mesh
    initial_value
        Value of solution at starting point of mesh.
    """

    def __init__(self, func, x_min, x_max, initial_value):
        super(AdaptiveMethod, self).__init__()

        if not callable(func):
            raise TypeError('Input func is not a callable function')

        self.func = func
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.initial_value = float(initial_value)
        self.initial_mesh = 0.5

    def ode23(self, abs_tol=1e-6, rel_tol=1e-3):

        abs_tol = abs(abs_tol)
        rel_tol = abs(rel_tol)

        y_n = [self.initial_value]
        x_n = [self.x_min]

        y_temp = 2 * (abs_tol + rel_tol)

        while x_n[-1] < self.x_max:

            mesh = self.initial_mesh * 3
            error = 2 * (abs_tol + rel_tol)
            count = 0

            while error > max(abs_tol, rel_tol * abs(y_temp)):

                mesh = mesh / 3
                coef1 = self.func(x_n[-1], y_n[-1])
                coef2 = self.func(
                    x_n[-1] + mesh / 2,
                    y_n[-1] + mesh / 2 * coef1)
                coef3 = self.func(
                    x_n[-1] + mesh * 3 / 4,
                    y_n[-1] + mesh * 3 / 4 * coef2)

                y_temp = y_n[-1] + mesh / 9 * (
                    2 * coef1 + 3 * coef2 + 4 * coef3)

                coef4 = self.func(
                    x_n[-1] + mesh, y_temp)

                error = mesh / 72 * (
                    -5 * coef1 + 6 * coef2 + 8 * coef3 - 9 * coef4)
                error = abs(error)
                count += 1

            y_n.append(y_temp)
            x_n.append(x_n[-1] + mesh)

        return x_n, y_n
