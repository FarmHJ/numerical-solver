#
# This file is part of numerical-solver
# (https://github.com/FarmHJ/numerical-solver/) which is released under the BSD
# 3-clause license. See accompanying LICENSE.md for copyright notice and full
# license details.
#

import numpy as np


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
            raise TypeError('Input func is not a callable function.')

        if not isinstance(initial_value, list):
            raise TypeError('Input initial_value is not a list.')

        self.func = func
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.initial_value = initial_value
        self.mesh_points = int(mesh_points)

        # Calculate the size of mesh.
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

        # Calculate approximated solution for each mesh point.
        for n in range(1, self.mesh_points + 1):
            step = [self.mesh_size * f for f in self.func(x_n[-1], y_n[-1])]
            y_n.append([a + b for a, b in zip(y_n[-1], step)])
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

        # Approximate solution with fixed point iteration.
        while np.linalg.norm(
            np.array(prediction) - np.array(
                next_prediction)) >= threshold and iteration_counts <= 1000:
            prediction = next_prediction
            next_prediction = numerical_method(prediction)
            iteration_counts += 1

        # Raise error if algorithm doesn't converge.
        if np.linalg.norm(
            np.array(prediction) - np.array(
                next_prediction)) < threshold:
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

        # Use value at previous mesh point as prediction for
        # fixed point iteration if no prediction is given.
        if prediction is None:
            prediction = y_n[-1]

        # Calculate approximated solution for each mesh point.
        # Use fixed point iteration to solve numerical equation.
        for n in range(1, self.mesh_points + 1):

            def num_method(prediction):
                step = [self.mesh_size * f for f in self.func(
                    x_n[-1] + self.mesh_size, prediction)]
                return [a + b for a, b in zip(y_n[-1], step)]

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

        # Calculate approximated solution for each mesh point.
        for n in range(1, self.mesh_points + 1):
            # step = [self.mesh_size * f for f in self.func(x_n[-1], y_n[-1])]
            # y_n.append([a + b for a, b in zip(y_n[-1], step)])

            k1 = self.func(x_n[-1], y_n[-1])

            k2_input = [
                a + self.mesh_size / 2 * b for a, b in zip(y_n[-1], k1)]
            k2 = self.func(
                x_n[-1] + self.mesh_size / 2, k2_input)

            k3_input = [
                a + self.mesh_size / 2 * b for a, b in zip(y_n[-1], k2)]
            k3 = self.func(
                x_n[-1] + self.mesh_size / 2, k3_input)

            k4_input = [
                a + self.mesh_size * b for a, b in zip(y_n[-1], k3)]
            k4 = self.func(
                x_n[-1] + self.mesh_size, k4_input)

            funcs_value = [
                a + 2 * b + 2 * c + d for a, b, c, d in zip(k1, k2, k3, k4)]
            y_n.append(
                [a + self.mesh_size / 6 * b for a, b in zip(
                    y_n[-1], funcs_value)])
            x_n.append(self.x_min + n * self.mesh_size)

        return x_n, y_n

    def trapezium_rule(self, prediction=None):
        r"""
        Runs the trapezium rule numerical method to approximate
        the solution to the initial value problem.

        .. math::
            y_{n+1} = y_n + \frac{1}{2}h(f(x_{n}, y_{n}), f(x_{n+1}, y_{n+1}))

        where :math:`h` is the mesh size and function :math:`f` is the ODE.
        """

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Use value at previous mesh point as prediction for
        # fixed point iteration if no prediction is given.
        if prediction is None:
            prediction = y_n[-1]

        # Calculate approximated solution for each mesh point.
        # Use fixed point iteration to solve numerical equation.
        for n in range(1, self.mesh_points + 1):

            def num_method(prediction):
                previous_func = self.func(x_n[-1], y_n[-1])
                new_func = self.func(x_n[-1] + self.mesh_size, prediction)
                return [a + self.mesh_size / 2 * (b + c) for a, b, c in zip(
                    y_n[-1], previous_func, new_func)]

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

        if not isinstance(initial_value, list):
            raise TypeError('Input initial_value is not a list.')

        self.func = func
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.initial_value = initial_value
        self.mesh_points = int(mesh_points)

        # Calculate the size of mesh
        self.mesh_size = (self.x_max - self.x_min) / self.mesh_points

    def Euler_trapezium(self, tol=0.001):
        r"""
        Runs the Euler-trapezium predictor-corrector method to approximate
        the solution to the initial value problem.

        The Euler's explicit method is used as a predictor for the implicit
        trapezium rule. The trapezium rule, a corrector method, is then
        run iteratively to estimate the actual solution.

        Predictor:

        .. math::
            y_{n+1} = y_n + hf(x_n, y_n)

        Corrector:

        .. math::
            y_{n+1} = y_n + \frac{1}{2}h(f(x_{n}, y_{n}), f(x_{n+1}, y_{n+1}))

        where :math:`h` is the mesh size and function :math:`f` is the ODE.
        """

        y_n = [self.initial_value]
        x_n = [self.x_min]
        tol = abs(tol)

        # Define the trapezium rule, which is the corrector method
        def trapezium(x_point, y_n, prediction):
            previous_func = self.func(x_point, y_n[-1])
            new_func = self.func(x_point + self.mesh_size, prediction)
            return [a + self.mesh_size / 2 * (b + c) for a, b, c in zip(
                y_n[-1], previous_func, new_func)]

        # Calculate approximated solution for each mesh point.
        # Use Euler's explicit method as predictor
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

            while np.linalg.norm(
                np.array(correction) - np.array(
                    next_correction)) > tol and iteration_counts <= 1000:
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
    initial value problem with controlled errors.

    Parameters
    ----------
    func: callable function
        ODE function to be solved numerically
    x_min
        Starting value of mesh
    initial_value
        Value of solution at starting point of mesh.
    """

    def __init__(self, func, x_min, x_max, initial_value, initial_mesh=0.2):
        super(AdaptiveMethod, self).__init__()

        if not callable(func):
            raise TypeError('Input func is not a callable function')

        if not isinstance(initial_value, list):
            raise TypeError('Input initial_value is not a list.')

        self.func = func
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.initial_value = initial_value
        self.initial_mesh = float(initial_mesh)

    def ode23(self, abs_tol=1e-6, rel_tol=1e-3):
        r"""
        Runs the BS23 algorithm to approximate the solution to the initial
        value problem. It involves 2 one step method, one of order 2 and
        the other of order 3.

        The order 2 method is used to estimate the solution, while the
        order 3 method is used to estimate the error of the numerical
        solution. The algorithm terminates when the error is smaller than
        given tolerance.

        Solution estimation:

        .. math::
            y_{n+1} = y_n + \frac{h}{9}(2k_1 + 3k_2 + 4k_3)

        where

        .. math::
            s_1 = f(x_n, y_n)
        .. math::
            s_2 = f(x_n + \frac{h}{2}, y_n + \frac{h}{2}s_1)
        .. math::
            s_3 = f(x_n + \frac{3h}{4}, y_n + \frac{3h}{4}s_2)

        Error estimation:

        .. math::
            e_{n+1} = \frac{h}{72}(-5s_1 + 6s_2 + 8s_3 - 9s_4)

        where

        .. math::
            s_4 = f(x_{n+1}, y_{n+1})

        and :math:`h` is the mesh size and function :math:`f` is the ODE.

        If the error is larger than the tolerance, the next mesh point is
        calculated to be

        .. math::
            h_{\text{new}} = 0.9h(\frac{\text{tol}}{\text{err}})^\frac{1}{3}
        """

        # Set the absolute and relative tolerance for the solution.
        abs_tol = abs(abs_tol)
        rel_tol = abs(rel_tol)

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Initialise a temporary solution so that the error is
        # larger than given tolerance.
        y_temp = [2 * (abs_tol + rel_tol)] * len(self.initial_value)

        # The adaptive method is run until the mesh point exceeds
        # given evaluation boundary.
        while x_n[-1] < self.x_max:

            error = 2 * (abs_tol + rel_tol)
            count = 0

            # BS23 algorithm, similar to Matlab ode23 function
            while error > max(abs_tol, rel_tol * np.linalg.norm(
                    np.array(y_temp))):

                if count == 0:
                    mesh = self.initial_mesh
                else:
                    mesh = 0.9 * mesh * pow(max(
                        abs_tol, rel_tol * np.linalg.norm(
                            np.array(y_temp))) / error, 1 / 3)

                coef1 = self.func(x_n[-1], y_n[-1])

                coef2_input = [
                    a + mesh / 2 * b for a, b in zip(y_n[-1], coef1)]
                coef2 = self.func(
                    x_n[-1] + mesh / 2, coef2_input)

                coef3_input = [
                    a + mesh * 3 / 4 * b for a, b in zip(y_n[-1], coef2)]
                coef3 = self.func(
                    x_n[-1] + mesh * 3 / 4, coef3_input)

                y_temp = [a + mesh / 9 * (
                    2 * b + 3 * c + 4 * d) for a, b, c, d in zip(
                        y_n[-1], coef1, coef2, coef3)]

                coef4 = self.func(x_n[-1] + mesh, y_temp)

                error = [mesh / 72 * (
                    -5 * a + 6 * b + 8 * c - 9 * d) for a, b, c, d in zip(
                        coef1, coef2, coef3, coef4)]

                error = np.linalg.norm(np.array(error))
                count += 1

            y_n.append(y_temp)
            x_n.append(x_n[-1] + mesh)

        return x_n, y_n

    def ode45(self, abs_tol=1e-6, rel_tol=1e-3):
        r"""
        Runs the RKF45 algorithm to approximate the solution to the initial
        value problem. It involves 2 one step method, one of order 4 and
        the other of order 5.

        The order 4 method is used to estimate the solution, while the
        order 5 method is used to estimate the error of the numerical
        solution. The algorithm terminates when the error is smaller than
        given tolerance.

        Solution estimation:

        .. math::
            y_{n+1} = y_n + h(\frac{35}{384}k_1 + \frac{500}{1113}k_3
                + \frac{125}{192}k_4 - \frac{2187}{6784}k_5 + \frac{11}{84}k_6)

        where

        .. math::
            s_1 = f(x_n, y_n)
        .. math::
            s_2 = f(x_n + \frac{h}{5}, y_n + \frac{h}{5}s_1)
        .. math::
            s_3 = f(x_n + \frac{3h}{10}, y_n + \frac{3h}{40}s_1
                + \frac{9h}{40}s_2)
        .. math::
            s_4 = f(x_n + \frac{4h}{5}, y_n + \frac{44h}{45}s_1
                - \frac{56h}{15}s_2 + \frac{32h}{9}s_3)
        .. math::
            s_5 = f(x_n + \frac{8h}{9}, y_n + \frac{19372h}{6561}s_1
                - \frac{25360h}{2187}s_2 + \frac{64448h}{6561}s_3
                - \frac{212h}{729}s_4)
        .. math::
            s_6 = f(x_n + h, y_n + \frac{9017h}{3168}s_1
                - \frac{355h}{33}s_2 + \frac{46732h}{5247}s_3
                + \frac{49h}{176}s_4 - \frac{5103h}{18656}s_5)

        Error estimation:

        .. math::
            e_{n+1} = h(\frac{71}{57600}s_1 - \frac{71}{16695}s_3
                + \frac{71}{1920}s_4 - \frac{17253}{339200}s_5
                + \frac{22}{525}s_6 - \frac{1}{40}s_7)

        where

        .. math::
            s_7 = f(x_{n+1}, y_{n+1})

        and :math:`h` is the mesh size and function :math:`f` is the ODE.

        If the error is larger than the tolerance, the next mesh point is
        calculated to be

        .. math::
            h_{\text{new}} = 0.9h(\frac{\text{tol}}{\text{err}})^\frac{1}{5}
        """

        # Set the absolute and relative tolerance for the solution.
        abs_tol = abs(abs_tol)
        rel_tol = abs(rel_tol)

        y_n = [self.initial_value]
        x_n = [self.x_min]

        # Initialise a temporary solution so that the error is
        # larger than given tolerance.
        y_temp = [2 * (abs_tol + rel_tol)] * len(self.initial_value)

        # The adaptive method is run until the mesh point exceeds
        # given evaluation boundary.
        while x_n[-1] < self.x_max:

            error = 2 * (abs_tol + rel_tol)
            count = 0

            # Modified RKF45 algorithm, similar to Matlab ode45 function
            while error > max(abs_tol, rel_tol * np.linalg.norm(
                    np.array(y_temp))):

                if count == 0:
                    mesh = self.initial_mesh
                else:
                    mesh = 0.9 * mesh * pow(max(
                        abs_tol, rel_tol * np.linalg.norm(
                            np.array(y_temp))) / error, 0.2)

                coef1 = self.func(x_n[-1], y_n[-1])

                coef2_input = [
                    a + mesh / 5 * b for a, b in zip(y_n[-1], coef1)]
                coef2 = self.func(
                    x_n[-1] + mesh / 5, coef2_input)

                coef3_input = [a + mesh * (
                    3 / 40 * b + 9 / 40 * c) for a, b, c in zip(
                        y_n[-1], coef1, coef2)]
                coef3 = self.func(
                    x_n[-1] + mesh * 3 / 10, coef3_input)

                coef4_input = [a + mesh * (
                    44 / 45 * b - 56 / 15 * c
                    + 32 / 9 * d) for a, b, c, d in zip(
                        y_n[-1], coef1, coef2, coef3)]
                coef4 = self.func(
                    x_n[-1] + mesh * 4 / 5, coef4_input)

                coef5_input = [a + mesh * (
                    19372 / 6561 * b - 25360 / 2187 * c + 64448 / 6561 * d
                    - 212 / 729 * e) for a, b, c, d, e in zip(
                        y_n[-1], coef1, coef2, coef3, coef4)]
                coef5 = self.func(
                    x_n[-1] + mesh * 8 / 9, coef5_input)

                coef6_input = [a + mesh * (
                    9017 / 3168 * b - 355 / 33 * c + 46732 / 5247 * d
                    - 49 / 176 * e - 5103 / 18656 * f) for (
                        a, b, c, d, e, f) in zip(
                            y_n[-1], coef1, coef2, coef3, coef4, coef5)]
                coef6 = self.func(
                    x_n[-1] + mesh, coef6_input)

                y_temp = [a + mesh * (
                    35 / 384 * b + 500 / 1113 * c + 125 / 192 * d
                    - 2187 / 6784 * e + 11 / 84 * f) for (
                        a, b, c, d, e, f) in zip(
                            y_n[-1], coef1, coef3, coef4, coef5, coef6)]

                coef7 = self.func(x_n[-1] + mesh, y_temp)

                error = [mesh * (
                    71 / 57600 * a - 71 / 16695 * b + 71 / 1920
                    * c - 17253 / 339200 * d + 22 / 525 * e
                    - 1 / 40 * f) for (a, b, c, d, e, f) in zip(
                        coef1, coef3, coef4, coef5, coef6, coef7)]
                error = np.linalg.norm(np.array(error))
                count += 1

            y_n.append(y_temp)
            x_n.append(x_n[-1] + mesh)

        return x_n, y_n
