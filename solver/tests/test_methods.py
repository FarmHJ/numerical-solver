#
# This file is part of numerical-solver
# (https://github.com/FarmHJ/numerical-solver/) which is released under the BSD
# 3-clause license. See accompanying LICENSE.md for copyright notice and full
# license details.
#

import unittest

import matplotlib.pyplot as plt
import numpy as np
import os

import solver


class TestOneStepMethods(unittest.TestCase):
    """
    Test the 'OneStepMethod' class.
    """
    def test__init__(self):

        def func(x, y):
            return -y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)

        self.assertEqual(problem.x_min, 0)
        self.assertEqual(problem.mesh_points, 10)

        with self.assertRaises(TypeError):
            solver.OneStepMethods(
                x_min, x_min, x_max, initial_value, mesh_points)

    def test_Euler_explicit(self):

        def func(x, y):
            return -y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)
        mesh, soln = problem.Euler_explicit()

        self.assertEqual(np.shape(mesh), (11,))
        self.assertEqual(np.shape(soln), (11,))
        self.assertEqual(soln[1], 0.9)
        self.assertEqual(soln[2], 0.81)

    # def test_fixed_pt_iteration(self):

        # def func(x, y):
        #     return -y
        # x_min = 0
        # x_max = 1
        # initial_value = 1
        # mesh_points = 10

        # problem = solver.OneStepMethods(
        #     func, x_min, x_max, initial_value, mesh_points)

        # def method(value):
        #     return 0.1 * func(0, value)

        # y_pred = problem.fixed_pt_iteration(initial_value, method)

        # self.assertEqual(y_pred, 0.909)

        # def func(x, y):
        #     return -10 * y
        # initial_value = 1.1
        # mesh_points = 10

        # problem = solver.OneStepMethods(
        #     func, x_min, x_max, initial_value, mesh_points)

        # with self.assertRaises(RuntimeError):
        #     problem.fixed_pt_iteration(initial_value, 0.1)

    def test_Euler_implicit(self):

        def func(x, y):
            return -y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)
        mesh, soln = problem.Euler_implicit()

        self.assertEqual(np.shape(mesh), (11,))
        self.assertEqual(np.shape(soln), (11,))
        self.assertEqual(soln[1], 0.9091)
        self.assertAlmostEqual(round(soln[2],5), 0.82647)

    def test_RungeKutta4(self):

        def func(x, y):
            return y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)
        mesh, soln = problem.RungeKutta4()

        self.assertAlmostEqual(soln[1], 1.10517083)


class TestPredictorCorrector(unittest.TestCase):
    """
    Test the 'PredictorCorrector' class.
    """
    def test__init__(self):

        def func(x, y):
            return -y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.PredictorCorrector(
            func, x_min, x_max, initial_value, mesh_points)

        self.assertEqual(problem.x_min, 0)
        self.assertEqual(problem.mesh_points, 10)

        with self.assertRaises(TypeError):
            solver.PredictorCorrector(
                x_min, x_min, x_max, initial_value, mesh_points)

    def test_Euler_trapezium(self):

        def func(x, y):
            return y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.PredictorCorrector(
            func, x_min, x_max, initial_value, mesh_points)

        _, soln = problem.Euler_trapezium()

        self.assertEqual(soln[1], 1.10525)
        self.assertEqual(round(soln[2], 5), 1.22158)


class TestAdaptiveMethod(unittest.TestCase):
    """
    Test the 'AdaptiveMethod' class.
    """
    def test__init__(self):

        def func(x, y):
            return -y
        x_min = 0
        x_max = 1
        initial_value = 1

        problem = solver.AdaptiveMethod(
            func, x_min, x_max, initial_value)

        self.assertEqual(problem.x_min, 0)
        self.assertEqual(problem.initial_value, 1)

        with self.assertRaises(TypeError):
            solver.AdaptiveMethod(
                x_min, x_min, x_max, initial_value)

    def test_ode23(self):

        def func(x, y):
            return 10 * np.exp(-(x - 2) * (x - 2) / (2 * 0.075**2)) - 0.6* y
            # - 1000 * y
        x_min = 0
        x_max = 4
        initial_value = 0.5

        problem = solver.AdaptiveMethod(
            func, x_min, x_max, initial_value)
        mesh, soln = problem.ode23()

    def test_ode45(self):

        def func(x, y):
            return 10 * np.exp(-(x - 2) * (x - 2) / (2 * 0.075**2)) - 0.6* y
            #-1000 * y
        x_min = 0
        x_max = 4
        initial_value = 0.5

        problem = solver.AdaptiveMethod(
            func, x_min, x_max, initial_value)
        mesh, soln = problem.ode45()


if __name__ == '__main__':
    unittest.main()
