#
# This file is part of numerical-solver
# (https://github.com/FarmHJ/numerical-solver/) which is released under the BSD
# 3-clause license. See accompanying LICENSE.md for copyright notice and full
# license details.
#

import unittest

import numpy as np

import solver


class TestOneStepMethods(unittest.TestCase):
    """
    Test the 'OneStepMethod' class.
    """
    def test__init__(self):

        def func(x, y):
            return [-y[0]]
        x_min = 0
        x_max = 1
        initial_value = [1]
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)

        # Test initialisation
        self.assertEqual(problem.x_min, 0)
        self.assertEqual(problem.mesh_points, 10)

        # Test raised error for callable function
        with self.assertRaises(TypeError):
            solver.OneStepMethods(
                x_min, x_min, x_max, initial_value, mesh_points)

        # Test raised error if initial_value not list
        with self.assertRaises(TypeError):
            solver.OneStepMethods(
                func, x_min, x_max, 1, mesh_points)

    def test_Euler_explicit(self):

        def func(x, y):
            return [-y[0], y[1]]
        x_min = 0
        x_max = 1
        initial_value = [1, 1]
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)
        mesh, soln = problem.Euler_explicit()

        # Test shape of output: mesh and solution
        self.assertEqual(np.shape(mesh), (11,))
        self.assertEqual(np.shape(soln), (11, 2))

        # Test solution at first stepsize
        self.assertEqual(soln[1][0], 0.9)
        self.assertEqual(soln[2][0], 0.81)

        # Test solution at second stepsize
        self.assertEqual(soln[1][1], 1.1)
        self.assertAlmostEqual(soln[2][1], 1.21)

    def test_fixed_pt_iteration(self):

        def func(x, y):
            return [-y[0]]
        x_min = 0
        x_max = 1
        initial_value = [1]
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)

        def method(value):
            return [0.1 * i + 1.1 for i in func(0, value)]

        y_pred = problem.fixed_pt_iteration(initial_value, method)

        # Test fixed point iteration results
        self.assertAlmostEqual(y_pred, [1])

        def func(x, y):
            return [-10 * y[0]]
        initial_value = [1.1]
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)

        # Test raised error when fixed point iteration does not converge
        with self.assertRaises(RuntimeError):
            problem.fixed_pt_iteration(initial_value, method)

    def test_Euler_implicit(self):

        def func(x, y):
            return [-y[0], y[1]]
        x_min = 0
        x_max = 1
        initial_value = [1, 1]
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)
        mesh, soln = problem.Euler_implicit()

        # Test shape of output: mesh and solution
        self.assertEqual(np.shape(mesh), (11,))
        self.assertEqual(np.shape(soln), (11, 2))

        # Test solution at first stepsize
        self.assertEqual(soln[1][0], 0.9091)
        self.assertAlmostEqual(round(soln[2][0], 5), 0.82647)

        # Test solution at second stepsize
        self.assertEqual(soln[1][1], 1.1111)
        self.assertAlmostEqual(soln[2][1], 1.2345321)

    def test_RungeKutta4(self):

        def func(x, y):
            return [y[0]]
        x_min = 0
        x_max = 1
        initial_value = [1]
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)
        mesh, soln = problem.RungeKutta4()

        # Test shape of output: mesh and solution
        self.assertEqual(np.shape(mesh), (11,))
        self.assertEqual(np.shape(soln), (11, 1))

        # Test solution at first stepsize
        self.assertAlmostEqual(soln[1][0], 1.10517083)

    def test_trapezium_rule(self):

        def func(x, y):
            return [-y[0]]
        x_min = 0
        x_max = 1
        initial_value = [1]
        mesh_points = 10

        problem = solver.OneStepMethods(
            func, x_min, x_max, initial_value, mesh_points)
        mesh, soln = problem.trapezium_rule()

        # Test shape of output: mesh and solution
        self.assertEqual(np.shape(mesh), (11,))
        self.assertEqual(np.shape(soln), (11, 1))

        # Test solution at first stepsize
        self.assertAlmostEqual(soln[1][0], 0.904749999999)


class TestPredictorCorrector(unittest.TestCase):
    """
    Test the 'PredictorCorrector' class.
    """
    def test__init__(self):

        def func(x, y):
            return [-y[0]]
        x_min = 0
        x_max = 1
        initial_value = [1]
        mesh_points = 10

        problem = solver.PredictorCorrector(
            func, x_min, x_max, initial_value, mesh_points)

        # Test initialisation
        self.assertEqual(problem.x_min, 0)
        self.assertEqual(problem.mesh_points, 10)

        # Test raised error for callable function
        with self.assertRaises(TypeError):
            solver.PredictorCorrector(
                x_min, x_min, x_max, initial_value, mesh_points)

        # Test raised error if initial_value not list
        with self.assertRaises(TypeError):
            solver.PredictorCorrector(
                func, x_min, x_max, 1, mesh_points)

    def test_Euler_trapezium(self):

        def func(x, y):
            return [y[0]]
        x_min = 0
        x_max = 1
        initial_value = [1]
        mesh_points = 10

        problem = solver.PredictorCorrector(
            func, x_min, x_max, initial_value, mesh_points)

        _, soln = problem.Euler_trapezium()

        # Test solution at first stepsize
        self.assertEqual(soln[1][0], 1.10525)

        # Test solution at second stepsize
        self.assertEqual(round(soln[2][0], 5), 1.22158)


class TestAdaptiveMethod(unittest.TestCase):
    """
    Test the 'AdaptiveMethod' class.
    """
    def test__init__(self):

        def func(x, y):
            return [-y[0]]
        x_min = 0
        x_max = 1
        initial_value = [1]

        problem = solver.AdaptiveMethod(
            func, x_min, x_max, initial_value)

        # Test initialisation
        self.assertEqual(problem.x_min, 0)
        self.assertEqual(problem.initial_value, [1])

        # Test raised error for callable function
        with self.assertRaises(TypeError):
            solver.AdaptiveMethod(
                x_min, x_min, x_max, initial_value)

        # Test raised error if initial_value not list
        with self.assertRaises(TypeError):
            solver.AdaptiveMethod(
                func, x_min, x_max, 1)

    def test_ode23(self):

        def func(x, y):
            return [-y[0]]
        x_min = 0
        x_max = 1
        initial_value = [1]

        problem = solver.AdaptiveMethod(
            func, x_min, x_max, initial_value, initial_mesh=0.5)
        mesh, soln = problem.ode23()

        # Test end point of mesh
        self.assertGreaterEqual(mesh[-1], 1.0)

        # Test mesh point
        self.assertAlmostEqual(mesh[1], 0.3483788976565)

        # Test solution at first stepsize
        self.assertAlmostEqual(soln[1][0], 0.7052580305097)

    def test_ode45(self):

        def func(x, y):
            return [-y[0]]
        x_min = 0
        x_max = 1
        initial_value = [1]

        problem = solver.AdaptiveMethod(
            func, x_min, x_max, initial_value)
        mesh, soln = problem.ode45()

        # Test mesh points
        self.assertGreaterEqual(mesh[-1], 1.0)
        self.assertAlmostEqual(mesh[1], 0.2)

        # Test solution at first stepsize
        self.assertAlmostEqual(soln[1][0], 0.816247173333)

        mesh, soln = problem.ode45(rel_tol=7e-4)

        # Test mesh points
        self.assertAlmostEqual(mesh[1], 0.1679939278)


if __name__ == '__main__':
    unittest.main()
