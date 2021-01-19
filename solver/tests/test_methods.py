#
# This file is part of numerical-solver
# (https://github.com/FarmHJ/numerical-solver/) which is released under the BSD
# 3-clause license. See accompanying LICENSE.md for copyright notice and full
# license details.
#

import unittest

import numpy as np

import solver


class TestSolverMethod(unittest.TestCase):
    """
    Test the 'SolverMethod' class.
    """
    def test__init__(self):

        func = lambda x, y: -y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.SolverMethod(
            func, x_min, x_max, initial_value, mesh_points)

        self.assertEqual(problem.x_min, 0)
        self.assertEqual(problem.mesh_points, 10)

    def test_Euler_explicit(self):

        func = lambda x, y: -y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.SolverMethod(
            func, x_min, x_max, initial_value, mesh_points)
        mesh, soln = problem.Euler_explicit()

        self.assertEqual(np.shape(mesh), (11,))
        self.assertEqual(np.shape(soln), (11,))
        self.assertEqual(soln[1], 0.9)
        self.assertEqual(soln[2], 0.81)

    def test_fixed_pt_iteration(self):

        func = lambda x, y: -y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.SolverMethod(
            func, x_min, x_max, initial_value, mesh_points)
        y_pred = problem.fixed_pt_iteration(initial_value, 0.1)

        self.assertEqual(y_pred, 0.909)

        func = lambda x, y: -10 * y
        initial_value = 1.1
        mesh_points = 10

        problem = solver.SolverMethod(
            func, x_min, x_max, initial_value, mesh_points)

        with self.assertRaises(RuntimeError):
            problem.fixed_pt_iteration(initial_value, 0.1)

    def test_Euler_implicit(self):

        func = lambda x, y: -y
        x_min = 0
        x_max = 1
        initial_value = 1
        mesh_points = 10

        problem = solver.SolverMethod(
            func, x_min, x_max, initial_value, mesh_points)
        mesh, soln = problem.Euler_implicit()

        self.assertEqual(np.shape(mesh), (11,))
        self.assertEqual(np.shape(soln), (11,))
        self.assertEqual(soln[1], 0.909)
        self.assertAlmostEqual(soln[2], 0.82719)


if __name__ == '__main__':
    unittest.main()
