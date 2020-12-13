import unittest
import solver


class TestSolverMethod(unittest.TestCase):
    """
    Test the 'SolverMethod' class.
    """
    def test__init__(self):

        func = lambda x, y: -y
        x_min = 0
        x_max = 10
        initial_value = 1
        mesh_points = 100

        solver.SolverMethod(func, x_min, x_max, initial_value, mesh_points)


if __name__ == '__main__':
    unittest.main()
