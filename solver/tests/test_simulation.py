#
# This file is part of numerical-solver (https://github.com/FarmHJ/numerical-solver/)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import unittest
import solver


class TestSimulationController(unittest.TestCase):
    """
    Test the 'SimulationController' class.
    """
    def test__init__(self):

        start = 0
        end = 10
        with self.assertRaises(TypeError):
            solver.SimulationController(solver.SimulationController, start, end) # noqa
            solver.SimulationController('1', start, end)


if __name__ == '__main__':
    unittest.main()
