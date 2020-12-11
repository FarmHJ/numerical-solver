#
# This file is part of seirmo (https://github.com/SABS-R3-Epidemiology/seirmo/)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import numpy as np

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
