#
# This file is part of seirmo (https://github.com/SABS-R3-Epidemiology/seirmo/)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import numpy as np

import unittest
import solver


class TestForwardModel(unittest.TestCase):
    """
    Test the 'ForwardModel' class.
    """
    def test__init__(self):
        solver.ForwardModel()

    def test_simulate(self):
        forward_model = solver.ForwardModel()
        with self.assertRaises(NotImplementedError):
            forward_model.simulate(0, 1)

if __name__ == '__main__':
    unittest.main()
