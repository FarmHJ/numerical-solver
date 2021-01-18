#
# This file is part of numerical-solver (https://github.com/FarmHJ/numerical-solver/)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import numpy as np

import solver


class SimulationController(object):
    """SimulationController Class:

    Runs the simulation of any model and controls outputs

    Parameters
    ----------
    model: solver.ForwardModel class
    start: simulation start time
    end: simulation end time
    """

    def __init__(self, model, start, end): # noqa
        super(SimulationController, self).__init__()

        if not issubclass(model, solver.ForwardModel):
            raise TypeError(
                'Model has to be a subclass of solver.ForwardModel.')

        self._model = model()
        self._simulation_times = np.linspace(start, end)

    def run(self, parameters, return_incidence=False):

        output = self._model.simulate(
            parameters,
            self._simulation_times,
            return_incidence)

        return output
