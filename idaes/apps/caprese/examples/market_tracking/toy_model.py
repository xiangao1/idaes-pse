##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
"""
import random

import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.dae.initialization import solve_consistent_initial_conditions

from idaes.apps.caprese.controller import ControllerBlock
from idaes.apps.caprese.util import apply_noise_with_bounds
from idaes.apps.caprese.nmpc_var import InputVar

import idaes.logger as idaeslog


__author__ = "Robert Parker and Xian Gao"


# See if ipopt is available and set up solver
if pyo.SolverFactory('ipopt').available():
    solver = pyo.SolverFactory('ipopt')
    solver.options = {
            'tol': 1e-6,
            'bound_push': 1e-8,
            'halt_on_ampl_error': 'yes',
            'linear_solver': 'ma57',
            }
else:
    solver = None


def build_model(price, e0=0):
    '''
    Create optimization model for MPC

    Arguments (inputs):
        price: NumPy array with energy price timeseries
        e0: initial value for energy storage level
   
    Returns (outputs):
        my_model: Pyomo optimization model
    '''
   
    # Create a concrete Pyomo model. We'll learn more about this in a few weeks
    my_model = pyo.ConcreteModel()

    ## Define Sets

    # Number of timesteps in planning horizon
    my_model.HORIZON = dae.ContinuousSet(initialize = range(len(price)))

    ## Define Parameters

    # Square root of round trip efficiency
    my_model.sqrteta = pyo.Param(initialize = pyo.sqrt(0.88))

    # Energy in battery at t=0
    my_model.E0 = pyo.Param(initialize = e0, mutable=True)

    ## Define variables
   
    # Charging rate [MW]
    my_model.c = pyo.Var(my_model.HORIZON, initialize = 0.0, bounds=(0, 1))

    # Discharging rate [MW]
    my_model.d = pyo.Var(my_model.HORIZON, initialize = 0.0, bounds=(0, 1))

    # Energy (state-of-charge) [MWh]
    my_model.E = pyo.Var(my_model.HORIZON, initialize = 0.0, bounds=(0, 4))

    ## Define constraints
   
    # Define Energy Balance constraints. [MWh] = [MW]*[1 hr]
    # Note: this model assumes 1-hour timestep in price data and control
    # actions.
    def EnergyBalance(model,t):
        # First timestep
        if t == 0 :
            return (model.E[t] ==
                    model.E0 + model.c[t] * model.sqrteta -
                    model.d[t] / model.sqrteta)
       
        # Subsequent timesteps
        else:
            return (model.E[t] ==
                    model.E[t-1] + model.c[t] * model.sqrteta -
                    model.d[t] / model.sqrteta)
   
    my_model.EnergyBalance_Con = pyo.Constraint(my_model.HORIZON,
            rule = EnergyBalance)
   
    # Enforce the amount of energy is the storage at the final time must equal
    # the initial time.
    # [MWh] = [MWh]
    my_model.PeriodicBoundaryCondition = pyo.Constraint(expr=my_model.E0
            == my_model.E[len(price)-1])
   
    ## Define the objective function (profit)
    # Receding horizon
    def objfun(model):
        return sum((-model.c[t] + model.d[t]) * price[t] for t in model.HORIZON)
    my_model.OBJ = pyo.Objective(rule = objfun, sense = pyo.maximize)
   
    return my_model


def main(plot_switch=False):

    price = [20, 20, 25, 30, 25, 25, 20]
    model = build_model(price)

    simulation_horizon = len(price)
    sample_time = 1

    # We must identify for the controller which variables are our
    # inputs and measurements.
    inputs = [
            model.E[0],
            ]
    measurements = [
            #model.E[0],
            ]

    model.c.fix()
    model.d.fix()
    
    time = model.HORIZON
    controller = ControllerBlock(
            model=model,
            time=time,
            inputs=inputs,
            measurements=measurements,
            )
    controller.construct()
    controller.set_sample_time(sample_time)

    t0 = time.first()
    ts = t0 + sample_time

    controller.initialize_to_initial_conditions(ctype=InputVar)

    model.PeriodicBoundaryCondition.deactivate()

    # Solve the first control problem
    controller.vectors.input[...].unfix()
    solver.solve(controller, tee=True)

    # Extract inputs from controller and inject them into plant
    inputs = controller.generate_inputs_at_time(t0)
    print('Inputs:', list(inputs))

    # TODO: Do something with the inputs... 

    for i in range(1, len(price)):
        print('\nENTERING NMPC LOOP ITERATION %s\n' % i)
        # Here we would:
        # - update the price vector,
        # - update energy storage values,
        # - get the new market signal c and d
        controller.advance_one_sample()
        price.append(price.pop(0))

        solver.solve(controller, tee=True)
        inputs = controller.generate_inputs_at_time(t0)
        print('Inputs:', list(inputs))

if __name__ == '__main__':
    main()

