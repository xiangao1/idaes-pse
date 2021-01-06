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
NRTL property model for a benzene-toluene mixture.
The example model is from the IDAES tutorial,
https://github.com/IDAES/examples-pse/blob/main/src/Tutorials/Advanced/ParamEst/
"""
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import Flash
from idaes.generic_models.properties.activity_coeff_models.BTX_activity_coeff_VLE import BTXParameterBlock
import idaes.logger as idaeslog
from pyomo.environ import *

def NRTL_model(data):
    """This function generates an instance of the NRTL Pyomo model using 'data' as the input argument
    
    Parameters
    ----------
    data: pandas DataFrame, list of dictionaries, or list of json file names
        Data that is used to build an instance of the Pyomo model
    
    Returns
    -------
    m: an instance of the Pyomo model
        for estimating parameters and covariance
    """
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.properties = BTXParameterBlock(default={"valid_phase":
                                                 ('Liq', 'Vap'),
                                                 "activity_coeff_model":
                                                 'NRTL'})
    m.fs.flash = Flash(default={"property_package": m.fs.properties})

    # Initialize at a certain inlet condition
    m.fs.flash.inlet.flow_mol.fix(1)
    m.fs.flash.inlet.temperature.fix(368) 
    m.fs.flash.inlet.pressure.fix(101325)
    m.fs.flash.inlet.mole_frac_comp[0, "benzene"].fix(0.5)
    m.fs.flash.inlet.mole_frac_comp[0, "toluene"].fix(0.5)

    # Set Flash unit specifications
    m.fs.flash.heat_duty.fix(0)
    m.fs.flash.deltaP.fix(0)

    # Fix NRTL specific variables
    # alpha values (set at 0.3)
    m.fs.properties.alpha["benzene", "benzene"].fix(0)
    m.fs.properties.alpha["benzene", "toluene"].fix(0.3)
    m.fs.properties.alpha["toluene", "toluene"].fix(0)
    m.fs.properties.alpha["toluene", "benzene"].fix(0.3)

    # initial tau values
    m.fs.properties.tau["benzene", "benzene"].fix(0)
    m.fs.properties.tau["benzene", "toluene"].fix(0.1690)
    m.fs.properties.tau["toluene", "toluene"].fix(0)
    m.fs.properties.tau["toluene", "benzene"].fix(-0.1559)

    # Initialize the flash unit
    m.fs.flash.initialize(outlvl=idaeslog.INFO_LOW)

    # Fix at actual temperature
    m.fs.flash.inlet.temperature.fix(float(data["temperature"]))

    # Set bounds on variables to be estimated
    m.fs.properties.tau["benzene", "toluene"].setlb(-5)
    m.fs.properties.tau["benzene", "toluene"].setub(5)

    m.fs.properties.tau["toluene", "benzene"].setlb(-5)
    m.fs.properties.tau["toluene", "benzene"].setub(5)

    # Return initialized flash model
    return m


def NRTL_model_opt(theta, theta_names):
    """This function generates an instance of the NRTL Pyomo model using 'theta' and 'theta_names'  as the input arguments
    
    Parameters
    ----------
    theta: dict
        Estimated parameters 
    theta_names: list of strings
        List of estimated Var names
    
    Returns
    -------
    m: an instance of the Pyomo model
        for uncertainty propagation
    """
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
      
    props = m.fs.properties = BTXParameterBlock(default={"valid_phase":
                                                 ('Liq', 'Vap'),
                                                 "activity_coeff_model":
                                                 'NRTL'})
    # Fix NRTL specific variables
    # alpha values (set at 0.3)
    props.alpha["benzene", "benzene"].fix(0)
    props.alpha["benzene", "toluene"].fix(0.3)
    props.alpha["toluene", "toluene"].fix(0)
    props.alpha["toluene", "benzene"].fix(0.3)

    # initial tau values
    props.tau["benzene", "benzene"].fix(0)
    props.tau["benzene", "toluene"].fixed = False #To get the gradients of theta, kaug requires the theta to be unfirxed 
    props.tau["toluene", "toluene"].fix(0)
    props.tau["toluene", "benzene"].fixed = False #To get the gradients of theta, kaug requires the theta to be unfirxed 

    # Set bounds on variables to be estimated
    props.tau["benzene", "toluene"].setlb(-5)
    props.tau["benzene", "toluene"].setub(5)

    props.tau["toluene", "benzene"].setlb(-5)
    props.tau["toluene", "benzene"].setub(5)


    m.fs.flash = Flash(default={"property_package": m.fs.properties})

    # Inlet specifications given above
    m.fs.flash.inlet.flow_mol.fix(1)
    m.fs.flash.inlet.temperature.fix(368)
    m.fs.flash.inlet.pressure.fix(101325)
    m.fs.flash.inlet.mole_frac_comp[0, "benzene"].fix()
    m.fs.flash.inlet.mole_frac_comp[0, "toluene"].fix(0.5)

 
    m.fs.flash.heat_duty.fix(0)
    m.fs.flash.deltaP.fix(0)
    

    m.obj = Objective(expr = 0*m.fs.properties.tau["benzene","toluene"] + exp(-m.fs.properties.alpha['toluene','benzene'].value * m.fs.properties.tau['toluene','benzene']), sense=minimize)
    return m

