# -*- coding: utf-8 -*-
#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

import pyomo.environ as pyo

from pyomo.core.base.componentuid import ComponentUID
from pyomo.common.collections import ComponentMap
import pandas as pd
from collections import OrderedDict
# from idaes.apps.caprese.plotlibrary import (
#                         PLANT_PlotLibrary,
#                         NMPC_PlotLibrary,
#                         MHE_PlotLibrary,)

__author__ = "Robert Parker and Kuan-Han Lin"


def empty_dataframe_from_variables(variables, rename_map=None):
    '''
    This function creates a pandas dataframe with variables in columns.

    Parameters
    ----------
    variables : list of variables of interest
    rename_map : dictionary or componentmap that maps the variable to a 
                    specific name string.

    Returns
    -------
    dataframe : pandas dataframe

    '''
    if rename_map is None:
        # Sometimes we construct a variable with a programatically
        # generated name that is not that meaningful to the user, e.g.
        # ACTUALMEASUREMENTBLOCK[1].var. This argument lets us rename
        # to something more meaningful, such as temperature_actual.
        rename_map = ComponentMap()
    var_dict = OrderedDict()
    var_dict["iteration"] = []
    for var in variables:
        if var in rename_map:
            key = rename_map[var]
        else:
            if var.is_reference():
                # NOTE: This will fail if var is a reference-to-list
                # or reference-to-dict.
                key = str(ComponentUID(var.referent))
            else:
                key = str(ComponentUID(var))
        var_dict[key] = []
        dataframe = pd.DataFrame(var_dict)
        # Change the type of iteration column from float to integer
        dataframe["iteration"] = dataframe["iteration"].astype(int)
    return dataframe


def add_variable_values_to_dataframe(
        dataframe,
        variables,
        iteration,
        time_subset=None,
        rename_map=None,
        time_map=None,
        ):
    '''
    This function appends data to the dataframe. 

    Parameters
    ----------
    dataframe : pandas dataframe that we want to append the data to.
    variables : list of variables of interest
    iteration : current iteration.
    time_subset : time indices of interset in the variables.
    rename_map : dictionary or componentmap that maps the variable to a 
                    specific name string.
    time_map : map the time in time_subset to real time points
    '''

    if rename_map is None:
        rename_map = ComponentMap()

    if len(dataframe.index) == 0:
        initial_time = 0.0
    else:
        initial_time = dataframe.index[-1]

    if (len(dataframe.index) != 0 and time_subset is not None
            and time_subset[0] == 0 and time_map is None):
        # The data frame has data in it that we would override with
        # the provided time subset.
        raise RuntimeError()

    df_map = OrderedDict()
    hash_set = set()
    for var in variables:
        idx_set = var.index_set()
        set_hash = hash(tuple(idx_set))
        if len(hash_set) == 0:
            hash_set.add(set_hash)

            if time_subset is None:
                time_subset = list(idx_set)
                if (len(dataframe.index) != 0 and time_subset[0] == 0
                    and time_map is None):
                    time_subset = time_subset[1:]
                
            df_map["iteration"] = len(time_subset)*[iteration]

        elif set_hash not in hash_set:
            raise ValueError()

        if var in rename_map:
            key = rename_map[var]
        else:
            if var.is_reference():
                # NOTE: This will fail if var is a reference-to-list
                # or reference-to-dict.
                key = str(ComponentUID(var.referent))                
            else:
                key = str(ComponentUID(var))

        # append only the desired values
        df_map[key] = [var[t].value for t in time_subset] # Values to append for each variable

    if time_map:
        time_points = [time_map[t] for t in time_subset]
    else:
        time_points = [initial_time + t for t in time_subset]
        # If we aren't using initial_time, we should avoid doing floating
        # point arithmetic.
        #time_points = [t for t in time_subset]

    df = pd.DataFrame(df_map, index=time_points)

    return dataframe.append(df)

def add_variable_setpoints_to_dataframe(dataframe, 
                                        variables, 
                                        iteration,
                                        time_subset, 
                                        map_for_user_given_vars = None):
    '''
    Save the setpoints for states of interest in nmpc's dataframe.

    TODO: replace this function with 'add_variable_values_to_dataframe' when we
    don't save setpoints in attributes'
    '''

    df_map = OrderedDict()
    df_map["iteration"] = len(time_subset)*[iteration]

    for var in variables:
        column_name = str(ComponentUID(var.referent))
        if var in map_for_user_given_vars:
            # User given variables are not nmpc_var, so they don't have setpoint attribute.
            # User a componentmap to get corresponding nmpc_var.            
            df_map[column_name] = len(time_subset)* \
                                    [map_for_user_given_vars[var].setpoint]
        else:
            df_map[column_name] = len(time_subset)*[var.setpoint]

    if len(dataframe.index) == 0:
        initial_time = 0.0
    else:
        initial_time = dataframe.index[-1]
    time_points = [initial_time + t for t in time_subset]
    df = pd.DataFrame(df_map, index=time_points)         

    return dataframe.append(df)

class PlantDataManager(object):
    def __init__(self, 
                 plantblock,
                 user_interested_states = None,
                 user_interested_inputs = None):
                
        # Make sure given variables are all in plantblock
        if user_interested_states is not None:
            cuid_states = [ComponentUID(var.referent) for var in user_interested_states]
            self.plant_user_interested_states = [cuid.find_component_on(plantblock)
                                                 for cuid in cuid_states]
        else:
            self.plant_user_interested_states = []
            
        if user_interested_inputs is not None:
            cuid_inputs = [ComponentUID(var.referent) for var in user_interested_inputs]
            self.plant_user_interested_inputs = [cuid.find_component_on(plantblock)
                                                 for cuid in cuid_inputs]
        else:
            self.plant_user_interested_inputs = []
        
        self.plantblock = plantblock
        self.plant_vars_of_interest = self.plant_user_interested_states + \
                                        self.plant_user_interested_inputs + \
                                            self.plantblock.differential_vars + \
                                                self.plantblock.input_vars

        self.plant_df = empty_dataframe_from_variables(self.plant_vars_of_interest)
        
    def get_plant_dataframe(self):
        return self.plant_df
        
    def save_initial_plant_data(self):
        plant_states_to_save = self.plant_user_interested_states + self.plantblock.differential_vars
        self.plant_df = add_variable_values_to_dataframe(self.plant_df,
                                                         plant_states_to_save, #no inputs
                                                         iteration = 0,
                                                         time_subset = [self.plantblock.time.first()])

    def save_plant_data(self, iteration):
        #skip time.first()
        time_subset = self.plantblock.time.ordered_data()[1:]
        self.plant_df = add_variable_values_to_dataframe(self.plant_df,
                                                         self.plant_vars_of_interest,
                                                         iteration,
                                                         time_subset = time_subset,)


class ControllerDataManager(object):
    def __init__(self, 
                 controllerblock, 
                 user_interested_states = None,
                 user_interested_inputs = None):

        self.controllerblock = controllerblock
        # Convert vars in plant to vars in controller
        if user_interested_states is not None:
            cuid_states = [ComponentUID(var.referent) for var in user_interested_states]
            self.controller_user_interested_states = [cuid.find_component_on(controllerblock)
                                                      for cuid in cuid_states]
        else:
            self.controller_user_interested_states = []

        if user_interested_inputs is not None:
            cuid_inputs = [ComponentUID(var.referent) for var in user_interested_inputs]
            self.controller_user_interested_inputs = [cuid.find_component_on(controllerblock)
                                                      for cuid in cuid_inputs]
        else:
            self.controller_user_interested_inputs = []

        self.controller_vars_of_interest = self.controller_user_interested_inputs + \
                                                    controllerblock.input_vars   
        self.controller_df = empty_dataframe_from_variables(self.controller_vars_of_interest)

        self.states_need_setpoints = self.controller_user_interested_states + \
                                            controllerblock.differential_vars

        self.setpoint_df = empty_dataframe_from_variables(self.states_need_setpoints)
        
        # Important!!!
        # User given variables are not nmpc_var, so they don't have setpoint attribute.
        # Create this componentmap to map the given vars to corresponding nmpc_vars.
        vardata_map = self.controllerblock.vardata_map
        t0 = self.controllerblock.time.first()
        self.user_given_vars_map_nmpcvar = ComponentMap((var, vardata_map[var[t0]]) 
                                                        for var in self.controller_user_interested_states)

    def get_controller_dataframe(self):
        return self.controller_df
    
    def get_setpoint_dataframe(self):
        return self.setpoint_df

    def save_controller_data(self, iteration):
        #skip time.first()
        time = self.controllerblock.time
        #Save values in the first sample time
        time_subset = [t for t in time if t <= self.controllerblock.sample_points[1] 
                                           and t != time.first()]
        self.controller_df = add_variable_values_to_dataframe(self.controller_df,
                                                              self.controller_vars_of_interest,
                                                              iteration,
                                                              time_subset = time_subset,)

        self.setpoint_df = add_variable_setpoints_to_dataframe(self.setpoint_df,
                                                               self.states_need_setpoints,
                                                               iteration,
                                                               time_subset,
                                                               self.user_given_vars_map_nmpcvar,)

class EstimatorDataManager(object):
    def __init__(self, 
                 estimatorblock, 
                 user_interested_states = None,):
        
        self.estimatorblock = estimatorblock
        # Convert vars in plant to vars in estimator
        if user_interested_states is not None:
            cuid_states = [ComponentUID(var.referent) for var in user_interested_states]
            self.estimator_user_interested_states = [cuid.find_component_on(estimatorblock)
                                                     for cuid in cuid_states]
        else:
            self.estimator_user_interested_states = []
            
        self.estimator_vars_of_interest = self.estimator_user_interested_states + \
                                                estimatorblock.differential_vars
        self.estimator_df = empty_dataframe_from_variables(self.estimator_vars_of_interest)

    def get_estimator_dataframe(self):
        return self.estimator_df

    def save_estimator_data(self, iteration):
        time = self.estimatorblock.time
        t_last = time.last()
        # Estimation starts from the very first sample time.
        time_map = OrderedDict([(t_last, (iteration+1)*self.estimatorblock.sample_time),])
        self.estimator_df = add_variable_values_to_dataframe(self.estimator_df,
                                                             self.estimator_vars_of_interest,
                                                             iteration,
                                                             time_subset = [t_last],
                                                             time_map = time_map,)
