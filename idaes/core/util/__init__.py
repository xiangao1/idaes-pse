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
from .model_serializer import to_json, from_json, StoreSpec
from .misc import svg_tag, copy_port_values, TagReference
from .tags import ModelTag, ModelTagGroup

from pyomo.common.deprecation import relocated_module_attribute

relocated_module_attribute(
    "get_solver", "idaes.core.solvers.get_solver", version="2.0.0.alpha0"
)
