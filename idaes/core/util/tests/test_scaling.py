##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
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
This module contains tests for scaling.
"""

import pytest
import pyomo.environ as pyo
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_statistics import number_activated_objectives
import idaes.core.util.scaling as sc

__author__ = "John Eslick, Tim Bartholomew"


@pytest.mark.unit
def test_find_badly_scaled_vars():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1e6)
    m.y = pyo.Var(initialize=1e-8)
    m.z = pyo.Var(initialize=1e-20)
    m.b = pyo.Block()
    m.b.w = pyo.Var(initialize=1e10)

    a = [id(v) for v, sv in sc.badly_scaled_var_generator(m)]
    assert id(m.x) in a
    assert id(m.y) in a
    assert id(m.b.w) in a
    assert id(m.z) not in a

    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.b.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.x] = 1e-6
    m.scaling_factor[m.y] = 1e6
    m.scaling_factor[m.z] = 1
    m.b.scaling_factor[m.b.w] = 1e-5

    a = [id(v) for v, sv in sc.badly_scaled_var_generator(m)]
    assert id(m.x) not in a
    assert id(m.y) not in a
    assert id(m.b.w) in a
    assert id(m.z) not in a


@pytest.mark.unit
def test_find_unscaled_vars_and_constraints():
    m = pyo.ConcreteModel()
    m.b = pyo.Block()
    m.x = pyo.Var(initialize=1e6)
    m.y = pyo.Var(initialize=1e-8)
    m.z = pyo.Var(initialize=1e-20)
    m.c1 = pyo.Constraint(expr=m.x==0)
    m.c2 = pyo.Constraint(expr=m.y==0)
    m.b.w = pyo.Var([1,2,3], initialize=1e10)
    m.b.c1 = pyo.Constraint(expr=m.b.w[1]==0)
    m.b.c2 = pyo.Constraint(expr=m.b.w[2]==0)

    sc.set_scaling_factor(m.x, 1)
    sc.set_scaling_factor(m.b.w[1], 2)
    sc.set_scaling_factor(m.c1, 1)
    sc.set_scaling_factor(m.b.c1, 1)

    a = [id(v) for v in sc.unscaled_variables_generator(m)]
    # Make sure we pick up the right variales
    assert id(m.x) not in a
    assert id(m.y) in a
    assert id(m.z) in a
    assert id(m.b.w[1]) not in a
    assert id(m.b.w[2]) in a
    assert id(m.b.w[3]) in a
    assert len(a) == 4 #make sure we didn't pick up any other random stuff

    a = [id(v) for v in sc.unscaled_constraints_generator(m)]
    assert id(m.c1) not in a
    assert id(m.b.c1) not in a
    assert id(m.c2) in a
    assert id(m.b.c2) in a
    assert len(a) == 2 #make sure we didn't pick up any other random stuff

@pytest.mark.unit
def test_propogate_indexed_scaling():
    m = pyo.ConcreteModel()
    m.b = pyo.Block()
    m.x = pyo.Var([1,2,3], initialize=1e6)
    m.y = pyo.Var([1,2,3], initialize=1e-8)
    m.z = pyo.Var([1,2,3], initialize=1e-20)
    @m.Constraint([1,2,3])
    def c1(b, i):
        return m.x[i] == 0
    @m.Constraint([1,2,3])
    def c2(b, i):
        return m.y[i] == 0
    m.b.w = pyo.Var([1,2,3], initialize=1e10)
    m.b.c1 = pyo.Constraint(expr=m.b.w[1]==0)
    m.b.c2 = pyo.Constraint(expr=m.b.w[2]==0)

    sc.set_scaling_factor(m.x, 11)
    sc.set_scaling_factor(m.y, 13)
    sc.set_scaling_factor(m.b.w, 16)
    sc.set_scaling_factor(m.c1, 14)

    for i in [1,2,3]:
        assert sc.get_scaling_factor(m.x[i]) is None
        assert sc.get_scaling_factor(m.y[i]) is None
        assert sc.get_scaling_factor(m.z[i]) is None
        assert sc.get_scaling_factor(m.b.w[i]) is None
        assert sc.get_scaling_factor(m.c1[i]) is None
        assert sc.get_scaling_factor(m.c2[i]) is None
    assert sc.get_scaling_factor(m.x) == 11
    assert sc.get_scaling_factor(m.y) == 13
    assert sc.get_scaling_factor(m.z) is None
    assert sc.get_scaling_factor(m.b.w) == 16
    assert sc.get_scaling_factor(m.c1) == 14
    assert sc.get_scaling_factor(m.c2) is None

    sc.propagate_indexed_component_scaling_factors(m)
    for i in [1,2,3]:
        assert sc.get_scaling_factor(m.x[i]) is 11
        assert sc.get_scaling_factor(m.y[i]) is 13
        assert sc.get_scaling_factor(m.z[i]) is None
        assert sc.get_scaling_factor(m.b.w[i]) is 16
        assert sc.get_scaling_factor(m.c1[i]) is 14
        assert sc.get_scaling_factor(m.c2[i]) is None


class TestSingleConstraintScalingTransform():
    @pytest.fixture(scope="class")
    def model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=500)
        m.c1 = pyo.Constraint(expr=m.x <= 1e3)
        m.c2 = pyo.Constraint(expr=m.x == 1e3)
        m.c3 = pyo.Constraint(expr=m.x >= 1e3)
        return m

    @pytest.mark.unit
    def test_unscaled_constraints(self, model):
        assert model.c1.lower is None
        assert model.c1.body is model.x
        assert model.c1.upper.value == pytest.approx(1e3)
        assert model.c2.lower.value == pytest.approx(1e3)
        assert model.c2.body is model.x
        assert model.c2.upper.value == pytest.approx(1e3)
        assert model.c3.lower.value == pytest.approx(1e3)
        assert model.c3.body is model.x
        assert model.c3.upper is None

    @pytest.mark.unit
    def test_not_constraint(self, model):
        with pytest.raises(TypeError):
            sc.constraint_scaling_transform(model.x, 1)

    @pytest.mark.unit
    def test_less_than_constraint(self, model):
        sc.constraint_scaling_transform(model.c1, 1e-3)
        assert model.c1.lower is None
        assert model.c1.body() == pytest.approx(model.x.value / 1e3)
        assert model.c1.upper.value == pytest.approx(1)
        assert sc.get_scaling_factor(model.c1) is None
        sc.constraint_scaling_transform_undo(model.c1)
        assert model.c1.lower is None
        assert model.c1.body() == pytest.approx(model.x.value)
        assert model.c1.upper.value == pytest.approx(1e3)

    @pytest.mark.unit
    def test_equality_constraint(self, model):
        sc.constraint_scaling_transform(model.c2, 1e-3)
        # Do transformation again to be sure that, despite being called twice in
        # a row, the transformation only happens once.
        sc.constraint_scaling_transform(model.c2, 1e-3)
        assert model.c2.lower.value == pytest.approx(1)
        assert model.c2.body() == pytest.approx(model.x.value / 1e3)
        assert model.c2.upper.value == pytest.approx(1)
        assert sc.get_constarint_tranform_applied_scaling_factor(model.c2) is 1e-3
        sc.constraint_scaling_transform_undo(model.c2)
        assert sc.get_constarint_tranform_applied_scaling_factor(model.c2) is None
        assert model.c2.lower.value == pytest.approx(1e3)
        assert model.c2.body() == pytest.approx(model.x.value)
        assert model.c2.upper.value == pytest.approx(1e3)

    @pytest.mark.unit
    def test_greater_than_constraint(self, model):
        sc.constraint_scaling_transform(model.c3, 1e-3)
        assert sc.get_scaling_factor(model.c3) == None
        assert model.c3.lower.value == pytest.approx(1)
        assert model.c3.body() == pytest.approx(model.x.value / 1e3)
        assert model.c3.upper is None
        sc.constraint_scaling_transform_undo(model.c3)
        assert model.c3.lower.value == pytest.approx(1e3)
        assert model.c3.body() == pytest.approx(model.x.value)


class TestScaleSingleConstraint():
    @pytest.fixture(scope="class")
    def model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=500)
        m.c1 = pyo.Constraint(expr=m.x <= 1e3)
        m.c2 = pyo.Constraint(expr=m.x == 1e3)
        m.c3 = pyo.Constraint(expr=m.x >= 1e3)
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.c1] = 1 / 1e3
        m.scaling_factor[m.c2] = 1 / 1e3
        m.scaling_factor[m.c3] = 1 / 1e3
        return m

    @pytest.mark.unit
    def test_unscaled_constraints(self, model):
        assert model.c1.lower is None
        assert model.c1.body is model.x
        assert model.c1.upper.value == pytest.approx(1e3)
        assert model.c2.lower.value == pytest.approx(1e3)
        assert model.c2.body is model.x
        assert model.c2.upper.value == pytest.approx(1e3)
        assert model.c3.lower.value == pytest.approx(1e3)
        assert model.c3.body is model.x
        assert model.c3.upper is None

    @pytest.mark.unit
    def test_not_constraint(self, model):
        with pytest.raises(TypeError):
            sc.scale_single_constraint(model.x)

    @pytest.mark.unit
    def test_less_than_constraint(self, model):
        sc.scale_single_constraint(model.c1)
        assert model.c1.lower is None
        assert model.c1.body() == pytest.approx(model.x.value / 1e3)
        assert model.c1.upper.value == pytest.approx(1)

    @pytest.mark.unit
    def test_equality_constraint(self, model):
        sc.scale_single_constraint(model.c2)
        assert model.c2.lower.value == pytest.approx(1)
        assert model.c2.body() == pytest.approx(model.x.value / 1e3)
        assert model.c2.upper.value == pytest.approx(1)

    @pytest.mark.unit
    def test_greater_than_constraint(self, model):
        sc.scale_single_constraint(model.c3)
        assert model.c3.lower.value == pytest.approx(1)
        assert model.c3.body() == pytest.approx(model.x.value / 1e3)
        assert model.c3.upper is None

    @pytest.mark.unit
    def test_scaling_factor_and_expression_replacement(self, model):
        model.c4 = pyo.Constraint(expr=model.x <= 1e6)
        model.scaling_factor[model.c4] = 1e-6
        sc.scale_single_constraint(model.c4)
        assert model.c4.upper.value == pytest.approx(1)
        assert model.c4 not in model.scaling_factor

    @pytest.fixture(scope="class")
    def model2(self):
        m = pyo.ConcreteModel()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.y <= 1e3)
        return m

    @pytest.mark.unit
    def test_no_scaling_factor(self, model2):
        model2.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        sc.scale_single_constraint(model2.c)
        assert model2.c.upper.value == pytest.approx(1e3)


class TestScaleConstraintsPynumero():
    def model(self):
        m = pyo.ConcreteModel()
        x = m.x = pyo.Var(initialize=1e3)
        y = m.y = pyo.Var(initialize=1e6)
        z = m.z = pyo.Var(initialize=1e4)
        m.c1 = pyo.Constraint(expr=0 == -x * y + z)
        m.c2 = pyo.Constraint(expr=0 == 3*x + 4*y + 2*z)
        m.c3 = pyo.Constraint(expr=0 <= z**3)
        return m

    @pytest.mark.unit
    def test_set_get_unset(self):
        """Make sure the Jacobian from Pynumero matches expectation.  This is
        mostly to ensure we understand the interface and catch if things change.
        """
        m = self.model()
        assert sc.get_scaling_factor(m.x) is None
        assert sc.get_scaling_factor(m.x, default=1) == 1
        sc.set_scaling_factor(m.x, 1e-4)
        assert sc.get_scaling_factor(m.x) == 1e-4
        sc.unset_scaling_factor(m.x)
        assert sc.get_scaling_factor(m.x) is None

    @pytest.mark.unit
    def test_jacobian(self):
        """Make sure the Jacobian from Pynumero matches expectation.  This is
        mostly to ensure we understand the interface and catch if things change.
        """
        m = self.model()
        assert number_activated_objectives(m) == 0
        jac, jac_scaled, nlp = sc.constraint_autoscale_large_jac(m, no_scale=True)
        assert number_activated_objectives(m) == 0

        c1_row = nlp._condata_to_idx[m.c1]
        c2_row = nlp._condata_to_idx[m.c2]
        c3_row = nlp._condata_to_idx[m.c3]
        x_col = nlp._vardata_to_idx[m.x]
        y_col = nlp._vardata_to_idx[m.y]
        z_col = nlp._vardata_to_idx[m.z]

        assert jac[c1_row, x_col] == pytest.approx(-1e6)
        assert jac[c1_row, y_col] == pytest.approx(-1e3)
        assert jac[c1_row, z_col] == pytest.approx(1)

        assert jac[c2_row, x_col] == pytest.approx(3)
        assert jac[c2_row, y_col] == pytest.approx(4)
        assert jac[c2_row, z_col] == pytest.approx(2)

        assert jac[c3_row, z_col] == pytest.approx(3e8)

        # Make sure scaling factors don't affect the result
        sc.set_scaling_factor(m.c1, 1e-6)
        sc.set_scaling_factor(m.x, 1e-3)
        sc.set_scaling_factor(m.y, 1e-6)
        sc.set_scaling_factor(m.z, 1e-4)
        jac, jac_scaled, nlp = sc.constraint_autoscale_large_jac(m, no_scale=True)
        assert jac[c1_row, x_col] == pytest.approx(-1e6)
        # Check the scaled jacobian calculation
        assert jac_scaled[c1_row, x_col] == pytest.approx(-1000)
        assert jac_scaled[c1_row, y_col] == pytest.approx(-1000)
        assert jac_scaled[c1_row, z_col] == pytest.approx(0.01)

    @pytest.mark.unit
    def test_scale_no_var_scale(self):
        """Make sure the Jacobian from Pynumero matches expectation.  This is
        mostly to ensure we understand the interface and catch if things change.
        """
        m = self.model()
        jac, jac_scaled, nlp = sc.constraint_autoscale_large_jac(m)

        c1_row = nlp._condata_to_idx[m.c1]
        c2_row = nlp._condata_to_idx[m.c2]
        c3_row = nlp._condata_to_idx[m.c3]
        x_col = nlp._vardata_to_idx[m.x]
        y_col = nlp._vardata_to_idx[m.y]
        z_col = nlp._vardata_to_idx[m.z]

        assert jac_scaled[c1_row, x_col] == pytest.approx(-100)
        assert jac_scaled[c1_row, y_col] == pytest.approx(-0.1)
        assert jac_scaled[c1_row, z_col] == pytest.approx(1e-4)
        assert m.scaling_factor[m.c1] == pytest.approx(1e-4)

        assert jac_scaled[c2_row, x_col] == pytest.approx(3)
        assert jac_scaled[c2_row, y_col] == pytest.approx(4)
        assert jac_scaled[c2_row, z_col] == pytest.approx(2)

        assert jac_scaled[c3_row, z_col] == pytest.approx(3e2)

    @pytest.mark.unit
    def test_scale_with_var_scale(self):
        """Make sure the Jacobian from Pynumero matches expectation.  This is
        mostly to ensure we understand the interface and catch if things change.
        """
        m = self.model()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.x] = 1e-3
        m.scaling_factor[m.y] = 1e-6
        m.scaling_factor[m.z] = 1e-4

        jac, jac_scaled, nlp = sc.constraint_autoscale_large_jac(m)

        c1_row = nlp._condata_to_idx[m.c1]
        c2_row = nlp._condata_to_idx[m.c2]
        c3_row = nlp._condata_to_idx[m.c3]
        x_col = nlp._vardata_to_idx[m.x]
        y_col = nlp._vardata_to_idx[m.y]
        z_col = nlp._vardata_to_idx[m.z]

        assert jac_scaled[c1_row, x_col] == pytest.approx(-1000)
        assert jac_scaled[c1_row, y_col] == pytest.approx(-1000)
        assert jac_scaled[c1_row, z_col] == pytest.approx(1e-2)
        assert m.scaling_factor[m.c1] == pytest.approx(1e-6)

        assert jac_scaled[c2_row, x_col] == pytest.approx(0.075)
        assert jac_scaled[c2_row, y_col] == pytest.approx(100)
        assert jac_scaled[c2_row, z_col] == pytest.approx(0.5)
        assert m.scaling_factor[m.c2] == pytest.approx(2.5e-5)

        assert jac_scaled[c3_row, z_col] == pytest.approx(3e6)
        assert m.scaling_factor[m.c3] == pytest.approx(1e-6)


    @pytest.mark.unit
    def test_scale_with_ignore_var_scale(self):
        """Make sure the Jacobian from Pynumero matches expectation.  This is
        mostly to ensure we understand the interface and catch if things change.
        """
        m = self.model()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.x] = 1e-3
        m.scaling_factor[m.y] = 1e-6
        m.scaling_factor[m.z] = 1e-4

        jac, jac_scaled, nlp = sc.constraint_autoscale_large_jac(
            m, ignore_variable_scaling=True)

        c1_row = nlp._condata_to_idx[m.c1]
        c2_row = nlp._condata_to_idx[m.c2]
        c3_row = nlp._condata_to_idx[m.c3]
        x_col = nlp._vardata_to_idx[m.x]
        y_col = nlp._vardata_to_idx[m.y]
        z_col = nlp._vardata_to_idx[m.z]

        assert jac_scaled[c1_row, x_col] == pytest.approx(-100)
        assert jac_scaled[c1_row, y_col] == pytest.approx(-0.1)
        assert jac_scaled[c1_row, z_col] == pytest.approx(1e-4)
        assert m.scaling_factor[m.c1] == pytest.approx(1e-4)

        assert jac_scaled[c2_row, x_col] == pytest.approx(3)
        assert jac_scaled[c2_row, y_col] == pytest.approx(4)
        assert jac_scaled[c2_row, z_col] == pytest.approx(2)
        assert m.scaling_factor[m.c2] == pytest.approx(1)

        assert jac_scaled[c3_row, z_col] == pytest.approx(3e2)
        assert m.scaling_factor[m.c3] == pytest.approx(1e-6)

    @pytest.mark.unit
    def test_scale_with_ignore_var_scale_constraint_scale(self):
        """Make sure the Jacobian from Pynumero matches expectation.  This is
        mostly to ensure we understand the interface and catch if things change.
        """
        m = self.model()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.c1] = 1e-6
        m.scaling_factor[m.x] = 1e-3
        m.scaling_factor[m.y] = 1e-6
        m.scaling_factor[m.z] = 1e-4

        jac, jac_scaled, nlp = sc.constraint_autoscale_large_jac(
            m, ignore_variable_scaling=True)

        c1_row = nlp._condata_to_idx[m.c1]
        c2_row = nlp._condata_to_idx[m.c2]
        c3_row = nlp._condata_to_idx[m.c3]
        x_col = nlp._vardata_to_idx[m.x]
        y_col = nlp._vardata_to_idx[m.y]
        z_col = nlp._vardata_to_idx[m.z]

        assert jac_scaled[c1_row, x_col] == pytest.approx(-1)
        assert jac_scaled[c1_row, y_col] == pytest.approx(-1e-3)
        assert jac_scaled[c1_row, z_col] == pytest.approx(1e-6)
        assert m.scaling_factor[m.c1] == pytest.approx(1e-6)

        assert jac_scaled[c2_row, x_col] == pytest.approx(3)
        assert jac_scaled[c2_row, y_col] == pytest.approx(4)
        assert jac_scaled[c2_row, z_col] == pytest.approx(2)
        assert m.scaling_factor[m.c2] == pytest.approx(1)

        assert jac_scaled[c3_row, z_col] == pytest.approx(3e2)
        assert m.scaling_factor[m.c1] == pytest.approx(1e-6)


class TestScaleConstraints():
    @pytest.fixture(scope="class")
    def model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1e3)
        m.y = pyo.Var(initialize=1e6)
        m.c1 = pyo.Constraint(expr=m.x == 1e3)
        m.c2 = pyo.Constraint(expr=m.y == 1e6)
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.c1] = 1e-3
        m.scaling_factor[m.c2] = 1e-6

        m.b1 = pyo.Block()
        m.b1.c1 = pyo.Constraint(expr=m.x <= 1e9)
        m.b1.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.b1.scaling_factor[m.b1.c1] = 1e-9

        m.b1.b2 = pyo.Block()
        m.b1.b2.c1 = pyo.Constraint(expr=m.x <= 1e12)
        m.b1.b2.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.b1.b2.scaling_factor[m.b1.b2.c1] = 1e-12

        return m

    @pytest.mark.unit
    def test_scale_one_block(self, model):
        sc.scale_constraints(model, descend_into=False)
        # scaled
        assert model.c1.lower.value == pytest.approx(1)
        assert model.c1.body() == pytest.approx(model.x.value / 1e3)
        assert model.c1.upper.value == pytest.approx(1)
        assert model.c2.lower.value == pytest.approx(1)
        assert model.c2.body() == pytest.approx(model.y.value / 1e6)
        assert model.c2.upper.value == pytest.approx(1)
        # unscaled
        assert model.b1.c1.upper.value == pytest.approx(1e9)
        assert model.b1.b2.c1.upper.value == pytest.approx(1e12)

    @pytest.mark.unit
    def test_scale_model(self, model):
        sc.scale_constraints(model)
        assert model.c1.upper.value == pytest.approx(1)
        assert model.b1.c1.upper.value == pytest.approx(1)
        assert model.b1.b2.c1.upper.value == pytest.approx(1)
