import pandas as pd
import pyomo.environ as pyo
from pyomo.opt.base.solvers import OptSolver
import os
from itertools import combinations

class Bidder:
    def __init__(self, bidding_model_object, n_scenario, solver, forecaster, bid_type):

        """
        Initializes the bidder object.

        Arguments:
            bidding_model_object: the model object for tracking for bidding

            n_scenario: number of LMP scenarios

            solver: a Pyomo mathematical programming solver object

            forecaster: an initialized LMP forecaster object


        Returns:
            None
        """

        # copy the inputs
        self.bidding_model_object = bidding_model_object
        self.n_scenario = n_scenario
        self.solver = solver
        self.forecaster = forecaster
        self.bid_type = bid_type

        self._check_inputs()

        # get the generator name
        self.generator = self.bidding_model_object.generator

        # add flowsheets to model
        self.model = pyo.ConcreteModel()

        # declare scenario set
        self.model.SCENARIOS = pyo.Set(initialize=range(self.n_scenario))

        # populate scenario blocks
        self.model.fs = pyo.Block(self.model.SCENARIOS)
        for i in self.model.SCENARIOS:
            self.bidding_model_object.populate_model(self.model.fs[i])

        # save power output variable in the model object
        self._save_power_outputs()

        self.formulate_bidding_problem()

        # declare a list to store results
        self.bids_result_list = []


    def _check_inputs(self):

        """
        Check if the inputs to construct the tracker is valid. If not raise errors.
        """

        self._check_bidding_model_object()
        self._check_n_scenario()
        self._check_solver()

    def _check_bidding_model_object(self):

        """
        Check if tracking model object has the necessary methods and attributes.
        """

        method_list = ["populate_model", "update_model"]
        attr_list = ["power_output", "total_cost", "generator", "pmin", "default_bids"]
        msg = "Tracking model object does not have a "

        for m in method_list:
            obtained_m = getattr(self.bidding_model_object, m, None)
            if obtained_m is None:
                raise AttributeError(
                    msg
                    + m
                    + "() method. "
                    + "The bidder object needs the users to "
                    + "implement this method in their model object."
                )

        for attr in attr_list:
            obtained_attr = getattr(self.bidding_model_object, attr, None)
            if obtained_attr is None:
                raise AttributeError(
                    msg
                    + attr
                    + " property. "
                    + "The bidder object needs the users to "
                    + "specify this property in their model object."
                )

    def _check_n_scenario(self):

        """
        Check if the number of LMP scenarios is an integer and greater than 0.
        """

        # check if it is an integer
        if not isinstance(self.n_scenario, int):
            raise TypeError(
                "The number of LMP scenarios should be an integer, "
                + "but a {} was given.".format(type(self.n_scenario).__name__)
            )

        if self.n_scenario <= 0:
            raise ValueError(
                "The number of LMP scenarios should be greater than zero, "
                + "but {} was given.".format(self.n_scenario)
            )

    def _check_solver(self):

        """
        Check if provides solver is a valid Pyomo solver object.
        """

        if not isinstance(self.solver, OptSolver):
            raise TypeError(
                "The provided solver {} is not a valid Pyomo solver.".format(
                    self.solver
                )
            )

    def _save_power_outputs(self):

        """
        Create references of the power output variable in each price scenario
        block.

        Arguments:
            None

        Returns:
            None
        """

        for i in self.model.SCENARIOS:
            # get the power output
            power_output_name = self.bidding_model_object.power_output
            self.model.fs[i].power_output_ref = pyo.Reference(
                getattr(self.model.fs[i], power_output_name)
            )

        return

    def formulate_bidding_problem(self):

        """
        Formulate the bidding optimization problem by adding necessary
        parameters, constraints, and objective function.

        Arguments:
            None

        Returns:
            None
        """

        # if the number of scenarios is 1, we do a static bidding or optimal bidding, 
        # neither case need extra constaints.

        self._add_bidding_params()
        self._add_bidding_objective()  
        if self.n_scenario != 1:
            self._add_bidding_constraints()

        return

    def _add_bidding_params(self):

        """
        Add necessary bidding parameters to the model, i.e., market energy price.

        Arguments:
            None

        Returns:
            None
        """

        for i in self.model.SCENARIOS:
            time_index = self.model.fs[i].power_output_ref.index_set()
            self.model.fs[i].energy_price = pyo.Param(
                time_index, initialize=0, mutable=True
            )
        return

    def _add_bidding_constraints(self):

        """
        Add bidding constraints to the model, i.e., the bid curves need to be
        nondecreasing.

        Arguments:
            None

        Returns:
            None
        """

        # declare a constraint list
        self.model.bidding_constraints = pyo.ConstraintList()

        # generate scenarios combinations
        scenario_comb = list(combinations(self.model.SCENARIOS, 2))

        for k in scenario_comb:
            time_index = self.model.fs[k[0]].power_output_ref.index_set()
            for t in time_index:
                self.model.bidding_constraints.add(
                    (
                        self.model.fs[k[0]].power_output_ref[t]
                        - self.model.fs[k[1]].power_output_ref[t]
                    )
                    * (
                        self.model.fs[k[0]].energy_price[t]
                        - self.model.fs[k[1]].energy_price[t]
                    )
                    >= 0
                )
        return

    def _add_bidding_objective(self):

        """
        Add objective function to the model, i.e., maximizing the expected profit
        of the energy system.

        Arguments:
            None

        Returns:
            None
        """

        # declare an empty objective
        self.model.obj = pyo.Objective(expr=0, sense=pyo.maximize)

        for k in self.model.SCENARIOS:
            time_index = self.model.fs[k].power_output_ref.index_set()

            cost_name = self.bidding_model_object.total_cost[0]
            cost = getattr(self.model.fs[k], cost_name)
            weight = self.bidding_model_object.total_cost[1]

            for t in time_index:
                self.model.obj.expr += (
                    self.model.fs[k].power_output_ref[t]
                    * self.model.fs[k].energy_price[t]
                    - weight * cost[t]
                )

    def compute_bids(self, date, hour=None, **kwargs):

        """
        Solve the model to bid into the markets. After solving, record the bids
        from the solve.

        Arguments:
            price_forecasts: price forecasts needed to solve the bidding problem. {LMP scenario: [forecast timeseries] }

            date: current simulation date

            hour: current simulation hour

        Returns:
            None
        """
        # write a simple forecaster
        # move the static_bids out. Do not need to specify n_scenario == 1. 

        # if self.n_scenario == 1:
        #     static_bids = {}
        #     price_forecasts = self.forecaster.forecast(date=date, hour=hour, **kwargs)
        #     for price in price_forecasts:
        #         self.static_pass_price_forecasts(price)
        #         self.solver.solve(self.model, tee=True)
        #         bids = self._assemble_bids()
        #         self.record_bids(bids, date=date, hour=hour)
        #         static_bids[price] = bids

        #     return static_bids

        # make it an independent function compute_static_bids(self, lmp) 

        price_forecasts = self.forecaster.forecast(date=date, hour=hour, **kwargs)

        # update the price forecasts
        self._pass_price_forecasts(price_forecasts)
        self.solver.solve(self.model, tee=True)
        bids = self._assemble_bids()
        self.record_bids(bids, date=date, hour=hour)

        return bids


    def compute_static_bids(self,lmp,date,hour):

        if not isinstance(lmp,list):
            raise RuntimeError(
                f"Argument 'lmp' should be a list")

        for price in lmp:
            self.static_pass_price_forecasts(price)
            self.solver.solve(self.model, tee=True)
            bids = self.static_assemble_record_bids(date)
            # self.record_bids(bids, date=date, hour=hour)

        return bids


    def update_model(self, **kwargs):

        """
        Update the flowsheets in all the price scenario blocks to advance time
        step.

        Arguments:
            kwargs: necessary profiles to update the underlying model. {stat_name: [...]}

        Returns:
            None
        """

        for i in self.model.SCENARIOS:
            self.bidding_model_object.update_model(b=self.model.fs[i], **kwargs)


    def static_pass_price_forecasts(self, price):
        
        '''
        Set the price forecasts into model parameters.

        Arguments:
            The price_forecast is the price we choose for the optimization problem. 
            Differ from the optimal bidding, we only use one price.
            In the function compute_bids, if scenario == 1, we loop over forecasted lmps.
            The price here is a single number.
            Both static bidding and optimal bidding use the same energy_price data structure
            which is a vector with length of time_index.

        Returns 
            None
        '''
        # if it is a scalar # merge 2 into 1
        for k in self.model.SCENARIOS:
            time_index = self.model.fs[k].energy_price.index_set()
            for t in time_index:
                self.model.fs[k].energy_price[t] = price
        
        return

    def _pass_price_forecasts(self, price_forecasts):

        """
        Pass the price forecasts into model parameters.

        Arguments:
            price_forecasts: price forecasts needed to solve the bidding problem. {LMP scenario: [forecast timeseries] }

        Returns:
            None
        """

        for i in self.model.SCENARIOS:
            time_index = self.model.fs[i].energy_price.index_set()
            for t, p in zip(time_index, price_forecasts[i]):
                self.model.fs[i].energy_price[t] = p

        return

    def static_assemble_record_bids(self,date):
        bids = {}
        gen = self.generator
        for i in self.model.SCENARIOS:
            time_index = self.model.fs[i].energy_price.index_set()
            for t in time_index:
                bids[t] = {}
                bids[t][gen] = {}
                power = round(pyo.value(self.model.fs[i].power_output_ref[t]), 2)
                marginal_cost = round(pyo.value(self.model.fs[i].energy_price[t]), 2)
                bids[t][gen][power] = marginal_cost
                if True:
                    break
        df_list = []
        for t in bids:
            for gen in bids[t]:
                result_dict = {}
                result_dict["Generator"] = gen
                result_dict["Date"] = date
                result_dict["LMP"] = bids[t][gen][power]
                result_dict["Power"] = power
                result_df = pd.DataFrame.from_dict(result_dict, orient="index")
                df_list.append(result_df.T)
        self.bids_result_list.append(pd.concat(df_list))



    def _assemble_bids(self):

        """
        This methods extract the bids out of the stochastic programming model and
        organize them.

        Arguments:

        Returns:
            bids: the bid we computed. It is a dictionary that has this structure. {t: {gen:{power: cost}}}.
        """

        bids = {}
        gen = self.generator

        for i in self.model.SCENARIOS:
            time_index = self.model.fs[i].energy_price.index_set()
            for t in time_index:

                if t not in bids:
                    bids[t] = {}
                if gen not in bids[t]:
                    bids[t][gen] = {}

                power = round(pyo.value(self.model.fs[i].power_output_ref[t]), 2)
                marginal_cost = round(pyo.value(self.model.fs[i].energy_price[t]), 2)

                if power < self.bidding_model_object.pmin:
                    continue
                elif power in bids[t][gen]:
                    bids[t][gen][power] = min(bids[t][gen][power], marginal_cost)
                else:
                    bids[t][gen][power] = marginal_cost

                # if self.bid_type == 's':
                #     break

        for t in time_index:

            # make sure the orignal points in the bids
            for power, marginal_cost in self.bidding_model_object.default_bids.items():
                if power not in bids[t][gen]:
                    bids[t][gen][power] = marginal_cost

            pmin = self.bidding_model_object.pmin

            # sort the curves by power
            bids[t][gen] = dict(sorted(bids[t][gen].items()))

            # make sure the curve is nondecreasing
            pre_power = pmin
            for power, marginal_cost in bids[t][gen].items():

                # ignore pmin, because min load cost is special
                if pre_power == pmin:
                    pre_power = power
                    continue
                bids[t][gen][power] = max(bids[t][gen][power], bids[t][gen][pre_power])
                pre_power = power

            # calculate the actual cost
            pre_power = 0
            pre_cost = 0
            for power, marginal_cost in bids[t][gen].items():

                delta_p = power - pre_power
                bids[t][gen][power] = pre_cost + marginal_cost * delta_p
                pre_power = power
                pre_cost += marginal_cost * delta_p

            # if self.bid_type == 's':
            #     break

        print(self.bidding_model_object.default_bids.items())
        # check if bids are convex
        for t in bids:
            for gen in bids[t]:
                if not self._is_convex_bid(bids[t][gen]):
                    raise RuntimeError(
                        f"Bids for generator {gen} at hour {t} is not convex!"
                    )

        return bids

    @staticmethod
    def _is_convex_bid(bids):

        """
        This method checks the convexity of a bid at a single time period from a
         single generator.

        Arguments:
            bids: a bids at at a single time period from a single generator,
            which is a dictionary whose keys are the power outputs and the values
             are the corresponding costs. {power: cost}

        Returns:
            bids: the bid we computed. It is a dictionary that has this structure. {t: {gen:{power: cost}}}.
        """

        power = list(bids.keys())
        power.sort()

        idx = 0
        delta_p = []
        marginal_cost = []

        # calculate marginal costs (slope)
        while idx < len(power) - 1:
            delta_p.append(power[idx + 1] - power[idx])
            marginal_cost.append(
                (bids[power[idx + 1]] - bids[power[idx]]) / delta_p[-1]
            )
            idx += 1

        # check whether the marginal costs are sorted <=> convex
        idx = 0
        while idx < len(marginal_cost) - 1:
            if round(marginal_cost[idx],8) > round(marginal_cost[idx + 1],8):
                return False
            idx += 1

        return True


    def record_bids(self, bids, date, hour):

        """
        This function records the bids we computed for the given date into a
        DataFrame. This DataFrame has the following columns: gen, date, hour,
        power 1, ..., power n, price 1, ..., price n. And concatenate the
        DataFrame into a class property 'bids_result_list'.

        Arguments:
            bids: the obtained bids for this date.

            date: the date we bid into

            hour: the hour we bid into

        Returns:
            None

        """
            
        df_list = []
        for t in bids:
            for gen in bids[t]:

                result_dict = {}
                result_dict["Generator"] = gen
                result_dict["Date"] = date
                result_dict["Hour"] = t

                pair_cnt = 0
                for power, cost in bids[t][gen].items():
                    result_dict["Power {} [MW]".format(pair_cnt)] = power
                    result_dict["Cost {} [$]".format(pair_cnt)] = cost

                    pair_cnt += 1

                # place holder, in case different len of bids
                while pair_cnt < self.n_scenario:

                    result_dict["Power {} [MW]".format(pair_cnt)] = None
                    result_dict["Cost {} [$]".format(pair_cnt)] = None

                    pair_cnt += 1

                # if self.bid_type == 's':
                #     result_dict["LMP"] = self.static_price

                result_df = pd.DataFrame.from_dict(result_dict, orient="index")
                df_list.append(result_df.T)

            # if self.bid_type == 's':
            #     break
        # save the result to object property
        # wait to be written when simulation ends
        self.bids_result_list.append(pd.concat(df_list))

        return

    def write_results(self, path):
        """
        This methods writes the saved operation stats into an csv file.

        Arguments:
            path: the path to write the results.

        Return:
            None
        """

        print("")
        print("Saving bidding results to disk...")
        pd.concat(self.bids_result_list).to_csv(
            os.path.join(path, "bidding_detail.csv"), index=False
        )
'''
Pseudo code for compute static bids

def compute_static_bids(self, bidding_model_objectives, lmp_prices):

    create bidder object, create forecaster (lmp_prices are from the forecaster? Yes)
    
    create bidding optimization problem with 1 timestep and 1 scenario.
    
    for i in list_lmps():
        set price in optimization to i (update param pyomo repeated solve)
        solver.solve(model)
        record solution
    return and save bids (self.record_bids, self.write_results)

    after the for loop _check the convexity (self._is_convex_bid)

Questions are on the google doc
'''
