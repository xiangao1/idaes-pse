import pyomo.environ as pyo

class MultiPeriodModel:

    """
        Initialize a MultiPeriodModel

        Arguments:
            n_time_points: number of time points in horizon
            process_model_func: a multiperiod capable steady state model builder
            linking_variable_func: a function that returns a tuple of variable pairs to link between time steps
            periodic_variable_func: a function that returns a tuple of variable pairs to link between last and first time steps
    """
    def __init__(self, n_time_points, process_model_func, linking_variable_func,  periodic_variable_func=None):#, state_variable_func=None):
        self.n_time_points = n_time_points

        #user provided functions
        self.create_process_model = process_model_func
        self.get_linking_variable_pairs = linking_variable_func
        self.get_periodic_variable_pairs = periodic_variable_func
        #self.get_state_variable_pairs = state_variable_func

        #populated on 'build_multi_period_model'
        self._pyomo_model = None                 #pyomo model
        self._first_active_time = None           #index of first active time in problem horizon

        #optional initialzation features
        self.initialization_points = None       #library of possible initial points
        self.initialize_func = None             #function to perform the initialize

    """
        Build a multi-period capable model using user-provided functions

        Arguments:
            model_data_args: dictionary with {time:(arguments,)} where `time` is the time in the horizon.
    """
    def build_multi_period_model(self, model_data_kwargs={}):
        assert(list(range(len(model_data_kwargs)))==sorted(model_data_kwargs))
        m = pyo.ConcreteModel()
        m.TIME = pyo.Set(initialize=range(self.n_time_points))

        #create user defined steady-state models. Each block is a multi-period capable model.
        m.blocks = pyo.Block(m.TIME)
        for t in m.TIME:
            m.blocks[t].process = self.create_process_model(**model_data_kwargs[t])

        #link blocks together. loop over every time index except the last one
        for t in m.TIME.data()[:self.n_time_points-1]: 
            link_variable_pairs = self.get_linking_variable_pairs(m.blocks[t].process,m.blocks[t+1].process)
            self._create_linking_constraints(m.blocks[t].process,link_variable_pairs)

        if self.get_periodic_variable_pairs is not None:
            N = len(m.blocks)
            periodic_variable_pairs = self.get_periodic_variable_pairs(m.blocks[N-1].process,m.blocks[0].process)
            self._create_periodic_constraints(m.blocks[N-1].process,periodic_variable_pairs)

        self._pyomo_model = m
        self._first_active_time = m.TIME.first()
        return m

    """
        Advance the current model instance to the next time period

        Arguments:
            model_data_args: arguments passed to user provided steady-state builder function
    """
    def advance_time(self, **model_data_kwargs):
        m = self._pyomo_model
        previous_time = self._first_active_time
        current_time = m.TIME.next(previous_time)

        #deactivate previous time
        m.blocks[previous_time].process.deactivate()

        #track the first time in the problem horizon
        self._first_active_time = current_time

        #populate new time for the end of the horizon
        last_time = m.TIME.last()
        new_time = last_time + 1
        m.TIME.add(new_time)
        m.blocks[new_time].process = self.create_process_model(**model_data_kwargs)

        #sequential time coupling
        link_variable_pairs = self.get_linking_variable_pairs(m.blocks[last_time].process, m.blocks[new_time].process)
        self._create_linking_constraints(m.blocks[last_time].process,link_variable_pairs)

        #periodic time coupling
        if self.get_periodic_variable_pairs is not None:
            periodic_variable_pairs = self.get_periodic_variable_pairs(m.blocks[new_time].process,m.blocks[current_time].process)
            self._create_periodic_constraints(m.blocks[new_time].process,periodic_variable_pairs)
            #deactivate old periodic constraint
            m.blocks[last_time].process.periodic_constraints.deactivate()

        # TODO: discuss where state goes. sometimes the user might want to fix values based on a 'real' process
        # also TODO: inspect argument and use fix() if possible
        # if self.get_state_variable_pairs is not None:
        #     state_variable_pairs = self.get_state_variable_pairs(m.blocks[previous_time].process, m.blocks[current_time].process)
        #     self._fix_initial_states(m.blocks[current_time].process,state_variable_pairs)

    """
        Retrieve the underlying pyomo model
    """
    @property
    def pyomo_model(self):
        return self._pyomo_model

    """
        Retrieve the current multiperiod model time
    """
    @property
    def current_time(self):
        return self._first_active_time

    """
        Retrieve the active time blocks of the pyomo model
    """
    def get_active_process_blocks(self):
        return [b.process for b in self._pyomo_model.blocks.values() if b.process.active]

    #create linking constraint on the "from block"
    def _create_linking_constraints(self,b1,variable_pairs):
        b1.link_constraints = pyo.Constraint(range(len(variable_pairs)))
        for (i,pair) in enumerate(variable_pairs):
            b1.link_constraints[i] = pair[0]==pair[1]

    #create periodic constraint on the "from block"
    def _create_periodic_constraints(self,b1,variable_pairs):
        b1.periodic_constraints = pyo.Constraint(range(len(variable_pairs)))
        for (i,pair) in enumerate(variable_pairs):
            b1.periodic_constraints[i] = pair[0]==pair[1]
