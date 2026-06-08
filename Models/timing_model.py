import pyomo.environ as pyo


def build_timing_model(
    patients,
    scenarios,
    weights,
    t_open=0,
    t_close=480,
    changeover=10
):

    model = pyo.ConcreteModel()

    P = range(len(patients))
    Z = range(len(scenarios))

    model.P = pyo.Set(initialize=P)
    model.Z = pyo.Set(initialize=Z)

    beta_w, beta_i, beta_o = weights

    # ==================================================
    # VARIABLES
    # ==================================================

    # Planned start times
    model.S = pyo.Var(
        model.P,
        domain=pyo.NonNegativeReals
    )

    # Actual start times
    model.A = pyo.Var(
        model.P,
        model.Z,
        domain=pyo.NonNegativeReals
    )

    # Waiting time
    model.W = pyo.Var(
        model.P,
        model.Z,
        domain=pyo.NonNegativeReals
    )

    # Idle time
    model.I = pyo.Var(
        model.P,
        model.Z,
        domain=pyo.NonNegativeReals
    )

    # Overtime
    model.O = pyo.Var(
        model.Z,
        domain=pyo.NonNegativeReals
    )

    # ==================================================
    # CONSTRAINTS
    # ==================================================

    # First actual surgery starts at opening time
    def first_patient_rule(m, z):
        return m.A[0, z] == t_open

    model.first_patient = pyo.Constraint(
        model.Z,
        rule=first_patient_rule
    )

    # Propagation of actual surgery times
    def recursion_rule(m, p, z):

        if p == 0:
            return pyo.Constraint.Skip

        return (
            m.A[p, z]
            >=
            m.A[p - 1, z]
            + scenarios[z][p - 1]
            + changeover
        )

    model.recursion = pyo.Constraint(
        model.P,
        model.Z,
        rule=recursion_rule
    )

    # Surgeries cannot start before planned time
    def punctuality_rule(m, p, z):

        return m.A[p, z] >= m.S[p]

    model.punctuality = pyo.Constraint(
        model.P,
        model.Z,
        rule=punctuality_rule
    )

    # Waiting time definition
    def waiting_rule(m, p, z):

        return (
            m.W[p, z]
            >=
            m.A[p, z] - m.S[p]
        )

    model.waiting = pyo.Constraint(
        model.P,
        model.Z,
        rule=waiting_rule
    )

    # Idle time definition
    def idle_rule(m, p, z):

        if p == 0:
            return pyo.Constraint.Skip

        return (
            m.I[p, z]
            >=
            m.A[p, z]
            -
            (
                m.A[p - 1, z]
                + scenarios[z][p - 1]
                + changeover
            )
        )

    model.idle = pyo.Constraint(
        model.P,
        model.Z,
        rule=idle_rule
    )

    # Overtime definition
    def overtime_rule(m, z):

        last = len(patients) - 1

        return (
            m.O[z]
            >=
            (
                m.A[last, z]
                + scenarios[z][last]
                - t_close
            )
        )

    model.overtime = pyo.Constraint(
        model.Z,
        rule=overtime_rule
    )

    # First planned surgery fixed at opening time
    def first_start_fix_rule(m):

        return m.S[0] == t_open

    model.first_start_fix = pyo.Constraint(
        rule=first_start_fix_rule
    )

    # Optional upper bound on planned starts
    def upper_bound_rule(m, p):

        return m.S[p] <= t_close

    model.upper_bound = pyo.Constraint(
        model.P,
        rule=upper_bound_rule
    )

    # ==================================================
    # OBJECTIVE
    # ==================================================

    def objective_rule(m):

        total_waiting = sum(
            beta_w * m.W[p, z]
            for p in m.P
            for z in m.Z
        )

        total_idle = sum(
            beta_i * m.I[p, z]
            for p in m.P
            for z in m.Z
            if p != 0
        )

        total_overtime = sum(
            beta_o * m.O[z]
            for z in m.Z
        )

        total_cost = (
            total_waiting
            + total_idle
            + total_overtime
        )

        return total_cost / len(scenarios)

    model.obj = pyo.Objective(
        rule=objective_rule,
        sense=pyo.minimize
    )

    return model