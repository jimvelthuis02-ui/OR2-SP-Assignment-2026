import pyomo.environ as pyo

def build_timing_model(patients, scenarios, weights, t_open=0, t_close=480):
    model = pyo.ConcreteModel()

    P = range(len(patients))
    Z = range(len(scenarios))

    model.P = pyo.Set(initialize=P)
    model.Z = pyo.Set(initialize=Z)

    beta_w, beta_i, beta_o = weights

    # ---------- VARIABLES ----------
    model.S = pyo.Var(model.P, domain=pyo.NonNegativeReals)  # planned start times

    # recourse variables per scenario
    model.A = pyo.Var(model.P, model.Z, domain=pyo.NonNegativeReals)  # actual start
    model.W = pyo.Var(model.P, model.Z, domain=pyo.NonNegativeReals)  # waiting
    model.I = pyo.Var(model.P, model.Z, domain=pyo.NonNegativeReals)  # idle
    model.O = pyo.Var(model.Z, domain=pyo.NonNegativeReals)           # overtime

    changeover = 10

    # ---------- CONSTRAINTS ----------

    def first_patient_rule(m, z):
        return m.A[0, z] == t_open
    model.first_patient = pyo.Constraint(model.Z, rule=first_patient_rule)

    def recursion_rule(m, p, z):
        if p == 0:
            return pyo.Constraint.Skip
        return m.A[p, z] >= m.A[p-1, z] + scenarios[z][p-1] + changeover
    model.recursion = pyo.Constraint(model.P, model.Z, rule=recursion_rule)

    def punctuality_rule(m, p, z):
        return m.A[p, z] >= m.S[p]
    model.punctuality = pyo.Constraint(model.P, model.Z, rule=punctuality_rule)

    # waiting definition
    def waiting_rule(m, p, z):
        return m.W[p, z] >= m.A[p, z] - m.S[p]
    model.waiting = pyo.Constraint(model.P, model.Z, rule=waiting_rule)

    # idle definition
    def idle_rule(m, p, z):
        if p == 0:
            return pyo.Constraint.Skip
        return m.I[p, z] >= m.A[p, z] - (
            m.A[p-1, z] + scenarios[z][p-1] + changeover
        )
    model.idle = pyo.Constraint(model.P, model.Z, rule=idle_rule)

    # overtime
    def overtime_rule(m, z):
        last = len(patients)-1
        return m.O[z] >= (
            m.A[last, z] + scenarios[z][last] - t_close
        )
    model.overtime = pyo.Constraint(model.Z, rule=overtime_rule)

    # ---------- OBJECTIVE ----------
    def objective_rule(m):
        total = 0
        for z in m.Z:
            total += sum(beta_w * m.W[p,z] for p in m.P)
            total += sum(beta_i * m.I[p,z] for p in m.P if p != 0)
            total += beta_o * m.O[z]

        return total / len(scenarios)

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model