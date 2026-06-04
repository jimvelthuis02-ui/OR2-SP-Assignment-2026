import numpy as np

def generate_scenarios(mean, std, n_patients, Z):
    """
    Simple lognormal sampling 
    """
    scenarios = []

    for z in range(Z):
        durations = np.random.lognormal(mean, std, n_patients)
        scenarios.append(durations)

    return np.array(scenarios)