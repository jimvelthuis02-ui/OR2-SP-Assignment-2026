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

def generate_empirical_scenarios(durations_data, n_patients, Z):
    """
    Empirical sampling from real dataset (recommended for assignment)
    """
    scenarios = []

    for z in range(Z):
        sampled = np.random.choice(
            durations_data,
            size=n_patients,
            replace=True
        )
        scenarios.append(sampled)

    return np.array(scenarios)