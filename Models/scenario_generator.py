import numpy as np


def generate_specialty_scenarios(
    patient_specialties,
    specialty_duration_dict,
    Z,
    seed=None
):
    """
    Generate empirical duration scenarios based on patient specialties.
    """

    if seed is not None:
        np.random.seed(seed)

    scenarios = []

    n_patients = len(patient_specialties)

    for z in range(Z):

        scenario = []

        for p in range(n_patients):

            specialty = patient_specialties[p]

            sampled_duration = np.random.choice(
                specialty_duration_dict[specialty]
            )

            scenario.append(sampled_duration)

        scenarios.append(scenario)

    return np.array(scenarios)