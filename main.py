from Models.timing_model import build_timing_model
from Models.scenario_generator import generate_specialty_scenarios

from pyomo.opt import SolverFactory

import pyomo.environ as pyo

from pathlib import Path
from typing import Any, cast

import pandas as pd
import numpy as np
import scipy.stats as st
import csv
import time


# ==================================================
# HELPER FUNCTION
# ==================================================

def as_float(pyomo_expr: Any) -> float:

    val = pyo.value(pyomo_expr)

    if val is None:
        raise ValueError(
            "Expected a numeric Pyomo value, got None."
        )

    return float(val)


# ==================================================
# LOAD DATA
# ==================================================

data_path = Path(
    "Data/SOL-SP-2026-Surgery scheduling dataset.csv"
)

if not data_path.exists():

    alt_path = Path("data/dataset.csv")

    if alt_path.exists():
        data_path = alt_path

    else:
        raise FileNotFoundError(
            "Dataset not found."
        )

df = pd.read_csv(data_path)

print("\nDataset loaded successfully.")
print(f"Rows in dataset: {len(df)}")

script_start_time = time.perf_counter()


# ==================================================
# COLUMN NAMES
# ==================================================

specialty_column = "Specialty"
duration_column = "Realised surgery duration"
session_id_column = "Session ID"
session_sequence_column = "Session-sequence position"
patient_id_column = "Patient ID"


# ==================================================
# SELECT PATIENTS WITH KNOWN SESSION + SEQUENCE
# ==================================================

df_subset = df.dropna(
    subset=[
        patient_id_column,
        specialty_column,
        session_id_column,
        session_sequence_column,
    ]
).copy()

sequence_parts = df_subset[session_sequence_column].astype(str).str.extract(
    r"^S(?P<sequence_session_id>\d+)-P(?P<sequence_position>\d+)$"
)

df_subset = pd.concat([df_subset, sequence_parts], axis=1)
df_subset = df_subset.dropna(
    subset=["sequence_session_id", "sequence_position"]
).copy()

df_subset["sequence_session_id"] = df_subset["sequence_session_id"].astype(int)
df_subset["sequence_position"] = df_subset["sequence_position"].astype(int)

df_subset["session_id_numeric"] = pd.to_numeric(
    df_subset[session_id_column],
    errors="coerce",
)
df_subset = df_subset.dropna(subset=["session_id_numeric"]).copy()
df_subset["session_id_numeric"] = df_subset["session_id_numeric"].astype(int)

df_subset = df_subset[
    df_subset["session_id_numeric"] == df_subset["sequence_session_id"]
].copy()

df_subset = df_subset.sort_values(
    by=["session_id_numeric", "sequence_position", patient_id_column]
).reset_index(drop=True)

if df_subset.empty:
    raise ValueError(
        "No patients found with both Session ID and valid Session-sequence position (Sx-Py)."
    )

patients = list(range(len(df_subset)))
n_patients = len(patients)

patient_ids = list(df_subset[patient_id_column])
patient_sessions = list(df_subset[session_id_column])
patient_sequence_positions = list(df_subset[session_sequence_column])

session_count = df_subset["session_id_numeric"].nunique()

print(f"\nEligible sessions found: {session_count}")
print(f"Number of eligible patients: {n_patients}")


# ==================================================
# STORE PATIENT SPECIALTIES
# ==================================================

patient_specialties = list(
    df_subset[specialty_column]
)

print("\nSelected patient specialties:")
specialty_counts = (
    pd.Series(patient_specialties)
    .value_counts()
    .to_dict()
)
print(f"Specialty mix: {specialty_counts}")


# ==================================================
# CREATE SPECIALTY DURATION DICTIONARY
# ==================================================

specialty_duration_dict = {}

specialties = (
    df[specialty_column]
    .dropna()
    .unique()
)

for specialty in specialties:

    subset = df[
        df[specialty_column] == specialty
    ][duration_column].dropna()

    if len(subset) > 0:

        specialty_duration_dict[
            specialty
        ] = subset.values

        print(
            f"Specialty {specialty}: "
            f"{len(subset)} durations"
        )


# ==================================================
# OVERALL DURATION STATISTICS
# ==================================================

all_durations = (
    pd.to_numeric(
        df[duration_column],
        errors="coerce"
    )
    .dropna()
    .to_numpy(dtype=float)
)

print("\nHistorical duration statistics:")

print(
    f"Mean duration: "
    f"{np.mean(all_durations):.2f}"
)

print(
    f"Std duration: "
    f"{np.std(all_durations):.2f}"
)

print(
    f"Min duration: "
    f"{np.min(all_durations):.2f}"
)

print(
    f"Max duration: "
    f"{np.max(all_durations):.2f}"
)


# ==================================================
# PARAMETERS
# ==================================================

weights = (0.6, 0.2, 0.2)

Z = 100
N = 10

Z_test = 500

random_seed = 42


# ==================================================
# SOLVER SETUP
# ==================================================

solver = None
selected_solver_name = None

for solver_name in [
    "gurobi",
    "appsi_highs",
    "highs",
    "cbc",
    "glpk"
]:

    candidate = SolverFactory(solver_name)

    if (
        candidate is not None
        and candidate.available(False)
    ):

        solver = candidate
        selected_solver_name = solver_name

        print(f"\nUsing solver: {solver_name}")

        break


if solver is None:

    raise RuntimeError(
        "No supported solver found."
    )


# ==================================================
# SAA LOOP
# ==================================================

objective_values = []
iteration_runtime_seconds = []

session_groups = list(
    df_subset.groupby("session_id_numeric", sort=True)
)

final_session_models = {}

for i in range(N):

    print(f"\n--- SAA Iteration {i + 1} ---")
    iteration_start_time = time.perf_counter()

    iteration_total_obj = 0.0
    iteration_models = {}

    for session_id, session_df in session_groups:

        session_id_int = int(cast(Any, session_id))

        session_specialties = list(session_df[specialty_column])
        session_patients = list(range(len(session_specialties)))

        scenarios = generate_specialty_scenarios(
            patient_specialties=session_specialties,
            specialty_duration_dict=specialty_duration_dict,
            Z=Z,
            seed=random_seed + i * 1000 + session_id_int
        )

        model = build_timing_model(
            patients=session_patients,
            scenarios=scenarios,
            weights=weights
        )

        solver.solve(
            model,
            tee=False
        )

        iteration_total_obj += as_float(model.obj)
        iteration_models[session_id_int] = model

    objective_values.append(iteration_total_obj)
    iteration_seconds = time.perf_counter() - iteration_start_time
    iteration_runtime_seconds.append(iteration_seconds)

    print(
        f"Objective value (sum over sessions): "
        f"{iteration_total_obj:.2f}"
    )
    print(f"Runtime (s): {iteration_seconds:.2f}")

    final_session_models = iteration_models


if not final_session_models:

    raise RuntimeError(
        "No model generated."
    )


# ==================================================
# SAA SUMMARY
# ==================================================

mean_obj = np.mean(objective_values)

std_obj = np.std(
    objective_values,
    ddof=1
)

avg_iteration_runtime = float(np.mean(iteration_runtime_seconds))
total_saa_runtime = float(np.sum(iteration_runtime_seconds))

ci = st.t.interval(
    confidence=0.95,
    df=len(objective_values) - 1,
    loc=mean_obj,
    scale=std_obj / np.sqrt(len(objective_values))
)

print("\n--- SAA SUMMARY ---")

print(
    f"Mean objective: "
    f"{mean_obj:.2f}"
)

print(
    f"Std deviation: "
    f"{std_obj:.2f}"
)

print(
    f"95% confidence interval: "
    f"[{ci[0]:.2f}, {ci[1]:.2f}]"
)
print(f"Avg SAA iteration runtime (s): {avg_iteration_runtime:.2f}")
print(f"Total SAA runtime (s): {total_saa_runtime:.2f}")


# ==================================================
# OUT-OF-SAMPLE TEST
# ==================================================

print("\n--- OUT-OF-SAMPLE TEST ---")
oos_start_time = time.perf_counter()

session_test_rows = []

for idx, (session_id, session_df) in enumerate(session_groups, start=1):

    session_id_int = int(cast(Any, session_id))

    session_specialties = list(session_df[specialty_column])
    session_patients = list(range(len(session_specialties)))

    test_scenarios = generate_specialty_scenarios(
        patient_specialties=session_specialties,
        specialty_duration_dict=specialty_duration_dict,
        Z=Z_test,
        seed=999 + session_id_int
    )

    test_model = build_timing_model(
        patients=session_patients,
        scenarios=test_scenarios,
        weights=weights
    )

    solver.solve(
        test_model,
        tee=False
    )

    session_test_obj = as_float(test_model.obj)

    session_test_rows.append(
        {
            "session_id": session_id_int,
            "n_patients": len(session_patients),
            "test_objective": session_test_obj,
        }
    )

    if idx % 20 == 0 or idx == len(session_groups):
        print(
            f"Out-of-sample progress: {idx}/{len(session_groups)} sessions solved"
        )

test_obj = float(np.mean([
    row["test_objective"]
    for row in session_test_rows
]))

test_obj_weighted = float(np.average(
    [row["test_objective"] for row in session_test_rows],
    weights=[row["n_patients"] for row in session_test_rows],
))

oos_runtime_seconds = time.perf_counter() - oos_start_time

print(f"Test objective (session mean): {test_obj:.2f}")
print(f"Test objective (patient-weighted): {test_obj_weighted:.2f}")
print(f"Out-of-sample runtime (s): {oos_runtime_seconds:.2f}")


# ==================================================
# FINAL SOLUTION
# ==================================================

print("\n--- PLANNED START TIMES ---")

df_subset["local_index_in_session"] = df_subset.groupby("session_id_numeric").cumcount()

planned_start_times = []

for row in df_subset.itertuples(index=False):

    session_id_int = int(cast(Any, row.session_id_numeric))
    local_idx = int(cast(Any, row.local_index_in_session))

    if session_id_int not in final_session_models:
        raise KeyError(f"Missing final model for session {session_id_int}.")

    planned_start_times.append(
        as_float(final_session_models[session_id_int].S[local_idx])
    )

print("Detailed patient-level start times are exported to CSV.")


# ==================================================
# PERFORMANCE BREAKDOWN
# ==================================================

waiting_vals = []
idle_vals = []
overtime_vals = []

for model in final_session_models.values():
    waiting_vals.extend(
        as_float(model.W[p, z])
        for p in model.P
        for z in model.Z
    )

    idle_vals.extend(
        as_float(model.I[p, z])
        for p in model.P
        for z in model.Z
        if p != 0
    )

    overtime_vals.extend(
        as_float(model.O[z])
        for z in model.Z
    )

avg_waiting = float(np.mean(waiting_vals))
avg_idle = float(np.mean(idle_vals)) if idle_vals else 0.0
avg_overtime = float(np.mean(overtime_vals))

print("\n--- PERFORMANCE BREAKDOWN ---")

print(
    f"Average waiting: "
    f"{avg_waiting:.2f}"
)

print(
    f"Average idle: "
    f"{avg_idle:.2f}"
)

print(
    f"Average overtime: "
    f"{avg_overtime:.2f}"
)


# ==================================================
# EXPORT RESULTS
# ==================================================

output_dir = Path("outputs")

output_dir.mkdir(exist_ok=True)


# --------------------------------------------------
# Planned start times
# --------------------------------------------------

rows = [
    {
        "patient_index": p,
        "patient_id": patient_ids[p],
        "session_id": patient_sessions[p],
        "session_sequence_position": patient_sequence_positions[p],
        "specialty": patient_specialties[p],
        "planned_start_time": planned_start_times[p]
    }
    for p in patients
]

planned_times_path = (
    output_dir /
    "planned_start_times.csv"
)

with planned_times_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "patient_index",
            "patient_id",
            "session_id",
            "session_sequence_position",
            "specialty",
            "planned_start_time"
        ]
    )

    writer.writeheader()

    writer.writerows(rows)

print(f"\nSaved: {planned_times_path}")


# --------------------------------------------------
# SAA iteration results
# --------------------------------------------------

saa_path = (
    output_dir /
    "saa_iterations.csv"
)

with saa_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "iteration",
            "objective_value"
        ]
    )

    writer.writeheader()

    writer.writerows([
        {
            "iteration": i + 1,
            "objective_value": obj
        }
        for i, obj in enumerate(objective_values)
    ])

print(f"Saved: {saa_path}")


# --------------------------------------------------
# Out-of-sample per-session results
# --------------------------------------------------

oos_session_path = (
    output_dir /
    "out_of_sample_by_session.csv"
)

with oos_session_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "session_id",
            "n_patients",
            "test_objective"
        ]
    )

    writer.writeheader()
    writer.writerows(session_test_rows)

print(f"Saved: {oos_session_path}")


# --------------------------------------------------
# Summary results
# --------------------------------------------------

summary_path = (
    output_dir /
    "run_summary.csv"
)

summary_rows = [

    {"metric": "solver",
     "value": selected_solver_name},

    {"metric": "n_patients",
     "value": n_patients},

    {"metric": "Z",
     "value": Z},

    {"metric": "N",
     "value": N},

    {"metric": "Z_test",
     "value": Z_test},

    {"metric": "weight_waiting",
     "value": weights[0]},

    {"metric": "weight_idle",
     "value": weights[1]},

    {"metric": "weight_overtime",
     "value": weights[2]},

    {"metric": "mean_objective",
     "value": mean_obj},

    {"metric": "std_objective",
     "value": std_obj},

    {"metric": "ci_lower",
     "value": ci[0]},

    {"metric": "ci_upper",
     "value": ci[1]},

    {"metric": "test_objective_session_mean",
     "value": test_obj},

    {"metric": "test_objective_patient_weighted",
     "value": test_obj_weighted},

    {"metric": "avg_saa_iteration_runtime_seconds",
     "value": avg_iteration_runtime},

    {"metric": "total_saa_runtime_seconds",
     "value": total_saa_runtime},

    {"metric": "out_of_sample_runtime_seconds",
     "value": oos_runtime_seconds},

    {"metric": "avg_waiting",
     "value": avg_waiting},

    {"metric": "avg_idle",
     "value": avg_idle},

    {"metric": "avg_overtime",
     "value": avg_overtime},
]

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "metric",
            "value"
        ]
    )

    writer.writeheader()

    writer.writerows(summary_rows)

print(f"Saved: {summary_path}")

total_runtime_seconds = time.perf_counter() - script_start_time
print("\n--- RUNTIME SUMMARY ---")
print(f"Total script runtime (s): {total_runtime_seconds:.2f}")