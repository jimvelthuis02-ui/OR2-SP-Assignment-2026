from Models.timing_model import build_timing_model
from Models.scenario_generator import generate_empirical_scenarios
from pyomo.opt import SolverFactory
import pyomo.environ as pyo
from typing import Any, cast
import numpy as np
import pandas as pd
import csv
from pathlib import Path


def as_float(pyomo_expr: Any) -> float:
    val = pyo.value(pyomo_expr)
    if val is None:
        raise ValueError("Expected a numeric Pyomo value, got None.")
    return float(val)


# ===============================
# STEP 1: LOAD DATA
# ===============================
data_path = Path("Data/SOL-SP-2026-Surgery scheduling dataset.csv")
if not data_path.exists():
    alt_path = Path("data/dataset.csv")
    if alt_path.exists():
        data_path = alt_path
    else:
        raise FileNotFoundError(
            "Dataset not found. Expected either 'Data/SOL-SP-2026-Surgery scheduling dataset.csv' "
            "or 'data/dataset.csv'."
        )

df = pd.read_csv(data_path)

# Select one waiting list
waiting_list_id = df["Waiting list"].iloc[0]
df_subset = df[df["Waiting list"] == waiting_list_id].copy()

# Take first 6 patients (fixed sequence)
df_subset = df_subset.head(6)

patients = list(range(len(df_subset)))
n_patients = len(patients)

# Extract real durations
durations_data = df["Realised surgery duration"].dropna().values


# ===============================
# STEP 2: PARAMETERS
# ===============================
weights = (0.6, 0.2, 0.2)

Z = 20       # scenarios per SAA
N = 5        # SAA replications


# ===============================
# SOLVER SETUP
# ===============================
solver = None
selected_solver_name = None

for solver_name in ["gurobi", "appsi_highs", "highs", "cbc", "glpk"]:
    candidate = SolverFactory(solver_name)
    if candidate is not None and candidate.available(False):
        solver = candidate
        selected_solver_name = solver_name
        print(f"Using solver: {solver_name}")
        break

if solver is None:
    raise RuntimeError("No supported solver found.")


# ===============================
# STEP 3: SAA LOOP
# ===============================
objective_values = []
last_model = None

for i in range(N):
    print(f"\n--- SAA Iteration {i+1} ---")

    scenarios = generate_empirical_scenarios(
        durations_data,
        n_patients=n_patients,
        Z=Z
    )

    model = build_timing_model(patients, scenarios, weights)
    results = solver.solve(model, tee=False)

    obj_value = as_float(model.obj)
    objective_values.append(obj_value)

    print(f"Objective value: {obj_value:.2f}")
    last_model = model

if last_model is None:
    raise RuntimeError("SAA loop did not produce a model. Check N and solver status.")

final_model: Any = cast(Any, last_model)


# ===============================
# STEP 4: SUMMARY
# ===============================
mean_obj = np.mean(objective_values)
std_obj = np.std(objective_values)

print("\n--- SAA Summary ---")
print(f"Mean objective: {mean_obj:.2f}")
print(f"Std deviation: {std_obj:.2f}")


# ===============================
# STEP 5: OUT-OF-SAMPLE TEST
# ===============================
Z_test = 200

test_scenarios = generate_empirical_scenarios(
    durations_data,
    n_patients=n_patients,
    Z=Z_test
)

test_model = build_timing_model(patients, test_scenarios, weights)
solver.solve(test_model, tee=False)
test_obj = as_float(test_model.obj)

print("\n--- Out-of-sample performance ---")
print(f"Test objective: {test_obj:.2f}")


# ===============================
# STEP 6: FINAL OUTPUT
# ===============================
print("\nPlanned start times:")
for p in patients:
    print(f"Patient {p}: {as_float(final_model.S[p]):.2f}")


# ===============================
# STEP 7: EXPORT
# ===============================
rows = [
    {"patient": p, "planned_start_time": as_float(final_model.S[p])}
    for p in patients
]

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "planned_start_times.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["patient", "planned_start_time"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved CSV: {csv_path}")

saa_iterations_path = output_dir / "saa_iterations.csv"
with saa_iterations_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["iteration", "objective_value"])
    writer.writeheader()
    writer.writerows(
        [
            {"iteration": i + 1, "objective_value": obj}
            for i, obj in enumerate(objective_values)
        ]
    )

summary_path = output_dir / "run_summary.csv"
summary_rows = [
    {"metric": "solver", "value": selected_solver_name},
    {"metric": "n_patients", "value": n_patients},
    {"metric": "saa_scenarios_per_replication_Z", "value": Z},
    {"metric": "saa_replications_N", "value": N},
    {"metric": "test_scenarios_Z_test", "value": Z_test},
    {"metric": "weight_waiting", "value": weights[0]},
    {"metric": "weight_idle", "value": weights[1]},
    {"metric": "weight_overtime", "value": weights[2]},
    {"metric": "mean_objective", "value": mean_obj},
    {"metric": "std_objective", "value": std_obj},
    {"metric": "test_objective", "value": test_obj},
]

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "value"])
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Saved CSV: {saa_iterations_path}")
print(f"Saved CSV: {summary_path}")
