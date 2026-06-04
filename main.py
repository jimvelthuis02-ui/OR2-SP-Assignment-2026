from Models.timing_model import build_timing_model
from Models.scenario_generator import generate_scenarios
from pyomo.opt import SolverFactory
import pyomo.environ as pyo
from typing import Any
import csv
from pathlib import Path


def as_float(pyomo_expr: Any) -> float:
    val = pyo.value(pyomo_expr)
    if val is None:
        raise ValueError("Expected a numeric Pyomo value, got None.")
    return float(val)

# --- Example data ---
n_patients = 6
patients = list(range(n_patients))

# --- Scenario generation ---
Z = 20
scenarios = generate_scenarios(mean=4, std=0.5, n_patients=n_patients, Z=Z)

# --- Weights ---
weights = (0.6, 0.2, 0.2)

# --- Build model ---
model: Any = build_timing_model(patients, scenarios, weights)

# --- Solve ---
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
    raise RuntimeError(
        "No supported solver found. Install one of: gurobi, highspy (for highs), cbc, or glpk."
    )

results = solver.solve(model, tee=True)

# --- Output ---
print("\nPlanned start times:")
for p in patients:
    print(f"Patient {p}: {as_float(model.S[p]):.2f}")

# --- Export ---
rows = [
    {
        "patient": p,
        "planned_start_time": as_float(model.S[p]),
    }
    for p in patients
]

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "planned_start_times.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["patient", "planned_start_time"])
    writer.writeheader()
    writer.writerows(rows)

summary_path = output_dir / "solve_summary.txt"
with summary_path.open("w", encoding="utf-8") as f:
    f.write(f"solver={selected_solver_name}\n")
    f.write(f"status={results.solver.status}\n")
    f.write(f"termination={results.solver.termination_condition}\n")
    f.write(f"objective={as_float(model.obj)}\n")

print(f"\nSaved CSV: {csv_path}")
print(f"Saved summary: {summary_path}")

try:
    import pandas as pd

    xlsx_path = output_dir / "planned_start_times.xlsx"
    pd.DataFrame(rows).to_excel(xlsx_path, index=False)
    print(f"Saved Excel: {xlsx_path}")
except ImportError:
    print("Excel export skipped (install pandas and openpyxl to enable .xlsx export).")
