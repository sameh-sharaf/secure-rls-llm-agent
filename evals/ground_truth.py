"""Ground truth for the correctness suite.

Computed with pandas straight from employees.csv, filtered by tenant in this
module and nowhere else. This deliberately does *not* go through db.py, the
gateway, or any tool: if the evaluator shared a code path with the system under
test, a single bug would produce a wrong answer and a passing test.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

from db import CSV_PATH


@functools.lru_cache(maxsize=8)
def frame(tenant: str, csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df.tenant_id == tenant].copy()


def headcount(tenant: str) -> float:
    return float(len(frame(tenant)))


def headcount_by_department(tenant: str, department: str) -> float:
    df = frame(tenant)
    return float((df.department == department).sum())


def avg_salary_by_department(tenant: str, department: str) -> float:
    df = frame(tenant)
    subset = df[df.department == department]
    return float(subset.salary.mean()) if len(subset) else 0.0


def top_department_by_avg_salary(tenant: str) -> str:
    df = frame(tenant)
    return str(df.groupby("department").salary.mean().idxmax())


def bottom_department_by_avg_salary(tenant: str) -> str:
    df = frame(tenant)
    return str(df.groupby("department").salary.mean().idxmin())


def max_salary(tenant: str) -> float:
    return float(frame(tenant).salary.max())


def median_salary(tenant: str) -> float:
    return float(frame(tenant).salary.median())


def total_payroll(tenant: str) -> float:
    return float(frame(tenant).salary.sum())


def department_count(tenant: str) -> float:
    return float(frame(tenant).department.nunique())


def avg_performance(tenant: str) -> float:
    return float(frame(tenant).performance_score.mean())


def department_of(tenant: str, name: str) -> str:
    df = frame(tenant)
    rows = df[df.name == name]
    return str(rows.department.iloc[0]) if len(rows) else ""


def hires_in_year(tenant: str, year: int) -> float:
    df = frame(tenant)
    years = pd.to_datetime(df.hire_date).dt.year
    return float((years == int(year)).sum())


REGISTRY = {
    "headcount": headcount,
    "headcount_by_department": headcount_by_department,
    "avg_salary_by_department": avg_salary_by_department,
    "top_department_by_avg_salary": top_department_by_avg_salary,
    "bottom_department_by_avg_salary": bottom_department_by_avg_salary,
    "max_salary": max_salary,
    "median_salary": median_salary,
    "total_payroll": total_payroll,
    "department_count": department_count,
    "avg_performance": avg_performance,
    "department_of": department_of,
    "hires_in_year": hires_in_year,
}


def resolve(tenant: str, spec: dict) -> float | str:
    """Look up and call a ground-truth function from a YAML `truth:` block."""
    spec = dict(spec)
    fn_name = spec.pop("fn")
    fn = REGISTRY[fn_name]
    return fn(tenant, **spec)
