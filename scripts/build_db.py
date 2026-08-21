"""Load employees.csv into SQLite. Privileged -- run once at setup."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import CSV_PATH, DB_PATH, build_database, iter_tenants, tenant_user_ids  # noqa: E402


def main() -> None:
    n = build_database(CSV_PATH, DB_PATH)
    print(f"loaded {n} rows into {DB_PATH}")
    for tenant in iter_tenants():
        print(f"  {tenant:6} {len(tenant_user_ids(tenant)):4} rows")


if __name__ == "__main__":
    main()
