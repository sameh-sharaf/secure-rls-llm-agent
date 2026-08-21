"""Build the per-tenant note indexes. Privileged -- run once at setup."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secure_rls.rag.ingest import build_index  # noqa: E402


def main() -> None:
    counts = build_index()
    total = sum(counts.values())
    print(f"indexed {total} notes into {len(counts)} tenant-private collections")
    for tenant, n in sorted(counts.items()):
        print(f"  notes_{tenant:6} {n:4}")


if __name__ == "__main__":
    main()
