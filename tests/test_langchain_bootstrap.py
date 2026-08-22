"""An optional dependency we do not use must not be on the import path.

`langchain_core.language_models.base` ends with a guarded
`from transformers import GPT2TokenizerFast`, a fallback token counter used only
when a model reports no count of its own. Nothing here calls it, and neither
`transformers` nor `torch` is in `requirements.txt` -- but both are present in
some developer environments, where the optional import succeeds and drags torch
in behind it. 34.7s to `import agent`, down to 3.3s once it is hidden.

CI installs `requirements.txt` and never had the packages, so CI never paid the
cost and never would have caught this. The slow path was the local one, which is
the awkward direction for a performance problem to run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from secure_rls import _langchain_bootstrap  # noqa: E402


def test_langchain_did_not_load_the_tokenizer() -> None:
    """The property the whole module exists for."""
    import langchain_core.language_models.base as base

    assert base._HAS_TRANSFORMERS is False


def test_transformers_is_left_importable() -> None:
    """The block is scoped to one import and undone in a `finally`.

    Assigning `sys.modules["transformers"] = None` and walking away would be
    simpler and would make the module permanently unimportable for the rest of
    the process -- a rude thing for a library to do to its host.
    """
    assert sys.modules.get("transformers", "absent") is not None

    pytest.importorskip("transformers", reason="not installed here, as in CI")
    import transformers

    assert transformers.__version__


def test_prepare_is_idempotent() -> None:
    """It runs on import; a second call must be a no-op, not a second block."""
    assert _langchain_bootstrap.prepare() is False


def test_importing_the_agent_does_not_pull_in_torch() -> None:
    """Asserted in a fresh interpreter, because sys.modules is process-wide.

    A timing assertion would be the obvious test and a flaky one. This checks
    the thing that costs the time instead, and it cannot pass by accident.
    """
    probe = (
        f"import sys; sys.path.insert(0, r'{ROOT}'); import agent; "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", out.stdout


def test_the_escape_hatch_is_documented() -> None:
    """If hiding it ever breaks something, there has to be a way out."""
    assert _langchain_bootstrap._OPT_OUT == "SECURE_RLS_ALLOW_TRANSFORMERS"
