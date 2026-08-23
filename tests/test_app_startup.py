"""The sign-in page must not wait for the ML stack.

`agent` pulls in langchain and, through it, an optional `transformers` import
that drags `torch` behind it. At module scope that put the whole ML stack in
front of the login form.

Measured with `AppTest`, which runs the script as a connecting session does:
47.1s to render the sign-in page originally, 8.8s with the three imports
deferred, 4.1s once `secure_rls/_langchain_bootstrap.py` also stopped the
`transformers` import from happening at all.

Deferring alone was not enough, and the reason is worth keeping: it moved the
wait rather than removing it. The page appeared quickly and then the *sign-in
click* blocked on the same import, which reads to a user as a broken login.

These tests are cheap and exist to keep it that way. A regression here is one
moved import line and produces no error, no failing behaviour and no clue.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Modules that drag in langchain, chromadb or torch. Everything the login page
#: needs must stay outside this set.
DEFERRED = ("agent", "secure_rls.session", "secure_rls.tools")


def _module_level_imports(path: Path) -> set[str]:
    """Imports executed when the file is loaded -- not those inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:  # top level only, deliberately
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_the_login_page_does_not_import_the_agent_stack() -> None:
    at_module_scope = _module_level_imports(ROOT / "app.py")
    offenders = {
        name
        for name in at_module_scope
        if any(name == d or name.startswith(f"{d}.") for d in DEFERRED)
    }
    assert not offenders, (
        f"{sorted(offenders)} imported at module scope in app.py. Every use site "
        f"runs after login; importing here costs ~30s before the form renders."
    )


def test_the_deferred_modules_are_still_reachable() -> None:
    """A guard that passes because the name was deleted would be worse than none."""
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    for expected in ("from agent import SecureAgent", "from secure_rls.session import build_session"):
        assert expected in source, f"{expected!r} is gone -- was it deferred, or lost?"


def test_default_model_does_not_duplicate_the_constant() -> None:
    """The cheap fix was a second copy of the default. This is why it wasn't taken."""
    import agent
    import app

    assert app.default_model() == agent.DEFAULT_MODEL


@pytest.mark.parametrize("module", ["db", "secure_rls.security.principal"])
def test_the_security_modules_stay_light(module: str) -> None:
    """The login page imports these directly; they must not pull the stack in."""
    import importlib

    importlib.import_module(module)
    assert "torch" not in sys.modules or "agent" in sys.modules, (
        f"importing {module} pulled in torch"
    )


# --------------------------------------------------------------------------
# the stale-process detector
# --------------------------------------------------------------------------

def test_stale_modules_names_edited_modules_but_not_the_script(monkeypatch) -> None:
    """Pretend the process started at the epoch: every local module is stale.

    The detector exists because Streamlit re-reads `app.py` on every rerun and
    leaves what it imports in `sys.modules`. Editing `app.py` is therefore not
    stale and must not be reported -- reporting it would train the reader to
    ignore the warning, which is the only way this can fail badly.
    """
    import app

    monkeypatch.setattr(app, "_code_loaded_at", lambda: 0.0)
    stale = app.stale_modules()

    assert stale, "with a zero start time every loaded local module is stale"
    assert any(name.endswith("principal.py") for name in stale)
    assert not any(name == "app.py" for name in stale)


def test_stale_modules_is_quiet_when_nothing_changed(monkeypatch) -> None:
    import time

    import app

    monkeypatch.setattr(app, "_code_loaded_at", lambda: time.time() + 3600)
    assert app.stale_modules() == []


# ------------------------------------------------------------- the layer lab ---


def test_the_layer_lab_is_off_unless_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """A control labelled "switch the security layers off" must not ship visible.

    Not because it is unsafe here -- every object it builds is a throwaway bound
    to the caller's own tenant, and the live session is untouched -- but because
    a reviewer who sees that button before reading the caption forms a fast and
    wrong impression. Opt in with SECURE_RLS_LAB=1.
    """
    import importlib

    for value in ("", "0", "false", "no"):
        monkeypatch.setenv("SECURE_RLS_LAB", value)
        app = importlib.reload(importlib.import_module("app"))
        assert app.LAB_ENABLED is False, value

    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("SECURE_RLS_LAB", value)
        app = importlib.reload(importlib.import_module("app"))
        assert app.LAB_ENABLED is True, value


def test_the_lab_probes_all_name_the_base_table_or_the_catalog() -> None:
    """A probe that cannot possibly leak would demonstrate nothing.

    Each one has to reach for something outside the tenant's own relation,
    or switching the layers off would change nothing visible and the panel
    would teach the opposite of what it is for.
    """
    import importlib

    app = importlib.reload(importlib.import_module("app"))
    for name, sql in app.LAB_PROBES.items():
        assert "employees_base" in sql or "sqlite_master" in sql, name
