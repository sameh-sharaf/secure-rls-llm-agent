"""Keep an optional dependency we do not use out of the import path.

`langchain_core.language_models.base` ends with:

    try:
        from transformers import GPT2TokenizerFast
        _HAS_TRANSFORMERS = True
    except ImportError:
        _HAS_TRANSFORMERS = False

It is a fallback token counter, used only by `get_num_tokens` when a model
provides no count of its own. Nothing here calls it -- token accounting comes
from Ollama -- and neither `transformers` nor `torch` appears in
`requirements.txt`.

They are in this developer's environment anyway, so the optional import
succeeds, and `transformers` pulls in `torch`. Measured: 27 of the 34.7 seconds
it took to `import agent`, to define a tokenizer that is never constructed. CI
installs only `requirements.txt` and so never paid it, which is exactly why it
went unnoticed -- the slow path was the local one.

    import agent, transformers present    34.7s
    import agent, transformers blocked     7.5s

The fix is to import that one module while `transformers` is unavailable, so the
`except ImportError` branch is taken, then put the world back. `_HAS_TRANSFORMERS`
is then `False` for the life of the process and every later langchain import is
cheap.

Scoped deliberately. The obvious version -- assign `sys.modules["transformers"]
= None` and leave it -- makes the module permanently unimportable for anything
else in the process, which is a rude thing for a library to do to its host. Here
the block lasts for one import and is removed in a `finally`, so anything that
genuinely wants `transformers` can still have it.

Importing this module performs the work, so that callers need only an import
line and not a statement wedged between their imports -- which would put every
import after it in violation of E402. `prepare()` stays public for the tests.
"""

from __future__ import annotations

import os
import sys

#: Escape hatch. Set to any non-empty value to keep the optional import.
_OPT_OUT = "SECURE_RLS_ALLOW_TRANSFORMERS"

_TARGET = "langchain_core.language_models.base"

_done = False


def prepare() -> bool:
    """Import langchain's base module with `transformers` hidden.

    Returns whether this call did the work. Safe and cheap to call repeatedly;
    safe to call when `transformers` is not installed at all, which is the case
    in CI and the reason this is an optimisation rather than a requirement.
    """
    global _done
    if _done or os.environ.get(_OPT_OUT):
        return False
    _done = True

    if _TARGET in sys.modules:
        # Something imported it first and has already paid. Blocking now would
        # cost the same and change nothing.
        return False

    # Only interfere if `transformers` has not been imported already: if it is
    # in `sys.modules`, the cost is spent and hiding it would be pure loss.
    hide = "transformers" not in sys.modules
    if hide:
        # `None` is the documented way to make an import fail: Python raises
        # ImportError rather than returning the entry, which is precisely the
        # branch langchain_core is written to handle.
        sys.modules["transformers"] = None
    try:
        __import__(_TARGET)
    except ImportError:
        # langchain is not installed, or its layout changed. Not our problem to
        # solve here -- the real import downstream will raise where it means
        # something.
        return False
    finally:
        if hide:
            sys.modules.pop("transformers", None)
    return True


prepare()
