"""Secure multi-tenant analyst.

Importing anything from this package first hides an optional dependency that
langchain would otherwise pull `torch` in for -- 27 seconds to define a token
counter nothing calls. See `_langchain_bootstrap`.

`secure_rls.session` and `secure_rls.tools.factory` import langchain, and both
live in this package, so Python runs this file before either of them. `agent.py`
does not, and calls `prepare()` itself.
"""

from __future__ import annotations

from secure_rls import _langchain_bootstrap  # noqa: F401  (imported for effect)
