"""The chat conveniences, and the one part of them that is not cosmetic.

`chat_ux` injects a script that reaches into Streamlit's DOM. Everything it does
is presentation, so the tests here are narrow on purpose -- there is no value in
asserting CSS. What is worth pinning is the payload: message text goes from
Python into a `<script>` block, and text that closes the block early would break
the page and, with a hostile enough note, do worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat_ux import _JS, inject_chat_ux  # noqa: E402


def test_the_script_cannot_be_closed_early_by_message_text() -> None:
    """A message containing `</script>` must not end the script block.

    Employee notes reach the transcript, and the notes column is the one place
    in this dataset an attacker is assumed to control -- the indirect-injection
    case in the threat model. It never had a route into a script tag before,
    and it does now.
    """
    captured: dict = {}
    import chat_ux

    real = chat_ux.st.html
    try:
        chat_ux.st.html = lambda html, **k: captured.update(html=html)
        inject_chat_ux(["</script><script>window.__pwned=1</script>"])
    finally:
        chat_ux.st.html = real

    html = captured["html"]
    # The template's own closing tag is the only one that may survive.
    assert html.count("</script>") == _JS.count("</script>") == 1
    assert r"<\/script>" in html
    assert "window.__pwned" in html, "the text is still delivered, just inert"


def test_the_placeholders_are_both_replaced(monkeypatch) -> None:
    captured: dict = {}

    def fake_html(html, **kwargs):
        captured["html"] = html
        captured["kwargs"] = kwargs

    import chat_ux

    monkeypatch.setattr(chat_ux.st, "html", fake_html)
    inject_chat_ux(["hello", "hi there"])

    assert "__TEXTS__" not in captured["html"]
    assert "__TOKEN__" not in captured["html"]
    assert "hello" in captured["html"]
    assert captured["kwargs"]["unsafe_allow_javascript"] is True


def test_the_token_changes_with_the_transcript(monkeypatch) -> None:
    """The token is what triggers the scroll-to-latest; a constant would not."""
    seen: list[str] = []

    import chat_ux

    monkeypatch.setattr(chat_ux.st, "html", lambda html, **k: seen.append(html))
    inject_chat_ux(["a", "b"])
    inject_chat_ux(["a", "b", "c", "d"])
    assert seen[0] != seen[1]


def test_no_streamlit_state_is_touched() -> None:
    """The injection is presentation only; if it starts reading state, stop it."""
    source = Path(__file__).resolve().parent.parent.joinpath("chat_ux.py").read_text("utf-8")
    assert "session_state" not in source
    assert "principal" not in source


def test_the_script_finds_the_document_holding_the_chat() -> None:
    """Rather than assuming, and rather than reaching blindly into a parent.

    `st.html` runs the script in the app's own document, so `window.parent` is
    the host page if the app is embedded anywhere -- a place this has no
    business writing to. Picking by "where are the chat messages" is right in
    both arrangements and wrong in neither.
    """
    assert "function pick()" in _JS
    assert "window.parent !== window" in _JS
    assert _JS.count("catch (e) {") >= 4, "every DOM reach should degrade, not raise"


def test_the_observer_cannot_react_to_its_own_writes() -> None:
    """`apply` inserts elements; an observer watching that is a spin waiting."""
    assert "__rlsObserver.disconnect()" in _JS
    assert "finally {" in _JS
