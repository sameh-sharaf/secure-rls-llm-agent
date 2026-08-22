"""Browser-side chat conveniences: copy buttons, jump-to-latest, arrival scroll.

All three need DOM access and Streamlit has no API for any of them, so this
injects a script -- `st.html(..., unsafe_allow_javascript=True)`, which runs in
the app's own document. It lives in its own module so `app.py` stays readable
and so the blast radius is visible.

The obvious route, `components.html`, was the wrong one twice over: it is
deprecated in favour of `st.iframe` (which takes a src, not markup, so it is no
replacement for this), and it renders inside an iframe, which means reaching
back out through `window.parent` for every DOM operation. `st.html` needs
neither. The script still picks its document by looking for where the chat
actually is, so it works either way and cannot wander into a host page if the
app is itself embedded.

Contained the way the spinner CSS is. It touches nothing but presentation, every
step is guarded, and if Streamlit's internals move the page loses three
conveniences rather than breaking. Nothing here reads or writes application
state: message text is passed in from Python rather than scraped out of the
page, which also keeps the reasoning expander and the caption out of what gets
copied.
"""

from __future__ import annotations

import json

import streamlit as st

_JS = """
<script>
(function () {
  var MSG = '[data-testid="stChatMessage"]';

  // Pick whichever document actually holds the chat, rather than assuming.
  // `st.html` runs here in the app's own document; if that ever changes, or if
  // the app is embedded, this finds the right one instead of guessing at
  // `window.parent` and landing in a host page.
  function pick() {
    try { if (document.querySelector(MSG)) return window; } catch (e) {}
    try {
      if (window.parent && window.parent !== window &&
          window.parent.document.querySelector(MSG)) return window.parent;
    } catch (e) {}
    return window;
  }

  var win = pick();
  var doc = win.document;
  if (!doc || !doc.body) return;

  var TEXTS = __TEXTS__;
  var TOKEN = "__TOKEN__";

  function scroller() {
    var seen = [
      doc.querySelector('section.main'),
      doc.querySelector('[data-testid="stMain"]'),
      doc.querySelector('[data-testid="stAppViewContainer"] section')
    ];
    for (var i = 0; i < seen.length; i++) {
      var el = seen[i];
      if (el && el.scrollHeight > el.clientHeight + 4) return el;
    }
    return doc.scrollingElement || doc.documentElement;
  }

  function toBottom(smooth) {
    var el = scroller();
    if (!el) return;
    try { el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" }); }
    catch (e) { el.scrollTop = el.scrollHeight; }
  }

  function legacyCopy(text, done) {
    try {
      var area = doc.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      doc.body.appendChild(area);
      area.select();
      doc.execCommand("copy");
      doc.body.removeChild(area);
      done();
    } catch (e) { /* no clipboard here; the button just does nothing */ }
  }

  function copy(text, button) {
    var done = function () {
      button.textContent = "copied";
      setTimeout(function () { button.textContent = "copy"; }, 1200);
    };
    try {
      // The clipboard of the window the click happened in, which is not
      // necessarily this one.
      win.navigator.clipboard.writeText(text).then(done, function () {
        legacyCopy(text, done);
      });
    } catch (e) { legacyCopy(text, done); }
  }

  function addCopyButtons() {
    var messages = doc.querySelectorAll(MSG);
    for (var i = 0; i < messages.length; i++) {
      var node = messages[i];
      if (node.dataset.rlsCopy === "1") continue;
      var text = TEXTS[i];
      if (typeof text !== "string" || !text) continue;
      node.dataset.rlsCopy = "1";
      if (!node.style.position) node.style.position = "relative";
      var button = doc.createElement("button");
      button.textContent = "copy";
      button.className = "rls-copy";
      button.setAttribute("aria-label", "Copy this message");
      button.addEventListener("click", (function (t, b) {
        return function (ev) { ev.preventDefault(); ev.stopPropagation(); copy(t, b); };
      })(text, button));
      node.appendChild(button);
    }
  }

  function styles() {
    if (doc.getElementById("rls-chat-ux-style")) return;
    var css = doc.createElement("style");
    css.id = "rls-chat-ux-style";
    css.textContent = [
      '[data-testid="stChatMessage"] .rls-copy {',
      '  position: absolute; top: 0.4rem; right: 0.5rem;',
      '  padding: 0.1rem 0.5rem; font-size: 0.72rem; line-height: 1.5;',
      '  border-radius: 0.35rem; cursor: pointer; opacity: 0;',
      '  transition: opacity 0.15s ease;',
      '  border: 1px solid rgba(128,128,128,0.35);',
      '  background: rgba(128,128,128,0.12); color: inherit;',
      '}',
      '[data-testid="stChatMessage"]:hover .rls-copy,',
      '[data-testid="stChatMessage"] .rls-copy:focus { opacity: 0.85; }',
      '[data-testid="stChatMessage"] .rls-copy:hover { opacity: 1; }',
      '#rls-jump {',
      '  position: fixed; left: 50%; transform: translateX(-50%);',
      '  bottom: 6.5rem; z-index: 999;',
      '  padding: 0.3rem 0.9rem; font-size: 0.78rem;',
      '  border-radius: 999px; cursor: pointer;',
      '  border: 1px solid rgba(128,128,128,0.35);',
      '  background: var(--secondary-background-color, rgba(38,42,50,0.96));',
      '  color: inherit; box-shadow: 0 2px 10px rgba(0,0,0,0.28);',
      '  opacity: 0; pointer-events: none; transition: opacity 0.18s ease;',
      '}',
      '#rls-jump.rls-on { opacity: 0.95; pointer-events: auto; }',
      '#rls-jump:hover { opacity: 1; }',
      '@media (prefers-reduced-motion: reduce) {',
      '  [data-testid="stChatMessage"] .rls-copy, #rls-jump { transition: none; }',
      '}'
    ].join("\\n");
    doc.head.appendChild(css);
  }

  function jumpButton() {
    var button = doc.getElementById("rls-jump");
    if (!button) {
      button = doc.createElement("button");
      button.id = "rls-jump";
      button.textContent = "\\u2193 latest";
      button.addEventListener("click", function () { toBottom(true); });
      doc.body.appendChild(button);
    }
    var el = scroller();
    if (!el) return;
    var update = function () {
      var away = el.scrollHeight - el.scrollTop - el.clientHeight;
      button.classList.toggle("rls-on", away > 140);
    };
    if (win.__rlsScrollTarget && win.__rlsScrollHandler) {
      try {
        win.__rlsScrollTarget.removeEventListener("scroll", win.__rlsScrollHandler);
      } catch (e) {}
    }
    win.__rlsScrollTarget = el;
    win.__rlsScrollHandler = update;
    el.addEventListener("scroll", update, { passive: true });
    update();
  }

  function apply() { styles(); addCopyButtons(); jumpButton(); }

  apply();

  // Streamlit rebuilds the message nodes on reruns that leave this script
  // unchanged, so re-attach on DOM changes rather than assuming one pass holds.
  //
  // Debounced, and the observer is disconnected while `apply` runs: it inserts
  // elements itself, and an observer that reacts to its own writes is one
  // careless edit away from a loop that pins a core.
  try {
    if (win.__rlsObserver) win.__rlsObserver.disconnect();
    var queued = null;
    win.__rlsObserver = new MutationObserver(function () {
      if (queued) return;
      queued = setTimeout(function () {
        queued = null;
        win.__rlsObserver.disconnect();
        try { apply(); } finally {
          win.__rlsObserver.observe(doc.body, { childList: true, subtree: true });
        }
      }, 120);
    });
    win.__rlsObserver.observe(doc.body, { childList: true, subtree: true });
  } catch (e) {}

  // Scroll to the newest message whenever the transcript changes -- which
  // includes arriving with a restored history, the "scroll down on login" case.
  if (win.__rlsToken !== TOKEN) {
    win.__rlsToken = TOKEN;
    setTimeout(function () { toBottom(false); }, 60);
    setTimeout(function () { toBottom(false); }, 350);
  }
})();
</script>
"""


def inject_chat_ux(texts: list[str]) -> None:
    """Wire up the copy buttons, the jump control and the arrival scroll.

    `texts` is the message text in render order -- user, assistant, user, ... --
    matching the `stChatMessage` elements one for one.
    """
    # `</` is escaped so a message cannot close the script block early. The
    # notes column is the one field in this dataset an attacker is assumed to
    # control -- the indirect-injection case in the threat model -- and it
    # reaches the transcript, so a note containing `</script>` is a real input.
    payload = json.dumps(texts).replace("</", "<\\/")
    html = _JS.replace("__TEXTS__", payload).replace("__TOKEN__", str(len(texts)))
    st.html(html, unsafe_allow_javascript=True)
