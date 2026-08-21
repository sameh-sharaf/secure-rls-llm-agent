"""Persisted conversation history.

A transcript is a new store of tenant data -- salaries and names written to
disk, outliving the session that produced them. The threat model already says
the LLM stack creates copies that inherit no access control unless you give it
to them, so these tests treat the store the way the boundary tests treat the
table: assert the isolation, do not assume it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secure_rls.security.conversation import (  # noqa: E402
    MAX_TURNS_PER_USER,
    ConversationStore,
)
from secure_rls.security.principal import Role, authenticate  # noqa: E402


@pytest.fixture
def store(tmp_path: Path) -> ConversationStore:
    return ConversationStore(tmp_path / "conversations.db")


def _p(user: str):
    return authenticate(user, f"{user.split('_')[0]}123")


def test_a_turn_round_trips(store: ConversationStore) -> None:
    p = _p("acme_admin")
    store.append(p, "How many in Sales?", "71 people work in Sales.", [{"kind": "tool"}])
    turns = store.load(p)
    assert len(turns) == 1
    assert turns[0].question == "How many in Sales?"
    assert turns[0].answer.startswith("71 people")
    assert turns[0].trace == [{"kind": "tool"}]


def test_turns_come_back_oldest_first(store: ConversationStore) -> None:
    p = _p("acme_admin")
    for i in range(5):
        store.append(p, f"q{i}", f"a{i}", [])
    assert [t.question for t in store.load(p)] == ["q0", "q1", "q2", "q3", "q4"]


# ------------------------------------------------------------- isolation ---


def test_users_in_the_same_tenant_do_not_share_a_transcript(store: ConversationStore) -> None:
    admin, analyst = _p("acme_admin"), _p("acme_analyst")
    store.append(admin, "admin question", "admin answer", [])
    assert store.load(analyst) == []
    assert len(store.load(admin)) == 1


def test_tenants_do_not_share_a_transcript(store: ConversationStore) -> None:
    acme, beta = _p("acme_admin"), _p("beta_admin")
    store.append(acme, "acme question", "ZZ_CANARY_ACME earns 999999", [])
    assert store.load(beta) == []
    blob = " ".join(t.answer for t in store.load(acme))
    assert "ZZ_CANARY_ACME" in blob


def test_a_role_change_does_not_replay_the_old_role_s_answers(store: ConversationStore) -> None:
    """The clause that is easy to omit and expensive to omit.

    An HR admin sees individual salaries. Demote them to analyst and their own
    history would otherwise replay answers the live system now refuses -- access
    control that ignores yesterday's answers is not access control.
    """
    from dataclasses import replace

    admin = _p("acme_admin")
    store.append(admin, "Who earns most?", "Jane Doe on 163,500.", [])

    demoted = replace(admin, role=Role.ANALYST)
    assert store.load(demoted) == [], "an old role's answers replayed after a demotion"
    assert len(store.load(admin)) == 1


# ------------------------------------------------------- redaction & size ---


def test_direct_identifiers_are_masked_before_they_are_written(
    store: ConversationStore,
) -> None:
    """A store that outlives the session should not hold more than the screen."""
    p = _p("acme_admin")
    store.append(p, "contact bob@example.com", "reach them on bob@example.com", [])
    turn = store.load(p)[0]
    assert "bob@example.com" not in turn.answer
    assert "bob@example.com" not in turn.question
    assert "redacted" in turn.answer


def test_history_is_capped_per_user(store: ConversationStore) -> None:
    p = _p("acme_admin")
    for i in range(MAX_TURNS_PER_USER + 12):
        store.append(p, f"q{i}", f"a{i}", [])
    turns = store.load(p)
    assert len(turns) == MAX_TURNS_PER_USER
    assert turns[-1].question == f"q{MAX_TURNS_PER_USER + 11}", "kept the wrong end"


def test_trimming_one_user_leaves_another_alone(store: ConversationStore) -> None:
    a, b = _p("acme_admin"), _p("beta_admin")
    store.append(b, "beta q", "beta a", [])
    for i in range(MAX_TURNS_PER_USER + 5):
        store.append(a, f"q{i}", f"a{i}", [])
    assert len(store.load(b)) == 1, "trimming reached into another tenant's rows"


def test_clear_erases_only_the_caller(store: ConversationStore) -> None:
    a, b = _p("acme_admin"), _p("beta_admin")
    store.append(a, "acme q", "acme a", [])
    store.append(b, "beta q", "beta a", [])
    removed = store.clear(a)
    assert removed == 1
    assert store.load(a) == []
    assert len(store.load(b)) == 1


def test_clear_removes_every_role_for_that_user(store: ConversationStore) -> None:
    """Erasure means erasure -- not "the turns your current role can see"."""
    from dataclasses import replace

    admin = _p("acme_admin")
    store.append(admin, "as admin", "admin answer", [])
    store.append(replace(admin, role=Role.ANALYST), "as analyst", "analyst answer", [])
    assert store.clear(admin) == 2
    assert store.count(admin) == 0


def test_store_is_separate_from_the_employee_database(tmp_path: Path) -> None:
    """Never mix a constantly-written log into the read-only, authorizer-locked db."""
    from db import DB_PATH
    from secure_rls.security.conversation import STORE_PATH

    assert STORE_PATH != DB_PATH
    assert STORE_PATH.name != DB_PATH.name


# ------------------------------------------------------- which model answered ---
# The model can be switched mid-conversation, and the bake-off measured answer
# accuracy from 72% to 100% across the three local models. A transcript where
# half the turns came from a weaker model and half from a stronger one, with no
# way to tell which, is a transcript you cannot judge.


def test_the_answering_model_round_trips(store: ConversationStore) -> None:
    p = _p("acme_admin")
    store.append(p, "q", "a", [], "qwen2.5:7b")
    assert store.load(p)[0].model == "qwen2.5:7b"


def test_each_turn_keeps_its_own_model(store: ConversationStore) -> None:
    """A mid-conversation switch must not relabel earlier turns."""
    p = _p("acme_admin")
    store.append(p, "q1", "a1", [], "qwen2.5:7b")
    store.append(p, "q2", "a2", [], "llama3.1:8b")
    assert [t.model for t in store.load(p)] == ["qwen2.5:7b", "llama3.1:8b"]


def test_model_is_optional(store: ConversationStore) -> None:
    p = _p("acme_admin")
    store.append(p, "q", "a", [])
    assert store.load(p)[0].model == ""


def test_a_database_from_before_the_column_still_opens(tmp_path: Path) -> None:
    """Migrate rather than discard.

    The store is local and gitignored, so this could be "delete the file" --
    except the entire point of persisting a transcript is that it survives.
    Dropping someone's history to add a column undercuts the feature.
    """
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        """
        CREATE TABLE turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL, username TEXT NOT NULL, role TEXT NOT NULL,
            ts REAL NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
            trace TEXT NOT NULL
        )
        """
    )
    old.execute(
        "INSERT INTO turns (tenant_id, username, role, ts, question, answer, trace)"
        " VALUES ('acme', 'acme_admin', 'hr_admin', 1.0, 'old q', 'old a', '[]')"
    )
    old.commit()
    old.close()

    store = ConversationStore(path)          # must migrate, not explode
    p = _p("acme_admin")
    turns = store.load(p)
    assert len(turns) == 1, "the pre-existing turn was lost"
    assert turns[0].question == "old q"
    assert turns[0].model == ""

    store.append(p, "new q", "new a", [], "gemma4:26b-a4b-it-q4_K_M")
    assert [t.model for t in store.load(p)] == ["", "gemma4:26b-a4b-it-q4_K_M"]
