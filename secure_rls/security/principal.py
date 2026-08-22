"""Layer 1 -- identity binding.

The `Principal` is the answer to "who is asking". It is created once, at login,
from the server-side session, and from then on it is passed *around* the model,
never *through* it. No principal field is ever a tool parameter, appears in a
prompt as an instruction the model could rewrite, or is round-tripped through
the browser.

Authentication here is hardcoded, as the brief specifies. In a real deployment
this is where an identity provider goes (Entra ID group claims mapping to row
filters); everything downstream of `Principal` is unchanged by that swap, which
is the point of isolating it here.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from enum import StrEnum

from db import AGENT_COLUMNS

# Demo-only. Real deployments delegate to an IdP and never see a password at
# all; this exists so the demo can switch tenants in front of an audience.
_SALT = b"secure-rls-demo-salt"


def _hash(password: str) -> str:
    return hashlib.sha256(_SALT + password.encode()).hexdigest()


class Role(StrEnum):
    """Access level *within* a tenant.

    The brief asks only for tenant-level isolation. This second axis exists to
    show that the security context generalises: user-level restrictions are a
    policy lookup, not a rewrite.
    """

    ANALYST = "analyst"
    HR_ADMIN = "hr_admin"


@dataclass(frozen=True)
class ColumnPolicy:
    """What a role may see, and at what granularity.

    Two tiers, and the difference matters. A *hidden* column does not exist as
    far as the role is concerned -- it is absent from the schema the model is
    shown, and naming it is refused outright. A *masked* column can be reasoned
    about in aggregate but never read for an individual: an analyst may ask for
    the average salary in Engineering and may not list salaries beside names.

    Hiding is the blunter instrument and is the right one when a column should
    not inform an answer at all. Masking is what you want when the column is
    legitimately part of the analysis and only the granularity is the problem.
    """

    #: Columns the role may name at all. Anything outside this is hidden.
    visible: frozenset[str]
    #: May the role see an individual person's salary, or only aggregates?
    row_level_salary: bool

    def masked_columns(self) -> frozenset[str]:
        return frozenset() if self.row_level_salary else frozenset({"salary"})

    def hidden_columns(self) -> frozenset[str]:
        """Everything the table has that this role may not name.

        Derived by subtraction from the live column set rather than listed, so
        a column added to the table is hidden by default from any role whose
        `visible` set was written out explicitly. Fail closed: a new column
        should have to be granted, not remembered.
        """
        return frozenset(AGENT_COLUMNS) - self.visible


#: Every column the agent's table exposes. Derived from the catalog (ADR-0005)
#: -- this was a hand-written seventh-column list, which is exactly the drift
#: that refactor removed everywhere else.
_ALL_COLUMNS = frozenset(AGENT_COLUMNS)

#: Neither shipped role hides anything: the case study is about tenant
#: isolation, and hiding columns by default would narrow the demo without
#: demonstrating anything the mask does not. The mechanism is enforced and
#: tested so that a role *can* hide a column -- see
#: `tests/test_column_policy.py`.
ROLE_POLICY: dict[Role, ColumnPolicy] = {
    # HR administrators see individual compensation.
    Role.HR_ADMIN: ColumnPolicy(visible=_ALL_COLUMNS, row_level_salary=True),
    # Analysts may compute salary statistics but may not read a named person's
    # pay. This is the aggregate/individual distinction that plain RLS does not
    # express -- RLS answers "which rows", not "how precisely may you look".
    Role.ANALYST: ColumnPolicy(visible=_ALL_COLUMNS, row_level_salary=False),
}


@dataclass(frozen=True)
class Principal:
    """An authenticated identity. Immutable by construction."""

    username: str
    tenant_id: str
    role: Role
    display_name: str

    @property
    def policy(self) -> ColumnPolicy:
        return ROLE_POLICY[self.role]

    def cache_key(self, *parts: str) -> str:
        """Namespace any cache or memory key by tenant.

        A shared cache is a cross-tenant read channel with a performance
        justification attached; every key in this system starts with the tenant.
        """
        return "|".join([self.tenant_id, self.role.value, *parts])

    def __repr__(self) -> str:  # keep principals out of logs by accident
        return f"Principal({self.username}@{self.tenant_id}/{self.role.value})"


@dataclass(frozen=True)
class _Account:
    principal: Principal
    password_hash: str


def _account(username: str, tenant: str, role: Role, name: str, password: str) -> _Account:
    return _Account(Principal(username, tenant, role, name), _hash(password))


# Credentials are documented in the README, as the brief requires.
_ACCOUNTS: dict[str, _Account] = {
    a.principal.username: a
    for a in [
        _account("acme_admin", "acme", Role.HR_ADMIN, "Dana Kovac (Acme HR)", "acme123"),
        _account("acme_analyst", "acme", Role.ANALYST, "Ravi Patel (Acme People Ops)", "acme123"),
        _account("beta_admin", "beta", Role.HR_ADMIN, "Milos Horak (Beta HR)", "beta123"),
        _account("beta_analyst", "beta", Role.ANALYST, "Sara Lindqvist (Beta Ops)", "beta123"),
        _account("gamma_admin", "gamma", Role.HR_ADMIN, "Nadia Haddad (Gamma HR)", "gamma123"),
        _account("gamma_analyst", "gamma", Role.ANALYST, "Tom Bakker (Gamma Ops)", "gamma123"),
    ]
}


class AuthenticationError(RuntimeError):
    """Login failed. Deliberately says nothing about which half was wrong."""


def authenticate(username: str, password: str) -> Principal:
    """Return the principal for valid credentials, else raise.

    Uses a constant-time comparison and gives an identical error for an unknown
    user and a wrong password, so the endpoint does not enumerate accounts.
    """
    account = _ACCOUNTS.get(username)
    candidate = _hash(password or "")
    expected = account.password_hash if account else _hash("\x00never")
    ok = hmac.compare_digest(candidate, expected)
    if not account or not ok:
        raise AuthenticationError("invalid username or password")
    return account.principal


def demo_accounts() -> list[tuple[str, str, str]]:
    """(username, tenant, role) for every demo account -- used by the UI and docs."""
    return [
        (a.principal.username, a.principal.tenant_id, a.principal.role.value)
        for a in _ACCOUNTS.values()
    ]
