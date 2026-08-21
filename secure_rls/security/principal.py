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
    """What a role may see, and at what granularity."""

    #: Columns the role may name at all.
    visible: frozenset[str]
    #: May the role see an individual person's salary, or only aggregates?
    row_level_salary: bool

    def masked_columns(self) -> frozenset[str]:
        return frozenset() if self.row_level_salary else frozenset({"salary"})


_ALL_COLUMNS = frozenset(
    {"user_id", "name", "department", "salary", "performance_score", "hire_date", "notes"}
)

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
