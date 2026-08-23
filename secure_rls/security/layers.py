"""Which layer refused, and why that is worth showing.

Every refusal in this system is made by a specific layer, and saying which one
turns a generic "I can't do that" into an explanation of the architecture. It
is also the fastest way to check the design is behaving: if an attack that
should die at the database is being turned away at the query gateway, the
gateway is doing work the boundary was supposed to do, and the ablation study
would tell a different story than the demo.

Attribution is explicit, not inferred from the message text. Exceptions carry a
`layer` attribute where the raising site knows better than the type does --
notably the role-based column policy, which is decided by layer 1 (the
principal's role) even though the check runs inside layer 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Layer(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"

    @property
    def title(self) -> str:
        return {
            Layer.L1: "identity & role policy",
            Layer.L2: "tool contract",
            Layer.L3: "query gateway",
            Layer.L4: "database boundary",
            Layer.L5: "output guard",
        }[self]

    @property
    def label(self) -> str:
        return f"{self.value} {self.title}"


@dataclass(frozen=True)
class LayerConfig:
    """Which defensive layers are active for one gateway.

    This exists so the ablation study -- and the lab panel in the UI -- can
    build a *weakened* stack by construction rather than by monkey-patching
    module globals. The harness used to do the latter, and it is worth saying
    why that had to stop:

    * Patching `OutputGuard.check_rows` on the class disables layer 5 for every
      session in the process. Streamlit serves every logged-in browser from one
      process, so one person's experiment would have silently removed a control
      for everyone else signed in.
    * The patch/restore pair is not exception-safe. A raise in between leaves
      the process running with a control switched off and nothing saying so.
    * It is also easy to patch the wrong name: `gateway.py` binds `guard_sql`
      into its own namespace at import, so patching `sql_guard.guard_sql`
      changes nothing the gateway calls. The harness did exactly that and
      reported a confident 0.00% for every arm, including the one built to
      leak (see `tests/test_ablation_harness.py`).

    Configuration passed to a constructor has none of those failure modes: the
    weakened object is a separate object, and the live session keeps its own.

    **There is deliberately no switch for layers 1 and 2.** They are not
    runtime checks that can be skipped -- L1 is what *constructs* the session,
    so "off" is not a weaker system but no session at all, and L2 is the shape
    of the tool schema, so "off" means building a different tool that takes a
    tenant argument. That is writing the vulnerability, not disabling a check,
    and invariant 1 plus the pre-commit hook exist to stop exactly that. The
    absence of those two fields is a statement about the architecture, not an
    omission.
    """

    #: Validate and rewrite model-written SQL on the sqlglot AST.
    l3_query_gateway: bool = True
    #: Materialise the tenant's rows into a private database and detach the
    #: source. Off means the naive build: the full table, an app-code WHERE.
    l4_database_boundary: bool = True
    #: Verify every result against a privileged id set before it is returned.
    l5_output_guard: bool = True

    @property
    def all_on(self) -> bool:
        return self.l3_query_gateway and self.l4_database_boundary and self.l5_output_guard

    def disabled(self) -> list[Layer]:
        """The layers switched off, for labelling a result or an audit entry."""
        out = []
        if not self.l3_query_gateway:
            out.append(Layer.L3)
        if not self.l4_database_boundary:
            out.append(Layer.L4)
        if not self.l5_output_guard:
            out.append(Layer.L5)
        return out

    def describe(self) -> str:
        off = self.disabled()
        if not off:
            return "all layers active"
        return "disabled: " + ", ".join(layer.label for layer in off)


#: The only configuration the application itself ever builds. Anything else is
#: a laboratory object, and every caller that makes one says so out loud.
ALL_LAYERS = LayerConfig()


#: Fallback mapping by exception type, used when a raise site has not tagged the
#: exception itself. Keyed by class name to avoid importing every module here
#: and creating a cycle.
_BY_TYPE: dict[str, Layer] = {
    # Layer 1 -- who is asking, and what their role permits.
    "AuthenticationError": Layer.L1,
    # Layer 2 -- the tool contract. A field the model invented, or a tool that
    # does not exist, is rejected before any data is touched.
    "ValidationError": Layer.L2,
    "CrossTenantRetrieval": Layer.L2,
    # Layer 3 -- the query gateway: shape, allowlists, k-anonymity.
    "SqlRejected": Layer.L3,
    "SpecError": Layer.L3,
    "CohortTooSmall": Layer.L3,
    # Layer 4 -- the boundary itself. The engine refused to prepare or run the
    # statement; nothing above it had to be correct for this to hold.
    "SecurityError": Layer.L4,
    # Layer 5 -- verification after the fact.
    "LeakDetected": Layer.L5,
}


def layer_of(exc: BaseException) -> Layer | None:
    """The layer that refused, or None if the exception is not a policy refusal."""
    tagged = getattr(exc, "layer", None)
    if isinstance(tagged, Layer):
        return tagged
    if isinstance(tagged, str):
        try:
            return Layer(tagged)
        except ValueError:
            return None
    for klass in type(exc).__mro__:
        found = _BY_TYPE.get(klass.__name__)
        if found is not None:
            return found
    return None


def tag(exc: Exception, layer: Layer) -> Exception:
    """Attach an explicit layer to an exception at the point it is raised."""
    exc.layer = layer  # type: ignore[attr-defined]
    return exc
