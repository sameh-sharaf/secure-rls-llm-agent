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
