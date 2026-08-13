"""
Pre-activation tests for the opcode slots of
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

Before the EIP activates, the opcode bytes chosen for TXTRACE,
EVENTDATACOPY, and TXDIFF are undefined instructions: executing them
in any context is an invalid-opcode exceptional halt, so no EIP-7906
semantics or pricing can leak across the fork boundary.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    EIPChecklist,
    StateTestFiller,
    Transaction,
)

from .helpers import HALT_PROBE_POST, OPCODE_USES, deploy_halt_probe
from .spec import ref_spec_7906

REFERENCE_SPEC_GIT_PATH = ref_spec_7906.git_path
REFERENCE_SPEC_VERSION = ref_spec_7906.version

# These arms pin the world before the pseudo-fork gating EIP-7906:
# Amsterdam is the last fork where the chosen opcode slots must still
# be undefined.
pytestmark = [
    pytest.mark.valid_from("Amsterdam"),
    pytest.mark.valid_until("Amsterdam"),
]


@EIPChecklist.Opcode.Test.ForkTransition.Invalid()
@EIPChecklist.GasCostChanges.Test.ForkTransition.Before()
@pytest.mark.parametrize("opcode_name", sorted(OPCODE_USES))
def test_new_opcode_bytes_undefined_before_activation(
    state_test: StateTestFiller,
    pre: Alloc,
    opcode_name: str,
) -> None:
    """
    Execute each chosen opcode byte from a legacy transaction before
    the EIP activates: the byte is an undefined instruction, so the
    call containing it exceptionally halts consuming all its gas —
    there is no EIP-7906 behavior or pricing to misapply before the
    fork.

    Rules: R-097, R-094.
    """
    sender = pre.fund_eoa()
    probe = deploy_halt_probe(pre, OPCODE_USES[opcode_name])

    tx = Transaction(sender=sender, to=probe, gas_price=10)

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=1),
            probe: HALT_PROBE_POST,
        },
    )
