"""
EIP-7702 set-code composition tests for
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

The state-diff opcode gate keys on the enclosing top-level frame's
mode, so code reached through an EIP-7702 delegation behaves exactly
like directly deployed code: live inside a `POST_TX` frame, halting
everywhere else.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    EIPChecklist,
    Op,
    StateTestFiller,
    Transaction,
)

from tests.amsterdam.eip8141_frame_transactions.helpers import (
    default_frame,
    verify_frame,
)

from .helpers import (
    BODY_FRAME_GAS,
    HALT_PROBE_POST,
    SENTINEL_MARKER,
    SENTINEL_SLOT,
    assert_eq,
    deploy_halt_probe,
    deploy_sentinel,
    post_tx_frame,
)
from .spec import Spec, ref_spec_7906

REFERENCE_SPEC_GIT_PATH = ref_spec_7906.git_path
REFERENCE_SPEC_VERSION = ref_spec_7906.version

# EIP-7906 extends EIP-8141, which is slated for the fork after
# Amsterdam, so fixtures are labeled with the pseudo `Bogota` fork.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")


@EIPChecklist.Opcode.Test.ExecutionContext.SetCode()
@pytest.mark.parametrize(
    "inside_post_tx",
    [
        pytest.param(True, id="delegated_target_of_post_tx_frame"),
        pytest.param(False, id="delegated_target_of_legacy_tx"),
    ],
)
def test_new_opcodes_via_set_code_delegation(
    state_test: StateTestFiller,
    pre: Alloc,
    inside_post_tx: bool,
) -> None:
    """
    Execute TXTRACE through an EIP-7702 delegated EOA: targeted by a
    `POST_TX` frame the delegated code runs and asserts correctly,
    while the same delegated account reached from a legacy transaction
    halts — the gate follows the enclosing frame mode through the
    delegation.

    Rule: R-020.
    """
    if inside_post_tx:
        sentinel = deploy_sentinel(pre)
        asserter_code = (
            assert_eq(Op.ADD(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT), 7), 7)
            + Op.STOP
        )
        delegated = pre.fund_eoa(
            amount=0,
            delegation=pre.deploy_contract(code=asserter_code),
        )
        tx = Transaction(
            sender=pre.fund_eoa(),
            frames=[
                verify_frame(),
                default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
                post_tx_frame(target=delegated),
            ],
        )
        post = {
            sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER}),
        }
    else:
        halting_code = (
            Op.POP(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT)) + Op.STOP
        )
        delegated = pre.fund_eoa(
            amount=0,
            delegation=pre.deploy_contract(code=halting_code),
        )
        probe = deploy_halt_probe(pre, delegated)
        tx = Transaction(sender=pre.fund_eoa(), to=probe, gas_price=10)
        post = {
            probe: HALT_PROBE_POST,
        }

    state_test(pre=pre, tx=tx, post=post)
