"""
Stack and return-data tests for the state-diff opcodes of
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

Each opcode takes its operands from the top of the stack and leaves
everything below untouched, whatever the stack's height; and none of
them writes to the return data buffer, at its own call context or at
its caller's.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Conditional,
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
    FORWARDED_GAS,
    SENTINEL_MARKER,
    SENTINEL_SLOT,
    assert_eq,
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

BURIED_MARKER = 0xDEAD
"""Value left underneath the operands to detect a deeper stack read."""


@EIPChecklist.Opcode.Test.StackComplexOperations.StackHeights.Zero()
@EIPChecklist.Opcode.Test.StackComplexOperations.StackHeights.Odd()
@EIPChecklist.Opcode.Test.StackComplexOperations.StackHeights.Even()
@pytest.mark.parametrize(
    "buried_items,halts",
    [
        pytest.param(0, True, id="zero_height_stack"),
        pytest.param(1, False, id="odd_height_stack"),
        pytest.param(2, False, id="even_height_stack"),
    ],
)
def test_new_opcodes_at_various_stack_heights(
    state_test: StateTestFiller,
    pre: Alloc,
    buried_items: int,
    halts: bool,
) -> None:
    """
    Run TXTRACE with a marker buried under its operands and check that
    it answers from its own two operands and leaves the marker in
    place, at both an odd and an even stack height — and that the same
    opcode on a stack holding nothing at all halts.

    The buried marker is read back after the call, so an opcode
    reaching one item too deep is caught rather than merely producing
    an unchecked stack.

    Rules: R-016, R-017.
    """
    sentinel = deploy_sentinel(pre)
    emitter = pre.deploy_contract(code=Op.LOG0(0, 0) + Op.STOP)

    # TXTRACE pops two items, so the two arms differ in whether the
    # marker sits at an odd or an even height beneath them.
    if halts:
        probe = Op.TXTRACE + Op.STOP
    else:
        probe = (
            Op.PUSH2[BURIED_MARKER] * buried_items
            + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT), 1)
            # `EQ` with one operand compares against what the opcode
            # left below its own: the marker must still be on top.
            + Conditional(
                condition=Op.EQ(BURIED_MARKER, unchecked=True),
                if_false=Op.REVERT(0, 0),
            )
            + Op.STOP
        )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=emitter, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=pre.deploy_contract(code=probe)),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: 0 if halts else SENTINEL_MARKER}
            ),
        },
    )


@EIPChecklist.Opcode.Test.ReturnData.Buffer.Parent()
def test_new_opcodes_preserve_caller_return_buffer(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Execute the three state-diff opcodes inside a sub-call and check
    the return data buffer the caller sees afterwards: it holds exactly
    what that sub-call returned, so none of the opcodes wrote to the
    buffer at its caller's context.

    Rules: R-016, R-053, R-090.
    """
    returned_value = 0xF00D
    sentinel = deploy_sentinel(pre)
    emitter = pre.deploy_contract(code=Op.LOG0(0, 0) + Op.STOP)
    inner = pre.deploy_contract(
        code=Op.POP(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT))
        + Op.POP(Op.TXDIFF(0, Op.ADDRESS, Spec.TXDIFF_ACCOUNT_CHANGE_FLAGS))
        + Op.EVENTDATACOPY(0, 0, 0, 0)
        + Op.MSTORE(0, returned_value)
        + Op.RETURN(0, 32)
    )
    asserter = pre.deploy_contract(
        code=assert_eq(Op.CALL(FORWARDED_GAS, inner, 0, 0, 0, 0, 0), 1)
        + assert_eq(Op.RETURNDATASIZE, 32)
        + Op.RETURNDATACOPY(0x20, 0, 32)
        + assert_eq(Op.MLOAD(0x20), returned_value)
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=emitter, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER})},
    )
