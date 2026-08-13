"""
Context gating tests for the TXTRACE, TXDIFF, and EVENTDATACOPY
opcodes of [EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

The three state-diff opcodes are valid only inside the call subtree of
a `POST_TX` frame; executing them in any other context — other
transaction types, other frame modes, or initcode — exceptionally
halts.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytecode,
    EIPChecklist,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    compute_create_address,
)
from execution_testing import (
    Macros as Om,
)

from tests.amsterdam.eip8141_frame_transactions.helpers import (
    default_frame,
    sender_frame,
    verify_frame,
)
from tests.amsterdam.eip8141_frame_transactions.spec import Spec as FrameSpec

from .helpers import (
    BODY_FRAME_GAS,
    FORWARDED_GAS,
    HALT_PROBE_POST,
    OPCODE_USES,
    SENTINEL_MARKER,
    SENTINEL_SLOT,
    SLOT_CALL_RESULT_PLUS_ONE,
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


@EIPChecklist.Opcode.Test.ExceptionalAbort()
@pytest.mark.parametrize("opcode_name", sorted(OPCODE_USES))
@pytest.mark.parametrize(
    "context",
    [
        pytest.param("legacy_tx", id="legacy_tx"),
        pytest.param("fee_market_tx", id="fee_market_tx"),
        pytest.param("default_frame", id="default_frame"),
        pytest.param("sender_frame", id="sender_frame"),
    ],
)
def test_new_opcodes_halt_outside_post_tx_frames(
    state_test: StateTestFiller,
    pre: Alloc,
    opcode_name: str,
    context: str,
) -> None:
    """
    Execute each state-diff opcode outside a `POST_TX` frame — from
    legacy and fee-market transactions and from `DEFAULT` and `SENDER`
    frames — and check that the call containing it exceptionally
    halts, consuming all its forwarded gas, without invalidating the
    surrounding transaction.

    Rules: R-018, R-019, R-095.
    """
    sender = pre.fund_eoa()
    probe = deploy_halt_probe(pre, OPCODE_USES[opcode_name])

    if context == "legacy_tx":
        tx = Transaction(sender=sender, to=probe, gas_price=10)
    elif context == "fee_market_tx":
        tx = Transaction(
            sender=sender,
            to=probe,
            max_fee_per_gas=10,
            max_priority_fee_per_gas=1,
        )
    elif context == "default_frame":
        tx = Transaction(
            sender=sender,
            frames=[
                verify_frame(),
                default_frame(target=probe, gas_limit=BODY_FRAME_GAS),
            ],
        )
    else:
        tx = Transaction(
            sender=sender,
            frames=[
                verify_frame(),
                sender_frame(target=probe, gas_limit=BODY_FRAME_GAS),
            ],
        )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=1),
            probe: HALT_PROBE_POST,
        },
    )


@pytest.mark.exception_test
@pytest.mark.parametrize("opcode_name", sorted(OPCODE_USES))
def test_new_opcodes_in_verify_frame_invalidate_transaction(
    state_test: StateTestFiller,
    pre: Alloc,
    opcode_name: str,
) -> None:
    """
    Execute each state-diff opcode as the code of a `VERIFY` frame's
    target: the exceptional halt fails the frame, and a failing
    `VERIFY` frame — unlike a failing body frame — invalidates the
    whole transaction.

    Rule: R-019.
    """
    sender = pre.fund_eoa()
    helper = pre.deploy_contract(code=OPCODE_USES[opcode_name])

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            verify_frame(
                flags=FrameSpec.APPROVE_NONE,
                target=helper,
            ),
        ],
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=0)},
    )


@EIPChecklist.Opcode.Test.ExecutionContext.Call()
@EIPChecklist.Opcode.Test.ExecutionContext.Delegatecall()
@EIPChecklist.Opcode.Test.ExecutionContext.Callcode()
@EIPChecklist.Opcode.Test.ExecutionContext.Staticcall()
@pytest.mark.with_all_call_opcodes
@pytest.mark.parametrize(
    "inside_post_tx",
    [
        pytest.param(True, id="inside_post_tx_subtree"),
        pytest.param(False, id="outside_post_tx_subtree"),
    ],
)
def test_opcode_gate_keys_on_enclosing_frame_mode(
    state_test: StateTestFiller,
    pre: Alloc,
    call_opcode: Op,
    inside_post_tx: bool,
) -> None:
    """
    Reach identical helper code through every call kind at depth 1:
    inside a `POST_TX` frame's call subtree the state-diff opcode
    executes — including under the context swaps of `DELEGATECALL` and
    `CALLCODE` — while the same path from a `DEFAULT` frame halts. The
    gate keys on the enclosing top-level frame's mode, not on the call
    kind or depth.

    Rule: R-020.
    """
    sender = pre.fund_eoa()
    # The helper's answer is offset by a constant so the expected
    # value is non-zero: a zeroed return buffer cannot pass.
    helper = pre.deploy_contract(
        code=Op.MSTORE(0, Op.ADD(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT), 7))
        + Op.RETURN(0, 32)
    )

    # Bound the forwarded gas: in the halting arms the helper burns
    # everything forwarded, and the probe still needs room for its
    # recording write afterwards.
    if call_opcode in (Op.CALL, Op.CALLCODE):
        call_code = call_opcode(FORWARDED_GAS, helper, 0, 0, 0, 0, 32)
    else:
        call_code = call_opcode(FORWARDED_GAS, helper, 0, 0, 0, 32)

    if inside_post_tx:
        sentinel = deploy_sentinel(pre)
        asserter = pre.deploy_contract(
            code=assert_eq(call_code, 1) + assert_eq(Op.MLOAD(0), 7) + Op.STOP
        )
        tx = Transaction(
            sender=sender,
            frames=[
                verify_frame(),
                default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
                post_tx_frame(target=asserter),
            ],
        )
        post = {
            sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER}),
        }
    else:
        probe = pre.deploy_contract(
            code=Op.SSTORE(SLOT_CALL_RESULT_PLUS_ONE, Op.ADD(call_code, 1))
            + Op.STOP
        )
        tx = Transaction(
            sender=sender,
            frames=[
                verify_frame(),
                default_frame(target=probe, gas_limit=BODY_FRAME_GAS),
            ],
        )
        post = {
            probe: Account(storage={SLOT_CALL_RESULT_PLUS_ONE: 1}),
        }

    state_test(pre=pre, tx=tx, post=post)


@pytest.mark.parametrize(
    "inside_post_tx",
    [
        pytest.param(True, id="inside_post_tx_subtree"),
        pytest.param(False, id="outside_post_tx_subtree"),
    ],
)
def test_opcode_gate_reaches_call_depth_two(
    state_test: StateTestFiller,
    pre: Alloc,
    inside_post_tx: bool,
) -> None:
    """
    Reach the state-diff opcode through a two-level call chain: the
    gate still admits it inside the `POST_TX` subtree and still halts
    it from a `DEFAULT` frame's subtree.

    Rule: R-020.
    """
    sender = pre.fund_eoa()
    helper = pre.deploy_contract(
        code=Op.MSTORE(0, Op.ADD(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT), 7))
        + Op.RETURN(0, 32)
    )
    # The middle contract forwards the helper's answer, failing loudly
    # if the inner call failed. The inner forward is bounded so a
    # halting helper cannot starve the callers above it.
    middle = pre.deploy_contract(
        code=assert_eq(Op.CALL(FORWARDED_GAS, helper, 0, 0, 0, 0, 32), 1)
        + Op.RETURN(0, 32)
    )

    if inside_post_tx:
        sentinel = deploy_sentinel(pre)
        asserter = pre.deploy_contract(
            code=assert_eq(Op.CALL(Op.GAS, middle, 0, 0, 0, 0, 32), 1)
            + assert_eq(Op.MLOAD(0), 7)
            + Op.STOP
        )
        tx = Transaction(
            sender=sender,
            frames=[
                verify_frame(),
                default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
                post_tx_frame(target=asserter),
            ],
        )
        post = {
            sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER}),
        }
    else:
        probe = pre.deploy_contract(
            code=Op.SSTORE(
                SLOT_CALL_RESULT_PLUS_ONE,
                Op.ADD(Op.CALL(Op.GAS, middle, 0, 0, 0, 0, 32), 1),
            )
            + Op.STOP
        )
        tx = Transaction(
            sender=sender,
            frames=[
                verify_frame(),
                default_frame(target=probe, gas_limit=BODY_FRAME_GAS),
            ],
        )
        post = {
            probe: Account(storage={SLOT_CALL_RESULT_PLUS_ONE: 1}),
        }

    state_test(pre=pre, tx=tx, post=post)


@EIPChecklist.Opcode.Test.ExecutionContext.Initcode.Behavior()
@pytest.mark.parametrize(
    "creation_context",
    [
        pytest.param("create_opcode", id="create_from_default_frame"),
        pytest.param("creation_tx", id="legacy_creation_tx"),
    ],
)
def test_initcode_cannot_reach_new_opcodes(
    state_test: StateTestFiller,
    pre: Alloc,
    creation_context: str,
) -> None:
    """
    Run TXTRACE inside initcode reached from non-`POST_TX` contexts —
    a `CREATE` in a `DEFAULT` frame and a legacy contract-creation
    transaction: the initcode exceptionally halts and nothing is
    deployed. Inside a `POST_TX` frame `CREATE` itself already halts
    under the static restriction, before any initcode could run.

    Rule: R-019.
    """
    sender = pre.fund_eoa()
    initcode = Op.POP(Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED)) + Op.STOP

    if creation_context == "create_opcode":
        factory = pre.deploy_contract(
            code=Om.MSTORE(bytes(initcode), 0)
            + Op.SSTORE(
                SLOT_CALL_RESULT_PLUS_ONE,
                Op.ADD(Op.CREATE(0, 0, len(bytes(initcode))), 1),
            )
            + Op.STOP
        )
        child = compute_create_address(address=factory, nonce=1)
        tx = Transaction(
            sender=sender,
            frames=[
                verify_frame(),
                default_frame(target=factory, gas_limit=BODY_FRAME_GAS),
            ],
        )
        post = {
            # CREATE pushed 0: the halted initcode deployed nothing.
            factory: Account(storage={SLOT_CALL_RESULT_PLUS_ONE: 1}, nonce=2),
            child: Account.NONEXISTENT,
        }
    else:
        child = compute_create_address(address=sender, nonce=0)
        tx = Transaction(
            sender=sender,
            to=None,
            gas_price=10,
            data=initcode,
            gas_limit=200_000,
        )
        post = {
            sender: Account(nonce=1),
            child: Account.NONEXISTENT,
        }

    state_test(pre=pre, tx=tx, post=post)


@EIPChecklist.Opcode.Test.StackUnderflow()
@pytest.mark.parametrize(
    "opcode_name,underflow_code",
    [
        pytest.param("txtrace", Op.PUSH0 + Op.TXTRACE, id="txtrace_one_item"),
        pytest.param(
            "txdiff", Op.PUSH0 + Op.PUSH0 + Op.TXDIFF, id="txdiff_two_items"
        ),
        pytest.param(
            "eventdatacopy",
            Op.PUSH0 + Op.PUSH0 + Op.PUSH0 + Op.EVENTDATACOPY,
            id="eventdatacopy_three_items",
        ),
    ],
)
def test_new_opcodes_stack_underflow(
    state_test: StateTestFiller,
    pre: Alloc,
    opcode_name: str,
    underflow_code: Bytecode,
) -> None:
    """
    Run each state-diff opcode with one stack item fewer than its
    arity, inside a `POST_TX` frame — the only context where the
    opcodes are live: the frame exceptionally halts, consuming its
    whole gas limit and rolling back the body.

    Rules: R-016, R-053, R-090.
    """
    sentinel = deploy_sentinel(pre)
    asserter = pre.deploy_contract(code=underflow_code)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(storage={SENTINEL_SLOT: 0}),
        },
    )


@EIPChecklist.Opcode.Test.ReturnData.Buffer.Current()
def test_new_opcodes_preserve_return_buffer(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Check that none of the three state-diff opcodes disturbs the
    return data buffer: after a call populates it, running TXTRACE,
    TXDIFF, and a zero-length EVENTDATACOPY leaves both its size and
    its contents intact.
    """
    buffer_value = 0xBEEF
    sentinel = deploy_sentinel(pre)
    emitter = pre.deploy_contract(code=Op.LOG0(0, 0) + Op.STOP)
    returner = pre.deploy_contract(
        code=Op.MSTORE(0, buffer_value) + Op.RETURN(0, 32)
    )
    asserter = pre.deploy_contract(
        code=assert_eq(Op.CALL(Op.GAS, returner, 0, 0, 0, 0, 0), 1)
        + Op.POP(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT))
        + Op.POP(Op.TXDIFF(0, Op.ADDRESS, Spec.TXDIFF_ACCOUNT_CHANGE_FLAGS))
        + Op.EVENTDATACOPY(0, 0, 0, 0)
        + assert_eq(Op.RETURNDATASIZE, 32)
        + Op.RETURNDATACOPY(0x20, 0, 32)
        + assert_eq(Op.MLOAD(0x20), buffer_value)
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=emitter),
            post_tx_frame(target=asserter),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER}),
        },
    )
