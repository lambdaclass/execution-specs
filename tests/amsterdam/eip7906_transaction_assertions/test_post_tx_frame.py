"""
`POST_TX` frame execution semantics tests for
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

A `POST_TX` frame executes as a `STATICCALL` from the frame entry
point after the execution body. The tests observe its behavior through
the sentinel contract's post state — a body write that survives a
passing assertion and is rolled back by a failing one — and through
the per-frame receipt statuses and gas.
"""

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Bytecode,
    Conditional,
    EIPChecklist,
    Fork,
    FrameReceipt,
    Hash,
    Op,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
    While,
)
from execution_testing import (
    Macros as Om,
)

from tests.amsterdam.eip8141_frame_transactions.helpers import (
    default_frame,
    verify_frame,
)
from tests.amsterdam.eip8141_frame_transactions.spec import Spec as FrameSpec

from .helpers import (
    ASSERTER_FRAME_GAS,
    BODY_FRAME_GAS,
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


@pytest.mark.parametrize(
    "expected_caller_offset",
    [
        pytest.param(0, id="correct_expectations"),
        pytest.param(1, id="wrong_expectation_bites"),
    ],
)
def test_post_tx_caller_and_origin_are_entry_point(
    state_test: StateTestFiller,
    pre: Alloc,
    expected_caller_offset: int,
) -> None:
    """
    Check that the caller of a `POST_TX` frame is the frame entry point
    at depth 0, and that `ORIGIN` returns the entry point at every call
    depth while a nested call's `CALLER` is the assertion contract.

    The wrong-expectation twin shifts the expected entry point by one,
    proving the conditional-revert oracle bites.

    Rules: R-007, R-112.
    """
    sentinel = deploy_sentinel(pre)
    expected_entry_point = int.from_bytes(FrameSpec.ENTRY_POINT, "big")

    # The helper reverts unless `ORIGIN` is still the entry point at
    # depth 1, then reports its own caller for the asserter to check.
    helper = pre.deploy_contract(
        code=assert_eq(Op.ORIGIN, FrameSpec.ENTRY_POINT)
        + Op.MSTORE(0, Op.CALLER)
        + Op.RETURN(0, 32)
    )

    asserter_code = (
        assert_eq(Op.CALLER, expected_entry_point + expected_caller_offset)
        + assert_eq(Op.ORIGIN, FrameSpec.ENTRY_POINT)
        + assert_eq(Op.CALL(Op.GAS, helper, 0, 0, 0, 0, 32), 1)
        + assert_eq(Op.MLOAD(0), Op.ADDRESS)
        + Op.STOP
    )
    asserter = pre.deploy_contract(code=asserter_code)

    passes = expected_caller_offset == 0
    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS
                    if passes
                    else FrameSpec.STATUS_FAILURE
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: SENTINEL_MARKER if passes else 0}
            ),
        },
    )


@EIPChecklist.Opcode.Test.ExecutionContext.Staticcall()
@pytest.mark.parametrize(
    "write_op,passes",
    [
        pytest.param(Op.SSTORE(0, 1), False, id="sstore"),
        pytest.param(Op.TSTORE(0, 1), False, id="tstore"),
        pytest.param(Op.LOG0(0, 0), False, id="log0"),
        pytest.param(Op.CREATE(0, 0, 0), False, id="create"),
        pytest.param(Op.CREATE2(0, 0, 0, 0), False, id="create2"),
        pytest.param(Op.SELFDESTRUCT(0), False, id="selfdestruct"),
        pytest.param(
            Op.CALL(Op.GAS, Op.ADDRESS, 1, 0, 0, 0, 0),
            False,
            id="call_with_value",
        ),
        pytest.param(
            # Reads and memory writes stay allowed inside the static
            # frame; the frame succeeds and the body survives.
            Op.MSTORE(0, 1)
            + Op.POP(Op.MLOAD(0))
            + Op.POP(Op.SLOAD(0))
            + Op.POP(Op.TLOAD(0)),
            True,
            id="reads_allowed_control",
        ),
    ],
)
def test_post_tx_static_context_bans_state_writes(
    state_test: StateTestFiller,
    pre: Alloc,
    write_op: Bytecode,
    passes: bool,
) -> None:
    """
    Check that a `POST_TX` frame executes as a `STATICCALL`: every
    state-manipulating instruction exceptionally halts the frame,
    consuming its whole gas limit and rolling back the execution body,
    while the transaction stays valid.

    Rule: R-008.
    """
    sentinel = deploy_sentinel(pre)
    asserter = pre.deploy_contract(code=write_op + Op.STOP)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS)
                if passes
                # An exceptional halt consumes the frame's whole gas
                # limit, unlike a revert.
                else FrameReceipt(
                    status=FrameSpec.STATUS_FAILURE,
                    gas_used=ASSERTER_FRAME_GAS,
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: SENTINEL_MARKER if passes else 0}
            ),
        },
    )


def test_post_tx_static_ban_propagates_to_subcalls(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Check that the static restriction propagates through a plain,
    zero-value `CALL` made from a `POST_TX` frame: the callee's
    `SSTORE` fails the inner call while the assertion frame itself
    proceeds, observing the failure.

    The asserter expects the inner call to report failure, so the
    passing sentinel proves the write ban reached depth 1.

    Rule: R-008.
    """
    sentinel = deploy_sentinel(pre)
    writer = pre.deploy_contract(code=Op.SSTORE(0, 1) + Op.STOP)

    # Forward a bounded sub-call gas so the inner halt's consumption
    # cannot exhaust the assertion frame.
    asserter = pre.deploy_contract(
        code=assert_eq(Op.CALL(100_000, writer, 0, 0, 0, 0, 0), 0) + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER}),
            # The inner write never landed.
            writer: Account(storage={0: 0}),
        },
    )


@pytest.mark.parametrize(
    "frame_flags",
    [
        pytest.param(FrameSpec.APPROVE_NONE, id="no_allowed_scope"),
        pytest.param(
            # Even a frame whose flags would allow the payment scope
            # cannot approve: the mode ban fires first.
            FrameSpec.APPROVE_PAYMENT,
            id="allowed_scope_still_forbidden",
        ),
    ],
)
def test_post_tx_approve_forbidden(
    state_test: StateTestFiller,
    pre: Alloc,
    frame_flags: int,
) -> None:
    """
    Check that executing `APPROVE` inside a `POST_TX` frame fails the
    frame: the frame consumes its whole gas limit — the exceptional
    halt reading of "usage is forbidden" — the execution body is
    rolled back, and the transaction stays valid.

    Rule: R-009.
    """
    sentinel = deploy_sentinel(pre)
    approver = pre.deploy_contract(
        code=Op.APPROVE(0, 0, FrameSpec.APPROVE_PAYMENT)
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=approver, flags=frame_flags),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_FAILURE,
                    gas_used=ASSERTER_FRAME_GAS,
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(storage={SENTINEL_SLOT: 0}),
        },
    )


@pytest.mark.parametrize(
    "target_kind",
    [
        pytest.param("funded_eoa", id="funded_eoa"),
        pytest.param("nonexistent", id="nonexistent_account"),
    ],
)
def test_post_tx_empty_code_target_succeeds(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    target_kind: str,
) -> None:
    """
    Check that a `POST_TX` frame targeting an account with no code
    succeeds as a no-op under EIP-8141's default code handling,
    consuming exactly the cold target access charged at frame entry.

    Rule: R-015.
    """
    sentinel = deploy_sentinel(pre)
    target: Address
    if target_kind == "funded_eoa":
        target = pre.fund_eoa(amount=1)
    else:
        target = pre.nonexistent_account()

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=target),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS,
                    gas_used=fork.gas_costs().COLD_ACCOUNT_ACCESS,
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER}),
        },
    )


@EIPChecklist.GasCostChanges.Test.OutOfGas()
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("out_of_gas", id="out_of_gas_consumes_stipend"),
        pytest.param("revert", id="revert_refunds_remainder"),
    ],
)
def test_post_tx_oog_reverts_body(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    failure: str,
) -> None:
    """
    Check that an assertion frame running out of gas is handled as a
    frame failure exactly like a revert — the whole execution body is
    rolled back and the transaction stays valid — while the gas
    observables differ: the exceptional halt consumes the frame's
    whole gas limit, the revert only the gas up to the `REVERT`.

    Rules: R-104, R-014.
    """
    sentinel = deploy_sentinel(pre)

    if failure == "out_of_gas":
        asserter_code: Bytecode = Om.OOG
        # An exceptional halt forfeits the frame's whole gas limit.
        asserter_gas_used = ASSERTER_FRAME_GAS
    else:
        asserter_code = Op.REVERT(0, 0)
        # A revert charges only the cold target access at frame entry
        # plus the executed code.
        asserter_gas_used = fork.gas_costs().COLD_ACCOUNT_ACCESS + Op.REVERT(
            0, 0
        ).gas_cost(fork)
    asserter = pre.deploy_contract(code=asserter_code)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_FAILURE,
                    gas_used=asserter_gas_used,
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(storage={SENTINEL_SLOT: 0}),
        },
    )


def test_assertion_oog_from_event_padding(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Starve a full event-enumeration assertion with adversarial event
    padding: the body emits two hundred events, and the assertion
    frame's budget dies mid-loop — an assertion that has not verified
    the full outcome fails, rolling back the whole body.

    Rule: R-104.
    """
    padding_events = 200
    # Enough for the frame entry and a few dozen loop iterations, but
    # nowhere near the two hundred the enumeration needs.
    starved_frame_gas = 20_000

    sentinel = deploy_sentinel(pre)
    emitter = pre.deploy_contract(code=Op.LOG0(0, 0) * padding_events)

    # Enumerate every event's emitter and data length, then check the
    # count last, so the budget dies inside the loop.
    enumerator = pre.deploy_contract(
        code=Op.MSTORE(0, 0)
        + While(
            body=Op.POP(Op.TXTRACE(Op.MLOAD(0), Spec.TXTRACE_EVENT_ADDRESS))
            + Op.POP(Op.TXTRACE(Op.MLOAD(0), Spec.TXTRACE_EVENT_DATA_LENGTH))
            + Op.MSTORE(0, Op.ADD(Op.MLOAD(0), 1)),
            condition=Op.LT(
                Op.MLOAD(0),
                Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT),
            ),
        )
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=emitter, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=enumerator, gas_limit=starved_frame_gas),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                # The emitter's logs are discarded with the rollback.
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS, logs=[]),
                FrameReceipt(
                    status=FrameSpec.STATUS_FAILURE,
                    gas_used=starved_frame_gas,
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(storage={SENTINEL_SLOT: 0}),
        },
    )


@EIPChecklist.GasCostChanges.Test.OutOfGas()
@pytest.mark.parametrize(
    "budget_delta",
    [
        pytest.param(0, id="exact_budget_succeeds"),
        pytest.param(-1, id="one_below_fails_despite_earlier_surplus"),
    ],
)
def test_post_tx_frame_gas_isolation(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    budget_delta: int,
) -> None:
    """
    Check that unused gas from one assertion frame is not available to
    the next: the first frame leaves nearly its whole large budget
    unused, yet the second frame fails when its own budget is one
    below its exact cost — the cold target access charged at entry.

    Rule: R-110.
    """
    sentinel = deploy_sentinel(pre)
    first_target = pre.deploy_contract(code=Op.STOP)
    second_target = pre.deploy_contract(code=Op.STOP)

    cold_access = fork.gas_costs().COLD_ACCOUNT_ACCESS
    passes = budget_delta == 0

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=first_target),
            post_tx_frame(
                target=second_target,
                gas_limit=cold_access + budget_delta,
            ),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS, gas_used=cold_access
                ),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS, gas_used=cold_access
                )
                if passes
                else FrameReceipt(
                    status=FrameSpec.STATUS_FAILURE,
                    gas_used=cold_access - 1,
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: SENTINEL_MARKER if passes else 0}
            ),
        },
    )


@pytest.mark.parametrize(
    "expected_transient_value,passes",
    [
        pytest.param(0, True, id="transient_storage_cleared"),
        pytest.param(42, False, id="wrong_expectation_bites"),
    ],
)
def test_post_tx_transient_storage_cleared(
    state_test: StateTestFiller,
    pre: Alloc,
    expected_transient_value: int,
    passes: bool,
) -> None:
    """
    Check that transient storage is discarded between frames: the body
    stores 42 in the writer's transient storage, and the assertion
    frame's static callback into the same contract must read 0.

    The wrong-expectation twin expects the stale 42, proving the
    oracle bites.

    Rule: R-114.
    """
    writer = pre.deploy_contract(
        code=Conditional(
            condition=Op.CALLDATASIZE,
            if_true=Op.TSTORE(0, 42)
            + Op.SSTORE(SENTINEL_SLOT, SENTINEL_MARKER)
            + Op.STOP,
            if_false=Op.MSTORE(0, Op.TLOAD(0)) + Op.RETURN(0, 32),
        )
    )
    asserter = pre.deploy_contract(
        code=assert_eq(Op.STATICCALL(Op.GAS, writer, 0, 0, 0, 32), 1)
        + assert_eq(Op.MLOAD(0), expected_transient_value)
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(
                target=writer, data=b"\x01", gas_limit=BODY_FRAME_GAS
            ),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS
                    if passes
                    else FrameSpec.STATUS_FAILURE
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            writer: Account(
                storage={SENTINEL_SLOT: SENTINEL_MARKER if passes else 0}
            ),
        },
    )


@pytest.mark.parametrize(
    "expected_mode,passes",
    [
        pytest.param(Spec.MODE_POST_TX, True, id="correct_expectations"),
        pytest.param(
            Spec.MODE_POST_TX + 1, False, id="wrong_expectation_bites"
        ),
    ],
)
def test_post_tx_introspection_opcodes(
    state_test: StateTestFiller,
    pre: Alloc,
    expected_mode: int,
    passes: bool,
) -> None:
    """
    Check that the EIP-8141 introspection instructions work unchanged
    inside a `POST_TX` frame: `FRAMEPARAM` reports mode 3 for the
    assertion frame and the completed body frame's success status,
    `TXPARAM` reports the frame count and the executing frame's index,
    and `FRAMEDATALOAD` reads the assertion frame's own data.

    Rule: R-116.
    """
    sentinel = deploy_sentinel(pre)
    frame_data = Hash(0x7906ABCD << 192)

    # Frame layout: 0 = VERIFY, 1 = body, 2 = the assertion frame.
    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.FRAMEPARAM(2, FrameSpec.FRAMEPARAM_MODE), expected_mode
        )
        + assert_eq(
            Op.FRAMEPARAM(1, FrameSpec.FRAMEPARAM_STATUS),
            FrameSpec.STATUS_SUCCESS,
        )
        + assert_eq(Op.TXPARAM(FrameSpec.TXPARAM_FRAME_COUNT), 3)
        + assert_eq(Op.TXPARAM(FrameSpec.TXPARAM_FRAME_INDEX), 2)
        + assert_eq(Op.FRAMEPARAM(2, FrameSpec.FRAMEPARAM_DATA_LENGTH), 32)
        + assert_eq(Op.FRAMEDATALOAD(0, 2), frame_data)
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter, data=frame_data),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS
                    if passes
                    else FrameSpec.STATUS_FAILURE
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: SENTINEL_MARKER if passes else 0}
            ),
        },
    )
