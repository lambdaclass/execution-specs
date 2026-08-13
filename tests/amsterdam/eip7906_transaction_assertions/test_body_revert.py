"""
Whole-body revert tests for
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

A failing `POST_TX` frame reverts the entire execution body back to
the end of the validation prefix — overriding atomic-batch unrolling —
while the transaction stays valid, is included with a receipt, and
keeps the payer charged for the gas consumed.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Fork,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
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
    SENTINEL_MARKER,
    SENTINEL_SLOT,
    deploy_sentinel,
    post_tx_frame,
)
from .spec import ref_spec_7906

REFERENCE_SPEC_GIT_PATH = ref_spec_7906.git_path
REFERENCE_SPEC_VERSION = ref_spec_7906.version

# EIP-7906 extends EIP-8141, which is slated for the fork after
# Amsterdam, so fixtures are labeled with the pseudo `Bogota` fork.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

WRITER_SLOT = 0x01
"""Storage slot the writer contract modifies."""

WRITER_PRESTATE_VALUE = 0x0101
"""The writer slot's pre-transaction value."""

WRITER_BODY_VALUE = 0x1111
"""The value the execution body writes into the writer slot."""


def test_assertion_revert_rolls_back_entire_body(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Roll back every kind of execution-body side effect with a single
    unconditionally failing assertion: a value transfer, a storage
    write over a non-zero prestate, a contract creation, and an event
    — while the transaction stays valid and included, with the
    validation prefix's sender nonce increment permanently committed.

    Rules: R-010, R-012, R-013.
    """
    alice = pre.fund_eoa()
    bob = pre.fund_eoa(amount=1)
    transfer_value = 10**15

    writer = pre.deploy_contract(
        code=Op.SSTORE(WRITER_SLOT, WRITER_BODY_VALUE) + Op.STOP,
        storage={WRITER_SLOT: WRITER_PRESTATE_VALUE},
    )
    factory = pre.deploy_contract(code=Op.POP(Op.CREATE(0, 0, 0)) + Op.STOP)
    # Deployed contracts start at nonce 1, so the body-created child
    # is the factory's nonce-1 creation.
    child = compute_create_address(address=factory, nonce=1)
    emitter = pre.deploy_contract(code=Op.LOG1(0, 0, 0x790601) + Op.STOP)
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            sender_frame(target=bob, value=transfer_value),
            default_frame(target=writer, gas_limit=BODY_FRAME_GAS),
            default_frame(target=factory, gas_limit=BODY_FRAME_GAS),
            default_frame(target=emitter),
            post_tx_frame(target=reverter),
        ],
        expected_receipt=TransactionReceipt(
            payer=alice,
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                # Unrolled body frames keep their status and gas, but
                # their logs are discarded with their state changes.
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS, logs=[]),
                FrameReceipt(status=FrameSpec.STATUS_FAILURE),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            # The prefix's nonce increment is permanently committed.
            alice: Account(nonce=1),
            bob: Account(balance=1),
            writer: Account(storage={WRITER_SLOT: WRITER_PRESTATE_VALUE}),
            factory: Account(nonce=1),
            child: Account.NONEXISTENT,
        },
    )


def test_assertion_revert_preserves_validation_prefix(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Check that state changes made within the validation prefix survive
    a failing assertion: a `DEFAULT` frame that runs before any frame
    approves payment is part of the prefix, so its storage write stays
    committed while the post-approval body write is rolled back.

    Rules: R-013, R-118.
    """
    alice = pre.fund_eoa()
    prefix_writer = pre.deploy_contract(
        code=Op.SSTORE(SENTINEL_SLOT, SENTINEL_MARKER) + Op.STOP
    )
    sentinel = deploy_sentinel(pre)
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))

    tx = Transaction(
        sender=alice,
        frames=[
            # Runs before payment approval: part of the prefix.
            default_frame(target=prefix_writer, gas_limit=BODY_FRAME_GAS),
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=reverter),
        ],
        expected_receipt=TransactionReceipt(
            payer=alice,
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_FAILURE),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            alice: Account(nonce=1),
            # The prefix write survives the failed assertion; the
            # body write does not.
            prefix_writer: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER}),
            sentinel: Account(storage={SENTINEL_SLOT: 0}),
        },
    )


def test_assertion_revert_overrides_atomic_batch(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Roll back a successfully completed atomic batch with a later
    failing assertion: a `POST_TX` revert unwinds the whole execution
    body, not merely an atomic batch, so the batch's committed writes
    are discarded along with the rest of the body.

    Rules: R-011, R-101.
    """
    alice = pre.fund_eoa()
    writers = [
        pre.deploy_contract(
            code=Op.SSTORE(WRITER_SLOT, WRITER_BODY_VALUE) + Op.STOP
        )
        for _ in range(3)
    ]
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            # A two-frame batch terminated by the third, unflagged
            # writer frame; the batch completes successfully.
            default_frame(
                target=writers[0],
                flags=FrameSpec.ATOMIC_BATCH_FLAG,
                gas_limit=BODY_FRAME_GAS,
            ),
            default_frame(
                target=writers[1],
                flags=FrameSpec.ATOMIC_BATCH_FLAG,
                gas_limit=BODY_FRAME_GAS,
            ),
            default_frame(target=writers[2], gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=reverter),
        ],
        expected_receipt=TransactionReceipt(
            payer=alice,
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_FAILURE),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            alice: Account(nonce=1),
            **{
                writer: Account(storage={WRITER_SLOT: 0}) for writer in writers
            },
        },
    )


@pytest.mark.parametrize(
    "failing_position",
    [
        pytest.param(0, id="first_assertion_fails"),
        pytest.param(1, id="second_assertion_fails"),
        pytest.param(2, id="third_assertion_fails"),
    ],
)
def test_assertion_failure_at_each_suffix_position(
    state_test: StateTestFiller,
    pre: Alloc,
    failing_position: int,
) -> None:
    """
    Fail exactly one assertion of a three-frame suffix at each
    position: any single assertion module independently rolls back the
    whole body, and the frames after the failing one never execute —
    they are skipped with no gas consumed, their budgets refunded.

    Rules: R-010, R-102.
    """
    alice = pre.fund_eoa()
    sentinel = deploy_sentinel(pre)
    passer = pre.deploy_contract(code=Op.STOP)
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))

    suffix_receipts = []
    for position in range(3):
        if position < failing_position:
            suffix_receipts.append(
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS)
            )
        elif position == failing_position:
            suffix_receipts.append(
                FrameReceipt(status=FrameSpec.STATUS_FAILURE)
            )
        else:
            suffix_receipts.append(
                FrameReceipt(status=FrameSpec.STATUS_SKIPPED, gas_used=0)
            )

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            *[
                post_tx_frame(
                    target=reverter if position == failing_position else passer
                )
                for position in range(3)
            ],
        ],
        expected_receipt=TransactionReceipt(
            payer=alice,
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                *suffix_receipts,
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            alice: Account(nonce=1),
            sentinel: Account(storage={SENTINEL_SLOT: 0}),
        },
    )


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("revert", id="revert_charges_up_to_revert"),
        pytest.param("out_of_gas", id="out_of_gas_charges_full_stipend"),
    ],
)
def test_assertion_revert_gas_charged_up_to_revert(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    failure: str,
) -> None:
    """
    Pin the exact total gas a failed-assertion transaction charges:
    the intrinsic cost plus each frame's consumption, with the failing
    frame contributing only the gas up to its `REVERT` — or its whole
    stipend in the out-of-gas twin.

    A contract sender keeps the intrinsic cost content-independent:
    no signature entries and no frame data, so the intrinsic reduces
    to the base and per-frame constants.

    Rules: R-014, R-103.
    """
    sender = pre.deploy_contract(
        code=Op.APPROVE(0, 0, FrameSpec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )
    gas_costs = fork.gas_costs()

    # The costing twin carries the gas metadata of the deployed code;
    # both produce identical bytes.
    sentinel_code = (
        Op.SSTORE(
            SENTINEL_SLOT,
            SENTINEL_MARKER,
            key_warm=False,
            original_value=0,
            current_value=0,
            new_value=SENTINEL_MARKER,
        )
        + Op.STOP
    )
    sentinel = pre.deploy_contract(code=sentinel_code)

    starved_gas = 30_000
    if failure == "revert":
        asserter_code = Op.REVERT(0, 0)
        asserter_frame = post_tx_frame(
            target=pre.deploy_contract(code=asserter_code)
        )
        # A revert charges the cold target access plus the executed
        # code, refunding the frame's remainder.
        asserter_gas_used = (
            gas_costs.COLD_ACCOUNT_ACCESS + asserter_code.gas_cost(fork)
        )
    else:
        asserter_frame = post_tx_frame(
            target=pre.deploy_contract(code=Om.OOG),
            gas_limit=starved_gas,
        )
        # An exceptional halt forfeits the frame's whole gas limit.
        asserter_gas_used = starved_gas

    # The sender contract's `VERIFY` frame: warm target access (the
    # sender seeds the warm journal) plus its three `PUSH1`s; the
    # `APPROVE` itself is free.
    verify_gas_used = gas_costs.WARM_ACCESS + (Op.PUSH1(0) * 3).gas_cost(fork)
    body_gas_used = gas_costs.COLD_ACCOUNT_ACCESS + sentinel_code.gas_cost(
        fork
    )
    intrinsic = (
        FrameSpec.FRAME_TX_INTRINSIC_COST
        + 3 * FrameSpec.FRAME_TX_PER_FRAME_COST
    )

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            asserter_frame,
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            cumulative_gas_used=intrinsic
            + verify_gas_used
            + body_gas_used
            + asserter_gas_used,
            frame_receipts=[
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS,
                    gas_used=verify_gas_used,
                ),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS,
                    gas_used=body_gas_used,
                ),
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
            # The prefix's nonce increment survives; the body write
            # does not.
            sender: Account(nonce=2),
            sentinel: Account(storage={SENTINEL_SLOT: 0}),
        },
    )


def test_assertion_pass_keeps_body_and_receipt(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Check the success path: a passing assertion keeps every body write,
    and its frame receipt carries success status, exactly the cold
    target access as gas used, and an empty log list — a static frame
    can emit no logs.

    Rules: R-012, R-111, R-103.
    """
    alice = pre.fund_eoa()
    sentinel = deploy_sentinel(pre)
    passer = pre.deploy_contract(code=Op.STOP)

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=passer),
        ],
        expected_receipt=TransactionReceipt(
            payer=alice,
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS,
                    gas_used=fork.gas_costs().COLD_ACCOUNT_ACCESS,
                    logs=[],
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            alice: Account(nonce=1),
            sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER}),
        },
    )
