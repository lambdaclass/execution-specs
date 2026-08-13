"""
Transaction scoping tests for
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

The state diff an assertion frame reads is scoped to its own
transaction: its prestate is the state at the start of that
transaction, not of the block. And a frame transaction carrying no
assertion frame keeps the EIP-8141 behavior it had before this EIP.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    EIPChecklist,
    Fork,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
)

from tests.amsterdam.eip8141_frame_transactions.helpers import (
    default_frame,
    verify_frame,
)
from tests.amsterdam.eip8141_frame_transactions.spec import Spec as FrameSpec

from .helpers import (
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

SHARED_SLOT = 0x07
"""Storage key both transactions of the mixed block write."""

VALUE_FROM_EARLIER_TRANSACTION = 11
"""Value the earlier transaction of the block leaves in the slot."""

VALUE_FROM_ASSERTION_TRANSACTION = 22
"""Value the assertion transaction's body writes over it."""


def test_frame_transaction_without_post_tx_unaffected(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Run a frame transaction that carries no assertion frame and check
    that nothing about it moved: the body commits, and each frame's
    receipt still reports the EIP-8141 cost of entering its target.

    The negative control for the whole EIP — every rule it adds is
    conditioned on a `POST_TX` frame being present, so a transaction
    without one must be indistinguishable from the same transaction
    before this EIP existed.

    Rules: R-094, R-099.
    """
    sentinel = deploy_sentinel(pre)
    no_op_code = Op.STOP
    no_op = pre.deploy_contract(code=no_op_code)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=no_op, gas_limit=BODY_FRAME_GAS),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                # A frame whose target only stops costs the cold
                # access charged for that target at frame entry.
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS,
                    gas_used=fork.gas_costs().COLD_ACCOUNT_ACCESS
                    + no_op_code.gas_cost(fork),
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER})},
    )


@EIPChecklist.Opcode.Test.ExecutionContext.TxContext()
@pytest.mark.parametrize(
    "expected_value_before",
    [
        pytest.param(
            VALUE_FROM_EARLIER_TRANSACTION, id="prestate_is_transaction_start"
        ),
        pytest.param(0, id="block_start_expectation_bites"),
    ],
)
def test_block_mixing_assertion_and_legacy_transactions(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    expected_value_before: int,
) -> None:
    """
    Put a legacy transaction and an assertion transaction in one block,
    both writing the same slot, and check what the assertion sees: one
    storage change, whose before value is what the legacy transaction
    left rather than what the block started with.

    The twin expecting the block's starting value proves the two are
    distinguishable — the slot holds a different value at each.

    Rule: R-046.
    """
    writer_code = Op.SSTORE(SHARED_SLOT, Op.CALLDATALOAD(0)) + Op.STOP
    writer = pre.deploy_contract(code=writer_code)

    earlier_tx = Transaction(
        sender=pre.fund_eoa(),
        to=writer,
        gas_price=10,
        data=VALUE_FROM_EARLIER_TRANSACTION.to_bytes(32, "big"),
    )

    asserter = pre.deploy_contract(
        code=assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOTS_CHANGED), 1)
        + assert_eq(
            Op.TXDIFF(SHARED_SLOT, writer, Spec.TXDIFF_SLOT_VALUE_BEFORE),
            expected_value_before,
        )
        + assert_eq(
            Op.TXDIFF(SHARED_SLOT, writer, Spec.TXDIFF_SLOT_VALUE_AFTER),
            VALUE_FROM_ASSERTION_TRANSACTION,
        )
        + Op.STOP
    )
    assertion_tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(
                target=writer,
                gas_limit=BODY_FRAME_GAS,
                data=VALUE_FROM_ASSERTION_TRANSACTION.to_bytes(32, "big"),
            ),
            post_tx_frame(target=asserter),
        ],
    )

    passes = expected_value_before == VALUE_FROM_EARLIER_TRANSACTION
    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[earlier_tx, assertion_tx])],
        post={
            writer: Account(
                storage={
                    SHARED_SLOT: (
                        VALUE_FROM_ASSERTION_TRANSACTION
                        if passes
                        # A failed assertion rolls its own body back,
                        # leaving the earlier transaction's write.
                        else VALUE_FROM_EARLIER_TRANSACTION
                    )
                }
            ),
        },
    )
