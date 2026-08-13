"""Block-order validity tests for EIP-8250 keyed nonces."""

import pytest
from execution_testing import (
    EOA,
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    Frame,
    FrameReceipt,
    Transaction,
    TransactionException,
    TransactionReceipt,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")


def approval_transaction(
    sender: EOA,
    nonce_keys: list[int],
    nonce_seq: int,
    first_use_count: int,
    error: TransactionException | None = None,
) -> Transaction:
    """Build a one-frame transaction that consumes its selected keys."""
    first_use_gas = Spec.KEYED_NONCE_FIRST_USE_GAS * first_use_count
    return Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=nonce_seq,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=first_use_gas,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=first_use_gas,
                )
            ],
        )
        if error is None
        else None,
        error=error,
    )


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            "stale_overlap",
            id="stale_overlap_invalid",
            marks=pytest.mark.exception_test,
        ),
        pytest.param("disjoint", id="disjoint_valid"),
    ],
)
def test_block_order_overlap_and_disjoint_domains(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    scenario: str,
) -> None:
    """
    Pin per-position sequence evaluation.

    In the overlap case A writes keys 1/2 to one, making B's stale sequence
    zero invalid at its exact block position; an invalid block commits none of
    A. In the disjoint control A and C independently write keys 1/2 and 3 to
    one. Both outcomes are hand-derived by comparing the selected domains.
    """
    sender = pre.fund_eoa()
    first = approval_transaction(sender, [1, 2], 0, 2)
    if scenario == "stale_overlap":
        error = TransactionException.NONCE_MISMATCH_TOO_LOW
        transactions = [
            first,
            approval_transaction(sender, [2, 3], 0, 1, error=error),
        ]
        block = Block(txs=transactions, exception=error)
        post = {
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, key): 0 for key in (1, 2, 3)}
            )
        }
    else:
        transactions = [
            first,
            approval_transaction(sender, [3], 0, 1),
        ]
        block = Block(txs=transactions)
        post = {
            sender: Account(nonce=0),
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, key): 1 for key in (1, 2, 3)}
            ),
        }

    blockchain_test(
        pre=pre,
        blocks=[block],
        post=post,
    )


def test_intra_block_chained_same_key(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the valid direction of the same rule.

    `test_block_order_overlap_and_disjoint_domains` shows that a *stale*
    sequence on an overlapping key set is invalid; the same rule also says
    overlapping transactions are valid when each one's `nonce_seq` equals the
    current sequence of every selected key *at its own block position*. Two
    transactions from one sender select the identical key set here, so nothing
    is disjoint and only per-position evaluation can accept them.

    Both expectations are re-derived from the spec pseudocode rather than
    observed. At the first transaction's position the slot is absent, so
    `current_nonce_seq` reads zero for an absent slot and `nonce_seq == 0`
    satisfies the equality; `consume_nonce_set` stores `0 + 1 == 1`. At the
    second
    transaction's position that same slot now reads one, so only
    `nonce_seq == 1` is valid there and the slot ends at `1 + 1 == 2`.

    The two frame gas limits are the discriminating part. `first_use_count`
    counts slots whose pre-read value is zero, so it is one for the
    first transaction and zero for the second: the first approving frame is
    given exactly `KEYED_NONCE_FIRST_USE_GAS` and the second exactly zero. An
    implementation that re-charged the surcharge for an already-written slot,
    or that evaluated both transactions against the block's opening state,
    fails here rather than producing the same post-state by another route.
    """
    sender = pre.fund_eoa()
    nonce_key = 7

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[
                    approval_transaction(sender, [nonce_key], 0, 1),
                    approval_transaction(sender, [nonce_key], 1, 0),
                ]
            )
        ],
        post={
            # The keyed domain never advances the legacy account nonce.
            sender: Account(nonce=0),
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, nonce_key): 2}
            ),
        },
    )
