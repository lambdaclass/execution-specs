"""Block-order validity tests for EIP-8250 keyed nonces."""

import pytest
from execution_testing import (
    EOA,
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    EIPChecklist,
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
@EIPChecklist.TransactionType.Test.BlockInteractions.LastTx.Valid()
@EIPChecklist.TransactionType.Test.BlockInteractions.LastTx.Invalid()
def test_block_order_overlap_and_disjoint_domains(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    scenario: str,
) -> None:
    """
    Pin R-069, R-143, and R-159.

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
