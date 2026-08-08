"""Block-composition tests for EIP-8250 keyed nonces."""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    EIPChecklist,
    Frame,
    FrameReceipt,
    Op,
    Transaction,
    TransactionReceipt,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

CANARY = 0xC0DE
"""Sentinel proving the non-frame transaction executed to completion."""

PROBE_GAS = 1_000_000
"""Gas budget covering the single `SSTORE` each probe performs."""


@EIPChecklist.TransactionType.Test.BlockInteractions.MixedTxs()
def test_mixed_transaction_types_share_block(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-085.

    Legacy account objects and non-frame transaction types are unchanged by
    this EIP, so a plain transaction, a keyed frame transaction, and a
    key-zero frame transaction must coexist in one block without
    interference. Each sender is distinct, so the three expectations are
    independent and hand-derived per domain: the plain sender's account nonce
    advances to one and its target records the canary, the keyed sender's
    account nonce stays at zero while its manager slot advances to
    `nonce_seq + 1 == 1`, and the key-zero sender's account nonce advances to
    one while writing no manager slot at all.
    """
    plain_sender = pre.fund_eoa()
    keyed_sender = pre.fund_eoa()
    legacy_sender = pre.fund_eoa()
    probe = pre.deploy_contract(code=Op.SSTORE(0, CANARY) + Op.STOP)
    keyed_key = 5

    plain_tx = Transaction(
        sender=plain_sender,
        to=probe,
        gas_limit=PROBE_GAS,
        expected_receipt=TransactionReceipt(status=1),
    )
    keyed_tx = Transaction(
        sender=keyed_sender,
        nonce_keys=[keyed_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=keyed_sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=Spec.KEYED_NONCE_FIRST_USE_GAS,
                )
            ],
        ),
    )
    legacy_tx = Transaction(
        sender=legacy_sender,
        nonce_keys=[0],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=legacy_sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS, gas_used=0)
            ],
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[plain_tx, keyed_tx, legacy_tx])],
        post={
            plain_sender: Account(nonce=1),
            keyed_sender: Account(nonce=0),
            legacy_sender: Account(nonce=1),
            probe: Account(storage={0: CANARY}),
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(keyed_sender, keyed_key): 1,
                    # The key-zero domain never touches manager storage.
                    Spec.nonce_slot(legacy_sender, 0): 0,
                }
            ),
        },
    )


@EIPChecklist.TransactionType.Test.SenderAccount.Nonce()
def test_keyed_validity_independent_of_legacy_nonce_advance(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-092, R-093, and R-094.

    A non-zero-key frame transaction does not advance the sender's legacy
    account nonce, and conversely a transaction that advances that nonce
    first must not invalidate it — so the "send another transaction with the
    same legacy nonce" cancellation strategy fails against keyed
    transactions. One sender issues a plain transaction and then a keyed
    frame transaction in the same block. The expected final account nonce is
    hand-derived as exactly one: the plain transaction contributes the single
    increment and keyed payment approval contributes none. A client that
    still incremented on keyed approval would leave two, and one that
    required `nonce_seq` to track the account nonce would reject the second
    transaction outright.
    """
    sender = pre.fund_eoa()
    probe = pre.deploy_contract(code=Op.SSTORE(0, CANARY) + Op.STOP)
    nonce_key = 11

    cancellation_attempt = Transaction(
        sender=sender,
        to=probe,
        gas_limit=PROBE_GAS,
        expected_receipt=TransactionReceipt(status=1),
    )
    keyed_tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=Spec.KEYED_NONCE_FIRST_USE_GAS,
                )
            ],
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[cancellation_attempt, keyed_tx])],
        post={
            # One increment from the plain transaction, none from approval.
            sender: Account(nonce=1),
            probe: Account(storage={0: CANARY}),
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, nonce_key): 1}
            ),
        },
    )
