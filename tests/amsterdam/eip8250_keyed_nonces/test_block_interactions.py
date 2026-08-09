"""Block-composition tests for EIP-8250 keyed nonces."""

from typing import Tuple

import pytest
from execution_testing import (
    EOA,
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    EIPChecklist,
    Environment,
    Fork,
    Frame,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
)

from .helpers import gas_components, nonce_calldata
from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

CANARY = 0xC0DE
"""Sentinel proving the non-frame transaction executed to completion."""

PROBE_GAS = 1_000_000
"""Gas budget covering the single `SSTORE` each probe performs."""


GAS_SINK_ALLOCATION = 200_000
"""
Gas handed to the trailing `INVALID` frame of an anchored transaction.

Two jobs. It keeps the standard cost clear of the calldata floor, and it
lifts the block gas limits derived below clear of EIP-7928's block-access-list
item budget, which allows one list item per 2,000 gas of block limit: an
anchor built from the surcharge alone leaves room for fewer items than these
blocks legitimately record, and the block would then be rejected for the
wrong reason.
"""


def anchored_transaction(
    pre: Alloc, fork: Fork, nonce_key: int
) -> Tuple[EOA, Transaction, int]:
    """
    Return a keyed transaction whose gas is exactly its inclusion anchor.

    EIP-8141 admits a frame transaction only if the block can supply
    `max(standard_gas_limit, calldata_floor_gas)`, and EIP-8250 folds
    `nonce_calldata` into both quantities. The transaction built here has a
    single fresh key and two frames, each given exactly the gas it is about
    to spend: the approving `VERIFY` frame gets the one
    `KEYED_NONCE_FIRST_USE_GAS` its single fresh key costs, and the trailing
    frame runs `INVALID`, which consumes its whole allocation. No frame gas
    goes unused, so the transaction's `gas_used` equals `standard_gas_limit`
    rather than sitting somewhere below the anchor, and the caller can treat
    one number as both what the block must be able to supply and what the
    block actually spends.

    The standard-over-floor ordering is asserted rather than assumed, because
    in the floor regime `gas_used` would be clamped to the floor and the two
    numbers would separate.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=pre.deploy_contract(code=Op.INVALID),
                gas_limit=GAS_SINK_ALLOCATION,
            ),
        ],
    )
    standard, floor = gas_components(
        tx.with_signature_and_sender(), fork, nonce_calldata([nonce_key], 0)
    )
    assert standard > floor, "anchor is not the standard gas limit"
    return sender, tx, standard


def anchor_receipt(cumulative_gas_used: int) -> TransactionReceipt:
    """
    Return the receipt an `anchored_transaction` must produce.

    The frame entries are what make the surrounding block-gas arithmetic
    checkable: they state that the approving frame spent exactly its single
    first-use surcharge and the sink frame exactly its whole allocation, so
    the transaction's contribution to the block total is the anchor and not
    some smaller amount that happens to fit.
    """
    return TransactionReceipt(
        cumulative_gas_used=cumulative_gas_used,
        frame_receipts=[
            FrameReceipt(
                status=Spec.STATUS_SUCCESS,
                gas_used=Spec.KEYED_NONCE_FIRST_USE_GAS,
            ),
            FrameReceipt(
                status=Spec.STATUS_FAILURE, gas_used=GAS_SINK_ALLOCATION
            ),
        ],
    )


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


@pytest.mark.parametrize(
    "shortfall,valid",
    [
        pytest.param(0, True, id="block_supplies_the_anchor"),
        pytest.param(
            1,
            False,
            id="block_one_gas_short_of_the_anchor",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@EIPChecklist.TransactionType.Test.BlockInteractions.SingleTx.Valid()
@EIPChecklist.TransactionType.Test.BlockInteractions.SingleTx.Invalid()
def test_sole_transaction_gas_allowance_boundary(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    shortfall: int,
    valid: bool,
) -> None:
    """
    Pin R-021 and R-023 at the block boundary for a block's only
    transaction: the gas the block must supply is the transaction's own
    inclusion anchor, `nonce_calldata` included.

    A frame transaction has no `gas_limit` field, so the checklist's
    "`tx.gas_limit == block.gas_limit`" pair is expressed against the
    quantity that field stands in for -- the derived anchor
    `max(standard_gas_limit, calldata_floor_gas)` that EIP-8141 requires the
    block to be able to supply, and into which R-021 adds
    `nonce_calldata_cost` and R-023 leaves every surrounding definition
    unchanged.

    The two arms are the exact boundary pair. `shortfall` of zero gives the
    block exactly the anchor and the transaction is included; a shortfall of
    one gas leaves the block one short and it is rejected. Nothing else
    differs between them, so the threshold is located at the anchor itself
    rather than anywhere below it -- and because the anchor carries the
    `nonce_calldata` term, a client that priced the nonce fields at zero
    would derive a smaller anchor, fit the transaction into the one-short
    block, and fail the rejecting arm.

    Both expectations are hand-derived from `consume_nonce_set`: the
    included arm writes `nonce_seq + 1 == 1` to the selected slot and spends
    exactly the anchor, and the rejected arm leaves that slot absent, which
    is the zero the post-state asserts.
    """
    nonce_key = 0xB10C
    sender, tx, anchor = anchored_transaction(pre, fork, nonce_key)
    gas_allowance = anchor - shortfall

    state_test(
        env=Environment(gas_limit=gas_allowance),
        pre=pre,
        tx=tx.copy(expected_receipt=anchor_receipt(anchor))
        if valid
        else tx.copy(error=TransactionException.GAS_ALLOWANCE_EXCEEDED),
        post={
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, nonce_key): 1 if valid else 0}
            )
        },
    )


@pytest.mark.parametrize(
    "shortfall,valid",
    [
        pytest.param(0, True, id="block_supplies_both_anchors"),
        pytest.param(
            1,
            False,
            id="block_one_gas_short_of_the_last_anchor",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@EIPChecklist.TransactionType.Test.BlockInteractions.LastTx.Valid()
@EIPChecklist.TransactionType.Test.BlockInteractions.LastTx.Invalid()
def test_last_transaction_gas_allowance_boundary(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
    shortfall: int,
    valid: bool,
) -> None:
    """
    Pin R-021 and R-023 at the block boundary for the last transaction of a
    block: what the closing transaction must fit into is what the block has
    left after the opening one, not the block limit itself.

    Two keyed transactions from two senders share one block. Each is built so
    that its `gas_used` equals its own inclusion anchor -- one fresh key, and
    every frame given exactly the gas it spends -- so the block's gas limit
    can be set to `txs[0].gas_used + txs[1].anchor` and both quantities are
    the EIP's own arithmetic rather than an observation.
    That is the checklist's last-transaction relation stated for a
    transaction type with no gas-limit field, with the anchor standing in for
    it exactly as in `test_sole_transaction_gas_allowance_boundary` above.

    The `shortfall` of one takes that limit down by a single gas, which is
    the checklist's invalid leg: the opening transaction still fits on its
    own -- asserted below, so the rejection cannot be attributed to it -- and
    only the closing transaction no longer does. An implementation that
    charged the closing transaction against the whole block limit instead of
    the remainder would include it in both arms.

    The post-state is hand-derived per arm from `consume_nonce_set`. In the
    accepted arm both selected slots hold `nonce_seq + 1 == 1`. In the
    rejected arm an invalid transaction makes its block invalid and nothing
    of the block is applied, so *both* slots are absent -- including the
    opening transaction's, which would otherwise have been written.
    """
    first_sender, first_tx, first_anchor = anchored_transaction(pre, fork, 3)
    last_sender, last_tx, last_anchor = anchored_transaction(pre, fork, 5)
    gas_limit = first_anchor + last_anchor - shortfall
    assert first_anchor < gas_limit, "opening transaction must fit alone"

    error = TransactionException.GAS_ALLOWANCE_EXCEEDED
    block = Block(
        txs=[
            first_tx.copy(expected_receipt=anchor_receipt(first_anchor))
            if valid
            else first_tx,
            last_tx.copy(
                expected_receipt=anchor_receipt(first_anchor + last_anchor)
            )
            if valid
            else last_tx.copy(error=error),
        ],
        exception=None if valid else error,
    )

    blockchain_test(
        genesis_environment=Environment(gas_limit=gas_limit),
        pre=pre,
        blocks=[block],
        post={
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(first_sender, 3): 1 if valid else 0,
                    Spec.nonce_slot(last_sender, 5): 1 if valid else 0,
                }
            )
        },
    )
