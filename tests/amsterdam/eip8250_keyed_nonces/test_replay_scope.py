"""
Replay-scope tests for EIP-8250 keyed nonces.

The Security Considerations scope replay protection to the triple
`(sender, nonce_keys, nonce_seq)`. Two of that triple's three components are
exercised elsewhere in this directory: `nonce_seq` by the sequence-mismatch
tests and `nonce_keys` by the same-block overlap test. These tests cover the
`sender` component, which no other test varies, and the two lifetime
properties the same section states about the slots themselves -- that a
consumed entry persists and is never deleted, and that a fresh key of a
sender with other consumed keys still reads as an absent slot.
"""

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


def consuming_transaction(
    sender: EOA,
    nonce_keys: list[int],
    nonce_seq: int,
    first_use_count: int,
    *,
    max_priority_fee_per_gas: int = 0,
    error: TransactionException | None = None,
) -> Transaction:
    """
    Return a one-frame transaction that consumes its whole selected key set.

    The frame is given exactly the first-use surcharge its fresh keys attract
    and nothing more, so a receipt that reports any other figure is a failure
    rather than slack absorbed by a generous limit.
    """
    first_use_gas = Spec.KEYED_NONCE_FIRST_USE_GAS * first_use_count
    return Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=nonce_seq,
        max_priority_fee_per_gas=max_priority_fee_per_gas,
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
                    logs=[],
                )
            ],
        )
        if error is None
        else None,
        error=error,
    )


@EIPChecklist.TransactionType.Test.BlockInteractions.SingleTx.Valid()
def test_same_key_different_senders_are_independent(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the sender component of R-088, and R-036 for a key that is fresh for
    one sender while consumed for another.

    Both transactions select the same numeric key at the same sequence zero,
    so the only thing separating their replay domains is the sender. The
    expectation follows from the slot preimage `keccak256(left_pad_32(sender)
    || uint256_to_bytes32(nonce_key))`: the sender occupies the first 32 bytes,
    so two senders derive two slots for one key, and each reads as absent --
    zero -- independently of the other. Both transactions are therefore valid
    at sequence zero and both pay the full first-use surcharge.

    The build-time inequality of the two slots is asserted before use, because
    a preimage that dropped the sender would make the two post-state entries
    collapse into one and the storage expectation would then be satisfied by a
    single write. The second transaction being charged 20,000 rather than 0 is
    the other half of the same argument: had it found the first sender's slot,
    it would have read a non-zero sequence, been invalid at zero, and could
    not have been charged for a first use at all.
    """
    shared_key = 0x5EED
    first_sender = pre.fund_eoa()
    second_sender = pre.fund_eoa()
    first_slot = Spec.nonce_slot(first_sender, shared_key)
    second_slot = Spec.nonce_slot(second_sender, shared_key)
    assert first_slot != second_slot, "slot preimage must include the sender"

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[
                    consuming_transaction(first_sender, [shared_key], 0, 1),
                    consuming_transaction(second_sender, [shared_key], 0, 1),
                ]
            )
        ],
        post={
            first_sender: Account(nonce=0),
            second_sender: Account(nonce=0),
            Spec.NONCE_MANAGER: Account(
                storage={first_slot: 1, second_slot: 1}
            ),
        },
    )


@pytest.mark.exception_test
@EIPChecklist.TransactionType.Test.BlockInteractions.LastTx.Invalid()
def test_consumed_slots_persist_and_replay_tuple_is_rejected(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-087, R-095, and the tuple scoping of R-088 across blocks.

    Three blocks from one sender. The first consumes keys 1 and 2 at sequence
    zero; the second consumes the disjoint key 3, also at sequence zero; the
    third repeats the first block's `(sender, nonce_keys, nonce_seq)` exactly.

    Every expectation is read off `current_nonce_seq` and the consumption
    rule, one key at a time. Keys 1 and 2 hold `nonce_seq + 1 == 1` after the
    first block, so repeating sequence zero against them in the third block is
    one below their current sequence and rejected as too low; an invalid
    transaction makes its block invalid, so nothing of the third block is
    applied. Key 3 was never written, so it still reads absent -- zero -- in
    the second block and is valid there at sequence zero and charged for a
    first use, which is R-087's replay independence of disjoint key sets and
    R-036's absent-reads-as-zero for a sender that already has consumed keys.
    The final storage holds all three keys at 1: none was deleted or reset by
    a later consumption or by the rejected block, which is R-095.

    The replayed transaction carries a priority fee of one wei where the
    original carried none. That leaves the replay tuple identical while giving
    the two transactions different hashes, so the rejection is attributable to
    the tuple rather than to a client recognising a hash it had already seen.
    """
    sender = pre.fund_eoa()
    slots = {key: Spec.nonce_slot(sender, key) for key in (1, 2, 3)}

    original = consuming_transaction(sender, [1, 2], 0, 2)
    disjoint = consuming_transaction(sender, [3], 0, 1)
    replay = consuming_transaction(
        sender,
        [1, 2],
        0,
        2,
        max_priority_fee_per_gas=1,
        error=TransactionException.NONCE_MISMATCH_TOO_LOW,
    )
    assert original.rlp() != replay.rlp(), "replay must not reuse the hash"

    blockchain_test(
        pre=pre,
        blocks=[
            Block(txs=[original]),
            Block(txs=[disjoint]),
            Block(
                txs=[replay],
                exception=TransactionException.NONCE_MISMATCH_TOO_LOW,
            ),
        ],
        post={
            sender: Account(nonce=0),
            Spec.NONCE_MANAGER: Account(
                storage=dict.fromkeys(slots.values(), 1)
            ),
        },
    )
