"""
Payload-layout tests for EIP-8250 keyed nonces.

The EIP replaces one EIP-8141 payload field with two: "`nonce_keys` followed
by `nonce_seq`. No other payload field is changed", and "the frame layout and
all other EIP-8141 transaction fields are unchanged".

`test_vectors.py` and `test_wire_encoding.py` pin that schema against a
minimal transaction, in which every field after `frames` is empty or zero and
each frame is entirely default. A serializer that emitted the trailing fields
in the wrong order, or a frame whose subfields were transposed, produces the
same bytes for such a transaction as a correct one does, so the claim that
nothing else moved is not yet observable there. These tests populate every
field with a distinct value, so each one's position is carried by its own
bytes, and then move one field to prove the positions are load-bearing.
"""

from typing import List, Sequence

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytes,
    EIPChecklist,
    Environment,
    Frame,
    FrameReceipt,
    FrameSignature,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
    TransactionTestFiller,
)

from .helpers import rlp_encode_bytes, rlp_encode_integer, rlp_encode_list
from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

# Distinct, deliberately wide values for the fields whose position the
# minimal vectors elsewhere cannot pin. `nonce_seq` is eight bytes of
# ascending values, so a field read one position early or late produces
# visibly different bytes rather than another run of zeros.
KEYS = [7, 2**256 - 1]
NONCE_SEQ = 0x0102030405060708
PRIORITY_FEE = 1
MAX_FEE = 7
# `max_fee_per_blob_gas` and `blob_versioned_hashes` are the two fields that
# must stay at their defaults: EIP-8141 requires a zero blob fee on a
# transaction that carries no blob hashes, so no valid transaction can give
# either a distinguishing value. Their positions follow from the other eight
# and from the payload being exactly ten items long.
BLOB_FEE = 0
FRAME_VALUE = 3
FRAME_DATA = b"\x11\x22\x33"
# The frame's own gas limit only has to be distinctive on the wire and ample
# for the three stores the target performs; nothing here measures gas.
FRAME_GAS_LIMIT = 1_000_000
SENTINEL = 0xC0DE

# A value legal in `max_priority_fee_per_gas`, which is a `uint256`, and
# illegal in `nonce_seq`, which is a `uint64`. Swapping the two positions
# therefore fails on the field that moved into the narrower type.
UINT64_OVERFLOW = 2**64


def encoded_frame(frame: Frame) -> bytes:
    """Return the RLP encoding of one frame, in EIP-8141's field order."""
    return rlp_encode_list(
        [
            rlp_encode_integer(int(frame.mode)),
            rlp_encode_integer(int(frame.flags)),
            rlp_encode_bytes(
                b"" if frame.target is None else bytes(frame.target)
            ),
            rlp_encode_integer(int(frame.gas_limit)),
            rlp_encode_integer(int(frame.value)),
            rlp_encode_bytes(bytes(frame.data)),
        ]
    )


def encoded_signature(signature: FrameSignature) -> bytes:
    """Return the RLP encoding of one signature entry."""
    return rlp_encode_list(
        [
            rlp_encode_integer(int(signature.scheme)),
            rlp_encode_bytes(bytes(signature.signer)),
            rlp_encode_bytes(bytes(signature.msg)),
            rlp_encode_bytes(bytes(signature.signature)),
        ]
    )


def payload_items(tx: Transaction) -> List[bytes]:
    """
    Return the ten already-encoded payload items, in the EIP's order.

    Written out as an explicit list rather than delegated to the framework's
    serializer, because the order of this list is the property under test:

        [chain_id, nonce_keys, nonce_seq, sender, frames, signatures,
         max_priority_fee_per_gas, max_fee_per_gas,
         max_fee_per_blob_gas, blob_versioned_hashes]
    """
    assert tx.nonce_keys is not None
    assert tx.nonce_seq is not None
    return [
        rlp_encode_integer(int(tx.chain_id)),
        rlp_encode_list(
            [rlp_encode_integer(int(key)) for key in tx.nonce_keys]
        ),
        rlp_encode_integer(int(tx.nonce_seq)),
        rlp_encode_bytes(b"" if tx.sender is None else bytes(tx.sender)),
        rlp_encode_list([encoded_frame(f) for f in tx.frames or []]),
        rlp_encode_list([encoded_signature(s) for s in tx.signatures or []]),
        rlp_encode_integer(int(tx.max_priority_fee_per_gas or 0)),
        rlp_encode_integer(int(tx.max_fee_per_gas or 0)),
        rlp_encode_integer(int(tx.max_fee_per_blob_gas or 0)),
        rlp_encode_list(
            [
                rlp_encode_bytes(bytes(h))
                for h in tx.blob_versioned_hashes or []
            ]
        ),
    ]


def envelope(items: Sequence[bytes]) -> bytes:
    """Return the typed envelope wrapping already-encoded payload items."""
    return bytes([Spec.FRAME_TX_TYPE]) + rlp_encode_list(list(items))


@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.Encoding.NewTypes.IncorrectEncoding()
def test_full_field_payload_layout_vector(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-014, R-015 and R-016 against a transaction that defaults nothing
    it can legally populate.

    Eight of the ten payload fields, and every subfield of the frames inside
    them, carry a distinct value: two keys of different widths, an eight-byte
    `nonce_seq`, an explicit sender, a `VERIFY` frame whose mode and flags are
    both non-zero followed by a `SENDER` frame carrying a target, a gas limit,
    a value and calldata, a populated signature entry, and two distinct fee
    fields. The two blob fields are the exception, for the reason recorded
    beside `BLOB_FEE`. The expected envelope is then assembled by hand from
    the field list the EIP quotes, in that order, using only RLP primitives,
    and must equal the transaction's own serialization.

    That equality is what makes the claim falsifiable. Two adjacent fields
    holding different values cannot be transposed without changing the bytes,
    so a serializer that emitted `max_fee_per_gas` before
    `max_priority_fee_per_gas`, or a frame whose `value` and `gas_limit` were
    swapped, fails here while satisfying every minimal vector in this
    directory. `nonce_seq` is eight bytes wide and `nonce_keys` is a list, so
    the two fields the EIP introduces are likewise distinguishable from each
    other and from the `sender` that follows them.

    The transaction is executed as well as serialized, so the fields are not
    merely well-placed but accepted: both keys advance from the seeded
    `nonce_seq` to `nonce_seq + 1`, hand-derived from the consumption rule,
    and the `SENDER` frame's target records the calldata and value it
    received. The calldata expectation is the three data bytes right-padded
    to a word, which is what `CALLDATALOAD` at offset zero returns; recording
    it rules out a frame whose `data` never reached the target.

    Both keys are seeded rather than fresh, so the approving frame is due no
    first-use surcharge and is given zero gas.
    """
    sender = pre.fund_eoa()
    recorder = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.CALLDATALOAD(0))
            + Op.SSTORE(1, Op.CALLVALUE)
            + Op.SSTORE(2, SENTINEL)
            + Op.STOP
        )
    )
    slots = {key: Spec.nonce_slot(sender, key) for key in KEYS}
    pre[Spec.NONCE_MANAGER] = Account(
        nonce=1,
        code=Spec.NONCE_MANAGER_CODE,
        storage=dict.fromkeys(slots.values(), NONCE_SEQ),
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=KEYS,
        nonce_seq=NONCE_SEQ,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                target=recorder,
                gas_limit=FRAME_GAS_LIMIT,
                value=FRAME_VALUE,
                data=Bytes(FRAME_DATA),
            ),
        ],
        signatures=[
            FrameSignature(
                scheme=Spec.SCHEME_SECP256K1,
                signer=Bytes(sender),
            ),
        ],
        max_priority_fee_per_gas=PRIORITY_FEE,
        max_fee_per_gas=MAX_FEE,
        max_fee_per_blob_gas=BLOB_FEE,
        blob_versioned_hashes=[],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS, gas_used=0, logs=[]),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )
    signed = tx.with_signature_and_sender()
    assert envelope(payload_items(signed)) == bytes(signed.rlp())

    state_test(
        env=Environment(),
        pre=pre,
        tx=signed,
        post={
            sender: Account(nonce=0),
            recorder: Account(
                balance=FRAME_VALUE,
                storage={
                    0: int.from_bytes(FRAME_DATA.ljust(32, b"\x00"), "big"),
                    1: FRAME_VALUE,
                    2: SENTINEL,
                },
            ),
            Spec.NONCE_MANAGER: Account(
                storage=dict.fromkeys(slots.values(), NONCE_SEQ + 1)
            ),
        },
    )


@pytest.mark.exception_test
@EIPChecklist.TransactionType.Test.Encoding.NewTypes.IncorrectEncoding()
def test_nonce_field_position_swap_rejected(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the positional half of R-014: `nonce_seq` sits at payload index 2,
    where EIP-8141's `nonce` used to be, and not wherever a field of the same
    RLP shape happens to fit.

    Ten integer-or-list fields give a decoder no way to notice that two of
    them were transposed, unless one position admits a value the other
    forbids. `nonce_seq` is a `uint64` and `max_priority_fee_per_gas` is a
    `uint256`, so `2**64` is exactly such a value: legal at index 6 and
    rejected at index 2 by the EIP's own `nonce_seq >= 2**64` clause. The
    envelope below is the canonical one with those two items exchanged, which
    leaves the field count, every other field and the total shape untouched;
    the only thing a client can be failing on is which index it read the
    sequence from.

    The declared exception is the invalid-nonce-field label rather than a
    sequence overflow, matching the over-wide field arms in
    `test_wire_encoding.py`: the value cannot be read into `nonce_seq`'s type
    at all, which is a decoding failure and not the stateful rejection the
    reserved `MAX_NONCE_SEQ` receives.

    This arm is emitted as a rejection only. The signing preimage covers the
    whole payload, so an envelope whose fields have been exchanged after
    signing carries a signature over the unexchanged payload -- fine for an
    arm that never gets past the decoder, and a false claim of validity for
    one that does.
    """
    tx = Transaction(
        sender=pre.fund_eoa(),
        nonce_keys=[1],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS,
            )
        ],
        max_priority_fee_per_gas=UINT64_OVERFLOW,
    ).with_signature_and_sender()

    items = payload_items(tx)
    assert envelope(items) == bytes(tx.rlp())
    assert int(items[2][0]) == 0x80, (
        "canonical sequence zero is the empty item"
    )

    swapped = list(items)
    swapped[2], swapped[6] = swapped[6], swapped[2]
    mutated = envelope(swapped)
    assert mutated != bytes(tx.rlp()), "swap did not change the wire"

    tx.error = TransactionException.RLP_INVALID_NONCE
    tx.rlp_override = Bytes(mutated)
    transaction_test(pre=pre, tx=tx)
