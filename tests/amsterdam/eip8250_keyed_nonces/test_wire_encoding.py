"""
Decoder-phase wire-format tests for EIP-8250 keyed nonces.

The transactions here are emitted as raw bytes in the `transaction_test`
fixture format, which ships the envelope to a consuming client's decoder
instead of executing a structured transaction. That is the only format in
which the EIP's "Decoders MUST reject" clauses are expressible at all:
every other format drives the transition tool from the structured
transaction fields, so a malformed encoding never reaches a decoder.

Three properties of that format are worth stating here rather than
leaving to be rediscovered.

First, filling this module executes nothing in the reference
implementation. `TransactionTest.generate` discards the transition tool
and copies the declared error straight into the fixture, so a rejection
arm cannot fail at fill time however wrong the decoder is; these arms are
artifacts for clients to consume, not a local gate. The assertions here
that a fill can fail are the accepted arms' checks that the hand-built
canonical envelope equals the framework's own serialization, which do pin
the payload field order.

Second, the decoder reports a malformed envelope as a decoding error,
never as the invalid-frame error that `TYPE_6_INVALID_FRAME_FORMAT`
resolves to -- the reference implementation raises that one only from the
static-constraint validator, which an envelope that failed to decode
never reaches. The arms below therefore declare the `RLP_*` transaction
exceptions that name the specific encoding defect, and the semantic
rejections that really do reach the validator are in `test_validation.py`
where that label is correct.

Third, an overridden envelope is only ever a rejection. The signing
preimage covers the whole payload, `nonce_keys` and `nonce_seq` included,
so replacing the wire bytes under a signature taken over different values
produces an envelope that decodes and is then rejected as an
unauthenticated frame. A rejection arm is indifferent to that -- it never
reaches signature validation -- but an accepted arm would be declaring
valid a transaction that every conformant client rejects, and no local
gate would notice. Accepted arms therefore vary the field through the
structured transaction and let the framework sign what it ships; the
`accept` and `reject` helpers below enforce that split.
"""

from typing import List, Sequence, Tuple

import pytest
from execution_testing import (
    Alloc,
    Bytes,
    EIPChecklist,
    Frame,
    Transaction,
    TransactionException,
    TransactionTestFiller,
)

from .helpers import rlp_encode_bytes, rlp_encode_integer, rlp_encode_list
from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

# Canonical encodings of the donor's own `nonce_keys` and `nonce_seq`.
CANONICAL_NONCE_KEYS = rlp_encode_list([rlp_encode_integer(1)])
CANONICAL_NONCE_SEQ = rlp_encode_integer(0)

# Widest values the EIP's field types can hold: a `nonce_keys` item is read
# as a 256-bit integer, and an ordinary `nonce_seq` stops one below the
# reserved `MAX_NONCE_SEQ`.
MAX_NONCE_KEY = 2**256 - 1
MAX_ORDINARY_NONCE_SEQ = Spec.MAX_NONCE_SEQ - 1


def signed_transaction(
    pre: Alloc,
    *,
    nonce_keys: Sequence[int] = (1,),
    nonce_seq: int = 0,
) -> Transaction:
    """
    Return a signed, minimal keyed-nonce transaction over the given fields.

    The nonce fields are set on the transaction itself so that the framework
    signs the payload it goes on to serialize. Splicing a different value
    into the wire bytes afterwards would leave the signature covering the
    superseded payload, which is fine for an arm that never gets past the
    decoder and fatal for one declared valid.
    """
    tx = Transaction(
        sender=pre.fund_eoa(),
        nonce_keys=list(nonce_keys),
        nonce_seq=nonce_seq,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS,
            )
        ],
    )
    return tx.with_signature_and_sender()


def signed_donor(pre: Alloc) -> Transaction:
    """
    Return the canonically encoded donor every rejection arm mutates.

    The signature is taken over the canonical payload so that each arm below
    differs from the accepted control in exactly the encoding defect it is
    named for, and in nothing else.
    """
    return signed_transaction(pre)


def canonical_nonce_items(tx: Transaction) -> Tuple[bytes, bytes]:
    """Return the canonical RLP items for the transaction's nonce fields."""
    nonce_keys = tx.nonce_keys if tx.nonce_keys is not None else []
    return (
        rlp_encode_list(
            [rlp_encode_integer(int(nonce_key)) for nonce_key in nonce_keys]
        ),
        rlp_encode_integer(int(tx.nonce_seq or 0)),
    )


def encoded_frames(tx: Transaction) -> bytes:
    """Return the RLP encoding of the transaction's `frames` field."""
    frames = tx.frames if tx.frames is not None else []
    return rlp_encode_list(
        [
            rlp_encode_list(
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
            for frame in frames
        ]
    )


def encoded_signatures(tx: Transaction) -> bytes:
    """Return the RLP encoding of the transaction's `signatures` field."""
    signatures = tx.signatures if tx.signatures is not None else []
    return rlp_encode_list(
        [
            rlp_encode_list(
                [
                    rlp_encode_integer(int(signature.scheme)),
                    rlp_encode_bytes(bytes(signature.signer)),
                    rlp_encode_bytes(bytes(signature.msg)),
                    rlp_encode_bytes(bytes(signature.signature)),
                ]
            )
            for signature in signatures
        ]
    )


def encode_transaction(
    tx: Transaction,
    *,
    nonce_keys_item: bytes = CANONICAL_NONCE_KEYS,
    nonce_seq_item: bytes | None = CANONICAL_NONCE_SEQ,
    extra_items: Sequence[bytes] = (),
) -> bytes:
    """
    Assemble the EIP-8250 envelope by hand from already-encoded field items.

    The field order is the one the EIP specifies -- the EIP-8141 payload with
    `nonce` replaced by consecutive `nonce_keys` and `nonce_seq` fields:

        [chain_id, nonce_keys, nonce_seq, sender, frames, signatures,
         max_priority_fee_per_gas, max_fee_per_gas,
         max_fee_per_blob_gas, blob_versioned_hashes]

    Passing `nonce_seq_item=None` drops that field; `extra_items` are spliced
    in directly after it.
    """
    items: List[bytes] = [rlp_encode_integer(int(tx.chain_id))]
    items.append(nonce_keys_item)
    if nonce_seq_item is not None:
        items.append(nonce_seq_item)
    items.extend(extra_items)
    blob_hashes = tx.blob_versioned_hashes or []
    sender = bytes(tx.sender) if tx.sender is not None else b""
    items.extend(
        [
            rlp_encode_bytes(sender),
            encoded_frames(tx),
            encoded_signatures(tx),
            rlp_encode_integer(int(tx.max_priority_fee_per_gas or 0)),
            rlp_encode_integer(int(tx.max_fee_per_gas or 0)),
            rlp_encode_integer(int(tx.max_fee_per_blob_gas or 0)),
            rlp_encode_list([rlp_encode_bytes(bytes(h)) for h in blob_hashes]),
        ]
    )
    return bytes([Spec.FRAME_TX_TYPE]) + rlp_encode_list(items)


def reject(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    donor: Transaction,
    envelope: bytes,
    error: TransactionException,
) -> None:
    """
    Emit `envelope` as a transaction rejected for `error`.

    Asserts first that the mutation actually changed the wire bytes, so an
    arm that silently produced the canonical encoding fails instead of
    reporting a rejection it never tested.
    """
    assert envelope != bytes(donor.rlp()), "mutation did not change the wire"
    donor.error = error
    donor.rlp_override = Bytes(envelope)
    transaction_test(pre=pre, tx=donor)


def accept(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    tx: Transaction,
) -> None:
    """
    Emit `tx` as a transaction a client must accept.

    Asserts that the envelope is the transaction's own serialization rather
    than an override, because an override keeps a signature taken over the
    superseded payload and so ships an envelope no conformant client
    accepts -- a claim of validity that nothing at fill time can refute.
    """
    assert tx.rlp_override is None, (
        "an accepted arm must not override the wire"
    )
    assert tx.error is None, "an accepted arm must not declare an error"
    transaction_test(pre=pre, tx=tx)


@EIPChecklist.TransactionType.Test.Encoding.NewTypes.IncorrectEncoding()
def test_canonical_encoding_accepted(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """
    Accepted control for the whole module.

    This is the accepted control for every rejection in this module. It also
    checks the hand-written encoder against the transaction's own
    serialization, which is what makes each rejection arm below a
    single-defect mutation of a known-good envelope rather than an
    independently assembled guess.
    """
    donor = signed_donor(pre)
    assert encode_transaction(donor) == bytes(donor.rlp())

    accept(transaction_test, pre, donor)


@pytest.mark.parametrize(
    "nonce_keys_item,nonce_seq_item",
    [
        pytest.param(
            rlp_encode_list([rlp_encode_bytes(b"\x00\x01")]),
            CANONICAL_NONCE_SEQ,
            id="nonce_key_leading_zero",
        ),
        pytest.param(
            rlp_encode_list([b"\x00"]),
            CANONICAL_NONCE_SEQ,
            id="nonce_key_zero_as_zero_byte",
        ),
        pytest.param(
            CANONICAL_NONCE_KEYS,
            rlp_encode_bytes(b"\x00\x01"),
            id="nonce_seq_leading_zero",
        ),
        pytest.param(
            CANONICAL_NONCE_KEYS,
            b"\x00",
            id="nonce_seq_zero_as_zero_byte",
        ),
    ],
)
@pytest.mark.exception_test
@EIPChecklist.TransactionType.Test.Encoding.FieldSizes.LeadingZero()
def test_non_canonical_integer_encoding_rejected(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    nonce_keys_item: bytes,
    nonce_seq_item: bytes,
) -> None:
    """
    Pin the canonical-RLP-integer clause.

    The two defect shapes are hand-written from the RLP integer rules the EIP
    invokes. A leading zero byte makes `0x820001` a non-minimal encoding of
    `1`, whose canonical form is the single byte `0x01`; and `0x00` is a
    non-minimal encoding of `0`, whose canonical form is the empty string
    `0x80`. Both shapes are applied once to a `nonce_keys` item and once to
    `nonce_seq`, because the EIP states the rule over both fields.

    The declared exception is the leading-zeros-in-the-nonce-field label:
    EIP-8250 replaces EIP-8141's single `nonce` with these two fields, so
    it is the nonce field of this transaction type whose integer encoding
    is non-minimal.
    """
    donor = signed_donor(pre)
    reject(
        transaction_test,
        pre,
        donor,
        encode_transaction(
            donor,
            nonce_keys_item=nonce_keys_item,
            nonce_seq_item=nonce_seq_item,
        ),
        TransactionException.RLP_LEADING_ZEROS_NONCE,
    )


@pytest.mark.parametrize(
    "nonce_keys,nonce_seq,over_wide_keys_item,over_wide_seq_item",
    [
        pytest.param(
            [MAX_NONCE_KEY],
            0,
            rlp_encode_list([rlp_encode_bytes(b"\x01" + b"\x00" * 32)]),
            None,
            id="nonce_key_thirty_three_bytes",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [MAX_NONCE_KEY],
            0,
            None,
            None,
            id="nonce_key_thirty_two_bytes",
        ),
        pytest.param(
            [1],
            MAX_ORDINARY_NONCE_SEQ,
            None,
            rlp_encode_bytes(b"\x01" + b"\x00" * 8),
            id="nonce_seq_nine_bytes",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [1],
            MAX_ORDINARY_NONCE_SEQ,
            None,
            None,
            id="nonce_seq_eight_bytes",
        ),
    ],
)
@EIPChecklist.TransactionType.Test.Encoding.FieldSizes.RemoveByte()
def test_integer_field_width_bounds(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    nonce_keys: List[int],
    nonce_seq: int,
    over_wide_keys_item: bytes | None,
    over_wide_seq_item: bytes | None,
) -> None:
    """
    Pin the `2**256` and `2**64` field bounds.

    Each rejected width is paired with the same field one byte narrower,
    which the EIP still admits. `0x01` followed by 32 zero bytes is the
    smallest 33-byte canonical integer, exactly `2**256`; removing its
    trailing byte gives a 32-byte integer, and `2**256 - 1` is the widest
    value the EIP accepts. `0x01` followed by 8 zero bytes is likewise
    exactly `2**64`, and `2**64 - 2` is the widest ordinary sequence value
    (`2**64 - 1` is reserved, and is covered by `test_invalid_nonce_fields`).
    Pairing each rejection with its accepted neighbour is what proves the
    boundary sits between them rather than somewhere below both.

    Both members of a pair are built from the same transaction, carrying the
    widest admitted value in the structured field, so the signature covers
    that value and the accepted arm really is acceptable. The rejected arm
    then widens that one field on the wire by a byte and nothing else,
    leaving the two envelopes identical up to the width under test. The
    equality check against the framework's own serialization is what proves
    the wide value reaches the wire at all: were it silently narrowed or
    dropped, the hand-built envelope would not match and the accepted arm
    would fail rather than bracket a boundary it never tested.

    The declared exception is the invalid-nonce-field label rather than a
    nonce overflow: an over-wide field is rejected because it cannot be
    read into the field's type at all, which is a different rule from the
    reserved `2**64 - 1` sequence that decodes and is then rejected by the
    stateful checks.
    """
    tx = signed_transaction(pre, nonce_keys=nonce_keys, nonce_seq=nonce_seq)
    keys_item, seq_item = canonical_nonce_items(tx)
    assert encode_transaction(
        tx, nonce_keys_item=keys_item, nonce_seq_item=seq_item
    ) == bytes(tx.rlp())

    if over_wide_keys_item is None and over_wide_seq_item is None:
        accept(transaction_test, pre, tx)
        return
    reject(
        transaction_test,
        pre,
        tx,
        encode_transaction(
            tx,
            nonce_keys_item=over_wide_keys_item or keys_item,
            nonce_seq_item=over_wide_seq_item or seq_item,
        ),
        TransactionException.RLP_INVALID_NONCE,
    )


@pytest.mark.parametrize(
    "nonce_keys_item",
    [
        pytest.param(
            rlp_encode_bytes(b"\x01"),
            id="byte_string_instead_of_list",
        ),
        pytest.param(
            rlp_encode_list([rlp_encode_list([rlp_encode_integer(1)])]),
            id="nested_list_item",
        ),
    ],
)
@pytest.mark.exception_test
@EIPChecklist.TransactionType.Test.Encoding.NewTypes.IncorrectEncoding()
def test_nonce_keys_wrong_rlp_type_rejected(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    nonce_keys_item: bytes,
) -> None:
    """
    Pin "`nonce_keys` is not an RLP list".

    The first arm re-encodes the same single key as the byte string `0x8101`
    in place of the one-element list `0xc101`; the payload stays ten fields
    long and every other byte is unchanged, so only the RLP type of
    `nonce_keys` distinguishes it from the accepted control. The second arm
    keeps the outer list but wraps the key in an inner list, which is not a
    canonical RLP integer.

    Both defects are visible in an RLP header alone -- `0x81` where `0xc1`
    is required, and `0xc1` where an integer is required -- so the declared
    exception is the invalid-header label.
    """
    donor = signed_donor(pre)
    reject(
        transaction_test,
        pre,
        donor,
        encode_transaction(donor, nonce_keys_item=nonce_keys_item),
        TransactionException.RLP_INVALID_HEADER,
    )


@pytest.mark.parametrize(
    "drop_nonce_seq,extra_items,error",
    [
        pytest.param(
            True,
            (),
            TransactionException.RLP_TOO_FEW_ELEMENTS,
            id="missing_nonce_seq",
        ),
        pytest.param(
            False,
            (rlp_encode_integer(0),),
            TransactionException.RLP_TOO_MANY_ELEMENTS,
            id="extra_field_after_nonce_seq",
        ),
    ],
)
@pytest.mark.exception_test
@EIPChecklist.TransactionType.Test.Encoding.MissingFields()
@EIPChecklist.TransactionType.Test.Encoding.ExtraFields()
def test_payload_field_count_mismatch_rejected(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    drop_nonce_seq: bool,
    extra_items: Sequence[bytes],
    error: TransactionException,
) -> None:
    """
    Pin the ten-field schema.

    The schema has exactly ten fields. Dropping `nonce_seq` leaves nine, and
    splicing one extra item after it leaves eleven; both are hand-counted
    against the field list quoted in `encode_transaction`. Neither can be
    salvaged by re-reading the remaining items in order, because dropping
    `nonce_seq` shifts the 20-byte `sender` into the `nonce_seq` position and
    the extra item shifts `sender` past it. Each arm declares the element
    count it violates, so the two directions are not interchangeable.
    """
    donor = signed_donor(pre)
    reject(
        transaction_test,
        pre,
        donor,
        encode_transaction(
            donor,
            nonce_seq_item=None if drop_nonce_seq else CANONICAL_NONCE_SEQ,
            extra_items=extra_items,
        ),
        error,
    )


@pytest.mark.parametrize(
    "truncate,error",
    [
        pytest.param(True, TransactionException.RLP_ERROR_EOF, id="truncated"),
        pytest.param(
            False, TransactionException.RLP_ERROR_SIZE, id="extra_byte"
        ),
    ],
)
@pytest.mark.exception_test
@EIPChecklist.TransactionType.Test.Encoding.Truncated()
@EIPChecklist.TransactionType.Test.Encoding.ExtraBytes()
def test_envelope_byte_length_mismatch_rejected(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    truncate: bool,
    error: TransactionException,
) -> None:
    """
    Pin exact envelope consumption.

    Removing the final byte leaves the outer list header promising one more
    byte than the payload supplies, and appending a byte leaves one byte
    unconsumed after the outer list ends. Both are derived from the RLP
    length prefix the encoder wrote, not from any client's parser, and each
    arm declares the failure its own direction produces: a stream that ends
    early against a size that does not match the envelope.
    """
    donor = signed_donor(pre)
    canonical = bytes(donor.rlp())
    envelope = canonical[:-1] if truncate else canonical + b"\x00"
    reject(transaction_test, pre, donor, envelope, error)
