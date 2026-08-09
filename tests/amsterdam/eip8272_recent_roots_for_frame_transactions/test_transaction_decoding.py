"""
Static validity tests for
[EIP-8272: Recent Roots for Frame Transactions](https://eips.ethereum.org/EIPS/eip-8272).

The "Static validity" section rejects a frame transaction on the shape of
its bytes alone, before any state is consulted: the payload gains a tenth
element, each element of that element is a three-item list, the two
identifiers are 32 bytes each, and the slot is a canonical integer below
`2**64`. These are decoder rules, so the fixtures here carry the encoded
transaction and the rejection it must produce, and nothing else.

Every malformed shape is built by giving the reference — or the payload
itself — a different serialization, so the bytes on the wire differ from
the well-formed ones in exactly the one place the arm names. Only
rejected shapes appear here: a `transaction_test` states the sender and
the intrinsic gas of an accepted transaction, and neither is defined for
a shape no decoder accepts. The accepted counterpart of each arm is the
well-formed reference every other module in this directory sends.

**Filling proves nothing about these fixtures.** A `transaction_test`
packages the encoded transaction and the declared rejection without
running a transition tool, so the declared exception is asserted against
a client, not against the reference implementation. Each arm's bytes were
therefore checked by hand, by decoding the emitted `txbytes` with the
fork's own `decode_transaction`: all fourteen are rejected, each with a
distinct decoder message naming the defect its arm claims, and the same
transaction with a well-formed reference decodes. Anything observable
*after* decoding belongs in one of the `state_test` modules instead,
where the fill does check it.

The reference-shaped defects declare
`TYPE_6_INVALID_RECENT_ROOT_REFERENCE_FORMAT`, mirroring how EIP-7702
labels a malformed authorization; the payload-level ones declare the
generic `RLP_*` members, which describe an element count or a length that
is not specific to this transaction type.
"""  # noqa: E501

from typing import Any, ClassVar, List, Type, cast

import pytest
from execution_testing import (
    Alloc,
    Bytes,
    EIPChecklist,
    Frame,
    Hash,
    Op,
    RecentRootReference,
    Transaction,
    TransactionException,
    TransactionTestFiller,
)
from execution_testing.base_types import FixedSizeBytes

from .helpers import as_int, verify_and_sender_frames
from .spec import CanonicalVector, Spec, ref_spec_8272

REFERENCE_SPEC_GIT_PATH = ref_spec_8272.git_path
REFERENCE_SPEC_VERSION = ref_spec_8272.version

pytestmark = [
    pytest.mark.valid_from("Bogota"),
    # Every shape below is rejected by the decoder.
    pytest.mark.exception_test,
]

INVALID_REFERENCE_FORMAT = (
    TransactionException.TYPE_6_INVALID_RECENT_ROOT_REFERENCE_FORMAT
)

REFERENCE_FIELDS: List[str] = ["source_id", "slot", "root"]
"""The three items the specification requires, in order."""

WELL_FORMED_SLOT = 8
"""
Slot the well-formed reference names.

Its value is immaterial here — no arm reaches the window check — but it
is small enough to encode in one byte, which keeps the arms that vary
the slot's encoding varying only that.
"""


def without_last_byte(value: Hash) -> int:
    """
    Return the integer a 32-byte value's first 31 bytes denote.

    Re-encoded at 31 bytes this is the value with its last byte removed,
    which is the deformation the shortened-identifier arms apply.
    """
    return as_int(bytes(value)[:-1])


class UndersizedHash(FixedSizeBytes[31]):  # type: ignore
    """A 31-byte identifier, one short of the required length."""

    pass


class OversizedHash(FixedSizeBytes[33]):  # type: ignore
    """A 33-byte identifier, one over the required length."""

    pass


class LeadingZeroSlot(FixedSizeBytes[2]):  # type: ignore
    """
    A slot encoded in two bytes.

    Values below `2**8` therefore carry a leading zero byte, which RLP's
    integer encoding forbids.
    """

    pass


class UndersizedSourceIdReference(RecentRootReference):
    """A reference whose `source_id` is 31 bytes."""

    source_id: UndersizedHash  # type: ignore


class OversizedSourceIdReference(RecentRootReference):
    """A reference whose `source_id` is 33 bytes."""

    source_id: OversizedHash  # type: ignore


class UndersizedRootReference(RecentRootReference):
    """A reference whose `root` is 31 bytes."""

    root: UndersizedHash  # type: ignore


class OversizedRootReference(RecentRootReference):
    """A reference whose `root` is 33 bytes."""

    root: OversizedHash  # type: ignore


class LeadingZeroSlotReference(RecentRootReference):
    """A reference whose `slot` is not a canonical RLP integer."""

    slot: LeadingZeroSlot  # type: ignore


class TwoItemReference(RecentRootReference):
    """A reference that drops `root`, leaving a two-item list."""

    rlp_fields: ClassVar[List[str]] = REFERENCE_FIELDS[:-1]


class FourItemReference(RecentRootReference):
    """A reference that repeats `root`, making a four-item list."""

    rlp_fields: ClassVar[List[str]] = REFERENCE_FIELDS + ["root"]


class ByteStringReference(RecentRootReference):
    """A reference serialized as one byte string rather than as a list."""

    def to_list(self, signing: bool = False) -> List[Any]:
        """Return the three fields concatenated into one byte string."""
        del signing
        # The caller hands whatever this returns straight to the RLP
        # encoder, so a byte string here becomes a string item where the
        # specification requires a list item.
        return cast(
            List[Any],
            bytes(self.source_id)
            + int(self.slot).to_bytes(Spec.SLOT_ENCODING_LENGTH, "big")
            + bytes(self.root),
        )


class ReferenceFreePayloadTransaction(Transaction):
    """A frame transaction whose payload omits the reference list."""

    def get_rlp_fields(self) -> List[str]:
        """Return the payload fields without `recent_root_references`."""
        return [
            field
            for field in super().get_rlp_fields()
            if field != "recent_root_references"
        ]


class DoubledReferencePayloadTransaction(Transaction):
    """A frame transaction whose payload repeats the reference list."""

    def get_rlp_fields(self) -> List[str]:
        """Return the payload fields with one extra element appended."""
        return super().get_rlp_fields() + ["recent_root_references"]


def well_formed_reference() -> RecentRootReference:
    """Return the reference every arm below deforms in one place."""
    return RecentRootReference(
        source_id=CanonicalVector.SOURCE_ID,
        slot=WELL_FORMED_SLOT,
        root=CanonicalVector.ROOT,
    )


@pytest.fixture
def decoder_frames(pre: Alloc) -> List[Frame]:
    """
    Frames of the transaction under test.

    Their content never runs — the transaction is rejected before
    execution — but the payload must still carry a well-formed frame
    list so that the arms differ only in their reference encoding.
    """
    return verify_and_sender_frames(pre.deploy_contract(code=Op.STOP))


@EIPChecklist.TransactionType.Test.Encoding.FieldSizes.RemoveByte()
@EIPChecklist.TransactionType.Test.Encoding.FieldSizes.LeadingZero()
@pytest.mark.parametrize(
    "reference",
    [
        pytest.param(
            UndersizedSourceIdReference(
                source_id=UndersizedHash(
                    without_last_byte(CanonicalVector.SOURCE_ID)
                ),
                slot=WELL_FORMED_SLOT,
                root=CanonicalVector.ROOT,
            ),
            id="source_id_31_bytes",
        ),
        pytest.param(
            OversizedSourceIdReference(
                source_id=OversizedHash(as_int(CanonicalVector.SOURCE_ID)),
                slot=WELL_FORMED_SLOT,
                root=CanonicalVector.ROOT,
            ),
            id="source_id_33_bytes",
        ),
        pytest.param(
            UndersizedRootReference(
                source_id=CanonicalVector.SOURCE_ID,
                slot=WELL_FORMED_SLOT,
                root=UndersizedHash(without_last_byte(CanonicalVector.ROOT)),
            ),
            id="root_31_bytes",
        ),
        pytest.param(
            OversizedRootReference(
                source_id=CanonicalVector.SOURCE_ID,
                slot=WELL_FORMED_SLOT,
                root=OversizedHash(as_int(CanonicalVector.ROOT)),
            ),
            id="root_33_bytes",
        ),
    ],
)
def test_reference_identifier_field_sizes(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    decoder_frames: List[Frame],
    reference: RecentRootReference,
) -> None:
    """
    Reject a reference whose `source_id` or `root` is not 32 bytes.

    Derived from "`source_id` and `root` MUST each be exactly 32 bytes":
    a fixed-length field admits neither a shorter nor a longer string,
    and RLP carries the length explicitly, so both directions are
    distinguishable on the wire from the well-formed encoding. One byte
    either side of the boundary is the whole of the rule.

    Each arm deforms the well-formed value rather than replacing it: the
    short arms drop its last byte, the long arms prepend a zero one. The
    long arms are the ones a decoder reading a length-prefixed field into
    a big-endian integer would still accept, since the numeric value is
    unchanged.
    """
    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=decoder_frames,
        recent_root_references=[reference],
        error=INVALID_REFERENCE_FORMAT,
    )

    transaction_test(pre=pre, tx=tx)


@EIPChecklist.TransactionType.Test.Encoding.FieldSizes.LeadingZero()
@EIPChecklist.TransactionType.Test.OutOfBounds.MaxPlusOne()
@pytest.mark.parametrize(
    "reference",
    [
        pytest.param(
            LeadingZeroSlotReference(
                source_id=CanonicalVector.SOURCE_ID,
                slot=LeadingZeroSlot(WELL_FORMED_SLOT),
                root=CanonicalVector.ROOT,
            ),
            id="slot_leading_zero",
        ),
        pytest.param(
            RecentRootReference(
                source_id=CanonicalVector.SOURCE_ID,
                slot=Spec.MAX_SLOT + 1,
                root=CanonicalVector.ROOT,
            ),
            id="slot_two_to_the_sixty_fourth",
        ),
        pytest.param(
            RecentRootReference(
                source_id=CanonicalVector.SOURCE_ID,
                slot=2**256 - 1,
                root=CanonicalVector.ROOT,
            ),
            id="slot_max_word",
        ),
    ],
)
def test_reference_slot_encoding(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    decoder_frames: List[Frame],
    reference: RecentRootReference,
) -> None:
    """
    Reject a reference whose `slot` is not a canonical integer below
    `2**64`.

    Derived from "`slot` MUST be a canonically encoded integer strictly
    less than `2**64`", which is two rules over one field. The first arm
    encodes the value 8 in two bytes, so the payload is a well-formed RLP
    string but not a well-formed RLP integer; the other two encode
    `2**64` and `2**256 - 1` canonically and are out of range instead.
    Splitting them this way means neither arm can pass for the other's
    reason.
    """
    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=decoder_frames,
        recent_root_references=[reference],
        error=INVALID_REFERENCE_FORMAT,
    )

    transaction_test(pre=pre, tx=tx)


@EIPChecklist.TransactionType.Test.Encoding.NewTypes.IncorrectEncoding()
@pytest.mark.parametrize(
    "reference",
    [
        pytest.param(
            ByteStringReference(
                source_id=CanonicalVector.SOURCE_ID,
                slot=WELL_FORMED_SLOT,
                root=CanonicalVector.ROOT,
            ),
            id="not_a_list",
        ),
        pytest.param(
            TwoItemReference(
                source_id=CanonicalVector.SOURCE_ID,
                slot=WELL_FORMED_SLOT,
                root=CanonicalVector.ROOT,
            ),
            id="two_items",
        ),
        pytest.param(
            FourItemReference(
                source_id=CanonicalVector.SOURCE_ID,
                slot=WELL_FORMED_SLOT,
                root=CanonicalVector.ROOT,
            ),
            id="four_items",
        ),
    ],
)
def test_reference_item_shape(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    decoder_frames: List[Frame],
    reference: RecentRootReference,
) -> None:
    """
    Reject a reference that is not an RLP list of exactly three items.

    Derived from "each element MUST be an RLP list of exactly three
    items". The first arm concatenates the same three values into one
    string, which is the shape a decoder reading fixed offsets rather
    than RLP structure would still accept; the other two keep the list
    but drop and repeat an item, which is the shape a decoder reading
    only the items it needs would still accept. The three fields are
    otherwise the well-formed ones, so no arm can be rejected for its
    contents.
    """
    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=decoder_frames,
        recent_root_references=[reference],
        error=INVALID_REFERENCE_FORMAT,
    )

    transaction_test(pre=pre, tx=tx)


@EIPChecklist.TransactionType.Test.Encoding.MissingFields()
@EIPChecklist.TransactionType.Test.Encoding.ExtraFields()
@pytest.mark.parametrize(
    "transaction_type,error",
    [
        pytest.param(
            ReferenceFreePayloadTransaction,
            TransactionException.RLP_TOO_FEW_ELEMENTS,
            id="nine_element_payload",
        ),
        pytest.param(
            DoubledReferencePayloadTransaction,
            TransactionException.RLP_TOO_MANY_ELEMENTS,
            id="eleven_element_payload",
        ),
    ],
)
def test_payload_element_count(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    decoder_frames: List[Frame],
    transaction_type: Type[Transaction],
    error: TransactionException,
) -> None:
    """
    Reject a frame transaction payload that does not have exactly ten
    elements.

    EIP-8272 appends `recent_root_references` to EIP-8141's nine-element
    payload, so from activation the arity is ten and both neighbours are
    invalid. The nine-element arm is the load-bearing one: it is the
    encoding EIP-8141 alone defines, and a client that kept accepting it
    would silently read pre-fork transactions as valid post-fork ones.
    The eleven-element arm repeats the reference list, so the surplus
    element is well-formed on its own and only the count is wrong.

    The reference the payload does carry is well-formed, and both arms
    are otherwise the same transaction, so neither can be rejected for
    anything the previous tests cover.
    """
    tx = transaction_type(
        sender=pre.fund_eoa(),
        frames=decoder_frames,
        recent_root_references=[well_formed_reference()],
        error=error,
    )

    transaction_test(pre=pre, tx=tx)


@EIPChecklist.TransactionType.Test.Encoding.Truncated()
@EIPChecklist.TransactionType.Test.Encoding.ExtraBytes()
@pytest.mark.parametrize(
    "trailing_bytes,error",
    [
        pytest.param(-1, TransactionException.RLP_ERROR_EOF, id="truncated"),
        pytest.param(1, TransactionException.RLP_ERROR_SIZE, id="extra_bytes"),
    ],
)
def test_transaction_serialization_length(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    decoder_frames: List[Frame],
    trailing_bytes: int,
    error: TransactionException,
) -> None:
    """
    Reject a serialization that does not end where the payload does.

    `recent_root_references` is the last element of the payload, so the
    transaction's last byte is the last byte of the last reference's
    `root`. Dropping it leaves a payload whose declared length outruns
    the bytes that follow, and appending one leaves bytes after the
    payload ends — the two length errors an RLP decoder must raise
    instead of reading a shorter or longer reference.

    Both arms are byte edits of a serialization the previous tests build
    from objects, which is the only way to express a length that no
    well-formed object has. The unedited bytes decode: they are the same
    transaction the arity test sends, with the payload it was signed
    over.
    """
    sender = pre.fund_eoa()
    well_formed = Transaction(
        sender=sender,
        frames=decoder_frames,
        recent_root_references=[well_formed_reference()],
    )
    encoded = bytes(well_formed.with_signature_and_sender().rlp())
    edited = (
        encoded[:trailing_bytes]
        if trailing_bytes < 0
        else encoded + bytes(trailing_bytes)
    )
    assert len(edited) == len(encoded) + trailing_bytes

    tx = Transaction(
        sender=sender,
        frames=decoder_frames,
        recent_root_references=[well_formed_reference()],
        error=error,
    )
    tx.rlp_override = Bytes(edited)

    transaction_test(pre=pre, tx=tx)
