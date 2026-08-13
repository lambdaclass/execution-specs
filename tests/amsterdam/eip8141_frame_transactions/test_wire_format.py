"""
Wire-format decoder tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

Every case ships raw transaction bytes that violate exactly one clause
of the payload encoding `[chain_id, nonce, sender, frames, signatures,
max_priority_fee_per_gas, max_fee_per_gas, max_fee_per_blob_gas,
blob_versioned_hashes]` and expects decoding to reject them. Each
malformed shape is expressed on the transaction object itself —
through field-type overrides, field-list overrides, or the envelope
prefix — so the shipped bytes are the object's own serialization and
arms that must differ do differ. No donor transaction is byte-patched
except the envelope truncation and trailing-byte arms, where the
corrupted byte stream is itself the shape under test.

The accept side of every clause is pinned by the state fixtures of
this directory, which embed the canonical serialization of the same
objects in their blocks. The expected exception class follows the
merged set-code transaction precedent of reporting the type-specific
format error for every encoding malformation.
"""

from typing import Any, Dict, List, Sequence

import pytest
from execution_testing import (
    Alloc,
    Bytes,
    EIPChecklist,
    Frame,
    FrameSignature,
    Hash,
    Transaction,
    TransactionException,
    TransactionTestFiller,
)
from execution_testing.base_types import FixedSizeBytes, HexNumber

from .helpers import AMPLE_FRAME_GAS, verify_frame
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

# EIP-8141 is slated for the fork after Amsterdam, so fixtures are
# labeled with the pseudo `Bogota` fork (Amsterdam + EIP-8141), even
# though the spec prototypes the EIP inside the Amsterdam fork module.
# Fill these tests with `--fork Bogota`.
pytestmark = [
    pytest.mark.valid_from("Bogota"),
    pytest.mark.exception_test,
]

FORMAT_ERROR = TransactionException.TYPE_6_INVALID_FRAME_FORMAT


class OversizedWord(FixedSizeBytes[33]):  # type: ignore
    """33-byte integer encoding, one byte above the 32-byte bound."""

    pass


class OversizedNonce(FixedSizeBytes[9]):  # type: ignore
    """9-byte integer encoding, one byte above the 8-byte bound."""

    pass


class PaddedZero(FixedSizeBytes[1]):  # type: ignore
    """
    Zero encoded as a single `0x00` byte.

    The canonical RLP integer encoding of zero is the empty string, so
    this adds a leading zero byte.
    """

    pass


class UndersizedAddressBytes(FixedSizeBytes[19]):  # type: ignore
    """19-byte address encoding, one byte below the address length."""

    pass


class OversizedAddressBytes(FixedSizeBytes[21]):  # type: ignore
    """21-byte address encoding, one byte above the address length."""

    pass


class UndersizedHash(FixedSizeBytes[31]):  # type: ignore
    """31-byte hash encoding, one byte below the hash length."""

    pass


class OversizedHash(FixedSizeBytes[33]):  # type: ignore
    """33-byte hash encoding, one byte above the hash length."""

    pass


def canonical_transaction_kwargs(pre: Alloc) -> Dict[str, Any]:
    """
    Return constructor arguments for the canonical valid frame
    transaction every arm mutates in exactly one clause.

    The single `VERIFY` frame approves execution and payment against
    the sender's default code, so the unmutated transaction is valid
    in full, and the framework signs the default secp256k1 signature
    entry over each mutant's own serialization.
    """
    return dict(
        sender=pre.fund_eoa(),
        nonce=0,
        frames=[verify_frame()],
        error=FORMAT_ERROR,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_field_zero",
        "extra_field_nonzero",
        "missing_field",
        "truncated",
        "extra_bytes",
        "wrong_type_byte",
    ],
)
@EIPChecklist.TransactionType.Test.Encoding.ExtraFields()
@EIPChecklist.TransactionType.Test.Encoding.MissingFields()
@EIPChecklist.TransactionType.Test.Encoding.Truncated()
@EIPChecklist.TransactionType.Test.Encoding.ExtraBytes()
def test_payload_structure(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    mutation: str,
) -> None:
    """
    Reject payloads that are not the exact nine-element list.

    Pins R-024 (payload field list), and through the extra-element
    arms R-225/R-226 (the payload admits no tenth list such as an
    authorization or access list). The wrong-type-byte arm pins the
    wire side of R-001: a frame transaction body under any other
    envelope byte is not a frame transaction.
    """
    kwargs = canonical_transaction_kwargs(pre)
    tx: Transaction
    if mutation in ("extra_field_zero", "extra_field_nonzero"):

        class ExtraPayloadFieldTransaction(Transaction):
            """Frame transaction serializing a tenth payload element."""

            extra_element: HexNumber

            def get_rlp_fields(self) -> List[str]:
                """Append the extra element to the payload field list."""
                return super().get_rlp_fields() + ["extra_element"]

        extra_element = 1 if mutation == "extra_field_nonzero" else 0
        tx = ExtraPayloadFieldTransaction(
            **kwargs, extra_element=extra_element
        )
    elif mutation == "missing_field":

        class MissingPayloadFieldTransaction(Transaction):
            """Frame transaction dropping the last payload element."""

            def get_rlp_fields(self) -> List[str]:
                """Drop the blob hashes element from the field list."""
                return [
                    field
                    for field in super().get_rlp_fields()
                    if field != "blob_versioned_hashes"
                ]

        tx = MissingPayloadFieldTransaction(**kwargs)
    elif mutation == "truncated":
        tx = Transaction(**kwargs).with_signature_and_sender()
        tx.rlp_override = Bytes(tx.rlp()[:-1])
    elif mutation == "extra_bytes":
        tx = Transaction(**kwargs).with_signature_and_sender()
        tx.rlp_override = Bytes(tx.rlp() + b"\x00")
    else:
        assert mutation == "wrong_type_byte"

        class WrongTypePrefixTransaction(Transaction):
            """Frame transaction body under an undefined type byte."""

            def get_rlp_prefix(self) -> bytes:
                """Return an undefined transaction type byte."""
                return b"\x07"

        kwargs["error"] = TransactionException.TYPE_NOT_SUPPORTED
        tx = WrongTypePrefixTransaction(**kwargs)

    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize(
    "mutation",
    [
        "chain_id_out_of_range",
        "nonce_out_of_range",
        "nonce_leading_zero",
        "priority_fee_out_of_range",
        "max_fee_out_of_range",
        "blob_fee_out_of_range",
        "sender_undersized",
        "sender_oversized",
        "blob_hash_undersized",
        "blob_hash_oversized",
    ],
)
@EIPChecklist.TransactionType.Test.Encoding.FieldSizes.LeadingZero()
@EIPChecklist.TransactionType.Test.Encoding.FieldSizes.RemoveByte()
def test_payload_field_encoding(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    mutation: str,
) -> None:
    """
    Reject payload field encodings that violate one field bound.

    The out-of-range arms encode the first integer that no canonical
    encoding can represent within each field's bound — `2**256` for
    chain id and the three fee fields (R-036, R-039, R-040, R-041)
    and `2**64` for the nonce (R-038) — so the bound is only reachable
    at the wire level. The leading-zero arm violates the canonical
    integer form without changing the value. The sender and blob hash
    arms change only the byte length of one fixed-length field (R-044,
    R-045), keeping the blob hash version byte valid.
    """
    kwargs = canonical_transaction_kwargs(pre)
    tx: Transaction
    if mutation == "chain_id_out_of_range":

        class OversizedChainIdTransaction(Transaction):
            """Frame transaction with a 33-byte chain id encoding."""

            chain_id: OversizedWord  # type: ignore

        tx = OversizedChainIdTransaction(**kwargs, chain_id=2**256)
    elif mutation == "nonce_out_of_range":

        class OversizedNonceTransaction(Transaction):
            """Frame transaction with a 9-byte nonce encoding."""

            nonce: OversizedNonce  # type: ignore

        kwargs["nonce"] = 2**64
        tx = OversizedNonceTransaction(**kwargs)
    elif mutation == "nonce_leading_zero":

        class PaddedNonceTransaction(Transaction):
            """Frame transaction with a zero-padded nonce encoding."""

            nonce: PaddedZero  # type: ignore

        tx = PaddedNonceTransaction(**kwargs)
    elif mutation == "priority_fee_out_of_range":

        class OversizedPriorityFeeTransaction(Transaction):
            """Frame transaction with a 33-byte priority fee."""

            max_priority_fee_per_gas: OversizedWord  # type: ignore

        tx = OversizedPriorityFeeTransaction(
            **kwargs, max_priority_fee_per_gas=2**256
        )
    elif mutation == "max_fee_out_of_range":

        class OversizedMaxFeeTransaction(Transaction):
            """Frame transaction with a 33-byte max fee encoding."""

            max_fee_per_gas: OversizedWord  # type: ignore

        tx = OversizedMaxFeeTransaction(**kwargs, max_fee_per_gas=2**256)
    elif mutation == "blob_fee_out_of_range":

        class OversizedBlobFeeTransaction(Transaction):
            """Frame transaction with a 33-byte blob fee encoding."""

            max_fee_per_blob_gas: OversizedWord  # type: ignore

        # One valid versioned hash keeps a nonzero blob fee otherwise
        # legal, isolating the encoding bound as the violated clause.
        tx = OversizedBlobFeeTransaction(
            **kwargs,
            max_fee_per_blob_gas=2**256,
            blob_versioned_hashes=[Hash(b"\x01" + (0).to_bytes(31, "big"))],
        )
    elif mutation in ("sender_undersized", "sender_oversized"):
        if mutation == "sender_undersized":

            class UndersizedSenderTransaction(Transaction):
                """Frame transaction with a 19-byte sender encoding."""

                sender: UndersizedAddressBytes  # type: ignore

            tx = UndersizedSenderTransaction(
                **{**kwargs, "sender": bytes(kwargs["sender"])[:19]},
                # The sender bytes carry no key, so no signature entry
                # can be signed; an empty signature list is valid.
                signatures=[],
            )
        else:

            class OversizedSenderTransaction(Transaction):
                """Frame transaction with a 21-byte sender encoding."""

                sender: OversizedAddressBytes  # type: ignore

            tx = OversizedSenderTransaction(
                **{**kwargs, "sender": bytes(kwargs["sender"]) + b"\x00"},
                signatures=[],
            )
    else:
        if mutation == "blob_hash_undersized":

            class UndersizedBlobHashTransaction(Transaction):
                """Frame transaction with a 31-byte blob hash."""

                blob_versioned_hashes: Sequence[  # type: ignore
                    UndersizedHash
                ]

            tx = UndersizedBlobHashTransaction(
                **kwargs,
                max_fee_per_blob_gas=1,
                blob_versioned_hashes=[
                    UndersizedHash(b"\x01" + (0).to_bytes(30, "big"))
                ],
            )
        else:
            assert mutation == "blob_hash_oversized"

            class OversizedBlobHashTransaction(Transaction):
                """Frame transaction with a 33-byte blob hash."""

                blob_versioned_hashes: Sequence[  # type: ignore
                    OversizedHash
                ]

            tx = OversizedBlobHashTransaction(
                **kwargs,
                max_fee_per_blob_gas=1,
                blob_versioned_hashes=[
                    OversizedHash(b"\x01" + (0).to_bytes(32, "big"))
                ],
            )

    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize(
    "mutation",
    [
        "five_elements",
        "seven_elements",
        "target_undersized",
        "target_oversized",
        "value_out_of_range",
        "mode_leading_zero",
        "flags_leading_zero",
        "frame_encoded_as_bytes",
    ],
)
@EIPChecklist.TransactionType.Test.Encoding.NewTypes.IncorrectEncoding()
def test_frame_item_encoding(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    mutation: str,
) -> None:
    """
    Reject frame items that are not the exact six-element tuple.

    Pins R-025 (frame tuple field list), R-055 (a present target is
    exactly twenty bytes; the accept side of the empty-string null
    target ships in every default-code state fixture of this
    directory), and R-057 (`2**256` as the first unrepresentable
    frame value, reachable only at the wire level). The mutated frame
    rides behind the canonical `VERIFY` frame so the tuple shape is
    the only violated clause.
    """
    kwargs = canonical_transaction_kwargs(pre)
    target = pre.nonexistent_account()
    frame: Any
    if mutation == "five_elements":

        class ShortTupleFrame(Frame):
            """Frame serializing five tuple elements."""

            def get_rlp_fields(self) -> List[str]:
                """Drop the data element from the frame tuple."""
                return [
                    field
                    for field in super().get_rlp_fields()
                    if field != "data"
                ]

        frame = ShortTupleFrame(gas_limit=AMPLE_FRAME_GAS, target=target)
    elif mutation == "seven_elements":

        class LongTupleFrame(Frame):
            """Frame serializing seven tuple elements."""

            extra_element: HexNumber

            def get_rlp_fields(self) -> List[str]:
                """Append an extra element to the frame tuple."""
                return super().get_rlp_fields() + ["extra_element"]

        frame = LongTupleFrame(
            gas_limit=AMPLE_FRAME_GAS, target=target, extra_element=1
        )
    elif mutation == "target_undersized":

        class UndersizedTargetFrame(Frame):
            """Frame with a 19-byte target encoding."""

            target: UndersizedAddressBytes | None  # type: ignore

        frame = UndersizedTargetFrame(
            gas_limit=AMPLE_FRAME_GAS, target=bytes(target)[:19]
        )
    elif mutation == "target_oversized":

        class OversizedTargetFrame(Frame):
            """Frame with a 21-byte target encoding."""

            target: OversizedAddressBytes | None  # type: ignore

        frame = OversizedTargetFrame(
            gas_limit=AMPLE_FRAME_GAS, target=bytes(target) + b"\x00"
        )
    elif mutation == "value_out_of_range":

        class OversizedValueFrame(Frame):
            """Frame with a 33-byte value encoding."""

            value: OversizedWord  # type: ignore

        # A `SENDER` frame keeps a nonzero value otherwise legal,
        # isolating the encoding bound as the violated clause.
        frame = OversizedValueFrame(
            mode=Spec.MODE_SENDER,
            gas_limit=AMPLE_FRAME_GAS,
            target=target,
            value=2**256,
        )
    elif mutation == "mode_leading_zero":

        class PaddedModeFrame(Frame):
            """Frame with a zero-padded mode encoding."""

            mode: PaddedZero  # type: ignore

        frame = PaddedModeFrame(
            mode=Spec.MODE_DEFAULT, gas_limit=AMPLE_FRAME_GAS, target=target
        )
    elif mutation == "flags_leading_zero":

        class PaddedFlagsFrame(Frame):
            """Frame with a zero-padded flags encoding."""

            flags: PaddedZero  # type: ignore

        frame = PaddedFlagsFrame(
            flags=Spec.APPROVE_NONE,
            gas_limit=AMPLE_FRAME_GAS,
            target=target,
        )
    else:
        assert mutation == "frame_encoded_as_bytes"

        class FramesAsBytesTransaction(Transaction):
            """Frame transaction encoding a frame item as bytes."""

            frames: List[Bytes] | None  # type: ignore

        tx = FramesAsBytesTransaction(
            **{**kwargs, "frames": [Bytes(verify_frame().rlp())]},
        )
        transaction_test(pre=pre, tx=tx)
        return

    kwargs["frames"] = [verify_frame(), frame]
    transaction_test(pre=pre, tx=Transaction(**kwargs))


@pytest.mark.parametrize(
    "mutation",
    [
        "three_elements",
        "five_elements",
        "entry_encoded_as_bytes",
    ],
)
@EIPChecklist.TransactionType.Test.Encoding.NewTypes.IncorrectEncoding()
def test_signature_item_encoding(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    mutation: str,
) -> None:
    """
    Reject signature items that are not the exact four-element tuple.

    Pins R-026 (signature tuple field list). Every arm keeps the entry
    contents valid — the scheme, the explicit twenty-byte sender as
    signer, the empty message, and a real secp256k1 signature where
    the tuple still serializes one — so the tuple shape is the only
    violated clause.
    """
    kwargs = canonical_transaction_kwargs(pre)
    signer_bytes = Bytes(kwargs["sender"])
    tx: Transaction
    if mutation == "three_elements":

        class ShortTupleSignature(FrameSignature):
            """Signature entry serializing three tuple elements."""

            def get_rlp_fields(self) -> List[str]:
                """Drop the signature element from the tuple."""
                return [
                    field
                    for field in super().get_rlp_fields()
                    if field != "signature"
                ]

        tx = Transaction(
            **kwargs,
            signatures=[
                ShortTupleSignature(
                    scheme=Spec.SCHEME_SECP256K1, signer=signer_bytes
                )
            ],
        )
    elif mutation == "five_elements":

        class LongTupleSignature(FrameSignature):
            """Signature entry serializing five tuple elements."""

            extra_element: HexNumber

            def get_rlp_fields(self) -> List[str]:
                """Append an extra element to the tuple."""
                return super().get_rlp_fields() + ["extra_element"]

        tx = Transaction(
            **kwargs,
            signatures=[
                LongTupleSignature(
                    scheme=Spec.SCHEME_SECP256K1,
                    signer=signer_bytes,
                    extra_element=1,
                )
            ],
        )
    else:
        assert mutation == "entry_encoded_as_bytes"

        class SignaturesAsBytesTransaction(Transaction):
            """Frame transaction encoding a signature entry as bytes."""

            signatures: List[Bytes] | None  # type: ignore

            def _sign_frame_signatures(self) -> None:
                """Skip signing; the entry is pre-encoded bytes."""
                return

            @property
            def signing_signatures(self) -> List[FrameSignature]:
                """Return no entries; the raw list is pre-encoded."""
                return []

        # Sign an identical twin to derive a real, fully populated
        # signature entry, then serialize that entry as a byte string
        # in place of the four-element tuple.
        twin = Transaction(
            sender=kwargs["sender"],
            nonce=0,
            frames=[verify_frame()],
        ).with_signature_and_sender()
        assert twin.signatures is not None
        tx = SignaturesAsBytesTransaction(
            **{**kwargs, "signatures": [Bytes(twin.signatures[0].rlp())]},
        )

    transaction_test(pre=pre, tx=tx)
