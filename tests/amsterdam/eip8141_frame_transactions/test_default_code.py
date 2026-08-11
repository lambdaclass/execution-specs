"""
Default code tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

An empty-codehash resolved target runs the default code. In `VERIFY`
mode it approves the frame's allowed scope only when a canonical
secp256k1 entry — empty message, resolved signer equal to the
resolved target — sits at the scope-selected signature index; each
requirement violated in isolation reverts the frame and invalidates
the transaction. The happy paths are exercised throughout the suite.
"""

import pytest
from execution_testing import (
    Alloc,
    Bytes,
    FrameSignature,
    StateTestFiller,
    Transaction,
    TransactionException,
    keccak256,
)

from .helpers import verify_frame
from .signature_helpers import P256_SIGNATURE, signed_digest_entry
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

# EIP-8141 is slated for the fork after Amsterdam, so fixtures are
# labeled with the pseudo `Bogota` fork (Amsterdam + EIP-8141), even
# though the spec prototypes the EIP inside the Amsterdam fork module.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")


@pytest.mark.exception_test
def test_default_code_rejects_empty_scope(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Revert the default code when the `VERIFY` frame allows no
    approval scope: with zero flags there is nothing to approve, so
    the frame reverts and the transaction is invalid.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        frames=[verify_frame(flags=Spec.APPROVE_NONE)],
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(pre=pre, tx=tx, post={})


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "entry_shape",
    [
        pytest.param("no_entries", id="empty_signature_list"),
        pytest.param("p256", id="p256_entry_at_index"),
        pytest.param("explicit_digest", id="explicit_digest_at_index"),
    ],
)
def test_default_code_signature_requirements(
    state_test: StateTestFiller,
    pre: Alloc,
    entry_shape: str,
) -> None:
    """
    Violate one default-code entry requirement at a time at signature
    index zero: no entry at all, a valid P-256 entry — wrong scheme
    and a signer that can never equal the sender — and a valid
    secp256k1 entry over an explicit digest instead of the canonical
    hash. Each reverts the approving frame and invalidates the
    transaction.
    """
    sender = pre.fund_eoa()
    if entry_shape == "no_entries":
        signatures = []
    elif entry_shape == "p256":
        signatures = [
            FrameSignature(
                scheme=Spec.SCHEME_P256,
                msg=Bytes(b"\x01" * 32),
                signer=Bytes(keccak256(P256_SIGNATURE[64:])[12:]),
                signature=Bytes(P256_SIGNATURE),
            )
        ]
    else:
        signatures = [signed_digest_entry(sender.key)]

    tx = Transaction(
        sender=sender,
        frames=[verify_frame()],
        signatures=signatures,
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(pre=pre, tx=tx, post={})


@pytest.mark.exception_test
def test_default_code_signature_index_selection(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Swap the sender's and payer's entries in a sponsored flow: both
    canonical entries are present, but the execution-approving frame
    selects index zero and finds the payer's entry, whose resolved
    signer is not the resolved target — pinning that the default code
    selects by index, not by searching the list.
    """
    sender = pre.fund_eoa()
    payer = pre.fund_eoa()

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(flags=Spec.APPROVE_EXECUTION),
            verify_frame(flags=Spec.APPROVE_PAYMENT, target=payer),
        ],
        signatures=[
            # The payer's entry sits at the execution index zero and
            # the sender's at the payment index one, inverting the
            # scope-selected order.
            FrameSignature(
                scheme=Spec.SCHEME_SECP256K1,
                signer=Bytes(payer),
                secret_key=payer.key,
            ),
            FrameSignature(
                scheme=Spec.SCHEME_SECP256K1,
                signer=Bytes(sender),
            ),
        ],
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={},
    )
