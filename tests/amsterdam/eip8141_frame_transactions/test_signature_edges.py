"""
Signature component edge tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

Sweep the secp256k1 component bounds around their exact limits. Values
outside the ranges — the recovery identifier above one, `r` at or
above the curve order, `s` above the low-half bound — fail the range
check and report the signature exception; values inside the ranges
that no longer match the signed digest pass the range check and fail
recovery, reporting the format exception. The differing classes pin
the exact comparison bounds.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytes,
    FrameSignature,
    StateTestFiller,
    Transaction,
    TransactionException,
)

from .helpers import verify_frame
from .signature_helpers import (
    SECP256K1N,
    signed_digest_entry,
    with_tampered_components,
)
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

# EIP-8141 is slated for the fork after Amsterdam, so fixtures are
# labeled with the pseudo `Bogota` fork (Amsterdam + EIP-8141), even
# though the spec prototypes the EIP inside the Amsterdam fork module.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "component,value,error",
    [
        # Recovery identifiers beyond one, including the legacy and
        # EIP-155 encodings, fail the range check.
        pytest.param(
            "v",
            28,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
            id="v_28",
        ),
        pytest.param(
            "v",
            35,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
            id="v_35",
        ),
        pytest.param(
            "v",
            36,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
            id="v_36",
        ),
        pytest.param(
            "v",
            255,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
            id="v_255",
        ),
        # Components beyond their range bounds fail the range check.
        pytest.param(
            "r",
            SECP256K1N + 1,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
            id="r_above_curve_order",
        ),
        pytest.param(
            "r",
            2**256 - 1,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
            id="r_max_word",
        ),
        pytest.param(
            "s",
            SECP256K1N,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
            id="s_at_curve_order",
        ),
        pytest.param(
            "s",
            2**256 - 1,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
            id="s_max_word",
        ),
        # A substituted `r` inside the range is not the x-coordinate
        # of a curve point, so recovery itself fails cryptographically
        # and reports the signature exception.
        pytest.param(
            "r",
            SECP256K1N - 1,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
            id="r_below_curve_order_off_curve",
        ),
        # Components inside their ranges but no longer matching the
        # signed digest pass the range check and fail recovery with a
        # mismatched signer.
        pytest.param(
            "s",
            SECP256K1N // 2,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="s_at_half_order_mismatches",
        ),
        pytest.param(
            "s",
            SECP256K1N // 2 - 1,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="s_below_half_order_mismatches",
        ),
    ],
)
def test_signature_component_edges(
    state_test: StateTestFiller,
    pre: Alloc,
    component: str,
    value: int,
    error: TransactionException,
) -> None:
    """
    Replace one component of an otherwise valid appended entry with a
    boundary value and check the rejection class: range failures and
    recovery failures report different exceptions, pinning the exact
    comparison widths of the component bounds.
    """
    sender = pre.fund_eoa()
    assert sender.key is not None
    entry = with_tampered_components(
        signed_digest_entry(sender.key), **{component: value}
    )

    tx = Transaction(
        sender=sender,
        frames=[verify_frame()],
        signatures=[
            FrameSignature(scheme=Spec.SCHEME_SECP256K1, signer=Bytes(sender)),
            entry,
        ],
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=0)},
    )
