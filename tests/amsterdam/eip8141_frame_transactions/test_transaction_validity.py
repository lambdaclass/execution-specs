"""
Transaction-level admission tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

Each case varies one admission rule — the sender nonce, the chain id,
or a fee bound — around its boundary, using a minimal frame
transaction whose other fields are valid.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytes,
    Environment,
    Fork,
    FrameSignature,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
)

from .helpers import default_frame, verify_frame
from .signature_helpers import signed_digest_entry, with_tampered_components
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

# EIP-8141 is slated for the fork after Amsterdam, so fixtures are
# labeled with the pseudo `Bogota` fork (Amsterdam + EIP-8141), even
# though the spec prototypes the EIP inside the Amsterdam fork module.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

SLOT_EXECUTED = 0x01
"""Storage slot used by target contracts to record execution."""


@pytest.mark.parametrize(
    "nonce_delta,error",
    [
        pytest.param(0, None, id="nonce_exact"),
        pytest.param(
            1,
            TransactionException.NONCE_MISMATCH_TOO_HIGH,
            id="nonce_plus_one",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            -1,
            TransactionException.NONCE_MISMATCH_TOO_LOW,
            id="nonce_minus_one",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_nonce_mismatch(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_delta: int,
    error: TransactionException | None,
) -> None:
    """
    Require the transaction nonce to equal the sender's state nonce.

    A contract sender starts at nonce one, so both the one-below and
    the one-above mismatches are reachable without mutating the shared
    pre-allocation.
    """
    sender = pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )
    tx = Transaction(
        sender=sender,
        nonce=1 + nonce_delta,
        frames=[verify_frame()],
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        # The sender's nonce only increments if the transaction is
        # valid and executes.
        post={sender: Account(nonce=1 if error else 2)},
    )


@pytest.mark.exception_test
def test_nonce_and_signature_both_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Reject a transaction whose nonce mismatches the state nonce and
    whose appended signature entry is out of range at the same time.

    The spec's behavior section checks the nonce before validating
    signatures, while the reference implementation validates the
    signature entries during static validation, before reading the
    sender's state nonce; both orders reject the transaction, so
    either exception class is accepted here. The divergence is
    recorded in the run's spec feedback.

    Either way no frame may run: the storing target must stay
    untouched.
    """
    sender = pre.fund_eoa()
    assert sender.key is not None
    target = pre.deploy_contract(code=Op.SSTORE(SLOT_EXECUTED, 1) + Op.STOP)
    # A structurally well-formed entry whose r component is out of
    # range, appended after the canonical entry the default code
    # consumes.
    bad_entry = with_tampered_components(signed_digest_entry(sender.key), r=0)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[
            verify_frame(),
            default_frame(target=target),
        ],
        signatures=[
            FrameSignature(scheme=Spec.SCHEME_SECP256K1, signer=Bytes(sender)),
            bad_entry,
        ],
        error=[
            TransactionException.NONCE_MISMATCH_TOO_HIGH,
            TransactionException.TYPE_6_INVALID_SIGNATURE,
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=0),
            target: Account(storage={SLOT_EXECUTED: 0}),
        },
    )


@pytest.mark.parametrize(
    "chain_id_delta,error",
    [
        pytest.param(0, None, id="chain_id_correct"),
        pytest.param(
            1,
            TransactionException.INVALID_CHAINID,
            id="chain_id_mismatch",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_wrong_chain_id(
    state_test: StateTestFiller,
    pre: Alloc,
    chain_id_delta: int,
    error: TransactionException | None,
) -> None:
    """
    Reject a frame transaction whose chain id is not the executing
    chain's.

    The pinned spec's static constraints only bound the chain id's
    range; the equality with the executing chain follows from the
    field definition, matching every other typed transaction. The
    run's spec feedback records the missing assert.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        chain_id=1 + chain_id_delta,
        frames=[verify_frame()],
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=0 if error else 1)},
    )


BASE_FEE = 1_000
"""Base fee of the block executing the fee-boundary transactions."""


@pytest.mark.parametrize(
    "max_fee,error",
    [
        pytest.param(BASE_FEE, None, id="max_fee_at_base_fee"),
        pytest.param(
            BASE_FEE - 1,
            TransactionException.INSUFFICIENT_MAX_FEE_PER_GAS,
            id="max_fee_below_base_fee",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_max_fee_below_base_fee(
    state_test: StateTestFiller,
    pre: Alloc,
    max_fee: int,
    error: TransactionException | None,
) -> None:
    """
    Apply the inherited EIP-1559 inclusion gate to frame transactions:
    the maximum fee must cover the block's base fee exactly, and one
    below it is rejected.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        max_fee_per_gas=max_fee,
        max_priority_fee_per_gas=0,
        frames=[verify_frame()],
        error=error,
    )

    state_test(
        env=Environment(base_fee_per_gas=BASE_FEE),
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=0 if error else 1)},
    )


# The largest power of two the derived transaction gas limit can reach
# under the per-transaction gas cap, so an in-range fee lands the
# maximum cost product exactly on the 2**256 boundary.
OVERFLOW_MAX_GAS = 2**24


@pytest.mark.parametrize(
    "max_fee,error",
    [
        pytest.param(
            2**232 - 1,
            None,
            id="max_cost_below_overflow",
        ),
        pytest.param(
            2**232,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="max_cost_at_overflow",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_max_cost_overflow(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    max_fee: int,
    error: TransactionException | None,
) -> None:
    """
    Drive the transaction's maximum cost across the word boundary.

    The frame gas limit is sized so the derived transaction gas limit
    is exactly `2**24`; a maximum fee of `2**232` then makes
    `max_gas * max_fee_per_gas` exactly `2**256`, which the spec's gas
    accounting rejects — the maximum cost the payer escrows must fit in
    a machine word. One fee step below, the product fits and a
    maximally funded payer covers it.

    The bound is checked as the maximum cost is derived, before any
    frame executes, so the observable class is a static one.
    """
    cap = fork.transaction_gas_limit_cap()
    assert cap is not None and cap >= OVERFLOW_MAX_GAS
    intrinsic = Spec.FRAME_TX_INTRINSIC_COST + Spec.FRAME_TX_PER_FRAME_COST

    sender = pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=2**256 - 1,
    )
    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=max_fee,
        max_priority_fee_per_gas=0,
        frames=[verify_frame(gas_limit=OVERFLOW_MAX_GAS - intrinsic)],
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=1 if error else 2)},
    )
