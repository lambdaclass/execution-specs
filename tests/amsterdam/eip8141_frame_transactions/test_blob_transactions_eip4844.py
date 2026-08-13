"""
Blob handling tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141)
composed with [EIP-4844](https://eips.ethereum.org/EIPS/eip-4844).

Frame transactions may carry blobs: the inclusion gate compares the
blob fee cap against the block's blob base fee, the payer is charged
exactly the blob gas at the base fee — never the cap — and `BLOBHASH`
serves every frame from the same transaction-level list.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytes,
    EIPChecklist,
    Environment,
    Fork,
    Hash,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
)

from .helpers import default_frame, verify_frame
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

# EIP-8141 is slated for the fork after Amsterdam, so fixtures are
# labeled with the pseudo `Bogota` fork (Amsterdam + EIP-8141), even
# though the spec prototypes the EIP inside the Amsterdam fork module.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

SLOT_RESULT = 0x01
"""Storage slot the probe contract writes what it read into."""

PROBE_FRAME_GAS = 500_000
"""Gas limit of frames whose code writes storage, leaving room for
the state gas of fresh writes under EIP-8037."""


def excess_for_priced_blobs(fork: Fork) -> int:
    """
    Return an excess blob gas that pushes the blob base fee to at
    least ten, solving against the fork's own price curve so the
    boundary arms stay meaningful under fee-curve changes.
    """
    price_of = fork.blob_gas_price_calculator()
    step = fork.blob_gas_per_blob()
    excess = 0
    while price_of(excess_blob_gas=excess) < 10:
        excess += step
    return excess


BASE_FEE = 7
PRIORITY_FEE = 0
MAX_FEE = 1_000
FUNDS = 10**18

APPROVE_ALL_CODE = Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT)
"""Sender code approving execution and payment."""


def blob_hash(index: int) -> Hash:
    """Return a distinct KZG-versioned hash for the given index."""
    return Hash(b"\x01" + index.to_bytes(31, "big"))


@pytest.mark.parametrize(
    "blob_count",
    [
        pytest.param(1, id="single_blob"),
        pytest.param(None, id="max_blobs"),
    ],
)
def test_blob_frame_transaction_acceptance(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    blob_count: int | None,
) -> None:
    """
    Accept a frame transaction carrying one blob and one carrying the
    per-transaction maximum, with correctly versioned hashes; the
    over-maximum and wrong-version sides are covered by the static
    validity suite.
    """
    count = blob_count if blob_count else fork.max_blobs_per_tx()
    sender = pre.fund_eoa()

    tx = Transaction(
        sender=sender,
        max_fee_per_blob_gas=1,
        blob_versioned_hashes=[blob_hash(i) for i in range(count)],
        frames=[verify_frame()],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=1)},
    )


@pytest.mark.parametrize(
    "fee_shortfall,error",
    [
        pytest.param(0, None, id="blob_fee_cap_at_base_fee"),
        pytest.param(
            1,
            TransactionException.INSUFFICIENT_MAX_FEE_PER_BLOB_GAS,
            id="blob_fee_cap_below_base_fee",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_blob_fee_inclusion_gate(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    fee_shortfall: int,
    error: TransactionException | None,
) -> None:
    """
    Gate inclusion on the blob fee cap: with nonzero excess blob gas
    the block's blob base fee exceeds one, and a cap exactly at that
    fee is accepted while one below is rejected.
    """
    excess_blob_gas = excess_for_priced_blobs(fork)
    blob_base_fee = fork.blob_gas_price_calculator()(
        excess_blob_gas=excess_blob_gas
    )
    sender = pre.fund_eoa()

    tx = Transaction(
        sender=sender,
        max_fee_per_blob_gas=blob_base_fee - fee_shortfall,
        blob_versioned_hashes=[blob_hash(0)],
        frames=[verify_frame()],
        error=error,
    )

    state_test(
        env=Environment(excess_blob_gas=excess_blob_gas),
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=0 if error else 1)},
    )


def test_blob_fee_settlement(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Charge the payer the blob gas at the blob base fee exactly: the
    fee cap is set far higher and must not be collected, pinned by the
    payer's post balance next to the hand-computed execution fee.
    """
    excess_blob_gas = excess_for_priced_blobs(fork)
    blob_base_fee = fork.blob_gas_price_calculator()(
        excess_blob_gas=excess_blob_gas
    )
    blob_count = 2
    blob_gas = blob_count * fork.blob_gas_per_blob()
    sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=FUNDS)
    gas_costs = fork.gas_costs()

    verify_gas_used = gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    gas_used = (
        Spec.FRAME_TX_INTRINSIC_COST
        + Spec.FRAME_TX_PER_FRAME_COST
        + verify_gas_used
    )
    effective_price = BASE_FEE + PRIORITY_FEE

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=PRIORITY_FEE,
        max_fee_per_blob_gas=blob_base_fee * 100,
        blob_versioned_hashes=[blob_hash(i) for i in range(blob_count)],
        frames=[verify_frame()],
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=gas_used,
            payer=sender,
        ),
    )

    state_test(
        env=Environment(
            base_fee_per_gas=BASE_FEE, excess_blob_gas=excess_blob_gas
        ),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                nonce=2,
                balance=FUNDS
                - gas_used * effective_price
                - blob_gas * blob_base_fee,
            ),
        },
    )


@EIPChecklist.TransactionType.Test.TxScopedAttributes.Persistent.Throughout()
def test_blobhash_across_frames(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Serve `BLOBHASH` from the transaction-level hash list in every
    frame: two frames read the same two hashes, the out-of-range
    index reads zero, and the blob count selector reports two.
    """
    sender = pre.fund_eoa()
    hashes = [blob_hash(0), blob_hash(1)]
    slot = Op.CALLDATALOAD(0)
    probe = pre.deploy_contract(
        code=Op.SSTORE(slot, Op.BLOBHASH(0))
        + Op.SSTORE(Op.ADD(slot, 1), Op.BLOBHASH(1))
        + Op.SSTORE(Op.ADD(slot, 2), Op.ADD(Op.BLOBHASH(2), 1))
        + Op.SSTORE(Op.ADD(slot, 3), Op.TXPARAM(Spec.TXPARAM_BLOB_COUNT))
        + Op.STOP
    )

    def probe_frame_at(base_slot: int) -> "Bytes":
        return Bytes(base_slot.to_bytes(32, "big"))

    tx = Transaction(
        sender=sender,
        max_fee_per_blob_gas=1,
        blob_versioned_hashes=hashes,
        frames=[
            verify_frame(),
            default_frame(
                target=probe,
                data=probe_frame_at(0x00),
                gas_limit=PROBE_FRAME_GAS,
            ),
            default_frame(
                target=probe,
                data=probe_frame_at(0x10),
                gas_limit=PROBE_FRAME_GAS,
            ),
        ],
    )

    expected = {
        0x00: hashes[0],
        0x01: hashes[1],
        0x02: 1,
        0x03: 2,
        0x10: hashes[0],
        0x11: hashes[1],
        0x12: 1,
        0x13: 2,
    }
    state_test(
        pre=pre,
        tx=tx,
        post={probe: Account(storage=expected)},
    )
