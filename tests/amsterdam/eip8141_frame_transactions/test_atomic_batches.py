"""
Atomic batch tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

An atomic batch is a maximal contiguous run of flagged frames plus
the first unflagged frame terminating it. A failure inside the batch
rolls the state back to just before the batch and skips its remaining
frames; executed members keep their status and gas in the receipts,
with their logs discarded. Frames outside the failing batch — and
earlier, disjoint batches — are unaffected.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytes,
    CodeGasMeasure,
    Fork,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
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

MARKER = 0xC0DE
"""Distinctive nonzero marker value."""

# A fresh SSTORE costs STATE_BYTES_PER_STORAGE_SET * COST_PER_STATE_BYTE
# of state gas under EIP-8037, and a frame transaction holds no state
# gas reservoir, so frames whose code writes storage need room for the
# writes.
WRITE_FRAME_GAS = 500_000

LOG_FRAME_GAS = 10_000
"""Gas limit of frames that only emit an empty log."""


def slot_word(slot: int) -> Bytes:
    """Encode a storage slot as a 32-byte frame data word."""
    return Bytes(slot.to_bytes(32, "big"))


def test_multiple_atomic_batches(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Fail the second of two disjoint atomic batches: the first batch's
    writes survive, and only the frames of the failing batch are
    rolled back — an unflagged frame terminates the batch extent.
    """
    sender = pre.fund_eoa()
    writer = pre.deploy_contract(
        code=Op.SSTORE(Op.CALLDATALOAD(0), MARKER) + Op.STOP
    )
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            # First batch: flagged writer plus unflagged terminator.
            default_frame(
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=writer,
                data=slot_word(0),
                gas_limit=WRITE_FRAME_GAS,
            ),
            default_frame(
                target=writer,
                data=slot_word(1),
                gas_limit=WRITE_FRAME_GAS,
            ),
            # Second batch: flagged writer terminated by a reverter.
            default_frame(
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=writer,
                data=slot_word(2),
                gas_limit=WRITE_FRAME_GAS,
            ),
            default_frame(target=reverter, gas_limit=100_000),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                # The unrolled member keeps its execution status.
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_FAILURE),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=1),
            writer: Account(
                storage={0: MARKER, 1: MARKER, 2: 0},
            ),
        },
    )


BATCH_LOGGER_COUNT = 60
"""Flagged logging frames in the large batch, which with the writer
and the terminator exhausts most of the frame count limit."""


def test_large_batch_unroll(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Unroll a batch spanning nearly the whole frame list: the failing
    terminator discards the first member's storage write and every
    member's log, while all sixty-one executed members keep their
    success status in the receipts.
    """
    sender = pre.fund_eoa()
    writer = pre.deploy_contract(
        code=Op.SSTORE(Op.CALLDATALOAD(0), MARKER) + Op.LOG0(0, 0) + Op.STOP
    )
    logger = pre.deploy_contract(code=Op.LOG0(0, 0) + Op.STOP)
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=writer,
                data=slot_word(0),
                gas_limit=WRITE_FRAME_GAS,
            ),
            *[
                default_frame(
                    flags=Spec.ATOMIC_BATCH_FLAG,
                    target=logger,
                    gas_limit=LOG_FRAME_GAS,
                )
                for _ in range(BATCH_LOGGER_COUNT)
            ],
            default_frame(target=reverter, gas_limit=100_000),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                *[
                    # Unrolled members keep their status; their logs
                    # are discarded with their state changes.
                    FrameReceipt(status=Spec.STATUS_SUCCESS, logs=[])
                    for _ in range(BATCH_LOGGER_COUNT + 1)
                ],
                FrameReceipt(status=Spec.STATUS_FAILURE),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=1),
            writer: Account(storage={0: 0}),
        },
    )


def test_atomic_batch_unwinds_warmth(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Discard the warm journal of an unrolled batch: a batch member
    warms an address, the terminator fails, and a later frame pays
    the cold access again — the rollback restores the pre-batch
    journal, not just the pre-batch state.
    """
    sender = pre.fund_eoa()
    subject = pre.fund_eoa(amount=1)
    toucher = pre.deploy_contract(
        code=Op.POP(Op.BALANCE(subject)) + Op.STOP
    )
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))
    cold_access = Op.BALANCE(address_warm=False).gas_cost(fork)
    measured = Op.BALANCE(address=subject, address_warm=False)
    probe_code = CodeGasMeasure(
        code=measured,
        overhead_cost=measured.gas_cost(fork) - cold_access,
        extra_stack_items=1,
        sstore_key=0,
    )
    probe = pre.deploy_contract(code=probe_code)

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=toucher,
                gas_limit=200_000,
            ),
            default_frame(target=reverter, gas_limit=100_000),
            default_frame(target=probe, gas_limit=WRITE_FRAME_GAS),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_FAILURE),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=1),
            probe: Account(storage={0: cold_access}),
        },
    )
