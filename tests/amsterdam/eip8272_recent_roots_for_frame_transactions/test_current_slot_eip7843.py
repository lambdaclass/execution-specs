"""
Tests that EIP-8272's `current_slot` is EIP-7843's `slotNumber`.

[EIP-8272](https://eips.ethereum.org/EIPS/eip-8272) requires clients to
take `current_slot` from the
[EIP-7843](https://eips.ethereum.org/EIPS/eip-7843) `slotNumber` header
field, and forbids deriving it from `block.timestamp` with a fixed slot
duration. That is only observable when the two disagree.
"""

from typing import List

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Environment,
    Frame,
    Hash,
    Op,
    RecentRootReference,
    StateTestFiller,
    Storage,
    Transaction,
    TransactionException,
)

from .helpers import (
    CANARY_VALUE,
    EXECUTED_VALUE,
    SLOT_EXECUTED,
    seed_entries,
    verify_and_sender_frames,
)
from .spec import Spec, ref_spec_8272

REFERENCE_SPEC_GIT_PATH = ref_spec_8272.git_path
REFERENCE_SPEC_VERSION = ref_spec_8272.version

pytestmark = [
    pytest.mark.valid_from("Bogota"),
    pytest.mark.pre_alloc_mutable,
]

ASSUMED_SLOT_DURATION = 12
"""
Seconds per slot a client would use to derive a slot from a timestamp.

EIP-8272 forbids that derivation; the constant exists only to build the
timestamps that make a client doing it disagree with the header.
"""

HEADER_SLOT = 1_000
"""Slot number one of the mirrored arms puts in the header."""

TIMESTAMP_SLOT = 500
"""Slot number the other arm's timestamp implies."""

REFERENCE_SLOT = HEADER_SLOT - 1
"""
Slot the reference names in both arms.

One below `HEADER_SLOT` and well above `TIMESTAMP_SLOT`, so the same
tuple is in-window under one reading and in the future under the other.
"""


@pytest.mark.parametrize(
    "slot_number,timestamp,valid",
    [
        pytest.param(
            HEADER_SLOT,
            TIMESTAMP_SLOT * ASSUMED_SLOT_DURATION,
            True,
            id="header_ahead_of_timestamp",
        ),
        pytest.param(
            TIMESTAMP_SLOT,
            HEADER_SLOT * ASSUMED_SLOT_DURATION,
            False,
            id="header_behind_timestamp",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_current_slot_comes_from_the_header(
    state_test: StateTestFiller,
    pre: Alloc,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    root: Hash,
    slot_number: int,
    timestamp: int,
    valid: bool,
) -> None:
    """
    Two mirrored blocks carry the same reference tuple, with the header
    slot and the timestamp-implied slot swapped between them.

    The reference names slot 999. Under the header it is one slot old in
    the first arm and five hundred slots in the future in the second; a
    client deriving the slot from the timestamp reads it exactly the
    other way round. So the header reading accepts the first and rejects
    the second, and the timestamp reading does the opposite — no
    implementation can pass both arms while using the wrong source.

    The expectation is structural: neither arm asserts a slot number, only
    which of the two identical tuples the block accepts.
    """
    sender = pre.fund_eoa()
    references = [
        RecentRootReference(
            source_id=source_id, slot=REFERENCE_SLOT, root=root
        )
    ]
    seed_entries(pre, references)

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=(
            None
            if valid
            else TransactionException.TYPE_6_INVALID_RECENT_ROOT_REFERENCE
        ),
    )

    state_test(
        env=Environment(slot_number=slot_number, timestamp=timestamp),
        pre=pre,
        tx=tx,
        post={
            target: Account(
                storage={
                    SLOT_EXECUTED: (EXECUTED_VALUE if valid else CANARY_VALUE)
                }
            )
        },
    )


@pytest.mark.parametrize(
    "delta",
    [
        pytest.param(1, id="newest_referenceable_slot"),
        pytest.param(
            Spec.RECENT_ROOT_USABLE_WINDOW, id="oldest_referenceable_slot"
        ),
    ],
)
def test_slotnum_agrees_with_the_accepted_window(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    delta: int,
) -> None:
    """
    The slot a frame reads from `SLOTNUM` and the slot the reference
    check compared against are the same number.

    A frame subtracts the reference's own slot, read back through
    `RECENTROOTREFLOAD`, from `SLOTNUM` and stores the difference. That
    difference must be the distance the check just accepted — asserted at
    both ends of the window, so a client that validated against one slot
    number and exposes another is caught even if its two numbers differ
    by a constant.
    """
    reference_slot = current_slot - delta
    references = [
        RecentRootReference(
            source_id=source_id, slot=reference_slot, root=root
        )
    ]
    seed_entries(pre, references)

    storage = Storage()
    probe_code = (
        Op.SSTORE(
            storage.store_next(delta),
            Op.SUB(
                Op.SLOTNUM,
                Op.RECENTROOTREFLOAD(field=Spec.FIELD_SLOT, index=0),
            ),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    prober = pre.deploy_contract(code=probe_code)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(prober),
        recent_root_references=references,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )
