"""
Tests that a reference is checked against the containing block's own
pre-state for
[EIP-8272: Recent Roots for Frame Transactions](https://eips.ethereum.org/EIPS/eip-8272).

Reference validity is not a property of the tuple: the same transaction
bytes are valid in a block whose transaction pre-state holds the entry
and invalid in one that does not. That pre-state includes the earlier
transactions of the very same block.
"""  # noqa: E501

from typing import List

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    Environment,
    Op,
    RecentRootReference,
    Transaction,
    TransactionException,
)

from .helpers import (
    CANARY_VALUE,
    EXECUTED_VALUE,
    SLOT_EXECUTED,
    as_int,
    compute_entry_hash,
    compute_storage_key,
    entries_for,
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

INVALID_REFERENCE = TransactionException.TYPE_6_INVALID_RECENT_ROOT_REFERENCE


@pytest.mark.parametrize(
    "entry_seeded",
    [
        pytest.param(True, id="entry_present"),
        pytest.param(
            False, id="entry_absent", marks=pytest.mark.exception_test
        ),
    ],
)
def test_reference_validity_follows_the_block_prestate(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    current_slot: int,
    references: List[RecentRootReference],
    entry_seeded: bool,
) -> None:
    """
    The same transaction is valid or invalid according to what the block
    it lands in holds.

    Both arms build byte-identical transactions in blocks at the same
    slot; the only difference is whether the referenced entry exists in
    the pre-state. That is the whole of "a reference is valid only in a
    block whose transaction pre-state contains the referenced entry" —
    and it is why two competing blocks at one slot can disagree about the
    same tuple.
    """
    seed_entries(pre, references if entry_seeded else [])
    target = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE) + Op.STOP,
        storage={SLOT_EXECUTED: CANARY_VALUE},
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(target),
        recent_root_references=references,
        error=None if entry_seeded else INVALID_REFERENCE,
    )

    blockchain_test(
        genesis_environment=Environment(slot_number=current_slot),
        pre=pre,
        blocks=[
            Block(
                slot_number=current_slot,
                txs=[tx],
                exception=None if entry_seeded else INVALID_REFERENCE,
            )
        ],
        post={
            target: Account(
                storage={
                    SLOT_EXECUTED: (
                        EXECUTED_VALUE if entry_seeded else CANARY_VALUE
                    )
                }
            )
        },
    )


@pytest.mark.parametrize(
    "write_in_same_block",
    [
        pytest.param(True, id="entry_written_by_earlier_tx"),
        pytest.param(
            False, id="no_earlier_write", marks=pytest.mark.exception_test
        ),
    ],
)
def test_reference_to_an_entry_written_earlier_in_the_block(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    current_slot: int,
    references: List[RecentRootReference],
    write_in_same_block: bool,
) -> None:
    """
    A reference may name an entry that an earlier transaction of the same
    block wrote.

    "The transaction pre-state includes all prior transactions in the
    same block", and this is the only construction that shows it: a
    writer is substituted at the recent root contract — the specification
    still leaves `RECENT_ROOT_CODE` undefined — so the first transaction
    can store the pair a reference needs, and the second transaction
    references it.

    The key and the entry are both computed test-side from the spec's
    concatenation rules and handed to the writer as calldata, so no
    client logic is being echoed back. The control arm drops the writing
    transaction and nothing else, which turns the very same reference
    invalid.
    """
    reference = references[0]
    storage_key = compute_storage_key(reference.source_id, int(reference.slot))
    entry_hash = compute_entry_hash(reference)

    # The recent root contract's own runtime code is `TBD`, so a writer
    # that stores exactly the pair the check reads stands in for it.
    pre[Spec.RECENT_ROOT_ADDRESS] = Account(
        nonce=1,
        code=Op.SSTORE(Op.CALLDATALOAD(0), Op.CALLDATALOAD(32)) + Op.STOP,
    )
    target = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE) + Op.STOP,
        storage={SLOT_EXECUTED: CANARY_VALUE},
    )

    writing_tx = Transaction(
        sender=pre.fund_eoa(),
        to=Spec.RECENT_ROOT_ADDRESS,
        data=bytes(storage_key) + bytes(entry_hash),
        gas_limit=200_000,
    )
    referencing_tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(target),
        recent_root_references=references,
        error=None if write_in_same_block else INVALID_REFERENCE,
    )

    blockchain_test(
        genesis_environment=Environment(slot_number=current_slot),
        pre=pre,
        blocks=[
            Block(
                slot_number=current_slot,
                txs=(
                    [writing_tx, referencing_tx]
                    if write_in_same_block
                    else [referencing_tx]
                ),
                exception=(None if write_in_same_block else INVALID_REFERENCE),
            )
        ],
        post={
            target: Account(
                storage={
                    SLOT_EXECUTED: (
                        EXECUTED_VALUE if write_in_same_block else CANARY_VALUE
                    )
                }
            ),
            Spec.RECENT_ROOT_ADDRESS: Account(
                storage=(
                    entries_for(references) if write_in_same_block else {}
                )
            ),
        },
    )


def test_current_slot_write_cannot_invalidate_a_reference(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    current_slot: int,
    references: List[RecentRootReference],
) -> None:
    """
    A write made during the current slot cannot invalidate a reference
    that is valid.

    The substituted writer overwrites the cell the current slot maps to —
    index `current_slot mod RECENT_ROOT_LENGTH` — in the first
    transaction, and the second transaction references an entry one slot
    old, which the window keeps in a different cell. The reference still
    validates.

    The two keys are asserted to differ before the block is built, so the
    fixture cannot pass by accident on a client whose ring indexing is
    wrong in a way that separates them anyway.
    """
    reference = references[0]
    referenced_key = compute_storage_key(
        reference.source_id, int(reference.slot)
    )
    current_slot_key = compute_storage_key(reference.source_id, current_slot)
    assert referenced_key != current_slot_key

    entries = entries_for(references)
    pre[Spec.RECENT_ROOT_ADDRESS] = Account(
        nonce=1,
        code=Op.SSTORE(Op.CALLDATALOAD(0), Op.CALLDATALOAD(32)) + Op.STOP,
        storage=entries,
    )
    target = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE) + Op.STOP,
        storage={SLOT_EXECUTED: CANARY_VALUE},
    )

    overwritten_entry = as_int(referenced_key)
    writing_tx = Transaction(
        sender=pre.fund_eoa(),
        to=Spec.RECENT_ROOT_ADDRESS,
        data=bytes(current_slot_key) + overwritten_entry.to_bytes(32, "big"),
        gas_limit=200_000,
    )
    referencing_tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(target),
        recent_root_references=references,
    )

    blockchain_test(
        genesis_environment=Environment(slot_number=current_slot),
        pre=pre,
        blocks=[
            Block(
                slot_number=current_slot,
                txs=[writing_tx, referencing_tx],
            )
        ],
        post={
            target: Account(storage={SLOT_EXECUTED: EXECUTED_VALUE}),
            Spec.RECENT_ROOT_ADDRESS: Account(
                storage={
                    **entries,
                    as_int(current_slot_key): overwritten_entry,
                }
            ),
        },
    )
