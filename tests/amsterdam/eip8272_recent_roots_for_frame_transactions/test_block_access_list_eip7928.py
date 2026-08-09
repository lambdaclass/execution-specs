"""
Tests for how EIP-8272's reference check appears in
[EIP-7928](https://eips.ethereum.org/EIPS/eip-7928) block access lists.

EIP-8272 says a valid reference "affects warm/cold gas accounting only".
Read literally that excludes the block access list: the check consults
the transaction pre-state, which is not execution, so no account and no
storage key enters the list on its account.

**This reading is contested.** The check unquestionably reads
`RECENT_ROOT_ADDRESS[storage_key]`, and an implementation that records
that read produces a different block access list — which is committed to
in the header, so the two readings give different block hashes. At least
one client records the read. The negative arm below therefore pins the
literal reading of the sentence, and is the one assertion in this suite
whose expected value should be re-checked if EIP-8272 ever says what it
means here; see `SPEC_FEEDBACK.md`.
"""

from typing import List

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    BalAccountExpectation,
    Block,
    BlockAccessListExpectation,
    BlockchainTestFiller,
    Environment,
    Op,
    RecentRootReference,
    Transaction,
)

from .helpers import (
    CANARY_VALUE,
    EXECUTED_VALUE,
    SENDER_FRAME_GAS,
    SLOT_EXECUTED,
    as_int,
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


def test_reference_check_adds_no_block_access_list_entry(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    current_slot: int,
    references: List[RecentRootReference],
) -> None:
    """
    Checking a reference leaves the recent root contract out of the
    block access list entirely.

    The transaction declares one valid reference and no frame goes
    anywhere near `RECENT_ROOT_ADDRESS`, so under the literal reading of
    "affects warm/cold gas accounting only" the contract contributes no
    account entry at all — not a storage read, not a touch. `None` as an
    account expectation is EIP-7928's assertion that an address is absent
    from the list.

    The absence is only meaningful next to
    [`test_frame_read_of_the_recent_root_contract_is_recorded`][control],
    which puts a real `SLOAD` of the very same key in the very same
    account and shows the entry appearing. Without that control this arm
    would also pass against a client whose block access list omitted the
    account for some unrelated reason.

    [control]: ref:tests.amsterdam.eip8272_recent_roots_for_frame_transactions.test_block_access_list_eip7928.test_frame_read_of_the_recent_root_contract_is_recorded
    """  # noqa: E501
    seed_entries(pre, references)
    target: Address = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE) + Op.STOP,
        storage={SLOT_EXECUTED: CANARY_VALUE},
    )
    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(target, SENDER_FRAME_GAS),
        recent_root_references=references,
    )

    blockchain_test(
        genesis_environment=Environment(slot_number=current_slot),
        pre=pre,
        blocks=[
            Block(
                slot_number=current_slot,
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations={Spec.RECENT_ROOT_ADDRESS: None}
                ),
            )
        ],
        post={target: Account(storage={SLOT_EXECUTED: EXECUTED_VALUE})},
    )


def test_frame_read_of_the_recent_root_contract_is_recorded(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    current_slot: int,
    references: List[RecentRootReference],
) -> None:
    """
    A frame that really reads a recent root entry does put the account
    and the key in the block access list.

    This is the control for
    [`test_reference_check_adds_no_block_access_list_entry`][negative]:
    same declared reference, same storage key, same block access list
    machinery — the only difference is that here a frame calls the recent
    root contract and executes an `SLOAD` on the declared key, so the
    read happens inside execution rather than inside the pre-execution
    check. The key appears under `storage_reads`, which is what makes the
    other test's `None` an observation rather than an artefact.

    The contract's runtime code is substituted into the pre-state,
    because EIP-8272 still lists `RECENT_ROOT_CODE` as `TBD` and there is
    no defined bytecode to call.

    [negative]: ref:tests.amsterdam.eip8272_recent_roots_for_frame_transactions.test_block_access_list_eip7928.test_reference_check_adds_no_block_access_list_entry
    """  # noqa: E501
    declared_key = as_int(
        compute_storage_key(references[0].source_id, int(references[0].slot))
    )
    entries = entries_for(references)
    pre[Spec.RECENT_ROOT_ADDRESS] = Account(
        nonce=1,
        code=Op.SSTORE(SLOT_EXECUTED, Op.SLOAD(declared_key)) + Op.STOP,
        storage={**entries, SLOT_EXECUTED: CANARY_VALUE},
    )
    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(
            Spec.RECENT_ROOT_ADDRESS, SENDER_FRAME_GAS
        ),
        recent_root_references=references,
    )

    blockchain_test(
        genesis_environment=Environment(slot_number=current_slot),
        pre=pre,
        blocks=[
            Block(
                slot_number=current_slot,
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations={
                        Spec.RECENT_ROOT_ADDRESS: BalAccountExpectation(
                            storage_reads=[declared_key],
                        )
                    }
                ),
            )
        ],
        post={
            Spec.RECENT_ROOT_ADDRESS: Account(
                storage={
                    **entries,
                    SLOT_EXECUTED: entries[declared_key],
                }
            )
        },
    )
