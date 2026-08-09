"""
Tests for how EIP-8272's references compose with
[EIP-8141](https://eips.ethereum.org/EIPS/eip-8141) frames.

The reference set sits inside the signed envelope, so no frame can change
it and no frame can be reached without it having been checked first. What
validation code gets in return is the ability to bind a root without
reading anybody's storage.
"""

from typing import Dict, List

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Bytecode,
    Bytes,
    Environment,
    Fork,
    Frame,
    FrameReceipt,
    FrameSignature,
    Hash,
    Op,
    RecentRootReference,
    StateTestFiller,
    Storage,
    Transaction,
    TransactionException,
    TransactionReceipt,
)

from ..eip8141_frame_transactions.spec import Spec as Spec8141
from .helpers import (
    CANARY_VALUE,
    EXECUTED_VALUE,
    SENDER_FRAME_GAS,
    SLOT_EXECUTED,
    VERIFY_FRAME_GAS,
    as_int,
    compute_source_id,
    entries_for,
    rlp_encode_references,
    seed_entries,
    verify_and_sender_frames,
)
from .spec import CanonicalVector, Spec, ref_spec_8272

REFERENCE_SPEC_GIT_PATH = ref_spec_8272.git_path
REFERENCE_SPEC_VERSION = ref_spec_8272.version

pytestmark = [
    pytest.mark.valid_from("Bogota"),
    pytest.mark.pre_alloc_mutable,
]

INVALID_REFERENCE = TransactionException.TYPE_6_INVALID_RECENT_ROOT_REFERENCE

DECOY_ROOT = Hash(
    0x7777777777777777777777777777777777777777777777777777777777777777
)
"""
A second root, as valid as the first once seeded.

Used wherever a test needs a reference that would satisfy the protocol's
own check but not the application's, so that a failure is attributable to
the binding rather than to the reference being unsatisfied.
"""


def binding_sender_code(reference: RecentRootReference) -> Bytecode:
    """
    Return sender code that approves only for one exact reference tuple.

    All three fields are read back through `RECENTROOTREFLOAD` and
    compared with constants baked into the code — the privacy-proof
    pattern the specification's security considerations describe, where
    the tuple is a public input the validation logic must re-check. The
    code reads no storage of any account, which is the restriction
    EIP-8272 exists to work around.
    """
    matches = Op.AND(
        Op.AND(
            Op.EQ(
                Op.RECENTROOTREFLOAD(field=Spec.FIELD_SOURCE_ID, index=0),
                as_int(reference.source_id),
            ),
            Op.EQ(
                Op.RECENTROOTREFLOAD(field=Spec.FIELD_SLOT, index=0),
                int(reference.slot),
            ),
        ),
        Op.EQ(
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=0),
            as_int(reference.root),
        ),
    )
    # A mismatch multiplies the requested scope down to `APPROVE_NONE`,
    # so the frame still runs to completion but authorizes nothing and
    # the transaction is left without an approved payer.
    return Op.APPROVE(
        0, 0, Op.MUL(matches, Spec8141.APPROVE_EXECUTION_AND_PAYMENT)
    )


@pytest.mark.parametrize(
    "binds_declared_reference",
    [
        pytest.param(True, id="tuple_matches"),
        pytest.param(
            False,
            id="tuple_mismatched",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_verify_frame_binds_reference(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    source_id: Hash,
    root: Hash,
    reference_slot: int,
    binds_declared_reference: bool,
) -> None:
    """
    A `VERIFY` frame authorizes the transaction only when the declared
    tuple is the one its code expects.

    Both arms declare a reference that the protocol accepts — both are
    seeded and in-window — so the protocol-level check passes either way
    and the only thing that differs is whether the sender's code
    approves. The mismatched arm therefore fails as an unapproved frame
    transaction, not as an invalid reference, which is what makes it a
    test of the binding rather than of the check.

    This is also the negative the security considerations call for: a
    root that is genuinely published, but by the wrong source, must not
    satisfy an application that expects a specific one.
    """
    declared = RecentRootReference(
        source_id=source_id, slot=reference_slot, root=root
    )
    expected = (
        declared
        if binds_declared_reference
        else RecentRootReference(
            source_id=compute_source_id(CanonicalVector.SOURCE_ADDRESS, 9),
            slot=reference_slot,
            root=DECOY_ROOT,
        )
    )
    seed_entries(pre, [declared])

    sender = pre.deploy_contract(
        code=binding_sender_code(expected), balance=10**18
    )
    target = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE) + Op.STOP,
        storage={SLOT_EXECUTED: CANARY_VALUE},
    )

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=verify_and_sender_frames(target),
        recent_root_references=[declared],
        error=(
            None
            if binds_declared_reference
            else TransactionException.TYPE_6_INVALID_FRAME_EXECUTION
        ),
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            target: Account(
                storage={
                    SLOT_EXECUTED: (
                        EXECUTED_VALUE
                        if binds_declared_reference
                        else CANARY_VALUE
                    )
                }
            )
        },
    )


@pytest.mark.exception_test
def test_reference_set_is_signed(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    target: Address,
    source_id: Hash,
    root: Hash,
    reference_slot: int,
) -> None:
    """
    A signature made over one reference set does not authorize another.

    The transaction is transmitted with a reference set that is itself
    valid and seeded, carrying the signature bytes produced over a
    different set. Everything else — sender, frames, nonce, fees — is
    identical, so the reference check passes and the only broken thing is
    the signature: EIP-8141's signature hash covers the payload, and
    EIP-8272 inserted its field into that payload.

    Without the transmitted set also being valid, this fixture would be
    passed by an implementation that never signed the field at all.
    """
    sender = pre.fund_eoa()
    signed_over = [
        RecentRootReference(
            source_id=source_id, slot=reference_slot, root=root
        )
    ]
    transmitted = [
        RecentRootReference(
            source_id=source_id, slot=reference_slot, root=DECOY_ROOT
        )
    ]
    assert rlp_encode_references(signed_over) != rlp_encode_references(
        transmitted
    )
    # Both sets name the same source and slot, so they share one storage
    # key; the entry seeded is the transmitted set's, which is the only
    # one the client ever sees and checks.
    seed_entries(pre, transmitted)

    donor = Transaction(
        sender=sender,
        frames=verify_and_sender_frames(target),
        recent_root_references=signed_over,
    )
    donor.sign()
    assert donor.signatures is not None
    donor_signature = donor.signatures[0].signature

    tx = Transaction(
        sender=sender,
        frames=verify_and_sender_frames(target),
        recent_root_references=transmitted,
        signatures=[
            FrameSignature(
                scheme=Spec8141.SCHEME_SECP256K1,
                signer=Bytes(sender),
                msg=Bytes(b""),
                signature=donor_signature,
            )
        ],
        # The signature recovers a key other than the declared signer,
        # which is EIP-8141's signature-entry validity failure rather
        # than anything EIP-8272 checks.
        error=TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: CANARY_VALUE})},
    )


def test_reference_set_is_immutable_across_frames(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    source_id: Hash,
    root: Hash,
    reference_slot: int,
) -> None:
    """
    Frame data cannot add to, remove from or modify the reference set.

    The middle frame's `data` is a byte-for-byte copy of a well-formed
    encoding of a *different* reference set — the exact bytes that would
    have to be interpreted for the set to change. Frames on both sides of
    it read the count and all three fields, and every read must return
    the declared set.

    Comparing the two frames against each other, rather than against a
    single expectation, means an implementation that let the data through
    would have to corrupt both reads identically to pass.
    """
    declared = RecentRootReference(
        source_id=source_id, slot=reference_slot, root=root
    )
    impostor = RecentRootReference(
        source_id=compute_source_id(CanonicalVector.SOURCE_ADDRESS, 9),
        slot=reference_slot - 1,
        root=DECOY_ROOT,
    )
    seed_entries(pre, [declared])

    def reading_code(storage: Storage) -> Bytecode:
        return (
            Op.SSTORE(
                storage.store_next(1),
                Op.TXPARAM(param=Spec.TXPARAM_RECENT_ROOT_REFERENCE_COUNT),
            )
            + Op.SSTORE(
                storage.store_next(as_int(declared.source_id)),
                Op.RECENTROOTREFLOAD(field=Spec.FIELD_SOURCE_ID, index=0),
            )
            + Op.SSTORE(
                storage.store_next(int(declared.slot)),
                Op.RECENTROOTREFLOAD(field=Spec.FIELD_SLOT, index=0),
            )
            + Op.SSTORE(
                storage.store_next(as_int(declared.root)),
                Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=0),
            )
            + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
            + Op.STOP
        )

    before_storage = Storage()
    after_storage = Storage()
    reader_before = pre.deploy_contract(code=reading_code(before_storage))
    reader_after = pre.deploy_contract(code=reading_code(after_storage))
    smuggler = pre.deploy_contract(code=Op.STOP)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            Frame(
                mode=Spec8141.MODE_VERIFY,
                flags=Spec8141.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=VERIFY_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                target=reader_before,
                gas_limit=SENDER_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                target=smuggler,
                gas_limit=SENDER_FRAME_GAS,
                data=Bytes(rlp_encode_references([impostor])),
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                target=reader_after,
                gas_limit=SENDER_FRAME_GAS,
            ),
        ],
        recent_root_references=[declared],
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            reader_before: Account(storage=before_storage),
            reader_after: Account(storage=after_storage),
        },
    )


def test_recentrootrefload_in_default_and_sender_frames(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    references: List[RecentRootReference],
) -> None:
    """
    The opcode may be used in any frame mode.

    A `DEFAULT` frame and a `SENDER` frame of one transaction read the
    same reference and store identical words. The `VERIFY` mode is
    exercised separately by the binding test above, where the opcode runs
    inside the static context a `VERIFY` frame executes in.
    """
    seed_entries(pre, references)
    reference = references[0]

    def reading_code(storage: Storage) -> Bytecode:
        return (
            Op.SSTORE(
                storage.store_next(as_int(reference.root)),
                Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=0),
            )
            + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
            + Op.STOP
        )

    default_storage = Storage()
    sender_storage = Storage()
    default_reader = pre.deploy_contract(code=reading_code(default_storage))
    sender_reader = pre.deploy_contract(code=reading_code(sender_storage))

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            Frame(
                mode=Spec8141.MODE_VERIFY,
                flags=Spec8141.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=VERIFY_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_DEFAULT,
                target=default_reader,
                gas_limit=SENDER_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                target=sender_reader,
                gas_limit=SENDER_FRAME_GAS,
            ),
        ],
        recent_root_references=references,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            default_reader: Account(storage=default_storage),
            sender_reader: Account(storage=sender_storage),
        },
    )


@pytest.mark.exception_test
def test_invalid_reference_prevents_frame_execution(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    source_id: Hash,
    root: Hash,
    reference_slot: int,
) -> None:
    """
    When any reference is invalid, no frame executes at all.

    Two references are declared, the first satisfied and the second not,
    and every frame of the transaction would leave a mark: the first
    writes storage, the second is the sender's own nonce increment.
    Afterwards the storage still holds its pre-value and the sender's
    nonce is unchanged — the transaction is not merely failed, it never
    ran.

    Seeding the first reference makes the point precise: the check is not
    all-or-nothing on the set being empty, it rejects on any member.
    """
    sender = pre.fund_eoa()
    satisfied = RecentRootReference(
        source_id=source_id, slot=reference_slot, root=root
    )
    unsatisfied = RecentRootReference(
        source_id=source_id, slot=reference_slot - 1, root=root
    )
    entries = entries_for([satisfied])
    seed_entries(pre, [satisfied])

    target = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE) + Op.STOP,
        storage={SLOT_EXECUTED: CANARY_VALUE},
    )

    tx = Transaction(
        sender=sender,
        frames=verify_and_sender_frames(target),
        recent_root_references=[satisfied, unsatisfied],
        error=INVALID_REFERENCE,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            target: Account(storage={SLOT_EXECUTED: CANARY_VALUE}),
            sender: Account(nonce=0),
            Spec.RECENT_ROOT_ADDRESS: Account(nonce=1, storage=entries),
        },
    )


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "reference_satisfied",
    [
        pytest.param(False, id="reference_unsatisfied"),
        pytest.param(True, id="reference_satisfied"),
    ],
)
def test_nonce_is_checked_before_the_references(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    target: Address,
    references: List[RecentRootReference],
    reference_satisfied: bool,
) -> None:
    """
    A transaction whose nonce and references are both bad is rejected for
    its nonce.

    EIP-8272 places its check in EIP-8141's existing sequence: references
    "are checked after the EIP-8141 nonce check and before frame
    execution". EIP-8141's step 1 is `tx.nonce == state[tx.sender].nonce`,
    so when both fail the nonce is the one reached, and the rejection
    carries its reason and not the reference's.

    Only a transaction that violates both clauses can distinguish the two
    orders — either defect alone is rejected the same way whichever check
    runs first. The sender is pre-set to nonce 1 and the transaction
    declares nonce 0, which is "nonce too low" against a sender that has
    already sent once.

    The `reference_satisfied` arm is the control. Its entry is seeded, so
    the reference check would pass if it ran, and the only reason left
    for the rejection is the nonce. The two arms differ in nothing but
    whether the entry is present, so a client that reported the reference
    failure would part company with the specification on the first arm
    alone — which is what makes the ordering, rather than the rejection
    itself, the thing under test.

    Neither arm is sensitive to how EIP-8272 orders against EIP-8141's
    signature validation, which the specification does not state: the
    sender here is an EOA carrying no signature entries.
    """
    sender = pre.fund_eoa(nonce=1)
    recent_root_storage: Dict[int, int] = {}
    if reference_satisfied:
        seed_entries(pre, references)
        recent_root_storage = entries_for(references)

    tx = Transaction(
        sender=sender,
        nonce=0,
        frames=verify_and_sender_frames(target),
        recent_root_references=references,
        error=TransactionException.NONCE_MISMATCH_TOO_LOW,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            target: Account(storage={SLOT_EXECUTED: CANARY_VALUE}),
            sender: Account(nonce=1),
            Spec.RECENT_ROOT_ADDRESS: Account(
                nonce=1, storage=recent_root_storage
            ),
        },
    )


def test_references_survive_an_atomic_batch_rollback(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    references: List[RecentRootReference],
) -> None:
    """
    An atomic batch that fails unrolls its writes but not the reference
    set.

    The batch's first frame reads the reference and writes it to storage;
    its terminator reverts, so that write is unrolled. A later frame
    reads the same reference again and must still see it — the set lives
    in the envelope, not in the state the batch rolled back.
    """
    seed_entries(pre, references)
    reference = references[0]

    batch_storage = Storage()
    batch_reader = pre.deploy_contract(
        code=Op.SSTORE(
            batch_storage.store_next(CANARY_VALUE),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=0),
        )
        + Op.STOP,
        storage={0: CANARY_VALUE},
    )
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))

    after_storage = Storage()
    after_reader = pre.deploy_contract(
        code=Op.SSTORE(
            after_storage.store_next(as_int(reference.root)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=0),
        )
        + Op.SSTORE(after_storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            Frame(
                mode=Spec8141.MODE_VERIFY,
                flags=Spec8141.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=VERIFY_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                flags=Spec8141.ATOMIC_BATCH_FLAG,
                target=batch_reader,
                gas_limit=SENDER_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                target=reverter,
                gas_limit=VERIFY_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                target=after_reader,
                gas_limit=SENDER_FRAME_GAS,
            ),
        ],
        recent_root_references=references,
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=Spec8141.STATUS_SUCCESS),
                FrameReceipt(status=Spec8141.STATUS_SUCCESS, logs=[]),
                FrameReceipt(status=Spec8141.STATUS_FAILURE),
                FrameReceipt(status=Spec8141.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            batch_reader: Account(storage=batch_storage),
            after_reader: Account(storage=after_storage),
        },
    )


SLOT_FRAME_COUNT = 0x40
"""Slot holding the transaction's frame count."""

SLOT_FRAME_INDEX = 0x41
"""Slot holding the index of the frame the probe runs in."""

SLOT_OWN_GAS_LIMIT = 0x42
"""Slot holding the probe's own frame gas limit."""

SLOT_OWN_MODE = 0x43
"""Slot holding the probe's own frame mode."""

SLOT_OWN_FLAGS_PLUS_ONE = 0x44
"""
Slot holding the probe's own frame flags, offset by one.

The flags of a plain `SENDER` frame are `APPROVE_NONE`, and a stored
zero cannot be told apart from a slot that was never written.
"""

SLOT_OWN_DATA_LENGTH_PLUS_ONE = 0x45
"""Slot holding the probe's own frame data length, offset by one."""

SLOT_VERIFY_GAS_LIMIT = 0x46
"""Slot holding the gas limit of the transaction's `VERIFY` frame."""

SLOT_VERIFY_MODE = 0x47
"""Slot holding the mode of the transaction's `VERIFY` frame."""


@pytest.mark.parametrize(
    "reference_count",
    [0, Spec.MAX_RECENT_ROOT_REFERENCES],
    ids=lambda count: f"references_{count}",
)
def test_reference_free_frame_tx_matches_baseline(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    reference_count: int,
) -> None:
    """
    Declaring references changes nothing about a frame transaction's
    frame layout or frame execution.

    This is the negative control the rest of the suite is read against.
    Everything EIP-8272 adds is in the envelope and in the intrinsic
    cost; "the frame layout and frame execution rules are otherwise
    unchanged", so two transactions that differ only in their reference
    set — none against the maximum sixteen — must present the identical
    frame list to the code running inside it, and must run their frames
    on the identical budgets.

    Every expected word is a constant the transaction was built from:
    the frame count and the probe's own index are positions in the list
    this test writes, and the four `FRAMEPARAM` values are the gas
    limits, modes and flags it declared. None of them may shift when
    sixteen references are inserted into the payload. An implementation
    that let the reference list occupy a position in the frame list, or
    that renumbered frames around it, reports a different count or
    index; one that changed the frames' declared budgets reports
    different gas limits.

    The two frames whose gas is pinned execute nothing. The `VERIFY`
    frame has no target at all, so it spends nothing and reports zero.
    The last frame's whole code is `STOP`, which is free, so all it
    spends is the fork's cold account access charge for reaching its
    target — a quantity of the fork's schedule, not of this EIP, and so
    the same in both arms. That is where a reference charge misapplied
    to a frame rather than to the transaction would surface: the
    sixteen-reference arm would report the per-reference charge on top.
    It is also where pre-warming would surface if it reached further
    than the recent root contract, since a warm target would be charged
    `WARM_ACCESS` instead. `status` is asserted alongside both, so a
    frame that never ran cannot masquerade as a frame that ran for
    free.
    """
    references = [
        RecentRootReference(
            source_id=source_id, slot=current_slot - 1 - index, root=root
        )
        for index in range(reference_count)
    ]
    seed_entries(pre, references)

    probe_frame_index = 1
    verify_frame_index = 0
    frame_count = 3

    storage = Storage()
    probe_code = (
        Op.SSTORE(
            SLOT_FRAME_COUNT, Op.TXPARAM(param=Spec8141.TXPARAM_FRAME_COUNT)
        )
        + Op.SSTORE(
            SLOT_FRAME_INDEX, Op.TXPARAM(param=Spec8141.TXPARAM_FRAME_INDEX)
        )
        + Op.SSTORE(
            SLOT_OWN_GAS_LIMIT,
            Op.FRAMEPARAM(
                frame_index=probe_frame_index,
                param=Spec8141.FRAMEPARAM_GAS_LIMIT,
            ),
        )
        + Op.SSTORE(
            SLOT_OWN_MODE,
            Op.FRAMEPARAM(
                frame_index=probe_frame_index,
                param=Spec8141.FRAMEPARAM_MODE,
            ),
        )
        + Op.SSTORE(
            SLOT_OWN_FLAGS_PLUS_ONE,
            Op.ADD(
                Op.FRAMEPARAM(
                    frame_index=probe_frame_index,
                    param=Spec8141.FRAMEPARAM_FLAGS,
                ),
                1,
            ),
        )
        + Op.SSTORE(
            SLOT_OWN_DATA_LENGTH_PLUS_ONE,
            Op.ADD(
                Op.FRAMEPARAM(
                    frame_index=probe_frame_index,
                    param=Spec8141.FRAMEPARAM_DATA_LENGTH,
                ),
                1,
            ),
        )
        + Op.SSTORE(
            SLOT_VERIFY_GAS_LIMIT,
            Op.FRAMEPARAM(
                frame_index=verify_frame_index,
                param=Spec8141.FRAMEPARAM_GAS_LIMIT,
            ),
        )
        + Op.SSTORE(
            SLOT_VERIFY_MODE,
            Op.FRAMEPARAM(
                frame_index=verify_frame_index,
                param=Spec8141.FRAMEPARAM_MODE,
            ),
        )
        + Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE)
        + Op.STOP
    )
    storage[SLOT_FRAME_COUNT] = frame_count
    storage[SLOT_FRAME_INDEX] = probe_frame_index
    storage[SLOT_OWN_GAS_LIMIT] = SENDER_FRAME_GAS
    storage[SLOT_OWN_MODE] = Spec8141.MODE_SENDER
    storage[SLOT_OWN_FLAGS_PLUS_ONE] = Spec8141.APPROVE_NONE + 1
    storage[SLOT_OWN_DATA_LENGTH_PLUS_ONE] = 1
    storage[SLOT_VERIFY_GAS_LIMIT] = VERIFY_FRAME_GAS
    storage[SLOT_VERIFY_MODE] = Spec8141.MODE_VERIFY
    storage[SLOT_EXECUTED] = EXECUTED_VALUE

    prober = pre.deploy_contract(
        code=probe_code, storage={SLOT_EXECUTED: CANARY_VALUE}
    )
    quiet_target = pre.deploy_contract(code=Op.STOP)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            Frame(
                mode=Spec8141.MODE_VERIFY,
                flags=Spec8141.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=VERIFY_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                target=prober,
                gas_limit=SENDER_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                target=quiet_target,
                gas_limit=VERIFY_FRAME_GAS,
            ),
        ],
        recent_root_references=references,
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=Spec8141.STATUS_SUCCESS, gas_used=0),
                FrameReceipt(status=Spec8141.STATUS_SUCCESS),
                FrameReceipt(
                    status=Spec8141.STATUS_SUCCESS,
                    gas_used=fork.gas_costs().COLD_ACCOUNT_ACCESS,
                ),
            ],
        ),
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )
