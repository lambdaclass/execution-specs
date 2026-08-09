"""
Reference validity tests for
[EIP-8272: Recent Roots for Frame Transactions](https://eips.ethereum.org/EIPS/eip-8272).

A declared reference is satisfied only when the root source it names
stored, for the slot it names, an entry committing to that same source,
slot and root — and only while that slot is still inside the rolling
window. These tests pin both halves, and pin the two derivations the
check performs against a hand-computed vector.
"""  # noqa: E501

from typing import List

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    EIPChecklist,
    Environment,
    Frame,
    Hash,
    RecentRootReference,
    StateTestFiller,
    Transaction,
    TransactionException,
)

from ..eip8141_frame_transactions.spec import Spec as Spec8141
from .helpers import (
    CANARY_VALUE,
    EXECUTED_VALUE,
    SLOT_EXECUTED,
    as_int,
    compute_entry_hash,
    compute_source_id,
    compute_storage_key,
    entries_for,
    seed_entries,
)
from .spec import CanonicalVector, Spec, ref_spec_8272

REFERENCE_SPEC_GIT_PATH = ref_spec_8272.git_path
REFERENCE_SPEC_VERSION = ref_spec_8272.version

pytestmark = [
    pytest.mark.valid_from("Bogota"),
    # Every test here seeds entries under the recent root contract,
    # which lives at a fixed, fork-allocated address.
    pytest.mark.pre_alloc_mutable,
]

INVALID_REFERENCE = TransactionException.TYPE_6_INVALID_RECENT_ROOT_REFERENCE


@pytest.mark.parametrize(
    "delta,valid",
    [
        pytest.param(
            0,
            False,
            id="current_slot",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(1, True, id="previous_slot"),
        pytest.param(2, True, id="two_slots_back"),
        pytest.param(
            Spec.RECENT_ROOT_USABLE_WINDOW - 1, True, id="window_8190"
        ),
        pytest.param(Spec.RECENT_ROOT_USABLE_WINDOW, True, id="window_8191"),
        pytest.param(
            Spec.RECENT_ROOT_LENGTH,
            False,
            id="window_8192",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            Spec.RECENT_ROOT_LENGTH + 1,
            False,
            id="window_8193",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_reference_window_boundaries(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    delta: int,
    valid: bool,
) -> None:
    """
    Sweep the rolling window with the named entry always correctly
    seeded, so the only clause under test is the slot arithmetic.

    Expectation derived from `1 <= current_slot - slot <=
    RECENT_ROOT_USABLE_WINDOW` read directly: delta 0 fails the lower
    bound because the current slot is not referenceable, delta 8191 is
    the last accepted distance, and delta 8192 — which shares a window
    index with the current slot — is the first expired one. The 8191/8192
    flip is the assertion that pins `RECENT_ROOT_USABLE_WINDOW`.
    """
    sender = pre.fund_eoa()
    references = [
        RecentRootReference(
            source_id=source_id, slot=current_slot - delta, root=root
        )
    ]
    seed_entries(pre, references)

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=None if valid else INVALID_REFERENCE,
    )

    state_test(
        env=env,
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


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "reference_slot_offset",
    [
        pytest.param(1, id="next_slot"),
        pytest.param(1_000, id="far_future_slot"),
    ],
)
def test_reference_future_slot(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    reference_slot_offset: int,
) -> None:
    """
    Reject a reference naming a slot after the containing block's.

    Derived from "References MUST target slots strictly before
    `current_slot`". The entry is seeded so the tuple would satisfy the
    hash check, isolating the ordering clause.
    """
    sender = pre.fund_eoa()
    references = [
        RecentRootReference(
            source_id=source_id,
            slot=current_slot + reference_slot_offset,
            root=root,
        )
    ]
    seed_entries(pre, references)

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=INVALID_REFERENCE,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: CANARY_VALUE})},
    )


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "current_slot,reference_slot",
    [
        pytest.param(0, Spec.MAX_SLOT, id="wraps_to_one"),
        pytest.param(
            Spec.RECENT_ROOT_USABLE_WINDOW // 2,
            Spec.MAX_SLOT - 7,
            id="wraps_mid_window",
        ),
        pytest.param(
            Spec.RECENT_ROOT_USABLE_WINDOW - 1,
            Spec.MAX_SLOT,
            id="wraps_to_window_edge",
        ),
    ],
)
def test_reference_slot_underflows_window(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    reference_slot: int,
) -> None:
    """
    Reject a future slot whose unsigned distance wraps back into the
    window.

    An implementation that subtracts before it compares evaluates
    `(current_slot - slot) mod 2**64` and finds a small positive number,
    so it accepts a reference the spec rejects. The parameters make that
    trap bite: each arm is asserted, before the transaction is built, to
    name a slot strictly after `current_slot` whose wrapped distance
    nevertheless lands inside `[1, RECENT_ROOT_USABLE_WINDOW]`. The
    wrapped distance is `current_slot + (2**64 - slot)`, so only a
    current slot no larger than the window can produce one — the edge arm
    wraps to exactly `RECENT_ROOT_USABLE_WINDOW`, the last distance the
    spec accepts.

    Derived by evaluating the spec's two clauses in their stated order:
    "references MUST target slots strictly before `current_slot`" is
    decided on the ordering alone, before the difference means anything.
    """
    sender = pre.fund_eoa()
    assert reference_slot > current_slot
    wrapped_distance = (current_slot - reference_slot) % (Spec.MAX_SLOT + 1)
    assert 1 <= wrapped_distance <= Spec.RECENT_ROOT_USABLE_WINDOW

    references = [
        RecentRootReference(
            source_id=source_id, slot=reference_slot, root=root
        )
    ]
    seed_entries(pre, references)

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=INVALID_REFERENCE,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: CANARY_VALUE})},
    )


@pytest.mark.parametrize(
    "current_slot,valid",
    [
        pytest.param(
            0, False, id="slot_zero_current", marks=pytest.mark.exception_test
        ),
        pytest.param(1, True, id="slot_zero_previous"),
    ],
)
def test_reference_slot_zero(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    root: Hash,
    valid: bool,
) -> None:
    """
    A root written during slot `S` becomes referenceable in slot `S + 1`,
    read at the arithmetic edge `S = 0`.

    Slot zero is rejected while the block is itself in slot zero and
    accepted one slot later, which is the spec's sentence stated at its
    smallest instance. Zero is also the slot value with the shortest
    canonical RLP encoding, so this fixture exercises that edge too.
    """
    sender = pre.fund_eoa()
    references = [RecentRootReference(source_id=source_id, slot=0, root=root)]
    seed_entries(pre, references)

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=None if valid else INVALID_REFERENCE,
    )

    state_test(
        env=env,
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


@pytest.mark.exception_test
def test_reference_missing_entry(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    references: List[RecentRootReference],
) -> None:
    """
    Reject an in-window reference whose storage key was never written.

    All entries are initially zero, and zero commits to nothing, so a
    reference to a slot the source never wrote in cannot be satisfied.
    Nothing is seeded here: the transaction is otherwise identical to the
    accepted `previous_slot` arm of the window sweep.
    """
    sender = pre.fund_eoa()

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=INVALID_REFERENCE,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            target: Account(storage={SLOT_EXECUTED: CANARY_VALUE}),
            Spec.RECENT_ROOT_ADDRESS: Account(nonce=1, storage={}),
        },
    )


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "defect",
    [
        pytest.param("missing_entry", id="missing_entry"),
        pytest.param("future_slot", id="future_slot"),
    ],
)
@pytest.mark.parametrize(
    "position",
    [
        pytest.param(0, id="first_position"),
        pytest.param(Spec.MAX_RECENT_ROOT_REFERENCES - 1, id="last_position"),
    ],
)
def test_one_invalid_reference_in_a_full_list(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    position: int,
    defect: str,
) -> None:
    """
    Reject a full reference list in which exactly one reference is
    invalid, wherever it sits.

    Every reference is checked, so a list is valid only if all of it is:
    "if any reference is invalid, the transaction is invalid and no frame
    is executed". The other rejection fixtures declare one reference, or
    a few, and put the defect at the front — which is also what an
    implementation that checked a prefix of the list and stopped would
    reject. Moving the defect to the last of sixteen separates the two:
    the fifteen references before it are in-window and correctly seeded,
    so a checker that gave up early sees nothing wrong and admits a
    transaction that must be rejected.

    The `first_position` arm is the control. It differs from the
    `last_position` arm only in which of the sixteen references carries
    the defect, so if the two disagree it is position alone that decided
    it.

    Both defects are ones the specification already gives a verdict on,
    used here for where they sit rather than for what they are: an entry
    that was never written cannot satisfy `RECENT_ROOT_ADDRESS[
    storage_key] == entry_hash`, and a reference to the current slot
    fails `1 <= current_slot - slot`.
    """
    references = [
        RecentRootReference(
            source_id=source_id, slot=current_slot - 1 - index, root=root
        )
        for index in range(Spec.MAX_RECENT_ROOT_REFERENCES)
    ]
    if defect == "future_slot":
        references[position] = RecentRootReference(
            source_id=source_id, slot=current_slot, root=root
        )
        seeded = references
    else:
        seeded = references[:position] + references[position + 1 :]
    assert len(references) == Spec.MAX_RECENT_ROOT_REFERENCES
    assert len(entries_for(seeded)) == len(seeded)
    seed_entries(pre, seeded)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=frames,
        recent_root_references=references,
        error=INVALID_REFERENCE,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: CANARY_VALUE})},
    )


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param("root", id="wrong_root"),
        pytest.param("slot", id="wrong_slot"),
        pytest.param("source_id", id="wrong_source_id"),
    ],
)
def test_reference_entry_hash_binding(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    root: Hash,
    reference_slot: int,
    corrupt: str,
) -> None:
    """
    Reject a reference that differs from the stored entry in exactly one
    of the three fields the entry commits to.

    The seeded entry is the one for `(source_id, reference_slot, root)`.
    Each arm then declares a reference differing in one field only, so
    the sole violated clause is the entry-hash equality. Because the
    entry hash covers the slot while the storage key covers only the
    window index, the `slot` arm reaches a written cell and still fails.
    """
    sender = pre.fund_eoa()
    seeded = RecentRootReference(
        source_id=source_id, slot=reference_slot, root=root
    )
    other_source_id = compute_source_id(
        CanonicalVector.SOURCE_ADDRESS, CanonicalVector.SALT + 1
    )
    declared = RecentRootReference(
        source_id=other_source_id if corrupt == "source_id" else source_id,
        slot=(
            reference_slot - Spec.RECENT_ROOT_LENGTH
            if corrupt == "slot"
            else reference_slot
        ),
        root=Hash(as_int(root) ^ 1) if corrupt == "root" else root,
    )
    assert compute_entry_hash(declared) != compute_entry_hash(seeded)
    if corrupt == "slot":
        # Same window index, so the declared reference reads the very
        # cell the seeded entry occupies.
        assert compute_storage_key(
            declared.source_id, int(declared.slot)
        ) == compute_storage_key(seeded.source_id, int(seeded.slot))

    seed_entries(pre, [seeded])

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=[declared],
        error=INVALID_REFERENCE,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: CANARY_VALUE})},
    )


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "flipped_bit",
    [
        pytest.param(0, id="lowest_bit"),
        pytest.param(127, id="middle_bit"),
        pytest.param(255, id="highest_bit"),
    ],
)
def test_reference_entry_hash_near_miss(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    references: List[RecentRootReference],
    flipped_bit: int,
) -> None:
    """
    Reject a reference whose own key holds a value differing from its
    entry hash in a single bit.

    [`test_reference_entry_hash_binding`][b] seeds the entry of a
    *different* reference; this seeds a *near miss* of the right one.
    Only the second measures the width of the comparison. The clause is
    `RECENT_ROOT_ADDRESS[storage_key] == entry_hash`, an equality over all
    32 bytes, and a wrong entry that disagrees in most of them is
    rejected by any implementation that inspects any part of the word — a
    prefix, a suffix, or a folded digest of it.

    The seeded value is derived by hand rather than sampled: it is the
    entry hash of the declared reference, exclusive-ored with
    `1 << flipped_bit`. That changes exactly one of the 32 bytes and
    leaves the other 31 in agreement, so each arm names the region an
    implementation must be reading. Bit 0 lies in the last byte, bit 255
    in the first, and bit 127 in the seventeenth — so a comparison over
    the leading byte alone admits `lowest_bit`, one over the trailing
    byte alone admits `highest_bit`, and one over both ends admits
    `middle_bit`.

    [b]: ref:tests.amsterdam.eip8272_recent_roots_for_frame_transactions.test_reference_validity.test_reference_entry_hash_binding
    """  # noqa: E501
    sender = pre.fund_eoa()
    reference = references[0]
    key = as_int(compute_storage_key(reference.source_id, int(reference.slot)))
    entry = as_int(compute_entry_hash(reference))
    near_miss = entry ^ (1 << flipped_bit)

    # A near miss is only a near miss if it is a miss, and only reaches
    # the equality if it is distinguishable from the never-written zero
    # that `test_reference_missing_entry` already covers.
    assert near_miss != entry
    assert near_miss != 0
    assert (near_miss ^ entry).bit_count() == 1

    pre[Spec.RECENT_ROOT_ADDRESS] = Account(nonce=1, storage={key: near_miss})

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=INVALID_REFERENCE,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            target: Account(storage={SLOT_EXECUTED: CANARY_VALUE}),
            Spec.RECENT_ROOT_ADDRESS: Account(
                nonce=1, storage={key: near_miss}
            ),
        },
    )


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "decoy",
    [
        pytest.param("wrong_key", id="entry_at_wrong_key"),
        pytest.param("wrong_account", id="entry_under_expiry_verifier"),
        pytest.param("neighbours", id="unrelated_neighbouring_keys"),
    ],
)
def test_reference_check_ignores_unrelated_storage(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    references: List[RecentRootReference],
    decoy: str,
) -> None:
    """
    Reject a reference whose correct entry sits anywhere other than its
    own key on the recent root contract.

    Negative control for the check's reach: the correct 32-byte entry is
    placed one key over, on a different account, or surrounded by
    unrelated values, and none of those satisfies the reference. Proves
    the check reads exactly one cell of exactly one account, which is
    what bounds its validation work.
    """
    sender = pre.fund_eoa()
    reference = references[0]
    key = as_int(compute_storage_key(reference.source_id, int(reference.slot)))
    entry = as_int(compute_entry_hash(reference))

    if decoy == "wrong_key":
        pre[Spec.RECENT_ROOT_ADDRESS] = Account(
            nonce=1, storage={key + 1: entry}
        )
    elif decoy == "wrong_account":
        pre[Spec8141.EXPIRY_VERIFIER] = Account(
            nonce=0,
            code=Spec8141.EXPIRY_VERIFIER_CODE,
            storage={key: entry},
        )
    else:
        pre[Spec.RECENT_ROOT_ADDRESS] = Account(
            nonce=1,
            storage={key + offset: entry + offset for offset in range(1, 21)},
        )

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=INVALID_REFERENCE,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: CANARY_VALUE})},
    )


@EIPChecklist.TransactionType.Test.Encoding.ListField.Max()
@EIPChecklist.TransactionType.Test.Encoding.ListField.MaxPlusOne()
@EIPChecklist.NewTransactionValidityConstraint.Test()
@pytest.mark.parametrize(
    "reference_count,valid",
    [
        pytest.param(Spec.MAX_RECENT_ROOT_REFERENCES - 1, True, id="fifteen"),
        pytest.param(Spec.MAX_RECENT_ROOT_REFERENCES, True, id="sixteen"),
        pytest.param(
            Spec.MAX_RECENT_ROOT_REFERENCES + 1,
            False,
            id="seventeen",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_reference_count_boundary(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    reference_count: int,
    valid: bool,
) -> None:
    """
    Accept up to `MAX_RECENT_ROOT_REFERENCES` references and reject one
    more.

    Every reference names a distinct slot inside the window and is
    correctly seeded, so the over-cap arm's only defect is its length.
    The under/exact/over triple is read straight off "The number of
    references MUST NOT exceed `MAX_RECENT_ROOT_REFERENCES`".
    """
    sender = pre.fund_eoa()
    references = [
        RecentRootReference(
            source_id=source_id, slot=current_slot - 1 - index, root=root
        )
        for index in range(reference_count)
    ]
    assert len(entries_for(references)) == reference_count
    seed_entries(pre, references)

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=(
            None
            if valid
            else (
                TransactionException.TYPE_6_RECENT_ROOT_REFERENCE_COUNT_EXCEEDED
            )
        ),
    )

    state_test(
        env=env,
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


def test_duplicate_references_accepted(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    references: List[RecentRootReference],
) -> None:
    """
    Accept a transaction declaring the cap's worth of byte-identical
    references.

    "Duplicate references are valid. They are checked, charged, and
    preserved independently." The list is at the cap while naming a
    single storage key, which is asserted here before the transaction is
    built so a decoder that deduplicates before counting is caught by
    both this fixture and the count boundary.
    """
    sender = pre.fund_eoa()
    duplicates = references * Spec.MAX_RECENT_ROOT_REFERENCES
    assert len(duplicates) == Spec.MAX_RECENT_ROOT_REFERENCES
    assert len(entries_for(duplicates)) == 1
    seed_entries(pre, duplicates)

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=duplicates,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: EXECUTED_VALUE})},
    )


@pytest.mark.exception_test
def test_reference_count_cap_counts_duplicates(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    references: List[RecentRootReference],
) -> None:
    """
    Reject `MAX_RECENT_ROOT_REFERENCES + 1` byte-identical references.

    The cap is on "the number of references", and duplicates are
    "preserved independently" — so seventeen copies of one reference are
    seventeen references, even though they name one storage key and one
    entry hash. The count boundary above declares references that all
    differ, where the number declared and the number distinct are the
    same; a cap applied to distinct references passes it and admits this
    transaction.

    [`test_duplicate_references_accepted`][control] is the control for
    this arm: the same reference, repeated one time fewer, in the same
    frames, over the same single seeded entry — and accepted. So the only
    thing this transaction has that the accepted one does not is its
    seventeenth reference.

    [control]: ref:tests.amsterdam.eip8272_recent_roots_for_frame_transactions.test_reference_validity.test_duplicate_references_accepted
    """  # noqa: E501
    over_cap = references * (Spec.MAX_RECENT_ROOT_REFERENCES + 1)
    assert len(over_cap) == Spec.MAX_RECENT_ROOT_REFERENCES + 1
    assert len(entries_for(over_cap)) == 1
    seed_entries(pre, over_cap)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=frames,
        recent_root_references=over_cap,
        error=TransactionException.TYPE_6_RECENT_ROOT_REFERENCE_COUNT_EXCEEDED,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: CANARY_VALUE})},
    )


@pytest.mark.parametrize(
    "root_value",
    [
        pytest.param(Hash(0), id="zero_root"),
        pytest.param(Hash(2**256 - 1), id="max_root"),
    ],
)
def test_reference_opaque_root_values(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    reference_slot: int,
    root_value: Hash,
) -> None:
    """
    Accept a correctly seeded reference whatever the root's bytes are.

    Consensus treats `root` as opaque, so neither an all-zero nor an
    all-ones root is special. The all-zero arm is the load-bearing one:
    it separates "the root is zero" from "the cell is unset", which a
    check short-circuiting on a zero-valued root would conflate — and the
    stored entry is a hash, so it stays non-zero either way.
    """
    sender = pre.fund_eoa()
    references = [
        RecentRootReference(
            source_id=source_id, slot=reference_slot, root=root_value
        )
    ]
    seed_entries(pre, references)

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: EXECUTED_VALUE})},
    )


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "delta",
    [
        pytest.param(0, id="written_this_slot"),
        pytest.param(Spec.RECENT_ROOT_LENGTH, id="expired_same_index"),
    ],
)
def test_ring_index_collision(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    delta: int,
) -> None:
    """
    Reject both references that share the current slot's window index.

    `i = slot mod RECENT_ROOT_LENGTH` makes slots `C` and
    `C - RECENT_ROOT_LENGTH` collide on one cell — asserted here before
    anything is seeded, so a wrong modulus breaks the assertion rather
    than the outcome. That single cell is then reachable by two
    references, and the window rejects both: `C` for not being over yet
    and `C - 8192` for being expired. Together they are why a write made
    during the current slot cannot invalidate a reference that is valid.
    """
    sender = pre.fund_eoa()
    assert compute_storage_key(source_id, current_slot) == (
        compute_storage_key(source_id, current_slot - Spec.RECENT_ROOT_LENGTH)
    )
    assert compute_storage_key(source_id, current_slot) != (
        compute_storage_key(
            source_id, current_slot - Spec.RECENT_ROOT_USABLE_WINDOW
        )
    )

    references = [
        RecentRootReference(
            source_id=source_id, slot=current_slot - delta, root=root
        )
    ]
    seed_entries(pre, references)

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=references,
        error=INVALID_REFERENCE,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: CANARY_VALUE})},
    )


@pytest.mark.parametrize("current_slot", [1_000])
def test_reference_derivation_matches_hand_vector(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
) -> None:
    """
    Accept a reference whose entry is seeded from hand-computed literals.

    The storage key and entry hash placed in the pre-state are the
    literals of `CanonicalVector`, computed offline from the spec's
    concatenation rules; the transaction declares the tuple they were
    computed from. The client must arrive at the same key and the same
    32-byte entry to accept it, so this fixture pins both domain
    separators, the 20-byte address and 32-byte salt of `source_id`, the
    8-byte big-endian slot and window index, and the field order of both
    preimages — none of which is read back from the client.

    The helpers are cross-checked against the same literals here, so a
    drifting helper fails loudly instead of agreeing with itself.
    """
    sender = pre.fund_eoa()
    reference = RecentRootReference(
        source_id=CanonicalVector.SOURCE_ID,
        slot=CanonicalVector.SLOT,
        root=CanonicalVector.ROOT,
    )
    assert (
        compute_source_id(CanonicalVector.SOURCE_ADDRESS, CanonicalVector.SALT)
        == CanonicalVector.SOURCE_ID
    )
    assert compute_entry_hash(reference) == CanonicalVector.ENTRY_HASH
    assert (
        compute_storage_key(CanonicalVector.SOURCE_ID, CanonicalVector.SLOT)
        == CanonicalVector.STORAGE_KEY
    )

    pre[Spec.RECENT_ROOT_ADDRESS] = Account(
        nonce=1,
        storage={
            as_int(CanonicalVector.STORAGE_KEY): as_int(
                CanonicalVector.ENTRY_HASH
            )
        },
    )

    tx = Transaction(
        sender=sender,
        frames=frames,
        recent_root_references=[reference],
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: EXECUTED_VALUE})},
    )


def test_one_address_addresses_many_root_sources(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    frames: List[Frame],
    target: Address,
    source_address: Address,
    reference_slot: int,
    root: Hash,
) -> None:
    """
    One source address addresses several independent root sources by
    varying the salt.

    `source_id = keccak256(source_address || salt)`, and a root source is
    created implicitly by the first write under a new
    `(source_address, salt)` pair — no registration and no creation
    transaction. So a single address owns as many root sources as it
    picks salts, and the reference check must treat them as unrelated.

    Three references share an address and a slot and differ only in
    their salt. The fill-time assertions record what that has to produce:
    three distinct source identifiers, and therefore three distinct
    storage keys under the recent root contract, even though the ring
    index `slot mod RECENT_ROOT_LENGTH` is the same for all three. An
    implementation that dropped the salt from the derivation would
    collapse all three onto one identifier and one cell, where at most
    one of the three entries can live — so at least two references would
    fail their check and the transaction would be rejected.

    Each reference also names a different root, so the entries differ in
    both of the two derivations rather than only in the storage key.
    """
    salts = [1, 2, 2**256 - 1]
    references = [
        RecentRootReference(
            source_id=compute_source_id(source_address, salt),
            slot=reference_slot,
            root=Hash(as_int(bytes(root)) ^ salt),
        )
        for salt in salts
    ]

    identifiers = {bytes(reference.source_id) for reference in references}
    assert len(identifiers) == len(salts)
    assert len(entries_for(references)) == len(salts)

    seed_entries(pre, references)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=frames,
        recent_root_references=references,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: EXECUTED_VALUE})},
    )
