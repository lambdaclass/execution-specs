"""
Introspection tests for
[EIP-8272: Recent Roots for Frame Transactions](https://eips.ethereum.org/EIPS/eip-8272).

`RECENTROOTREFLOAD` and `TXPARAM(0x0F)` are the only way validation code
reaches a checked reference. Both read the signed envelope, never the
recent root contract's storage.
"""  # noqa: E501

from typing import List

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Bytecode,
    CodeGasMeasure,
    EIPChecklist,
    Environment,
    Fork,
    Hash,
    Op,
    RecentRootReference,
    StateTestFiller,
    Storage,
    Transaction,
)

from ..eip8141_frame_transactions.spec import Spec as Spec8141
from .helpers import (
    CANARY_VALUE,
    EXECUTED_VALUE,
    SLOT_EXECUTED,
    TXPARAM_GAS,
    as_int,
    compute_source_id,
    entries_for,
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

SLOT_CALL_STATUS = 0x20
"""Slot holding the status of a call into a probe that may halt."""

SLOT_MEASURED_GAS = 0x21
"""Slot holding a measured opcode cost."""

HALT_PROBE_GAS = 100_000
"""Gas forwarded to a probe expected to consume all of it and halt."""

FULL_LIST_PROBE_GAS = 10_000_000
"""
Gas limit of the frame that reads every field of a full reference list.

Forty-nine cold stores need more than the frame gas limit the rest of
this file uses; a frame that runs out rolls its writes back, which is
indistinguishable from a probe that never ran.
"""


def three_references(
    source_id: Hash, current_slot: int
) -> List[RecentRootReference]:
    """
    Return three references that differ in every field.

    No two of the nine field values collide, so a fixture reading
    `(index, field)` pairs cannot pass with the operands, the indices or
    the selectors transposed.
    """
    return [
        RecentRootReference(
            source_id=source_id,
            slot=current_slot - 1,
            root=Hash(
                0x1111111111111111111111111111111111111111111111111111111111111111
            ),
        ),
        RecentRootReference(
            source_id=compute_source_id(CanonicalVector.SOURCE_ADDRESS, 2),
            slot=current_slot - 2,
            root=Hash(
                0x2222222222222222222222222222222222222222222222222222222222222222
            ),
        ),
        RecentRootReference(
            source_id=compute_source_id(CanonicalVector.SOURCE_ADDRESS, 3),
            slot=current_slot - 3,
            root=Hash(
                0x3333333333333333333333333333333333333333333333333333333333333333
            ),
        ),
    ]


def maximal_references(current_slot: int) -> List[RecentRootReference]:
    """
    Return `MAX_RECENT_ROOT_REFERENCES` references, no two of which agree
    on any field.

    Each source identifier is taken under a different salt, each slot is
    a different offset back from the current one, and each root is a
    different byte repeated thirty-two times. So the forty-eight words a
    full read produces are forty-eight distinct values, and an index that
    resolves to the wrong reference cannot coincide with the value
    expected of it.
    """
    return [
        RecentRootReference(
            source_id=compute_source_id(
                CanonicalVector.SOURCE_ADDRESS, index + 1
            ),
            slot=current_slot - 1 - index,
            root=Hash(bytes([index + 1]) * 32),
        )
        for index in range(Spec.MAX_RECENT_ROOT_REFERENCES)
    ]


def frame_transaction(
    sender: Address,
    target: Address,
    references: List[RecentRootReference],
) -> Transaction:
    """Return a frame transaction whose `SENDER` frame calls `target`."""
    return Transaction(
        sender=sender,
        frames=verify_and_sender_frames(target),
        recent_root_references=references,
    )


@EIPChecklist.TransactionType.Test.TxScopedAttributes.Read()
@EIPChecklist.Opcode.Test.OutOfBounds.Verify.Max()
def test_recentrootrefload_reads_every_field(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    source_id: Hash,
    current_slot: int,
) -> None:
    """
    Read all three fields of all three declared references.

    Every expected word is the envelope value the transaction was built
    with, so the fixture compares the opcode's output against the
    transaction's own input rather than against anything read from state.
    Field `1` is asserted as a full 256-bit word, which catches a slot
    that is left-aligned instead of zero-extended.

    The three references differ in source, slot and root at once, so an
    implementation returning the wrong index or the wrong selector
    produces a value that appears nowhere in the expected storage.
    """
    references = three_references(source_id, current_slot)
    seed_entries(pre, references)

    storage = Storage()
    probe_code = Bytecode()
    for index, reference in enumerate(references):
        probe_code += Op.SSTORE(
            storage.store_next(as_int(reference.source_id)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_SOURCE_ID, index=index),
        )
        probe_code += Op.SSTORE(
            storage.store_next(int(reference.slot)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_SLOT, index=index),
        )
        probe_code += Op.SSTORE(
            storage.store_next(as_int(reference.root)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=index),
        )
    probe_code += Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
    probe_code += Op.STOP

    prober = pre.deploy_contract(code=probe_code)
    tx = frame_transaction(pre.fund_eoa(), prober, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )


def test_recentrootrefload_reads_the_whole_reference_list(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    current_slot: int,
) -> None:
    """
    Read all three fields of all sixteen references a transaction may
    declare, index by index.

    `RECENTROOTREFLOAD` indexes `recent_root_references` directly, so
    every declared position must be reachable and must answer for itself.
    Reading only the first few positions leaves the rest of the list
    unpinned: an implementation that resolved every index past some
    small bound onto a single reference — the first one, say — returns
    correct words for the positions a short read visits and wrong words
    for the others, and only a read that reaches the top of the list
    separates the two. This fixture reaches index
    `MAX_RECENT_ROOT_REFERENCES - 1`, the highest index any transaction
    can make valid.

    Every expected word is the value the transaction was built with, and
    the sixteen references agree on no field, so each of the
    forty-eight stored words identifies one `(index, field)` pair
    uniquely. Slots are asserted as full 256-bit words, which keeps the
    zero extension of the 64-bit field pinned across the whole list
    rather than at index zero alone.
    """
    references = maximal_references(current_slot)
    assert len(references) == Spec.MAX_RECENT_ROOT_REFERENCES
    assert len({bytes(reference.root) for reference in references}) == len(
        references
    )
    seed_entries(pre, references)

    storage = Storage()
    probe_code = Bytecode()
    for index, reference in enumerate(references):
        probe_code += Op.SSTORE(
            storage.store_next(as_int(reference.source_id)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_SOURCE_ID, index=index),
        )
        probe_code += Op.SSTORE(
            storage.store_next(int(reference.slot)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_SLOT, index=index),
        )
        probe_code += Op.SSTORE(
            storage.store_next(as_int(reference.root)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=index),
        )
    probe_code += Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
    probe_code += Op.STOP

    prober = pre.deploy_contract(code=probe_code)
    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(prober, FULL_LIST_PROBE_GAS),
        recent_root_references=references,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )


def test_recentrootrefload_stack_order(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    source_id: Hash,
    current_slot: int,
) -> None:
    """
    `field` is the top stack item and `index` is the one below it.

    Both probes use `field != index`, and the two operand pairs are
    mirror images: `(index=0, field=2)` must return the first
    reference's root and `(index=2, field=0)` the third reference's
    source identifier. An implementation with the operands transposed
    swaps the two answers, and both stored words are wrong at once —
    whereas any probe with `field == index` would pass either way.
    """
    references = three_references(source_id, current_slot)
    seed_entries(pre, references)

    storage = Storage()
    probe_code = (
        Op.SSTORE(
            storage.store_next(as_int(references[0].root)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=0),
        )
        + Op.SSTORE(
            storage.store_next(as_int(references[2].source_id)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_SOURCE_ID, index=2),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    prober = pre.deploy_contract(code=probe_code)
    tx = frame_transaction(pre.fund_eoa(), prober, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )


@pytest.mark.parametrize("current_slot", [Spec.MAX_SLOT])
def test_recentrootrefload_slot_zero_extension(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    source_id: Hash,
    root: Hash,
    current_slot: int,
) -> None:
    """
    A slot is pushed zero-extended into a 256-bit word.

    The reference names the largest slot that is both representable and
    referenceable — one below the block's own slot, itself the maximum a
    `uint64` holds — so the expected word is
    `0x0000...0000FFFFFFFFFFFFFFFE`. Asserting the whole word rejects a
    left-aligned or high-byte-packed encoding, which would agree with
    this fixture for no slot value at all.
    """
    reference_slot = current_slot - 1
    assert reference_slot == 2**64 - 2
    references = [
        RecentRootReference(
            source_id=source_id, slot=reference_slot, root=root
        )
    ]
    seed_entries(pre, references)

    storage = Storage()
    probe_code = (
        Op.SSTORE(
            storage.store_next(reference_slot),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_SLOT, index=0),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    prober = pre.deploy_contract(code=probe_code)
    tx = frame_transaction(pre.fund_eoa(), prober, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )


@EIPChecklist.Opcode.Test.ExecutionContext.Call()
@EIPChecklist.Opcode.Test.ExceptionalAbort()
@EIPChecklist.Opcode.Test.OutOfBounds.Verify.MaxPlusOne()
@pytest.mark.parametrize(
    "declared_count,index,field",
    [
        pytest.param(1, 1, Spec.FIELD_SOURCE_ID, id="index_equals_length"),
        pytest.param(1, 2**64, Spec.FIELD_SOURCE_ID, id="index_huge"),
        pytest.param(1, 2**256 - 1, Spec.FIELD_SOURCE_ID, id="index_max_word"),
        pytest.param(
            1, 0, Spec.FIRST_UNDEFINED_FIELD, id="field_just_undefined"
        ),
        pytest.param(1, 0, 2**256 - 1, id="field_max_word"),
        pytest.param(0, 0, Spec.FIELD_SOURCE_ID, id="empty_reference_list"),
    ],
)
def test_recentrootrefload_out_of_range_operands_halt(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    references: List[RecentRootReference],
    declared_count: int,
    index: int,
    field: int,
) -> None:
    """
    An index past the end of the list, or a field above `2`, halts
    exceptionally rather than returning zero.

    The probe is called from a caller that records the call's status and
    then keeps going, so an exceptional halt shows up as a zero status
    with the caller's own sentinel still written — a `REVERT` would look
    the same from here, but the probe contains nothing that could revert.

    The empty-list arm is the minimal out-of-range read and catches an
    implementation that special-cases an empty list to push zero.
    """
    declared = references[:declared_count]
    seed_entries(pre, declared)

    probe = pre.deploy_contract(
        code=Op.RECENTROOTREFLOAD(field=field, index=index) + Op.STOP
    )
    storage = Storage()
    caller = pre.deploy_contract(
        code=Op.SSTORE(
            storage.store_next(0),
            Op.CALL(gas=HALT_PROBE_GAS, address=probe),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    tx = frame_transaction(pre.fund_eoa(), caller, declared)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={caller: Account(storage=storage)},
    )


@EIPChecklist.Opcode.Test.StackUnderflow()
@pytest.mark.parametrize(
    "stack_height",
    [
        pytest.param(0, id="empty_stack"),
        pytest.param(1, id="one_operand"),
    ],
)
def test_recentrootrefload_stack_underflow(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    references: List[RecentRootReference],
    stack_height: int,
) -> None:
    """
    `RECENTROOTREFLOAD` halts when fewer than its two operands are on the
    stack.

    It pops `field` and then `index`, so both a bare instruction and one
    preceded by a single push are underflows; one operand short is the
    boundary, and it is the arm an implementation that read `index` from
    a default would pass. The reference the transaction declares is valid
    and correctly seeded, so index 0 exists and nothing but the stack
    height is wrong.

    Observed the same way as the out-of-range reads above: the caller
    records the call's status, which stays zero, and then writes its own
    sentinel to show it survived.
    """
    seed_entries(pre, references)

    probe = pre.deploy_contract(
        code=(Op.PUSH1[0] * stack_height) + Op.RECENTROOTREFLOAD + Op.STOP
    )
    storage = Storage()
    caller = pre.deploy_contract(
        code=Op.SSTORE(
            storage.store_next(0),
            Op.CALL(gas=HALT_PROBE_GAS, address=probe),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    tx = frame_transaction(pre.fund_eoa(), caller, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={caller: Account(storage=storage)},
    )


@EIPChecklist.Opcode.Test.GasUsage.Normal()
def test_recentrootrefload_gas(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    references: List[RecentRootReference],
) -> None:
    """
    `RECENTROOTREFLOAD` costs `RECENTROOTREFLOAD_GAS`.

    Measured around the bare opcode, with the two operands pushed before
    the measurement starts so only the opcode itself is inside it. The
    expected value is the fork's cost for this opcode, which EIP-8272
    fixes at three — the same as the other very-low instructions, so the
    new opcode carries no bespoke price.
    """
    seed_entries(pre, references)
    assert Op.RECENTROOTREFLOAD.gas_cost(fork) == Spec.RECENTROOTREFLOAD_GAS

    storage = Storage()
    probe_code = (
        CodeGasMeasure(
            code=Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=0),
            # The two operand pushes sit inside the measurement, so
            # their cost is subtracted back out and what remains is the
            # opcode's own charge.
            overhead_cost=2 * Op.PUSH1.gas_cost(fork),
            extra_stack_items=1,
            sstore_key=SLOT_MEASURED_GAS,
        )
        + Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE)
        + Op.STOP
    )
    storage[SLOT_MEASURED_GAS] = Op.RECENTROOTREFLOAD.gas_cost(fork)
    storage[SLOT_EXECUTED] = EXECUTED_VALUE

    prober = pre.deploy_contract(
        code=probe_code, storage={SLOT_EXECUTED: CANARY_VALUE}
    )
    tx = frame_transaction(pre.fund_eoa(), prober, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )


@EIPChecklist.Opcode.Test.GasUsage.OutOfGasExecution()
@EIPChecklist.GasCostChanges.Test.OutOfGas()
@pytest.mark.parametrize(
    "available_gas_delta",
    [
        pytest.param(0, id="exact_gas"),
        pytest.param(-1, id="one_gas_short"),
    ],
)
def test_recentrootrefload_out_of_gas(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    references: List[RecentRootReference],
    available_gas_delta: int,
) -> None:
    """
    The opcode succeeds with exactly its cost available and halts with
    one gas less.

    The flip between the two arms is the second, independent statement of
    `RECENTROOTREFLOAD_GAS`: the measurement above says what is charged,
    and this says the charge is enforced.
    """
    seed_entries(pre, references)
    probe = pre.deploy_contract(
        code=Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=0) + Op.STOP
    )
    # The operand pushes and the trailing STOP are paid by the callee
    # too, so the forwarded gas covers them before the opcode runs.
    probe_overhead = Op.PUSH1.gas_cost(fork) * 2 + Op.STOP.gas_cost(fork)
    forwarded_gas = (
        probe_overhead
        + Op.RECENTROOTREFLOAD.gas_cost(fork)
        + available_gas_delta
    )

    storage = Storage()
    caller = pre.deploy_contract(
        code=Op.SSTORE(
            storage.store_next(1 if available_gas_delta == 0 else 0),
            Op.CALL(gas=forwarded_gas, address=probe),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    tx = frame_transaction(pre.fund_eoa(), caller, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={caller: Account(storage=storage)},
    )


def test_recentrootrefload_reads_envelope_not_storage(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    references: List[RecentRootReference],
) -> None:
    """
    The opcode returns envelope values even when the contract's storage
    holds something else at the reference's own key.

    The recent root contract stores an entry hash, which by construction
    equals none of the three fields it commits to, so an implementation
    consulting storage returns the entry hash for at least one field.
    A second reference is declared whose cell is left empty — the
    transaction is invalid without it being seeded, so instead the
    reference is a duplicate of the first, which shares its cell and
    therefore reads back the same envelope values from a single stored
    word.
    """
    duplicated = references * 2
    seed_entries(pre, duplicated)
    reference = duplicated[0]

    storage = Storage()
    probe_code = Bytecode()
    for index in range(len(duplicated)):
        probe_code += Op.SSTORE(
            storage.store_next(as_int(reference.source_id)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_SOURCE_ID, index=index),
        )
        probe_code += Op.SSTORE(
            storage.store_next(int(reference.slot)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_SLOT, index=index),
        )
        probe_code += Op.SSTORE(
            storage.store_next(as_int(reference.root)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=index),
        )
    probe_code += Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
    probe_code += Op.STOP

    prober = pre.deploy_contract(code=probe_code)
    tx = frame_transaction(pre.fund_eoa(), prober, duplicated)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )


def test_recentrootrefload_does_not_displace_sigparam(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    references: List[RecentRootReference],
) -> None:
    """
    `0xB5` is the new opcode and `0xB4` is still EIP-8141's `SIGPARAM`.

    Both run in one contract in one transaction and return their own,
    unrelated values: the reference's root, and the resolved signer of
    the transaction's only signature entry. EIP-8272 originally claimed
    `0xB4`; this pins that it took the free byte next to it instead of
    overwriting its neighbour.
    """
    seed_entries(pre, references)
    assert Op.RECENTROOTREFLOAD.int() == Spec.RECENTROOTREFLOAD
    assert Op.SIGPARAM.int() == Spec.RECENTROOTREFLOAD - 1

    sender = pre.fund_eoa()
    storage = Storage()
    probe_code = (
        Op.SSTORE(
            storage.store_next(as_int(references[0].root)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=0),
        )
        + Op.SSTORE(
            storage.store_next(as_int(sender)),
            Op.SIGPARAM(signature_index=0, param=0),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    prober = pre.deploy_contract(code=probe_code)
    tx = frame_transaction(sender, prober, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )


@pytest.mark.parametrize(
    "reference_count",
    [0, 1, 2, Spec.MAX_RECENT_ROOT_REFERENCES],
    ids=lambda count: f"references_{count}",
)
def test_txparam_reference_count(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    reference_count: int,
) -> None:
    """
    `TXPARAM(0x0F)` returns the number of declared references.

    Swept from the empty list to the cap. The zero arm stores the count
    incremented by one rather than the count itself, so "returned zero"
    stays distinguishable from "never executed".
    """
    references = [
        RecentRootReference(
            source_id=source_id, slot=current_slot - 1 - index, root=root
        )
        for index in range(reference_count)
    ]
    seed_entries(pre, references)

    storage = Storage()
    probe_code = (
        Op.SSTORE(
            storage.store_next(reference_count + 1),
            Op.ADD(
                Op.TXPARAM(param=Spec.TXPARAM_RECENT_ROOT_REFERENCE_COUNT),
                1,
            ),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    prober = pre.deploy_contract(code=probe_code)
    tx = frame_transaction(pre.fund_eoa(), prober, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )


def test_txparam_counts_declared_not_distinct_references(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    source_id: Hash,
    root: Hash,
    current_slot: int,
) -> None:
    """
    `TXPARAM(0x0F)` returns `len(recent_root_references)`, which counts
    repeated references once each time they are declared.

    Duplicate references are valid, and the specification keeps them
    "checked, charged, and preserved independently" — so the list a
    transaction carries is the list it declared, not the set of distinct
    references in it. Every other count fixture declares references that
    differ from one another, where the number declared and the number
    distinct are the same number; a count taken over a deduplicated list
    passes all of them.

    Here the two are pulled apart. Sixteen references are declared and
    two are distinct, so `16` is the answer and `2` is the answer an
    implementation that counted distinct references would give. Both are
    non-zero, which keeps the arm that never executed distinguishable
    from either.

    The reads pin the other half of the same clause: the duplicates
    occupy their declared positions, so index `7` still resolves to the
    first reference and index `15` to the second. A list collapsed to
    its two distinct entries has no index `7` at all, and the read halts
    instead of answering.
    """
    first = RecentRootReference(
        source_id=source_id, slot=current_slot - 1, root=root
    )
    second = RecentRootReference(
        source_id=compute_source_id(CanonicalVector.SOURCE_ADDRESS, 2),
        slot=current_slot - 2,
        root=Hash(bytes([0x5A] * 32)),
    )
    half = Spec.MAX_RECENT_ROOT_REFERENCES // 2
    references = [first] * half + [second] * half

    assert len(references) == Spec.MAX_RECENT_ROOT_REFERENCES
    assert len(entries_for(references)) == 2
    seed_entries(pre, references)

    storage = Storage()
    probe_code = (
        Op.SSTORE(
            storage.store_next(Spec.MAX_RECENT_ROOT_REFERENCES),
            Op.TXPARAM(param=Spec.TXPARAM_RECENT_ROOT_REFERENCE_COUNT),
        )
        + Op.SSTORE(
            storage.store_next(as_int(first.root)),
            Op.RECENTROOTREFLOAD(field=Spec.FIELD_ROOT, index=half - 1),
        )
        + Op.SSTORE(
            storage.store_next(as_int(second.root)),
            Op.RECENTROOTREFLOAD(
                field=Spec.FIELD_ROOT,
                index=Spec.MAX_RECENT_ROOT_REFERENCES - 1,
            ),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    prober = pre.deploy_contract(code=probe_code)
    tx = frame_transaction(pre.fund_eoa(), prober, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )


@pytest.mark.parametrize(
    "param,halts",
    [
        pytest.param(0x0C, True, id="param_0x0c"),
        pytest.param(0x0D, True, id="param_0x0d"),
        pytest.param(0x0E, True, id="param_0x0e"),
        pytest.param(
            Spec.TXPARAM_RECENT_ROOT_REFERENCE_COUNT, False, id="param_0x0f"
        ),
        pytest.param(0x10, True, id="param_0x10"),
    ],
)
def test_txparam_opens_exactly_one_index(
    state_test: StateTestFiller,
    pre: Alloc,
    env: Environment,
    references: List[RecentRootReference],
    param: int,
    halts: bool,
) -> None:
    """
    EIP-8272 defines `TXPARAM` index `0x0F` and leaves its neighbours
    undefined.

    The three indices below it and the one above it must still
    exceptionally halt; only `0x0F` returns. Without the halting arms,
    the returning arm alone would also pass against an implementation
    that answers every index.
    """
    seed_entries(pre, references)

    probe = pre.deploy_contract(code=Op.TXPARAM(param=param) + Op.STOP)
    storage = Storage()
    caller = pre.deploy_contract(
        code=Op.SSTORE(
            storage.store_next(0 if halts else 1),
            Op.CALL(gas=HALT_PROBE_GAS, address=probe),
        )
        + Op.SSTORE(storage.store_next(EXECUTED_VALUE), EXECUTED_VALUE)
        + Op.STOP
    )
    tx = frame_transaction(pre.fund_eoa(), caller, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={caller: Account(storage=storage)},
    )


SLOT_BASELINE_MEASURED_GAS = 0x22
"""Slot holding the measured cost of a `TXPARAM` index EIP-8141 defines."""


@EIPChecklist.GasCostChanges.Test.GasUpdatesMeasurement()
def test_txparam_reference_count_gas(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    references: List[RecentRootReference],
) -> None:
    """
    `TXPARAM(TXPARAM_RECENT_ROOT_REFERENCE_COUNT)` costs the standard
    `TXPARAM` gas.

    EIP-8272 gives its new index no price of its own — it says only that
    the index costs what `TXPARAM` costs — so the expectation here is an
    equality between two measurements rather than a number this EIP
    defines. The new index `0x0F` and EIP-8141's own `0x09` are measured
    the same way in the same transaction, and both must land on the same
    value. An implementation that gave the new index a bespoke price,
    cheap or dear, separates the two.

    That value is also asserted against `helpers.TXPARAM_GAS`,
    transcribed from EIP-8141's opcode table, so the pair cannot agree
    on a wrong number. The framework has no gas entry for `TXPARAM`,
    which is why the literal is carried in this suite rather than read
    from the fork.

    Each measurement puts its operand push inside the measured region
    and subtracts one `PUSH1` back out, leaving the opcode's own charge.
    """
    seed_entries(pre, references)

    storage = Storage()
    probe_code = (
        CodeGasMeasure(
            code=Op.TXPARAM(param=Spec.TXPARAM_RECENT_ROOT_REFERENCE_COUNT),
            overhead_cost=Op.PUSH1.gas_cost(fork),
            extra_stack_items=1,
            sstore_key=SLOT_MEASURED_GAS,
        )
        + CodeGasMeasure(
            code=Op.TXPARAM(param=Spec8141.TXPARAM_FRAME_COUNT),
            overhead_cost=Op.PUSH1.gas_cost(fork),
            extra_stack_items=1,
            sstore_key=SLOT_BASELINE_MEASURED_GAS,
        )
        + Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE)
        + Op.STOP
    )
    storage[SLOT_MEASURED_GAS] = TXPARAM_GAS
    storage[SLOT_BASELINE_MEASURED_GAS] = TXPARAM_GAS
    storage[SLOT_EXECUTED] = EXECUTED_VALUE

    prober = pre.deploy_contract(
        code=probe_code, storage={SLOT_EXECUTED: CANARY_VALUE}
    )
    tx = frame_transaction(pre.fund_eoa(), prober, references)

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={prober: Account(storage=storage)},
    )
