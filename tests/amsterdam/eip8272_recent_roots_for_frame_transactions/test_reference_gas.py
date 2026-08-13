"""
Gas accounting tests for
[EIP-8272: Recent Roots for Frame Transactions](https://eips.ethereum.org/EIPS/eip-8272).

Declaring references costs a transaction two things: the calldata cost of
the encoded reference list, and a per-reference charge preceded once by
an address charge. It also buys the transaction something: the recent
root contract and every declared storage key start out warm.
"""  # noqa: E501

from typing import List

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Block,
    BlockchainTestFiller,
    Bytecode,
    CodeGasMeasure,
    EIPChecklist,
    Environment,
    Fork,
    Frame,
    FrameReceipt,
    Hash,
    Op,
    RecentRootReference,
    StateTestFiller,
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
    compute_storage_key,
    entries_for,
    rlp_encode_references,
    seed_entries,
    signature_free_calldata_floor_gas,
    signature_free_intrinsic_gas,
    verify_and_sender_frames,
)
from .spec import Spec, ref_spec_8272

REFERENCE_SPEC_GIT_PATH = ref_spec_8272.git_path
REFERENCE_SPEC_VERSION = ref_spec_8272.version

pytestmark = [
    pytest.mark.valid_from("Bogota"),
    pytest.mark.pre_alloc_mutable,
]

SLOT_RECENT_ROOT_ACCESS_GAS = 0x10
"""Slot holding the measured cost of touching the recent root contract."""

SLOT_NEIGHBOUR_ACCESS_GAS = 0x11
"""Slot holding the measured cost of touching an adjacent address."""

SLOT_DECLARED_KEY_GAS = 0x12
"""Slot holding the measured cost of loading a declared storage key."""

SLOT_UNDECLARED_KEY_GAS = 0x13
"""Slot holding the measured cost of loading an undeclared storage key."""

SLOT_SECOND_FRAME_ACCESS_GAS = 0x14
"""Slot holding a measurement taken after an intervening frame reverted."""

SLOT_FULL_LIST_KEY_GAS_BASE = 0x40
"""
First of `MAX_RECENT_ROOT_REFERENCES` consecutive slots, one per declared
key of a full reference list, each holding that key's measured cost.
"""

NEIGHBOUR_ADDRESS = Address(0x8271)
"""
An address adjacent to the recent root contract's.

Nothing warms it, so it is the control that proves a measured warm cost
came from the reference and not from some blanket pre-warming.
"""


def approving_sender(pre: Alloc) -> Address:
    """
    Deploy a contract sender that approves execution and payment.

    A contract sender authorizes through its own code, so the
    transaction carries no signature entry — which removes the signature
    verification gas and the signature byte fields from the intrinsic
    cost, leaving a quantity the test can state in full.
    """
    return pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec8141.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )


@EIPChecklist.TransactionType.Test.BlockInteractions.Eip7825.Valid()
@EIPChecklist.TransactionType.Test.BlockInteractions.Eip7825.Invalid()
@pytest.mark.parametrize(
    "at_cap",
    [
        pytest.param(True, id="derived_limit_at_cap"),
        pytest.param(
            False,
            id="derived_limit_over_cap",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@pytest.mark.parametrize(
    "reference_count",
    [0, 1, 2, Spec.MAX_RECENT_ROOT_REFERENCES],
    ids=lambda count: f"references_{count}",
)
@pytest.mark.parametrize(
    "zero_valued",
    [
        pytest.param(False, id="nonzero_bytes"),
        pytest.param(True, id="zero_heavy_bytes"),
    ],
)
def test_reference_intrinsic_gas_boundary(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    target: Address,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    reference_count: int,
    zero_valued: bool,
    at_cap: bool,
) -> None:
    """
    Pin the transaction's whole intrinsic cost, at four reference counts
    and two byte profiles, through the EIP-7825 cap.

    A frame transaction has no gas limit field: its limit is derived as
    the intrinsic cost plus the sum of the frame gas limits, and the
    transaction is invalid once that derived limit exceeds the cap. So
    setting the frame gas limits to `cap - intrinsic` accepts, and one
    gas more rejects — which pins `intrinsic` to a single value rather
    than to a range.

    The expected intrinsic is computed in `helpers.py` from the two EIPs'
    prose: EIP-8141's base and per-frame costs, plus EIP-8272's
    `recent_root_calldata_cost` over the re-encoded reference list and
    its `recent_root_reference_intrinsic_gas`. Three of EIP-8272's
    clauses fall out of the sweep:

    - `reference_count == 0` still pays a calldata term, because
      `rlp([])` is the single byte `0xc0` — the zero arm's accepted limit
      is short by that much and by nothing else;
    - the address charge appears once, not once per reference: the step
      from one reference to two is a per-reference charge alone;
    - the calldata term counts tokens, not bytes: the zero-heavy profile
      is charged roughly a third of the non-zero profile for a list of
      almost the same length.
    """
    sender = approving_sender(pre)
    references = [
        RecentRootReference(
            source_id=Hash(0) if zero_valued else source_id,
            slot=current_slot - 1 - index,
            root=Hash(0) if zero_valued else root,
        )
        for index in range(reference_count)
    ]
    seed_entries(pre, references)

    intrinsic = signature_free_intrinsic_gas(
        fork, frame_count=2, references=references
    )
    gas_limit_cap = fork.transaction_gas_limit_cap()
    assert gas_limit_cap is not None
    total_frame_gas = gas_limit_cap - intrinsic + (0 if at_cap else 1)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=verify_and_sender_frames(
            target, total_frame_gas - VERIFY_FRAME_GAS
        ),
        recent_root_references=references,
        error=(
            None if at_cap else TransactionException.GAS_LIMIT_EXCEEDS_MAXIMUM
        ),
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            target: Account(
                storage={
                    SLOT_EXECUTED: (EXECUTED_VALUE if at_cap else CANARY_VALUE)
                }
            )
        },
    )


@EIPChecklist.TransactionType.Test.Encoding.ListField.Zero()
def test_zero_references_are_charged_for_the_empty_list(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    target: Address,
) -> None:
    """
    A transaction declaring no references still pays for the encoded
    empty list.

    `recent_root_calldata_cost` is defined over `rlp(recent_root_
    references)`, and the RLP of an empty list is the single non-zero
    byte `0xc0` — four tokens, never zero bytes. The assertion below
    fixes that encoding, and the cap boundary then charges for it: an
    implementation that skips the calldata term when the list is empty
    accepts a derived limit this test rejects.
    """
    encoding = rlp_encode_references([])
    assert encoding == b"\xc0"

    sender = approving_sender(pre)
    intrinsic = signature_free_intrinsic_gas(
        fork, frame_count=2, references=[]
    )
    gas_limit_cap = fork.transaction_gas_limit_cap()
    assert gas_limit_cap is not None

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=verify_and_sender_frames(
            target, gas_limit_cap - intrinsic - VERIFY_FRAME_GAS
        ),
        recent_root_references=[],
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: EXECUTED_VALUE})},
    )


@pytest.mark.parametrize(
    "at_cap",
    [
        pytest.param(True, id="derived_limit_at_cap"),
        pytest.param(
            False,
            id="derived_limit_over_cap",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_duplicate_references_are_charged_independently(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    target: Address,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    at_cap: bool,
) -> None:
    """
    Sixteen byte-identical references pay sixteen per-reference charges.

    Duplicates are the one case where the two sets EIP-8272 maintains
    disagree. The accessed storage-key set deduplicates normally, so
    sixteen identical references name one key and warm one key; the
    charge does not follow it, because duplicate references are
    "checked, charged, and preserved independently". So the per-reference
    term is counted over the declared list and not over the keys it
    resolves to.

    Every other fixture that pins a reference charge declares references
    that differ, where the two counts coincide and a charge taken over
    either one lands on the same number. Here they differ by fifteen.
    The expected intrinsic is the same quantity `helpers.py` derives from
    the two EIPs' prose — EIP-8141's base and per-frame costs, plus
    EIP-8272's calldata cost over the encoded list and one address charge
    followed by sixteen per-reference charges — and the EIP-7825 cap
    turns it into an accept/reject pair: a frame transaction's gas limit
    is derived rather than declared, so frame limits summing to
    `cap - intrinsic` are valid and one gas more is not.

    An implementation charging per distinct reference is short by fifteen
    times `RECENT_ROOT_REFERENCE_GAS`, which leaves room under the cap
    for the arm that must be rejected, and that arm fails. One charging
    per warmed key rather than per reference fails the same way.
    """
    sender = approving_sender(pre)
    reference = RecentRootReference(
        source_id=source_id, slot=current_slot - 1, root=root
    )
    references = [reference] * Spec.MAX_RECENT_ROOT_REFERENCES
    assert len(entries_for(references)) == 1
    seed_entries(pre, references)

    intrinsic = signature_free_intrinsic_gas(
        fork, frame_count=2, references=references
    )
    gas_limit_cap = fork.transaction_gas_limit_cap()
    assert gas_limit_cap is not None
    total_frame_gas = gas_limit_cap - intrinsic + (0 if at_cap else 1)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=verify_and_sender_frames(
            target, total_frame_gas - VERIFY_FRAME_GAS
        ),
        recent_root_references=references,
        error=(
            None if at_cap else TransactionException.GAS_LIMIT_EXCEEDS_MAXIMUM
        ),
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            target: Account(
                storage={
                    SLOT_EXECUTED: (EXECUTED_VALUE if at_cap else CANARY_VALUE)
                }
            )
        },
    )


@EIPChecklist.TransactionType.Test.IntrinsicValidity.DataFloorAboveIntrinsicGasCost()
@EIPChecklist.GasCostChanges.Test.GasUpdatesMeasurement()
@pytest.mark.parametrize(
    "reference_count",
    [Spec.MAX_RECENT_ROOT_REFERENCES // 2, Spec.MAX_RECENT_ROOT_REFERENCES],
    ids=lambda count: f"references_{count}",
)
@pytest.mark.parametrize(
    "zero_valued",
    [
        pytest.param(False, id="nonzero_bytes"),
        pytest.param(True, id="zero_heavy_bytes"),
    ],
)
def test_reference_calldata_floor_is_charged(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    target: Address,
    source_id: Hash,
    root: Hash,
    current_slot: int,
    reference_count: int,
    zero_valued: bool,
) -> None:
    """
    Pin the gas a reference-carrying transaction pays when the calldata
    floor is what binds.

    EIP-8272 adds to the floor twice — `recent_root_reference_intrinsic_
    gas` joins the costs the floor is anchored on, and the encoded
    reference list joins the bytes the floor is counted over — and
    neither addition is observable through a transaction's accept/reject
    outcome, because a frame transaction's derived gas limit is built
    from the *standard* branch alone. They are observable in the gas the
    transaction is charged, which is what this test asserts.

    The expected value is `helpers.signature_free_calldata_floor_gas`,
    transcribed from EIP-8141's base and per-frame costs, EIP-8272's two
    floor additions and EIP-7976's uniform byte rate; the fill-time
    assertion below records that it exceeds the standard branch, which is
    what makes it the quantity the transaction pays. With sixteen
    non-zero references the two branches are 139,778 and 85,106 gas, so
    the frames have more than fifty thousand gas of room before the floor
    stops binding — the transaction's own work never reaches it.

    Both byte profiles of a given count expect the *same* number while
    their standard costs do not, because the floor counts bytes and the
    standard branch counts tokens: at sixteen references the two
    profiles encode to the same 1,139 bytes but to 4,556 and 1,484
    tokens, a threefold spread that carries the standard totals to
    85,106 and 72,818. An implementation that carried the token count
    into the floor, or left the reference bytes out of it, reports a
    smaller number for the zero-heavy arm and for both arms
    respectively.
    """
    sender = approving_sender(pre)
    references = [
        RecentRootReference(
            source_id=Hash(0) if zero_valued else source_id,
            slot=current_slot - 1 - index,
            root=Hash(0) if zero_valued else root,
        )
        for index in range(reference_count)
    ]
    seed_entries(pre, references)

    calldata_floor = signature_free_calldata_floor_gas(
        fork, frame_count=2, references=references
    )
    standard_intrinsic = signature_free_intrinsic_gas(
        fork, frame_count=2, references=references
    )
    assert calldata_floor > standard_intrinsic

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=verify_and_sender_frames(target),
        recent_root_references=references,
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=calldata_floor
        ),
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(storage={SLOT_EXECUTED: EXECUTED_VALUE})},
    )


@pytest.mark.parametrize(
    "declares_reference",
    [
        pytest.param(True, id="with_reference"),
        pytest.param(False, id="without_reference"),
    ],
)
def test_reference_warms_recent_root_address(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    references: List[RecentRootReference],
    declares_reference: bool,
) -> None:
    """
    A declared reference adds the recent root contract to the accessed
    address set before any frame runs.

    Measured with a `BALANCE` probe inside the transaction's `SENDER`
    frame: with a reference declared the probe pays the warm price, and
    without one it pays the cold price. Both expectations come from the
    same bytecode's own cost under the fork's schedule, so a repricing
    moves them together.

    An adjacent address is probed in the same frame and must stay cold in
    both arms — the reference warms one account, not a neighbourhood.
    """
    declared = references if declares_reference else []
    seed_entries(pre, declared)

    recent_root_probe = Op.BALANCE(
        address=Spec.RECENT_ROOT_ADDRESS, address_warm=declares_reference
    )
    neighbour_probe = Op.BALANCE(address=NEIGHBOUR_ADDRESS, address_warm=False)
    probe_code = (
        CodeGasMeasure(
            code=recent_root_probe,
            extra_stack_items=1,
            sstore_key=SLOT_RECENT_ROOT_ACCESS_GAS,
        )
        + CodeGasMeasure(
            code=neighbour_probe,
            extra_stack_items=1,
            sstore_key=SLOT_NEIGHBOUR_ACCESS_GAS,
        )
        + Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE)
        + Op.STOP
    )
    prober = pre.deploy_contract(
        code=probe_code, storage={SLOT_EXECUTED: CANARY_VALUE}
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(prober, SENDER_FRAME_GAS),
        recent_root_references=declared,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            prober: Account(
                storage={
                    SLOT_RECENT_ROOT_ACCESS_GAS: recent_root_probe.gas_cost(
                        fork
                    ),
                    SLOT_NEIGHBOUR_ACCESS_GAS: neighbour_probe.gas_cost(fork),
                    SLOT_EXECUTED: EXECUTED_VALUE,
                }
            )
        },
    )


@EIPChecklist.SystemContract.Test.ContractSubstitution()
def test_reference_warms_only_declared_storage_keys(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    references: List[RecentRootReference],
) -> None:
    """
    A declared reference warms its own storage key and no other key of
    the recent root contract.

    The probe runs as the recent root contract itself — substituted into
    the pre-state, since the specification leaves its runtime code
    undefined — so its `SLOAD`s read the very cells the reference check
    reads. The declared key pays the warm price and a sibling key of the
    same account pays the cold price, which is the whole of "This affects
    warm/cold gas accounting only": one key, not the account's storage.
    """
    declared_key = as_int(
        compute_storage_key(references[0].source_id, int(references[0].slot))
    )
    undeclared_key = declared_key ^ 1
    declared_probe = Op.SLOAD(key=declared_key, key_warm=True)
    undeclared_probe = Op.SLOAD(key=undeclared_key, key_warm=False)

    probe_code = (
        CodeGasMeasure(
            code=declared_probe,
            extra_stack_items=1,
            sstore_key=SLOT_DECLARED_KEY_GAS,
        )
        + CodeGasMeasure(
            code=undeclared_probe,
            extra_stack_items=1,
            sstore_key=SLOT_UNDECLARED_KEY_GAS,
        )
        + Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE)
        + Op.STOP
    )
    entries = entries_for(references)
    pre[Spec.RECENT_ROOT_ADDRESS] = Account(
        nonce=1,
        code=probe_code,
        storage={**entries, SLOT_EXECUTED: CANARY_VALUE},
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(
            Spec.RECENT_ROOT_ADDRESS, SENDER_FRAME_GAS
        ),
        recent_root_references=references,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            Spec.RECENT_ROOT_ADDRESS: Account(
                storage={
                    **entries,
                    SLOT_DECLARED_KEY_GAS: declared_probe.gas_cost(fork),
                    SLOT_UNDECLARED_KEY_GAS: undeclared_probe.gas_cost(fork),
                    SLOT_EXECUTED: EXECUTED_VALUE,
                }
            )
        },
    )


def test_every_key_of_a_full_reference_list_is_warm(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    source_id: Hash,
    root: Hash,
    current_slot: int,
) -> None:
    """
    Every one of a full list's sixteen declared keys starts warm, not
    just the first few.

    "For validation checks, each declared reference names exactly one
    storage key" — each, so the warm set a transaction buys has one key
    per declared reference and the last reference pays for its key as
    much as the first does. Every other pre-warming fixture declares one
    reference, which an implementation that warmed only the head of the
    list satisfies. This one measures all sixteen.

    The expectation is the fork's own warm and cold `SLOAD` prices rather
    than a number this EIP defines, because EIP-8272 sets no price here:
    it moves keys into the warm set and lets the existing schedule charge
    for them. The undeclared sibling key is measured in the same probe
    and must stay cold, which is what makes the sixteen warm
    measurements mean "declared" rather than "some blanket warming of
    the account".
    """
    references = [
        RecentRootReference(
            source_id=source_id, slot=current_slot - 1 - index, root=root
        )
        for index in range(Spec.MAX_RECENT_ROOT_REFERENCES)
    ]
    entries = entries_for(references)
    assert len(entries) == Spec.MAX_RECENT_ROOT_REFERENCES

    declared_keys = [
        as_int(compute_storage_key(reference.source_id, int(reference.slot)))
        for reference in references
    ]
    undeclared_key = declared_keys[-1] ^ 1
    warm_probe = Op.SLOAD(key=declared_keys[0], key_warm=True)
    cold_probe = Op.SLOAD(key=undeclared_key, key_warm=False)

    probe_code = Bytecode()
    expected_gas = {}
    for index, key in enumerate(declared_keys):
        measured_slot = SLOT_FULL_LIST_KEY_GAS_BASE + index
        probe_code += CodeGasMeasure(
            code=Op.SLOAD(key=key, key_warm=True),
            extra_stack_items=1,
            sstore_key=measured_slot,
        )
        expected_gas[measured_slot] = warm_probe.gas_cost(fork)
    probe_code += CodeGasMeasure(
        code=cold_probe,
        extra_stack_items=1,
        sstore_key=SLOT_UNDECLARED_KEY_GAS,
    )
    probe_code += Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE) + Op.STOP

    pre[Spec.RECENT_ROOT_ADDRESS] = Account(
        nonce=1,
        code=probe_code,
        storage={**entries, SLOT_EXECUTED: CANARY_VALUE},
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(
            Spec.RECENT_ROOT_ADDRESS, SENDER_FRAME_GAS
        ),
        recent_root_references=references,
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            Spec.RECENT_ROOT_ADDRESS: Account(
                storage={
                    **entries,
                    **expected_gas,
                    SLOT_UNDECLARED_KEY_GAS: cold_probe.gas_cost(fork),
                    SLOT_EXECUTED: EXECUTED_VALUE,
                }
            )
        },
    )


def test_prewarm_survives_frame_revert(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    references: List[RecentRootReference],
) -> None:
    """
    The warm set the references populate predates every frame, so a frame
    that reverts cannot cool it.

    Frame two touches the recent root contract and reverts; frame three
    touches it again and must still find it warm. EIP-8141 unwinds a
    reverted frame's accesses only as far as that frame's entry, and the
    reference check runs before any frame starts — so an implementation
    that warms lazily inside the first frame that touches the contract
    reports the cold price here.
    """
    seed_entries(pre, references)
    warm_probe = Op.BALANCE(
        address=Spec.RECENT_ROOT_ADDRESS, address_warm=True
    )

    reverting_prober = pre.deploy_contract(
        code=warm_probe + Op.POP + Op.REVERT(0, 0)
    )
    second_prober = pre.deploy_contract(
        code=CodeGasMeasure(
            code=warm_probe,
            extra_stack_items=1,
            sstore_key=SLOT_SECOND_FRAME_ACCESS_GAS,
        )
        + Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE)
        + Op.STOP,
        storage={SLOT_EXECUTED: CANARY_VALUE},
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
                target=reverting_prober,
                gas_limit=SENDER_FRAME_GAS,
            ),
            Frame(
                mode=Spec8141.MODE_SENDER,
                target=second_prober,
                gas_limit=SENDER_FRAME_GAS,
            ),
        ],
        recent_root_references=references,
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=Spec8141.STATUS_SUCCESS),
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
            second_prober: Account(
                storage={
                    SLOT_SECOND_FRAME_ACCESS_GAS: warm_probe.gas_cost(fork),
                    SLOT_EXECUTED: EXECUTED_VALUE,
                }
            )
        },
    )


def test_prewarm_does_not_leak_to_next_transaction(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
    current_slot: int,
    references: List[RecentRootReference],
) -> None:
    """
    Pre-warming is scoped to the transaction that declared the
    references.

    Two frame transactions share a block: the first declares a reference,
    the second declares none. The second finds the recent root contract
    cold again, which is what "the transaction's accessed address and
    storage-key sets" means — a block-scoped implementation reports the
    warm price for the second.
    """
    seed_entries(pre, references)
    cold_probe = Op.BALANCE(
        address=Spec.RECENT_ROOT_ADDRESS, address_warm=False
    )
    prober_code: Bytecode = (
        CodeGasMeasure(
            code=cold_probe,
            extra_stack_items=1,
            sstore_key=SLOT_RECENT_ROOT_ACCESS_GAS,
        )
        + Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE)
        + Op.STOP
    )
    prober = pre.deploy_contract(
        code=prober_code, storage={SLOT_EXECUTED: CANARY_VALUE}
    )
    first_sender = pre.fund_eoa()
    second_sender = pre.fund_eoa()
    quiet_target = pre.deploy_contract(code=Op.STOP)

    referencing_tx = Transaction(
        sender=first_sender,
        frames=verify_and_sender_frames(quiet_target, SENDER_FRAME_GAS),
        recent_root_references=references,
    )
    probing_tx = Transaction(
        sender=second_sender,
        frames=verify_and_sender_frames(prober, SENDER_FRAME_GAS),
        recent_root_references=[],
    )

    blockchain_test(
        genesis_environment=Environment(slot_number=current_slot),
        pre=pre,
        blocks=[
            Block(
                slot_number=current_slot,
                txs=[referencing_tx, probing_tx],
            )
        ],
        post={
            prober: Account(
                storage={
                    SLOT_RECENT_ROOT_ACCESS_GAS: cold_probe.gas_cost(fork),
                    SLOT_EXECUTED: EXECUTED_VALUE,
                }
            )
        },
    )
