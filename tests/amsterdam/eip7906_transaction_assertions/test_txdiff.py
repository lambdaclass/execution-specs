"""
TXDIFF opcode tests for
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

TXDIFF complements TXTRACE's enumeration with direct keyed lookups:
slot, balance, and code hash values before and after, per-address
views over the slot and event tables, and the account change flags.
Expected values are hand-derived and baked into the assertion frame's
code, which reverts on the first mismatch.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytecode,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
    compute_create_address,
    keccak256,
)

from tests.amsterdam.eip8141_frame_transactions.helpers import (
    default_frame,
    sender_frame,
    verify_frame,
)
from tests.amsterdam.eip8141_frame_transactions.spec import Spec as FrameSpec

from .helpers import (
    BODY_FRAME_GAS,
    MAX_UINT256,
    SENTINEL_MARKER,
    SENTINEL_SLOT,
    assert_eq,
    contract_sender_max_cost,
    deploy_approving_sender,
    deploy_sentinel,
    post_tx_frame,
    sorted_by_address,
)
from .spec import Spec, ref_spec_7906

REFERENCE_SPEC_GIT_PATH = ref_spec_7906.git_path
REFERENCE_SPEC_VERSION = ref_spec_7906.version

# EIP-7906 extends EIP-8141, which is slated for the fork after
# Amsterdam, so fixtures are labeled with the pseudo `Bogota` fork.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

MAX_FEE_PER_GAS = 10
"""Explicit maximum fee so the gas pre-charge is hand-computable."""


@pytest.mark.parametrize(
    "wrong_offset",
    [
        pytest.param(0, id="correct_expectations"),
        pytest.param(1, id="wrong_expectation_bites"),
    ],
)
def test_txdiff_slot_lookup_before_after(
    state_test: StateTestFiller,
    pre: Alloc,
    wrong_offset: int,
) -> None:
    """
    Look up a twice-written slot by key: the before variant returns
    the transaction prestate value and the after variant the latest
    write — the intermediate value is not observable, consistent with
    the TXTRACE collapse.

    Rules: R-053, R-054, R-055.
    """
    slot_key = 7
    writer = pre.deploy_contract(
        code=Op.SSTORE(slot_key, Op.CALLDATALOAD(0)) + Op.STOP,
        storage={slot_key: 0xA},
    )

    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.TXDIFF(slot_key, writer, Spec.TXDIFF_SLOT_VALUE_BEFORE),
            0xA,
        )
        + assert_eq(
            Op.TXDIFF(slot_key, writer, Spec.TXDIFF_SLOT_VALUE_AFTER),
            0xC + wrong_offset,
        )
        + Op.STOP
    )

    passes = wrong_offset == 0
    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(
                target=writer,
                data=(0xB).to_bytes(32, "big"),
                gas_limit=BODY_FRAME_GAS,
            ),
            default_frame(
                target=writer,
                data=(0xC).to_bytes(32, "big"),
                gas_limit=BODY_FRAME_GAS,
            ),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS
                    if passes
                    else FrameSpec.STATUS_FAILURE
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            writer: Account(storage={slot_key: 0xC if passes else 0xA}),
        },
    )


def test_txdiff_unmodified_key_returns_live_value(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Query keys the transaction never modified: an untouched slot on a
    modified contract, a slot on an untouched contract, the balance of
    an untouched account, and the code hash of an untouched contract
    all return the current live value for both the before and the
    after variant.

    Rule: R-065.
    """
    touched_key, untouched_key = 1, 2
    writer = pre.deploy_contract(
        code=Op.SSTORE(touched_key, 0x11) + Op.STOP,
        storage={untouched_key: 0x22},
    )
    bystander_code = Op.STOP
    bystander = pre.deploy_contract(code=bystander_code, storage={5: 0x55})
    dave_balance = 77
    dave = pre.fund_eoa(amount=dave_balance)

    checks = Bytecode()
    for param, expected in [
        (Spec.TXDIFF_SLOT_VALUE_BEFORE, 0x22),
        (Spec.TXDIFF_SLOT_VALUE_AFTER, 0x22),
    ]:
        checks += assert_eq(Op.TXDIFF(untouched_key, writer, param), expected)
        checks += assert_eq(Op.TXDIFF(5, bystander, param), 0x55)
    for param in [
        Spec.TXDIFF_BALANCE_BEFORE,
        Spec.TXDIFF_BALANCE_AFTER,
    ]:
        checks += assert_eq(Op.TXDIFF(0, dave, param), dave_balance)
    for param in [
        Spec.TXDIFF_CODEHASH_BEFORE,
        Spec.TXDIFF_CODEHASH_AFTER,
    ]:
        checks += assert_eq(
            Op.TXDIFF(0, bystander, param),
            keccak256(bytes(bystander_code)),
        )
    asserter = pre.deploy_contract(code=checks + Op.STOP)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=writer, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            writer: Account(storage={touched_key: 0x11, untouched_key: 0x22}),
        },
    )


def test_txdiff_balance_lookup(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Look up the payer's and the recipient's balances by key: the
    before variants return the funded prestate amounts, the after
    variants the transferred amounts with the payer down its full
    escrow, hand-computed from the EIP-8141 accounting with a
    contract sender.

    Rules: R-056, R-057.
    """
    sender_balance = 10**18
    bob_balance = 5
    transfer_value = 10**15
    sender = deploy_approving_sender(pre, balance=sender_balance)
    bob = pre.fund_eoa(amount=bob_balance)

    frames = [
        verify_frame(),
        sender_frame(target=bob, value=transfer_value),
        post_tx_frame(),
    ]
    max_cost = contract_sender_max_cost(frames, MAX_FEE_PER_GAS)

    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.TXDIFF(0, sender, Spec.TXDIFF_BALANCE_BEFORE),
            sender_balance,
        )
        + assert_eq(
            Op.TXDIFF(0, sender, Spec.TXDIFF_BALANCE_AFTER),
            sender_balance - max_cost - transfer_value,
        )
        + assert_eq(Op.TXDIFF(0, bob, Spec.TXDIFF_BALANCE_BEFORE), bob_balance)
        + assert_eq(
            Op.TXDIFF(0, bob, Spec.TXDIFF_BALANCE_AFTER),
            bob_balance + transfer_value,
        )
        + Op.STOP
    )
    frames[-1] = post_tx_frame(target=asserter)

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE_PER_GAS,
        max_priority_fee_per_gas=0,
        frames=frames,
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            bob: Account(balance=bob_balance + transfer_value),
        },
    )


def test_txdiff_codehash_lookup(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Look up code hashes by address: a body-deployed child moves from
    the empty-code hash to the keccak of its runtime code, an
    untouched contract reports its live hash in both variants, and —
    diverging from `EXTCODEHASH`'s zero for non-existent accounts —
    a funded EOA and a never-existing address both report the
    empty-code hash, asserted against an in-frame `EXTCODEHASH`
    contrast.

    Rules: R-058, R-059, R-084.
    """
    alice = pre.fund_eoa()
    bob = pre.fund_eoa(amount=1)
    ghost = pre.nonexistent_account()

    runtime = bytes([0xFE])
    initcode = Op.MSTORE8(0, 0xFE) + Op.RETURN(0, 1)
    factory = pre.deploy_contract(
        code=Op.MSTORE(0, bytes(initcode).ljust(32, b"\x00"))
        + Op.POP(Op.CREATE(0, 0, len(bytes(initcode))))
        + Op.STOP
    )
    child = compute_create_address(address=factory, nonce=1)
    bystander_code = Op.PUSH0 + Op.STOP
    bystander = pre.deploy_contract(code=bystander_code)

    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.TXDIFF(0, child, Spec.TXDIFF_CODEHASH_BEFORE),
            Spec.EMPTY_CODE_HASH,
        )
        + assert_eq(
            Op.TXDIFF(0, child, Spec.TXDIFF_CODEHASH_AFTER),
            keccak256(runtime),
        )
        + assert_eq(
            Op.TXDIFF(0, bystander, Spec.TXDIFF_CODEHASH_BEFORE),
            keccak256(bytes(bystander_code)),
        )
        + assert_eq(
            Op.TXDIFF(0, bystander, Spec.TXDIFF_CODEHASH_AFTER),
            keccak256(bytes(bystander_code)),
        )
        # A codeless EOA reports the empty-code hash, not zero.
        + assert_eq(
            Op.TXDIFF(0, bob, Spec.TXDIFF_CODEHASH_BEFORE),
            Spec.EMPTY_CODE_HASH,
        )
        # A never-existing account also reports the empty-code hash,
        # while EXTCODEHASH returns zero for it.
        + assert_eq(
            Op.TXDIFF(0, ghost, Spec.TXDIFF_CODEHASH_BEFORE),
            Spec.EMPTY_CODE_HASH,
        )
        + assert_eq(
            Op.TXDIFF(0, ghost, Spec.TXDIFF_CODEHASH_AFTER),
            Spec.EMPTY_CODE_HASH,
        )
        + assert_eq(Op.EXTCODEHASH(ghost), 0)
        + Op.STOP
    )

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            default_frame(target=factory, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            child: Account(code=runtime),
        },
    )


def test_txdiff_per_address_slot_view(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the per-address slot view: the count params return each
    address's entry count — zero for an untouched account — and the
    index params map local indexes to global enumeration indexes,
    round-tripped through TXTRACE to recover the expected slot keys.

    Rules: R-060, R-061, R-066, R-067.
    """
    alice = pre.fund_eoa()
    charlie = pre.fund_eoa(amount=1)

    keys_a = [10, 20, 30]
    keys_b = [5]
    writer_a = pre.deploy_contract(
        code=sum((Op.SSTORE(key, 0xA0 + key) for key in keys_a), Bytecode())
        + Op.STOP
    )
    writer_b = pre.deploy_contract(
        code=sum((Op.SSTORE(key, 0xB0 + key) for key in keys_b), Bytecode())
        + Op.STOP
    )

    # The global table is sorted by address, then slot key: compute
    # each writer's block of global indexes by hand.
    ordered = sorted_by_address([writer_a, writer_b])
    global_index = {}
    next_index = 0
    for address in ordered:
        keys = keys_a if address == writer_a else keys_b
        for key in sorted(keys):
            global_index[(address, key)] = next_index
            next_index += 1

    checks = (
        assert_eq(
            Op.TXDIFF(0, writer_a, Spec.TXDIFF_ADDRESS_SLOTS_COUNT),
            len(keys_a),
        )
        + assert_eq(
            Op.TXDIFF(0, writer_b, Spec.TXDIFF_ADDRESS_SLOTS_COUNT),
            len(keys_b),
        )
        + assert_eq(Op.TXDIFF(0, charlie, Spec.TXDIFF_ADDRESS_SLOTS_COUNT), 0)
    )
    for address, keys in [(writer_a, keys_a), (writer_b, keys_b)]:
        for local_index, key in enumerate(sorted(keys)):
            checks += assert_eq(
                Op.TXDIFF(
                    local_index, address, Spec.TXDIFF_ADDRESS_SLOT_INDEX
                ),
                global_index[(address, key)],
            )
            # Round-trip: the returned global index must resolve to
            # this address and slot key in the TXTRACE enumeration.
            checks += assert_eq(
                Op.TXTRACE(
                    Op.TXDIFF(
                        local_index,
                        address,
                        Spec.TXDIFF_ADDRESS_SLOT_INDEX,
                    ),
                    Spec.TXTRACE_SLOT_KEY,
                ),
                key,
            )
            checks += assert_eq(
                Op.TXTRACE(
                    Op.TXDIFF(
                        local_index,
                        address,
                        Spec.TXDIFF_ADDRESS_SLOT_INDEX,
                    ),
                    Spec.TXTRACE_SLOT_ADDRESS,
                ),
                address,
            )
    asserter = pre.deploy_contract(code=checks + Op.STOP)

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            default_frame(target=writer_a, gas_limit=BODY_FRAME_GAS),
            default_frame(target=writer_b, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            writer_a: Account(storage={key: 0xA0 + key for key in keys_a}),
            writer_b: Account(storage={key: 0xB0 + key for key in keys_b}),
        },
    )


def test_txdiff_per_address_event_view(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the per-address event view over an interleaved A, B, A, B
    emission pattern: each address's local view maps to its global
    log indexes in emission order, round-tripped through TXTRACE's
    emitter-address param.

    Rules: R-062, R-063, R-067.
    """
    alice = pre.fund_eoa()
    charlie = pre.fund_eoa(amount=1)

    emitter_a = pre.deploy_contract(code=Op.LOG0(0, 0) + Op.STOP)
    emitter_b = pre.deploy_contract(code=Op.LOG1(0, 0, 0xB) + Op.STOP)
    # One driver frame calls A, B, A, B: the interleaving breaks any
    # first-index-plus-offset shortcut.
    driver = pre.deploy_contract(
        code=sum(
            (
                Op.POP(Op.CALL(Op.GAS, emitter, 0, 0, 0, 0, 0))
                for emitter in [emitter_a, emitter_b, emitter_a, emitter_b]
            ),
            Bytecode(),
        )
        + Op.STOP
    )

    view = {emitter_a: [0, 2], emitter_b: [1, 3]}
    checks = assert_eq(
        Op.TXDIFF(0, charlie, Spec.TXDIFF_ADDRESS_EVENTS_COUNT), 0
    )
    for emitter, globals_ in view.items():
        checks += assert_eq(
            Op.TXDIFF(0, emitter, Spec.TXDIFF_ADDRESS_EVENTS_COUNT),
            len(globals_),
        )
        for local_index, expected_global in enumerate(globals_):
            checks += assert_eq(
                Op.TXDIFF(
                    local_index,
                    emitter,
                    Spec.TXDIFF_ADDRESS_EVENT_INDEX,
                ),
                expected_global,
            )
            checks += assert_eq(
                Op.TXTRACE(
                    Op.TXDIFF(
                        local_index,
                        emitter,
                        Spec.TXDIFF_ADDRESS_EVENT_INDEX,
                    ),
                    Spec.TXTRACE_EVENT_ADDRESS,
                ),
                emitter,
            )
    asserter = pre.deploy_contract(code=checks + Op.STOP)

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            default_frame(target=driver, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={alice: Account(nonce=1)},
    )


@pytest.mark.parametrize(
    "param",
    [
        pytest.param(Spec.TXDIFF_ADDRESS_SLOT_INDEX, id="slot_view"),
        pytest.param(Spec.TXDIFF_ADDRESS_EVENT_INDEX, id="event_view"),
    ],
)
@pytest.mark.parametrize(
    "index_kind,passes",
    [
        pytest.param("last", True, id="last_local_index"),
        pytest.param("count", False, id="count"),
        pytest.param("count_plus_one", False, id="count_plus_one"),
        pytest.param("max", False, id="max_index"),
        pytest.param("empty_view", False, id="empty_view_index_zero"),
    ],
)
def test_txdiff_view_local_index_out_of_bounds_halts(
    state_test: StateTestFiller,
    pre: Alloc,
    param: int,
    index_kind: str,
    passes: bool,
) -> None:
    """
    Probe the per-address view index params at their exact bounds over
    a two-entry view: the last local index succeeds, while the count,
    one above, the maximum index, and index zero on an untouched
    account's empty view each exceptionally halt.

    Rule: R-068.
    """
    sentinel = deploy_sentinel(pre)
    charlie = pre.fund_eoa(amount=1)
    subject = pre.deploy_contract(
        code=Op.SSTORE(1, 0x11)
        + Op.SSTORE(2, 0x22)
        + Op.LOG0(0, 0)
        + Op.LOG0(0, 0)
        + Op.STOP
    )

    view_count = 2
    index = {
        "last": view_count - 1,
        "count": view_count,
        "count_plus_one": view_count + 1,
        "max": MAX_UINT256,
        "empty_view": 0,
    }[index_kind]
    queried = charlie if index_kind == "empty_view" else subject

    asserter = pre.deploy_contract(
        code=Op.POP(Op.TXDIFF(index, queried, param)) + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=subject, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS
                    if passes
                    else FrameSpec.STATUS_FAILURE
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: SENTINEL_MARKER if passes else 0}
            ),
        },
    )


def test_txdiff_account_change_flags(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the account change flags for one persona per bit pattern: the
    sender (nonce and balance), a recipient (balance only), a storage
    writer (storage only), a factory (nonce only), an endowed child
    with constructor storage (all four bits), and two untouched
    accounts (zero) — each compared as a full 256-bit word, so any
    stray higher bit fails.

    Rules: R-064, R-069, R-070, R-071, R-072, R-073, R-075.
    """
    alice = pre.fund_eoa()
    bob = pre.fund_eoa(amount=1)
    charlie = pre.fund_eoa(amount=1)
    bystander = pre.deploy_contract(code=Op.STOP, storage={1: 1})
    writer = pre.deploy_contract(code=Op.SSTORE(1, 0x11) + Op.STOP)

    endowment = 5
    # The child's constructor writes a slot and returns one byte of
    # runtime code; with the endowment and the creation nonce, all
    # four account fields change. The endowment arrives with the
    # calling frame and leaves with the creation, so the factory's own
    # balance nets zero and only its nonce bit is set.
    initcode = Op.SSTORE(0, 1) + Op.MSTORE8(0, 0xFE) + Op.RETURN(0, 1)
    factory = pre.deploy_contract(
        code=Op.MSTORE(0, bytes(initcode).ljust(32, b"\x00"))
        + Op.POP(Op.CREATE(Op.CALLVALUE, 0, len(bytes(initcode))))
        + Op.STOP,
    )
    child = compute_create_address(address=factory, nonce=1)

    expected_flags = {
        alice: Spec.FLAG_NONCE_CHANGED | Spec.FLAG_BALANCE_CHANGED,
        bob: Spec.FLAG_BALANCE_CHANGED,
        writer: Spec.FLAG_STORAGE_CHANGED,
        factory: Spec.FLAG_NONCE_CHANGED,
        child: Spec.FLAG_NONCE_CHANGED
        | Spec.FLAG_BALANCE_CHANGED
        | Spec.FLAG_STORAGE_CHANGED
        | Spec.FLAG_CODE_CHANGED,
        charlie: 0,
        bystander: 0,
    }
    checks = Bytecode()
    for address, flags in expected_flags.items():
        checks += assert_eq(
            Op.TXDIFF(0, address, Spec.TXDIFF_ACCOUNT_CHANGE_FLAGS),
            flags,
        )
    asserter = pre.deploy_contract(code=checks + Op.STOP)

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            sender_frame(target=bob, value=10**15),
            default_frame(target=writer, gas_limit=BODY_FRAME_GAS),
            sender_frame(
                target=factory,
                value=endowment,
                gas_limit=BODY_FRAME_GAS,
            ),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            child: Account(
                balance=endowment, storage={0: 1}, code=bytes([0xFE])
            ),
        },
    )


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("all_restored", id="all_restored_flags_zero"),
        pytest.param("balance_kept", id="slot_restored_balance_bit_remains"),
    ],
)
def test_txdiff_flags_net_difference_only(
    state_test: StateTestFiller,
    pre: Alloc,
    scenario: str,
) -> None:
    """
    Restore a written slot to its prestate value within the body: with
    nothing else changed the account's flags collapse to zero — values
    modified and later restored set no bit — while the twin that also
    keeps a balance change reports exactly the balance bit.

    Rules: R-074, R-075.
    """
    slot_key = 9
    subject = pre.deploy_contract(
        code=Op.SSTORE(slot_key, Op.CALLDATALOAD(0)) + Op.STOP,
        storage={slot_key: 0x9},
    )

    if scenario == "all_restored":
        body_value = 0
        expected_flags = 0
    else:
        body_value = 3
        expected_flags = Spec.FLAG_BALANCE_CHANGED

    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.TXDIFF(0, subject, Spec.TXDIFF_ACCOUNT_CHANGE_FLAGS),
            expected_flags,
        )
        # Cross-check: the restored slot also vanished from the
        # per-address view.
        + assert_eq(Op.TXDIFF(0, subject, Spec.TXDIFF_ADDRESS_SLOTS_COUNT), 0)
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            sender_frame(
                target=subject,
                value=body_value,
                data=(0xF).to_bytes(32, "big"),
                gas_limit=BODY_FRAME_GAS,
            ),
            default_frame(
                target=subject,
                data=(0x9).to_bytes(32, "big"),
                gas_limit=BODY_FRAME_GAS,
            ),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            subject: Account(storage={slot_key: 0x9}),
        },
    )


def test_txdiff_nonce_flag_without_nonce_value(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Bump only an account's nonce — a creation whose empty initcode
    leaves no code, balance, or storage change on the factory — and
    check that the flags word reports exactly the nonce bit: the flag
    is the only nonce observable, since no TXTRACE or TXDIFF param
    exposes the nonce value itself.

    Rules: R-069, R-076.
    """
    factory = pre.deploy_contract(code=Op.POP(Op.CREATE(0, 0, 0)) + Op.STOP)

    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.TXDIFF(0, factory, Spec.TXDIFF_ACCOUNT_CHANGE_FLAGS),
            Spec.FLAG_NONCE_CHANGED,
        )
        + assert_eq(Op.TXDIFF(0, factory, Spec.TXDIFF_ADDRESS_SLOTS_COUNT), 0)
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=factory, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={factory: Account(nonce=2)},
    )


@pytest.mark.parametrize(
    "reserved_input",
    [
        pytest.param(0, id="zero_passes"),
        pytest.param(1, id="one_halts"),
        pytest.param(MAX_UINT256, id="max_halts"),
    ],
)
@pytest.mark.parametrize(
    "param",
    [
        pytest.param(Spec.TXDIFF_BALANCE_BEFORE, id="balance_before"),
        pytest.param(Spec.TXDIFF_BALANCE_AFTER, id="balance_after"),
        pytest.param(Spec.TXDIFF_CODEHASH_BEFORE, id="codehash_before"),
        pytest.param(Spec.TXDIFF_CODEHASH_AFTER, id="codehash_after"),
        pytest.param(
            Spec.TXDIFF_ADDRESS_SLOTS_COUNT, id="address_slots_count"
        ),
        pytest.param(
            Spec.TXDIFF_ADDRESS_EVENTS_COUNT, id="address_events_count"
        ),
        pytest.param(
            Spec.TXDIFF_ACCOUNT_CHANGE_FLAGS, id="account_change_flags"
        ),
    ],
)
def test_txdiff_reserved_in3_nonzero_halts(
    state_test: StateTestFiller,
    pre: Alloc,
    param: int,
    reserved_input: int,
) -> None:
    """
    Supply a non-zero value to the reserved third operand of every
    keyed TXDIFF param that requires it to be zero: each exceptionally
    halts the assertion frame, while the zero control passes.

    Rule: R-085.
    """
    sentinel = deploy_sentinel(pre)
    asserter = pre.deploy_contract(
        code=Op.POP(Op.TXDIFF(reserved_input, Op.ADDRESS, param)) + Op.STOP
    )
    passes = reserved_input == 0

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS
                    if passes
                    else FrameSpec.STATUS_FAILURE
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: SENTINEL_MARKER if passes else 0}
            ),
        },
    )


@pytest.mark.parametrize(
    "param,passes",
    [
        pytest.param(
            Spec.TXDIFF_ACCOUNT_CHANGE_FLAGS, True, id="last_defined"
        ),
        pytest.param(
            Spec.FIRST_UNDEFINED_TXDIFF_PARAM,
            False,
            id="first_undefined",
        ),
        pytest.param(0xFF, False, id="high_undefined"),
        pytest.param(MAX_UINT256, False, id="max_undefined"),
    ],
)
def test_txdiff_undefined_param_halts(
    state_test: StateTestFiller,
    pre: Alloc,
    param: int,
    passes: bool,
) -> None:
    """
    Query params on either side of TXDIFF's defined range: the last
    defined param succeeds while 0x0B, 0xFF, and the maximum selector
    each exceptionally halt.

    Rule: R-053.
    """
    sentinel = deploy_sentinel(pre)
    asserter = pre.deploy_contract(
        code=Op.POP(Op.TXDIFF(0, Op.ADDRESS, param)) + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: SENTINEL_MARKER if passes else 0}
            ),
        },
    )


def test_txdiff_address_operand_uint160_max(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Query the maximum clean address value: an untouched account at
    2**160 - 1 reports live-value semantics — a zero balance in both
    variants and the empty-code hash, a non-zero constant a wrong
    lookup cannot fake. Address operands with dirty high bits are
    deliberately not pinned: the spec is silent between masking and
    halting.

    Rule: R-065.
    """
    sentinel = deploy_sentinel(pre)
    max_address = 2**160 - 1

    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.TXDIFF(0, max_address, Spec.TXDIFF_BALANCE_BEFORE), 0
        )
        + assert_eq(Op.TXDIFF(0, max_address, Spec.TXDIFF_BALANCE_AFTER), 0)
        + assert_eq(
            Op.TXDIFF(0, max_address, Spec.TXDIFF_CODEHASH_BEFORE),
            Spec.EMPTY_CODE_HASH,
        )
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER}),
        },
    )
