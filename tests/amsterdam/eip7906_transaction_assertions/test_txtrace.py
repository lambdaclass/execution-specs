"""
TXTRACE opcode tests for
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

TXTRACE enumerates the transaction's collapsed state diff — balance,
storage, and deployment changes, emitted events, and the gas
pre-charge — from inside a `POST_TX` frame. Every expected value is
hand-derived from the spec's state-difference semantics and baked into
the assertion frame's code, which reverts on the first mismatch; the
sentinel contract's post state and the frame receipt statuses then
report whether all assertions held.
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
    While,
    compute_create_address,
    keccak256,
)
from execution_testing import (
    Macros as Om,
)
from execution_testing.base_types import StorageRootType

from tests.amsterdam.eip8141_frame_transactions.helpers import (
    AMPLE_FRAME_GAS,
    default_frame,
    sender_frame,
    verify_frame,
)
from tests.amsterdam.eip8141_frame_transactions.spec import Spec as FrameSpec

from .helpers import (
    ASSERTER_FRAME_GAS,
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


def test_txtrace_stack_operand_convention(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin TXTRACE's operand order — index on top of the stack, the
    parameter selector below it, the FRAMEPARAM convention the spec
    references — with inputs whose two readings diverge.

    `TXTRACE(index=1, param=0x03)` must return the second changed
    balance address; under the swapped reading it would be the scalar
    `slots_changed` with a non-zero reserved input, an exceptional
    halt. The twin `TXTRACE(index=3, param=0x01)` must halt on its
    reserved input, while the swapped reading would return a balance
    address and succeed — so a swapped implementation fails both arms.

    Rules: R-016, R-017.
    """
    alice = pre.fund_eoa()
    bob = pre.fund_eoa(amount=1)
    transfer_value = 10**15
    # Balance changes are enumerated ascending by address: the payer
    # and the recipient in numerical order.
    first, second = sorted_by_address([alice, bob])

    asserter = pre.deploy_contract(
        code=assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCE_ADDRESS), first)
        + assert_eq(Op.TXTRACE(1, Spec.TXTRACE_BALANCE_ADDRESS), second)
        + Op.STOP
    )
    # The swapped-orientation twin: a correct implementation halts on
    # the non-zero reserved input of the scalar param 0x01.
    halter = pre.deploy_contract(
        code=Op.POP(Op.TXTRACE(3, Spec.TXTRACE_SLOTS_CHANGED)) + Op.STOP
    )

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            sender_frame(target=bob, value=transfer_value),
            post_tx_frame(target=asserter),
            post_tx_frame(target=halter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_FAILURE),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            # The halting twin reverted the body: the transfer is
            # rolled back.
            bob: Account(balance=1),
        },
    )


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("full_body", id="full_body"),
        pytest.param("payer_only", id="payer_only_zero_boundaries"),
    ],
)
def test_txtrace_counts(
    state_test: StateTestFiller,
    pre: Alloc,
    scenario: str,
) -> None:
    """
    Pin the four scalar counts against a hand-enumerated body — three
    balance changes, four storage slot changes, one deployment, five
    events — and their zero boundaries against a payer-only body whose
    diff is a single balance change.

    Rules: R-021, R-022, R-023, R-033.
    """
    alice = pre.fund_eoa()

    if scenario == "full_body":
        bob = pre.fund_eoa(amount=1)
        charlie = pre.fund_eoa(amount=1)
        writer = pre.deploy_contract(
            code=sum(
                (Op.SSTORE(key, 0xA0 + key) for key in range(4)),
                Bytecode(),
            )
            + Op.STOP
        )
        # The initcode returns one non-zero byte of runtime code, so
        # the child counts as a deployed contract.
        initcode = Op.MSTORE8(0, 0xFE) + Op.RETURN(0, 1)
        factory = pre.deploy_contract(
            code=Op.MSTORE(0, bytes(initcode).ljust(32, b"\x00"))
            + Op.POP(Op.CREATE(0, 0, len(bytes(initcode))))
            + Op.STOP
        )
        emitter = pre.deploy_contract(code=Op.LOG0(0, 0) * 5 + Op.STOP)
        body_frames = [
            sender_frame(target=bob, value=10**15),
            sender_frame(target=charlie, value=10**15),
            default_frame(target=writer, gas_limit=BODY_FRAME_GAS),
            default_frame(target=factory, gas_limit=BODY_FRAME_GAS),
            default_frame(target=emitter),
        ]
        # Five explicit logs plus the two EIP-7708 transfer logs the
        # value-carrying frames emit.
        expected = (3, 4, 1, 7)
    else:
        body_frames = []
        # The gas pre-charge is the only diff entry of an empty body.
        expected = (1, 0, 0, 0)

    balances, slots, deployed, events = expected
    asserter = pre.deploy_contract(
        code=assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED), balances)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOTS_CHANGED), slots)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_CONTRACTS_DEPLOYED), deployed)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT), events)
        + Op.STOP
    )

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            *body_frames,
            post_tx_frame(target=asserter),
        ],
        # The assertion frame's success status is the oracle: a failed
        # count comparison would revert it and roll the body back.
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS)
                for _ in range(2 + len(body_frames))
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={alice: Account(nonce=1)},
    )


@pytest.mark.parametrize(
    "wrong_offset",
    [
        pytest.param(0, id="correct_expectations"),
        pytest.param(1, id="wrong_expectation_bites"),
    ],
)
def test_txtrace_balances_changed_enumeration(
    state_test: StateTestFiller,
    pre: Alloc,
    wrong_offset: int,
) -> None:
    """
    Pin the full balance enumeration of a two-transfer body: addresses
    ascending, each entry's before value from the transaction
    prestate, and each after value as of the TXTRACE call — including
    the payer's full gas pre-charge, hand-computed from the EIP-8141
    accounting with a contract sender.

    The wrong-expectation twin offsets one recipient's after balance
    by one wei, proving the oracle bites.

    Rules: R-024, R-025, R-026, R-049, R-050, R-086.
    """
    sender_balance = 10**18
    bob_balance = 1
    charlie_balance = 2
    value_to_bob = 10**15
    value_to_charlie = 3 * 10**15

    sender = deploy_approving_sender(pre, balance=sender_balance)
    bob = pre.fund_eoa(amount=bob_balance)
    charlie = pre.fund_eoa(amount=charlie_balance)

    frames = [
        verify_frame(),
        sender_frame(target=bob, value=value_to_bob),
        sender_frame(target=charlie, value=value_to_charlie),
        post_tx_frame(),  # placeholder for gas accounting
    ]
    max_cost = contract_sender_max_cost(frames, MAX_FEE_PER_GAS)

    balance_before = {
        sender: sender_balance,
        bob: bob_balance,
        charlie: charlie_balance,
    }
    balance_after = {
        # The payer's escrow is the full maximum cost: the unused-gas
        # refund happens only after all frames.
        sender: sender_balance - max_cost - value_to_bob - value_to_charlie,
        bob: bob_balance + value_to_bob + wrong_offset,
        charlie: charlie_balance + value_to_charlie,
    }

    checks = assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED), 3)
    for index, address in enumerate(sorted_by_address([sender, bob, charlie])):
        checks += assert_eq(
            Op.TXTRACE(index, Spec.TXTRACE_BALANCE_ADDRESS), address
        )
        checks += assert_eq(
            Op.TXTRACE(index, Spec.TXTRACE_BALANCE_BEFORE),
            balance_before[address],
        )
        checks += assert_eq(
            Op.TXTRACE(index, Spec.TXTRACE_BALANCE_AFTER),
            balance_after[address],
        )
    asserter = pre.deploy_contract(code=checks + Op.STOP)
    frames[-1] = post_tx_frame(target=asserter)

    passes = wrong_offset == 0
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
            bob: Account(
                balance=bob_balance + value_to_bob if passes else bob_balance
            ),
            charlie: Account(
                balance=charlie_balance + value_to_charlie
                if passes
                else charlie_balance
            ),
        },
    )


def test_txtrace_excludes_coinbase_and_refund(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin what the balance enumeration excludes at assertion time: with
    an empty body the diff holds exactly one entry — the payer, down
    the full maximum cost. The coinbase has not been credited and the
    unused-gas refund has not been applied; both happen only after all
    frames.

    Rule: R-049.
    """
    sender_balance = 10**18
    sender = deploy_approving_sender(pre, balance=sender_balance)

    frames = [verify_frame(), post_tx_frame()]
    max_cost = contract_sender_max_cost(frames, MAX_FEE_PER_GAS)

    asserter = pre.deploy_contract(
        code=assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED), 1)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCE_ADDRESS), sender)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCE_BEFORE), sender_balance)
        + assert_eq(
            Op.TXTRACE(0, Spec.TXTRACE_BALANCE_AFTER),
            sender_balance - max_cost,
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
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=2)},
    )


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("intermediary_restored", id="intermediary_restored"),
        pytest.param("payer_restored", id="payer_restored"),
    ],
)
def test_txtrace_balances_restored_to_prestate_excluded(
    state_test: StateTestFiller,
    pre: Alloc,
    scenario: str,
) -> None:
    """
    Check that accounts whose balance is restored to its prestate
    value vanish from the enumeration: a bouncer forwarding its whole
    incoming value nets zero and is absent, and a body refunding the
    payer exactly its escrow plus transfers empties the payer's entry
    — while the gas pre-charge params still report the charge.

    Rules: R-048, R-049, R-074.
    """
    sender_balance = 10**18
    sender = deploy_approving_sender(pre, balance=sender_balance)
    transfer_value = 10**15

    if scenario == "intermediary_restored":
        bob = pre.fund_eoa(amount=1)
        # The bouncer forwards its whole incoming value: net zero.
        bouncer = pre.deploy_contract(
            code=Op.POP(Op.CALL(Op.GAS, bob, Op.CALLVALUE, 0, 0, 0, 0))
            + Op.STOP
        )
        frames = [
            verify_frame(),
            sender_frame(
                target=bouncer,
                value=transfer_value,
                gas_limit=BODY_FRAME_GAS,
            ),
            post_tx_frame(),
        ]
        first, second = sorted_by_address([sender, bob])
        checks = (
            assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED), 2)
            + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCE_ADDRESS), first)
            + assert_eq(Op.TXTRACE(1, Spec.TXTRACE_BALANCE_ADDRESS), second)
        )
        post = {
            bob: Account(balance=1 + transfer_value),
            bouncer: Account(balance=0),
        }
    else:
        refunder_balance = 10**18
        # The frame plan is fixed first: the maximum cost depends only
        # on the frame count and gas limits, and the refunder's code
        # needs it as a constant.
        frame_gas_plan = [
            AMPLE_FRAME_GAS,
            BODY_FRAME_GAS,
            ASSERTER_FRAME_GAS,
        ]
        intrinsic = (
            FrameSpec.FRAME_TX_INTRINSIC_COST
            + len(frame_gas_plan) * FrameSpec.FRAME_TX_PER_FRAME_COST
        )
        max_cost = (intrinsic + sum(frame_gas_plan)) * MAX_FEE_PER_GAS
        # The refunder returns the incoming value plus the escrow, so
        # the payer's balance lands exactly on its prestate value.
        refunder = pre.deploy_contract(
            code=Op.POP(
                Op.CALL(
                    Op.GAS,
                    sender,
                    Op.ADD(Op.CALLVALUE, max_cost),
                    0,
                    0,
                    0,
                    0,
                )
            )
            + Op.STOP,
            balance=refunder_balance,
        )
        checks = (
            # The restored payer is gone from the enumeration...
            assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED), 1)
            + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCE_ADDRESS), refunder)
            + assert_eq(
                Op.TXTRACE(0, Spec.TXTRACE_BALANCE_BEFORE),
                refunder_balance,
            )
            + assert_eq(
                Op.TXTRACE(0, Spec.TXTRACE_BALANCE_AFTER),
                refunder_balance - max_cost,
            )
            # ...while the gas pre-charge params still report it.
            + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_GAS_PRE_CHARGE), max_cost)
            + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_GAS_PAYER_ADDRESS), sender)
        )
        frames = [
            verify_frame(gas_limit=frame_gas_plan[0]),
            sender_frame(
                target=refunder,
                value=transfer_value,
                gas_limit=frame_gas_plan[1],
            ),
            post_tx_frame(gas_limit=frame_gas_plan[2]),
        ]
        post = {}

    asserter = pre.deploy_contract(code=checks + Op.STOP)
    frames[-1] = post_tx_frame(
        target=asserter, gas_limit=int(frames[-1].gas_limit)
    )

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

    state_test(pre=pre, tx=tx, post=post)


def test_txtrace_slots_changed_enumeration_and_order(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the storage enumeration's double sort order with a body that
    writes in inverted chronological order: the higher-sorted contract
    is written first and each contract writes its keys descending —
    including key 0 and the maximum key — yet the enumeration comes
    back ascending by address and then by slot key, with the before
    values from the prestate.

    Rules: R-027, R-028, R-029, R-030, R-086, R-087.
    """
    alice = pre.fund_eoa()
    keys = [0, 5, MAX_UINT256]
    prestate: StorageRootType = {5: 0x50}

    def writer_code(value_base: int) -> Bytecode:
        # Write keys in descending order: max key first.
        return (
            sum(
                (
                    Op.SSTORE(key, value_base + offset)
                    for offset, key in enumerate(reversed(keys))
                ),
                Bytecode(),
            )
            + Op.STOP
        )

    writer_a = pre.deploy_contract(code=writer_code(0xA0), storage=prestate)
    writer_b = pre.deploy_contract(code=writer_code(0xB0), storage=prestate)
    low, high = sorted_by_address([writer_a, writer_b])
    value_base = {writer_a: 0xA0, writer_b: 0xB0}

    checks = assert_eq(
        Op.TXTRACE(0, Spec.TXTRACE_SLOTS_CHANGED), len(keys) * 2
    )
    index = 0
    for address in (low, high):
        for key in sorted(keys):
            # The write loop stores descending, so the value written
            # to the k-th ascending key is the (n-1-k)-th offset.
            offset = len(keys) - 1 - sorted(keys).index(key)
            checks += assert_eq(
                Op.TXTRACE(index, Spec.TXTRACE_SLOT_ADDRESS), address
            )
            checks += assert_eq(Op.TXTRACE(index, Spec.TXTRACE_SLOT_KEY), key)
            checks += assert_eq(
                Op.TXTRACE(index, Spec.TXTRACE_SLOT_VALUE_BEFORE),
                prestate.get(key, 0),
            )
            checks += assert_eq(
                Op.TXTRACE(index, Spec.TXTRACE_SLOT_VALUE_AFTER),
                value_base[address] + offset,
            )
            index += 1
    asserter = pre.deploy_contract(code=checks + Op.STOP)

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            # Chronological write order inverts the enumeration order.
            default_frame(target=high, gas_limit=BODY_FRAME_GAS),
            default_frame(target=low, gas_limit=BODY_FRAME_GAS),
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
            writer_a: Account(
                storage={
                    0: 0xA2,
                    5: 0xA1,
                    MAX_UINT256: 0xA0,
                }
            ),
            writer_b: Account(
                storage={
                    0: 0xB2,
                    5: 0xB1,
                    MAX_UINT256: 0xB0,
                }
            ),
        },
    )


def test_txtrace_slot_multiple_writes_collapse(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Write one slot twice and restore another: the enumeration holds a
    single entry with the prestate before value and the latest after
    value — the intermediate write is not observable — and the
    restored slot vanishes entirely.

    Rules: R-046, R-047, R-048.
    """
    alice = pre.fund_eoa()
    collapse_key, restore_key = 7, 8
    writer = pre.deploy_contract(
        # Calldata selects the writing pass, keeping each body frame a
        # single-purpose write.
        code=Op.SSTORE(collapse_key, Op.CALLDATALOAD(0))
        + Op.SSTORE(restore_key, Op.CALLDATALOAD(32))
        + Op.STOP,
        storage={collapse_key: 0xA, restore_key: 0x1},
    )

    def pass_data(collapse_value: int, restore_value: int) -> bytes:
        return collapse_value.to_bytes(32, "big") + restore_value.to_bytes(
            32, "big"
        )

    asserter = pre.deploy_contract(
        code=assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOTS_CHANGED), 1)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOT_KEY), collapse_key)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOT_VALUE_BEFORE), 0xA)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOT_VALUE_AFTER), 0xC)
        + Op.STOP
    )

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            default_frame(
                target=writer,
                data=pass_data(0xB, 0x2),
                gas_limit=BODY_FRAME_GAS,
            ),
            default_frame(
                target=writer,
                data=pass_data(0xC, 0x1),
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
            writer: Account(storage={collapse_key: 0xC, restore_key: 0x1}),
        },
    )


def test_txtrace_contracts_deployed(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the deployed-contracts enumeration: a `CREATE` child and a
    `CREATE2` child appear — addresses hand-derived from the creation
    formulas, code hashes as the keccak of their runtime code, sorted
    ascending by address per the reference implementation's choice
    (the pinned spec leaves the deployed table's order unstated) —
    while a third creation returning empty code is not enumerated.

    Rules: R-023, R-031, R-032, R-051, R-052.
    """
    alice = pre.fund_eoa()

    runtime_a = bytes([0xFE])
    runtime_b = bytes([0xFE, 0xFD])
    initcode_a = Op.MSTORE8(0, 0xFE) + Op.RETURN(0, 1)
    initcode_b = Op.MSTORE8(0, 0xFE) + Op.MSTORE8(1, 0xFD) + Op.RETURN(0, 2)
    salt = 0x5A17

    def store_initcode(initcode: Bytecode) -> Bytecode:
        return Op.MSTORE(0, bytes(initcode).ljust(32, b"\x00"))

    factory = pre.deploy_contract(
        code=store_initcode(initcode_a)
        + Op.POP(Op.CREATE(0, 0, len(bytes(initcode_a))))
        + store_initcode(initcode_b)
        + Op.POP(Op.CREATE2(0, 0, len(bytes(initcode_b)), salt))
        # The third creation runs empty initcode: the child keeps the
        # empty code hash and must not be enumerated.
        + Op.POP(Op.CREATE(0, 0, 0))
        + Op.STOP,
    )
    child_a = compute_create_address(address=factory, nonce=1)
    child_b = compute_create_address(
        address=factory,
        salt=salt,
        initcode=bytes(initcode_b),
        opcode=Op.CREATE2,
    )
    expected = {
        child_a: keccak256(runtime_a),
        child_b: keccak256(runtime_b),
    }

    checks = assert_eq(Op.TXTRACE(0, Spec.TXTRACE_CONTRACTS_DEPLOYED), 2)
    for index, child in enumerate(sorted_by_address([child_a, child_b])):
        checks += assert_eq(
            Op.TXTRACE(index, Spec.TXTRACE_DEPLOYED_ADDRESS), child
        )
        checks += assert_eq(
            Op.TXTRACE(index, Spec.TXTRACE_DEPLOYED_CODEHASH),
            expected[child],
        )
    asserter = pre.deploy_contract(code=checks + Op.STOP)

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            default_frame(target=factory, gas_limit=BODY_FRAME_GAS * 2),
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
            child_a: Account(code=runtime_a),
            child_b: Account(code=runtime_b),
        },
    )


def test_txtrace_deployment_edge_cases(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Deploy a child and `SELFDESTRUCT` it within the same transaction:
    at assertion time its code change is live, so the child is
    enumerated — the diff reflects the state as of the TXTRACE call —
    while EIP-6780 deletes the account only at the end of the
    transaction, leaving no trace in the post state.

    Rules: R-047, R-051.
    """
    alice = pre.fund_eoa()

    # Child runtime: self-destruct to the zero beneficiary.
    child_runtime = bytes(Op.SELFDESTRUCT(0))
    initcode = Op.MSTORE(0, child_runtime.ljust(32, b"\x00")) + Op.RETURN(
        0, len(child_runtime)
    )
    factory = pre.deploy_contract(
        code=Om.MSTORE(bytes(initcode), 0)
        + Op.POP(Op.CREATE(0, 0, len(bytes(initcode))))
        + Op.STOP,
    )
    child = compute_create_address(address=factory, nonce=1)
    destroyer = pre.deploy_contract(
        code=Op.POP(Op.CALL(Op.GAS, child, 0, 0, 0, 0, 0)) + Op.STOP
    )

    asserter = pre.deploy_contract(
        code=assert_eq(Op.TXTRACE(0, Spec.TXTRACE_CONTRACTS_DEPLOYED), 1)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_DEPLOYED_ADDRESS), child)
        + assert_eq(
            Op.TXTRACE(0, Spec.TXTRACE_DEPLOYED_CODEHASH),
            keccak256(child_runtime),
        )
        + Op.STOP
    )

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            default_frame(target=factory, gas_limit=BODY_FRAME_GAS),
            default_frame(target=destroyer, gas_limit=BODY_FRAME_GAS),
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
            # EIP-6780: created and destroyed in the same transaction,
            # the child is deleted at transaction end — after the
            # assertion observed its live code.
            child: Account.NONEXISTENT,
        },
    )


def test_txtrace_events_enumeration(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the event enumeration in global emission order across frames —
    the first event emitted by a pre-approval prefix frame — with the
    emitter address, topic count (zero to four), every topic's
    distinct full-width constant, and the data length of each event.

    Rules: R-033, R-034, R-035, R-036, R-037, R-038, R-039, R-040,
    R-088.
    """
    alice = pre.fund_eoa()

    topic = [0x11111111 << (i * 32) | (i + 1) for i in range(4)]
    prefix_emitter = pre.deploy_contract(
        code=Op.LOG1(0, 0, topic[0]) + Op.STOP
    )
    emitter_two_topics = pre.deploy_contract(
        # Five bytes of distinctive, non-zero data.
        code=Op.MSTORE(0, 0xD1D2D3D4D5 << 216)
        + Op.LOG2(0, 5, topic[0], topic[1])
        + Op.STOP
    )
    emitter_no_topics = pre.deploy_contract(code=Op.LOG0(0, 0) + Op.STOP)
    emitter_four_topics = pre.deploy_contract(
        code=Op.MSTORE(0, 0xE1E2E3E4)
        + Op.LOG4(0, 32, topic[0], topic[1], topic[2], topic[3])
        + Op.STOP
    )

    expected_events = [
        (prefix_emitter, [topic[0]], 0),
        (emitter_two_topics, [topic[0], topic[1]], 5),
        (emitter_no_topics, [], 0),
        (emitter_four_topics, topic, 32),
    ]
    checks = assert_eq(
        Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT), len(expected_events)
    )
    for index, (emitter, topics, data_length) in enumerate(expected_events):
        checks += assert_eq(
            Op.TXTRACE(index, Spec.TXTRACE_EVENT_ADDRESS), emitter
        )
        checks += assert_eq(
            Op.TXTRACE(index, Spec.TXTRACE_EVENT_TOPIC_COUNT),
            len(topics),
        )
        for topic_offset, topic_value in enumerate(topics):
            checks += assert_eq(
                Op.TXTRACE(index, Spec.TXTRACE_EVENT_TOPIC0 + topic_offset),
                topic_value,
            )
        checks += assert_eq(
            Op.TXTRACE(index, Spec.TXTRACE_EVENT_DATA_LENGTH),
            data_length,
        )
    asserter = pre.deploy_contract(code=checks + Op.STOP)

    tx = Transaction(
        sender=alice,
        frames=[
            # Emits before payment approval: a prefix event, first in
            # the global order.
            default_frame(target=prefix_emitter),
            verify_frame(),
            default_frame(target=emitter_two_topics),
            default_frame(target=emitter_no_topics),
            default_frame(target=emitter_four_topics),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
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
        post={alice: Account(nonce=1)},
    )


@pytest.mark.parametrize(
    "log_topic_count,queried_param,passes",
    [
        pytest.param(0, Spec.TXTRACE_EVENT_TOPIC0, False, id="log0_topic0"),
        pytest.param(1, Spec.TXTRACE_EVENT_TOPIC0, True, id="log1_topic0"),
        pytest.param(1, Spec.TXTRACE_EVENT_TOPIC1, False, id="log1_topic1"),
        pytest.param(2, Spec.TXTRACE_EVENT_TOPIC1, True, id="log2_topic1"),
        pytest.param(2, Spec.TXTRACE_EVENT_TOPIC2, False, id="log2_topic2"),
        pytest.param(3, Spec.TXTRACE_EVENT_TOPIC2, True, id="log3_topic2"),
        pytest.param(3, Spec.TXTRACE_EVENT_TOPIC3, False, id="log3_topic3"),
        pytest.param(4, Spec.TXTRACE_EVENT_TOPIC3, True, id="log4_topic3"),
    ],
)
def test_txtrace_event_topic_out_of_bounds_halts(
    state_test: StateTestFiller,
    pre: Alloc,
    log_topic_count: int,
    queried_param: int,
    passes: bool,
) -> None:
    """
    Query each topic slot on either side of an event's topic count:
    the last existing topic returns its hand-picked constant, and the
    first missing one exceptionally halts the assertion frame, rolling
    back the body.

    Rules: R-036, R-037, R-038, R-039, R-045.
    """
    sentinel = deploy_sentinel(pre)
    topics = [0x22222222 << (i * 32) | (i + 1) for i in range(4)]
    log_op = [Op.LOG0, Op.LOG1, Op.LOG2, Op.LOG3, Op.LOG4][log_topic_count]
    emitter = pre.deploy_contract(
        code=log_op(0, 0, *topics[:log_topic_count]) + Op.STOP
    )

    if passes:
        queried_topic = queried_param - Spec.TXTRACE_EVENT_TOPIC0
        asserter_code = (
            assert_eq(Op.TXTRACE(0, queried_param), topics[queried_topic])
            + Op.STOP
        )
    else:
        asserter_code = Op.POP(Op.TXTRACE(0, queried_param)) + Op.STOP
    asserter = pre.deploy_contract(code=asserter_code)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=emitter),
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


@pytest.mark.parametrize(
    "wrong_offset",
    [
        pytest.param(0, id="correct_expectations"),
        pytest.param(1, id="wrong_expectation_bites"),
    ],
)
def test_txtrace_gas_scalars(
    state_test: StateTestFiller,
    pre: Alloc,
    wrong_offset: int,
) -> None:
    """
    Pin the gas pre-charge and payer address params: the pre-charge is
    the total amount deducted from the payer — the derived maximum gas
    priced at the maximum fee, hand-computed from the EIP-8141
    accounting with a contract sender — and the payer is the approving
    frame's target.

    Rules: R-041, R-042.
    """
    sender = deploy_approving_sender(pre, balance=10**18)

    frames = [verify_frame(), post_tx_frame()]
    max_cost = contract_sender_max_cost(frames, MAX_FEE_PER_GAS)

    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.TXTRACE(0, Spec.TXTRACE_GAS_PRE_CHARGE),
            max_cost + wrong_offset,
        )
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_GAS_PAYER_ADDRESS), sender)
        + Op.STOP
    )
    frames[-1] = post_tx_frame(target=asserter)

    passes = wrong_offset == 0
    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE_PER_GAS,
        max_priority_fee_per_gas=0,
        frames=frames,
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
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
        post={sender: Account(nonce=2)},
    )


def test_gas_payer_address_paymaster(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Sponsor the transaction from a paymaster contract: the payer
    params report the paymaster — the EIP-8141 payer, distinct from
    the transaction sender — the paymaster's entry carries the full
    escrow deduction, and the sender's balance is untouched (only its
    nonce moves, bumped at payment approval).

    Rule: R-044.
    """
    sender_balance = 10**17
    paymaster_balance = 10**18
    sender = pre.deploy_contract(
        code=Op.APPROVE(0, 0, FrameSpec.APPROVE_EXECUTION),
        balance=sender_balance,
    )
    paymaster = pre.deploy_contract(
        code=Op.APPROVE(0, 0, FrameSpec.APPROVE_PAYMENT),
        balance=paymaster_balance,
    )

    frames = [
        verify_frame(flags=FrameSpec.APPROVE_EXECUTION),
        verify_frame(flags=FrameSpec.APPROVE_PAYMENT, target=paymaster),
        post_tx_frame(),  # placeholder for gas accounting
    ]
    max_cost = contract_sender_max_cost(frames, MAX_FEE_PER_GAS)

    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.TXTRACE(0, Spec.TXTRACE_GAS_PAYER_ADDRESS), paymaster
        )
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_GAS_PRE_CHARGE), max_cost)
        # The paymaster is the only balance change: the sender's
        # balance is untouched.
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED), 1)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_BALANCE_ADDRESS), paymaster)
        + assert_eq(
            Op.TXTRACE(0, Spec.TXTRACE_BALANCE_BEFORE),
            paymaster_balance,
        )
        + assert_eq(
            Op.TXTRACE(0, Spec.TXTRACE_BALANCE_AFTER),
            paymaster_balance - max_cost,
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
            payer=paymaster,
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
            sender: Account(nonce=2, balance=sender_balance),
        },
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
        pytest.param(Spec.TXTRACE_BALANCES_CHANGED, id="balances_changed"),
        pytest.param(Spec.TXTRACE_SLOTS_CHANGED, id="slots_changed"),
        pytest.param(Spec.TXTRACE_CONTRACTS_DEPLOYED, id="contracts_deployed"),
        pytest.param(Spec.TXTRACE_EVENTS_COUNT, id="events_count"),
        pytest.param(Spec.TXTRACE_GAS_PRE_CHARGE, id="gas_pre_charge"),
        pytest.param(Spec.TXTRACE_GAS_PAYER_ADDRESS, id="gas_payer_address"),
    ],
)
def test_txtrace_reserved_in2_nonzero_halts(
    state_test: StateTestFiller,
    pre: Alloc,
    param: int,
    reserved_input: int,
) -> None:
    """
    Supply a non-zero value to the reserved index operand of every
    scalar TXTRACE param: each exceptionally halts the assertion
    frame, while the zero control passes.

    Rule: R-085.
    """
    sentinel = deploy_sentinel(pre)
    asserter = pre.deploy_contract(
        code=Op.POP(Op.TXTRACE(reserved_input, param)) + Op.STOP
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
    "param,index,passes",
    [
        pytest.param(
            Spec.TXTRACE_BALANCE_ADDRESS, 1, True, id="balances_last"
        ),
        pytest.param(
            Spec.TXTRACE_BALANCE_ADDRESS, 2, False, id="balances_count"
        ),
        pytest.param(
            Spec.TXTRACE_BALANCE_BEFORE,
            3,
            False,
            id="balances_count_plus_one",
        ),
        pytest.param(
            Spec.TXTRACE_BALANCE_AFTER,
            MAX_UINT256,
            False,
            id="balances_max_index",
        ),
        pytest.param(Spec.TXTRACE_SLOT_KEY, 0, True, id="slots_last"),
        pytest.param(Spec.TXTRACE_SLOT_ADDRESS, 1, False, id="slots_count"),
        pytest.param(
            Spec.TXTRACE_SLOT_VALUE_AFTER,
            MAX_UINT256,
            False,
            id="slots_max_index",
        ),
        pytest.param(
            Spec.TXTRACE_DEPLOYED_ADDRESS, 0, True, id="deployed_last"
        ),
        pytest.param(
            Spec.TXTRACE_DEPLOYED_CODEHASH,
            1,
            False,
            id="deployed_count",
        ),
        # The body's events: the EIP-7708 transfer log of the value
        # transfer, then the factory's LOG0 — two entries.
        pytest.param(Spec.TXTRACE_EVENT_ADDRESS, 1, True, id="events_last"),
        pytest.param(
            Spec.TXTRACE_EVENT_TOPIC_COUNT, 2, False, id="events_count"
        ),
        pytest.param(
            Spec.TXTRACE_EVENT_DATA_LENGTH,
            MAX_UINT256,
            False,
            id="events_max_index",
        ),
    ],
)
def test_txtrace_index_out_of_bounds_halts(
    state_test: StateTestFiller,
    pre: Alloc,
    param: int,
    index: int,
    passes: bool,
) -> None:
    """
    Probe every indexed TXTRACE family at its exact boundary over a
    body with two balance changes, one storage change, one deployment,
    and one event: the last valid index succeeds, and the count, one
    above it, and the maximum index each exceptionally halt.

    Rule: R-045.
    """
    alice = pre.fund_eoa()
    bob = pre.fund_eoa(amount=1)
    writer = pre.deploy_contract(code=Op.SSTORE(0, 1) + Op.STOP)
    initcode = Op.MSTORE8(0, 0xFE) + Op.RETURN(0, 1)
    factory = pre.deploy_contract(
        code=Op.MSTORE(0, bytes(initcode).ljust(32, b"\x00"))
        + Op.POP(Op.CREATE(0, 0, len(bytes(initcode))))
        + Op.LOG0(0, 0)
        + Op.STOP
    )

    asserter = pre.deploy_contract(
        code=Op.POP(Op.TXTRACE(index, param)) + Op.STOP
    )

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            sender_frame(target=bob, value=10**15),
            default_frame(target=writer, gas_limit=BODY_FRAME_GAS),
            default_frame(target=factory, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
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
            # The transfer survives only if the assertion passed.
            bob: Account(balance=1 + 10**15 if passes else 1),
        },
    )


@pytest.mark.parametrize(
    "param,passes",
    [
        pytest.param(Spec.TXTRACE_GAS_PAYER_ADDRESS, True, id="last_defined"),
        pytest.param(
            Spec.FIRST_UNDEFINED_TXTRACE_PARAM,
            False,
            id="first_undefined",
        ),
        pytest.param(0xFF, False, id="high_undefined"),
        pytest.param(MAX_UINT256, False, id="max_undefined"),
    ],
)
def test_txtrace_undefined_param_halts(
    state_test: StateTestFiller,
    pre: Alloc,
    param: int,
    passes: bool,
) -> None:
    """
    Query params on either side of the defined range: the last defined
    param succeeds while 0x16, 0xFF, and the maximum selector each
    exceptionally halt, per the FRAMEPARAM precedent the spec models
    TXTRACE on.

    Rule: R-016.
    """
    sentinel = deploy_sentinel(pre)
    asserter = pre.deploy_contract(code=Op.POP(Op.TXTRACE(0, param)) + Op.STOP)

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


def test_txtrace_identical_across_post_tx_frames(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Run the same complete diff assertion in two consecutive `POST_TX`
    frames: the suffix placement guarantees both observe the final
    body outcome, and the first frame's reads change nothing the
    second can see.

    Rule: R-100.
    """
    alice = pre.fund_eoa()
    writer = pre.deploy_contract(code=Op.SSTORE(3, 0x33) + Op.STOP)

    asserter = pre.deploy_contract(
        code=assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOTS_CHANGED), 1)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOT_ADDRESS), writer)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOT_KEY), 3)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_SLOT_VALUE_AFTER), 0x33)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT), 0)
        + Op.STOP
    )

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            default_frame(target=writer, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
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
        post={writer: Account(storage={3: 0x33})},
    )


def test_txtrace_balance_conservation_invariant(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Assert value conservation over the whole balance enumeration: with
    no burns and no coinbase credit yet, the sum of all before values
    must exceed the sum of all after values by exactly the gas
    pre-charge — every internal transfer nets to zero.

    The invariant is computed entirely in-frame, so it holds for any
    sender kind without hand-computing the intrinsic cost.

    Rule: R-041 (with the State Difference Semantics).
    """
    alice = pre.fund_eoa()
    bob = pre.fund_eoa(amount=1)
    charlie = pre.fund_eoa(amount=1)
    # A two-hop transfer: alice funds the splitter, which forwards
    # part to bob and part to charlie.
    splitter = pre.deploy_contract(
        code=Op.POP(Op.CALL(Op.GAS, bob, 10**12, 0, 0, 0, 0))
        + Op.POP(Op.CALL(Op.GAS, charlie, 3 * 10**12, 0, 0, 0, 0))
        + Op.STOP
    )

    # Memory: 0x00 = loop index, 0x20 = sum of befores, 0x40 = sum of
    # afters.
    asserter = pre.deploy_contract(
        code=Op.MSTORE(0x00, 0)
        + Op.MSTORE(0x20, 0)
        + Op.MSTORE(0x40, 0)
        + While(
            body=Op.MSTORE(
                0x20,
                Op.ADD(
                    Op.MLOAD(0x20),
                    Op.TXTRACE(Op.MLOAD(0x00), Spec.TXTRACE_BALANCE_BEFORE),
                ),
            )
            + Op.MSTORE(
                0x40,
                Op.ADD(
                    Op.MLOAD(0x40),
                    Op.TXTRACE(Op.MLOAD(0x00), Spec.TXTRACE_BALANCE_AFTER),
                ),
            )
            + Op.MSTORE(0x00, Op.ADD(Op.MLOAD(0x00), 1)),
            condition=Op.LT(
                Op.MLOAD(0x00),
                Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED),
            ),
        )
        + assert_eq(
            Op.MLOAD(0x20),
            Op.ADD(
                Op.MLOAD(0x40),
                Op.TXTRACE(0, Spec.TXTRACE_GAS_PRE_CHARGE),
            ),
        )
        + Op.STOP
    )

    tx = Transaction(
        sender=alice,
        frames=[
            verify_frame(),
            sender_frame(
                target=splitter,
                value=10**15,
                gas_limit=BODY_FRAME_GAS,
            ),
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
            bob: Account(balance=1 + 10**12),
            charlie: Account(balance=1 + 3 * 10**12),
        },
    )
