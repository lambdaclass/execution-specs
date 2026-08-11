"""
Tests for the `APPROVE` instruction of
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

`APPROVE` exits the current call frame successfully and updates the
transaction-scoped approval context: execution approval admits later
`SENDER` frames, and payment approval escrows the transaction's
maximum cost from the approving frame's resolved target. A refused
approval reverts the requesting call frame; in a non-`VERIFY` frame
the revert is catchable and the transaction continues.
"""

from typing import Any, Dict

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    AuthorizationTuple,
    Bytecode,
    Bytes,
    Fork,
    FrameReceipt,
    FrameSignature,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
)

from .helpers import default_frame, sender_frame, verify_frame
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

# EIP-8141 is slated for the fork after Amsterdam, so fixtures are
# labeled with the pseudo `Bogota` fork (Amsterdam + EIP-8141), even
# though the spec prototypes the EIP inside the Amsterdam fork module.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

SLOT_MARKER = 0x01
"""Storage slot for the marker written before a probed `APPROVE`."""

SLOT_STATUS = 0x02
"""Storage slot recording a child call's success flag plus one."""

SLOT_EXECUTED = 0x01
"""Storage slot used by canary contracts to record execution."""

MARKER = 0xC0DE
"""Distinctive nonzero marker value."""

MARKER_WORD = int("c0de" * 16, 16)
"""Distinctive 32-byte marker word for return-data checks."""

# A fresh SSTORE costs STATE_BYTES_PER_STORAGE_SET * COST_PER_STATE_BYTE
# of state gas under EIP-8037, and a frame transaction holds no state
# gas reservoir, so frames whose code writes storage need room for the
# writes.
PROBE_FRAME_GAS = 500_000

CREATE_FRAME_GAS = 1_000_000
"""Gas limit of the frame attempting a contract creation."""

MAX_FEE = 1_000
"""Explicit maximum fee, making the escrow arithmetic visible."""

VERIFY_FRAME_GAS = 100_000
"""Gas limit of approving frames in the balance boundary test."""


def scope_word(scope: int) -> Bytes:
    """Encode an approval scope as a 32-byte frame data word."""
    return Bytes(scope.to_bytes(32, "big"))


def branched_sender_code(probe: Bytecode, canonical_scope: int) -> Bytecode:
    """
    Build sender code with a canonical branch and a probe branch.

    Empty calldata approves `canonical_scope` — safe inside `VERIFY`
    frames, since the branch only jumps and approves. Non-empty
    calldata runs `probe`, which must end in a terminating
    instruction.
    """
    canonical_offset = 5 + len(probe)
    assert canonical_offset < 256, "jump destination must fit a PUSH1"
    return (
        Op.JUMPI(canonical_offset, Op.ISZERO(Op.CALLDATASIZE))
        + probe
        + Op.JUMPDEST
        + Op.APPROVE(0, 0, canonical_scope)
    )


def scope_probing_sender_code(canonical_scope: int) -> Bytecode:
    """
    Build sender code whose probe branch records a marker and approves
    the scope read from the first frame data word: a refused approval
    reverts the frame and discards the marker.
    """
    return branched_sender_code(
        Op.SSTORE(SLOT_MARKER, MARKER) + Op.APPROVE(0, 0, Op.CALLDATALOAD(0)),
        canonical_scope,
    )


def canary_contract(pre: Alloc) -> Address:
    """Deploy a contract recording its execution in a storage marker."""
    return pre.deploy_contract(code=Op.SSTORE(SLOT_EXECUTED, MARKER) + Op.STOP)


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "approved_scope",
    [
        pytest.param(None, id="no_approval"),
        pytest.param(Spec.APPROVE_EXECUTION, id="execution_only"),
    ],
)
def test_payer_never_set(
    state_test: StateTestFiller,
    pre: Alloc,
    approved_scope: int | None,
) -> None:
    """
    Reject a frame transaction in which every frame succeeds but no
    frame ever approves payment — including one that approved
    execution only.

    The failure class is the execution one: the transaction fails
    only after all frames ran, and the storing target's state is
    discarded.
    """
    sender = pre.fund_eoa()
    target = canary_contract(pre)
    if approved_scope is None:
        frames = [default_frame(target=target, gas_limit=PROBE_FRAME_GAS)]
    else:
        frames = [
            verify_frame(flags=approved_scope),
            sender_frame(target=target, gas_limit=PROBE_FRAME_GAS),
        ]

    tx = Transaction(
        sender=sender,
        frames=frames,
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=0),
            target: Account(storage={SLOT_EXECUTED: 0}),
        },
    )


@pytest.mark.exception_test
def test_payment_before_execution_approval(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Reject a frame transaction whose first frame approves payment
    only: payment approval requires execution approval first, so the
    payer's default code reverts and the `VERIFY` frame invalidates
    the transaction.
    """
    sender = pre.fund_eoa()
    payer = pre.fund_eoa()

    tx = Transaction(
        sender=sender,
        frames=[verify_frame(flags=Spec.APPROVE_PAYMENT, target=payer)],
        signatures=[
            FrameSignature(scheme=Spec.SCHEME_SECP256K1, signer=Bytes(sender)),
            FrameSignature(
                scheme=Spec.SCHEME_SECP256K1,
                signer=Bytes(payer),
                secret_key=payer.key,
            ),
        ],
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=0),
            payer: Account(nonce=0),
        },
    )


@pytest.mark.parametrize(
    "frame_flags,probe_scope",
    [
        pytest.param(Spec.APPROVE_EXECUTION_AND_PAYMENT, 0, id="zero_scope"),
        pytest.param(
            Spec.APPROVE_EXECUTION_AND_PAYMENT, 4, id="batch_bit_scope"
        ),
        pytest.param(
            Spec.APPROVE_EXECUTION_AND_PAYMENT,
            5,
            id="batch_bit_and_payment_scope",
        ),
        pytest.param(
            Spec.APPROVE_PAYMENT, 3, id="scope_exceeds_payment_only_flags"
        ),
        pytest.param(Spec.APPROVE_PAYMENT, 2, id="execution_not_in_flags"),
        pytest.param(Spec.APPROVE_EXECUTION, 1, id="payment_not_in_flags"),
        pytest.param(
            Spec.APPROVE_PAYMENT,
            1,
            id="payment_before_execution_catchable",
        ),
    ],
)
def test_approve_scope_refusals(
    state_test: StateTestFiller,
    pre: Alloc,
    frame_flags: int,
    probe_scope: int,
) -> None:
    """
    Refuse an `APPROVE` whose scope is empty, carries bits beyond the
    approval mask, exceeds the frame's allowed flags, or fails the
    execution-first precondition.

    The probing `DEFAULT` frame reverts — discarding its marker — and
    the transaction continues: the canonical approval in the later
    `VERIFY` frame still succeeds, proving the refused attempt left
    no approval state behind.
    """
    sender = pre.deploy_contract(
        code=scope_probing_sender_code(Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[
            default_frame(
                flags=frame_flags,
                data=scope_word(probe_scope),
                gas_limit=PROBE_FRAME_GAS,
            ),
            verify_frame(),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_FAILURE),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=2, storage={SLOT_MARKER: 0})},
    )


def test_approve_strict_subset_scope(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Approve a strict subset of the frame's allowed flags: a frame
    allowing execution and payment may approve execution alone.

    The marker written next to the subset approval survives, a later
    payment-only canonical branch completes the approvals, and a
    trailing `SENDER` frame proves the execution approval took
    effect.
    """
    sender = pre.deploy_contract(
        code=scope_probing_sender_code(Spec.APPROVE_PAYMENT),
        balance=10**18,
    )
    canary = canary_contract(pre)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[
            default_frame(
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                data=scope_word(Spec.APPROVE_EXECUTION),
                gas_limit=PROBE_FRAME_GAS,
            ),
            verify_frame(flags=Spec.APPROVE_PAYMENT),
            sender_frame(target=canary, gas_limit=PROBE_FRAME_GAS),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=2, storage={SLOT_MARKER: MARKER}),
            canary: Account(storage={SLOT_EXECUTED: MARKER}),
        },
    )


def test_approve_duplicate_scopes(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Refuse every duplicate approval after the canonical one: both
    scopes again, execution again, and payment again from a third
    party.

    Each duplicate reverts its own `DEFAULT` frame — discarding the
    probe marker — while the original approvals survive: the payer
    stays the sender and one execution approval still powers two
    later `SENDER` frames at different targets.
    """
    sender = pre.deploy_contract(
        code=scope_probing_sender_code(Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )
    other_payer = pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec.APPROVE_PAYMENT),
        balance=10**18,
    )
    canary_a = canary_contract(pre)
    canary_b = canary_contract(pre)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[
            verify_frame(),
            default_frame(
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                data=scope_word(Spec.APPROVE_EXECUTION_AND_PAYMENT),
                gas_limit=PROBE_FRAME_GAS,
            ),
            default_frame(
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                data=scope_word(Spec.APPROVE_EXECUTION),
                gas_limit=PROBE_FRAME_GAS,
            ),
            default_frame(
                flags=Spec.APPROVE_PAYMENT,
                target=other_payer,
                gas_limit=PROBE_FRAME_GAS,
            ),
            sender_frame(target=canary_a, gas_limit=PROBE_FRAME_GAS),
            sender_frame(target=canary_b, gas_limit=PROBE_FRAME_GAS),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_FAILURE),
                FrameReceipt(status=Spec.STATUS_FAILURE),
                FrameReceipt(status=Spec.STATUS_FAILURE),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=2, storage={SLOT_MARKER: 0}),
            canary_a: Account(storage={SLOT_EXECUTED: MARKER}),
            canary_b: Account(storage={SLOT_EXECUTED: MARKER}),
            # The refused third-party payment left the other payer's
            # escrow untouched.
            other_payer: Account(nonce=1, balance=10**18),
        },
    )


def test_approve_return_data(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Observe an `APPROVE`'s designated memory region as the caller's
    return data.

    The sender self-calls; the child writes a marker word to memory
    and approves both scopes with a 32-byte return region. The parent
    pins the returned size and bytes — any operand-order swap between
    offset and length changes both — and a trailing `SENDER` frame
    proves the depth-two approval took effect.
    """
    child = Op.MSTORE(0, MARKER_WORD) + Op.APPROVE(
        0, 32, Spec.APPROVE_EXECUTION_AND_PAYMENT
    )
    parent = (
        Op.POP(Op.CALL(Op.GAS, Op.ADDRESS, 0, 0, 1, 0, 0))
        + Op.SSTORE(SLOT_STATUS, Op.RETURNDATASIZE)
        + Op.RETURNDATACOPY(0, 0, 32)
        + Op.SSTORE(SLOT_MARKER, Op.MLOAD(0))
        + Op.STOP
    )
    child_offset = 4 + len(parent)
    assert child_offset < 256, "jump destination must fit a PUSH1"
    sender = pre.deploy_contract(
        code=Op.JUMPI(child_offset, Op.CALLDATASIZE)
        + parent
        + Op.JUMPDEST
        + child,
        balance=10**18,
    )
    canary = canary_contract(pre)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[
            default_frame(
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=PROBE_FRAME_GAS,
            ),
            sender_frame(target=canary, gas_limit=PROBE_FRAME_GAS),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                nonce=2,
                storage={
                    SLOT_STATUS: 32,
                    SLOT_MARKER: MARKER_WORD,
                },
            ),
            canary: Account(storage={SLOT_EXECUTED: MARKER}),
        },
    )


def test_approve_from_child_call_reverts_callee(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Revert only the callee when a called contract's `APPROVE` runs
    with an address other than the frame's resolved target.

    The probing frame stores the child call's success flag plus one,
    and the canonical approval afterwards still succeeds — the
    refused attempt left no approval state behind.
    """
    foreign = pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec.APPROVE_PAYMENT),
        balance=10**18,
    )
    probe = (
        Op.SSTORE(
            SLOT_STATUS,
            Op.ADD(Op.CALL(Op.GAS, foreign, 0, 0, 0, 0, 0), 1),
        )
        + Op.STOP
    )
    sender = pre.deploy_contract(
        code=branched_sender_code(probe, Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )
    canary = canary_contract(pre)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[
            default_frame(
                flags=Spec.APPROVE_PAYMENT,
                data=Bytes(b"\x01"),
                gas_limit=PROBE_FRAME_GAS,
            ),
            verify_frame(),
            sender_frame(target=canary, gas_limit=PROBE_FRAME_GAS),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=2, storage={SLOT_STATUS: 1}),
            canary: Account(storage={SLOT_EXECUTED: MARKER}),
        },
    )


def test_approve_from_delegatecall(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Approve through a `DELEGATECALL`ed helper: the executing address
    is preserved, so the helper's `APPROVE` matches the resolved
    target and both approvals take effect from call depth two.
    """
    helper = pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT)
    )
    probe = (
        Op.SSTORE(
            SLOT_STATUS,
            Op.ADD(Op.DELEGATECALL(Op.GAS, helper, 0, 0, 0, 0), 1),
        )
        + Op.STOP
    )
    sender = pre.deploy_contract(
        code=branched_sender_code(probe, Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )
    canary = canary_contract(pre)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[
            default_frame(
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                data=Bytes(b"\x01"),
                gas_limit=PROBE_FRAME_GAS,
            ),
            sender_frame(target=canary, gas_limit=PROBE_FRAME_GAS),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=2, storage={SLOT_STATUS: 2}),
            canary: Account(storage={SLOT_EXECUTED: MARKER}),
        },
    )


def test_approve_in_initcode_reverts_create(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Revert a contract creation whose initcode calls `APPROVE`: the
    executing address is the new contract, never the frame's resolved
    target.

    The probing frame stores the creation result's zero-flag plus
    one, and the canonical approval afterwards still succeeds.
    """
    initcode = Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT)
    initcode_word = int.from_bytes(bytes(initcode).ljust(32, b"\x00"), "big")
    probe = (
        Op.MSTORE(0, initcode_word)
        + Op.SSTORE(
            SLOT_STATUS,
            Op.ADD(Op.ISZERO(Op.CREATE(0, 0, len(initcode))), 1),
        )
        + Op.STOP
    )
    sender = pre.deploy_contract(
        code=branched_sender_code(probe, Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )
    canary = canary_contract(pre)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[
            default_frame(
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                data=Bytes(b"\x01"),
                gas_limit=CREATE_FRAME_GAS,
            ),
            verify_frame(),
            sender_frame(target=canary, gas_limit=PROBE_FRAME_GAS),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            # The failed creation still bumps the creator's nonce, on
            # top of the deployment and payment-approval increments.
            sender: Account(nonce=3, storage={SLOT_STATUS: 2}),
            canary: Account(storage={SLOT_EXECUTED: MARKER}),
        },
    )


@pytest.mark.parametrize(
    "offset,length,expanded_bytes",
    [
        pytest.param(0, 0, 0, id="empty_region"),
        pytest.param(0, 64, 64, id="two_words"),
        pytest.param(2**32, 0, 0, id="empty_region_at_huge_offset"),
    ],
)
def test_approve_memory_expansion_gas(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    offset: int,
    length: int,
    expanded_bytes: int,
) -> None:
    """
    Charge `APPROVE` only the memory expansion of its return-data
    region: no base cost, and a zero-length region expands nothing
    even at a huge offset.

    The frame receipt's gas pins the total: the warm access charged
    for the frame's resolved target at entry — the sender seeds the
    warm journal — plus the sender code with its expansion metadata.
    """
    code = Op.APPROVE(
        offset=offset,
        size=length,
        scope=Spec.APPROVE_EXECUTION_AND_PAYMENT,
        new_memory_size=expanded_bytes,
        old_memory_size=0,
    )
    sender = pre.deploy_contract(code=code, balance=10**18)
    frame_gas = fork.gas_costs().WARM_ACCESS + code.gas_cost(fork)

    tx = Transaction(
        sender=sender,
        nonce=1,
        frames=[verify_frame()],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS, gas_used=frame_gas),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=2)},
    )


@pytest.mark.parametrize(
    "writer_code,error",
    [
        pytest.param(
            Op.SSTORE(0, 1) + Op.STOP,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="sstore",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            Op.LOG0(0, 0) + Op.STOP,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="log0",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            Op.TSTORE(0, 1) + Op.STOP,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="tstore",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            Op.POP(Op.CALL(Op.GAS, Address(0x1234), 1, 0, 0, 0, 0)) + Op.STOP,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="call_with_value",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            Op.POP(Op.CREATE(0, 0, 0)) + Op.STOP,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="create",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            Op.POP(Op.SLOAD(0)) + Op.POP(Op.BALANCE(Op.ADDRESS)) + Op.STOP,
            None,
            id="read_only_control",
        ),
    ],
)
def test_verify_frame_static_restrictions(
    state_test: StateTestFiller,
    pre: Alloc,
    writer_code: Bytecode,
    error: TransactionException | None,
) -> None:
    """
    Fail a `VERIFY` frame on any state mutation other than `APPROVE`:
    the frame executes as a static call, so a write halts it and the
    transaction is invalid. The read-only control passes.
    """
    sender = pre.fund_eoa()
    writer = pre.deploy_contract(code=writer_code, balance=1)

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            verify_frame(flags=Spec.APPROVE_NONE, target=writer),
        ],
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=0 if error else 1),
            writer: Account(storage={0: 0}),
        },
    )


@pytest.mark.exception_test
def test_verify_frame_after_sender_frame_unwinds(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Invalidate the whole transaction when a trailing `VERIFY` frame
    reverts after `SENDER` frames already executed: the canary write
    is discarded, the sender's nonce never moves, and its balance is
    fully restored — inclusion-time execution can still be voided
    late.
    """
    funding = 10**18
    sender = pre.fund_eoa(amount=funding)
    canary = canary_contract(pre)
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            sender_frame(target=canary, gas_limit=PROBE_FRAME_GAS),
            verify_frame(flags=Spec.APPROVE_NONE, target=reverter),
        ],
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=0, balance=funding),
            canary: Account(storage={SLOT_EXECUTED: 0}),
        },
    )


@pytest.mark.parametrize(
    "deficit,error",
    [
        pytest.param(0, None, id="balance_at_max_cost"),
        pytest.param(
            1,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="balance_below_max_cost",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@pytest.mark.parametrize(
    "payer_is_sender",
    [
        pytest.param(True, id="sender_pays"),
        pytest.param(False, id="paymaster_pays"),
    ],
)
def test_payment_balance_boundary(
    state_test: StateTestFiller,
    pre: Alloc,
    payer_is_sender: bool,
    deficit: int,
    error: TransactionException | None,
) -> None:
    """
    Fund the payer with exactly the transaction's maximum cost, or
    one wei less.

    The maximum cost is the derived transaction gas limit priced at
    the maximum fee, hand-computed from the base and per-frame
    constants plus the frame gas — contract senders and payers carry
    no signature entries and the frames no data. The exact balance is
    approved; one below reverts the approving `VERIFY` frame and
    invalidates the transaction, pinning the product on both sides.
    """
    frame_count = 1 if payer_is_sender else 2
    max_gas = (
        Spec.FRAME_TX_INTRINSIC_COST
        + frame_count * Spec.FRAME_TX_PER_FRAME_COST
        + frame_count * VERIFY_FRAME_GAS
    )
    max_cost = max_gas * MAX_FEE

    if payer_is_sender:
        sender = pre.deploy_contract(
            code=Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT),
            balance=max_cost - deficit,
        )
        payer = sender
        frames = [verify_frame(gas_limit=VERIFY_FRAME_GAS)]
    else:
        sender = pre.deploy_contract(
            code=Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION),
        )
        payer = pre.deploy_contract(
            code=Op.APPROVE(0, 0, Spec.APPROVE_PAYMENT),
            balance=max_cost - deficit,
        )
        frames = [
            verify_frame(
                flags=Spec.APPROVE_EXECUTION, gas_limit=VERIFY_FRAME_GAS
            ),
            verify_frame(
                flags=Spec.APPROVE_PAYMENT,
                target=payer,
                gas_limit=VERIFY_FRAME_GAS,
            ),
        ]

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=0,
        frames=frames,
        error=error,
        expected_receipt=None if error else TransactionReceipt(payer=payer),
    )

    post = {sender: Account(nonce=1 if error else 2)}
    if not payer_is_sender:
        # The sender was deployed without funds and pays nothing; the
        # nonce increment lands on the sender, never the payer.
        post[sender] = Account(nonce=1 if error else 2, balance=0)
        post[payer] = Account(nonce=1)

    state_test(pre=pre, tx=tx, post=post)


@pytest.mark.parametrize(
    "halting_op",
    [
        pytest.param(Op.POP(Op.TXPARAM(0)), id="txparam"),
        pytest.param(Op.POP(Op.FRAMEDATALOAD(0, 0)), id="framedataload"),
        pytest.param(Op.FRAMEDATACOPY(0, 0, 0, 0), id="framedatacopy"),
        pytest.param(Op.POP(Op.FRAMEPARAM(0, 0)), id="frameparam"),
        pytest.param(Op.POP(Op.SIGPARAM(0, 0)), id="sigparam"),
        pytest.param(
            Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT),
            id="approve",
        ),
    ],
)
@pytest.mark.parametrize("tx_type", [0, 2, 4])
def test_opcodes_undefined_in_other_tx_types(
    state_test: StateTestFiller,
    pre: Alloc,
    tx_type: int,
    halting_op: Bytecode,
) -> None:
    """
    Halt exceptionally on every frame-transaction instruction when
    executed outside a frame transaction.

    The probe writes a marker before the instruction, so an
    instruction that returned a value instead of halting would leave
    the marker behind; the halt also consumes the transaction's full
    gas limit, pinned through the receipt.
    """
    gas_limit = 500_000
    sender = pre.fund_eoa()
    probe = pre.deploy_contract(
        code=Op.SSTORE(SLOT_STATUS, MARKER) + halting_op + Op.STOP
    )

    tx_kwargs: Dict[str, Any] = dict(
        sender=sender,
        to=probe,
        gas_limit=gas_limit,
        expected_receipt=TransactionReceipt(cumulative_gas_used=gas_limit),
    )
    if tx_type == 2:
        tx_kwargs.update(max_fee_per_gas=10, max_priority_fee_per_gas=0)
    elif tx_type == 4:
        authority = pre.fund_eoa()
        tx_kwargs.update(
            max_fee_per_gas=10,
            max_priority_fee_per_gas=0,
            authorization_list=[
                AuthorizationTuple(
                    address=Address(0), nonce=0, signer=authority
                )
            ],
        )
    tx = Transaction(**tx_kwargs)
    assert tx.ty == tx_type

    state_test(
        pre=pre,
        tx=tx,
        post={probe: Account(storage={SLOT_STATUS: 0})},
    )
