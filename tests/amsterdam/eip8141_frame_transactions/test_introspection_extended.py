"""
Extended introspection tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

Pin the introspection instructions' gas costs, their behavior from
child call contexts and initcode, their stack underflow halts, reads
of not-yet-executed frames, and the canonical signature hash — both
its exact preimage and the elision of empty-message signature bytes.
"""

from typing import List, TypeAlias, Union

import pytest
from execution_testing import (
    EOA,
    Account,
    Alloc,
    Bytecode,
    Bytes,
    CodeGasMeasure,
    Fork,
    Frame,
    FrameReceipt,
    FrameSignature,
    Op,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
    compute_create_address,
    keccak256,
)
from spec256k1 import PrivateKey

from .helpers import default_frame, verify_frame
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

# EIP-8141 is slated for the fork after Amsterdam, so fixtures are
# labeled with the pseudo `Bogota` fork (Amsterdam + EIP-8141), even
# though the spec prototypes the EIP inside the Amsterdam fork module.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

SLOT_RESULT = 0x01
"""Storage slot the probe contract writes what it read into."""

MARKER = 0xC0DE
"""Distinctive nonzero marker value."""

# A fresh SSTORE costs STATE_BYTES_PER_STORAGE_SET * COST_PER_STATE_BYTE
# of state gas under EIP-8037, and a frame transaction holds no state
# gas reservoir, so probe frames writing several slots need room for
# every write.
PROBE_FRAME_GAS = 500_000

CREATE_FRAME_GAS = 2_000_000
"""Gas limit of the frame creating an introspecting contract."""

PRIORITY_FEE = 2
MAX_FEE = 1_000

FUTURE_FRAME_DATA = Bytes(b"\x5a" * 32)
"""Data of the not-yet-executed frame read by an earlier probe."""


@pytest.mark.parametrize(
    "measured,expected_cost,extra_stack_items",
    [
        pytest.param(Op.TXPARAM(0), Spec.GAS_TXPARAM, 1, id="txparam"),
        pytest.param(
            Op.FRAMEDATALOAD(0, 1),
            Spec.GAS_FRAMEDATALOAD,
            1,
            id="framedataload",
        ),
        pytest.param(
            Op.FRAMEPARAM(0, Spec.FRAMEPARAM_MODE),
            Spec.GAS_FRAMEPARAM,
            1,
            id="frameparam",
        ),
        pytest.param(
            Op.SIGPARAM(0, Spec.SIGPARAM_SCHEME),
            Spec.GAS_SIGPARAM,
            1,
            id="sigparam",
        ),
    ],
)
def test_introspection_gas_costs(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    measured: Bytecode,
    expected_cost: int,
    extra_stack_items: int,
) -> None:
    """
    Measure each introspection instruction's constant gas cost at
    runtime against the spec's literal value; the operand pushes are
    subtracted as overhead.
    """
    sender = pre.fund_eoa()
    probe_code = CodeGasMeasure(
        code=measured,
        overhead_cost=measured.gas_cost(fork) - expected_cost,
        extra_stack_items=extra_stack_items,
        sstore_key=SLOT_RESULT,
    )
    probe = pre.deploy_contract(code=probe_code)

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(target=probe, gas_limit=PROBE_FRAME_GAS),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={probe: Account(storage={SLOT_RESULT: expected_cost})},
    )


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(0, id="empty"),
        pytest.param(1, id="single_byte"),
        pytest.param(31, id="word_minus_one"),
        pytest.param(32, id="single_word"),
        pytest.param(33, id="word_plus_one"),
        pytest.param(64, id="two_words"),
    ],
)
def test_framedatacopy_gas(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    length: int,
) -> None:
    """
    Measure the frame data copy's gas: the base cost, three per copied
    word, and the standard memory expansion — the yellow paper's
    linear and quadratic terms, hand-applied per arm.

    Copying past the end of the frame's data zero-fills and costs the
    same, so the empty probe frame data exercises every length.
    """
    words = (length + 31) // 32
    gas_costs = fork.gas_costs()
    expansion = gas_costs.MEMORY_PER_WORD * words + (words * words) // 512
    expected_cost = (
        Spec.GAS_FRAMEDATACOPY_BASE
        + Spec.GAS_FRAMEDATACOPY_PER_WORD * words
        + expansion
    )
    measured = Op.FRAMEDATACOPY(0, 0, length, 1)
    # Four operand pushes are the only overhead.
    overhead = (Op.PUSH1(0) * 4).gas_cost(fork)

    sender = pre.fund_eoa()
    probe_code = CodeGasMeasure(
        code=measured,
        overhead_cost=overhead,
        extra_stack_items=0,
        sstore_key=SLOT_RESULT,
    )
    probe = pre.deploy_contract(code=probe_code)

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(target=probe, gas_limit=PROBE_FRAME_GAS),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={probe: Account(storage={SLOT_RESULT: expected_cost})},
    )


def test_framedatacopy_huge_length_out_of_gas(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Halt a frame data copy whose length's memory expansion exceeds
    any reachable gas; the probe's pre-write is rolled back and the
    frame forfeits its whole limit.
    """
    sender = pre.fund_eoa()
    probe = pre.deploy_contract(
        code=Op.SSTORE(SLOT_RESULT, MARKER)
        + Op.FRAMEDATACOPY(0, 0, 2**32, 1)
        + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(target=probe, gas_limit=PROBE_FRAME_GAS),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(
                    status=Spec.STATUS_FAILURE, gas_used=PROBE_FRAME_GAS
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={probe: Account(storage={SLOT_RESULT: 0})},
    )


@pytest.mark.parametrize(
    "offset,expected",
    [
        pytest.param(
            0,
            int.from_bytes(FUTURE_FRAME_DATA, "big") + 1,
            id="first_word",
        ),
        pytest.param(2**256 - 1, 1, id="max_offset_reads_zero"),
    ],
)
def test_framedataload_future_frame_and_max_offset(
    state_test: StateTestFiller,
    pre: Alloc,
    offset: int,
    expected: int,
) -> None:
    """
    Read the data of a frame that has not executed yet — frame data
    is transaction-static, not an execution artifact — and read the
    zero word at the maximal offset without any memory involvement.

    The probe stores the read plus one, so the zero read stays
    distinguishable from a probe that never ran.
    """
    sender = pre.fund_eoa()
    probe = pre.deploy_contract(
        code=Op.SSTORE(
            SLOT_RESULT, Op.ADD(Op.FRAMEDATALOAD(offset, 2), 1)
        )
        + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(target=probe, gas_limit=PROBE_FRAME_GAS),
            default_frame(data=FUTURE_FRAME_DATA),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={probe: Account(storage={SLOT_RESULT: expected})},
    )


@pytest.mark.parametrize(
    "short_stacked",
    [
        pytest.param(Op.TXPARAM, id="txparam_empty_stack"),
        pytest.param(Op.PUSH1(0) + Op.FRAMEDATALOAD, id="framedataload"),
        pytest.param(
            Op.PUSH1(0) + Op.PUSH1(0) + Op.PUSH1(0) + Op.FRAMEDATACOPY,
            id="framedatacopy",
        ),
        pytest.param(Op.PUSH1(0) + Op.FRAMEPARAM, id="frameparam"),
        pytest.param(Op.PUSH1(0) + Op.SIGPARAM, id="sigparam"),
        pytest.param(Op.PUSH1(0) + Op.PUSH1(0) + Op.APPROVE, id="approve"),
    ],
)
def test_introspection_stack_underflow(
    state_test: StateTestFiller,
    pre: Alloc,
    short_stacked: Bytecode,
) -> None:
    """
    Halt each frame transaction instruction when the stack holds one
    item fewer than it pops; the marker written before the underflow
    is rolled back with the frame.
    """
    sender = pre.fund_eoa()
    probe = pre.deploy_contract(
        code=Op.SSTORE(SLOT_RESULT, MARKER) + short_stacked + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(target=probe, gas_limit=PROBE_FRAME_GAS),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={probe: Account(storage={SLOT_RESULT: 0})},
    )


@pytest.mark.parametrize(
    "call_kind",
    [
        pytest.param(Op.CALL, id="call"),
        pytest.param(Op.CALLCODE, id="callcode"),
        pytest.param(Op.DELEGATECALL, id="delegatecall"),
        pytest.param(Op.STATICCALL, id="staticcall"),
    ],
)
def test_introspection_in_subcontexts(
    state_test: StateTestFiller,
    pre: Alloc,
    call_kind: Op,
) -> None:
    """
    Read transaction, frame, and signature information from a child
    call of every kind: the instructions are transaction-scoped, so
    depth-two reads return the same values as depth-one reads,
    regardless of the child's execution context.
    """
    sender = pre.fund_eoa()
    helper = pre.deploy_contract(
        code=Op.MSTORE(0, Op.TXPARAM(Spec.TXPARAM_FRAME_INDEX))
        + Op.MSTORE(32, Op.FRAMEPARAM(1, Spec.FRAMEPARAM_TARGET))
        + Op.MSTORE(64, Op.SIGPARAM(0, Spec.SIGPARAM_SCHEME))
        + Op.RETURN(0, 96)
    )
    if call_kind in (Op.CALL, Op.CALLCODE):
        invocation = call_kind(Op.GAS, helper, 0, 0, 0, 0, 96)
    else:
        invocation = call_kind(Op.GAS, helper, 0, 0, 0, 96)
    dispatcher = pre.deploy_contract(
        code=Op.POP(invocation)
        + Op.SSTORE(SLOT_RESULT, Op.MLOAD(0))
        + Op.SSTORE(SLOT_RESULT + 1, Op.MLOAD(32))
        + Op.SSTORE(SLOT_RESULT + 2, Op.MLOAD(64))
        + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(target=dispatcher, gas_limit=PROBE_FRAME_GAS),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            dispatcher: Account(
                storage={
                    SLOT_RESULT: 1,
                    SLOT_RESULT + 1: dispatcher,
                    SLOT_RESULT + 2: Spec.SCHEME_SECP256K1,
                }
            ),
        },
    )


def test_introspection_in_initcode(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Read transaction-scoped information from creation initcode: the
    created contract stores the executing frame's index plus one into
    its own fresh storage.
    """
    sender = pre.fund_eoa()
    initcode = (
        Op.SSTORE(
            SLOT_RESULT, Op.ADD(Op.TXPARAM(Spec.TXPARAM_FRAME_INDEX), 1)
        )
        + Op.STOP
    )
    initcode_word = int.from_bytes(
        bytes(initcode).ljust(32, b"\x00"), "big"
    )
    dispatcher = pre.deploy_contract(
        code=Op.MSTORE(0, initcode_word)
        + Op.POP(Op.CREATE(0, 0, len(initcode)))
        + Op.STOP
    )
    created = compute_create_address(address=dispatcher, nonce=1)

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(target=dispatcher, gas_limit=CREATE_FRAME_GAS),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            # The probe frame is at index one, so the created account
            # stores two.
            created: Account(storage={SLOT_RESULT: 2}),
        },
    )


def test_frameparam_status_of_unrolled_and_skipped_frames(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Read the failure and skipped statuses of earlier frames after a
    failed atomic batch; the probe stores each status plus one, so a
    zero read stays distinguishable from a probe that never ran.
    """
    sender = pre.fund_eoa()
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))
    skipped_target = pre.deploy_contract(code=Op.STOP)
    probe = pre.deploy_contract(
        code=Op.SSTORE(
            SLOT_RESULT,
            Op.ADD(Op.FRAMEPARAM(1, Spec.FRAMEPARAM_STATUS), 1),
        )
        + Op.SSTORE(
            SLOT_RESULT + 1,
            Op.ADD(Op.FRAMEPARAM(2, Spec.FRAMEPARAM_STATUS), 1),
        )
        + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=reverter,
                gas_limit=100_000,
            ),
            default_frame(target=skipped_target, gas_limit=100_000),
            default_frame(target=probe, gas_limit=PROBE_FRAME_GAS),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            probe: Account(
                storage={
                    SLOT_RESULT: Spec.STATUS_FAILURE + 1,
                    SLOT_RESULT + 1: Spec.STATUS_SKIPPED + 1,
                }
            ),
        },
    )


RlpItem: TypeAlias = Union[int, bytes, List["RlpItem"]]


def rlp_encode(item: RlpItem) -> bytes:
    """
    Encode an item as RLP, built here from the RLP definition so the
    signature hash preimage is derived independently of any library.
    """

    def length_prefix(length: int, offset: int) -> bytes:
        if length < 56:
            return bytes([offset + length])
        length_bytes = length.to_bytes(
            (length.bit_length() + 7) // 8, "big"
        )
        return bytes([offset + 55 + len(length_bytes)]) + length_bytes

    if isinstance(item, int):
        as_bytes = (
            item.to_bytes((item.bit_length() + 7) // 8, "big")
            if item
            else b""
        )
        return rlp_encode(as_bytes)
    if isinstance(item, bytes):
        if len(item) == 1 and item[0] < 0x80:
            return item
        return length_prefix(len(item), 0x80) + item
    encoded = b"".join(rlp_encode(element) for element in item)
    return length_prefix(len(encoded), 0xC0) + encoded


def hand_computed_signature_hash(
    sender: EOA,
    frames: List[Frame],
    signature_fields: List[List[RlpItem]],
    max_priority_fee: int,
    max_fee: int,
) -> bytes:
    """
    Recompute the canonical signature hash from the spec's payload
    field list: the transaction type byte followed by the RLP of the
    nine payload fields, with a null frame target encoded as the
    empty byte string.

    The caller passes the signature entries as raw field lists, with
    any empty-message entry's signature bytes already elided.
    """
    payload: List[RlpItem] = [
        1,  # chain id of the test chain
        1,  # sender nonce
        bytes(sender),
        [
            [
                int(frame.mode),
                int(frame.flags),
                bytes(frame.target) if frame.target is not None else b"",
                int(frame.gas_limit),
                int(frame.value),
                bytes(frame.data),
            ]
            for frame in frames
        ],
        signature_fields,
        max_priority_fee,
        max_fee,
        0,  # max fee per blob gas
        [],  # blob versioned hashes
    ]
    return keccak256(
        bytes([Spec.FRAME_TX_TYPE]) + rlp_encode(payload)
    )


def test_sig_hash_pinned(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the canonical signature hash to a preimage recomputed from
    the spec's field list with an in-test RLP encoder.

    The only signature entry carries an explicit digest, so nothing
    is elided and every payload byte is static; the probe stores the
    hash the transaction reports and the post state compares it with
    the hand-computed value.
    """
    witness = FrameSignature(
        scheme=Spec.SCHEME_ARBITRARY,
        msg=Bytes(b"\x5a" * 32),
        signature=Bytes(b"\xaa\xbb\xcc"),
    )
    sender = pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )
    probe = pre.deploy_contract(
        code=Op.SSTORE(SLOT_RESULT, Op.TXPARAM(Spec.TXPARAM_SIG_HASH))
        + Op.STOP
    )
    frames = [
        verify_frame(),
        default_frame(target=probe, gas_limit=PROBE_FRAME_GAS),
    ]
    expected_hash = hand_computed_signature_hash(
        sender,
        frames,
        [
            [
                Spec.SCHEME_ARBITRARY,
                b"",
                bytes(witness.msg),
                bytes(witness.signature),
            ]
        ],
        PRIORITY_FEE,
        MAX_FEE,
    )

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_priority_fee_per_gas=PRIORITY_FEE,
        max_fee_per_gas=MAX_FEE,
        frames=frames,
        signatures=[witness],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            probe: Account(
                storage={
                    SLOT_RESULT: int.from_bytes(expected_hash, "big")
                }
            ),
        },
    )


EMBEDDED_SIGNER_KEY = 2
"""Private key of the embedded signer; its address is below."""

EMBEDDED_SIGNER = bytes.fromhex("2b5ad5c4795c026514f8317c7a215e218dccd6cf")
"""The address of private key `2`."""


@pytest.mark.parametrize(
    "witness_bytes",
    [
        pytest.param(b"\xab" * 5, id="short_witness"),
        pytest.param(b"\xab" * 100, id="long_witness"),
    ],
)
def test_sig_hash_elides_empty_msg_bytes(
    state_test: StateTestFiller,
    pre: Alloc,
    witness_bytes: bytes,
) -> None:
    """
    Pin the elision of empty-message signature bytes from the
    canonical hash: a secp256k1 entry is signed in-test over the
    hand-computed elided preimage, so the transaction validates only
    if the protocol elides both entries' raw bytes — including the
    varied arbitrary witness, whose size must not affect the hash.

    The probe additionally stores the witness length, proving the
    varied bytes rode along in the transaction that validated.
    """
    sender = pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=10**18,
    )
    probe = pre.deploy_contract(
        code=Op.SSTORE(
            SLOT_RESULT,
            Op.SIGPARAM(1, Spec.SIGPARAM_SIGNATURE_LENGTH),
        )
        + Op.STOP
    )
    frames = [
        verify_frame(),
        default_frame(target=probe, gas_limit=PROBE_FRAME_GAS),
    ]
    elided_hash = hand_computed_signature_hash(
        sender,
        frames,
        [
            [Spec.SCHEME_SECP256K1, EMBEDDED_SIGNER, b"", b""],
            [Spec.SCHEME_ARBITRARY, b"", b"", b""],
        ],
        PRIORITY_FEE,
        MAX_FEE,
    )
    raw = PrivateKey(
        EMBEDDED_SIGNER_KEY.to_bytes(32, "big")
    ).sign_recoverable(elided_hash)
    signed_entry = FrameSignature(
        scheme=Spec.SCHEME_SECP256K1,
        signer=Bytes(EMBEDDED_SIGNER),
        signature=Bytes(bytes([raw[64]]) + raw[0:64]),
    )

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_priority_fee_per_gas=PRIORITY_FEE,
        max_fee_per_gas=MAX_FEE,
        frames=frames,
        signatures=[
            signed_entry,
            FrameSignature(
                scheme=Spec.SCHEME_ARBITRARY,
                signature=Bytes(witness_bytes),
            ),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            probe: Account(
                storage={SLOT_RESULT: len(witness_bytes)},
            ),
        },
    )
