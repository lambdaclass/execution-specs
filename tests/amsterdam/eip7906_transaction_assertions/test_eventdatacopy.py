"""
EVENTDATACOPY tests for
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

EVENTDATACOPY copies a window of one event's non-indexed data into
memory. It is priced exactly as CALLDATACOPY, but its bounds are
strict: a window reaching past the end of the event's data
exceptionally halts instead of copying zeroes, as does an event index
at or beyond the event count.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytecode,
    EIPChecklist,
    Fork,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
)
from execution_testing import (
    Macros as Om,
)

from tests.amsterdam.eip8141_frame_transactions.helpers import (
    default_frame,
    verify_frame,
)
from tests.amsterdam.eip8141_frame_transactions.spec import Spec as FrameSpec

from .helpers import (
    BODY_FRAME_GAS,
    MAX_UINT256,
    SENTINEL_MARKER,
    SENTINEL_SLOT,
    assert_eq,
    deploy_sentinel,
    post_tx_frame,
)
from .spec import ref_spec_7906

REFERENCE_SPEC_GIT_PATH = ref_spec_7906.git_path
REFERENCE_SPEC_VERSION = ref_spec_7906.version

# EIP-7906 extends EIP-8141, which is slated for the fork after
# Amsterdam, so fixtures are labeled with the pseudo `Bogota` fork.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

EVENT_DATA = bytes(range(1, 65))
"""
The 64 bytes of non-indexed data the emitted event carries.

Every byte is distinct and non-zero, so a copy landing at the wrong
memory offset, reading from the wrong data offset, or copying nothing
at all produces a value that no expected word can match.
"""

SECOND_EVENT_DATA = bytes(range(0x81, 0xC1))
"""Data of the second event, disjoint from the first event's bytes."""


def emitter_code(data: bytes) -> Bytecode:
    """Return code emitting one event carrying `data` unindexed."""
    return Om.MSTORE(data, 0) + Op.LOG0(0, len(data)) + Op.STOP


def assertion_frame_gas(fork: Fork, code: Bytecode) -> int:
    """
    Return the exact gas an assertion frame running `code` uses.

    An assertion frame is entered by a call to a target that no earlier
    frame touched, so its cost is the cold account access charged at
    entry plus the cost of the code itself — the two components the
    EIP-8141 suite pins for a `STOP`-only frame. Every gas expectation
    below is that sum, with no term of the schedule spelled out by
    hand.
    """
    return fork.gas_costs().COLD_ACCOUNT_ACCESS + code.gas_cost(fork)


def memory_word(
    memory_offset: int, data_offset: int, size: int, load_offset: int
) -> int:
    """
    Return the word an `MLOAD` at the absolute `load_offset` must
    return after the copy, computed by applying the copy to a zeroed
    memory image.

    Re-derives the spec's `memory[memory_offset:memory_offset+size] =
    event.data[data_offset:data_offset+size]` independently of any
    client, and reads back the same 32 bytes the assertion code loads —
    so a copy that writes past its window, or lands one byte off, is
    caught as well as one that copies nothing.
    """
    memory = bytearray(load_offset + memory_offset + size + 64)
    memory[memory_offset : memory_offset + size] = EVENT_DATA[
        data_offset : data_offset + size
    ]
    return int.from_bytes(memory[load_offset : load_offset + 32], "big")


@pytest.mark.parametrize(
    "expectation_delta",
    [
        pytest.param(0, id="correct_expectations"),
        pytest.param(1, id="wrong_expectation_bites"),
    ],
)
@pytest.mark.parametrize(
    "memory_offset,data_offset,size",
    [
        pytest.param(0, 0, 32, id="first_word_to_offset_zero"),
        # The two words share no byte value, so a copy that ignores
        # the data offset lands on the other word's expectation.
        pytest.param(0, 32, 32, id="second_word_to_offset_zero"),
        # Distinct memory and data offsets: swapping the two operands
        # writes the wrong bytes at the wrong place.
        pytest.param(0x40, 32, 32, id="second_word_to_high_offset"),
        pytest.param(0x20, 3, 5, id="unaligned_partial_window"),
        pytest.param(0, 63, 1, id="last_byte_only"),
        pytest.param(0, 0, 64, id="whole_data"),
    ],
)
def test_eventdatacopy_copies_event_data(
    state_test: StateTestFiller,
    pre: Alloc,
    memory_offset: int,
    data_offset: int,
    size: int,
    expectation_delta: int,
) -> None:
    """
    Copy windows of an event's data into memory and check the bytes
    that land there, reading back both the word the window starts in
    and the following word.

    The expected words are re-derived by applying the copy to a zeroed
    memory image in Python. The wrong-expectation twin shifts them by
    one, proving the comparison bites.

    Rules: R-090, R-091.
    """
    sentinel = deploy_sentinel(pre)
    emitter = pre.deploy_contract(code=emitter_code(EVENT_DATA))

    asserter = pre.deploy_contract(
        code=Op.EVENTDATACOPY(0, memory_offset, data_offset, size)
        + assert_eq(
            Op.MLOAD(memory_offset),
            memory_word(memory_offset, data_offset, size, memory_offset)
            + expectation_delta,
        )
        + assert_eq(
            Op.MLOAD(memory_offset + 32),
            memory_word(memory_offset, data_offset, size, memory_offset + 32)
            + expectation_delta,
        )
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=emitter, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={
                    SENTINEL_SLOT: (
                        SENTINEL_MARKER if expectation_delta == 0 else 0
                    )
                }
            ),
        },
    )


@EIPChecklist.Opcode.Test.ExceptionalAbort()
@pytest.mark.parametrize(
    "event_index,halts",
    [
        pytest.param(0, False, id="first_event"),
        pytest.param(1, False, id="last_event"),
        pytest.param(2, True, id="event_count_halts"),
        pytest.param(3, True, id="above_event_count_halts"),
        pytest.param(MAX_UINT256, True, id="max_index_halts"),
    ],
)
def test_eventdatacopy_event_index_out_of_bounds_halts(
    state_test: StateTestFiller,
    pre: Alloc,
    event_index: int,
    halts: bool,
) -> None:
    """
    Address each of the two emitted events by index and check that the
    first index at or beyond the event count halts, while the two live
    indexes copy their own event's data.

    The two events carry disjoint bytes, so an off-by-one on the index
    bound is caught by the copied value as well as by the halt.

    Rule: R-092.
    """
    sentinel = deploy_sentinel(pre)
    first_emitter = pre.deploy_contract(code=emitter_code(EVENT_DATA))
    second_emitter = pre.deploy_contract(code=emitter_code(SECOND_EVENT_DATA))

    source = EVENT_DATA if event_index == 0 else SECOND_EVENT_DATA
    asserter = pre.deploy_contract(
        code=Op.EVENTDATACOPY(event_index, 0, 0, 32)
        + assert_eq(Op.MLOAD(0), int.from_bytes(source[0:32], "big"))
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=first_emitter, gas_limit=BODY_FRAME_GAS),
            default_frame(target=second_emitter, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: 0 if halts else SENTINEL_MARKER}
            ),
        },
    )


@EIPChecklist.Opcode.Test.OutOfBounds.Verify.Max()
@EIPChecklist.Opcode.Test.OutOfBounds.Verify.MaxPlusOne()
@pytest.mark.parametrize(
    "data_offset,size,halts",
    [
        pytest.param(32, 32, False, id="window_ends_exactly_at_data_end"),
        pytest.param(33, 32, True, id="window_ends_one_past_data_end"),
        pytest.param(64, 0, False, id="empty_window_at_data_end"),
        # A zero-length read is still bounds-checked: unlike
        # CALLDATACOPY there is no zero fill to fall back on.
        pytest.param(65, 0, True, id="empty_window_past_data_end"),
        pytest.param(0, 65, True, id="size_one_past_data_length"),
        pytest.param(MAX_UINT256, 1, True, id="max_offset_halts"),
    ],
)
def test_eventdatacopy_data_bounds_halt(
    state_test: StateTestFiller,
    pre: Alloc,
    data_offset: int,
    size: int,
    halts: bool,
) -> None:
    """
    Read windows straddling the end of the event's data and check that
    any window reaching past it halts rather than copying zeroes, while
    the window ending exactly at the last byte succeeds.

    Rule: R-093.
    """
    sentinel = deploy_sentinel(pre)
    emitter = pre.deploy_contract(code=emitter_code(EVENT_DATA))

    # The halting arms never reach the comparison; the surviving ones
    # must still land the bytes their window names.
    expected_word = 0 if halts else memory_word(0, data_offset, size, 0)
    asserter = pre.deploy_contract(
        code=Op.EVENTDATACOPY(0, 0, data_offset, size)
        + assert_eq(Op.MLOAD(0), expected_word)
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            default_frame(target=emitter, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sentinel: Account(
                storage={SENTINEL_SLOT: 0 if halts else SENTINEL_MARKER}
            ),
        },
    )


def gas_pinned_transaction(
    pre: Alloc,
    asserter: Bytecode,
    expected_gas_used: int | None,
    frame_gas_limit: int | None = None,
) -> Transaction:
    """
    Return a transaction whose single assertion frame runs `asserter`,
    with the frame's gas usage pinned through its receipt.

    The body emits the event the assertion frame reads and writes the
    sentinel marker, so a frame that halts instead of completing is
    visible both in its receipt and in the rolled-back body.
    """
    frames = [
        verify_frame(),
        default_frame(target=deploy_sentinel(pre), gas_limit=BODY_FRAME_GAS),
        default_frame(
            target=pre.deploy_contract(code=emitter_code(EVENT_DATA)),
            gas_limit=BODY_FRAME_GAS,
        ),
        post_tx_frame(target=pre.deploy_contract(code=asserter))
        if frame_gas_limit is None
        else post_tx_frame(
            target=pre.deploy_contract(code=asserter),
            gas_limit=frame_gas_limit,
        ),
    ]
    succeeds = expected_gas_used is not None
    return Transaction(
        sender=pre.fund_eoa(),
        frames=frames,
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS,
                    gas_used=expected_gas_used,
                )
                if succeeds
                else FrameReceipt(
                    status=FrameSpec.STATUS_FAILURE,
                    gas_used=frame_gas_limit,
                ),
            ],
        ),
    )


@EIPChecklist.Opcode.Test.GasUsage.Normal()
@EIPChecklist.Opcode.Test.GasUsage.MemoryExpansion()
@EIPChecklist.Opcode.Test.MemExp.SingleByte()
@EIPChecklist.Opcode.Test.MemExp.ThirtyOneBytes()
@EIPChecklist.Opcode.Test.MemExp.ThirtyTwoBytes()
@EIPChecklist.Opcode.Test.MemExp.ThirtyThreeBytes()
@EIPChecklist.Opcode.Test.MemExp.SixtyFourBytes()
@pytest.mark.parametrize(
    "size",
    [
        pytest.param(1, id="single_byte"),
        pytest.param(31, id="31_bytes"),
        pytest.param(32, id="32_bytes"),
        pytest.param(33, id="33_bytes"),
        pytest.param(64, id="64_bytes"),
    ],
)
def test_eventdatacopy_memory_expansion_boundaries(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    size: int,
) -> None:
    """
    Copy windows straddling every word boundary and pin the frame's
    exact gas: the base cost, one copy charge per started word, and the
    expansion of a memory that starts empty.

    The word-granular sizes are what separate the per-word copy charge
    from a flat one — 31, 32 and 33 bytes cost one, one and two words.

    Rule: R-089.
    """
    code = (
        Op.EVENTDATACOPY(0, 0, 0, size, data_size=size, new_memory_size=size)
        + Op.STOP
    )

    state_test(
        pre=pre,
        tx=gas_pinned_transaction(pre, code, assertion_frame_gas(fork, code)),
        post={},
    )


@EIPChecklist.Opcode.Test.MemExp.ZeroBytesZeroOffset()
@EIPChecklist.Opcode.Test.MemExp.ZeroBytesMaxOffset()
@pytest.mark.parametrize(
    "memory_offset",
    [
        pytest.param(0, id="zero_offset"),
        pytest.param(MAX_UINT256, id="max_offset"),
    ],
)
def test_eventdatacopy_zero_length_no_memory_expansion(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    memory_offset: int,
) -> None:
    """
    Copy zero bytes and check that no memory is expanded and no word is
    charged, even at the maximum destination offset — the one offset
    that would price a single expanded word out of any budget.

    Rule: R-089.
    """
    code = Op.EVENTDATACOPY(0, memory_offset, 0, 0) + Op.STOP

    state_test(
        pre=pre,
        tx=gas_pinned_transaction(pre, code, assertion_frame_gas(fork, code)),
        post={},
    )


@EIPChecklist.Opcode.Test.GasUsage.OutOfGasMemory()
@EIPChecklist.Opcode.Test.MemExp.TwoThirtyTwoMinusOneBytes()
@EIPChecklist.Opcode.Test.MemExp.TwoThirtyTwoBytes()
@EIPChecklist.Opcode.Test.MemExp.TwoSixtyFourMinusOneBytes()
@EIPChecklist.Opcode.Test.MemExp.TwoSixtyFourBytes()
@EIPChecklist.Opcode.Test.MemExp.TwoTwoFiftySixMinusOneBytes()
@pytest.mark.parametrize(
    "memory_offset",
    [
        pytest.param(2**32 - 33, id="2_32_minus_one_bytes"),
        pytest.param(2**32, id="2_32_bytes"),
        pytest.param(2**64 - 33, id="2_64_minus_one_bytes"),
        pytest.param(2**64, id="2_64_bytes"),
        pytest.param(MAX_UINT256, id="2_256_minus_one_bytes"),
    ],
)
def test_eventdatacopy_huge_expansion_oog(
    state_test: StateTestFiller,
    pre: Alloc,
    memory_offset: int,
) -> None:
    """
    Copy an in-bounds window to a destination so far out that the
    memory expansion alone exceeds any budget: the frame runs out of
    gas, consuming its whole limit, and the body is rolled back.

    The window itself is in bounds, so the halt isolates the memory
    expansion charge from the data bounds check.

    Rule: R-089.
    """
    frame_gas_limit = 100_000
    code = Op.EVENTDATACOPY(0, memory_offset, 0, 32) + Op.STOP

    state_test(
        pre=pre,
        tx=gas_pinned_transaction(pre, code, None, frame_gas_limit),
        post={},
    )


@EIPChecklist.Opcode.Test.GasUsage.OutOfGasExecution()
@EIPChecklist.Opcode.Test.GasUsage.ExtraGas()
@EIPChecklist.Opcode.Test.GasUsage.OrderOfOperations.Exact()
@EIPChecklist.Opcode.Test.GasUsage.OrderOfOperations.Oog()
@pytest.mark.parametrize(
    "budget_delta",
    [
        pytest.param(1, id="one_above_succeeds"),
        pytest.param(0, id="exact_budget_succeeds"),
        pytest.param(-1, id="one_below_runs_out_of_gas"),
    ],
)
def test_eventdatacopy_exact_gas_boundary(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    budget_delta: int,
) -> None:
    """
    Run a copy in a frame budgeted to its exact cost, one gas above and
    one gas below: only the short budget halts, and it consumes every
    gas it was given.

    A single gas of difference is what distinguishes the copy charge
    from any other pricing of the same window.

    Rule: R-089.
    """
    size = 64
    code = (
        Op.EVENTDATACOPY(0, 0, 0, size, data_size=size, new_memory_size=size)
        + Op.STOP
    )
    exact_gas = assertion_frame_gas(fork, code)
    succeeds = budget_delta >= 0

    state_test(
        pre=pre,
        tx=gas_pinned_transaction(
            pre,
            code,
            exact_gas if succeeds else None,
            exact_gas + budget_delta,
        ),
        post={},
    )


def test_eventdatacopy_gas_matches_calldatacopy(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Charge the same window through EVENTDATACOPY and through
    CALLDATACOPY in two assertion frames, and pin both frames' gas.

    The two frames run identical code but for the copy opcode and the
    extra event index it pops, so the EIP's claim that EVENTDATACOPY is
    priced as CALLDATACOPY reduces to the frames differing by exactly
    one push.

    Rules: R-003, R-089.
    """
    size = 64
    event_code = (
        Op.EVENTDATACOPY(0, 0, 0, size, data_size=size, new_memory_size=size)
        + Op.STOP
    )
    calldata_code = (
        Op.CALLDATACOPY(0, 0, size, data_size=size, new_memory_size=size)
        + Op.STOP
    )
    calldata_gas = assertion_frame_gas(fork, calldata_code)
    # The event index is the only extra operand EVENTDATACOPY pops.
    event_gas = calldata_gas + Op.PUSH1[0].gas_cost(fork)

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(
                target=pre.deploy_contract(code=emitter_code(EVENT_DATA)),
                gas_limit=BODY_FRAME_GAS,
            ),
            post_tx_frame(target=pre.deploy_contract(code=calldata_code)),
            post_tx_frame(target=pre.deploy_contract(code=event_code)),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS, gas_used=calldata_gas
                ),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS, gas_used=event_gas
                ),
            ],
        ),
    )

    state_test(pre=pre, tx=tx, post={})
