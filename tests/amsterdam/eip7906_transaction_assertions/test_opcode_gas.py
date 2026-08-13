"""
Pricing tests for the TXTRACE and TXDIFF opcodes of
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

The TXDIFF parameters that may fall back to live state are priced with
the warm and cold access costs and record the access; every other
state-diff lookup is answered from what the transaction already
recorded, at a flat cost and without touching an access set.

Each expectation is a frame receipt's exact `gas_used`: an assertion
frame costs the cold access charged for its target at entry plus the
cost of its code, so a single mispriced lookup moves it.
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
    SENTINEL_MARKER,
    SENTINEL_SLOT,
    assert_eq,
    contract_sender_max_cost,
    deploy_approving_sender,
    deploy_sentinel,
    post_tx_frame,
)
from .spec import Spec, ref_spec_7906

REFERENCE_SPEC_GIT_PATH = ref_spec_7906.git_path
REFERENCE_SPEC_VERSION = ref_spec_7906.version

# EIP-7906 extends EIP-8141, which is slated for the fork after
# Amsterdam, so fixtures are labeled with the pseudo `Bogota` fork.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

MAX_FEE_PER_GAS = 10
"""Fee the gas pre-charge tests price the escrow at."""

SUBJECT_SLOT = 0x1234
"""Storage key of the queried account the assertions look up."""

SUBJECT_VALUE = 0xABCD
"""Value stored at `SUBJECT_SLOT` before the transaction."""


def assertion_frame_gas(fork: Fork, code: Bytecode) -> int:
    """
    Return the exact gas an assertion frame running `code` uses.

    A frame pays the cold access of its own target at entry — the
    EIP-8141 frame entry charge — and then the cost of the code, so
    the sum is the whole receipt.
    """
    return fork.gas_costs().COLD_ACCOUNT_ACCESS + code.gas_cost(fork)


def measured_transaction(
    pre: Alloc,
    fork: Fork,
    asserter_code: Bytecode,
    body: Bytecode | None = None,
) -> Transaction:
    """
    Return a transaction whose assertion frame runs `asserter_code`
    with its exact gas pinned through the frame receipt.

    An optional body frame runs first, for the diffs and events the
    assertion reads.
    """
    frames = [verify_frame()]
    receipts = [FrameReceipt(status=FrameSpec.STATUS_SUCCESS)]
    if body is not None:
        frames.append(
            default_frame(
                target=pre.deploy_contract(code=body),
                gas_limit=BODY_FRAME_GAS,
            )
        )
        receipts.append(FrameReceipt(status=FrameSpec.STATUS_SUCCESS))
    frames.append(
        post_tx_frame(target=pre.deploy_contract(code=asserter_code))
    )
    receipts.append(
        FrameReceipt(
            status=FrameSpec.STATUS_SUCCESS,
            gas_used=assertion_frame_gas(fork, asserter_code),
        )
    )
    return Transaction(
        sender=pre.fund_eoa(),
        frames=frames,
        expected_receipt=TransactionReceipt(frame_receipts=receipts),
    )


@EIPChecklist.Opcode.Test.GasUsage.Normal()
@pytest.mark.parametrize(
    "param",
    [
        pytest.param(Spec.TXDIFF_SLOT_VALUE_BEFORE, id="slot_value_before"),
        pytest.param(Spec.TXDIFF_SLOT_VALUE_AFTER, id="slot_value_after"),
    ],
)
def test_txdiff_slot_pricing_cold_warm(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    param: int,
) -> None:
    """
    Look the same storage key up twice and pin the frame's gas: the
    first lookup pays the cold storage access and records it, the
    second pays the warm access.

    The queried account is touched by nothing else in the transaction,
    so the key is cold when the assertion frame starts. Charging both
    lookups the same — either price — moves the receipt.

    Rules: R-077, R-080.
    """
    subject = pre.deploy_contract(
        code=Op.STOP, storage={SUBJECT_SLOT: SUBJECT_VALUE}
    )
    code = (
        Op.POP(Op.TXDIFF(SUBJECT_SLOT, subject, param, key_warm=False))
        + Op.POP(Op.TXDIFF(SUBJECT_SLOT, subject, param))
        + Op.STOP
    )

    state_test(pre=pre, tx=measured_transaction(pre, fork, code), post={})


@EIPChecklist.Opcode.Test.GasUsage.Normal()
@pytest.mark.parametrize(
    "param",
    [
        pytest.param(Spec.TXDIFF_BALANCE_BEFORE, id="balance_before"),
        pytest.param(Spec.TXDIFF_BALANCE_AFTER, id="balance_after"),
        pytest.param(Spec.TXDIFF_CODEHASH_BEFORE, id="codehash_before"),
        pytest.param(Spec.TXDIFF_CODEHASH_AFTER, id="codehash_after"),
    ],
)
def test_txdiff_account_pricing_cold_warm(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    param: int,
) -> None:
    """
    Look the same account up twice through each account-valued
    parameter and pin the frame's gas: the first lookup pays the cold
    account access and records it, the second pays the warm access.

    Rules: R-078, R-080.
    """
    subject = pre.deploy_contract(code=Op.STOP)
    code = (
        Op.POP(Op.TXDIFF(0, subject, param, address_warm=False))
        + Op.POP(Op.TXDIFF(0, subject, param))
        + Op.STOP
    )

    state_test(pre=pre, tx=measured_transaction(pre, fork, code), post={})


@EIPChecklist.Opcode.Test.GasUsage.Normal()
def test_txdiff_flat_params_uniform_cost(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Answer every parameter served from the transaction-local diff and
    then read the same account's balance for the first time: the flat
    lookups all cost the same regardless of their operands, and the
    balance still pays the cold access because none of them recorded
    one.

    Rules: R-079, R-083, R-002.
    """
    subject = pre.deploy_contract(code=Op.STOP)
    code = (
        Op.POP(Op.TXDIFF(0, subject, Spec.TXDIFF_ADDRESS_SLOTS_COUNT))
        + Op.POP(Op.TXDIFF(0, subject, Spec.TXDIFF_ADDRESS_EVENTS_COUNT))
        + Op.POP(Op.TXDIFF(0, subject, Spec.TXDIFF_ACCOUNT_CHANGE_FLAGS))
        # Still cold: none of the flat parameters above may record an
        # access for the address they were keyed by.
        + Op.POP(
            Op.TXDIFF(
                0,
                subject,
                Spec.TXDIFF_BALANCE_BEFORE,
                address_warm=False,
            )
        )
        + Op.STOP
    )

    state_test(pre=pre, tx=measured_transaction(pre, fork, code), post={})


@EIPChecklist.Opcode.Test.GasUsage.Normal()
@EIPChecklist.GasCostChanges.Test.GasUpdatesMeasurement()
@EIPChecklist.GasCostChanges.Test.ForkTransition.After()
@pytest.mark.parametrize(
    "event_count",
    [
        pytest.param(1, id="one_event_diff"),
        pytest.param(6, id="six_event_diff"),
    ],
)
def test_txtrace_cost_uniform(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    event_count: int,
) -> None:
    """
    Read the same eight parameters of a small and of a six-times
    larger diff, and pin the identical frame gas for both: TXTRACE is
    charged a flat cost per call, independent of the parameter it
    selects and of how much the transaction changed.

    Rule: R-002.
    """
    writer = Op.SSTORE(SUBJECT_SLOT, SUBJECT_VALUE)
    for topic in range(event_count):
        writer += Op.LOG1(0, 0, 0xFEED + topic)
    writer += Op.STOP
    code = (
        Op.POP(Op.TXTRACE(0, Spec.TXTRACE_SLOTS_CHANGED))
        + Op.POP(Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED))
        + Op.POP(Op.TXTRACE(0, Spec.TXTRACE_EVENTS_COUNT))
        + Op.POP(Op.TXTRACE(0, Spec.TXTRACE_SLOT_ADDRESS))
        + Op.POP(Op.TXTRACE(0, Spec.TXTRACE_SLOT_KEY))
        + Op.POP(Op.TXTRACE(0, Spec.TXTRACE_SLOT_VALUE_AFTER))
        + Op.POP(Op.TXTRACE(0, Spec.TXTRACE_EVENT_TOPIC0))
        + Op.POP(Op.TXTRACE(0, Spec.TXTRACE_GAS_PRE_CHARGE))
        + Op.STOP
    )

    state_test(
        pre=pre,
        tx=measured_transaction(pre, fork, code, body=writer),
        post={},
    )


@EIPChecklist.Opcode.Test.GasUsage.Normal()
def test_txtrace_and_eventdatacopy_add_no_accesses(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Enumerate an event whose topic names an untouched account, copy its
    data, and then read that account's balance: the read still pays the
    cold access, so neither opcode recorded one for the address it
    reported.

    An event topic is the only place the diff can name an address that
    no frame touched — every address in the balance and storage tables
    was warmed by the body that changed it.

    Rule: R-082.
    """
    cold_subject = pre.deploy_contract(code=Op.STOP)
    emitter = (
        Om.MSTORE(bytes(range(1, 33)), 0)
        + Op.LOG1(0, 32, int.from_bytes(cold_subject, "big"))
        + Op.STOP
    )
    code = (
        Op.POP(Op.TXTRACE(0, Spec.TXTRACE_EVENT_ADDRESS))
        + Op.POP(Op.TXTRACE(0, Spec.TXTRACE_EVENT_TOPIC0))
        + Op.EVENTDATACOPY(0, 0, 0, 32, data_size=32, new_memory_size=32)
        + Op.POP(Op.BALANCE(cold_subject, address_warm=False))
        + Op.STOP
    )

    state_test(
        pre=pre,
        tx=measured_transaction(pre, fork, code, body=emitter),
        post={},
    )


@EIPChecklist.Opcode.Test.GasUsage.OrderOfOperations.Exact()
@EIPChecklist.Opcode.Test.GasUsage.OrderOfOperations.Oog()
@EIPChecklist.Opcode.Test.GasUsage.OutOfGasExecution()
@pytest.mark.parametrize(
    "budget_delta",
    [
        pytest.param(0, id="exact_budget_succeeds"),
        pytest.param(-1, id="one_below_runs_out_of_gas"),
    ],
)
def test_txdiff_pricing_exact_gas_frames(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    budget_delta: int,
) -> None:
    """
    Budget an assertion frame to the exact cost of a cold storage
    lookup followed by a cold account lookup, and to one gas below:
    only the short budget halts, rolling the body back.

    A frame short by one gas cannot be paid for by any lower price of
    the two lookups, so the pair pins both cold costs from below.

    Rules: R-077, R-078.
    """
    sentinel = deploy_sentinel(pre)
    subject = pre.deploy_contract(
        code=Op.STOP, storage={SUBJECT_SLOT: SUBJECT_VALUE}
    )
    code = (
        Op.POP(
            Op.TXDIFF(
                SUBJECT_SLOT,
                subject,
                Spec.TXDIFF_SLOT_VALUE_BEFORE,
                key_warm=False,
            )
        )
        + Op.POP(
            Op.TXDIFF(
                0,
                subject,
                Spec.TXDIFF_BALANCE_AFTER,
                address_warm=False,
            )
        )
        + Op.STOP
    )
    exact_gas = assertion_frame_gas(fork, code)
    passes = budget_delta == 0

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(
                target=pre.deploy_contract(code=code),
                gas_limit=exact_gas + budget_delta,
            ),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS, gas_used=exact_gas
                )
                if passes
                else FrameReceipt(
                    status=FrameSpec.STATUS_FAILURE,
                    gas_used=exact_gas - 1,
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


def test_txdiff_cross_frame_warmth(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Look the same key and account up from two consecutive assertion
    frames: the first pays both cold accesses, the second pays warm
    ones, because a successful frame commits what it accessed to the
    warm journal the EIP-8141 frames share.

    Rule: R-113.
    """
    subject = pre.deploy_contract(
        code=Op.STOP, storage={SUBJECT_SLOT: SUBJECT_VALUE}
    )
    cold_code = (
        Op.POP(
            Op.TXDIFF(
                SUBJECT_SLOT,
                subject,
                Spec.TXDIFF_SLOT_VALUE_BEFORE,
                key_warm=False,
            )
        )
        + Op.POP(
            Op.TXDIFF(
                0, subject, Spec.TXDIFF_BALANCE_BEFORE, address_warm=False
            )
        )
        + Op.STOP
    )
    warm_code = (
        Op.POP(Op.TXDIFF(SUBJECT_SLOT, subject, Spec.TXDIFF_SLOT_VALUE_BEFORE))
        + Op.POP(Op.TXDIFF(0, subject, Spec.TXDIFF_BALANCE_BEFORE))
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            post_tx_frame(target=pre.deploy_contract(code=cold_code)),
            post_tx_frame(target=pre.deploy_contract(code=warm_code)),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS,
                    gas_used=assertion_frame_gas(fork, cold_code),
                ),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS,
                    gas_used=assertion_frame_gas(fork, warm_code),
                ),
            ],
        ),
    )

    state_test(pre=pre, tx=tx, post={})


@pytest.mark.parametrize(
    "assertion_frames",
    [
        pytest.param(1, id="one_assertion_frame"),
        pytest.param(2, id="two_assertion_frames"),
        pytest.param(3, id="three_assertion_frames"),
    ],
)
def test_per_frame_cost_charged_for_post_tx_frames(
    state_test: StateTestFiller,
    pre: Alloc,
    assertion_frames: int,
) -> None:
    """
    Grow a transaction by assertion frames and check the gas
    pre-charge each time: every `POST_TX` frame adds the EIP-8141
    per-frame constant to the intrinsic cost, as well as its own gas
    limit to the escrow.

    Dropping the per-frame constant for assertion frames — or their
    gas limits — lowers the pre-charge the diff reports.

    Rule: R-109.
    """
    sender = deploy_approving_sender(pre, balance=10**18)
    # A frame with no target of its own runs the frame entry point,
    # which a static assertion frame cannot execute; the filler frames
    # need a target that does nothing.
    no_op = pre.deploy_contract(code=Op.STOP)

    frames = [verify_frame()] + [
        post_tx_frame(target=no_op) for _ in range(assertion_frames)
    ]
    max_cost = contract_sender_max_cost(frames, MAX_FEE_PER_GAS)

    asserter = pre.deploy_contract(
        code=assert_eq(Op.TXTRACE(0, Spec.TXTRACE_GAS_PRE_CHARGE), max_cost)
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
            payer=sender,
            frame_receipts=[FrameReceipt(status=FrameSpec.STATUS_SUCCESS)]
            * (assertion_frames + 1),
        ),
    )

    state_test(pre=pre, tx=tx, post={sender: Account(nonce=2)})


@pytest.mark.parametrize(
    "padding_events",
    [
        pytest.param(2, id="two_padding_events"),
        pytest.param(200, id="two_hundred_padding_events"),
    ],
)
def test_per_address_view_survives_event_padding(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    padding_events: int,
) -> None:
    """
    Pad a transaction with cheap logs from an unrelated contract and
    assert about one account's events through its per-address view:
    the view reports the account's own single event and the global
    index it landed at, and the assertion frame costs the same gas
    whatever the padding.

    This is the attack the per-address views exist to defeat — an
    assertion that had to walk the global enumeration would pay for
    every padded event, and its cost would move between the two arms.

    Rules: R-066, R-067, R-102.
    """
    subject_topic = 0xBEEF
    padder = pre.deploy_contract(code=Op.LOG0(0, 0) * padding_events + Op.STOP)
    subject = pre.deploy_contract(code=Op.LOG1(0, 0, subject_topic) + Op.STOP)

    checker = (
        assert_eq(Op.TXDIFF(0, subject, Spec.TXDIFF_ADDRESS_EVENTS_COUNT), 1)
        # The account's only event sits after all the padding.
        + assert_eq(
            Op.TXDIFF(0, subject, Spec.TXDIFF_ADDRESS_EVENT_INDEX),
            padding_events,
        )
        + assert_eq(
            Op.TXTRACE(padding_events, Spec.TXTRACE_EVENT_TOPIC0),
            subject_topic,
        )
        + Op.STOP
    )
    # The metered frame names no padding-dependent value, so its code
    # is identical in both arms and its receipt must be too.
    meter = (
        Op.POP(Op.TXDIFF(0, subject, Spec.TXDIFF_ADDRESS_EVENTS_COUNT))
        + Op.POP(Op.TXDIFF(0, subject, Spec.TXDIFF_ADDRESS_EVENT_INDEX))
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=padder, gas_limit=BODY_FRAME_GAS),
            default_frame(target=subject, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=pre.deploy_contract(code=checker)),
            post_tx_frame(target=pre.deploy_contract(code=meter)),
        ],
        expected_receipt=TransactionReceipt(
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(
                    status=FrameSpec.STATUS_SUCCESS,
                    gas_used=assertion_frame_gas(fork, meter),
                ),
            ],
        ),
    )

    state_test(pre=pre, tx=tx, post={})
