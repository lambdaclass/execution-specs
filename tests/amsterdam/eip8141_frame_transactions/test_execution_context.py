"""
Execution-context tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

Frames set the execution context by mode: `DEFAULT` and `VERIFY`
frames execute with the frame entry point as caller and origin, while
`SENDER` frames execute as the transaction sender. Value moves only in
`SENDER` frames, transient storage is discarded between frames, and
the payer's escrow leaves the sender's spendable balance before any
`SENDER` frame runs.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytes,
    EIPChecklist,
    Frame,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
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

SLOT_CALLER = 0x01
"""Storage slot recording the frame's top-level `CALLER`."""

SLOT_ORIGIN = 0x02
"""Storage slot recording `ORIGIN` at call depth one."""

SLOT_ORIGIN_DEPTH_TWO = 0x03
"""Storage slot recording `ORIGIN` at call depth two."""

# A fresh SSTORE costs STATE_BYTES_PER_STORAGE_SET * COST_PER_STATE_BYTE
# of state gas under EIP-8037, and a frame transaction holds no state
# gas reservoir, so probe frames writing several slots need room for
# every write.
PROBE_FRAME_GAS = 500_000


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(Spec.MODE_DEFAULT, id="default_frame"),
        pytest.param(Spec.MODE_SENDER, id="sender_frame"),
    ],
)
def test_frame_caller_and_origin(
    state_test: StateTestFiller,
    pre: Alloc,
    mode: int,
) -> None:
    """
    Record `CALLER` and `ORIGIN` inside a frame, at call depths one
    and two.

    A `DEFAULT` frame executes with the frame entry point as caller,
    and a `SENDER` frame with the transaction sender; `ORIGIN` returns
    the frame's caller at every depth, never a transaction-wide
    origin.
    """
    sender = pre.fund_eoa()
    helper = pre.deploy_contract(
        code=Op.SSTORE(SLOT_ORIGIN_DEPTH_TWO, Op.ORIGIN) + Op.STOP
    )
    probe = pre.deploy_contract(
        code=Op.SSTORE(SLOT_CALLER, Op.CALLER)
        + Op.SSTORE(SLOT_ORIGIN, Op.ORIGIN)
        + Op.POP(Op.CALL(Op.GAS, helper, 0, 0, 0, 0, 0))
        + Op.STOP
    )
    expected = Spec.ENTRY_POINT if mode == Spec.MODE_DEFAULT else sender

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            Frame(
                mode=mode,
                target=probe,
                gas_limit=PROBE_FRAME_GAS,
            ),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            probe: Account(
                storage={
                    SLOT_CALLER: expected,
                    SLOT_ORIGIN: expected,
                }
            ),
            helper: Account(
                storage={SLOT_ORIGIN_DEPTH_TWO: expected},
            ),
        },
    )


def test_caller_and_origin_in_verify_frame(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Check `CALLER` and `ORIGIN` inside a `VERIFY` frame without
    writing state: the checker expands memory beyond any reachable
    gas unless both equal the frame entry point, so acceptance of the
    transaction is the oracle.
    """
    sender = pre.fund_eoa()
    matches_entry_point = Op.AND(
        Op.EQ(Op.CALLER, Spec.ENTRY_POINT),
        Op.EQ(Op.ORIGIN, Spec.ENTRY_POINT),
    )
    checker = pre.deploy_contract(
        # On a mismatch the read lands at 2**32 and the memory
        # expansion charge exceeds the frame's gas, failing the
        # `VERIFY` frame and invalidating the transaction.
        code=Op.POP(Op.MLOAD(Op.MUL(Op.ISZERO(matches_entry_point), 2**32)))
        + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            verify_frame(flags=Spec.APPROVE_NONE, target=checker),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=1)},
    )


def test_origin_in_non_frame_transaction(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Record `ORIGIN` and `CALLER` in a regular type-two transaction as
    a negative control: outside frame transactions both remain the
    signing sender, so any frame-transaction origin semantics leaking
    into other types diverges here.
    """
    sender = pre.fund_eoa()
    probe = pre.deploy_contract(
        code=Op.SSTORE(SLOT_CALLER, Op.CALLER)
        + Op.SSTORE(SLOT_ORIGIN, Op.ORIGIN)
        + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        to=probe,
        max_fee_per_gas=10,
        max_priority_fee_per_gas=0,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            probe: Account(storage={SLOT_CALLER: sender, SLOT_ORIGIN: sender}),
        },
    )


TRANSFER_VALUE = 12_345
"""Value carried by the `SENDER` frame of the call-value test."""


def test_callvalue_in_sender_frame(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Report the frame's value through `CALLVALUE` in the top-level
    frame call, and transfer it from the sender.

    The probes store the read value plus one, so a zero read is
    distinguishable from a probe that never ran. The `DEFAULT` frame
    control pins that its statically forced zero value also reads
    back as zero.
    """
    sender = pre.fund_eoa()
    value_probe = pre.deploy_contract(
        code=Op.SSTORE(SLOT_CALLER, Op.ADD(Op.CALLVALUE, 1)) + Op.STOP
    )
    zero_probe = pre.deploy_contract(
        code=Op.SSTORE(SLOT_CALLER, Op.ADD(Op.CALLVALUE, 1)) + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            sender_frame(
                target=value_probe,
                value=TRANSFER_VALUE,
                gas_limit=PROBE_FRAME_GAS,
            ),
            default_frame(target=zero_probe, gas_limit=PROBE_FRAME_GAS),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            value_probe: Account(
                balance=TRANSFER_VALUE,
                storage={SLOT_CALLER: TRANSFER_VALUE + 1},
            ),
            zero_probe: Account(storage={SLOT_CALLER: 1}),
        },
    )


TRANSIENT_KEY = 0xBEEF
"""Transient storage key exercised across frames."""


def test_transient_storage_cleared_between_frames(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Discard transient storage between frames.

    Each frame stores the transient value plus one before and after
    its own `TSTORE`, into slots selected by its frame data: the
    before-read pins that nothing leaked from the previous frame, and
    the after-read is the same-frame visibility control. Storing the
    value plus one keeps a zero read distinguishable from a probe
    that never ran.
    """
    sender = pre.fund_eoa()
    slot = Op.CALLDATALOAD(0)
    probe = pre.deploy_contract(
        code=Op.SSTORE(slot, Op.ADD(Op.TLOAD(TRANSIENT_KEY), 1))
        + Op.TSTORE(TRANSIENT_KEY, 41)
        + Op.SSTORE(Op.ADD(slot, 1), Op.ADD(Op.TLOAD(TRANSIENT_KEY), 1))
        + Op.STOP
    )

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(
                target=probe,
                gas_limit=PROBE_FRAME_GAS,
                data=Bytes((0x00).to_bytes(32, "big")),
            ),
            default_frame(
                target=probe,
                gas_limit=PROBE_FRAME_GAS,
                data=Bytes((0x10).to_bytes(32, "big")),
            ),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            probe: Account(
                storage={
                    # First frame: nothing to leak from, then 41 + 1.
                    0x00: 1,
                    0x01: 42,
                    # Second frame: a leak would read 42 here.
                    0x10: 1,
                    0x11: 42,
                }
            ),
        },
    )


VERIFY_FRAME_GAS = 100_000
"""Gas limit of the approving frame in the balance boundary test."""

TRANSFER_FRAME_GAS = 100_000
"""Gas limit of the transferring frame in the balance boundary test."""

MAX_FEE = 1_000
"""Explicit maximum fee, making the escrow arithmetic visible."""

SENDER_FUNDS = 10**18
"""Funding of the sender in the balance boundary test."""


@pytest.mark.parametrize(
    "value_excess,transferred",
    [
        pytest.param(0, True, id="value_at_remaining_balance"),
        pytest.param(1, False, id="value_above_remaining_balance"),
    ],
)
@EIPChecklist.TransactionType.Test.IntrinsicValidity.ValueNonZeroSufficientBalance()
@EIPChecklist.TransactionType.Test.IntrinsicValidity.ValueNonZeroInsufficientBalance()
@EIPChecklist.TransactionType.Test.SenderAccount.Balance()
def test_sender_value_balance_boundary(
    state_test: StateTestFiller,
    pre: Alloc,
    value_excess: int,
    transferred: bool,
) -> None:
    """
    Transfer the sender's entire spendable balance in a `SENDER`
    frame.

    The payer's escrow — the maximum cost, the derived transaction
    gas limit priced at the maximum fee — leaves the balance when
    payment is approved, before any later frame runs, so the largest
    transferable value is the funding less the hand-computed maximum
    cost. One wei more reverts the frame, without invalidating the
    transaction.
    """
    recipient = pre.fund_eoa(amount=1)
    sender = pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT),
        balance=SENDER_FUNDS,
    )
    # Contract senders carry no signature entries and the frames carry
    # no data, so the derived gas limit reduces to the base and
    # per-frame constants plus the frame gas.
    max_gas = (
        Spec.FRAME_TX_INTRINSIC_COST
        + 2 * Spec.FRAME_TX_PER_FRAME_COST
        + VERIFY_FRAME_GAS
        + TRANSFER_FRAME_GAS
    )
    max_cost = max_gas * MAX_FEE
    value = SENDER_FUNDS - max_cost + value_excess

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=0,
        frames=[
            verify_frame(gas_limit=VERIFY_FRAME_GAS),
            sender_frame(
                target=recipient,
                value=value,
                gas_limit=TRANSFER_FRAME_GAS,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS
                    if transferred
                    else Spec.STATUS_FAILURE
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=2),
            recipient: Account(balance=1 + (value if transferred else 0)),
        },
    )
