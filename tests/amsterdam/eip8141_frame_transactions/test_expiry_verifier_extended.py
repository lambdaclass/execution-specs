"""
Extended expiry verifier tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

The predeploy's runtime behavior is pinned outside the expiry
verifier frame shape: called from `DEFAULT` frames, from child calls
of every kind, through a delegation designator, and with value from a
`SENDER` frame. The success path's gas is decoded from the runtime
bytecode by hand and pinned exactly, on both sides of the frame gas
boundary.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytes,
    EIPChecklist,
    Environment,
    Fork,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
    compute_create_address,
)
from execution_testing import (
    Macros as Om,
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

SLOT_RESULT = 0x01
"""Storage slot the probe contract writes what it observed into."""

BLOCK_TIMESTAMP = 1_000
"""Timestamp of the block executing the frame transaction."""

UNEXPIRED = Bytes((2**64 - 1).to_bytes(Spec.EXPIRY_DATA_LENGTH, "big"))
"""Expiry data that never expires."""

EXPIRED = Bytes((BLOCK_TIMESTAMP - 1).to_bytes(8, "big"))
"""Expiry data one second in the past."""

# A fresh SSTORE costs STATE_BYTES_PER_STORAGE_SET * COST_PER_STATE_BYTE
# of state gas under EIP-8037, and a frame transaction holds no state
# gas reservoir, so probe frames writing storage need room for the
# writes.
PROBE_FRAME_GAS = 500_000

# The canonical runtime's success path, decoded by hand from the
# spec's assembly listing: the length check — PUSH1, CALLDATASIZE, EQ,
# PUSH1, JUMPI — costs 3+2+3+3+10, and the deadline check — JUMPDEST,
# PUSH0, CALLDATALOAD, PUSH1, SHR, TIMESTAMP, GT, PUSH1, JUMPI — costs
# 1+2+3+3+3+2+3+3+10, ending on a free STOP.
EXPIRY_SUCCESS_EXECUTION_GAS = 21 + 30


@pytest.mark.parametrize(
    "expiry_data,status",
    [
        pytest.param(UNEXPIRED, Spec.STATUS_SUCCESS, id="unexpired"),
        pytest.param(EXPIRED, Spec.STATUS_FAILURE, id="expired"),
        pytest.param(
            Bytes(b"\xff" * 7),
            Spec.STATUS_FAILURE,
            id="data_too_short",
        ),
        pytest.param(
            Bytes(b"\xff" * 9),
            Spec.STATUS_FAILURE,
            id="data_too_long",
        ),
    ],
)
@EIPChecklist.SystemContract.Test.Inputs.Invalid()
@EIPChecklist.SystemContract.Test.Inputs.Invalid.Checks()
@EIPChecklist.SystemContract.Test.InputLengths.Static.Correct()
@EIPChecklist.SystemContract.Test.InputLengths.Static.TooShort()
@EIPChecklist.SystemContract.Test.InputLengths.Static.TooLong()
def test_expiry_runtime_length_check(
    state_test: StateTestFiller,
    pre: Alloc,
    expiry_data: Bytes,
    status: int,
) -> None:
    """
    Call the expiry verifier from a `DEFAULT` frame, where its shape
    rules do not bind: the runtime code itself rejects any calldata
    that is not eight bytes, and an expired deadline, as an ordinary
    revert that fails only the frame.
    """
    sender = pre.fund_eoa()

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(
                target=Spec.EXPIRY_VERIFIER,
                data=expiry_data,
                gas_limit=100_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=status),
            ],
        ),
    )

    state_test(
        env=Environment(timestamp=BLOCK_TIMESTAMP),
        pre=pre,
        tx=tx,
        post={
            Spec.EXPIRY_VERIFIER: Account(
                nonce=0, code=Spec.EXPIRY_VERIFIER_CODE
            ),
            sender: Account(nonce=1),
        },
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
@EIPChecklist.SystemContract.Test.CallContexts.Normal()
@EIPChecklist.SystemContract.Test.CallContexts.Delegate()
@EIPChecklist.SystemContract.Test.CallContexts.Static()
@EIPChecklist.SystemContract.Test.CallContexts.Callcode()
@EIPChecklist.SystemContract.Test.Inputs.MaxValues()
def test_expiry_verifier_from_child_call(
    state_test: StateTestFiller,
    pre: Alloc,
    call_kind: Op,
) -> None:
    """
    Call the expiry verifier from call depth two with every call
    kind: the code reads only calldata and the timestamp, so it
    succeeds with no return data in every context, including the
    read-only and caller-context ones.
    """
    sender = pre.fund_eoa()
    if call_kind in (Op.CALL, Op.CALLCODE):
        invocation = call_kind(Op.GAS, Spec.EXPIRY_VERIFIER, 0, 0, 8, 0, 0)
    else:
        invocation = call_kind(Op.GAS, Spec.EXPIRY_VERIFIER, 0, 8, 0, 0)
    dispatcher = pre.deploy_contract(
        # The unexpired deadline word occupies memory bytes zero to
        # eight after the shift below.
        code=Op.MSTORE(0, int.from_bytes(bytes(UNEXPIRED) + b"\x00" * 24))
        + Op.SSTORE(SLOT_RESULT, Op.ADD(invocation, 1))
        + Op.SSTORE(SLOT_RESULT + 1, Op.ADD(Op.RETURNDATASIZE, 1))
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
        env=Environment(timestamp=BLOCK_TIMESTAMP),
        pre=pre,
        tx=tx,
        post={
            dispatcher: Account(
                storage={
                    # Call success flag plus one, and an empty return
                    # data size plus one.
                    SLOT_RESULT: 2,
                    SLOT_RESULT + 1: 1,
                }
            ),
        },
    )


@EIPChecklist.SystemContract.Test.Inputs.AllZeros()
def test_expiry_verifier_all_zero_deadline(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Reject the all-zeros deadline at any nonzero block timestamp: the
    zero expiry is one second in the past of the earliest block, so
    the frame reverts.
    """
    sender = pre.fund_eoa()

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(
                target=Spec.EXPIRY_VERIFIER,
                data=Bytes(b"\x00" * Spec.EXPIRY_DATA_LENGTH),
                gas_limit=100_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_FAILURE),
            ],
        ),
    )

    state_test(
        env=Environment(timestamp=BLOCK_TIMESTAMP),
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=1)},
    )


@EIPChecklist.SystemContract.Test.ValueTransfer.NoFee()
def test_expiry_verifier_receives_value_from_sender_frame(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Transfer value to the expiry verifier from a `SENDER` frame: the
    shape rules bind only `VERIFY` frames targeting the predeploy, and
    its code ignores the value, so the transfer succeeds and the
    balance sticks.
    """
    sender = pre.fund_eoa()

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            sender_frame(
                target=Spec.EXPIRY_VERIFIER,
                data=UNEXPIRED,
                value=1,
                gas_limit=100_000,
            ),
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
        env=Environment(timestamp=BLOCK_TIMESTAMP),
        pre=pre,
        tx=tx,
        post={
            Spec.EXPIRY_VERIFIER: Account(
                balance=1, code=Spec.EXPIRY_VERIFIER_CODE
            ),
        },
    )


@EIPChecklist.SystemContract.Test.CallContexts.SetCode()
def test_expiry_verifier_behind_delegation(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Run the expiry verifier's code behind a delegation designator: a
    frame targeting a delegated account executes the predeploy's
    runtime under the account's own context, succeeding on an
    unexpired deadline.
    """
    sender = pre.fund_eoa()
    delegated = pre.fund_eoa(amount=1, delegation=Spec.EXPIRY_VERIFIER)

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(
                target=delegated,
                data=UNEXPIRED,
                gas_limit=100_000,
            ),
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
        env=Environment(timestamp=BLOCK_TIMESTAMP),
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=1)},
    )


@pytest.mark.parametrize(
    "gas_shortfall,error",
    [
        pytest.param(0, None, id="frame_gas_exact"),
        pytest.param(
            1,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="frame_gas_exact_minus_one",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@EIPChecklist.SystemContract.Test.GasUsage.Constant.Exact()
@EIPChecklist.SystemContract.Test.GasUsage.Constant.Oog()
def test_expiry_verifier_exact_gas(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_shortfall: int,
    error: TransactionException | None,
) -> None:
    """
    Budget an expiry verifier frame with exactly its execution cost —
    the hand-decoded success path plus the cold account access
    charged at frame entry — or one unit less, which fails the
    `VERIFY` frame and invalidates the transaction.
    """
    sender = pre.fund_eoa()
    exact = fork.gas_costs().COLD_ACCOUNT_ACCESS + EXPIRY_SUCCESS_EXECUTION_GAS

    expected_receipt = None
    if error is None:
        expected_receipt = TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS, gas_used=exact),
            ],
        )

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            verify_frame(
                flags=Spec.APPROVE_NONE,
                target=Spec.EXPIRY_VERIFIER,
                data=UNEXPIRED,
                gas_limit=exact - gas_shortfall,
            ),
        ],
        error=error,
        expected_receipt=expected_receipt,
    )

    state_test(
        env=Environment(timestamp=BLOCK_TIMESTAMP),
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=0 if error else 1)},
    )


VERIFIER_CALL_INITCODE = (
    # The verifier reads the first eight calldata bytes as a
    # big-endian deadline, so the word is left-aligned in memory.
    Op.MSTORE(0, int.from_bytes(UNEXPIRED, "big") << 192)
    + Op.SSTORE(
        SLOT_RESULT,
        Op.ADD(Op.CALL(Op.GAS, Spec.EXPIRY_VERIFIER, 0, 0, 8, 0, 0), 1),
    )
    + Op.STOP
)
"""
Initcode calling the expiry verifier and recording the call's success
flag plus one, so a successful call is distinguishable from a slot
that was never written.
"""


@pytest.mark.parametrize(
    "from_creating_transaction",
    [
        pytest.param(False, id="create_opcode_inside_frame"),
        pytest.param(True, id="contract_creating_transaction"),
    ],
)
@EIPChecklist.SystemContract.Test.CallContexts.Initcode.CREATE()
@EIPChecklist.SystemContract.Test.CallContexts.Initcode.Tx()
def test_expiry_verifier_from_initcode(
    state_test: StateTestFiller,
    pre: Alloc,
    from_creating_transaction: bool,
) -> None:
    """
    Call the expiry verifier from initcode — from a `CREATE` executed
    inside a frame, and from a contract-creating transaction.

    The verifier's runtime reads only its calldata and the block
    timestamp, so an unexpired deadline succeeds in either creation
    context; the created account keeps the recorded flag because the
    initcode deploys empty code rather than reverting.
    """
    sender = pre.fund_eoa()

    if from_creating_transaction:
        created = compute_create_address(address=sender, nonce=sender.nonce)
        tx = Transaction(
            sender=sender,
            to=None,
            data=VERIFIER_CALL_INITCODE,
            max_fee_per_gas=10,
            max_priority_fee_per_gas=0,
        )
    else:
        factory = pre.deploy_contract(
            code=Om.MSTORE(VERIFIER_CALL_INITCODE, 0)
            + Op.POP(Op.CREATE(0, 0, len(VERIFIER_CALL_INITCODE)))
            + Op.STOP
        )
        created = compute_create_address(address=factory, nonce=1)
        tx = Transaction(
            sender=sender,
            frames=[
                verify_frame(),
                default_frame(target=factory, gas_limit=PROBE_FRAME_GAS),
            ],
        )

    state_test(
        env=Environment(timestamp=BLOCK_TIMESTAMP),
        pre=pre,
        tx=tx,
        # One more than the call's success flag.
        post={created: Account(storage={SLOT_RESULT: 2})},
    )
