"""
Block-level tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

Frame transactions coexist with the other transaction types in a
block: receipts chain their cumulative gas across types, an invalid
frame transaction invalidates its whole block, the EIP-3607 origin
ban stays type-scoped, frame logs concatenate in frame order, and a
frame can call a contract deployed by an earlier frame of the same
transaction.
"""

from typing import Dict, List

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Block,
    BlockchainTestFiller,
    EIPChecklist,
    Environment,
    Fork,
    FrameReceipt,
    Hash,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionLog,
    TransactionReceipt,
    compute_create_address,
)

from .helpers import (
    AMPLE_FRAME_GAS,
    default_frame,
    sender_frame,
    verify_frame,
)
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

# EIP-8141 is slated for the fork after Amsterdam, so fixtures are
# labeled with the pseudo `Bogota` fork (Amsterdam + EIP-8141), even
# though the spec prototypes the EIP inside the Amsterdam fork module.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

SLOT_RESULT = 0x01
"""Storage slot target contracts write into."""

MARKER = 0xC0DE
"""Distinctive nonzero marker value."""

PROBE_FRAME_GAS = 500_000
"""Gas limit of frames whose code writes storage, leaving room for
the state gas of fresh writes under EIP-8037."""

APPROVE_ALL_CODE = Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT)
"""Sender code approving execution and payment."""


@EIPChecklist.TransactionType.Test.BlockInteractions.MixedTxs()
def test_mixed_transaction_block(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Execute legacy, fee-market, and frame transactions in one block,
    with the frame transaction last: each receipt's cumulative gas is
    the previous cumulative plus its own hand-computed gas.
    """
    legacy_sender = pre.fund_eoa()
    market_sender = pre.fund_eoa()
    frame_sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=10**18)
    recipient = pre.fund_eoa(amount=1)
    gas_costs = fork.gas_costs()

    transfer_gas = fork.transaction_intrinsic_cost_calculator()(
        calldata=b"", sends_value=True
    )
    verify_gas_used = gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    frame_gas_used = (
        Spec.FRAME_TX_INTRINSIC_COST
        + Spec.FRAME_TX_PER_FRAME_COST
        + verify_gas_used
    )

    txs = [
        Transaction(
            sender=legacy_sender,
            to=recipient,
            value=1,
            gas_limit=100_000,
            expected_receipt=TransactionReceipt(
                cumulative_gas_used=transfer_gas
            ),
        ),
        Transaction(
            sender=market_sender,
            to=recipient,
            value=1,
            gas_limit=100_000,
            max_fee_per_gas=10,
            max_priority_fee_per_gas=0,
            expected_receipt=TransactionReceipt(
                cumulative_gas_used=2 * transfer_gas
            ),
        ),
        Transaction(
            sender=frame_sender,
            nonce=1,
            frames=[verify_frame()],
            expected_receipt=TransactionReceipt(
                cumulative_gas_used=2 * transfer_gas + frame_gas_used,
                payer=frame_sender,
            ),
        ),
    ]

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=txs)],
        post={
            recipient: Account(balance=3),
            frame_sender: Account(nonce=2),
        },
    )


@pytest.mark.exception_test
def test_invalid_frame_transaction_invalidates_block(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Invalidate a whole block on an invalid frame transaction: the
    trailing frame transaction never approves payment, so the block
    is rejected and the leading transfer is not applied.
    """
    transfer_sender = pre.fund_eoa()
    frame_sender = pre.fund_eoa()
    recipient = pre.fund_eoa(amount=1)

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[
                    Transaction(
                        sender=transfer_sender,
                        to=recipient,
                        value=1,
                        gas_limit=100_000,
                    ),
                    Transaction(
                        sender=frame_sender,
                        frames=[default_frame()],
                        error=(
                            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION
                        ),
                    ),
                ],
                exception=(
                    TransactionException.TYPE_6_INVALID_FRAME_EXECUTION
                ),
            )
        ],
        post={
            recipient: Account(balance=1),
            transfer_sender: Account(nonce=0),
        },
    )


@pytest.mark.exception_test
# Funding an EOA with code mutates the shared pre-allocation.
@pytest.mark.pre_alloc_mutable
def test_code_bearing_sender_exemption_is_type_scoped(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Send a frame transaction from a code-bearing account, then a
    fee-market transaction from the same account: the frame
    transaction executes while the second block is rejected by the
    origin ban — the exemption is scoped to the frame transaction
    type, not to the account.
    """
    sender = pre.fund_eoa(amount=10**18, code=APPROVE_ALL_CODE)
    recipient = pre.fund_eoa(amount=1)

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[
                    Transaction(
                        sender=sender,
                        frames=[
                            verify_frame(),
                            sender_frame(target=recipient, value=1),
                        ],
                    ),
                ],
            ),
            Block(
                txs=[
                    Transaction(
                        sender=sender,
                        # The code-bearing account enters the pre-state
                        # at nonce two and the frame transaction bumps
                        # it; the correct nonce keeps the origin ban as
                        # the only rejection.
                        nonce=3,
                        to=recipient,
                        value=1,
                        gas_limit=100_000,
                        max_fee_per_gas=10,
                        max_priority_fee_per_gas=0,
                        error=TransactionException.SENDER_NOT_EOA,
                    ),
                ],
                exception=TransactionException.SENDER_NOT_EOA,
            ),
        ],
        post={
            recipient: Account(balance=2),
            sender: Account(nonce=3),
        },
    )


TOPIC_FIRST = 0x1111
TOPIC_DROPPED = 0x2222
TOPIC_LAST = 0x3333


def test_logs_concatenation(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Concatenate frame logs in frame order: three logging frames emit
    distinct topics, the middle one reverts and its log is dropped,
    and the surviving logs appear in their frames' receipts in order
    — any reordering or omission changes the pinned lists.
    """
    sender = pre.fund_eoa()
    first_logger = pre.deploy_contract(
        code=Op.LOG1(0, 0, TOPIC_FIRST) + Op.STOP
    )
    reverting_logger = pre.deploy_contract(
        code=Op.LOG1(0, 0, TOPIC_DROPPED) + Op.REVERT(0, 0)
    )
    last_logger = pre.deploy_contract(code=Op.LOG1(0, 0, TOPIC_LAST) + Op.STOP)

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(target=first_logger, gas_limit=100_000),
            default_frame(target=reverting_logger, gas_limit=100_000),
            default_frame(target=last_logger, gas_limit=100_000),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS, logs=[]),
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    logs=[
                        TransactionLog(
                            address=first_logger,
                            topics=[Hash(TOPIC_FIRST)],
                        )
                    ],
                ),
                # The reverted frame's log is discarded with its state.
                FrameReceipt(status=Spec.STATUS_FAILURE, logs=[]),
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    logs=[
                        TransactionLog(
                            address=last_logger,
                            topics=[Hash(TOPIC_LAST)],
                        )
                    ],
                ),
            ],
        ),
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=1)},
    )


def test_deploy_then_use(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Call a contract deployed by an earlier frame of the same
    transaction: the first frame's factory creates it at a
    hand-derived address — creator and nonce — and the second frame
    calls it, leaving its marker in the fresh account's storage.
    """
    sender = pre.fund_eoa()
    deployed_runtime = Op.SSTORE(SLOT_RESULT, MARKER) + Op.STOP
    runtime_word = int.from_bytes(
        bytes(deployed_runtime).ljust(32, b"\x00"), "big"
    )
    initcode = Op.MSTORE(0, runtime_word) + Op.RETURN(0, len(deployed_runtime))
    # The initcode exceeds one word, so the factory stores it in two.
    initcode_bytes = bytes(initcode).ljust(64, b"\x00")
    factory = pre.deploy_contract(
        code=Op.MSTORE(0, int.from_bytes(initcode_bytes[:32], "big"))
        + Op.MSTORE(32, int.from_bytes(initcode_bytes[32:], "big"))
        + Op.POP(Op.CREATE(0, 0, len(initcode)))
        + Op.STOP
    )
    created = compute_create_address(address=factory, nonce=1)

    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(target=factory, gas_limit=2_000_000),
            default_frame(target=created, gas_limit=PROBE_FRAME_GAS),
        ],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            created: Account(
                code=deployed_runtime,
                storage={SLOT_RESULT: MARKER},
            ),
        },
    )


@pytest.mark.parametrize(
    "pool_case",
    [
        "unused_gas_returned_exactly",
        pytest.param(
            "returned_gas_boundary_minus_one",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            "admission_charges_max_gas",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@EIPChecklist.TransactionType.Test.BlockInteractions.LastTx.Valid()
@EIPChecklist.TransactionType.Test.BlockInteractions.LastTx.Invalid()
@EIPChecklist.TransactionType.Test.BlockInteractions.SingleTx.Invalid()
def test_block_gas_pool_returns_unused(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
    pool_case: str,
) -> None:
    """
    Return a frame transaction's unused gas to the block gas pool.

    The frame transaction is admitted against its derived maximum gas
    — the intrinsic execution gas plus its frame gas limit — but the
    pool only keeps its hand-computed usage. A follow-up transaction
    sized to the returned gas fits exactly when the block gas limit
    equals the frame transaction's usage plus the follow-up's limit,
    and no longer fits when the limit is one less. The third arm pins
    the admission side: a block gas limit one below the derived
    maximum gas rejects the frame transaction outright, however
    little it would consume.
    """
    gas_costs = fork.gas_costs()
    frame_sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=10**18)
    follow_up_sender = pre.fund_eoa()
    canary = pre.deploy_contract(code=Op.SSTORE(SLOT_RESULT, MARKER) + Op.STOP)

    verify_gas_used = gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    frame_tx_intrinsic = (
        Spec.FRAME_TX_INTRINSIC_COST + Spec.FRAME_TX_PER_FRAME_COST
    )
    # The contract sender carries no signature entries and the frame
    # no data, so no calldata-priced bytes enter the formulas and the
    # standard path dominates the floor.
    frame_tx_gas_used = frame_tx_intrinsic + verify_gas_used
    frame_tx_max_gas = frame_tx_intrinsic + AMPLE_FRAME_GAS

    # The follow-up limit leaves room for the fresh write's state gas.
    follow_up_gas_limit = PROBE_FRAME_GAS
    exact_limit = frame_tx_gas_used + follow_up_gas_limit
    # The frame transaction must itself fit the pool by its maximum
    # gas; the verify frame's unused gas provides the slack.
    assert frame_tx_max_gas <= exact_limit
    # Were the pool charged the maximum gas instead of the usage, the
    # remainder would be far below the follow-up's limit, so the exact
    # arm distinguishes the two accountings.
    assert exact_limit - frame_tx_max_gas < follow_up_gas_limit

    blocks: List[Block]
    post: Dict[Address, Account]
    if pool_case == "unused_gas_returned_exactly":
        genesis_environment = Environment(gas_limit=exact_limit)
        blocks = [
            Block(
                txs=[
                    Transaction(
                        sender=frame_sender,
                        nonce=1,
                        frames=[verify_frame()],
                        expected_receipt=TransactionReceipt(
                            cumulative_gas_used=frame_tx_gas_used,
                            payer=frame_sender,
                        ),
                    ),
                    Transaction(
                        sender=follow_up_sender,
                        to=canary,
                        gas_limit=follow_up_gas_limit,
                    ),
                ]
            )
        ]
        post = {
            canary: Account(storage={SLOT_RESULT: MARKER}),
            frame_sender: Account(nonce=2),
        }
    elif pool_case == "returned_gas_boundary_minus_one":
        genesis_environment = Environment(gas_limit=exact_limit - 1)
        blocks = [
            Block(
                txs=[
                    Transaction(
                        sender=frame_sender,
                        nonce=1,
                        frames=[verify_frame()],
                    ),
                    Transaction(
                        sender=follow_up_sender,
                        to=canary,
                        gas_limit=follow_up_gas_limit,
                        error=TransactionException.GAS_ALLOWANCE_EXCEEDED,
                    ),
                ],
                exception=TransactionException.GAS_ALLOWANCE_EXCEEDED,
            )
        ]
        post = {
            canary: Account(storage={SLOT_RESULT: 0}),
            frame_sender: Account(nonce=1),
        }
    else:
        assert pool_case == "admission_charges_max_gas"
        genesis_environment = Environment(gas_limit=frame_tx_max_gas - 1)
        blocks = [
            Block(
                txs=[
                    Transaction(
                        sender=frame_sender,
                        nonce=1,
                        frames=[verify_frame()],
                        error=TransactionException.GAS_ALLOWANCE_EXCEEDED,
                    ),
                ],
                exception=TransactionException.GAS_ALLOWANCE_EXCEEDED,
            )
        ]
        post = {frame_sender: Account(nonce=1)}

    blockchain_test(
        pre=pre,
        genesis_environment=genesis_environment,
        blocks=blocks,
        post=post,
    )
