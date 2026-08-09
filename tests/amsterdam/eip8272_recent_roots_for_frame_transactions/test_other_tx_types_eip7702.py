"""
Tests that other transaction types are untouched by
[EIP-8272: Recent Roots for Frame Transactions](https://eips.ethereum.org/EIPS/eip-8272).

EIP-8272 inserts one field into EIP-8141's frame transaction and states
that it "does not modify EIP-7702 or other transaction types". Two things
it adds could leak across that boundary: the reference charge, which a
frame transaction pays even for an empty list because `rlp([])` is one
non-zero byte, and the pre-warming a checked reference buys. Neither may
reach a transaction of any other type, and a delegation set by an
EIP-7702 transaction must behave the same in a block that also carries
references.
"""  # noqa: E501

from typing import List

import pytest
from execution_testing import (
    EOA,
    AccessList,
    Account,
    Address,
    Alloc,
    AuthorizationTuple,
    Block,
    BlockchainTestFiller,
    Bytecode,
    CodeGasMeasure,
    Environment,
    Fork,
    Hash,
    Op,
    RecentRootReference,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
)

from ...prague.eip7702_set_code_tx.spec import Spec as Spec7702
from .helpers import (
    CANARY_VALUE,
    EXECUTED_VALUE,
    SENDER_FRAME_GAS,
    SLOT_EXECUTED,
    seed_entries,
    verify_and_sender_frames,
)
from .spec import Spec, ref_spec_8272

REFERENCE_SPEC_GIT_PATH = ref_spec_8272.git_path
REFERENCE_SPEC_VERSION = ref_spec_8272.version

pytestmark = [
    pytest.mark.valid_from("Bogota"),
    pytest.mark.pre_alloc_mutable,
]

SLOT_RECENT_ROOT_ACCESS_GAS = 0x30
"""Slot holding the measured cost of touching the recent root contract."""


@pytest.mark.parametrize(
    "tx_type",
    [
        pytest.param(0, id="legacy"),
        pytest.param(1, id="access_list"),
        pytest.param(2, id="dynamic_fee"),
    ],
)
def test_other_transaction_types_pay_no_reference_gas(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    env: Environment,
    tx_type: int,
) -> None:
    """
    A transaction of any type other than EIP-8141's pays exactly its
    unmodified intrinsic cost once EIP-8272 is active.

    The expected value is the fork's own intrinsic calculator for the
    transaction as built — no EIP-8272 term of any kind — and the
    transaction is arranged so that nothing else can contribute to the
    number: empty calldata, zero value, and a recipient whose entire code
    is `STOP`, which executes for no gas and writes nothing. Its gas used
    is therefore its intrinsic cost and nothing more.

    That makes the assertion sensitive to the smallest leak EIP-8272
    could produce. A frame transaction is charged for `rlp([])`, one
    non-zero byte, even when it declares no references — sixteen gas at
    the standard token rate. An implementation that applied the same
    empty-list charge to every transaction type would report sixteen gas
    too many here, and one that applied the once-only address charge
    would report thousands too many.

    The access-list arms carry one address and one storage key on
    purpose: EIP-8272 prices a reference on top of
    `TX_ACCESS_LIST_STORAGE_KEY`, so a client that conflated the two
    charges misprices exactly these transactions.
    """
    target = pre.deploy_contract(code=Op.STOP)
    access_list: List[AccessList] | None = None
    if tx_type >= 1:
        access_list = [
            AccessList(address=target, storage_keys=[Hash(SLOT_EXECUTED)])
        ]

    intrinsic_gas = fork.transaction_intrinsic_cost_calculator()(
        calldata=b"",
        access_list=access_list,
    )

    tx = Transaction(
        ty=tx_type,
        sender=pre.fund_eoa(),
        to=target,
        access_list=access_list,
        gas_limit=intrinsic_gas,
        expected_receipt=TransactionReceipt(cumulative_gas_used=intrinsic_gas),
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={target: Account(code=Op.STOP, storage={})},
    )


def test_delegation_is_unaffected_by_a_referencing_transaction(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
    current_slot: int,
    references: List[RecentRootReference],
) -> None:
    """
    An EIP-7702 transaction sharing a block with a reference-carrying
    frame transaction sets its delegation and finds the recent root
    contract cold.

    The frame transaction runs first and declares a reference, so the
    recent root contract and one of its storage keys are in *that*
    transaction's accessed sets. The EIP-7702 transaction that follows
    delegates an authority to a prober and calls it, and the prober
    measures the cost of touching the recent root contract. The expected
    value is the fork's cold account access price, derived by the
    framework from the fork's own schedule for a `BALANCE` whose address
    is declared cold — so an implementation that pre-warmed the contract
    for the whole block, or for every transaction type, reports the warm
    price instead and the stored word does not match.

    Both halves of "does not modify EIP-7702" are asserted: the
    delegation designator lands on the authority exactly as EIP-7702
    defines it, and the delegated code runs and writes its sentinel.
    """
    seed_entries(pre, references)
    cold_probe = Op.BALANCE(
        address=Spec.RECENT_ROOT_ADDRESS, address_warm=False
    )
    delegate_code: Bytecode = (
        CodeGasMeasure(
            code=cold_probe,
            extra_stack_items=1,
            sstore_key=SLOT_RECENT_ROOT_ACCESS_GAS,
        )
        + Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE)
        + Op.STOP
    )
    delegate = pre.deploy_contract(code=delegate_code)
    authority: EOA = pre.fund_eoa()
    frame_target: Address = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE) + Op.STOP,
        storage={SLOT_EXECUTED: CANARY_VALUE},
    )

    referencing_tx = Transaction(
        sender=pre.fund_eoa(),
        frames=verify_and_sender_frames(frame_target, SENDER_FRAME_GAS),
        recent_root_references=references,
    )
    delegating_tx = Transaction(
        ty=Spec7702.SET_CODE_TX_TYPE,
        sender=pre.fund_eoa(),
        to=authority,
        gas_limit=500_000,
        authorization_list=[
            AuthorizationTuple(address=delegate, signer=authority)
        ],
    )

    blockchain_test(
        genesis_environment=Environment(slot_number=current_slot),
        pre=pre,
        blocks=[
            Block(
                slot_number=current_slot,
                txs=[referencing_tx, delegating_tx],
            )
        ],
        post={
            frame_target: Account(storage={SLOT_EXECUTED: EXECUTED_VALUE}),
            authority: Account(
                code=Spec7702.delegation_designation(delegate),
                storage={
                    SLOT_RECENT_ROOT_ACCESS_GAS: cold_probe.gas_cost(fork),
                    SLOT_EXECUTED: EXECUTED_VALUE,
                },
            ),
        },
    )
