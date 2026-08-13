"""
Call-context and code-cost tests for the EIP-8250 nonce manager predeploy.

`NONCE_MANAGER` holds the keyed nonce state, but its code
(`NONCE_MANAGER_CODE = 0x60006000fd`) reverts unconditionally, so the
protocol is the only writer. `test_introspection.py` pins that a contract
calling it gets an empty revert; the tests here cover the contexts a
contract call cannot reach -- the transaction entry point and a frame
target -- and the two facts that follow from the code being exactly those
five bytes: it costs exactly six gas, and value sent to it is returned.

The manager is also the one account in this EIP that can hold a balance
nobody can spend, since every ordinary call reverts. That the balance is
inert -- it neither blocks nor perturbs a keyed consume -- is the last test
here.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Environment,
    Fork,
    Frame,
    FrameReceipt,
    Op,
    StateTestFiller,
    Storage,
    Transaction,
    TransactionReceipt,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

MANAGER_CODE_GAS = 6
"""
Gas the predeploy consumes before reverting.

Hand-disassembled from `NONCE_MANAGER_CODE = 0x60006000fd`: `PUSH1 0x00`
twice at 3 gas each, then `REVERT` over the zero-length region they pushed,
which charges nothing itself and expands no memory. 3 + 3 + 0 = 6.
"""

FORCED_BALANCE = 1
"""Wei force-sent to the manager, which has no way to spend it."""

CANARY = 0xC0DE
"""Sentinel proving the probing code ran to completion."""

PROBE_GAS = 1_000_000
"""Gas limit of a probing frame, ample for the writes it performs."""


def test_manager_as_tx_entry_point(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Pin the revert at the transaction entry point and the code's exact
    cost.

    A transaction whose `to` is the manager is the one call context no
    contract can construct, and the EIP grants it no exemption: the code
    runs and reverts like any other call. The transaction is therefore
    valid but fails, and its cost is fully determined -- the intrinsic cost
    for a value-bearing transaction with no calldata, plus the six gas of
    the code and nothing else. The top-level target is already warm under
    EIP-2929, and EIP-2780 charges the recipient's state gas only when
    value lands on an empty account, so neither adds to the total.

    The revert is observed through that gas total and through the returned
    wei, not through a receipt status: `verify_transaction_receipt` compares
    only `cumulative_gas_used` and `logs`, so a transaction-level `status`
    or `gas_used` expectation would be accepted and never checked. The two
    observables used here do discriminate. A target that ran to completion
    instead of reverting could not land on the same total, and the
    manager's balance staying zero is only consistent with the wei having
    come back.
    """
    sender = pre.fund_eoa()
    intrinsic_gas = fork.transaction_intrinsic_cost_calculator()(
        calldata=b"",
        sends_value=True,
    )

    tx = Transaction(
        sender=sender,
        to=Spec.NONCE_MANAGER,
        value=FORCED_BALANCE,
        state_gas_reservoir=0,
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=intrinsic_gas + MANAGER_CODE_GAS,
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=1),
            Spec.NONCE_MANAGER: Account(
                nonce=1,
                balance=0,
                code=Spec.NONCE_MANAGER_CODE,
                storage={},
            ),
        },
    )


def test_manager_as_frame_target(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Pin the revert for a frame target beside a successful protocol consume.

    A `DEFAULT` frame pointed at the manager is the closest a transaction
    can get to writing keyed nonce state through the manager's own code,
    and it fails: the frame reverts, while the protocol's own consume in
    the approving frame is untouched. The two halves of that expectation
    are opposed by construction -- the manager's storage carries the
    consumed slot at `nonce_seq + 1 == 1` and nothing else, and its
    balance stays zero.

    The frame carries no value because EIP-8141 permits value transfer
    only from a `SENDER` frame; the value-is-returned half of the rule is
    pinned at the transaction entry point above, where a top-level call
    can carry wei.

    The failing frame's `gas_used` accounts for exactly two things. A frame
    charges its target's access within its own gas limit and starts warm
    with neither its caller nor its target, so the manager -- which no
    earlier frame touched -- is charged as a cold account; then the code
    spends its hand-disassembled six gas and reverts. The access term is
    read from the fork's schedule instead of written out, so a repricing
    moves the expectation with it, while the six is the part this EIP
    fixes. Their sum distinguishes a revert from an exceptional halt, which
    would have consumed the frame's whole limit -- the same discriminator
    `test_undefined_txparam_index_halts` uses in the opposite direction.
    """
    sender = pre.fund_eoa()
    nonce_key = 3
    target_frame_gas = fork.gas_costs().COLD_ACCOUNT_ACCESS + MANAGER_CODE_GAS

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=Spec.NONCE_MANAGER,
                gas_limit=PROBE_GAS,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=Spec.KEYED_NONCE_FIRST_USE_GAS,
                ),
                FrameReceipt(
                    status=Spec.STATUS_FAILURE,
                    gas_used=target_frame_gas,
                ),
            ],
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=0),
            Spec.NONCE_MANAGER: Account(
                nonce=1,
                balance=0,
                code=Spec.NONCE_MANAGER_CODE,
                storage={Spec.nonce_slot(sender, nonce_key): 1},
            ),
        },
    )


def test_postfork_force_send_balance_inert(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin that a force-sent balance is unrecoverable and does not perturb
    keyed state.

    `SELFDESTRUCT` is the one way to move wei to an account that reverts on
    every call, so after it the manager holds a balance with no spender.
    The EIP gives the manager no withdrawal path and no balance-dependent
    behaviour, so the forced wei must be observable, unrecoverable, and
    irrelevant to keyed nonce accounting all at once. All three are checked
    in one transaction: the same frame that force-sends then tries to
    recover, while the approving frame's consume proceeds beside it.

    Under EIP-6780 the bomb is not deleted -- it was created before this
    transaction -- but its wei still moves, which is what makes this a
    force-send rather than a transfer the manager could have refused.

    The recovery attempt's status and its returndata length are both zero,
    and zero is what an unexecuted `SSTORE` leaves behind. Each is
    therefore stored incremented by one, so the slots hold `1` only if the
    call really happened and really failed; a skipped probe would leave
    them at zero and fail the test. The canary is the same guard for the
    frame as a whole.
    """
    sender = pre.fund_eoa()
    nonce_key = 9
    storage = Storage()

    bomb = pre.deploy_contract(
        balance=FORCED_BALANCE,
        code=Op.SELFDESTRUCT(Spec.NONCE_MANAGER),
    )
    driver = pre.deploy_contract(
        code=(
            Op.SSTORE(
                storage.store_next(1),
                Op.CALL(gas=PROBE_GAS, address=bomb),
            )
            + Op.SSTORE(
                storage.store_next(1),
                Op.ADD(
                    Op.CALL(gas=PROBE_GAS, address=Spec.NONCE_MANAGER),
                    1,
                ),
            )
            + Op.SSTORE(storage.store_next(1), Op.ADD(Op.RETURNDATASIZE, 1))
            + Op.SSTORE(
                storage.store_next(FORCED_BALANCE),
                Op.BALANCE(Spec.NONCE_MANAGER),
            )
            + Op.SSTORE(storage.store_next(CANARY), CANARY)
            + Op.STOP
        ),
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=driver,
                gas_limit=PROBE_GAS,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=Spec.KEYED_NONCE_FIRST_USE_GAS,
                ),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            bomb: Account(balance=0),
            driver: Account(storage=storage),
            Spec.NONCE_MANAGER: Account(
                nonce=1,
                balance=FORCED_BALANCE,
                code=Spec.NONCE_MANAGER_CODE,
                storage={Spec.nonce_slot(sender, nonce_key): 1},
            ),
        },
    )
