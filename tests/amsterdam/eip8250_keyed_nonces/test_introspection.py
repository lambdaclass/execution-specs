"""Predeploy and TXPARAM tests for EIP-8250."""

import pytest
from execution_testing import (
    Account,
    Alloc,
    EIPChecklist,
    Environment,
    Frame,
    Op,
    StateTestFiller,
    Transaction,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")


@pytest.mark.pre_alloc_mutable
def test_manager_direct_call_empty_revert(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-003, R-011, R-037, and R-098.

    Bytecode 60 00 60 00 fd is manually decoded as REVERT(0,0). A caller
    records CALL, STATICCALL, and value-bearing CALL status and returndata
    length; all are hard-coded to zero, and the manager's forced 123 wei
    balance is unchanged because the value-bearing subcall reverts.

    The revert is observed through those seven storage slots and not through
    a transaction-level receipt: `verify_transaction_receipt` compares only
    `cumulative_gas_used` and `logs`, so a `status` expectation would be
    accepted and never checked.
    """
    sender = pre.fund_eoa()
    caller = pre.deploy_contract(
        balance=1,
        code=(
            Op.SSTORE(
                0,
                Op.CALL(
                    gas=100_000,
                    address=Spec.NONCE_MANAGER,
                    value=0,
                ),
            )
            + Op.SSTORE(1, Op.RETURNDATASIZE)
            + Op.SSTORE(
                2,
                Op.STATICCALL(gas=100_000, address=Spec.NONCE_MANAGER),
            )
            + Op.SSTORE(3, Op.RETURNDATASIZE)
            + Op.SSTORE(
                4,
                Op.CALL(
                    gas=100_000,
                    address=Spec.NONCE_MANAGER,
                    value=1,
                ),
            )
            + Op.SSTORE(5, Op.RETURNDATASIZE)
            + Op.SSTORE(6, Op.BALANCE(Spec.NONCE_MANAGER))
            + Op.STOP
        ),
    )
    pre[Spec.NONCE_MANAGER] = Account(
        nonce=1,
        balance=123,
        code=Spec.NONCE_MANAGER_CODE,
    )
    tx = Transaction(
        sender=sender,
        to=caller,
        gas_limit=1_000_000,
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            Spec.NONCE_MANAGER: Account(
                nonce=1,
                balance=123,
                code=Spec.NONCE_MANAGER_CODE,
            ),
            caller: Account(
                balance=1,
                storage=dict.fromkeys(range(6), 0) | {6: 123},
            ),
        },
    )


@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.TxScopedAttributes.Read()
def test_keyed_nonce_txparams(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-007--R-010, R-061, and R-063--R-067.

    Scalar returns are literal inputs and the key hash is a hard-coded Keccak
    vector over bytes32(2)||bytes32(1)||bytes32(2).
    """
    legacy_nonce = 7
    nonce_keys = [1, 2]
    sender = pre.fund_eoa(nonce=legacy_nonce)
    target = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.TXPARAM(Spec.TXPARAM_NONCE_SEQ))
            + Op.SSTORE(1, Op.TXPARAM(Spec.TXPARAM_LEGACY_NONCE))
            + Op.SSTORE(2, Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_COUNT))
            + Op.SSTORE(3, Op.TXPARAM(Spec.TXPARAM_NONCE_KEYS_HASH))
            + Op.SSTORE(4, Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_0))
            + Op.STOP
        )
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=target,
                gas_limit=1_000_000,
            ),
        ],
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=legacy_nonce),
            target: Account(
                storage={
                    0: 0,
                    1: legacy_nonce,
                    2: len(nonce_keys),
                    3: int(
                        "cfeb0db821474afb76aa0658ea10c92266529072ed43958d6"
                        "e2fa8274443b0f4",
                        16,
                    ),
                    4: nonce_keys[0],
                }
            ),
        },
    )


@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.TxScopedAttributes.Persistent.Throughout()
def test_legacy_nonce_snapshot_survives_key_zero_approval(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-063, R-069, and R-070.

    The pre-frame nonce seven is hard-coded in both introspection slots, while
    the independently computed live post-approval account nonce is eight.
    """
    initial_nonce = 7
    sender = pre.fund_eoa(nonce=initial_nonce)
    target = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.TXPARAM(Spec.TXPARAM_NONCE_SEQ))
            + Op.SSTORE(1, Op.TXPARAM(Spec.TXPARAM_LEGACY_NONCE))
            + Op.SSTORE(2, Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_COUNT))
            + Op.SSTORE(3, Op.TXPARAM(Spec.TXPARAM_NONCE_KEYS_HASH))
            + Op.SSTORE(4, Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_0))
            + Op.STOP
        )
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=[0],
        nonce_seq=initial_nonce,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=target,
                gas_limit=1_000_000,
            ),
        ],
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=initial_nonce + 1),
            target: Account(
                storage={
                    0: initial_nonce,
                    1: initial_nonce,
                    2: 1,
                    3: int(
                        "ada5013122d395ba3c54772283fb069b10426056ef8ca5475"
                        "0cb9bb552a59e7d",
                        16,
                    ),
                    4: 0,
                }
            ),
        },
    )
