"""State transition and gas tests for EIP-8250 keyed nonces."""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytes,
    Conditional,
    EIPChecklist,
    Environment,
    Frame,
    FrameReceipt,
    FrameSignature,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
)

from .helpers import approve_bytecode
from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")


@pytest.mark.parametrize(
    "nonce_keys",
    [
        pytest.param([1], id="one_key"),
        pytest.param([1, 2], id="two_keys"),
        pytest.param([2**256 - 1], id="maximum_uint256_key"),
        pytest.param(list(range(1, 17)), id="maximum_key_count"),
    ],
)
@EIPChecklist.TransactionType.Test.SenderAccount.Nonce()
def test_first_use_consumes_key_set(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_keys: list[int],
) -> None:
    """
    Pin R-057--R-060, R-072, R-086--R-089, and R-175.

    Every absent slot starts at zero, so expected storage is hand-derived as
    nonce_seq+1=1 and gas as 20,000 times the enumerated key count.
    """
    sender = pre.fund_eoa()
    first_use_gas = Spec.KEYED_NONCE_FIRST_USE_GAS * len(nonce_keys)

    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=first_use_gas,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=first_use_gas,
                    logs=[],
                )
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
                storage={
                    Spec.nonce_slot(sender, nonce_key): 1
                    for nonce_key in nonce_keys
                }
            ),
        },
    )


@pytest.mark.parametrize(
    "initial_nonce",
    [
        pytest.param(0, id="zero"),
        pytest.param(1, id="one"),
        pytest.param(2**64 - 2, id="highest_usable"),
    ],
)
@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.IntrinsicValidity.NonceExact()
def test_legacy_zero_alias_never_manager_slot(
    state_test: StateTestFiller,
    pre: Alloc,
    initial_nonce: int,
) -> None:
    """
    Pin R-050, R-055, R-071, and R-182.

    Expected sender nonce is the supplied legacy nonce plus one; the manager
    remains empty because the zero-key branch is selected before slot hashing.
    """
    sender = pre.fund_eoa(nonce=initial_nonce)
    tx = Transaction(
        sender=sender,
        nonce_keys=[0],
        nonce_seq=initial_nonce,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=0,
                    logs=[],
                )
            ],
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=initial_nonce + 1),
            Spec.NONCE_MANAGER: Account(storage={}),
        },
    )


@pytest.mark.pre_alloc_mutable
def test_subsequent_key_increment_has_zero_surcharge(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-058, R-072, and R-152.

    Pre-seeded value 4 advances to the independently computed value 5, while a
    zero-gas frame proves that only absent (zero-valued) slots are surcharged.
    """
    sender = pre.fund_eoa(nonce=9)
    nonce_keys = [3, 7]
    pre[Spec.NONCE_MANAGER] = Account(
        nonce=1,
        code=Spec.NONCE_MANAGER_CODE,
        storage={
            Spec.nonce_slot(sender, nonce_key): 4 for nonce_key in nonce_keys
        },
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=4,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=0,
                    logs=[],
                )
            ],
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=9),
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(sender, nonce_key): 5
                    for nonce_key in nonce_keys
                }
            ),
        },
    )


@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.OutOfBounds.Max()
def test_highest_keyed_sequence_becomes_exhausted(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-065, R-072, R-177, and R-178.

    The seeded value MAX-1 advances by literal addition to MAX, independently
    exercising the final permitted transition into the exhausted state.
    """
    sender = pre.fund_eoa(nonce=3)
    nonce_key = 5
    usable_sequence = Spec.MAX_NONCE_SEQ - 1
    nonce_slot = Spec.nonce_slot(sender, nonce_key)
    pre[Spec.NONCE_MANAGER] = Account(
        nonce=1,
        code=Spec.NONCE_MANAGER_CODE,
        storage={nonce_slot: usable_sequence},
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=usable_sequence,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            )
        ],
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=3),
            Spec.NONCE_MANAGER: Account(
                storage={nonce_slot: Spec.MAX_NONCE_SEQ}
            ),
        },
    )


@pytest.mark.parametrize(
    "approval_gas,success",
    [
        pytest.param(
            19_999,
            False,
            id="one_below",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(20_000, True, id="exact"),
        pytest.param(20_001, True, id="one_above"),
    ],
)
@EIPChecklist.GasCostChanges.Test.GasUpdatesMeasurement()
@EIPChecklist.GasCostChanges.Test.OutOfGas()
def test_first_use_surcharge_one_key_gas_triptych(
    state_test: StateTestFiller,
    pre: Alloc,
    approval_gas: int,
    success: bool,
) -> None:
    """
    Pin R-004, R-086--R-089, and R-194.

    The three gas limits are literal 20,000 minus one, equal, and plus one;
    equality succeeds and writes nonce_seq+1, while the lower case writes none.
    """
    sender = pre.fund_eoa()
    nonce_key = 1

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=approval_gas,
            )
        ],
        error=(
            None
            if success
            else TransactionException.TYPE_6_INVALID_FRAME_EXECUTION
        ),
        expected_receipt=(
            TransactionReceipt(
                payer=sender,
                frame_receipts=[
                    FrameReceipt(
                        status=Spec.STATUS_SUCCESS,
                        gas_used=Spec.KEYED_NONCE_FIRST_USE_GAS,
                    )
                ],
            )
            if success
            else None
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(sender, nonce_key): 1 if success else 0
                }
            )
        },
    )


@pytest.mark.parametrize(
    "approval_gas,success",
    [
        pytest.param(
            319_999,
            False,
            id="one_below",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(320_000, True, id="exact"),
        pytest.param(320_001, True, id="one_above"),
    ],
)
@EIPChecklist.GasCostChanges.Test.GasUpdatesMeasurement()
@EIPChecklist.GasCostChanges.Test.OutOfGas()
def test_first_use_surcharge_sixteen_key_aggregate(
    state_test: StateTestFiller,
    pre: Alloc,
    approval_gas: int,
    success: bool,
) -> None:
    """
    Pin R-006 and R-086--R-089.

    Sixteen absent reads require the hand product 16*20,000=320,000. The
    one-below case checks that the aggregate transition leaves every slot zero.
    """
    sender = pre.fund_eoa()
    nonce_keys = list(range(1, 17))
    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=approval_gas,
            )
        ],
        error=(
            None
            if success
            else TransactionException.TYPE_6_INVALID_FRAME_EXECUTION
        ),
        expected_receipt=(
            TransactionReceipt(
                payer=sender,
                frame_receipts=[
                    FrameReceipt(
                        status=Spec.STATUS_SUCCESS,
                        gas_used=16 * Spec.KEYED_NONCE_FIRST_USE_GAS,
                    )
                ],
            )
            if success
            else None
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(sender, key): 1 if success else 0
                    for key in nonce_keys
                }
            )
        },
    )


@pytest.mark.parametrize(
    "stored_sequence,tx_sequence,error",
    [
        pytest.param(
            0,
            1,
            TransactionException.NONCE_MISMATCH_TOO_HIGH,
            id="transaction_sequence_too_high",
        ),
        pytest.param(
            2,
            1,
            TransactionException.NONCE_MISMATCH_TOO_LOW,
            id="transaction_sequence_too_low",
        ),
    ],
)
@pytest.mark.exception_test
@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.IntrinsicValidity.NonceMinusOne()
@EIPChecklist.TransactionType.Test.IntrinsicValidity.NoncePlusOne()
def test_keyed_sequence_mismatch(
    state_test: StateTestFiller,
    pre: Alloc,
    stored_sequence: int,
    tx_sequence: int,
    error: TransactionException,
) -> None:
    """
    Pin R-063--R-068.

    The expected high/low error is obtained by directly comparing tx sequence
    one with the hand-seeded current values zero and two before any frame runs.
    """
    sender = pre.fund_eoa()
    nonce_key = 19
    pre[Spec.NONCE_MANAGER] = Account(
        nonce=1,
        code=Spec.NONCE_MANAGER_CODE,
        storage={Spec.nonce_slot(sender, nonce_key): stored_sequence},
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=tx_sequence,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=100_000,
            )
        ],
        error=error,
    )

    state_test(env=Environment(), pre=pre, tx=tx, post={})


@pytest.mark.parametrize(
    "second_sequence,error",
    [
        pytest.param(4, None, id="shared_sequence"),
        pytest.param(
            5,
            TransactionException.NONCE_MISMATCH_TOO_LOW,
            id="one_domain_mismatch",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@pytest.mark.pre_alloc_mutable
def test_keyset_requires_one_shared_sequence(
    state_test: StateTestFiller,
    pre: Alloc,
    second_sequence: int,
    error: TransactionException | None,
) -> None:
    """
    Pin R-066, R-148, and R-149.

    The expected validity is re-derived by comparing scalar 4 independently
    against both hand-seeded slots; success writes scalar+1 to both domains.
    """
    sender = pre.fund_eoa()
    nonce_keys = [1, 2]
    pre[Spec.NONCE_MANAGER] = Account(
        nonce=1,
        code=Spec.NONCE_MANAGER_CODE,
        storage={
            Spec.nonce_slot(sender, 1): 4,
            Spec.nonce_slot(sender, 2): second_sequence,
        },
    )
    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=4,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            )
        ],
        error=error,
    )
    post = {}
    if error is None:
        post[Spec.NONCE_MANAGER] = Account(
            storage={Spec.nonce_slot(sender, key): 5 for key in nonce_keys}
        )

    state_test(env=Environment(), pre=pre, tx=tx, post=post)


def test_payment_approval_consumes_only_once(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-070, R-073--R-075, R-090, and R-091.

    Receipt statuses are hand-enumerated success/success/failure; only the
    first payer transition writes nonce_seq+1, so the final slot equals one.
    """
    sender = pre.fund_eoa()
    first_payer = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT),
        balance=10**18,
    )
    second_payer = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT),
        balance=10**18,
    )
    nonce_key = 23

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION,
                gas_limit=0,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=Spec.APPROVE_PAYMENT,
                target=first_payer,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=Spec.APPROVE_PAYMENT,
                target=second_payer,
                gas_limit=100_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=first_payer,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS, gas_used=0),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_FAILURE),
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
                storage={Spec.nonce_slot(sender, nonce_key): 1}
            ),
        },
    )


def test_approval_survives_later_frame_revert(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-093--R-098, R-144, R-172, R-188, and R-189.

    The hard-coded post-state keeps nonce_seq+1 after a later REVERT, directly
    re-deriving the EIP's outer-journal rule rather than using a state helper.
    """
    sender = pre.fund_eoa()
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))
    nonce_key = 29

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
                target=reverter,
                gas_limit=100_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=Spec.KEYED_NONCE_FIRST_USE_GAS,
                ),
                FrameReceipt(status=Spec.STATUS_FAILURE),
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
                storage={Spec.nonce_slot(sender, nonce_key): 1}
            ),
        },
    )


def test_approval_survives_revert_of_approving_frame(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-093--R-098 and R-144.

    A DELEGATECALL reaches APPROVE before the enclosing REVERT; the expected
    slot one and payer effects follow the outer approval journal explicitly.
    """
    approval_library = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION_AND_PAYMENT)
    )
    sender = pre.deploy_contract(
        code=(
            Op.POP(Op.DELEGATECALL(gas=200_000, address=approval_library))
            + Op.REVERT(0, 0)
        ),
        balance=10**18,
    )
    target = pre.deploy_contract(code=Op.SSTORE(1, 1) + Op.STOP)
    nonce_key = 31

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=500_000,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                target=target,
                gas_limit=300_000,
            ),
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
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=1),
            target: Account(storage={1: 1}),
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, nonce_key): 1}
            ),
        },
    )


@pytest.mark.parametrize(
    "nonce_keys,approval_gas,expected_sender_nonce",
    [
        pytest.param([0], 0, 1, id="legacy_nonce"),
        pytest.param(
            [37],
            Spec.KEYED_NONCE_FIRST_USE_GAS,
            0,
            id="keyed_nonce",
        ),
    ],
)
def test_protocol_bookkeeping_remains_cold_unmetered(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_keys: list[int],
    approval_gas: int,
    expected_sender_nonce: int,
) -> None:
    """
    Pin R-104 and R-190--R-191.

    The expected 6,005 gas is hand-summed from target entry and opcode costs:
    3,000 frame entry, 3 for the PUSH20, the full cold 3,000 manager access
    charged by BALANCE after protocol bookkeeping, 2 for the POP and 0 for
    the STOP.
    """
    sender = pre.fund_eoa()
    target = pre.deploy_contract(
        code=Op.POP(Op.BALANCE(Spec.NONCE_MANAGER)) + Op.STOP
    )
    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=approval_gas,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=target,
                gas_limit=100_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=approval_gas,
                ),
                # 3,000 target entry + 3 PUSH/BALANCE/POP/STOP costs,
                # including a cold 3,000-gas manager access.
                FrameReceipt(status=Spec.STATUS_SUCCESS, gas_used=6_005),
            ],
        ),
    )

    manager_storage = {}
    if nonce_keys != [0]:
        manager_storage[Spec.nonce_slot(sender, nonce_keys[0])] = 1

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=expected_sender_nonce),
            Spec.NONCE_MANAGER: Account(storage=manager_storage),
        },
    )


def test_approval_survives_failed_atomic_batch(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-096, R-098, R-099, and R-144.

    Expected batch receipts are enumerated success/success/failure/skipped.
    The final frame has status 2 and no state effect because the preceding
    flagged frame failed; the independently specified outer approval journal
    nevertheless keeps slot nonce_seq+1.
    """
    sender = pre.fund_eoa()
    payer = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT),
        balance=10**18,
    )
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))
    skipped_target = pre.deploy_contract(code=Op.SSTORE(1, 1) + Op.STOP)
    nonce_key = 0xBEEF

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=(Spec.APPROVE_PAYMENT | Spec.ATOMIC_BATCH_FLAG),
                target=payer,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=reverter,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=skipped_target,
                gas_limit=100_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=payer,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS, logs=[]),
                FrameReceipt(status=Spec.STATUS_FAILURE),
                FrameReceipt(
                    status=Spec.STATUS_SKIPPED,
                    gas_used=0,
                    logs=[],
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
                storage={Spec.nonce_slot(sender, nonce_key): 1}
            ),
            skipped_target: Account(storage={1: 0}),
        },
    )


@EIPChecklist.TransactionType.Test.SenderAccount.Balance()
def test_payment_approval_preserves_transiently_funded_payer(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-096, R-098, R-099, R-102, and R-103.

    The payer balance is independently computed as the transient 10**18 credit
    minus the hand-summed 47,284 gas at the fixed price of seven wei.
    """
    sender_balance = 10**21
    transfer_value = 10**18
    # 21,128 intrinsic + frame costs 3,021 + 20,129 + 3,006.
    gas_used = 47_284
    gas_price = 7
    sender = pre.fund_eoa(amount=sender_balance)
    payer = pre.deploy_contract(
        code=Conditional(
            condition=Op.CALLVALUE,
            if_true=Op.STOP,
            if_false=approve_bytecode(Spec.APPROVE_PAYMENT),
        )
    )
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))
    nonce_key = 0xCAFE

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        max_priority_fee_per_gas=0,
        max_fee_per_gas=gas_price,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=payer,
                value=transfer_value,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=(Spec.APPROVE_PAYMENT | Spec.ATOMIC_BATCH_FLAG),
                target=payer,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=reverter,
                gas_limit=100_000,
            ),
        ],
        signatures=[
            FrameSignature(
                scheme=Spec.SCHEME_SECP256K1,
                signer=Bytes(sender),
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=payer,
            cumulative_gas_used=gas_used,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_FAILURE),
            ],
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=0, balance=sender_balance),
            payer: Account(
                nonce=1,
                balance=transfer_value - gas_used * gas_price,
            ),
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, nonce_key): 1}
            ),
        },
    )


@pytest.mark.exception_test
def test_execution_only_approval_reverts_outside_payment_scope(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-073--R-075, R-093--R-096.

    Two delegate calls separate execution-only from payment approval; after the
    frame REVERT, the missing durable execution approval makes the tx invalid.
    """
    execution_library = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION)
    )
    payment_library = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT)
    )
    sender = pre.deploy_contract(
        code=(
            Op.POP(Op.DELEGATECALL(gas=200_000, address=execution_library))
            + Op.POP(Op.DELEGATECALL(gas=200_000, address=payment_library))
            + Op.REVERT(0, 0)
        ),
        balance=10**18,
    )
    target = pre.deploy_contract(code=Op.STOP)

    tx = Transaction(
        sender=sender,
        nonce_keys=[0xDEAD],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=600_000,
            ),
            Frame(
                mode=Spec.MODE_SENDER,
                target=target,
                gas_limit=100_000,
            ),
        ],
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(env=Environment(), pre=pre, tx=tx, post={})


def test_key_zero_preserves_live_nonce_across_batch_rollback(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-071, R-076, R-077, R-096, and R-098.

    Starting nonce one is incremented by CREATE to two and by approval to
    three; the latter is retained outside the later batch rollback snapshot.
    """
    sender = pre.deploy_contract(
        code=Conditional(
            condition=Op.ISZERO(Op.TXPARAM(0x0A)),
            if_true=Op.POP(Op.CREATE(0, 0, 0)) + Op.STOP,
            if_false=approve_bytecode(Spec.APPROVE_EXECUTION_AND_PAYMENT),
        ),
        balance=10**18,
        nonce=1,
    )
    reverter = pre.deploy_contract(code=Op.REVERT(0, 0))

    tx = Transaction(
        sender=sender,
        nonce_keys=[0],
        nonce_seq=1,
        frames=[
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=Spec.ATOMIC_BATCH_FLAG,
                gas_limit=1_000_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=(
                    Spec.APPROVE_EXECUTION_AND_PAYMENT | Spec.ATOMIC_BATCH_FLAG
                ),
                gas_limit=300_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=reverter,
                gas_limit=100_000,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
                FrameReceipt(status=Spec.STATUS_FAILURE),
            ],
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=3)},
    )


@pytest.mark.exception_test
def test_key_zero_approval_rejects_live_nonce_overflow(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-078--R-080, R-177, and R-178.

    MAX-1 plus the earlier CREATE equals MAX; another hand-computed increment
    would exceed the uint64 bound, so approval must fail without effects.
    """
    initial_nonce = Spec.MAX_NONCE_SEQ - 1
    sender = pre.deploy_contract(
        code=Conditional(
            condition=Op.ISZERO(Op.TXPARAM(0x0A)),
            if_true=Op.POP(Op.CREATE(0, 0, 0)) + Op.STOP,
            if_false=approve_bytecode(Spec.APPROVE_EXECUTION_AND_PAYMENT),
        ),
        balance=10**18,
        nonce=initial_nonce,
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=[0],
        nonce_seq=initial_nonce,
        frames=[
            Frame(mode=Spec.MODE_DEFAULT, gas_limit=1_000_000),
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=300_000,
            ),
        ],
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(env=Environment(), pre=pre, tx=tx, post={})
