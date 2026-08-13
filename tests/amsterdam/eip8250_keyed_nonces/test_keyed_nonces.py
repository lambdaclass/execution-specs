"""State transition and gas tests for EIP-8250 keyed nonces."""

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Bytecode,
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
    compute_create_address,
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
    Pin first use of a fresh key set.

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
    Pin the key-zero alias.

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
    Prove only absent slots are surcharged.

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
    Pin the final permitted advance.

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
    Boundary the one-key surcharge.

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
    Boundary the aggregate surcharge.

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
    Pin the per-key sequence equality.

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

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        # The seeded sequence must be exactly what it was: a rejected
        # transaction neither advances the slot nor clears it, and both
        # seeded values are asserted literally rather than left unstated.
        post={
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, nonce_key): stored_sequence}
            )
        },
    )


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
    Prove every selected key must match.

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
    Pin exactly-once consumption.

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


@EIPChecklist.TransactionType.Test.SenderAccount.Balance()
def test_refused_payment_approval_leaves_key_unconsumed(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin that a refused payment approval consumes nothing.

    `test_payment_approval_consumes_only_once` covers a second payment
    approval refused *after* a successful one. This covers the opposite
    order: the first payment-scoped `APPROVE` is refused, and a later one
    succeeds. The EIP requires clients to apply every EIP-8141 `APPROVE`
    exceptional-condition check before the keyed-nonce steps, and
    makes steps 3 through 5 a single transition, so when the approval does
    not complete no approval effect occurs at all.

    The paymaster holds one wei. A frame transaction's maximum cost is its
    fee cap times a gas limit that starts at the EIP-8141 intrinsic cost of
    15,000, so the maximum cost exceeds one wei by construction and the
    EIP-8141 "resolved target can cover the maximum cost" check refuses the
    approval; no arithmetic beyond that inequality is needed.

    The discriminating expectation is the third frame's `gas_used`. Its
    approval is the first one that completes, so the read finds an absent
    slot, `first_use_count` is one, and the EIP deducts exactly one
    `KEYED_NONCE_FIRST_USE_GAS` from that frame. Had the refused approval
    consumed the key, the slot would already hold a non-zero value, the
    surcharge would be zero, and this frame would report zero gas while still
    succeeding within the same limit -- so the exact value, not merely the
    success, is what separates the two outcomes. The paymaster's untouched
    balance pins that no maximum cost was collected either.

    Scope, stated so it is not read as wider than it is: this pins that a
    refused approval performs no keyed-nonce *consumption*. It does not pin
    the order of the first-use *gas charge* against the payer-balance check.
    An implementation that deducts the surcharge and only then refuses on
    balance leaves the slot absent, so the third frame still reports the full
    surcharge and this test still passes. Isolating that ordering needs an
    exact `gas_used` on the refused frame, which for a contract-target frame
    means hard-coding the fork's cold-account-access price.
    """
    sender = pre.fund_eoa()
    paymaster_balance = 1
    paymaster = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT),
        balance=paymaster_balance,
    )
    nonce_key = 0x5EED
    first_use_gas = Spec.KEYED_NONCE_FIRST_USE_GAS

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
                target=paymaster,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_PAYMENT,
                gas_limit=first_use_gas,
            ),
        ],
        # A payment-only `VERIFY` frame is authorized by the signature entry
        # at index 1, so the sender signs twice: index 0 for the execution
        # approval and index 1 for the payment approval it falls back to.
        signatures=[
            FrameSignature(
                scheme=Spec.SCHEME_SECP256K1,
                signer=Bytes(sender),
            ),
            FrameSignature(
                scheme=Spec.SCHEME_SECP256K1,
                signer=Bytes(sender),
                secret_key=sender.key,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS, gas_used=0),
                FrameReceipt(status=Spec.STATUS_FAILURE),
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=first_use_gas,
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
            paymaster: Account(balance=paymaster_balance),
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
    Pin durability against a later revert.

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
    Pin durability against the approving frame's own revert.

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
    Prove bookkeeping does not warm the manager.

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
    Pin durability against a batch rollback.

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
    Pin paymaster payment.

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
    Prove execution-only approval is not payment.

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
    Pin that the approval's nonce effect outlives an atomic-batch rollback.

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
    Pin key-zero overflow.

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


@pytest.mark.parametrize(
    "create_count",
    [
        pytest.param(1, id="single_create"),
        pytest.param(2, id="two_creates"),
        pytest.param(3, id="three_creates"),
    ],
)
@EIPChecklist.TransactionType.Test.SenderAccount.Nonce()
def test_key_zero_approval_increments_live_nonce_not_sequence(
    state_test: StateTestFiller,
    pre: Alloc,
    create_count: int,
) -> None:
    """
    Pin that key-zero approval increments the *live* account nonce rather
    than assigning `nonce_seq+1`.

    The EIP: "increments the sender's current account nonce; it does not set
    the account nonce to `tx.nonce_seq + 1`". Stateful validity forces
    `nonce_seq` to equal the account nonce at transaction start, so the two
    readings only diverge once an earlier frame has already moved the live
    nonce -- the case the EIP names explicitly ("by executing `CREATE` or
    `CREATE2` at `tx.sender`").

    Frame zero runs `create_count` CREATEs at the sender, taking the live
    nonce from one to `1 + create_count`; frame one takes the payment
    approval. Hand-derived expectation, straight from the EIP's wording:
    final nonce = `1 + create_count + 1`. An implementation assigning
    `tx.nonce_seq + 1` yields two for every arm, so the arms disagree with
    it by one, two and three respectively.

    No frame carries `ATOMIC_BATCH_FLAG`, which is what makes the assertion
    load-bearing: with a batch present, a rollback restores the nonce from
    the separately captured approval snapshot rather than from the
    consumption step, and the increment itself stops being observable.

    CREATE derives its address from the creator's nonce, so storing each
    return value makes the live nonce sequence directly observable:
    slot `i + 1` must hold `keccak256(rlp([sender, 1 + i]))[12:]`. The
    sentinel in the last slot is written after the CREATEs, so a frame that
    halts early leaves it zero and fails rather than passing quietly.
    """
    initial_nonce = 1
    canary_slot = 0xFF
    canary_value = 0xC0DE

    creating_frame_code = Bytecode()
    for index in range(create_count):
        creating_frame_code += Op.SSTORE(index + 1, Op.CREATE(0, 0, 0))
    creating_frame_code += Op.SSTORE(canary_slot, canary_value) + Op.STOP

    sender = pre.deploy_contract(
        code=Conditional(
            condition=Op.ISZERO(Op.TXPARAM(0x0A)),
            if_true=creating_frame_code,
            if_false=approve_bytecode(Spec.APPROVE_EXECUTION_AND_PAYMENT),
        ),
        balance=10**19,
        nonce=initial_nonce,
    )

    expected_storage: dict[int, int | Address] = {
        index + 1: compute_create_address(
            address=sender, nonce=initial_nonce + index
        )
        for index in range(create_count)
    }
    expected_storage[canary_slot] = canary_value

    tx = Transaction(
        sender=sender,
        nonce_keys=[0],
        nonce_seq=initial_nonce,
        frames=[
            # Sized well above the creating frame's cost rather than
            # tuned to it: Amsterdam prices each CREATE at 32,000 plus
            # 183,600 state gas and each fresh SSTORE at 107,920, so a
            # one-million budget silently starves the three-CREATE arm
            # and the whole probe collapses back to a bare increment.
            Frame(mode=Spec.MODE_DEFAULT, gas_limit=4_000_000),
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=300_000,
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
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                nonce=initial_nonce + create_count + 1,
                storage=expected_storage,
            )
        },
    )
