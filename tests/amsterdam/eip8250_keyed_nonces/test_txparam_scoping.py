"""TXPARAM index-gap and transaction-scoping tests for EIP-8250."""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytecode,
    EIPChecklist,
    Environment,
    Frame,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

CANARY = 0xC0DE
"""Sentinel proving the probing frame ran before the parameter read."""

PROBE_FRAME_GAS = 1_000_000
"""
Gas limit of the probing frame; an exceptional halt consumes all of it.

Sized well above the two `SSTORE`s the probe performs, which the defined
control arm demonstrates by completing both writes within it. A halt can
therefore only come from the parameter index under test, not exhaustion.
"""

SCOPING_FRAME_GAS = 8_000_000
"""Gas limit of the scoping probe, which performs eleven `SSTORE`s."""

SINGLE_ZERO_KEY_HASH = int(
    "ada5013122d395ba3c54772283fb069b10426056ef8ca54750cb9bb552a59e7d", 16
)
"""Hard-coded `nonce_keys_hash` vector for the key set `[0]`."""


@pytest.mark.parametrize(
    "param_index,defined",
    [
        pytest.param(0x0D, True, id="defined_key_count"),
        pytest.param(0x0F, False, id="undefined_reserved_gap"),
        pytest.param(0x11, False, id="undefined_above_key_zero"),
    ],
)
@EIPChecklist.TransactionType.Test.TxScopedAttributes.Read()
def test_undefined_txparam_index_halts(
    state_test: StateTestFiller,
    pre: Alloc,
    param_index: int,
    defined: bool,
) -> None:
    """
    Pin R-062 and R-105.

    EIP-8141 assigns `TXPARAM` indices through `0x0B` and this EIP adds only
    `0x0C`, `0x0D`, `0x0E`, and `0x10`, so `0x0F` and `0x11` stay undefined
    and must halt exceptionally rather than return zero. Every arm runs the
    same bytecode and differs only in the index, so the outcomes are opposed
    by construction: the defined control stores the canary and the
    independently counted key count two, while an undefined index discards
    both writes. The halt is distinguished from an ordinary revert by the
    hand-derived receipt value `gas_used == PROBE_FRAME_GAS`, because an
    exceptional halt consumes the frame's entire gas limit.
    """
    sender = pre.fund_eoa()
    nonce_keys = [1, 2]
    approval_gas = Spec.KEYED_NONCE_FIRST_USE_GAS * len(nonce_keys)
    probe = pre.deploy_contract(
        code=(
            Op.SSTORE(0, CANARY)
            + Op.SSTORE(1, Op.TXPARAM(param_index))
            + Op.STOP
        )
    )

    if defined:
        probe_receipt = FrameReceipt(status=Spec.STATUS_SUCCESS)
        probe_storage = {0: CANARY, 1: len(nonce_keys)}
    else:
        probe_receipt = FrameReceipt(
            status=Spec.STATUS_FAILURE,
            gas_used=PROBE_FRAME_GAS,
        )
        probe_storage = {0: 0, 1: 0}

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
                target=probe,
                gas_limit=PROBE_FRAME_GAS,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=approval_gas,
                ),
                probe_receipt,
            ],
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            probe: Account(storage=probe_storage),
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(sender, nonce_key): 1
                    for nonce_key in nonce_keys
                }
            ),
        },
    )


@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.TxScopedAttributes.Read()
@EIPChecklist.TransactionType.Test.TxScopedAttributes.Persistent.Throughout()
def test_txparam_values_are_transaction_scoped(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-063, R-069, and R-070.

    The five keyed-nonce parameters are transaction-scoped, so payment
    approval and a `CREATE` executed inside the transaction must not move
    them. The probe reads all five, performs a `CREATE` that advances its own
    account nonce, then reads all five again into a second slot range; the
    expected values in both ranges are the same hand-derived literals. Key
    zero is selected so that approval has already incremented the sender's
    account nonce from seven to eight before the probe runs, which makes
    `TXPARAM(0x0C)` return the pre-state seven rather than the live eight and
    keeps R-070's `TXPARAM(0x01) == TXPARAM(0x0C)` equality observable. The
    key-set hash is the hard-coded `[0]` vector, not a recomputation.
    """
    initial_nonce = 7
    sender = pre.fund_eoa(nonce=initial_nonce)

    def read_parameters(base_slot: int) -> Bytecode:
        return (
            Op.SSTORE(base_slot + 0, Op.TXPARAM(Spec.TXPARAM_NONCE_SEQ))
            + Op.SSTORE(base_slot + 1, Op.TXPARAM(Spec.TXPARAM_LEGACY_NONCE))
            + Op.SSTORE(
                base_slot + 2, Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_COUNT)
            )
            + Op.SSTORE(
                base_slot + 3, Op.TXPARAM(Spec.TXPARAM_NONCE_KEYS_HASH)
            )
            + Op.SSTORE(base_slot + 4, Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_0))
        )

    probe = pre.deploy_contract(
        code=(
            read_parameters(0)
            + Op.POP(Op.CREATE(0, 0, 0))
            + read_parameters(5)
            + Op.SSTORE(10, CANARY)
            + Op.STOP
        ),
        nonce=1,
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
                target=probe,
                gas_limit=SCOPING_FRAME_GAS,
            ),
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS, gas_used=0),
                FrameReceipt(status=Spec.STATUS_SUCCESS),
            ],
        ),
    )

    expected = {
        0: initial_nonce,
        1: initial_nonce,
        2: 1,
        3: SINGLE_ZERO_KEY_HASH,
        4: 0,
    }
    probe_storage = dict(expected)
    probe_storage.update({slot + 5: value for slot, value in expected.items()})
    probe_storage[10] = CANARY

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            # Approval advanced the live account nonce past the snapshot.
            sender: Account(nonce=initial_nonce + 1),
            probe: Account(nonce=2, storage=probe_storage),
        },
    )
