"""Durability boundary between keyed consumption and VERIFY reverts."""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Environment,
    Frame,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")


@pytest.mark.parametrize(
    "verify_frame_reverts",
    [
        pytest.param(
            True,
            id="verify_reverts_invalid",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(False, id="verify_completes_valid"),
    ],
)
def test_verify_revert_unrolls_keyed_consumption(
    state_test: StateTestFiller,
    pre: Alloc,
    verify_frame_reverts: bool,
) -> None:
    """
    Bound the durability rule.

    Keyed consumption is journaled outside the frame revert journal, so it
    survives a later ordinary frame revert and an atomic-batch restore. A
    reverting `VERIFY` frame is the deliberate exception: EIP-8141 makes the
    whole transaction invalid, and an invalid transaction is never included,
    so no manager slot may be written. This is the negative counterpart of
    `test_approval_survives_later_frame_revert`, which pins the surviving
    case with a `DEFAULT`-mode reverting frame.

    Both arms share one structure and differ only in the second frame's
    target bytecode, so the outcomes are opposed by construction rather than
    by separate setups. The expected slot values are hand-derived from
    `consume_nonce_set`: `nonce_seq + 1 == 1` when the transaction is
    included, and the absent-slot reading of zero when it is rejected.
    """
    sender = pre.fund_eoa()
    nonce_key = 41
    verifier = pre.deploy_contract(
        code=Op.REVERT(0, 0) if verify_frame_reverts else Op.STOP
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
                mode=Spec.MODE_VERIFY,
                target=verifier,
                gas_limit=100_000,
            ),
        ],
        error=(
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION
            if verify_frame_reverts
            else None
        ),
        expected_receipt=(
            None
            if verify_frame_reverts
            else TransactionReceipt(
                payer=sender,
                frame_receipts=[
                    FrameReceipt(
                        status=Spec.STATUS_SUCCESS,
                        gas_used=Spec.KEYED_NONCE_FIRST_USE_GAS,
                    ),
                    FrameReceipt(status=Spec.STATUS_SUCCESS),
                ],
            )
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(sender, nonce_key): (
                        0 if verify_frame_reverts else 1
                    )
                }
            )
        },
    )
