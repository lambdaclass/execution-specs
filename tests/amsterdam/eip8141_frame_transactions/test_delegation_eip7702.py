"""
Delegation tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141)
composed with [EIP-7702](https://eips.ethereum.org/EIPS/eip-7702).

A delegated account never runs the default code: frames resolving to
it execute the delegate under the account's own context, both when it
is the transaction sender and when it is an ordinary frame target.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
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
"""Storage slot the delegate writes what it read into."""

PROBE_FRAME_GAS = 500_000
"""Gas limit of frames whose code writes storage, leaving room for
the state gas of fresh writes under EIP-8037."""


@pytest.mark.parametrize(
    "delegate_approves,error",
    [
        pytest.param(True, None, id="delegate_approves"),
        pytest.param(
            False,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="delegate_reverts",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_delegated_sender_bypasses_default_code(
    state_test: StateTestFiller,
    pre: Alloc,
    delegate_approves: bool,
    error: TransactionException | None,
) -> None:
    """
    Run the delegate instead of the default code for a delegated
    sender's `VERIFY` frame: an approving delegate validates the
    transaction, and a reverting delegate invalidates it even though
    the canonical signature entry is present — proving the default
    code was bypassed.
    """
    delegate = pre.deploy_contract(
        code=Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT)
        if delegate_approves
        else Op.REVERT(0, 0)
    )
    sender = pre.fund_eoa(delegation=delegate)

    tx = Transaction(
        sender=sender,
        frames=[verify_frame()],
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        # A delegated account enters the pre-state at nonce one, as if
        # it had sent its authorization.
        post={sender: Account(nonce=2 if error is None else 1)},
    )


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(Spec.MODE_DEFAULT, id="default_frame"),
        pytest.param(Spec.MODE_SENDER, id="sender_frame"),
    ],
)
def test_delegated_target_in_frames(
    state_test: StateTestFiller,
    pre: Alloc,
    mode: int,
) -> None:
    """
    Execute the delegate for frames targeting a delegated account
    that is not the sender: the write lands in the delegated
    account's own storage, and the delegate reads frame transaction
    introspection — the executing frame's index plus one — proving
    the composition works under delegation.
    """
    sender = pre.fund_eoa()
    delegate = pre.deploy_contract(
        code=Op.SSTORE(
            SLOT_RESULT, Op.ADD(Op.TXPARAM(Spec.TXPARAM_FRAME_INDEX), 1)
        )
        + Op.STOP
    )
    delegated = pre.fund_eoa(amount=1, delegation=delegate)

    frame = (
        default_frame(target=delegated, gas_limit=PROBE_FRAME_GAS)
        if mode == Spec.MODE_DEFAULT
        else sender_frame(target=delegated, gas_limit=PROBE_FRAME_GAS)
    )
    tx = Transaction(
        sender=sender,
        frames=[verify_frame(), frame],
    )

    state_test(
        pre=pre,
        tx=tx,
        post={
            delegated: Account(storage={SLOT_RESULT: 2}),
            delegate: Account(storage={SLOT_RESULT: 0}),
        },
    )
