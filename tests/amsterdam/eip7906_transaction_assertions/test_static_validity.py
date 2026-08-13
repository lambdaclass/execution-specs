"""
Static validity tests for the `POST_TX` frame mode of
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

Each case starts from a minimal valid frame transaction carrying a
trailing `POST_TX` frame and applies one variation that violates a
static validity rule, or sits exactly on its boundary.
"""

from typing import Callable, List, Optional, Union

import pytest
from execution_testing import (
    EOA,
    Account,
    Alloc,
    EIPChecklist,
    Frame,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
)

from tests.amsterdam.eip8141_frame_transactions.helpers import (
    default_frame,
    sender_frame,
    verify_frame,
)
from tests.amsterdam.eip8141_frame_transactions.spec import Spec as FrameSpec

from .helpers import post_tx_frame
from .spec import Spec, ref_spec_7906

REFERENCE_SPEC_GIT_PATH = ref_spec_7906.git_path
REFERENCE_SPEC_VERSION = ref_spec_7906.version

# EIP-7906 extends EIP-8141, which is slated for the fork after
# Amsterdam, so fixtures are labeled with the pseudo `Bogota` fork
# (Amsterdam + EIP-8141 + EIP-7906), even though the spec prototypes
# the EIPs inside the Amsterdam fork module. Fill these tests with
# `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

FrameListOrFactory = Union[List[Frame], Callable[[Alloc, EOA], List[Frame]]]

ModifiedValidityChecklist = (
    EIPChecklist.ModifiedTransactionValidityConstraint.Test.ForkTransition
)


@ModifiedValidityChecklist.AcceptedAfterFork()
@ModifiedValidityChecklist.RejectedAfterFork()
@pytest.mark.parametrize(
    "mode,error",
    [
        pytest.param(Spec.MODE_POST_TX, None, id="post_tx_mode"),
        pytest.param(
            Spec.FIRST_UNDEFINED_MODE,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="first_undefined_mode",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            255,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="high_undefined_mode",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_post_tx_mode_boundary(
    state_test: StateTestFiller,
    pre: Alloc,
    mode: int,
    error: Optional[TransactionException],
) -> None:
    """
    Vary the mode of the trailing frame of an otherwise valid frame
    transaction across the boundary EIP-7906 moves: mode 3 (`POST_TX`)
    becomes valid, while 4 and above remain invalid.

    Rules: R-001, R-004, R-005.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        frames=[
            verify_frame(),
            default_frame(),
            post_tx_frame(mode=mode),
        ],
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        # The sender's nonce only increments if the transaction is
        # valid and executes.
        post={sender: Account(nonce=0 if error else 1)},
    )


@pytest.mark.parametrize(
    "frames,error",
    [
        pytest.param(
            [verify_frame(), post_tx_frame()],
            None,
            id="suffix_directly_after_prefix",
        ),
        pytest.param(
            [verify_frame(), default_frame(), post_tx_frame()],
            None,
            id="single_frame_suffix",
        ),
        pytest.param(
            [
                verify_frame(),
                default_frame(),
                post_tx_frame(),
                post_tx_frame(),
                post_tx_frame(),
            ],
            None,
            id="multi_frame_suffix",
        ),
        pytest.param(
            [verify_frame(), post_tx_frame(), default_frame()],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="default_frame_after_post_tx",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [verify_frame(), post_tx_frame(), sender_frame()],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="sender_frame_after_post_tx",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [
                verify_frame(),
                post_tx_frame(),
                verify_frame(flags=FrameSpec.APPROVE_NONE),
            ],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="verify_frame_after_post_tx",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [
                verify_frame(),
                default_frame(),
                post_tx_frame(),
                default_frame(),
                post_tx_frame(),
            ],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="interleaved_suffix",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_post_tx_suffix_contiguity(
    state_test: StateTestFiller,
    pre: Alloc,
    frames: List[Frame],
    error: Optional[TransactionException],
) -> None:
    """
    Check that `POST_TX` frames must form a contiguous trailing suffix
    of the frame list: any frame of another mode after the first
    `POST_TX` frame invalidates the transaction, while a suffix of any
    length — including one directly after the validation prefix, a
    position where a `VERIFY` frame would violate the mempool shape
    rule — is accepted.

    Rules: R-006, R-117.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        frames=frames,
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        # The sender's nonce only increments if the transaction is
        # valid and executes.
        post={sender: Account(nonce=0 if error else 1)},
    )


@pytest.mark.parametrize(
    "value,error",
    [
        pytest.param(0, None, id="zero_value"),
        pytest.param(
            1,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="nonzero_value",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_post_tx_value_must_be_zero(
    state_test: StateTestFiller,
    pre: Alloc,
    value: int,
    error: Optional[TransactionException],
) -> None:
    """
    Check that a `POST_TX` frame carrying a non-zero value is
    statically invalid: the EIP-8141 constraint restricting value
    transfer to `SENDER` frames applies unamended to the new mode.

    Rule: R-105.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        frames=[verify_frame(), post_tx_frame(value=value)],
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        # The sender's nonce only increments if the transaction is
        # valid and executes.
        post={sender: Account(nonce=0 if error else 1)},
    )


@pytest.mark.parametrize(
    "frames,error",
    [
        pytest.param(
            # The mode check is the tested rule; the batch bit on a
            # trailing frame overlaps with the terminator rules, but
            # every reading raises the same exception class.
            [
                verify_frame(),
                post_tx_frame(flags=FrameSpec.ATOMIC_BATCH_FLAG),
                post_tx_frame(),
            ],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="atomic_batch_flag_on_post_tx",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            # The flagged body frame is legal on its own; its batch
            # would be terminated by the first assertion frame, which
            # a failing batch could then skip.
            [
                verify_frame(),
                default_frame(flags=FrameSpec.ATOMIC_BATCH_FLAG),
                post_tx_frame(),
            ],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="atomic_batch_terminated_by_post_tx",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            # Identical to the previous case except for the unflagged
            # body frame closing the batch before the suffix.
            [
                verify_frame(),
                default_frame(flags=FrameSpec.ATOMIC_BATCH_FLAG),
                default_frame(),
                post_tx_frame(),
            ],
            None,
            id="atomic_batch_closed_before_suffix",
        ),
        pytest.param(
            [
                verify_frame(),
                post_tx_frame(
                    flags=FrameSpec.ATOMIC_BATCH_FLAG
                    | FrameSpec.APPROVE_PAYMENT
                ),
                post_tx_frame(),
            ],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="batch_and_payment_scope_flags",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [
                verify_frame(),
                post_tx_frame(
                    flags=FrameSpec.ATOMIC_BATCH_FLAG
                    | FrameSpec.APPROVE_EXECUTION_AND_PAYMENT
                ),
                post_tx_frame(),
            ],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="batch_and_full_approval_scope_flags",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            # The approval scope flags are statically valid with any
            # mode; the frame's code never calls APPROVE, so the dead
            # allowance stays unexercised at runtime.
            lambda pre, _sender: [
                verify_frame(),
                post_tx_frame(
                    flags=FrameSpec.APPROVE_PAYMENT,
                    target=pre.deploy_contract(code=Op.STOP),
                ),
            ],
            None,
            id="payment_scope_flag_statically_legal",
        ),
        pytest.param(
            # An execution-scope frame must target the sender, whose
            # empty code succeeds as a no-op.
            [
                verify_frame(),
                post_tx_frame(flags=FrameSpec.APPROVE_EXECUTION),
            ],
            None,
            id="execution_scope_flag_statically_legal",
        ),
        pytest.param(
            [
                verify_frame(),
                post_tx_frame(flags=FrameSpec.APPROVE_EXECUTION_AND_PAYMENT),
            ],
            None,
            id="full_approval_scope_flags_statically_legal",
        ),
    ],
)
def test_post_tx_flags_validity(
    state_test: StateTestFiller,
    pre: Alloc,
    frames: FrameListOrFactory,
    error: Optional[TransactionException],
) -> None:
    """
    Check the frame flag rules applied to the `POST_TX` mode: atomic
    batches may neither contain a `POST_TX` frame nor be terminated by
    one — a batch reaching into the assertion suffix could skip an
    assertion while keeping the batch's preceding state changes — while
    the approval scope bits remain statically valid with any mode.

    Rules: R-106, R-107, R-011 (static half).
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        frames=frames(pre, sender) if callable(frames) else frames,
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        # The sender's nonce only increments if the transaction is
        # valid and executes.
        post={sender: Account(nonce=0 if error else 1)},
    )


@pytest.mark.parametrize(
    "suffix_count,error",
    [
        pytest.param(FrameSpec.MAX_FRAMES - 1, None, id="suffix_fills_max"),
        pytest.param(
            FrameSpec.MAX_FRAMES,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="suffix_overflows_max",
            marks=pytest.mark.exception_test,
        ),
    ],
)
def test_post_tx_frames_count_toward_max_frames(
    state_test: StateTestFiller,
    pre: Alloc,
    suffix_count: int,
    error: Optional[TransactionException],
) -> None:
    """
    Overflow the frame count limit entirely with `POST_TX` frames:
    assertion frames count toward `MAX_FRAMES` like any other mode.

    Each suffix frame targets the sender's empty code and succeeds as
    a no-op within a small gas limit, keeping the transaction's
    derived gas limit far below the per-transaction cap.

    Rule: R-108.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        frames=[verify_frame()]
        + [post_tx_frame(gas_limit=1_000) for _ in range(suffix_count)],
        error=error,
    )

    state_test(
        pre=pre,
        tx=tx,
        # The sender's nonce only increments if the transaction is
        # valid and executes.
        post={sender: Account(nonce=0 if error else 1)},
    )


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "suffix_count",
    [
        pytest.param(1, id="single_post_tx_frame"),
        pytest.param(2, id="two_post_tx_frames"),
    ],
)
def test_all_post_tx_frames_never_sets_payer(
    state_test: StateTestFiller,
    pre: Alloc,
    suffix_count: int,
) -> None:
    """
    Reject a frame transaction consisting solely of `POST_TX` frames:
    `APPROVE` is forbidden inside the new mode, so no frame can ever
    approve gas payment, and the EIP-8141 end-of-execution payer check
    invalidates the transaction.

    Rule: R-115.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        frames=[post_tx_frame() for _ in range(suffix_count)],
        error=TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
    )

    state_test(
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=0)},
    )
