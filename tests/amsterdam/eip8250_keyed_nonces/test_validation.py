"""Static keyed-nonce validation tests for EIP-8250."""

from typing import Dict

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    EIPChecklist,
    Environment,
    Frame,
    StateTestFiller,
    Transaction,
    TransactionException,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")


@pytest.mark.parametrize(
    "nonce_keys,error",
    [
        pytest.param(
            [],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="empty",
            marks=pytest.mark.exception_test,
        ),
        pytest.param([1], None, id="minimum"),
        pytest.param(list(range(1, 17)), None, id="maximum"),
        pytest.param(
            list(range(1, 18)),
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="maximum_plus_one",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@EIPChecklist.TransactionType.Test.Encoding.ListField.Zero()
@EIPChecklist.TransactionType.Test.Encoding.ListField.Max()
@EIPChecklist.TransactionType.Test.Encoding.ListField.MaxPlusOne()
def test_nonce_key_collection_bounds(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_keys: list[int],
    error: TransactionException | None,
) -> None:
    """
    Pin R-006, R-020, R-021, R-042, and R-043.

    The cases are the direct 1..16 inclusive boundary at 0, 1, 16, and 17;
    valid controls receive exactly 20,000 gas per independently counted key.
    """
    sender = pre.fund_eoa()
    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=20_000 * len(nonce_keys),
            )
        ],
        error=error,
    )
    post = {}
    if error is None:
        post[Spec.NONCE_MANAGER] = Account(
            storage={Spec.nonce_slot(sender, key): 1 for key in nonce_keys}
        )

    state_test(env=Environment(), pre=pre, tx=tx, post=post)


@pytest.mark.parametrize(
    "nonce_keys,error",
    [
        pytest.param(
            [1, 1],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="duplicate",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [2, 1],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="descending",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [0, 1],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="mixed_zero",
            marks=pytest.mark.exception_test,
        ),
        pytest.param([0], None, id="zero_singleton"),
        pytest.param([1, 2], None, id="ascending"),
        pytest.param([255, 256], None, id="numeric_not_lexicographic"),
    ],
)
def test_strict_numeric_order_and_zero_singleton(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_keys: list[int],
    error: TransactionException | None,
) -> None:
    """
    Pin R-048 and R-049.

    Cases are hand-enumerated numeric comparisons; [255,256] is the control
    that distinguishes integer order from variable-width byte-string order.
    """
    sender = pre.fund_eoa()
    first_use_count = 0 if nonce_keys == [0] else len(nonce_keys)
    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=20_000 * first_use_count,
            )
        ],
        error=error,
    )
    # Keyed by Address: the sender is an EOA, the manager is a plain address.
    post: Dict[Address, Account] = {}
    if error is None and nonce_keys == [0]:
        post[sender] = Account(nonce=1)
    elif error is None:
        post[Spec.NONCE_MANAGER] = Account(
            storage={Spec.nonce_slot(sender, key): 1 for key in nonce_keys}
        )

    state_test(env=Environment(), pre=pre, tx=tx, post=post)


@pytest.mark.parametrize(
    "nonce_keys,nonce_seq,error",
    [
        pytest.param(
            [],
            0,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="empty_key_list",
        ),
        pytest.param(
            list(range(1, 18)),
            0,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="too_many_keys",
        ),
        pytest.param(
            [1, 1],
            0,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="duplicate_keys",
        ),
        pytest.param(
            [2, 1],
            0,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="keys_not_increasing",
        ),
        pytest.param(
            [0, 1],
            0,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="zero_with_another_key",
        ),
        pytest.param(
            [1],
            Spec.MAX_NONCE_SEQ,
            TransactionException.NONCE_IS_MAX,
            id="reserved_sequence",
        ),
    ],
)
@pytest.mark.exception_test
@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.OutOfBounds.MaxPlusOne()
def test_invalid_nonce_fields(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_keys: list[int],
    nonce_seq: int,
    error: TransactionException,
) -> None:
    """
    Pin R-042, R-043, R-048, R-049, and R-065.

    Each malformed list is directly constructed from the forbidden predicate;
    MAX_NONCE_SEQ is the hard-coded reserved value, not a client-derived limit.
    """
    sender = pre.fund_eoa()
    if nonce_seq == Spec.MAX_NONCE_SEQ and nonce_keys == [1]:
        pre[Spec.NONCE_MANAGER] = Account(
            nonce=1,
            code=Spec.NONCE_MANAGER_CODE,
            storage={Spec.nonce_slot(sender, 1): Spec.MAX_NONCE_SEQ},
        )
    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=nonce_seq,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=400_000,
            )
        ],
        error=error,
    )

    state_test(env=Environment(), pre=pre, tx=tx, post={})
