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
    Pin R-026, the `1 <= len(nonce_keys) <= MAX_NONCE_KEYS` bound.

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
    Pin R-030 and R-031.

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
    "nonce_keys,error",
    [
        pytest.param(
            [1, 3, 2],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="disorder_in_closing_pair",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            list(range(1, 16)) + [15],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="adjacent_equal_in_last_position",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [256, 2],
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="descending_numerically_ascending_bytewise",
            marks=pytest.mark.exception_test,
        ),
        pytest.param([1, 2, 3], None, id="ascending_control"),
    ],
)
def test_strict_increase_position_sweep(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_keys: list[int],
    error: TransactionException | None,
) -> None:
    """
    Pin R-030 where the violation sits away from the opening pair of keys.

    `test_strict_numeric_order_and_zero_singleton` above puts each of its
    violations in the only pair a two-key list has, so a comparison of just
    `nonce_keys[0]` against `nonce_keys[1]`, or of just the first item
    against the last, satisfies it. Every arm here keeps those two
    comparisons ascending and moves the single defect somewhere they cannot
    see:

    - `[1, 3, 2]` opens ascending and ends above where it started; only its
      closing pair is out of order.
    - `[1, ..., 15, 15]` fills the whole `MAX_NONCE_KEYS` allowance and
      repeats only in the sixteenth position, so a scan that stops short of
      the end still has to reach it.
    - `[256, 2]` is the reject-direction counterpart of the neighbouring
      `numeric_not_lexicographic` control. 256 is `0x0100` and 2 is `0x02`
      as minimal big-endian integers, so the pair ascends compared as byte
      strings and descends by the numeric value R-030 names.

    Expectations come from reading the rule literally: a key list that is not
    strictly increasing is rejected, whichever position the break occupies.
    `[1, 2, 3]` is the control, built from the reject arms' own key widths
    and given the same per-key frame gas, so the three rejections cannot be
    attributed to key count, key width or gas. Its post-state is
    `nonce_seq + 1 == 1` in each selected slot, straight from
    `consume_nonce_set`. The 16-key ascending twin of the second arm is
    `test_nonce_key_collection_bounds[maximum]` above and is not repeated
    here.
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
                gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS * len(nonce_keys),
            )
        ],
        error=error,
    )
    post: Dict[Address, Account] = {}
    if error is None:
        post[Spec.NONCE_MANAGER] = Account(
            storage={Spec.nonce_slot(sender, key): 1 for key in nonce_keys}
        )

    state_test(env=Environment(), pre=pre, tx=tx, post=post)


@pytest.mark.parametrize(
    "nonce_keys,nonce_seq,error",
    [
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
    Pin R-039, the `tx.nonce_seq < MAX_NONCE_SEQ` assertion that R-097's
    reserved exhausted state is expressed as.

    MAX_NONCE_SEQ is the hard-coded reserved value, not a client-derived
    limit. The slot is seeded to `MAX_NONCE_SEQ` so that the per-key
    sequence-match check cannot be what rejects the transaction and only
    the reserved-sequence rule can fire.

    The malformed-key-list predicates this module also rejects -- empty,
    seventeen keys, duplicated, descending, and zero beside another key --
    are covered by `test_nonce_key_collection_bounds` and
    `test_strict_numeric_order_and_zero_singleton` above, each against the
    same expected exception. They were duplicated here with a different
    frame gas limit, which changes nothing: every one of them is rejected
    before any frame runs.
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

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        # The seeded slot must survive untouched: a rejected transaction
        # consumes nothing, and asserting the exhausted value is still
        # there is stronger than asserting no post-state at all.
        post={
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(sender, key): Spec.MAX_NONCE_SEQ
                    for key in nonce_keys
                }
            )
        },
    )
