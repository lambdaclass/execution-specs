"""
Key-set introspection tests for EIP-8250 at the maximum key count.

`test_introspection.py` reads all four new `TXPARAM` indices for a two-key
set and for the zero-key alias. Both are counts of one and two, where the
count fits in a byte and the hash preimage is short enough that an
off-by-one in its length prefix or a dropped trailing key would be an
unlikely coincidence rather than an excluded possibility.

These tests read the same indices at `MAX_NONCE_KEYS`, the largest set the
EIP admits, over a key set that mixes the narrowest and the widest key the
type allows.
"""

from typing import List

import pytest
from execution_testing import (
    Account,
    Alloc,
    Conditional,
    EIPChecklist,
    Environment,
    Frame,
    Hash,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

# Sixteen strictly increasing keys: fifteen single-byte keys and the widest
# key the `uint256` type holds. The wide key is last, so a preimage that
# stopped one key short, or that left the length prefix out, is a different
# 544-byte string rather than a near-miss.
MAXIMUM_KEY_SET = list(range(1, 16)) + [2**256 - 1]

# keccak256 over bytes32(16) || bytes32(k) for each k, computed by hand from
# the preimage the EIP defines and hard-coded here rather than recomputed
# from the same helper the transaction uses.
MAXIMUM_KEY_SET_HASH = Hash(
    "0x334d3aa4d3b82734ae319acfa0c3eb5d1f425a3b40229df6c6711d5319d5464c"
)
MAXIMUM_KEY_SET_HASH_WORD = int.from_bytes(MAXIMUM_KEY_SET_HASH, "big")

SENTINEL = 0xC0DE


@EIPChecklist.TransactionType.Test.TxScopedAttributes.Read()
def test_key_set_introspection_at_maximum_count(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-064 and R-067 at `len(nonce_keys) == MAX_NONCE_KEYS`.

    `TXPARAM(0x0D)` must report 16, the hand-counted length of the key set,
    and `TXPARAM(0x0E)` must report the hash of
    `uint256_to_bytes32(16) || concat(uint256_to_bytes32(k) for k in keys)`.
    That expected hash is a hard-coded 544-byte-preimage vector: the length
    prefix is one word and the sixteen keys are one word each. Hard-coding it
    is what separates this from a tautology -- recomputing the digest with the
    same helper the transaction was built from would agree with any preimage,
    including one that omitted the length or reversed the keys.

    The preimage is checked against this directory's reference helper as well,
    so a future edit to either the vector or the helper has to explain itself
    against the other.

    `TXPARAM(0x10)` must report 1, the first key by numeric order, and not the
    widest key that happens to be the most conspicuous member of the set.
    `TXPARAM(0x01)` reports the shared sequence, which is zero here because
    all sixteen keys are fresh; a derived non-zero value is stored beside it
    so that a read returning zero for the wrong reason -- a parameter that
    never resolved -- is not indistinguishable from the correct answer.

    The sixteen fresh keys attract sixteen first-use surcharges, so the
    approving frame is given `16 * 20,000 == 320,000` gas, hand-multiplied
    from the constant. Each key's slot then holds `nonce_seq + 1 == 1`.
    """
    assert len(MAXIMUM_KEY_SET) == Spec.MAX_NONCE_KEYS
    assert MAXIMUM_KEY_SET == sorted(set(MAXIMUM_KEY_SET))
    assert Spec.nonce_keys_hash(MAXIMUM_KEY_SET) == MAXIMUM_KEY_SET_HASH

    sender = pre.fund_eoa()
    target = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_COUNT))
            + Op.SSTORE(1, Op.TXPARAM(Spec.TXPARAM_NONCE_KEYS_HASH))
            + Op.SSTORE(2, Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_0))
            + Op.SSTORE(3, Op.ADD(Op.TXPARAM(Spec.TXPARAM_NONCE_SEQ), 1))
            + Op.SSTORE(4, SENTINEL)
            + Op.STOP
        )
    )
    first_use_gas = Spec.KEYED_NONCE_FIRST_USE_GAS * len(MAXIMUM_KEY_SET)

    tx = Transaction(
        sender=sender,
        nonce_keys=MAXIMUM_KEY_SET,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=first_use_gas,
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
            sender: Account(nonce=0),
            target: Account(
                storage={
                    0: len(MAXIMUM_KEY_SET),
                    1: MAXIMUM_KEY_SET_HASH_WORD,
                    2: MAXIMUM_KEY_SET[0],
                    3: 1,
                    4: SENTINEL,
                }
            ),
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(sender, key): 1 for key in MAXIMUM_KEY_SET
                }
            ),
        },
    )


@pytest.mark.parametrize(
    "nonce_keys,accepted",
    [
        pytest.param([0], True, id="legacy_alias_admitted"),
        pytest.param(
            [7],
            False,
            id="single_non_zero_key_refused",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            [1, 2],
            False,
            id="two_key_set_refused",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@EIPChecklist.TransactionType.Test.TxScopedAttributes.Read()
def test_legacy_assumption_guard_admits_only_the_zero_key_alias(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_keys: List[int],
    accepted: bool,
) -> None:
    """
    Pin R-086: the guard the EIP prescribes for verifier code that predates
    keyed nonces -- `TXPARAM(0x0D) == 1 and TXPARAM(0x10) == 0` -- must admit
    exactly the transactions on which `TXPARAM(0x01)` is still the legacy
    account nonce, and refuse every other key set.

    The rule is advice to contract authors, so the only way to pin it is to
    deploy the advice and show it works. The verifier here is that guard and
    nothing else: it runs in a `VERIFY` frame, stops when both conjuncts
    hold, and reverts otherwise. EIP-8141 makes a reverting `VERIFY` frame
    invalidate the whole transaction, so "the guard refused" and "the
    transaction was not included" are the same observation.

    The expectations follow from reading the two conjuncts against each key
    set:

    - `[0]` is the alias the guard exists to admit. Count is one and the
      first key is zero, so both conjuncts hold.
    - `[7]` has count one as well, so the `TXPARAM(0x10) == 0` conjunct is
      the only one that can reject it, and it is the arm that isolates
      that conjunct: a client reporting a first key of zero for every key
      set would be admitted here and nowhere else in this test.
    - `[1, 2]` is the multi-key refusal R-091 warns about, where an
      attacker appends keys the verifier never authenticated. Both
      conjuncts fail on it, so it isolates neither; it is here because a
      guard demonstrated only against single-key sets leaves the case the
      security note actually names unstated.

    The count conjunct cannot be isolated at all, and that is a property of
    the EIP rather than a gap in this test: R-031 rejects any key set that
    contains `0` alongside another key, so `TXPARAM(0x10) == 0` already
    implies the singleton `[0]` and `TXPARAM(0x0D) == 1` with it. No key set
    exists that satisfies the second conjunct and violates the first.

    The sentinel is what makes the admitted arm capable of failing. It is
    written by a later frame, after the guard has run, so it is present only
    when the transaction completed; on the refused arms the transaction is
    never included and the slot stays zero, which the post-state asserts
    rather than leaving unstated. The approving frame is given exactly the
    surcharge its own key set costs -- nothing for `[0]`, one per fresh
    key otherwise -- so a refusal cannot be an out-of-gas in disguise.
    """
    sender = pre.fund_eoa()
    guard = pre.deploy_contract(
        code=Conditional(
            condition=Op.AND(
                Op.EQ(Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_COUNT), 1),
                Op.EQ(Op.TXPARAM(Spec.TXPARAM_NONCE_KEY_0), 0),
            ),
            if_true=Op.STOP,
            if_false=Op.REVERT(0, 0),
        )
    )
    probe = pre.deploy_contract(code=Op.SSTORE(0, SENTINEL) + Op.STOP)
    first_use_gas = (
        0
        if nonce_keys == [0]
        else Spec.KEYED_NONCE_FIRST_USE_GAS * len(nonce_keys)
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=first_use_gas,
            ),
            Frame(
                mode=Spec.MODE_VERIFY,
                target=guard,
                gas_limit=100_000,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=probe,
                gas_limit=1_000_000,
            ),
        ],
        error=(
            None
            if accepted
            else TransactionException.TYPE_6_INVALID_FRAME_EXECUTION
        ),
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            probe: Account(storage={0: SENTINEL if accepted else 0}),
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(sender, key): 0
                    for key in nonce_keys
                    if key != 0
                }
            ),
        },
    )
