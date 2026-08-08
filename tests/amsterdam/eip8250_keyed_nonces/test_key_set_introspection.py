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

import pytest
from execution_testing import (
    Account,
    Alloc,
    EIPChecklist,
    Environment,
    Frame,
    Hash,
    Op,
    StateTestFiller,
    Transaction,
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
