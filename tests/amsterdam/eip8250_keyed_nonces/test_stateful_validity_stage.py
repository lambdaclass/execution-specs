"""
Stage and pre-state tests for EIP-8250's stateful validity check.

The EIP is specific about two things beyond the check's arithmetic: the
sequence it compares against is read from "the transaction's actual
pre-state within block execution order", and the comparison "occurs at the
same stage as EIP-8141's existing nonce check, before any frame executes".

Both are statements about *when* the check happens, so neither is observable
from a transaction studied on its own with the outcome as the only signal.
The first needs a preceding transaction in the same block to move the domain
under the transaction being checked; the second needs a frame whose execution
would leave a mark, so that a rejection reached late is distinguishable from
one reached on time.
"""

from typing import Dict

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Block,
    BlockchainTestFiller,
    EIPChecklist,
    Environment,
    Frame,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

# Written by the frame that must not run when validity fails. Distinctive and
# non-zero, so an untouched slot is distinguishable from a slot the frame
# wrote: storing zero would be indistinguishable from never storing at all.
SENTINEL = 0xC0DE


@pytest.mark.parametrize(
    "nonce_seq,error",
    [
        pytest.param(1, None, id="sequence_follows_preceding_transaction"),
        pytest.param(
            0,
            TransactionException.NONCE_MISMATCH_TOO_LOW,
            id="sequence_stale_at_block_start",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@EIPChecklist.TransactionType.Test.BlockInteractions.MixedTxs()
def test_legacy_domain_tracks_preceding_transaction_in_block(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    nonce_seq: int,
    error: TransactionException | None,
) -> None:
    """
    Prove `tx_legacy_nonce` is read at the transaction's block position.

    One block, one sender, two transactions. The first is an ordinary
    value-transfer transaction, which by ordinary EVM rules takes the sender's
    account nonce from 0 to 1. The second is a frame transaction on the
    `nonce_keys == [0]` domain, whose selected sequence is therefore
    `state[sender].nonce` -- and the expectation is hand-derived by asking
    which value that is at the second transaction's position: 1, the value the
    first transaction left, not the 0 the block began with.

    Both arms are needed, and each is the other's discriminator. An
    implementation that snapshotted the sender's nonce at the start of the
    block would reject sequence 1 as too high and accept the stale sequence 0,
    inverting both verdicts. The valid arm additionally reads the two
    introspection parameters the EIP ties together for this domain -- "for
    `nonce_keys == [0]`, stateful validity requires
    `TXPARAM(0x01) == TXPARAM(0x0C)` at transaction start" -- and both must
    report the mid-block value 1, so an implementation that validated against
    the live nonce while reporting a block-start snapshot through `0x0C` fails
    here even though its verdict was right.

    The sender's final account nonce is 2: one increment from the ordinary
    transaction and one from payment approval, which for `[0]` increments the
    current nonce.
    """
    sender = pre.fund_eoa()
    recipient = pre.fund_eoa(amount=0)
    target = pre.deploy_contract(
        code=(
            Op.SSTORE(0, Op.TXPARAM(Spec.TXPARAM_NONCE_SEQ))
            + Op.SSTORE(1, Op.TXPARAM(Spec.TXPARAM_LEGACY_NONCE))
            + Op.SSTORE(2, SENTINEL)
            + Op.STOP
        )
    )

    ordinary = Transaction(sender=sender, to=recipient, value=1)
    keyed = Transaction(
        sender=sender,
        nonce_keys=[0],
        nonce_seq=nonce_seq,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=target,
                gas_limit=1_000_000,
            ),
        ],
        error=error,
    )

    # Keyed by Address: the senders are EOAs, the target a plain address.
    post: Dict[Address, Account]
    if error is None:
        post = {
            sender: Account(nonce=2),
            target: Account(storage={0: 1, 1: 1, 2: SENTINEL}),
            Spec.NONCE_MANAGER: Account(storage={}),
        }
    else:
        # The invalid transaction invalidates its block, so the ordinary
        # transaction that preceded it is not applied either.
        post = {
            sender: Account(nonce=0),
            target: Account(storage={}),
            Spec.NONCE_MANAGER: Account(storage={}),
        }

    blockchain_test(
        pre=pre,
        blocks=[Block(txs=[ordinary, keyed], exception=error)],
        post=post,
    )


@pytest.mark.parametrize(
    "nonce_keys,current_sequence,nonce_seq,error",
    [
        pytest.param([19], 5, 5, None, id="keyed_sequence_exact"),
        pytest.param(
            [19],
            5,
            6,
            TransactionException.NONCE_MISMATCH_TOO_HIGH,
            id="keyed_sequence_high",
            marks=pytest.mark.exception_test,
        ),
        pytest.param([0], 3, 3, None, id="legacy_sequence_exact"),
        pytest.param(
            [0],
            3,
            4,
            TransactionException.NONCE_MISMATCH_TOO_HIGH,
            id="legacy_sequence_high",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.IntrinsicValidity.NoncePlusOne()
def test_sequence_mismatch_precedes_all_frame_execution(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_keys: list[int],
    current_sequence: int,
    nonce_seq: int,
    error: TransactionException | None,
) -> None:
    """
    Prove the sequence check precedes every frame on both domains.

    Every transaction here carries a second frame that calls a contract whose
    only job is to write a sentinel. On the arms whose sequence is exact, that
    write lands and the selected domain advances, which is what makes the
    other arms capable of failing: a rejection is only evidence of ordering if
    the same setup demonstrably does execute the frame when the check passes.
    On the arms whose sequence is one above the current sequence, the
    transaction is rejected and the sentinel slot must still be zero. An
    implementation that ran frames first and rejected afterwards would leave
    the sentinel written, because a rejected transaction's frames are not
    unrolled -- they were never authorised to run.

    The current sequence is planted directly, one domain at a time, and the
    expected verdicts come from the EIP's assertion
    `tx.nonce_seq == current_nonce_seq(tx.sender, nonce_key)`: 5 against a
    planted 5 and 3 against a planted account nonce of 3 are exact, and each
    domain's `+1` arm exceeds what its own branch of `current_nonce_seq`
    returns. The advanced value on the passing arms is likewise the rule's
    own: `nonce_seq + 1 == 6` in the manager slot for the keyed domain, and
    one increment of the current account nonce -- 4 -- for the legacy alias.

    The approving frame is given zero gas on every arm. Both passing arms
    select a domain that is already non-zero, so the first-use count is zero
    and no surcharge is due; the keyed slot is planted rather than fresh
    precisely so that the surcharge cannot mask a frame-ordering fault by
    exhausting the frame for an unrelated reason.
    """
    sender = pre.fund_eoa(nonce=current_sequence if nonce_keys == [0] else 0)
    sentinel = pre.deploy_contract(code=Op.SSTORE(0, SENTINEL) + Op.STOP)

    keyed_slot = None
    if nonce_keys != [0]:
        keyed_slot = Spec.nonce_slot(sender, nonce_keys[0])
        pre[Spec.NONCE_MANAGER] = Account(
            nonce=1,
            code=Spec.NONCE_MANAGER_CODE,
            storage={keyed_slot: current_sequence},
        )

    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=nonce_seq,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=sentinel,
                gas_limit=1_000_000,
            ),
        ],
        error=error,
    )

    executed = error is None
    # Keyed by Address: the sender is an EOA, the others plain addresses.
    post: Dict[Address, Account] = {
        sentinel: Account(storage={0: SENTINEL} if executed else {}),
    }
    if nonce_keys == [0]:
        post[sender] = Account(
            nonce=current_sequence + 1 if executed else current_sequence
        )
        post[Spec.NONCE_MANAGER] = Account(storage={})
    else:
        assert keyed_slot is not None
        post[sender] = Account(nonce=0)
        post[Spec.NONCE_MANAGER] = Account(
            storage={
                keyed_slot: current_sequence + 1
                if executed
                else current_sequence
            }
        )

    state_test(env=Environment(), pre=pre, tx=tx, post=post)
