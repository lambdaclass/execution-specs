"""
Tests for
[EIP-8504: Proof Frames for leanSPHINCS Signatures](https://eips.ethereum.org/EIPS/eip-8504)
across the fork that activates it.

Before the fork a block has neither the list of dependency witnesses nor
the header field that commits to it. Both exist from the block that
activates the fork, which can already carry a witness.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Block,
    BlockchainTestFiller,
    Bytes,
    Dependency,
    Fork,
    Header,
    LeanSphincsKey,
    Op,
    Transaction,
)
from execution_testing.fixtures.blockchain import dependency_witnesses_hash

from ..eip8141_frame_transactions.helpers import (
    FrameTransaction,
    sender_frame,
)
from .helpers import account_code, account_frame, proof_frame, witnesses_of
from .spec import ref_spec_8504

REFERENCE_SPEC_GIT_PATH = ref_spec_8504.git_path
REFERENCE_SPEC_VERSION = ref_spec_8504.version

pytestmark = pytest.mark.valid_at_transition_to("Bogota")

FORK_TIMESTAMP = 15_000
"""Timestamp at which the transition fork activates EIP-8504."""

SLOT_EXECUTED = 0x01
"""Storage slot a target contract sets when a `SENDER` frame reaches it."""

KEY = LeanSphincsKey.from_index(1)
"""Key controlling the account the post-fork transaction is sent from."""


def test_first_block_of_the_fork_carries_a_witness(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Include a transaction with a dependency in the block that activates
    the fork, after a pre-fork block that has no dependency witnesses.

    The fork block's header commits to the one witness the block carries.
    The pre-fork block is built under the previous fork's rules, whose
    header has no such field.
    """
    post_fork = fork.transitions_to()
    plain_sender = pre.fund_eoa()
    recipient = pre.fund_eoa(amount=0)

    sender = pre.deploy_contract(
        code=account_code(KEY.public_key, frame_index=1), balance=10**18
    )
    target_code = Op.SSTORE(SLOT_EXECUTED, 1) + Op.STOP
    target = pre.deploy_contract(code=target_code)
    tx = FrameTransaction.build(
        fork=post_fork,
        sender=sender,
        nonce=1,
        signatures=[],
        frames=[
            account_frame(post_fork, KEY.public_key, frame_index=1),
            proof_frame(post_fork, [Dependency.lean_sphincs(KEY.public_key)]),
            sender_frame(target=target, code=target_code),
        ],
        dependency_signers=[KEY],
    )
    witnesses = [Bytes(witness) for witness in witnesses_of(tx, [KEY])]

    blockchain_test(
        pre=pre,
        post={
            recipient: Account(balance=1),
            sender: Account(nonce=2),
            target: Account(storage={SLOT_EXECUTED: 1}),
        },
        blocks=[
            Block(
                timestamp=FORK_TIMESTAMP - 1,
                txs=[Transaction(sender=plain_sender, to=recipient, value=1)],
            ),
            Block(
                timestamp=FORK_TIMESTAMP,
                txs=[tx],
                header_verify=Header(
                    dependency_witnesses_hash=dependency_witnesses_hash(
                        witnesses
                    )
                ),
            ),
        ],
    )
