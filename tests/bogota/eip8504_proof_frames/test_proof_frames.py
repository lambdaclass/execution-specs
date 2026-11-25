"""
Tests for the proof frame and the dependency witnesses of
[EIP-8504: Proof Frames for leanSPHINCS Signatures](https://eips.ethereum.org/EIPS/eip-8504).

Each test exercises one rule the EIP adds: a transaction that declares a
dependency and an account that approves on it, a dependency shared by two
transactions, a block whose witnesses do not match its dependencies, a
block whose header commits to another witness list, and a proof frame that
breaks a static constraint.
"""

from typing import Callable, List, Tuple

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Block,
    BlockchainTestFiller,
    BlockException,
    Bytes,
    Dependency,
    Fork,
    FrameFlag,
    Hash,
    Header,
    LeanSphincsKey,
    Op,
    StateTestFiller,
    TransactionException,
)

from ..eip8141_frame_transactions.helpers import (
    FrameCase,
    FrameTransaction,
    sender_frame,
    verify_frame,
)
from .helpers import account_code, account_frame, proof_frame, witnesses_of
from .spec import Spec, ref_spec_8504

REFERENCE_SPEC_GIT_PATH = ref_spec_8504.git_path
REFERENCE_SPEC_VERSION = ref_spec_8504.version

pytestmark = pytest.mark.valid_from("Bogota")

SLOT_EXECUTED = 0x01
"""Storage slot a target contract sets when a `SENDER` frame reaches it."""

KEY = LeanSphincsKey.from_index(1)
"""Key controlling the account most tests send from."""

OTHER_KEY = LeanSphincsKey.from_index(2)
"""A key that does not control that account."""

MESSAGE = bytes(range(1, 33))
"""An explicit message, signed as it is rather than as a signature hash."""

PROOF_FRAME_INDEX = 1
"""Position of the proof frame in the transactions built here."""


def account_transaction(
    pre: Alloc, fork: Fork, **proof_overrides: object
) -> Tuple[FrameTransaction, Address, Address]:
    """
    Return a transaction from an account controlled by `KEY`, with that
    account and the contract its `SENDER` frame calls.

    The account approves in a `VERIFY` frame because the proof frame that
    follows declares a dependency on `KEY`'s signature of the transaction's
    signature hash. `proof_overrides` replace fields of that proof frame.
    """
    sender = pre.deploy_contract(
        code=account_code(KEY.public_key, frame_index=PROOF_FRAME_INDEX),
        balance=10**18,
    )
    target_code = Op.SSTORE(SLOT_EXECUTED, 1) + Op.STOP
    target = pre.deploy_contract(code=target_code)
    tx = FrameTransaction.build(
        fork=fork,
        sender=sender,
        nonce=1,
        signatures=[],
        frames=[
            account_frame(fork, KEY.public_key, frame_index=PROOF_FRAME_INDEX),
            proof_frame(
                fork,
                [Dependency.lean_sphincs(KEY.public_key)],
                **proof_overrides,
            ),
            sender_frame(target=target, code=target_code),
        ],
        dependency_signers=[KEY],
        error=(
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT
            if proof_overrides
            else None
        ),
    )
    return tx, sender, target


def test_account_approves_on_its_dependency(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Send a transaction from an account that a leanSPHINCS key controls.

    The transaction declares, in a proof frame, a dependency on the key's
    signature of its own signature hash. The account's code reads that
    declaration and approves. The proof frame runs nothing and reports its
    whole execution budget as used, and the block carries the signature as
    its only dependency witness.
    """
    tx, sender, target = account_transaction(pre, fork)

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=2),
            target: Account(storage={SLOT_EXECUTED: 1}),
        },
    )


def test_dependency_shared_by_two_transactions(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Include two transactions that declare the same dependency on an
    explicit message.

    Both pay for the witness, and the block carries it once.
    """
    dependency = Dependency.lean_sphincs(KEY.public_key, MESSAGE)
    target_code = Op.SSTORE(SLOT_EXECUTED, 1) + Op.STOP
    targets = [pre.deploy_contract(code=target_code) for _ in range(2)]
    txs = [
        FrameTransaction.build(
            fork=fork,
            sender=pre.fund_eoa(),
            frames=[
                verify_frame(logs=[]),
                proof_frame(fork, [dependency]),
                sender_frame(target=target, code=target_code),
            ],
            # The receipt's total is cumulative over the block, which the
            # second transaction's own total does not give.
            settle=False,
        )
        for target in targets
    ]

    blockchain_test(
        pre=pre,
        post={
            target: Account(storage={SLOT_EXECUTED: 1}) for target in targets
        },
        blocks=[Block(txs=txs, dependency_witnesses=[KEY.sign(MESSAGE)])],
    )


def tampered(witnesses: List[Bytes]) -> List[Bytes]:
    """Flip one bit in the last byte of the only witness."""
    (witness,) = witnesses
    return [Bytes(witness[:-1] + bytes([witness[-1] ^ 1]))]


def truncated(witnesses: List[Bytes]) -> List[Bytes]:
    """Drop the last byte of the only witness."""
    (witness,) = witnesses
    return [Bytes(witness[:-1])]


def missing(witnesses: List[Bytes]) -> List[Bytes]:
    """Drop the only witness."""
    del witnesses
    return []


def repeated(witnesses: List[Bytes]) -> List[Bytes]:
    """Give the only witness twice."""
    return witnesses * 2


def of_another_key(witnesses: List[Bytes]) -> List[Bytes]:
    """Replace the only witness by another key's signature of `MESSAGE`."""
    del witnesses
    return [OTHER_KEY.sign(MESSAGE)]


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(tampered, id="tampered_witness"),
        pytest.param(truncated, id="truncated_witness"),
        pytest.param(of_another_key, id="witness_of_another_key"),
        pytest.param(missing, id="missing_witness"),
        pytest.param(repeated, id="repeated_witness"),
    ],
)
def test_invalid_dependency_witnesses(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
    corrupt: Callable[[List[Bytes]], List[Bytes]],
) -> None:
    """
    Reject a block whose dependency witnesses are not exactly one valid
    witness for each dependency of its transactions.

    The transaction is valid and executes the same whatever the witnesses
    are, so only the block-level check can reject these blocks.
    """
    tx, sender, target = account_transaction(pre, fork)

    blockchain_test(
        pre=pre,
        post={
            sender: Account(nonce=1),
            target: Account(storage={SLOT_EXECUTED: 0}),
        },
        blocks=[
            Block(
                txs=[tx],
                dependency_witnesses=corrupt(witnesses_of(tx, [KEY])),
                exception=BlockException.INVALID_DEPENDENCY_WITNESSES,
            )
        ],
    )


@pytest.mark.exception_test
def test_invalid_dependency_witnesses_hash(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Reject a block whose header commits to another list than the dependency
    witnesses the block carries, although those witnesses are valid.
    """
    tx, sender, target = account_transaction(pre, fork)

    blockchain_test(
        pre=pre,
        post={
            sender: Account(nonce=1),
            target: Account(storage={SLOT_EXECUTED: 0}),
        },
        blocks=[
            Block(
                txs=[tx],
                rlp_modifier=Header(dependency_witnesses_hash=Hash(1)),
                exception=[
                    BlockException.INVALID_DEPENDENCY_WITNESSES_HASH,
                    BlockException.INVALID_BLOCK_HASH,
                ],
            )
        ],
    )


def one_triple(scheme: int = Spec.LEAN_SPHINCS_SCHEME) -> bytes:
    """Return one encoded triple naming `scheme` and `KEY`."""
    return scheme.to_bytes(32, "big") + bytes(32) + bytes(KEY.public_key)


@pytest.mark.exception_test
@pytest.mark.parametrize(
    "proof_overrides",
    [
        pytest.param({"data": b""}, id="no_dependency"),
        pytest.param({"data": one_triple()[:-1]}, id="partial_triple"),
        pytest.param(
            {"data": b"\x01" + one_triple()[1:]}, id="scheme_not_right_aligned"
        ),
        pytest.param({"target": Address(0x1234)}, id="with_target"),
        pytest.param({"flags": 1}, id="with_flags"),
        pytest.param({"state_gas_limit": 1}, id="with_state_gas"),
        pytest.param({"gas_limit_offset": 1}, id="too_much_execution_gas"),
        pytest.param({"gas_limit_offset": -1}, id="too_little_execution_gas"),
    ],
)
def test_invalid_proof_frame(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    proof_overrides: dict,
) -> None:
    """
    Reject a transaction whose proof frame breaks one of its static
    constraints: its data is a non-empty list of whole triples with a
    right-aligned scheme, it has no target, no flags and no state gas, and
    its execution gas is exactly what its triples call for.

    The block still carries a witness for each dependency that can be read
    from the frame, so the transaction is its only defect.
    """
    overrides = dict(proof_overrides)
    if "gas_limit_offset" in overrides:
        overrides["gas_limit"] = (
            fork.gas_costs().PROOF_VERIFICATION
            + overrides.pop("gas_limit_offset")
        )
    tx, sender, target = account_transaction(pre, fork, **overrides)

    state_test(
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=1),
            target: Account(storage={SLOT_EXECUTED: 0}),
        },
    )


@pytest.mark.exception_test
def test_proof_frame_with_undefined_scheme(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Reject a block whose transaction declares a dependency in a scheme
    that is not defined.

    The transaction breaks a static constraint of proof frames. No witness
    satisfies such a dependency either, so a client that checks the block's
    witnesses before it validates the transaction rejects the block for
    its witnesses instead. Either reason is accepted.
    """
    tx, sender, target = account_transaction(
        pre, fork, data=one_triple(scheme=0x11)
    )

    blockchain_test(
        pre=pre,
        post={
            sender: Account(nonce=1),
            target: Account(storage={SLOT_EXECUTED: 0}),
        },
        blocks=[
            Block(
                txs=[tx],
                exception=[
                    TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
                    BlockException.INVALID_DEPENDENCY_WITNESSES,
                ],
            )
        ],
    )


@pytest.mark.exception_test
def test_proof_frame_in_atomic_batch(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Reject a transaction whose proof frame follows a frame that opens an
    atomic batch, which would make the proof frame part of the batch.
    """
    sender = pre.fund_eoa()
    target_code = Op.SSTORE(SLOT_EXECUTED, 1) + Op.STOP
    target = pre.deploy_contract(code=target_code)
    frames: List[FrameCase] = [
        verify_frame(logs=[]),
        sender_frame(
            target=target, code=target_code, flags=FrameFlag.ATOMIC_BATCH
        ),
        proof_frame(fork, [Dependency.lean_sphincs(KEY.public_key, MESSAGE)]),
        sender_frame(target=target, code=target_code),
    ]

    state_test(
        pre=pre,
        tx=FrameTransaction.build(
            fork=fork,
            sender=sender,
            frames=frames,
            dependency_signers=[KEY],
            error=TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
        ),
        post={target: Account(storage={SLOT_EXECUTED: 0})},
    )
