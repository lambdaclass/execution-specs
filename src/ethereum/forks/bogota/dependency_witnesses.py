"""
Dependency witnesses are what a block carries to satisfy the dependencies
its transactions declare, as introduced in [EIP-8504].

A frame transaction declares dependencies in its `PROOF` frames (see
[`dependencies`][d]). Executing such a frame checks nothing: whether a
dependency holds is decided for the block as a whole, by
[`validate_dependency_witnesses`][v], from the block's contents alone. The
check reads no state and can run before the block's transactions execute.

[EIP-8504]: https://eips.ethereum.org/EIPS/eip-8504
[d]: ref:ethereum.forks.bogota.transactions.frame_transaction.dependencies
[v]: ref:ethereum.forks.bogota.dependency_witnesses.validate_dependency_witnesses
"""  # noqa: E501

from typing import List, Sequence, Tuple

from ethereum_rlp import rlp
from ethereum_types.bytes import Bytes

from ethereum.crypto.hash import Hash32, keccak256
from ethereum.crypto.lean_sphincs import lean_sphincs_verify

from .exceptions import InvalidDependencyWitnessesError
from .transactions import Transaction
from .transactions.frame_transaction import (
    LEAN_SPHINCS_SCHEME,
    Dependency,
    FrameTransaction,
    deduplicate_and_sort,
    dependencies,
)


def block_dependencies(
    transactions: Sequence[Transaction],
) -> Tuple[Dependency, ...]:
    """
    Combine the dependencies of a block's transactions, sorted and
    deduplicated, so a dependency that several transactions declare appears
    once.
    """
    combined: List[Dependency] = []
    for tx in transactions:
        if isinstance(tx, FrameTransaction):
            combined.extend(dependencies(tx))
    return deduplicate_and_sort(combined)


def satisfies(dependency: Dependency, witness: Bytes) -> bool:
    """
    Decide whether `witness` satisfies `dependency`: whether it is a valid
    signature of the dependency's message under its public key, in the
    dependency's scheme.
    """
    if dependency.scheme != LEAN_SPHINCS_SCHEME:
        return False
    return lean_sphincs_verify(
        dependency.verification_key_hash, dependency.data_hash, witness
    )


def hash_dependency_witnesses(witnesses: Sequence[Bytes]) -> Hash32:
    """
    Compute the commitment a block header holds to the block's dependency
    witnesses.
    """
    return keccak256(rlp.encode(tuple(witnesses)))


def validate_dependency_witnesses(
    transactions: Sequence[Transaction], witnesses: Sequence[Bytes]
) -> None:
    """
    Check that `witnesses` holds exactly one witness for each of the
    [`block_dependencies`][bd] of `transactions`, in the same order, and
    that each witness [`satisfies`][s] its dependency.

    A key holder can sign one message more than once, so a dependency may
    have several valid witnesses, and any of them is accepted.

    [bd]: ref:ethereum.forks.bogota.dependency_witnesses.block_dependencies
    [s]: ref:ethereum.forks.bogota.dependency_witnesses.satisfies
    """
    required = block_dependencies(transactions)
    if len(witnesses) != len(required):
        raise InvalidDependencyWitnessesError(
            "Dependency witness count does not match the dependencies"
        )
    for dependency, witness in zip(required, witnesses, strict=True):
        if not satisfies(dependency, witness):
            raise InvalidDependencyWitnessesError(
                "Dependency witness does not satisfy its dependency"
            )
