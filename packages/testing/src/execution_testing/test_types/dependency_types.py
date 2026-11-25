"""
Dependencies of frame transactions and the witnesses that satisfy them
(EIP-8504).

A frame transaction declares dependencies in its `PROOF` frames. A block
carries one witness for each distinct dependency of its transactions, in
the order of those dependencies. The helpers here derive that order from
the transactions and produce the witnesses, so a test only has to say
which keys sign.
"""

from typing import List, NamedTuple, Sequence

from execution_testing.base_types import Bytes, Hash

from .lean_sphincs import LeanSphincsKey
from .transaction_types import Transaction

PROOF_FRAME_MODE = 4
"""Mode value of a `PROOF` frame."""

DEPENDENCY_LENGTH = 96
"""Length in bytes of one dependency triple in a `PROOF` frame's data."""

LEAN_SPHINCS_SCHEME = 0x10
"""Scheme of a dependency satisfied by a leanSPHINCS signature."""


class Dependency(NamedTuple):
    """
    A statement a block must carry a witness for: `data_hash` is signed
    under the public key `verification_key_hash`, in `scheme`.

    Dependencies order by scheme, then by `data_hash`, then by
    `verification_key_hash`, which is how tuples of them compare.
    """

    scheme: int
    data_hash: Hash
    verification_key_hash: Hash

    @classmethod
    def lean_sphincs(
        cls, public_key: bytes, message: bytes | None = None
    ) -> "Dependency":
        """
        Return the dependency on a leanSPHINCS signature under
        `public_key`, of `message` or, when none is given, of the signature
        hash of whichever transaction declares it.
        """
        return cls(
            LEAN_SPHINCS_SCHEME,
            Hash(message if message is not None else bytes(32)),
            Hash(public_key),
        )

    def encode(self) -> Bytes:
        """Return the triple as a `PROOF` frame's data carries it."""
        return Bytes(
            self.scheme.to_bytes(32, "big")
            + bytes(self.data_hash)
            + bytes(self.verification_key_hash)
        )


def transaction_dependencies(tx: Transaction) -> List[Dependency]:
    """
    Return the dependencies of a transaction: the triples of its `PROOF`
    frames, each all-zero `data_hash` replaced by the transaction's
    signature hash, sorted and deduplicated.

    The signature hash covers the signature entries the transaction ends up
    with, so the transaction must already have them.
    """
    if tx.frames is None:
        return []
    signature_hash: bytes | None = None
    declared = set()
    for frame in tx.frames:
        if int(frame.mode) != PROOF_FRAME_MODE:
            continue
        data = bytes(frame.data)
        for start in range(
            0, len(data) - DEPENDENCY_LENGTH + 1, DEPENDENCY_LENGTH
        ):
            triple = data[start : start + DEPENDENCY_LENGTH]
            data_hash = triple[32:64]
            if data_hash == bytes(32):
                if signature_hash is None:
                    signature_hash = bytes(tx.rlp_signing_bytes().keccak256())
                data_hash = signature_hash
            declared.add(
                Dependency(triple[31], Hash(data_hash), Hash(triple[64:96]))
            )
    return sorted(declared)


def block_dependencies(txs: Sequence[Transaction]) -> List[Dependency]:
    """
    Return the dependencies of a block's transactions, sorted and
    deduplicated: the order of the block's dependency witnesses.
    """
    combined = set()
    for tx in txs:
        combined.update(transaction_dependencies(tx))
    return sorted(combined)


def sign_dependencies(
    dependencies: Sequence[Dependency], signers: Sequence[LeanSphincsKey]
) -> List[Bytes]:
    """
    Return one witness for each dependency, in order, signed by whichever
    of `signers` holds the dependency's public key.
    """
    by_public_key = {bytes(key.public_key): key for key in signers}
    witnesses = []
    for dependency in dependencies:
        key = by_public_key.get(bytes(dependency.verification_key_hash))
        if key is None:
            raise ValueError(
                "no signer for the dependency on public key "
                f"{dependency.verification_key_hash}: attach its key to a "
                "transaction's `dependency_signers`, or give the block its "
                "`dependency_witnesses`"
            )
        witnesses.append(key.sign(bytes(dependency.data_hash)))
    return witnesses
