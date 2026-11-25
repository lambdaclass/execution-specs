"""
Test the leanSPHINCS signer against the vectors of the scheme's
specification, and the dependency helpers built on it.

The vectors are the key generation and signing cases of the
`Cryptography/Sphincs` module of the `ethereum/EthLib` repository, at the
commit the vector file names. A case with no output is one the module
rejects for the length of its input.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from ethereum.crypto.lean_sphincs import lean_sphincs_verify
from ethereum_types.bytes import Bytes as SpecBytes
from ethereum_types.bytes import Bytes32

from execution_testing.base_types import Address, Bytes, Hash

from ..dependency_types import (
    DEPENDENCY_LENGTH,
    LEAN_SPHINCS_SCHEME,
    PROOF_FRAME_MODE,
    Dependency,
    block_dependencies,
    sign_dependencies,
    transaction_dependencies,
)
from ..lean_sphincs import LeanSphincsKey
from ..transaction_types import Frame, Transaction

VECTORS = json.loads(
    (Path(__file__).parent / "lean_sphincs_vectors.json").read_text()
)

KEY = LeanSphincsKey.from_index(1)
OTHER_KEY = LeanSphincsKey.from_index(2)
MESSAGE = bytes(range(32))


def from_hex(value: str) -> bytes:
    """Decode a `0x`-prefixed hex string."""
    return bytes.fromhex(value[2:])


@pytest.mark.parametrize(
    "case", VECTORS["key_gen"], ids=lambda case: case["name"]
)
def test_key_generation_vector(case: Dict[str, Any]) -> None:
    """Derive the public key the specification derives from a seed."""
    seed = from_hex(case["seed"])
    if case["public_key"] is None:
        with pytest.raises(ValueError):
            LeanSphincsKey(seed)
        return
    assert LeanSphincsKey(seed).public_key == Hash(case["public_key"])


@pytest.mark.parametrize(
    "case", VECTORS["sign"], ids=lambda case: case["name"]
)
def test_signing_vector(case: Dict[str, Any]) -> None:
    """Produce, byte for byte, the signature the specification produces."""
    seed, message = from_hex(case["seed"]), from_hex(case["message"])
    if case["signature"] is None:
        with pytest.raises(ValueError):
            LeanSphincsKey(seed).sign(message)
        return
    key = LeanSphincsKey(seed)
    signature = key.sign(message)
    assert signature == Bytes(case["signature"])
    assert lean_sphincs_verify(
        Bytes32(key.public_key), Bytes32(message), SpecBytes(signature)
    )


def proof_frame(*dependencies: Dependency) -> Frame:
    """Return a frame of the proof mode declaring `dependencies`."""
    return Frame(
        mode=PROOF_FRAME_MODE,
        data=b"".join(dependency.encode() for dependency in dependencies),
    )


def transaction(*frames: Frame) -> Transaction:
    """Return a frame transaction with no signature entries."""
    return Transaction(
        sender=Address(0x1000), frames=list(frames), signatures=[]
    )


def test_dependency_encoding() -> None:
    """Encode a triple as a right-aligned scheme, a message and a key."""
    dependency = Dependency.lean_sphincs(KEY.public_key, MESSAGE)
    encoded = dependency.encode()
    assert len(encoded) == DEPENDENCY_LENGTH
    assert encoded[:32] == LEAN_SPHINCS_SCHEME.to_bytes(32, "big")
    assert encoded[32:64] == MESSAGE
    assert encoded[64:] == KEY.public_key
    # With no message, the triple asks for the signature hash.
    placeholder = Dependency.lean_sphincs(KEY.public_key)
    assert placeholder.data_hash == Hash(0)


def test_transaction_dependencies_are_sorted_and_deduplicated() -> None:
    """
    Derive a transaction's dependencies from all its proof frames: the
    placeholder becomes the signature hash, repeats collapse, and the
    result is ordered by message and then by key.
    """
    own = Dependency.lean_sphincs(KEY.public_key)
    explicit = Dependency.lean_sphincs(KEY.public_key, MESSAGE)
    other = Dependency.lean_sphincs(OTHER_KEY.public_key, MESSAGE)
    tx = transaction(
        proof_frame(explicit, own, explicit),
        Frame(mode=0, data=explicit.encode()),
        proof_frame(other, own),
    )
    signature_hash = tx.rlp_signing_bytes().keccak256()

    dependencies = transaction_dependencies(tx)

    expected = sorted(
        [
            Dependency.lean_sphincs(KEY.public_key, signature_hash),
            explicit,
            other,
        ]
    )
    assert dependencies == expected
    assert transaction_dependencies(transaction(Frame(mode=0))) == []


def test_block_dependencies_merge_those_of_its_transactions() -> None:
    """
    Combine the dependencies of several transactions: a shared explicit
    dependency appears once, and the two signature hashes differ.
    """
    explicit = Dependency.lean_sphincs(KEY.public_key, MESSAGE)
    own = Dependency.lean_sphincs(KEY.public_key)
    first = transaction(proof_frame(explicit, own))
    second = transaction(proof_frame(explicit, own), Frame(mode=0))

    dependencies = block_dependencies([first, second])

    assert len(dependencies) == 3
    assert dependencies == sorted(
        set(transaction_dependencies(first))
        | set(transaction_dependencies(second))
    )


def test_sign_dependencies_uses_the_key_each_dependency_names() -> None:
    """
    Sign each dependency with the key that holds its public key, in
    order, and refuse a dependency no given key can sign.
    """
    dependencies = sorted(
        [
            Dependency.lean_sphincs(KEY.public_key, MESSAGE),
            Dependency.lean_sphincs(OTHER_KEY.public_key, MESSAGE),
        ]
    )

    witnesses = sign_dependencies(dependencies, [OTHER_KEY, KEY])

    for dependency, witness in zip(dependencies, witnesses, strict=True):
        assert lean_sphincs_verify(
            Bytes32(dependency.verification_key_hash),
            Bytes32(dependency.data_hash),
            SpecBytes(witness),
        )
    with pytest.raises(ValueError, match="no signer"):
        sign_dependencies(dependencies, [KEY])
