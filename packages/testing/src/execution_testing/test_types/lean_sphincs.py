"""
leanSPHINCS key generation and signing, for tests that need witnesses.

The execution specification only verifies leanSPHINCS signatures
(`ethereum.crypto.lean_sphincs`). Tests also have to produce them, so this
module implements the other half of the scheme on top of the
specification's own primitives: the address encoding, the tweakable hash,
the message digest and the parameters.

Signing is deterministic: a seed and a message always give the same
signature, which keeps fixtures stable. It is also slow for Python, a
fraction of a second per signature, most of it spent rebuilding trees. The
tree at the top of a key is the same for every signature, so it is built
once per key, and each signature is computed once per process.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

from ethereum.crypto.hash import keccak256
from ethereum.crypto.lean_sphincs import (
    ADDRESS_FORS_ROOTS,
    ADDRESS_FORS_TREE,
    ADDRESS_TREE,
    ADDRESS_WOTS_ENCODING,
    ADDRESS_WOTS_HASH,
    ADDRESS_WOTS_PUBLIC_KEY,
    COUNTER_SIZE,
    FORS_HEIGHT,
    HASH_SIZE,
    LAYER_HEIGHTS,
    SIGNATURE_SIZE,
    WOTS_CHAINS,
    WOTS_CODEWORD_SUM,
    WOTS_DIGIT_BITS,
    WOTS_LAST_POSITION,
    address,
    message_digest,
    one_time_key,
    split_digest,
    tweakable_hash,
)

from execution_testing.base_types import Bytes, Hash

SEED_SIZE = 32
"""Length in bytes of the secret seed a key is derived from."""

MESSAGE_SIZE = 32
"""Length in bytes of a signed message."""

DOMAIN = b"leanSPHINCS"
"""Tag that separates the scheme's key derivations from other uses."""

ADDRESS_WOTS_SECRET = 5
"""Address type of the secret a chain of a one-time key starts from."""

ADDRESS_FORS_SECRET = 6
"""Address type of the secret under a leaf of a few-time key."""

MerkleLevels = Tuple[Tuple[bytes, ...], ...]
"""Every level of a Merkle tree, leaves first and the root alone last."""


def _merkle_levels(
    seed: bytes,
    layer: int,
    tree: int,
    kind: int,
    key_pair: int,
    first_index: int,
    leaves: List[bytes],
) -> MerkleLevels:
    """
    Build a Merkle tree over `leaves`.

    `first_index` is the index of the tree's first leaf among all the
    leaves that share its address space, which is not zero for the trees
    of a few-time key: they are numbered side by side.
    """
    levels = [tuple(leaves)]
    height = 0
    while len(levels[-1]) > 1:
        below = levels[-1]
        height += 1
        first_index >>= 1
        levels.append(
            tuple(
                tweakable_hash(
                    seed,
                    address(
                        layer, tree, kind, key_pair, height, first_index + j
                    ),
                    below[2 * j] + below[2 * j + 1],
                )
                for j in range(len(below) // 2)
            )
        )
    return tuple(levels)


def _authentication_path(levels: MerkleLevels, leaf_index: int) -> bytes:
    """Return the siblings on the way from a leaf up to the root."""
    return b"".join(
        level[(leaf_index >> height) ^ 1]
        for height, level in enumerate(levels[:-1])
    )


def _wots_secret(
    seed: bytes, secret_seed: bytes, layer: int, tree: int, key_pair: int
) -> List[bytes]:
    """Derive the start of every chain of a one-time key."""
    return [
        tweakable_hash(
            seed,
            address(layer, tree, ADDRESS_WOTS_SECRET, key_pair, chain),
            secret_seed,
        )
        for chain in range(WOTS_CHAINS)
    ]


def _chain(
    seed: bytes,
    layer: int,
    tree: int,
    key_pair: int,
    chain: int,
    value: bytes,
    start: int,
    end: int,
) -> bytes:
    """Advance a chain value from position `start` to position `end`."""
    for position in range(start, end):
        value = tweakable_hash(
            seed,
            address(layer, tree, ADDRESS_WOTS_HASH, key_pair, chain, position),
            value,
        )
    return value


def _wots_public_key(
    seed: bytes, secret_seed: bytes, layer: int, tree: int, key_pair: int
) -> bytes:
    """Compute a one-time public key: the hash of the ends of its chains."""
    ends = b"".join(
        _chain(
            seed, layer, tree, key_pair, chain, secret, 0, WOTS_LAST_POSITION
        )
        for chain, secret in enumerate(
            _wots_secret(seed, secret_seed, layer, tree, key_pair)
        )
    )
    return tweakable_hash(
        seed, address(layer, tree, ADDRESS_WOTS_PUBLIC_KEY, key_pair), ends
    )


def _layer_tree(
    seed: bytes, secret_seed: bytes, layer: int, tree: int
) -> MerkleLevels:
    """Build one tree of one-time keys."""
    leaves = [
        _wots_public_key(seed, secret_seed, layer, tree, key_pair)
        for key_pair in range(1 << LAYER_HEIGHTS[layer])
    ]
    return _merkle_levels(seed, layer, tree, ADDRESS_TREE, 0, 0, leaves)


def _digits(encoding: bytes) -> List[int]:
    """Split an encoding into the digits that say how far each chain goes."""
    value = int.from_bytes(encoding, "big")
    return [
        (value >> (WOTS_DIGIT_BITS * (WOTS_CHAINS - 1 - chain)))
        % (1 << WOTS_DIGIT_BITS)
        for chain in range(WOTS_CHAINS)
    ]


def _wots_sign(
    seed: bytes,
    secret_seed: bytes,
    layer: int,
    tree: int,
    key_pair: int,
    signed: bytes,
) -> bytes:
    """
    Sign `signed` with a one-time key: the least counter that encodes it as
    a codeword, then each chain advanced by its digit.
    """
    counter = 0
    while True:
        counter_bytes = counter.to_bytes(COUNTER_SIZE, "big")
        digits = _digits(
            tweakable_hash(
                seed,
                address(layer, tree, ADDRESS_WOTS_ENCODING, key_pair),
                signed + counter_bytes,
            )
        )
        if sum(digits) == WOTS_CODEWORD_SUM:
            break
        counter += 1
    secrets = _wots_secret(seed, secret_seed, layer, tree, key_pair)
    return counter_bytes + b"".join(
        _chain(seed, layer, tree, key_pair, chain, secrets[chain], 0, digit)
        for chain, digit in enumerate(digits)
    )


@lru_cache(maxsize=None)
def _expand(secret_seed: bytes) -> Tuple[bytes, bytes, MerkleLevels]:
    """
    Derive what every signature of a key needs: the public seed, the secret
    that randomizes messages, and the top tree, whose root completes the
    public key.
    """
    seed = bytes(keccak256(DOMAIN + secret_seed + b"\x00")[:HASH_SIZE])
    message_secret = bytes(keccak256(DOMAIN + secret_seed + b"\x01"))
    top_layer = len(LAYER_HEIGHTS) - 1
    return seed, message_secret, _layer_tree(seed, secret_seed, top_layer, 0)


@lru_cache(maxsize=None)
def _sign(secret_seed: bytes, message: bytes) -> bytes:
    """Sign a message. See `LeanSphincsKey.sign`."""
    seed, message_secret, top_tree = _expand(secret_seed)
    public_key = seed + top_tree[-1][0]

    randomizer = bytes(
        keccak256(
            message_secret
            + seed
            + b"\x00"
            + len(DOMAIN).to_bytes(1, "big")
            + DOMAIN
            + message
        )[:HASH_SIZE]
    )
    leaves, key_index = split_digest(
        message_digest(randomizer, public_key, message)
    )

    # The few-time key, addressed like the bottom layer's one-time key.
    tree, key_pair = one_time_key(0, key_index)
    signature = randomizer
    roots = b""
    for fors_tree, leaf_index in enumerate(leaves):
        first_leaf = fors_tree << FORS_HEIGHT
        secrets = [
            tweakable_hash(
                seed,
                address(
                    0, tree, ADDRESS_FORS_SECRET, key_pair, 0, first_leaf + i
                ),
                secret_seed,
            )
            for i in range(1 << FORS_HEIGHT)
        ]
        levels = _merkle_levels(
            seed,
            0,
            tree,
            ADDRESS_FORS_TREE,
            key_pair,
            first_leaf,
            [
                tweakable_hash(
                    seed,
                    address(
                        0, tree, ADDRESS_FORS_TREE, key_pair, 0, first_leaf + i
                    ),
                    secret,
                )
                for i, secret in enumerate(secrets)
            ],
        )
        signature += secrets[leaf_index]
        signature += _authentication_path(levels, leaf_index)
        roots += levels[-1][0]
    signed = tweakable_hash(
        seed, address(0, tree, ADDRESS_FORS_ROOTS, key_pair), roots
    )

    top_layer = len(LAYER_HEIGHTS) - 1
    for layer in range(len(LAYER_HEIGHTS)):
        tree, key_pair = one_time_key(layer, key_index)
        signature += _wots_sign(
            seed, secret_seed, layer, tree, key_pair, signed
        )
        levels = (
            top_tree
            if layer == top_layer
            else _layer_tree(seed, secret_seed, layer, tree)
        )
        signature += _authentication_path(levels, key_pair)
        signed = levels[-1][0]

    assert len(signature) == SIGNATURE_SIZE
    return signature


@dataclass(frozen=True)
class LeanSphincsKey:
    """
    A leanSPHINCS key pair, derived from a secret seed.

    The same seed always gives the same public key and the same signatures,
    so a test names its keys by their seeds.
    """

    seed: bytes
    """The secret seed, `SEED_SIZE` bytes long."""

    def __post_init__(self) -> None:
        """Reject a seed of the wrong length."""
        if len(self.seed) != SEED_SIZE:
            raise ValueError(
                f"a leanSPHINCS seed is {SEED_SIZE} bytes long, "
                f"got {len(self.seed)}"
            )

    @classmethod
    def from_index(cls, index: int) -> "LeanSphincsKey":
        """Return one of an unlimited series of distinct, fixed keys."""
        return cls(index.to_bytes(SEED_SIZE, "big"))

    @property
    def public_key(self) -> Hash:
        """
        Return the public key: the public seed followed by the root of the
        key's top tree. It is what a dependency names as its
        `verification_key_hash`.
        """
        seed, _, top_tree = _expand(bytes(self.seed))
        return Hash(seed + top_tree[-1][0])

    def sign(self, message: bytes) -> Bytes:
        """
        Return this key's signature of `message`, the one the scheme's
        reference signer gives: it uses the least counter that works at
        every layer.
        """
        if len(message) != MESSAGE_SIZE:
            raise ValueError(
                f"a leanSPHINCS message is {MESSAGE_SIZE} bytes long, "
                f"got {len(message)}"
            )
        return Bytes(_sign(bytes(self.seed), bytes(message)))
