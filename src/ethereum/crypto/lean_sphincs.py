"""
leanSPHINCS is a stateless hash-based signature scheme whose only primitive
is [`keccak256`].

A public key is a public seed followed by the root of a tree of one-time
keys. A signature leads from the message to that root in three steps:

1. The message is hashed with the signature's randomizer and the public key.
   The digest selects one few-time key, and one leaf in each of that key's
   trees. The signature reveals the secret of each selected leaf with its
   authentication path, which gives the few-time key's public value.
1. A stack of one-time signatures follows, bottom layer first. Each layer
   signs the value below it: the few-time key's public value at the bottom,
   and the root of the layer below everywhere else. A layer's counter must
   encode the value it signs as a codeword, whose digits say how far each
   revealed chain value already is along its chain. Completing the chains
   gives the one-time public key, and the layer's authentication path leads
   from it to the root of the layer's tree.
1. The signature is valid when the root of the top layer is the public
   key's.

Every hash is a tweakable hash: [`keccak256`] over the public seed, an
[address] naming the hash's position in the structure, and the input,
truncated to the scheme's hash size.

Only verification is specified, in [`lean_sphincs_verify`]. Producing keys
and signatures is left to wallets.

[`keccak256`]: ref:ethereum.crypto.hash.keccak256
[address]: ref:ethereum.crypto.lean_sphincs.address
[`lean_sphincs_verify`]: ref:ethereum.crypto.lean_sphincs.lean_sphincs_verify
"""

from typing import List, Optional, Tuple

from ethereum_types.bytes import Bytes, Bytes32

from .hash import keccak256

HASH_SIZE = 16
"""
Length in bytes of every hash value in the scheme: the public seed, the
public root, and each node, secret and chain value of a signature.
"""

COUNTER_SIZE = 4
"""
Length in bytes of the counter a one-time signature carries.
"""

WOTS_CHAINS = 64
"""
Number of hash chains in a one-time key.
"""

WOTS_DIGIT_BITS = 2
"""
Number of bits in one digit of an encoding. A chain has one position for
each value a digit can take.
"""

WOTS_LAST_POSITION = (1 << WOTS_DIGIT_BITS) - 1
"""
Position of the end of a chain, which is also the greatest value of a digit.
"""

WOTS_CODEWORD_SUM = 128
"""
Sum of the digits of every codeword. An encoding with another sum is not a
codeword.
"""

LAYER_HEIGHTS = (8, 8, 10)
"""
Height of the trees of one-time keys at each layer, bottom layer first.
"""

FORS_TREES = 24
"""
Number of trees in a few-time key.
"""

FORS_HEIGHT = 10
"""
Height of each tree of a few-time key.
"""

FORS_LEAVES_SIZE = (FORS_TREES * FORS_HEIGHT + 7) // 8
"""
Length in bytes of the part of the message digest that selects one leaf in
each tree of the few-time key.
"""

KEY_INDEX_SIZE = (sum(LAYER_HEIGHTS) + 7) // 8
"""
Length in bytes of the part of the message digest that selects the few-time
key.
"""

FORS_OPENING_SIZE = HASH_SIZE * (1 + FORS_HEIGHT)
"""
Length in bytes of what a signature reveals for one tree of the few-time
key: a secret and its authentication path.
"""

SIGNATURE_SIZE = (
    HASH_SIZE
    + FORS_TREES * FORS_OPENING_SIZE
    + sum(
        COUNTER_SIZE + HASH_SIZE * (WOTS_CHAINS + height)
        for height in LAYER_HEIGHTS
    )
)
"""
Length in bytes of a signature: the randomizer, the openings of the few-time
key, and one one-time signature with its authentication path for each layer.
"""

ADDRESS_WOTS_HASH = 0
"""
[Address] type of a step along a chain of a one-time key.

[Address]: ref:ethereum.crypto.lean_sphincs.address
"""

ADDRESS_WOTS_PUBLIC_KEY = 1
"""
[Address] type of the hash that compresses the ends of a one-time key's
chains into its public key.

[Address]: ref:ethereum.crypto.lean_sphincs.address
"""

ADDRESS_TREE = 2
"""
[Address] type of a node in a tree of one-time keys.

[Address]: ref:ethereum.crypto.lean_sphincs.address
"""

ADDRESS_FORS_TREE = 3
"""
[Address] type of a leaf or a node in a tree of a few-time key.

[Address]: ref:ethereum.crypto.lean_sphincs.address
"""

ADDRESS_FORS_ROOTS = 4
"""
[Address] type of the hash that compresses the roots of a few-time key's
trees into its public value.

[Address]: ref:ethereum.crypto.lean_sphincs.address
"""

ADDRESS_WOTS_ENCODING = 7
"""
[Address] type of the hash that encodes the value a one-time key signs.

[Address]: ref:ethereum.crypto.lean_sphincs.address
"""


def address(
    layer: int,
    tree: int,
    kind: int,
    key_pair: int = 0,
    chain_or_height: int = 0,
    index: int = 0,
) -> bytes:
    """
    Encode the position of a hash within the structure of a key.

    An address names the layer and the tree a hash belongs to, what the hash
    is for, and where it sits: the one-time key within the tree, then the
    chain within the key or the height within a tree, then the position
    along the chain or the index at that height. Unused words are zero.

    Two hashes at different positions never share an address, which keeps
    their inputs apart.
    """
    return (
        layer.to_bytes(4, "big")
        + tree.to_bytes(12, "big")
        + kind.to_bytes(4, "big")
        + key_pair.to_bytes(4, "big")
        + chain_or_height.to_bytes(4, "big")
        + index.to_bytes(4, "big")
    )


def tweakable_hash(seed: bytes, tweak: bytes, value: bytes) -> bytes:
    """
    Hash `value` at the position `tweak` names, under the public `seed`.
    """
    return bytes(keccak256(seed + tweak + value)[:HASH_SIZE])


def message_digest(
    randomizer: bytes, public_key: bytes, message: bytes
) -> bytes:
    """
    Hash a message into the digest that selects what a signature reveals.

    The digest is longer than one hash output, so the same input is hashed
    under successive counters and the outputs are concatenated.
    """
    prefix = randomizer + public_key + message
    size = FORS_LEAVES_SIZE + KEY_INDEX_SIZE
    digest = b""
    counter = 0
    while len(digest) < size:
        digest += keccak256(prefix + counter.to_bytes(4, "big"))
        counter += 1
    return digest[:size]


def split_digest(digest: bytes) -> Tuple[List[int], int]:
    """
    Split a message digest into the leaf selected in each tree of the
    few-time key, and the index of that few-time key.

    Both are read most significant bit first. The key index keeps the low
    bits of its bytes, and the bits above them are ignored.
    """
    leaf_bits = int.from_bytes(digest[:FORS_LEAVES_SIZE], "big")
    unused_bits = 8 * FORS_LEAVES_SIZE - FORS_TREES * FORS_HEIGHT
    leaves = [
        (leaf_bits >> (unused_bits + FORS_HEIGHT * (FORS_TREES - 1 - tree)))
        % (1 << FORS_HEIGHT)
        for tree in range(FORS_TREES)
    ]
    key_index = int.from_bytes(digest[FORS_LEAVES_SIZE:], "big") % (
        1 << sum(LAYER_HEIGHTS)
    )
    return leaves, key_index


def one_time_key(layer: int, key_index: int) -> Tuple[int, int]:
    """
    Locate the one-time key that signs at `layer` on the way up from the
    few-time key `key_index`: the tree it belongs to, and its position among
    that tree's leaves.
    """
    leaf = key_index >> sum(LAYER_HEIGHTS[:layer])
    height = LAYER_HEIGHTS[layer]
    return leaf >> height, leaf % (1 << height)


def fors_tree_root(
    seed: bytes,
    tree: int,
    key_pair: int,
    fors_tree: int,
    leaf_index: int,
    opening: bytes,
) -> bytes:
    """
    Recompute the root of one tree of a few-time key from the secret a
    signature reveals for the selected leaf and its authentication path.

    The trees of a few-time key are numbered side by side, so a node's index
    counts across all of them.
    """
    node = tweakable_hash(
        seed,
        address(
            0,
            tree,
            ADDRESS_FORS_TREE,
            key_pair,
            0,
            (fors_tree << FORS_HEIGHT) + leaf_index,
        ),
        opening[:HASH_SIZE],
    )
    for height in range(FORS_HEIGHT):
        start = HASH_SIZE * (1 + height)
        sibling = opening[start : start + HASH_SIZE]
        if (leaf_index >> height) % 2 == 0:
            children = node + sibling
        else:
            children = sibling + node
        parent_index = (fors_tree << (FORS_HEIGHT - height - 1)) + (
            leaf_index >> (height + 1)
        )
        node = tweakable_hash(
            seed,
            address(
                0, tree, ADDRESS_FORS_TREE, key_pair, height + 1, parent_index
            ),
            children,
        )
    return node


def layer_root(
    seed: bytes,
    layer: int,
    key_index: int,
    signed: bytes,
    layer_signature: bytes,
) -> Optional[bytes]:
    """
    Recompute the root of a layer's tree from the one-time signature of
    `signed` and its authentication path.

    Return `None` when the signature's counter does not encode `signed` as a
    codeword.
    """
    tree, key_pair = one_time_key(layer, key_index)

    counter = layer_signature[:COUNTER_SIZE]
    encoding = int.from_bytes(
        tweakable_hash(
            seed,
            address(layer, tree, ADDRESS_WOTS_ENCODING, key_pair),
            signed + counter,
        ),
        "big",
    )
    digits = [
        (encoding >> (WOTS_DIGIT_BITS * (WOTS_CHAINS - 1 - chain)))
        % (1 << WOTS_DIGIT_BITS)
        for chain in range(WOTS_CHAINS)
    ]
    if sum(digits) != WOTS_CODEWORD_SUM:
        return None

    chain_ends = b""
    for chain, digit in enumerate(digits):
        start = COUNTER_SIZE + HASH_SIZE * chain
        value = layer_signature[start : start + HASH_SIZE]
        for position in range(digit, WOTS_LAST_POSITION):
            value = tweakable_hash(
                seed,
                address(
                    layer, tree, ADDRESS_WOTS_HASH, key_pair, chain, position
                ),
                value,
            )
        chain_ends += value

    node = tweakable_hash(
        seed,
        address(layer, tree, ADDRESS_WOTS_PUBLIC_KEY, key_pair),
        chain_ends,
    )
    path = layer_signature[COUNTER_SIZE + HASH_SIZE * WOTS_CHAINS :]
    for height in range(LAYER_HEIGHTS[layer]):
        sibling = path[HASH_SIZE * height : HASH_SIZE * (height + 1)]
        if (key_pair >> height) % 2 == 0:
            children = node + sibling
        else:
            children = sibling + node
        node = tweakable_hash(
            seed,
            address(
                layer,
                tree,
                ADDRESS_TREE,
                0,
                height + 1,
                key_pair >> (height + 1),
            ),
            children,
        )
    return node


def lean_sphincs_verify(
    public_key: Bytes32, message: Bytes32, signature: Bytes
) -> bool:
    """
    Check a leanSPHINCS signature of `message` under `public_key`.

    A signature of any other length than [`SIGNATURE_SIZE`] is not valid.
    Apart from its length, a signature is rejected only when a layer's
    counter does not give a codeword, or when the recomputed root is not the
    public key's. In particular, nothing requires a counter to be the least
    one that works, so a key holder can produce several valid signatures of
    one message.

    [`SIGNATURE_SIZE`]: ref:ethereum.crypto.lean_sphincs.SIGNATURE_SIZE
    """
    if len(signature) != SIGNATURE_SIZE:
        return False

    seed = public_key[:HASH_SIZE]
    root = public_key[HASH_SIZE:]

    randomizer = signature[:HASH_SIZE]
    position = HASH_SIZE

    digest = message_digest(randomizer, public_key, message)
    leaves, key_index = split_digest(digest)

    # The few-time key is addressed like the bottom layer's one-time key.
    tree, key_pair = one_time_key(0, key_index)
    fors_roots = b""
    for fors_tree, leaf_index in enumerate(leaves):
        opening = signature[position : position + FORS_OPENING_SIZE]
        position += FORS_OPENING_SIZE
        fors_roots += fors_tree_root(
            seed, tree, key_pair, fors_tree, leaf_index, opening
        )
    signed = tweakable_hash(
        seed, address(0, tree, ADDRESS_FORS_ROOTS, key_pair), fors_roots
    )

    for layer, height in enumerate(LAYER_HEIGHTS):
        size = COUNTER_SIZE + HASH_SIZE * (WOTS_CHAINS + height)
        layer_signature = signature[position : position + size]
        position += size
        recomputed = layer_root(
            seed, layer, key_index, signed, layer_signature
        )
        if recomputed is None:
            return False
        signed = recomputed

    return signed == root
