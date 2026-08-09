"""
Independent re-derivations of the EIP-8272 formulas, for tests.

Every derivation here is transcribed from the prose of
[EIP-8272](https://eips.ethereum.org/EIPS/eip-8272) and uses nothing but
`keccak256`, integer arithmetic and RLP. None of them calls into the fork
under test, so an expectation built from them is derived rather than
echoed back.

The module also holds the shapes every test file reuses: the two-frame
transaction body and the storage slots its probes write to.
"""

from typing import Dict, List, Sequence

from execution_testing import (
    Account,
    Address,
    Alloc,
    Fork,
    Frame,
    Hash,
    RecentRootReference,
    keccak256,
)

from ..eip8141_frame_transactions.spec import Spec as Spec8141
from .spec import Spec

SLOT_EXECUTED = 0x01
"""Storage slot a frame's target overwrites once it runs."""

CANARY_VALUE = 0xBA5E
"""
Pre-existing value at [`SLOT_EXECUTED`][s].

A frame that never runs leaves this value in place, so an unexecuted
transaction is distinguishable from one that executed and stored zero.

[s]: ref:tests.amsterdam.eip8272_recent_roots_for_frame_transactions.helpers.SLOT_EXECUTED
"""  # noqa: E501

EXECUTED_VALUE = 0xC0DE
"""
Value a frame's target stores over [`CANARY_VALUE`][c].

[c]: ref:tests.amsterdam.eip8272_recent_roots_for_frame_transactions.helpers.CANARY_VALUE
"""  # noqa: E501

VERIFY_FRAME_GAS = 100_000
"""Gas limit of a `VERIFY` frame, at EIP-8141's ceiling for one."""

SENDER_FRAME_GAS = 5_000_000
"""
Gas limit of a `SENDER` frame.

Generous on purpose: a frame that runs out of gas fails without
invalidating the transaction, which looks exactly like a frame that
executed and wrote nothing.
"""


def verify_and_sender_frames(
    target: Address, sender_frame_gas: int = SENDER_FRAME_GAS
) -> List[Frame]:
    """
    Return the minimal frame list: a `VERIFY` frame approving execution
    and payment, and a `SENDER` frame calling `target`.
    """
    return [
        Frame(
            mode=Spec8141.MODE_VERIFY,
            flags=Spec8141.APPROVE_EXECUTION_AND_PAYMENT,
            gas_limit=VERIFY_FRAME_GAS,
        ),
        Frame(
            mode=Spec8141.MODE_SENDER,
            target=target,
            gas_limit=sender_frame_gas,
        ),
    ]


def compute_source_id(source_address: Address, salt: int) -> Hash:
    """
    Return `keccak256(source_address || salt)`, the identifier of a root
    source, over a 20-byte address and a 32-byte salt.
    """
    preimage = bytes(source_address) + salt.to_bytes(32, "big")
    assert len(preimage) == Spec.SOURCE_ID_PREIMAGE_LENGTH
    return keccak256(preimage)


def compute_entry_hash(reference: RecentRootReference) -> Hash:
    """
    Return the entry a root source commits for a reference:
    `keccak256(ENTRY_DOMAIN || source_id || uint64_be(slot) || root)`.
    """
    preimage = (
        keccak256(Spec.ENTRY_DOMAIN_PREIMAGE)
        + bytes(reference.source_id)
        + int(reference.slot).to_bytes(Spec.SLOT_ENCODING_LENGTH, "big")
        + bytes(reference.root)
    )
    assert len(preimage) == Spec.ENTRY_HASH_PREIMAGE_LENGTH
    return keccak256(preimage)


def compute_storage_key(source_id: Hash, slot: int) -> Hash:
    """
    Return the storage key under the recent root contract that holds a
    source's entry for a slot:
    `keccak256(STORAGE_DOMAIN || source_id || uint64_be(i))`, where
    `i = slot mod RECENT_ROOT_LENGTH`.
    """
    window_index = slot % Spec.RECENT_ROOT_LENGTH
    preimage = (
        keccak256(Spec.STORAGE_DOMAIN_PREIMAGE)
        + bytes(source_id)
        + window_index.to_bytes(Spec.SLOT_ENCODING_LENGTH, "big")
    )
    assert len(preimage) == Spec.STORAGE_KEY_PREIMAGE_LENGTH
    return keccak256(preimage)


def as_int(value: bytes) -> int:
    """Return the integer a big-endian byte string denotes."""
    return int.from_bytes(value, "big")


def entries_for(
    references: Sequence[RecentRootReference],
) -> Dict[int, int]:
    """
    Return the storage the recent root contract must hold for every one
    of `references` to be satisfied.

    Duplicate references collapse onto one key, exactly as the contract's
    own storage would.
    """
    return {
        as_int(
            compute_storage_key(reference.source_id, int(reference.slot))
        ): as_int(compute_entry_hash(reference))
        for reference in references
    }


def seed_entries(
    pre: Alloc,
    references: Sequence[RecentRootReference],
    extra_storage: Dict[int, int] | None = None,
) -> None:
    """
    Place the entries satisfying `references` in the pre-state of the
    recent root contract.

    Requires `pytest.mark.pre_alloc_mutable`: the contract lives at a
    fixed address the fork pre-allocates, so its storage is seeded in
    place rather than through a freshly deployed account.
    """
    storage = entries_for(references)
    if extra_storage:
        storage.update(extra_storage)
    pre[Spec.RECENT_ROOT_ADDRESS] = Account(nonce=1, storage=storage)


def rlp_encode_references(
    references: Sequence[RecentRootReference],
) -> bytes:
    """
    Return `rlp(recent_root_references)`, the encoding whose bytes the
    transaction is charged for as calldata.

    RLP is re-implemented here for the one shape this field takes — a
    list of three-element lists — so that the charged byte count is
    derived from the encoding rules rather than read back out of the
    transaction the test is about to submit.
    """
    items = [
        _encode_list(
            [
                _encode_bytes(bytes(reference.source_id)),
                _encode_uint(int(reference.slot)),
                _encode_bytes(bytes(reference.root)),
            ]
        )
        for reference in references
    ]
    return _encode_list(items)


def _encode_length(length: int, offset: int) -> bytes:
    """Return an RLP header for a payload of `length` bytes."""
    if length < 56:
        return bytes([offset + length])
    encoded = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([offset + 55 + len(encoded)]) + encoded


def _encode_bytes(data: bytes) -> bytes:
    """Return the RLP encoding of a byte string."""
    if len(data) == 1 and data[0] < 0x80:
        return data
    return _encode_length(len(data), 0x80) + data


def _encode_uint(value: int) -> bytes:
    """Return the canonical RLP encoding of a non-negative integer."""
    if value == 0:
        return b"\x80"
    return _encode_bytes(value.to_bytes((value.bit_length() + 7) // 8, "big"))


def _encode_list(items: Sequence[bytes]) -> bytes:
    """Return the RLP encoding of a list of already-encoded items."""
    payload = b"".join(items)
    return _encode_length(len(payload), 0xC0) + payload


TOKENS_PER_ZERO_BYTE = 1
"""Tokens EIP-7623 assigns a zero calldata byte."""

TOKENS_PER_NONZERO_BYTE = 4
"""
Tokens EIP-7623 assigns a non-zero calldata byte.

EIP-7976 also makes it the uniform rate of the calldata floor, where
every charged byte counts as a non-zero one whatever its value.
"""


def count_tokens(data: bytes) -> int:
    """
    Return `tokens_in(data)`: one token per zero byte and four per
    non-zero byte, as EIP-7623 defines it.
    """
    return sum(
        TOKENS_PER_ZERO_BYTE if byte == 0 else TOKENS_PER_NONZERO_BYTE
        for byte in data
    )


def reference_calldata_cost(
    fork: Fork, references: Sequence[RecentRootReference]
) -> int:
    """
    Return `recent_root_calldata_cost`: the standard token cost applied
    to the tokens of the whole encoded reference list.
    """
    tokens = count_tokens(rlp_encode_references(references))
    return tokens * fork.gas_costs().TX_DATA_TOKEN_STANDARD


def reference_gas(fork: Fork) -> int:
    """
    Return `RECENT_ROOT_REFERENCE_GAS`.

    The four fork parameters are read from the fork's own schedule
    because they are inputs to EIP-8272 rather than things it defines;
    the shape of the sum — one storage key, two Keccaks, seven words — is
    EIP-8272's and is written out here.
    """
    gas_costs = fork.gas_costs()
    return (
        gas_costs.TX_ACCESS_LIST_STORAGE_KEY
        + Spec.KECCAK_INVOCATIONS_PER_REFERENCE
        * gas_costs.OPCODE_KECCAK256_BASE
        + Spec.KECCAK_WORDS_PER_REFERENCE * gas_costs.OPCODE_KECCAK256_PER_WORD
    )


def reference_intrinsic_gas(fork: Fork, reference_count: int) -> int:
    """
    Return `recent_root_reference_intrinsic_gas`: zero without
    references, and otherwise one address charge plus one per-reference
    charge for every declared reference, duplicates included.
    """
    if reference_count == 0:
        return 0
    return (
        fork.gas_costs().TX_ACCESS_LIST_ADDRESS
        + reference_count * reference_gas(fork)
    )


def signature_free_intrinsic_gas(
    fork: Fork,
    *,
    frame_count: int,
    references: Sequence[RecentRootReference],
) -> int:
    """
    Return the standard intrinsic execution gas of a frame transaction
    that carries no signature entries and no frame data.

    EIP-8141 charges a base cost and a per-frame cost, plus the standard
    token cost of every byte field priced as calldata. With a contract
    sender there are no signature entries, and with empty frame data the
    only calldata left is EIP-8272's encoded reference list — so the
    whole quantity is computable here, and the sum is exact rather than
    a delta.
    """
    return (
        Spec8141.FRAME_TX_INTRINSIC_COST
        + frame_count * Spec8141.FRAME_TX_PER_FRAME_COST
        + reference_calldata_cost(fork, references)
        + reference_intrinsic_gas(fork, len(references))
    )


def signature_free_calldata_floor_gas(
    fork: Fork,
    *,
    frame_count: int,
    references: Sequence[RecentRootReference],
) -> int:
    """
    Return the calldata floor gas of a frame transaction that carries no
    signature entries and no frame data.

    The floor is the other branch of the intrinsic cost, and EIP-8272
    adds to it twice: `recent_root_reference_intrinsic_gas` joins the
    costs the floor is anchored on, and the bytes of the encoded
    reference list join the bytes the floor is counted over. EIP-7976
    prices those bytes uniformly — every charged byte counts as a
    non-zero one — which is why this quantity, unlike
    [`signature_free_intrinsic_gas`][sfig], is blind to the values the
    references carry and sees only their encoded length.

    [sfig]: ref:tests.amsterdam.eip8272_recent_roots_for_frame_transactions.helpers.signature_free_intrinsic_gas
    """  # noqa: E501
    gas_costs = fork.gas_costs()
    charged_bytes = len(rlp_encode_references(references))
    floor_tokens = charged_bytes * TOKENS_PER_NONZERO_BYTE
    return (
        Spec8141.FRAME_TX_INTRINSIC_COST
        + frame_count * Spec8141.FRAME_TX_PER_FRAME_COST
        + reference_intrinsic_gas(fork, len(references))
        + floor_tokens * gas_costs.TX_DATA_TOKEN_FLOOR
    )


def references_at_slots(
    source_id: Hash, slots: List[int], root: Hash
) -> List[RecentRootReference]:
    """Return one reference per slot, all naming the same source and root."""
    return [
        RecentRootReference(source_id=source_id, slot=slot, root=root)
        for slot in slots
    ]


TXPARAM_GAS = 2
"""
Gas EIP-8141 charges for `TXPARAM`, whatever index it is given.

EIP-8272 adds one index to that opcode and prices it at "the standard
`TXPARAM` gas", so this is the number its new index must cost. It is
transcribed from EIP-8141's opcode table rather than read from the
fork's schedule, which carries no entry for this opcode.
"""
