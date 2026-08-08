"""Hard-coded vectors for EIP-8250 keyed nonces."""

import pytest
from execution_testing import (
    EOA,
    Account,
    Address,
    Alloc,
    Environment,
    Frame,
    Hash,
    StateTestFiller,
    Transaction,
)

from ethereum.forks.amsterdam.transactions.frame_transaction import (
    KEYED_NONCE_FIRST_USE_GAS,
    MAX_NONCE_KEYS,
    MAX_NONCE_SEQ,
    NONCE_MANAGER,
    NONCE_MANAGER_CODE,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")


def fill_valid_legacy_key_control(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Fill a minimal valid keyed-nonce transaction for vector tests."""
    sender = pre.fund_eoa()
    state_test(
        env=Environment(),
        pre=pre,
        tx=Transaction(
            sender=sender,
            nonce_keys=[0],
            nonce_seq=0,
            frames=[
                Frame(
                    mode=Spec.MODE_VERIFY,
                    flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                    gas_limit=0,
                )
            ],
        ),
        post={sender: Account(nonce=1)},
    )


def test_pinned_constant_table(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-002 through R-006.

    Expected literals are copied from the pinned EIP constant table and are
    compared to the production implementation constants, so changing the
    implementation (rather than this test's reference helper) fails the test.
    """
    assert NONCE_MANAGER == bytes(Address(0x8250))
    assert NONCE_MANAGER_CODE == bytes.fromhex("60006000fd")
    assert KEYED_NONCE_FIRST_USE_GAS == 20_000
    assert MAX_NONCE_SEQ == 18_446_744_073_709_551_615
    assert MAX_NONCE_KEYS == 16

    fill_valid_legacy_key_control(state_test, pre)


def test_payload_exact_rlp_and_signature_vector(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-017--R-028, R-146, R-148, and R-181.

    The expected payload was hand-assembled from canonical RLP and its signing
    hash was independently fixed from Keccak(type-byte || payload).
    """
    sender = EOA(bytes.fromhex("11" * 20))
    tx = Transaction(
        chain_id=1,
        sender=sender,
        nonce_keys=[0],
        nonce_seq=0,
        frames=[Frame()],
        signatures=[],
        max_priority_fee_per_gas=0,
        max_fee_per_gas=0,
        max_fee_per_blob_gas=0,
        blob_versioned_hashes=[],
    )
    payload = bytes.fromhex(
        "e601c18080941111111111111111111111111111111111111111"
        "c7c6808080808080c0808080c0"
    )
    assert bytes(tx.rlp()) == bytes([Spec.FRAME_TX_TYPE]) + payload
    assert tx.rlp_signing_bytes().keccak256() == Hash(
        "0xbad93494571e032e57588470924fb7025de96857bf5edc713f82f9297be4b1f0"
    )

    fill_valid_legacy_key_control(state_test, pre)


@pytest.mark.parametrize(
    "nonce_keys,nonce_seq,encoded,tokens,standard_gas,floor_gas",
    [
        pytest.param([0], 0, "c18080", 12, 48, 120, id="zero_key"),
        pytest.param(
            [256], 256, "c3820100820100", 22, 88, 220, id="zero_bytes"
        ),
        pytest.param([1, 2], 0, "c2010280", 16, 64, 160, id="two_keys"),
    ],
)
def test_nonce_calldata_pricing_vectors(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_keys: list[int],
    nonce_seq: int,
    encoded: str,
    tokens: int,
    standard_gas: int,
    floor_gas: int,
) -> None:
    """
    Pin the `nonce_calldata` encoding vectors of R-029 and R-030.

    Each byte string is hand-encoded RLP, and the token counts beside it are
    independently counted as zeros + 4*nonzeros and 4*zeros + 16*nonzeros.

    These assertions are fill-time identities over hand-written vectors: they
    document the encoding, and nothing here reaches a client. The pricing
    rules themselves -- that `nonce_calldata_cost` is added to
    `standard_gas_limit` and `nonce_calldata_tokens` to `calldata_tokens` --
    are pinned against a client in `test_calldata_pricing.py`, which asserts
    receipt gas. The `floor_gas` column below is EIP-7623's `10 * tokens`;
    Amsterdam reprices that weight under EIP-7976, so it is documentation of
    the referenced formula rather than an expectation for this fork.
    """

    def rlp_integer(value: int) -> bytes:
        if value == 0:
            return b"\x80"
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        if len(raw) == 1 and raw[0] < 0x80:
            return raw
        return bytes([0x80 + len(raw)]) + raw

    key_items = b"".join(rlp_integer(key) for key in nonce_keys)
    nonce_calldata = (
        bytes([0xC0 + len(key_items)]) + key_items + rlp_integer(nonce_seq)
    )
    zero_count = nonce_calldata.count(0)
    nonzero_count = len(nonce_calldata) - zero_count

    assert nonce_calldata.hex() == encoded
    assert zero_count + 4 * nonzero_count == tokens
    assert 4 * zero_count + 16 * nonzero_count == standard_gas
    assert 10 * tokens == floor_gas

    fill_valid_legacy_key_control(state_test, pre)


def test_storage_slot_hardcoded_vectors(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-051--R-054, R-056, R-150, and R-151.

    Expected slots are hard-coded Keccak results over the explicitly built
    64-byte left-padded sender and big-endian key preimages.
    """
    sender = Address(0x1234)
    expected = {
        1: Hash(
            "0x63b939821a5be8a0d41f2b7a5fc118fa09d99968c6010c590a5daf26b37cd05d"
        ),
        2: Hash(
            "0xc49c6fd786de9890f7bec89204b39c495593ac32e33447545124940bf5f83bbd"
        ),
    }
    assert Spec.nonce_slot(sender, 1) == expected[1]
    assert Spec.nonce_slot(sender, 2) == expected[2]
    assert expected[1] != expected[2]

    fill_valid_legacy_key_control(state_test, pre)


@pytest.mark.parametrize(
    "nonce_key",
    [
        pytest.param(2**192 - 1, id="largest_erc4337_key"),
        pytest.param(2**192, id="first_wider_key"),
        pytest.param(2**256 - 1, id="largest_eip8250_key"),
    ],
)
def test_erc4337_key_width_subset_vector(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_key: int,
) -> None:
    """
    Pin R-021, R-146, and R-147.

    Fixed-width byte conversion proves the 24-byte maximum has eight leading
    zero bytes, while arithmetic shows wider EIP-8250 uint256 keys remain
    valid.
    """
    largest_4337_key = 2**192 - 1
    assert largest_4337_key.to_bytes(32, "big") == b"\x00" * 8 + b"\xff" * 24
    assert largest_4337_key < 2**192 < 2**256 - 1 < 2**256

    sender = pre.fund_eoa()
    state_test(
        env=Environment(),
        pre=pre,
        tx=Transaction(
            sender=sender,
            nonce_keys=[nonce_key],
            nonce_seq=0,
            frames=[
                Frame(
                    mode=Spec.MODE_VERIFY,
                    flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                    gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS,
                )
            ],
        ),
        post={
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, nonce_key): 1}
            )
        },
    )


def test_signature_commits_entire_visible_keyset(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-027, R-028, R-115, R-161, and R-162.

    The two expected hashes are hard-coded vectors; only the second numeric key
    differs, so unequal hashes independently demonstrate whole-keyset binding.
    """
    sender = EOA(bytes.fromhex("11" * 20))

    def signing_hash(nonce_keys: list[int]) -> Hash:
        tx = Transaction(
            chain_id=1,
            sender=sender,
            nonce_keys=nonce_keys,
            nonce_seq=0,
            frames=[Frame()],
            signatures=[],
            max_priority_fee_per_gas=0,
            max_fee_per_gas=0,
            max_fee_per_blob_gas=0,
            blob_versioned_hashes=[],
        )
        return tx.rlp_signing_bytes().keccak256()

    assert signing_hash([1, 2]) == Hash(
        "0x44873c81c58b57abb7aac782065d6712fcd41babaf39629534ceb2d9526101f1"
    )
    assert signing_hash([1, 3]) == Hash(
        "0x9a352da63b34d10d963a33d04cd9831601de76e38dd1c5632977b1b563b68da2"
    )

    fill_valid_legacy_key_control(state_test, pre)
