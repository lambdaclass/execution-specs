"""Defines EIP-8250 constants and reference helpers."""

from dataclasses import dataclass

from execution_testing import Address, Hash, keccak256


@dataclass(frozen=True)
class ReferenceSpec:
    """Reference EIP path and pinned commit."""

    git_path: str
    version: str


ref_spec_8250 = ReferenceSpec(
    "EIPS/eip-8250.md", "c9d962f194b9b167e045b3b68a7a292cdc4cec7f"
)


@dataclass(frozen=True)
class Spec:
    """EIP-8250 constants used by the tests."""

    FRAME_TX_TYPE = 0x06
    NONCE_MANAGER = Address(0x8250)
    NONCE_MANAGER_CODE = bytes.fromhex("60006000fd")
    KEYED_NONCE_FIRST_USE_GAS = 20_000
    MAX_NONCE_SEQ = 2**64 - 1
    MAX_NONCE_KEYS = 16

    # Inherited EIP-8141 and EIP-7623 pricing constants. EIP-8250 adds
    # `nonce_calldata` as one more priced input to these formulas without
    # changing any of them, so the keyed-nonce gas expectations are derived
    # from the constants rather than from any client's arithmetic.
    FRAME_TX_INTRINSIC_COST = 15_000
    FRAME_TX_PER_FRAME_COST = 475
    SIGNATURE_GAS_SECP256K1 = 2_800
    STANDARD_TOKEN_COST = 4

    MODE_DEFAULT = 0
    MODE_VERIFY = 1
    MODE_SENDER = 2

    SCHEME_SECP256K1 = 0x1

    APPROVE_NONE = 0x0
    APPROVE_PAYMENT = 0x1
    APPROVE_EXECUTION = 0x2
    APPROVE_EXECUTION_AND_PAYMENT = 0x3
    ATOMIC_BATCH_FLAG = 0x4

    STATUS_FAILURE = 0
    STATUS_SUCCESS = 1
    STATUS_SKIPPED = 2

    TXPARAM_NONCE_SEQ = 0x01
    # Highest index EIP-8141 assigns. EIP-8250 must not disturb it, so the
    # tests read it back alongside the indices this EIP adds.
    TXPARAM_SIGNATURE_COUNT = 0x0B
    TXPARAM_LEGACY_NONCE = 0x0C
    TXPARAM_NONCE_KEY_COUNT = 0x0D
    TXPARAM_NONCE_KEYS_HASH = 0x0E
    TXPARAM_NONCE_KEY_0 = 0x10

    @staticmethod
    def nonce_slot(sender: Address, nonce_key: int) -> Hash:
        """Return ``keccak256(left_pad_32(sender) || be32(key))``."""
        return keccak256(
            bytes(sender).rjust(32, b"\x00")
            + nonce_key.to_bytes(32, byteorder="big")
        )

    @staticmethod
    def nonce_keys_hash(nonce_keys: list[int]) -> Hash:
        """Return the fixed-width hash exposed by TXPARAM 0x0E."""
        encoded = len(nonce_keys).to_bytes(32, byteorder="big")
        encoded += b"".join(
            key.to_bytes(32, byteorder="big") for key in nonce_keys
        )
        return keccak256(encoded)
