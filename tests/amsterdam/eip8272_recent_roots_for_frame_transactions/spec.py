"""Defines EIP-8272 specification constants and reference vectors."""

from dataclasses import dataclass

from execution_testing import Address, Hash


@dataclass(frozen=True)
class ReferenceSpec:
    """Defines the reference spec version and git path."""

    git_path: str
    version: str


ref_spec_8272 = ReferenceSpec(
    "EIPS/eip-8272.md", "1ff890ed8ce03be2cf25f7528c78733d35a3dfe3"
)


@dataclass(frozen=True)
class Spec:
    """
    Parameters from the EIP-8272 specification as defined at
    https://eips.ethereum.org/EIPS/eip-8272.
    """

    RECENT_ROOT_ADDRESS = Address(0x8272)
    RECENT_ROOT_LENGTH = 8192
    RECENT_ROOT_USABLE_WINDOW = 8191
    MAX_RECENT_ROOT_REFERENCES = 16

    ENTRY_DOMAIN_PREIMAGE = b"RECENT_ROOT_ENTRY"
    STORAGE_DOMAIN_PREIMAGE = b"RECENT_ROOT_STORAGE"

    TXPARAM_RECENT_ROOT_REFERENCE_COUNT = 0x0F
    RECENTROOTREFLOAD = 0xB5
    RECENTROOTREFLOAD_GAS = 3

    # `RECENTROOTREFLOAD` field selectors.
    FIELD_SOURCE_ID = 0x00
    FIELD_SLOT = 0x01
    FIELD_ROOT = 0x02
    FIRST_UNDEFINED_FIELD = 0x03

    # Lengths, in bytes, of the three fixed-length concatenations the
    # specification hashes. They are asserted before every hash so a
    # variable-length or little-endian encoding fails on the length
    # rather than silently producing a different digest.
    SOURCE_ID_PREIMAGE_LENGTH = 52
    ENTRY_HASH_PREIMAGE_LENGTH = 104
    STORAGE_KEY_PREIMAGE_LENGTH = 72

    # `RECENT_ROOT_REFERENCE_GAS` is one access-list storage key plus two
    # Keccak invocations, whose preimages above occupy four and three
    # 32-byte words respectively.
    KECCAK_INVOCATIONS_PER_REFERENCE = 2
    KECCAK_WORDS_PER_REFERENCE = 7

    # Slot and window index are unsigned 64-bit big-endian integers.
    SLOT_ENCODING_LENGTH = 8
    MAX_SLOT = 2**64 - 1


@dataclass(frozen=True)
class CanonicalVector:
    """
    A hand-computed reference vector for the two derivations of EIP-8272,
    committed here as literals.

    The values were produced offline from the concatenation rules in the
    "Entry and storage keys" section, and are re-derived independently by
    `helpers.py` at test time. Two paths that must agree turn a helper
    bug into a failure rather than into a self-consistent wrong answer.

    `SOURCE_ADDRESS` is an input to `keccak256(source_address || salt)`
    and never has to name an account: nothing in the reference check
    reads it.
    """

    SOURCE_ADDRESS = Address(0xAA)
    SALT = 1
    SLOT = 8
    ROOT = Hash(b"\x42" * 32)

    ENTRY_DOMAIN = Hash(
        0x8F42481679C8E6FEFA040974B3C905E0CE3F2E464BA93ACDB074A41181617EFC
    )
    STORAGE_DOMAIN = Hash(
        0xBDC897DA2177D260FF5F4BE5D4B2AAD43F89C3347A305B584FA5A2546D053DAA
    )
    SOURCE_ID = Hash(
        0x1D9CC831D43CEBD5F9A4D865649395054531AC35AE2D9F2B4833375D7E5A53F5
    )
    ENTRY_HASH = Hash(
        0xEAE24C43789444BCA2266D5353294AF90C11FCE216AB963C2F4DEA78517C8957
    )
    STORAGE_KEY = Hash(
        0x88DA6D80A157849F5CAFCD6937A271A70CF54DB581409B1D9D3C08AF2142F6F1
    )
