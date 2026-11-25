"""Defines EIP-8504 specification constants and functions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceSpec:
    """Defines the reference spec version and git path."""

    git_path: str
    version: str


ref_spec_8504 = ReferenceSpec(
    "EIPS/eip-8504.md", "b3de28798212aeb49f2ef86479cba1cb1358c211"
)


class Spec:
    """
    Parameters from the EIP-8504 specification as defined at
    https://eips.ethereum.org/EIPS/eip-8504.
    """

    MODE_PROOF = 4
    """Mode of a proof frame, `PROOF_FRAME_MODE`."""

    LEAN_SPHINCS_SCHEME = 0x10
    """Scheme of a dependency satisfied by a leanSPHINCS signature."""

    DEPENDENCY_LENGTH = 96
    """Length in bytes of one triple in a proof frame's data."""

    MAX_DEPENDENCIES_PER_FRAME = 256
    """Greatest number of triples in a single proof frame."""

    LEAN_SPHINCS_SIGNATURE_SIZE = 7_740
    """Length in bytes of a leanSPHINCS signature."""
