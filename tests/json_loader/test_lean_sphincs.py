"""
Test leanSPHINCS verification against the vectors of its specification.

The vectors are the verification cases of the `Cryptography/Sphincs` module
of the `ethereum/EthLib` repository, at the commit the vector file names.
Two of that module's cases are left out, because they give a public key or
a message that is not 32 bytes long, which `lean_sphincs_verify` cannot be
called with.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from ethereum_types.bytes import Bytes, Bytes32

from ethereum.crypto.lean_sphincs import SIGNATURE_SIZE, lean_sphincs_verify
from ethereum.utils.hexadecimal import hex_to_bytes

VECTORS = json.loads(
    (Path(__file__).parent / "lean_sphincs_verify_vectors.json").read_text()
)


@pytest.mark.parametrize(
    "case", VECTORS["cases"], ids=lambda case: case["name"]
)
def test_verify_vector(case: Dict[str, Any]) -> None:
    """Accept exactly the signatures the specification's vectors accept."""
    valid = lean_sphincs_verify(
        Bytes32(hex_to_bytes(case["public_key"])),
        Bytes32(hex_to_bytes(case["message"])),
        Bytes(hex_to_bytes(case["signature"])),
    )
    assert valid is case["valid"]


def test_vectors_cover_both_outcomes_and_both_wrong_lengths() -> None:
    """Check the vector file holds what the cases above rely on."""
    cases = {case["name"]: case for case in VECTORS["cases"]}
    assert any(case["valid"] for case in cases.values())
    assert any(not case["valid"] for case in cases.values())
    short = hex_to_bytes(cases["short_signature"]["signature"])
    long = hex_to_bytes(cases["long_signature"]["signature"])
    assert len(short) < SIGNATURE_SIZE < len(long)
