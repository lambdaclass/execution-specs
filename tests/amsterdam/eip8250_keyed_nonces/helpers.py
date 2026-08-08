"""Bytecode, encoding and pricing helpers for EIP-8250 tests."""

from typing import Sequence

from execution_testing import Bytecode, Op

from .spec import Spec


def approve_bytecode(
    scope: int = Spec.APPROVE_EXECUTION_AND_PAYMENT,
) -> Bytecode:
    """Return code that approves ``scope`` with empty return data."""
    return Op.APPROVE(0, 0, scope)


def rlp_encode_bytes(data: bytes) -> bytes:
    """
    Return the canonical RLP encoding of a byte string.

    Written out longhand rather than delegating to a library so that the
    wire-format tests state the encoding they assert against.
    """
    if len(data) == 1 and data[0] < 0x80:
        return data
    if len(data) <= 55:
        return bytes([0x80 + len(data)]) + data
    length = len(data).to_bytes((len(data).bit_length() + 7) // 8, "big")
    return bytes([0xB7 + len(length)]) + length + data


def rlp_encode_integer(value: int) -> bytes:
    """Return the canonical RLP encoding of a non-negative integer."""
    if value == 0:
        return b"\x80"
    return rlp_encode_bytes(
        value.to_bytes((value.bit_length() + 7) // 8, "big")
    )


def rlp_encode_list(items: Sequence[bytes]) -> bytes:
    """Return the RLP list whose already-encoded members are ``items``."""
    body = b"".join(items)
    if len(body) <= 55:
        return bytes([0xC0 + len(body)]) + body
    length = len(body).to_bytes((len(body).bit_length() + 7) // 8, "big")
    return bytes([0xF7 + len(length)]) + length + body


def nonce_calldata(nonce_keys: Sequence[int], nonce_seq: int) -> bytes:
    """
    Return ``rlp(nonce_keys) || rlp(nonce_seq)``.

    This is the byte string EIP-8250 defines as `nonce_calldata` and prices
    as transaction data.
    """
    keys = rlp_encode_list(
        [rlp_encode_integer(nonce_key) for nonce_key in nonce_keys]
    )
    return keys + rlp_encode_integer(nonce_seq)


def data_tokens(data: bytes) -> int:
    """
    Return the EIP-7623 token count of ``data``.

    `tokens_in(data)` counts one token per zero byte and four per non-zero
    byte; EIP-8141 charges `STANDARD_TOKEN_COST` per token.
    """
    zero_bytes = data.count(0)
    return zero_bytes + (len(data) - zero_bytes) * 4
