"""Bytecode, encoding and pricing helpers for EIP-8250 tests."""

from typing import Sequence, Tuple

from execution_testing import Bytecode, Fork, Op, Transaction

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


def charged_data(tx: Transaction, nonce_bytes: bytes) -> bytes:
    """
    Return every byte EIP-8141 charges as transaction data, in one string.

    That is each frame's `data` and each signature entry's `signer`, `msg`
    and `signature`, with EIP-8250's `nonce_calldata` prepended. The
    signature bytes are read off the signed transaction instead of being
    hard-coded, because they differ per sender and the tests must stay
    correct under `execute` as well as under `fill`.
    """
    data = nonce_bytes
    for frame in tx.frames if tx.frames is not None else []:
        data += bytes(frame.data)
    for signature in tx.signatures if tx.signatures is not None else []:
        data += (
            bytes(signature.signer)
            + bytes(signature.msg)
            + bytes(signature.signature)
        )
    return data


def gas_components(
    tx: Transaction, fork: Fork, nonce_bytes: bytes
) -> Tuple[int, int]:
    """
    Return `(standard_gas_limit, calldata_floor_gas)` for a frame
    transaction, re-derived from the pinned EIP-8141 formulas.

        standard_gas_limit = FRAME_TX_INTRINSIC_COST
            + len(frames) * FRAME_TX_PER_FRAME_COST
            + signature_verification_cost
            + STANDARD_TOKEN_COST * tokens_in(charged data)
            + sum(frame.gas_limit)

        calldata_floor_gas = FRAME_TX_INTRINSIC_COST
            + len(frames) * FRAME_TX_PER_FRAME_COST
            + signature_verification_cost
            + <the fork's calldata floor for the charged data>

    Only the floor's per-token weight is taken from the framework rather
    than from the EIP text: Amsterdam reprices it under EIP-7976, so a
    literal here would encode a fork constant that this EIP does not own.
    Everything else is the EIP's own arithmetic.
    """
    frames = tx.frames if tx.frames is not None else []
    signatures = tx.signatures if tx.signatures is not None else []
    base = (
        Spec.FRAME_TX_INTRINSIC_COST
        + len(frames) * Spec.FRAME_TX_PER_FRAME_COST
        + len(signatures) * Spec.SIGNATURE_GAS_SECP256K1
    )
    data = charged_data(tx, nonce_bytes)
    standard = (
        base
        + Spec.STANDARD_TOKEN_COST * data_tokens(data)
        + sum(int(frame.gas_limit) for frame in frames)
    )
    floor_calculator = fork.transaction_data_floor_cost_calculator()
    floor = base + floor_calculator(data=data) - floor_calculator(data=b"")
    return standard, floor
