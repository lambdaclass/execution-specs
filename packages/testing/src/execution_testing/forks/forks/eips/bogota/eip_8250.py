"""EIP-8250: Keyed Nonces for Frame Transactions."""

from typing import List, Mapping, Sequence

import ethereum_rlp as eth_rlp

from execution_testing.base_types import Bytes

from ....base_fork import BaseFork, FrameGasInfo, FrameSignatureGasInfo

NONCE_MANAGER_ADDRESS = 0x0000000000000000000000000000000000008250
NONCE_MANAGER_BYTECODE = bytes.fromhex("60006000fd")

DEFAULT_NONCE_CALLDATA = Bytes(
    # `rlp(nonce_keys) || rlp(nonce_seq)` for the shape every frame
    # transaction has unless it declares keyed nonces explicitly: the
    # singleton key zero, and a sequence in the one-byte RLP range.
    eth_rlp.encode([b""]) + eth_rlp.encode(b"\x01")
)


class EIP8250(BaseFork):
    """EIP-8250 class."""

    @classmethod
    def frame_txparam_undefined_selector(cls) -> int:
        """
        EIP-8250 defines `0x0C`, `0x0D` and `0x0E`, and starts the keyed
        nonce selectors at `0x10`, leaving `0x0F` undefined.
        """
        return 0x0F

    @classmethod
    def _frame_transaction_charged_bytes(
        cls,
        frames: Sequence[FrameGasInfo],
        signatures: Sequence[FrameSignatureGasInfo],
    ) -> List[Bytes]:
        """
        Add the nonce fields, which EIP-8250 prices as transaction data.

        The specification says the `nonce_keys` and `nonce_seq` encodings
        "are priced as transaction data" with no conditionality, and a
        transaction that selects no keyed nonce still carries the
        singleton key zero — so every frame transaction pays for them.
        Both gas anchors read this helper, so extending it here is what
        keeps the intrinsic-cost and calldata-floor calculators exact
        once this EIP is active, without touching an EIP-8141 test.

        The calculator protocol carries no nonce argument, so this
        assumes the default shape (`[0]`, a one-byte sequence). That is
        exact for every test that does not declare keyed nonces; a test
        that does declares its own expectation instead, as the EIP-8250
        suite's pricing vectors do.
        """
        # The helper lives on EIP8141, which this mixin sits in front of in
        # Bogota's MRO rather than deriving from — the same reason the
        # pre-allocation overrides below carry an ignore.
        return super(EIP8250, cls)._frame_transaction_charged_bytes(  # type: ignore[misc]
            frames, signatures
        ) + [DEFAULT_NONCE_CALLDATA]

    @classmethod
    def pre_allocation(cls) -> Mapping:
        """Pre-allocate the keyed nonce manager."""
        return {
            NONCE_MANAGER_ADDRESS: {
                "nonce": 1,
                "code": NONCE_MANAGER_BYTECODE,
            }
        } | super(EIP8250, cls).pre_allocation()  # type: ignore

    @classmethod
    def pre_allocation_blockchain(cls) -> Mapping:
        """Pre-allocate the keyed nonce manager."""
        return {
            NONCE_MANAGER_ADDRESS: {
                "nonce": 1,
                "code": NONCE_MANAGER_BYTECODE,
            }
        } | super(EIP8250, cls).pre_allocation_blockchain()  # type: ignore
