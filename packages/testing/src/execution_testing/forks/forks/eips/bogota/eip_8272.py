"""
EIP-8272: Recent Roots for Frame Transactions.

Let a frame transaction declare recent roots in its signed envelope,
checked against its pre-state before any frame executes and readable
from validation code afterwards.

https://eips.ethereum.org/EIPS/eip-8272
"""

from typing import Callable, Dict, List, Mapping, Sequence

import ethereum_rlp as eth_rlp

from execution_testing.base_types import Bytes
from execution_testing.vm import (
    OpcodeBase,
    Opcodes,
)

from ....base_fork import BaseFork, FrameGasInfo, FrameSignatureGasInfo

RECENT_ROOT_ADDRESS = 0x0000000000000000000000000000000000008272

RECENT_ROOT_ACCOUNT = {
    # EIP-8272 creates the account with a nonce of one; its runtime code
    # is not yet defined by the specification, so none is installed.
    "nonce": 1,
}

EMPTY_REFERENCE_LIST = Bytes(eth_rlp.encode([]))
"""`rlp(recent_root_references)` when no reference is declared: one byte."""


class EIP8272(BaseFork):
    """EIP-8272 class."""

    @classmethod
    def _frame_transaction_charged_bytes(
        cls,
        frames: Sequence[FrameGasInfo],
        signatures: Sequence[FrameSignatureGasInfo],
    ) -> List[Bytes]:
        """
        Add the recent root reference list, which EIP-8272 prices as
        transaction data.

        Every frame transaction pays it: the specification adds
        `recent_root_calldata_cost` to `standard_gas_limit` and
        `recent_root_calldata_tokens` to `calldata_tokens` with no
        conditionality, and an empty list still encodes to one byte.
        Both gas anchors read this helper, so extending it here keeps the
        intrinsic-cost and calldata-floor calculators exact once this EIP
        is active, without touching an EIP-8141 test.

        The per-reference charge is not added here: it applies only to a
        transaction that declares at least one reference, and the
        calculator protocol carries no reference argument. The EIP-8272
        suite states those expectations itself.
        """
        # The helper lives on EIP8141, which this mixin sits in front of in
        # Bogota's MRO rather than deriving from — the same reason the
        # pre-allocation overrides below carry an ignore.
        return super(EIP8272, cls)._frame_transaction_charged_bytes(  # type: ignore[misc]
            frames, signatures
        ) + [EMPTY_REFERENCE_LIST]

    @classmethod
    def pre_allocation(cls) -> Mapping:
        """Pre-allocate the recent root contract."""
        return {RECENT_ROOT_ADDRESS: RECENT_ROOT_ACCOUNT} | super(
            EIP8272, cls
        ).pre_allocation()  # type: ignore

    @classmethod
    def pre_allocation_blockchain(cls) -> Mapping:
        """Pre-allocate the recent root contract."""
        return {RECENT_ROOT_ADDRESS: RECENT_ROOT_ACCOUNT} | super(
            EIP8272, cls
        ).pre_allocation_blockchain()  # type: ignore

    @classmethod
    def opcode_gas_map(
        cls,
    ) -> Dict[OpcodeBase, int | Callable[[OpcodeBase], int]]:
        """Add RECENTROOTREFLOAD opcode gas cost."""
        gas_costs = cls.gas_costs()
        base_map = super(EIP8272, cls).opcode_gas_map()
        return {
            **base_map,
            Opcodes.RECENTROOTREFLOAD: gas_costs.VERY_LOW,
        }

    @classmethod
    def valid_opcodes(cls) -> List[Opcodes]:
        """Add RECENTROOTREFLOAD opcode."""
        return [Opcodes.RECENTROOTREFLOAD] + super(
            EIP8272, cls
        ).valid_opcodes()
