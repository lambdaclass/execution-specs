"""
EIP-8272: Recent Roots for Frame Transactions.

Let a frame transaction declare recent roots in its signed envelope,
checked against its pre-state before any frame executes and readable
from validation code afterwards.

https://eips.ethereum.org/EIPS/eip-8272
"""

from typing import Callable, Dict, List, Mapping

from execution_testing.vm import (
    OpcodeBase,
    Opcodes,
)

from ....base_fork import BaseFork

RECENT_ROOT_ADDRESS = 0x0000000000000000000000000000000000008272

RECENT_ROOT_ACCOUNT = {
    # EIP-8272 creates the account with a nonce of one; its runtime code
    # is not yet defined by the specification, so none is installed.
    "nonce": 1,
}


class EIP8272(BaseFork):
    """EIP-8272 class."""

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
