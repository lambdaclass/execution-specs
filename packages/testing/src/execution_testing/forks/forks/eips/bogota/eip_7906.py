"""
EIP-7906: Transaction Assertions via State Diff Opcode.

Add a POST_TX frame mode to EIP-8141 frame transactions, executed as a
read-only trailing suffix, and the TXTRACE, TXDIFF, and EVENTDATACOPY
opcodes exposing the transaction's collapsed state diff inside such
frames.

https://eips.ethereum.org/EIPS/eip-7906
"""

from ....base_fork import BaseFork


class EIP7906(BaseFork):
    """EIP-7906 class."""

    # The EIP reuses the EIP-8141 transaction type and envelope: it
    # adds a frame mode value and three opcodes, so it introduces no
    # new transaction types, pre-allocations, or system contracts.
    pass
