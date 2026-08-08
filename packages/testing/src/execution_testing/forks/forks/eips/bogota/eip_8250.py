"""EIP-8250: Keyed Nonces for Frame Transactions."""

from typing import Mapping

from ....base_fork import BaseFork

NONCE_MANAGER_ADDRESS = 0x0000000000000000000000000000000000008250
NONCE_MANAGER_BYTECODE = bytes.fromhex("60006000fd")


class EIP8250(BaseFork):
    """EIP-8250 class."""

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
