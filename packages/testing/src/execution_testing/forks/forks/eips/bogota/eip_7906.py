"""
EIP-7906: Transaction Assertions via State Diff Opcode.

Add a POST_TX frame mode to EIP-8141 frame transactions, executed as a
read-only trailing suffix, and the TXTRACE, TXDIFF, and EVENTDATACOPY
opcodes exposing the transaction's collapsed state diff inside such
frames.

https://eips.ethereum.org/EIPS/eip-7906
"""

from typing import Callable, Dict

from execution_testing.vm import OpcodeBase, Opcodes

from ....base_fork import BaseFork

TXTRACE_GAS_COST = 100
"""
Flat cost of TXTRACE and of the TXDIFF parameters answered entirely
from the transaction-local diff.

The EIP leaves this constant to be determined, so this value models the
choice the specification repository pins: the same price as a warm
state access, since both answer from data the transaction already holds.
"""


class EIP7906(BaseFork):
    """EIP-7906 class."""

    # The EIP reuses the EIP-8141 transaction type and envelope: it
    # adds a frame mode value and three opcodes, so it introduces no
    # new transaction types, pre-allocations, or system contracts.

    @classmethod
    def opcode_gas_map(
        cls,
    ) -> Dict[OpcodeBase, int | Callable[[OpcodeBase], int]]:
        """Add the state diff opcode gas costs."""
        gas_costs = cls.gas_costs()
        memory_expansion_calculator = cls.memory_expansion_gas_calculator()
        base_map = super(EIP7906, cls).opcode_gas_map()

        def txdiff_gas(opcode: OpcodeBase) -> int:
            """
            Return the cost of one TXDIFF lookup.

            The parameters that may fall back to live state are priced
            like any other state read, so a lookup is charged the cold
            cost the first time it names a slot or an account and the
            warm cost afterwards. The parameters answered from the
            transaction-local diff touch no access set and are charged
            the flat cost, which the metadata defaults express.
            """
            if not opcode.metadata["key_warm"]:
                return gas_costs.COLD_STORAGE_ACCESS
            if not opcode.metadata["address_warm"]:
                return gas_costs.COLD_ACCOUNT_ACCESS
            return TXTRACE_GAS_COST

        return {
            **base_map,
            Opcodes.TXTRACE: TXTRACE_GAS_COST,
            # EVENTDATACOPY is priced exactly as CALLDATACOPY.
            Opcodes.EVENTDATACOPY: cls._with_memory_expansion(
                cls._with_data_copy(
                    gas_costs.OPCODE_CALLDATACOPY_BASE, gas_costs
                ),
                memory_expansion_calculator,
            ),
            Opcodes.TXDIFF: txdiff_gas,
        }
