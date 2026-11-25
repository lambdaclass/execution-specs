"""
EIP-8504: Proof Frames for leanSPHINCS Signatures.

Add a frame mode whose frame executes no code and declares the
leanSPHINCS signatures the transaction depends on. The signatures are
carried by the block, outside its transactions, and committed to by a
new header field.

https://eips.ethereum.org/EIPS/eip-8504
"""

from dataclasses import replace
from typing import Sequence

from execution_testing.base_types import Bytes

from ....base_fork import BaseFork, FrameGasInfo
from ....gas_costs import GasCosts

PROOF_MODE = 4
"""Frame mode value of a `PROOF` frame."""

DEPENDENCY_LENGTH = 96
"""Length in bytes of one dependency triple in a `PROOF` frame's data."""

LEAN_SPHINCS_SIGNATURE_SIZE = 7_740
"""Length in bytes of a leanSPHINCS signature, the witness of a dependency."""


class EIP8504(BaseFork):
    """EIP-8504 class."""

    @classmethod
    def _frame_modes_after_assert(cls) -> int:
        """The `PROOF` frame mode is introduced after `ASSERT`."""
        count = super(EIP8504, cls)._frame_modes_after_assert()
        assert count == 0, "PROOF must be the next frame mode"
        return count + 1

    @classmethod
    def gas_costs(cls) -> GasCosts:
        """
        Add the execution gas a `PROOF` frame carries per dependency and
        the intrinsic cost of a dependency's witness, whose bytes are
        priced as non-zero calldata at the floor rate.
        """
        parent = super(EIP8504, cls).gas_costs()
        return replace(
            parent,
            PROOF_VERIFICATION=3_000,
            TX_DEPENDENCY_WITNESS=(
                LEAN_SPHINCS_SIGNATURE_SIZE
                * parent.TX_DATA_TOKEN_STANDARD
                * parent.TX_DATA_TOKEN_FLOOR
            ),
        )

    @classmethod
    def _frame_transaction_dependency_witness_cost(
        cls, frames: Sequence[FrameGasInfo]
    ) -> int:
        """
        Charge the witness of each distinct dependency the transaction
        declares. A triple repeated within a `PROOF` frame or across
        several is one dependency.
        """
        triples = set()
        for frame in frames:
            if int(frame.mode) != PROOF_MODE:
                continue
            data = Bytes(frame.data)
            for start in range(
                0, len(data) - DEPENDENCY_LENGTH + 1, DEPENDENCY_LENGTH
            ):
                triples.add(bytes(data[start : start + DEPENDENCY_LENGTH]))
        return len(triples) * cls.gas_costs().TX_DEPENDENCY_WITNESS

    @classmethod
    def header_dependency_witnesses_hash_required(cls) -> bool:
        """Header must contain the dependency witnesses hash."""
        return True

    @classmethod
    def engine_execution_payload_dependency_witnesses(cls) -> bool:
        """Execution payload includes `dependency_witnesses`."""
        return True
