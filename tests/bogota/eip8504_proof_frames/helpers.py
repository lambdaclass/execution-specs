"""Helpers for EIP-8504 proof frame tests."""

from typing import Any, List, Sequence

from execution_testing import (
    Bytecode,
    Bytes,
    Dependency,
    Fork,
    LeanSphincsKey,
    Op,
    Transaction,
    sign_dependencies,
    transaction_dependencies,
)

from ..eip8141_frame_transactions.helpers import FrameCase, verify_frame
from ..eip8141_frame_transactions.spec import Spec as FrameSpec
from .spec import Spec


def proof_frame(
    fork: Fork, dependencies: Sequence[Dependency], **overrides: Any
) -> FrameCase:
    """
    Return a proof frame declaring `dependencies`, in the order given and
    without removing repeats.

    The frame is valid as it stands: it has no target, no value, no flags
    and no state gas, and its execution gas is what its triples call for,
    all of which its receipt reports as used.
    """
    execution_gas = len(dependencies) * fork.gas_costs().PROOF_VERIFICATION
    fields: dict[str, Any] = dict(
        mode=Spec.MODE_PROOF,
        flags=FrameSpec.APPROVE_NONE,
        gas_limit=execution_gas,
        state_gas_limit=0,
        data=b"".join(dependency.encode() for dependency in dependencies),
    )
    fields.update(overrides)
    # The frame runs nothing, so its receipt follows from its budget alone.
    fields.setdefault("gas_used", int(fields["gas_limit"]))
    fields.setdefault("state_gas_used", 0)
    return FrameCase(**fields)


def account_checks(
    public_key: bytes, *, frame_index: int, triple_index: int = 0
) -> Bytecode:
    """
    Return code leaving one on the stack when frame `frame_index` is a
    proof frame whose triple `triple_index` asks for a leanSPHINCS
    signature of the transaction's signature hash under `public_key`, and
    zero otherwise.
    """
    offset = triple_index * Spec.DEPENDENCY_LENGTH
    return Op.FRAMEDATACOPY(
        0,
        offset,
        Spec.DEPENDENCY_LENGTH,
        frame_index,
        data_size=Spec.DEPENDENCY_LENGTH,
        new_memory_size=Spec.DEPENDENCY_LENGTH,
    ) + Op.AND(
        Op.AND(
            Op.EQ(
                Op.FRAMEPARAM(frame_index, FrameSpec.FRAMEPARAM_MODE),
                Spec.MODE_PROOF,
            ),
            Op.EQ(Op.MLOAD(0), Spec.LEAN_SPHINCS_SCHEME),
        ),
        Op.AND(
            Op.ISZERO(Op.MLOAD(32)),
            Op.EQ(Op.MLOAD(64), int.from_bytes(public_key, "big")),
        ),
    )


def account_code(
    public_key: bytes, *, frame_index: int, triple_index: int = 0
) -> Bytecode:
    """
    Return the validation code of an account controlled by a leanSPHINCS
    key: it approves execution and payment when the transaction declares a
    dependency on that key's signature of its own signature hash, and
    halts otherwise.

    The code only reads what the transaction declares. Whether the
    signature exists is not its concern: a block is invalid unless it
    carries a witness for every dependency.
    """
    checks = account_checks(
        public_key, frame_index=frame_index, triple_index=triple_index
    )
    approve = len(checks) + 4
    return (
        checks
        + Op.PUSH1[approve]
        + Op.JUMPI
        + Op.INVALID
        + Op.JUMPDEST
        + Op.APPROVE(0, 0, FrameSpec.APPROVE_EXECUTION_AND_PAYMENT)
    )


def account_frame(
    fork: Fork, public_key: bytes, *, frame_index: int, triple_index: int = 0
) -> FrameCase:
    """
    Return the `VERIFY` frame in which an account running
    [`account_code`] approves, priced along the path that approves.
    """
    approving_path = (
        account_checks(
            public_key, frame_index=frame_index, triple_index=triple_index
        )
        + Op.PUSH1[0]
        + Op.JUMPI
        + Op.JUMPDEST
        + Op.APPROVE(0, 0, FrameSpec.APPROVE_EXECUTION_AND_PAYMENT)
    )
    return verify_frame(execution_gas=approving_path.execution_cost(fork))


def witnesses_of(
    tx: Transaction, signers: Sequence[LeanSphincsKey]
) -> List[Bytes]:
    """
    Return the witnesses of a signed transaction's dependencies, in the
    order a block holding only that transaction carries them.
    """
    return sign_dependencies(transaction_dependencies(tx), signers)
