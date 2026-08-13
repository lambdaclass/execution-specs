"""Helpers for EIP-7906 transaction assertion tests."""

from typing import Any, Dict, Iterable, List

from execution_testing import (
    Account,
    Address,
    Alloc,
    Bytecode,
    Conditional,
    Frame,
    Op,
)

from tests.amsterdam.eip8141_frame_transactions.spec import (
    Spec as FrameSpec,
)

from .spec import Spec

MAX_UINT256 = 2**256 - 1
"""The maximum 256-bit stack value."""

BODY_FRAME_GAS = 500_000
"""
Default gas limit of execution-body frames, with ample headroom for a
few storage writes: a fresh `SSTORE` draws its state gas from the
frame's execution gas, since frame transactions hold no state gas
reservoir.
"""

ASSERTER_FRAME_GAS = 1_000_000
"""
Default gas limit of `POST_TX` assertion frames, with ample headroom
for dozens of state-diff reads. Gas-sensitive tests size their frames
explicitly instead.
"""

SENTINEL_SLOT = 0x00
"""Storage slot the sentinel contract records the body write into."""

SENTINEL_MARKER = 0xC0DE
"""Distinctive value the sentinel contract stores when executed."""


def post_tx_frame(**overrides: Any) -> Frame:
    """
    Return a `POST_TX` frame with an ample gas limit.

    Keyword arguments override the corresponding frame fields, for
    variants that differ from the canonical frame in a single field.
    """
    kwargs: Dict[str, Any] = dict(
        mode=Spec.MODE_POST_TX,
        gas_limit=ASSERTER_FRAME_GAS,
    )
    kwargs.update(overrides)
    return Frame(**kwargs)


def assert_eq(actual: Bytecode, expected: Any) -> Bytecode:
    """
    Return asserter code that reverts unless `actual` pushes exactly
    `expected`.

    The building block of every assertion frame: a failed comparison
    reverts the frame, which rolls back the execution body — observed
    through the sentinel contract's post state and the frame receipt
    statuses.
    """
    return Conditional(
        condition=Op.EQ(actual, expected),
        if_false=Op.REVERT(0, 0),
    )


def deploy_sentinel(pre: Alloc) -> Address:
    """
    Deploy the sentinel contract recording that the execution body ran.

    A body frame targeting it stores `SENTINEL_MARKER`; the post state
    then distinguishes a passing assertion (the marker survived) from a
    failing one (the whole body, including the marker, was rolled
    back).
    """
    return pre.deploy_contract(
        code=Op.SSTORE(SENTINEL_SLOT, SENTINEL_MARKER) + Op.STOP
    )


SLOT_CALL_RESULT_PLUS_ONE = 0x00
"""Probe slot recording the sub-call result plus one (1 = failure)."""

SLOT_ALL_GAS_CONSUMED = 0x01
"""Probe slot recording whether the sub-call burned its whole gas."""

FORWARDED_GAS = 50_000
"""Gas the probe forwards to the halting helper."""

OPCODE_USES: Dict[str, Bytecode] = {
    "txtrace": Op.POP(Op.TXTRACE(0, Spec.TXTRACE_BALANCES_CHANGED)) + Op.STOP,
    "txdiff": Op.POP(Op.TXDIFF(0, 0, Spec.TXDIFF_SLOT_VALUE_BEFORE)) + Op.STOP,
    "eventdatacopy": Op.EVENTDATACOPY(0, 0, 0, 0) + Op.STOP,
}
"""Minimal well-formed use of each state-diff opcode."""


def deploy_halt_probe(pre: Alloc, helper: Bytecode | Address) -> Address:
    """
    Deploy a probe calling a helper that must exceptionally halt.

    The helper is either code to deploy or the address of an existing
    account to call. The probe records the sub-call's result plus one
    — so a recorded 1 means failure and an unwritten slot stays 0 —
    and whether the call consumed more than the gas it forwarded,
    distinguishing the exceptional halt from a clean return or a
    revert.
    """
    if isinstance(helper, Address):
        helper_address = helper
    else:
        helper_address = pre.deploy_contract(code=helper)
    return pre.deploy_contract(
        code=Op.MSTORE(0, Op.GAS)
        + Op.SSTORE(
            SLOT_CALL_RESULT_PLUS_ONE,
            Op.ADD(Op.CALL(FORWARDED_GAS, helper_address, 0, 0, 0, 0, 0), 1),
        )
        + Op.SSTORE(
            SLOT_ALL_GAS_CONSUMED,
            Op.GT(Op.SUB(Op.MLOAD(0), Op.GAS), FORWARDED_GAS),
        )
        + Op.STOP
    )


HALT_PROBE_POST = Account(
    storage={
        SLOT_CALL_RESULT_PLUS_ONE: 1,
        SLOT_ALL_GAS_CONSUMED: 1,
    }
)
"""Post state of a probe whose sub-call exceptionally halted."""


def sorted_by_address(addresses: Iterable[Address]) -> List[Address]:
    """
    Sort addresses ascending as numerical uint160 values: the order
    the TXTRACE enumerations use.
    """
    return sorted(addresses, key=lambda a: int.from_bytes(a, "big"))


def deploy_approving_sender(pre: Alloc, balance: int) -> Address:
    """
    Deploy a contract sender approving execution and payment.

    Contract senders carry no signature entries and their frames no
    data, so the transaction's intrinsic cost — and with it the exact
    gas pre-charge — reduces to the base and per-frame constants,
    hand-computable via `contract_sender_max_cost`.

    The code accepts plain value transfers (a non-zero `CALLVALUE`
    skips the approval), so refund-style bodies can send ether back to
    the sender without tripping `APPROVE`'s target check.
    """
    return pre.deploy_contract(
        code=Conditional(
            condition=Op.CALLVALUE,
            if_true=Op.STOP,
            if_false=Op.APPROVE(0, 0, FrameSpec.APPROVE_EXECUTION_AND_PAYMENT),
        ),
        balance=balance,
    )


def contract_sender_max_cost(frames: List[Frame], max_fee_per_gas: int) -> int:
    """
    Return the exact gas pre-charge escrowed from the payer of a
    contract-sender frame transaction with no blobs.

    Derived by hand from the EIP-8141 accounting: the intrinsic cost
    (base plus per-frame constants; no signature entries and no frame
    data for a contract sender) plus every frame's gas limit, priced
    at the maximum fee per gas. The calldata floor equals the
    data-free intrinsic, so it never exceeds the standard gas limit.
    """
    assert all(len(frame.data) == 0 for frame in frames), (
        "frame data would add data tokens to the intrinsic cost"
    )
    intrinsic = (
        FrameSpec.FRAME_TX_INTRINSIC_COST
        + len(frames) * FrameSpec.FRAME_TX_PER_FRAME_COST
    )
    standard_gas_limit = intrinsic + sum(
        int(frame.gas_limit) for frame in frames
    )
    return standard_gas_limit * max_fee_per_gas
