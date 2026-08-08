"""
Transaction-data pricing tests for EIP-8250 keyed nonces.

EIP-8250 prices `nonce_calldata = rlp(nonce_keys) || rlp(nonce_seq)` exactly
as EIP-8141 prices frame and signature data: its cost is added to
`standard_gas_limit` and its tokens to `calldata_tokens`. Both halves are
only observable in the transaction's total gas, so every test here asserts a
receipt gas value rather than a fill-time identity.
"""

from typing import Dict, List

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    EIPChecklist,
    Environment,
    Fork,
    Frame,
    FrameReceipt,
    Op,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
)

from .helpers import data_tokens, nonce_calldata
from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

# Gas handed to the trailing `INVALID` frame. Large enough that the standard
# cost clears the calldata floor for every arm, and consumed in full because
# `INVALID` burns the whole allocation.
GAS_SINK_ALLOCATION = 100_000


def charged_data(tx: Transaction, nonce_bytes: bytes) -> bytes:
    """
    Return every byte EIP-8141 charges as transaction data, in one string.

    That is each frame's `data` and each signature entry's `signer`, `msg`
    and `signature`, with EIP-8250's `nonce_calldata` prepended. The
    signature bytes are read off the signed transaction instead of being
    hard-coded, because they differ per sender and the test must stay
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
) -> tuple[int, int]:
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


@pytest.mark.parametrize(
    "nonce_keys,nonce_seq,encoded_nonce_calldata",
    [
        pytest.param([1], 0, "c10180", id="single_key"),
        pytest.param([1, 2, 3], 0, "c301020380", id="three_keys"),
        pytest.param(
            list(range(1, 17)),
            0,
            "d0" + "".join(f"{key:02x}" for key in range(1, 17)) + "80",
            id="sixteen_keys",
        ),
        pytest.param(
            [2**256 - 1], 0, "e1a0" + "ff" * 32 + "80", id="max_width_key"
        ),
        pytest.param([0], 0, "c18080", id="legacy_key_zero_sequence"),
        pytest.param(
            [0],
            2**64 - 2,
            "c18088fffffffffffffffe",
            id="legacy_key_wide_sequence",
        ),
    ],
)
@pytest.mark.pre_alloc_mutable
def test_nonce_calldata_priced_into_transaction_gas(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    nonce_keys: List[int],
    nonce_seq: int,
    encoded_nonce_calldata: str,
) -> None:
    """
    Pin R-029 through R-033: `nonce_calldata` is `rlp(nonce_keys)` followed
    by `rlp(nonce_seq)`, and its cost is added to `standard_gas_limit`.

    Every arm is in the standard regime -- asserted, not assumed -- so the
    transaction's gas is `standard_gas_limit` minus unused frame gas, and
    every frame is given exactly the gas it consumes so nothing is unused:
    a fresh keyed set costs `KEYED_NONCE_FIRST_USE_GAS` per key inside the
    approving frame, the legacy `[0]` domain costs nothing there, and the
    trailing frame runs `INVALID`, which consumes its whole allocation. The
    frame receipts pin both consumptions independently, so a frame that
    spent less than its limit fails here rather than being quietly absorbed
    into the total.

    The expected `nonce_calldata` bytes are hand-written per arm from the
    RLP rules; the two `legacy_key_*` arms differ only in the width of
    `rlp(nonce_seq)` (one byte against nine), which isolates the `nonce_seq`
    half of the definition from the `nonce_keys` half that the other arms
    vary.
    """
    keyed = nonce_keys != [0]
    first_use_gas = (
        Spec.KEYED_NONCE_FIRST_USE_GAS * len(nonce_keys) if keyed else 0
    )
    sender = pre.fund_eoa(nonce=0 if keyed else nonce_seq)
    gas_sink = pre.deploy_contract(code=Op.INVALID)

    nonce_bytes = nonce_calldata(nonce_keys, nonce_seq)
    assert nonce_bytes.hex() == encoded_nonce_calldata

    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=nonce_seq,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=first_use_gas,
            ),
            Frame(
                mode=Spec.MODE_DEFAULT,
                target=gas_sink,
                gas_limit=GAS_SINK_ALLOCATION,
            ),
        ],
    )
    standard, floor = gas_components(
        tx.with_signature_and_sender(), fork, nonce_bytes
    )
    assert standard > floor, "arm is not in the standard-cost regime"

    post: Dict[Address, Account] = {}
    if keyed:
        post[Spec.NONCE_MANAGER] = Account(
            storage={
                Spec.nonce_slot(sender, nonce_key): nonce_seq + 1
                for nonce_key in nonce_keys
            }
        )
    else:
        post[sender] = Account(nonce=nonce_seq + 1)

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx.copy(
            expected_receipt=TransactionReceipt(
                cumulative_gas_used=standard,
                frame_receipts=[
                    FrameReceipt(
                        status=Spec.STATUS_SUCCESS, gas_used=first_use_gas
                    ),
                    FrameReceipt(
                        status=Spec.STATUS_FAILURE,
                        gas_used=GAS_SINK_ALLOCATION,
                    ),
                ],
            )
        ),
        post=post,
    )


@pytest.mark.parametrize(
    "nonce_keys",
    [
        pytest.param([1], id="narrow_key"),
        pytest.param([2**256 - 1], id="widest_key"),
    ],
)
@EIPChecklist.TransactionType.Test.IntrinsicValidity.DataFloorAboveIntrinsicGasCost()
def test_nonce_calldata_counted_in_floor_above_standard_cost(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    nonce_keys: List[int],
) -> None:
    """
    Pin R-034 and R-035: `nonce_calldata` tokens are added to
    `calldata_tokens`, so they raise the floor that `gas_used` is clamped
    to when the floor exceeds the standard cost.

    The frame carries 512 bytes of data and only enough gas to cover the
    first-use surcharge, which puts `calldata_floor_gas` above
    `standard_gas_limit`; the assertion below states that ordering rather
    than trusting it. In that regime the EIP-8141 settlement rule
    `gas_used = max(gas_used_after_refund, calldata_floor_gas)` resolves to
    the floor whatever the frame consumed, so the expected total is the
    floor alone.

    Both arms name exactly one fresh key, so both pay the same single
    first-use surcharge and allocate the same frame gas; they differ only in
    how wide that key is, which moves `len(nonce_calldata)` from 3 bytes to
    35. Any difference in the totals is therefore attributable to
    `nonce_calldata` alone, and an implementation that left it out of
    `calldata_tokens` would report the same total for both.
    """
    first_use_gas = Spec.KEYED_NONCE_FIRST_USE_GAS * len(nonce_keys)
    sender = pre.fund_eoa()
    nonce_bytes = nonce_calldata(nonce_keys, 0)

    tx = Transaction(
        sender=sender,
        nonce_keys=nonce_keys,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=first_use_gas,
                data=b"\xff" * 512,
            )
        ],
    )
    standard, floor = gas_components(
        tx.with_signature_and_sender(), fork, nonce_bytes
    )
    assert floor > standard, "arm is not in the calldata-floor regime"

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx.copy(
            expected_receipt=TransactionReceipt(cumulative_gas_used=floor)
        ),
        post={
            Spec.NONCE_MANAGER: Account(
                storage={
                    Spec.nonce_slot(sender, nonce_key): 1
                    for nonce_key in nonce_keys
                }
            )
        },
    )


@pytest.mark.parametrize(
    "allowance,valid",
    [
        pytest.param(
            "standard_cost",
            False,
            id="allowance_at_standard_cost",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            "floor_excluding_nonce_calldata",
            False,
            id="allowance_at_floor_excluding_nonce_calldata",
            marks=pytest.mark.exception_test,
        ),
        pytest.param(
            "floor_minus_one",
            False,
            id="allowance_one_below_floor",
            marks=pytest.mark.exception_test,
        ),
        pytest.param("floor", True, id="allowance_at_floor"),
    ],
)
@EIPChecklist.TransactionType.Test.IntrinsicValidity.DataFloorAboveIntrinsicGasCost()
def test_calldata_floor_above_standard_cost_exceeds_gas_allowance(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    allowance: str,
    valid: bool,
) -> None:
    """
    Pin R-034 and R-035 on the rejecting side: when the calldata floor
    exceeds the standard cost the floor -- `nonce_calldata` included -- is
    the amount of gas the block must be able to supply, so a block that can
    only supply less rejects the transaction.

    A frame transaction has no gas limit field, so the checklist's
    "gas_limit == intrinsic_gas_cost" case is expressed through the only
    quantity that can be held at the standard cost while the floor sits
    above it: the gas the block has left. EIP-8141 derives the
    inclusion-facing anchor as `max(standard_gas_limit, calldata_floor_gas)`
    and admits the transaction only if the block can supply it, so the four
    arms bracket that anchor from both sides.

    Three of them reject and one accepts. `allowance_at_standard_cost` is
    the checklist case itself: the block supplies exactly the standard cost,
    which the floor is above. `allowance_one_below_floor` and
    `allowance_at_floor` are the exact boundary pair, and locate the
    threshold at the floor rather than anywhere below it.

    `allowance_at_floor_excluding_nonce_calldata` is the discriminating arm.
    Its allowance is the floor recomputed over the same charged bytes with
    `nonce_calldata` removed -- the value an implementation that left
    `nonce_calldata` out of `calldata_tokens` would derive. The gap between
    the two floors is `64 * len(nonce_calldata)` gas, hand-derived from
    EIP-7976's uniform floor weight of four tokens per byte at sixteen gas
    per token; with the widest key that is 35 bytes, so 2,240 gas. Under
    the EIP the transaction still does not fit and is rejected; under an
    implementation that omits those bytes it fits exactly and is accepted,
    which is the outcome this arm forbids. The assertion below states the
    ordering the three thresholds must be in, so an arm that stopped
    discriminating fails here instead of passing quietly.
    """
    nonce_key = 2**256 - 1
    first_use_gas = Spec.KEYED_NONCE_FIRST_USE_GAS
    sender = pre.fund_eoa()
    nonce_bytes = nonce_calldata([nonce_key], 0)

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=first_use_gas,
                data=b"\xff" * 512,
            )
        ],
    )
    signed = tx.with_signature_and_sender()
    standard, floor = gas_components(signed, fork, nonce_bytes)
    _, floor_excluding_nonce_calldata = gas_components(signed, fork, b"")
    assert floor - floor_excluding_nonce_calldata == 64 * len(nonce_bytes)
    assert floor > floor_excluding_nonce_calldata > standard, (
        "arm cannot discriminate nonce_calldata's share of the floor"
    )

    gas_allowance = {
        "standard_cost": standard,
        "floor_excluding_nonce_calldata": floor_excluding_nonce_calldata,
        "floor_minus_one": floor - 1,
        "floor": floor,
    }[allowance]

    state_test(
        env=Environment(gas_limit=gas_allowance),
        pre=pre,
        tx=tx.copy(
            expected_receipt=TransactionReceipt(cumulative_gas_used=floor)
        )
        if valid
        else tx.copy(error=TransactionException.GAS_ALLOWANCE_EXCEEDED),
        post={
            Spec.NONCE_MANAGER: Account(
                storage={Spec.nonce_slot(sender, nonce_key): 1 if valid else 0}
            )
        },
    )
