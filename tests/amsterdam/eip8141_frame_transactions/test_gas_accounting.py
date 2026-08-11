"""
Gas accounting tests for
[EIP-8141: Frame Transaction](https://eips.ethereum.org/EIPS/eip-8141).

A frame transaction has no gas limit field: its gas anchors derive
from the intrinsic constants, the calldata-priced byte fields, the
signature verification costs, and the frame gas limits. Every
expectation here is recomputed from the spec's formulas — with the
calldata floor counting each charged byte uniformly, per the
composition with EIP-7976 in this fork — and pinned through receipt
gas and balance arithmetic.

All senders are contracts: contract senders carry no protocol
signature entries, so every byte that enters the gas formulas is
chosen by the test.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytecode,
    Bytes,
    Environment,
    Fork,
    FrameReceipt,
    FrameSignature,
    Op,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
    keccak256,
)

from .helpers import default_frame, verify_frame
from .signature_helpers import DIGEST, P256_SIGNATURE
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

# EIP-8141 is slated for the fork after Amsterdam, so fixtures are
# labeled with the pseudo `Bogota` fork (Amsterdam + EIP-8141), even
# though the spec prototypes the EIP inside the Amsterdam fork module.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

BASE_FEE = 7
"""Base fee of the block executing the transactions."""

PRIORITY_FEE = 2
"""Priority fee, making the coinbase credit visible."""

MAX_FEE = 1_000
"""Maximum fee, far above the effective price to make the payer's
refund path visible."""

FUNDS = 10**18
"""Funding of the paying sender."""

APPROVE_ALL_CODE = Op.APPROVE(0, 0, Spec.APPROVE_EXECUTION_AND_PAYMENT)
"""Sender code approving execution and payment."""

EMBEDDED_SECP256K1_SIGNATURE = bytes.fromhex(
    "000d10c237d89a0f0c6f974d8d8a4283bbd1cf1d326332cdf0dbc418c88246e4"
    "3138d4f3b0f50965345a1faa1845f52ade58ced5915f335ad58fdddf46eead88"
    "79"
)
"""
A valid `v || r || s` secp256k1 signature over the explicit digest
`DIGEST`, produced deterministically (RFC 6979) with the private key
`2`. Embedded so the signature's byte content — which enters the gas
formulas — is fixed at authoring time.
"""

EMBEDDED_SECP256K1_SIGNER = bytes.fromhex(
    "2b5ad5c4795c026514f8317c7a215e218dccd6cf"
)
"""The address of private key `2`, the embedded signature's signer."""


def tokens_in(data: bytes) -> int:
    """
    Count calldata tokens as the spec's `tokens_in` pseudocode: one
    per zero byte and four per nonzero byte.
    """
    zero_bytes = data.count(0)
    return zero_bytes + 4 * (len(data) - zero_bytes)


def entry_data_bytes(entry: FrameSignature) -> bytes:
    """Concatenate the calldata-priced byte fields of an entry."""
    return (
        bytes(entry.signer) + bytes(entry.msg) + bytes(entry.signature)
    )


def gas_anchors(
    fork: Fork,
    frames_data: list[bytes],
    entries: list[FrameSignature],
    scheme_gas: dict[int, int],
) -> tuple[int, int]:
    """
    Recompute the spec's intrinsic execution gas and calldata floor.

    The floor counts every charged byte uniformly at four tokens, per
    the EIP-7976 composition in this fork, instead of the pinned
    spec's zero/nonzero token split.
    """
    gas_costs = fork.gas_costs()
    signature_gas = sum(scheme_gas[int(e.scheme)] for e in entries)
    charged_bytes = b"".join(frames_data) + b"".join(
        entry_data_bytes(e) for e in entries
    )
    base = (
        Spec.FRAME_TX_INTRINSIC_COST
        + len(frames_data) * Spec.FRAME_TX_PER_FRAME_COST
        + signature_gas
    )
    intrinsic_execution = base + gas_costs.TX_DATA_TOKEN_STANDARD * (
        tokens_in(charged_bytes)
    )
    calldata_floor = base + (
        4 * gas_costs.TX_DATA_TOKEN_FLOOR * len(charged_bytes)
    )
    return intrinsic_execution, calldata_floor


SCHEME_GAS = {
    Spec.SCHEME_ARBITRARY: Spec.GAS_SIGNATURE_ARBITRARY,
    Spec.SCHEME_SECP256K1: Spec.GAS_SIGNATURE_SECP256K1,
    Spec.SCHEME_P256: Spec.GAS_SIGNATURE_P256,
}
"""Signature verification gas per scheme, from the spec's table."""


def test_exact_gas_accounting_standard(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Pin every term of the standard gas path: the intrinsic base and
    per-frame constants, the frame data tokens, the per-frame gas
    used, and the fee settlement.

    The receipt pins the cumulative gas and both frame gas values;
    the post balances pin conservation — the payer is debited the gas
    used at the effective price, the coinbase collects exactly the
    priority portion, and the escrowed surplus up to the maximum fee
    returns to the payer.
    """
    worker_data = b"\x00" * 8 + b"\x01" * 4
    worker_code = Op.STOP
    sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=FUNDS)
    worker = pre.deploy_contract(code=worker_code)
    gas_costs = fork.gas_costs()

    intrinsic_execution, calldata_floor = gas_anchors(
        fork, [b"", worker_data], [], SCHEME_GAS
    )
    verify_gas_used = (
        gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    )
    worker_gas_used = (
        gas_costs.COLD_ACCOUNT_ACCESS + worker_code.gas_cost(fork)
    )
    gas_used = intrinsic_execution + verify_gas_used + worker_gas_used
    assert calldata_floor < gas_used, "arm must be standard-dominant"

    effective_price = BASE_FEE + PRIORITY_FEE
    env = Environment(base_fee_per_gas=BASE_FEE)

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=PRIORITY_FEE,
        frames=[
            verify_frame(),
            default_frame(target=worker, data=Bytes(worker_data)),
        ],
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=gas_used,
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS, gas_used=verify_gas_used
                ),
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS, gas_used=worker_gas_used
                ),
            ],
        ),
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                nonce=2, balance=FUNDS - gas_used * effective_price
            ),
            env.fee_recipient: Account(balance=gas_used * PRIORITY_FEE),
        },
    )


def test_exact_gas_accounting_floor_dominant(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Pin the calldata floor as the final gas when zero-heavy frame
    data outweighs the executed gas.

    In this fork the floor prices every charged byte uniformly, per
    the composition with EIP-7976; under the pinned spec's own
    zero/nonzero token split this arm's floor would fall below the
    standard path, so the expected value also discriminates the
    composed formula. Fees settle on the floored gas, pinned through
    the payer and coinbase balances.
    """
    zero_data = b"\x00" * 2_000
    sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=FUNDS)
    gas_costs = fork.gas_costs()

    intrinsic_execution, calldata_floor = gas_anchors(
        fork, [zero_data], [], SCHEME_GAS
    )
    verify_gas_used = (
        gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    )
    assert calldata_floor > intrinsic_execution + verify_gas_used, (
        "arm must be floor-dominant"
    )
    gas_used = calldata_floor

    effective_price = BASE_FEE + PRIORITY_FEE
    env = Environment(base_fee_per_gas=BASE_FEE)

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=PRIORITY_FEE,
        # The sender's approving code ignores its calldata, so the
        # zero-heavy data rides on the verify frame itself.
        frames=[verify_frame(data=Bytes(zero_data))],
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=gas_used,
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS, gas_used=verify_gas_used
                ),
            ],
        ),
    )

    state_test(
        env=env,
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                nonce=2, balance=FUNDS - gas_used * effective_price
            ),
            env.fee_recipient: Account(balance=gas_used * PRIORITY_FEE),
        },
    )


@pytest.mark.parametrize("entry_count", [1, 3])
@pytest.mark.parametrize(
    "scheme",
    [
        pytest.param(Spec.SCHEME_ARBITRARY, id="arbitrary"),
        pytest.param(Spec.SCHEME_SECP256K1, id="secp256k1"),
        pytest.param(Spec.SCHEME_P256, id="p256"),
    ],
)
def test_signature_gas_constants(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    scheme: int,
    entry_count: int,
) -> None:
    """
    Pin each signature scheme's verification gas constant through the
    cumulative gas of transactions carrying one and three entries of
    that scheme: two points solve the per-entry cost — the scheme
    constant plus the entry's byte cost — as a linear term.

    The entries' byte content is fixed at authoring time: the
    arbitrary witness is chosen, and the secp256k1 and P-256 entries
    embed deterministic signatures over an explicit digest. The
    contract sender approves via code, so the entries are validated
    but never consumed.
    """
    if scheme == Spec.SCHEME_ARBITRARY:
        entries = [
            FrameSignature(
                scheme=scheme, signature=Bytes(b"\x00\x00\x00\xab\xcd")
            )
            for _ in range(entry_count)
        ]
    elif scheme == Spec.SCHEME_SECP256K1:
        entries = [
            FrameSignature(
                scheme=scheme,
                signer=Bytes(EMBEDDED_SECP256K1_SIGNER),
                msg=Bytes(DIGEST),
                signature=Bytes(EMBEDDED_SECP256K1_SIGNATURE),
            )
            for _ in range(entry_count)
        ]
    else:
        entries = [
            FrameSignature(
                scheme=scheme,
                signer=Bytes(keccak256(P256_SIGNATURE[64:])[12:]),
                msg=Bytes(DIGEST),
                signature=Bytes(P256_SIGNATURE),
            )
            for _ in range(entry_count)
        ]

    sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=FUNDS)
    gas_costs = fork.gas_costs()

    intrinsic_execution, calldata_floor = gas_anchors(
        fork, [b""], entries, SCHEME_GAS
    )
    verify_gas_used = (
        gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    )
    gas_used = max(
        intrinsic_execution + verify_gas_used, calldata_floor
    )

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=PRIORITY_FEE,
        frames=[verify_frame()],
        signatures=entries,
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=gas_used,
            payer=sender,
        ),
    )

    state_test(
        env=Environment(base_fee_per_gas=BASE_FEE),
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=2)},
    )


@pytest.mark.parametrize(
    "gas_shortfall",
    [
        pytest.param(0, id="frame_gas_exact"),
        pytest.param(1, id="frame_gas_exact_minus_one"),
    ],
)
def test_frame_oog_isolation(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_shortfall: int,
) -> None:
    """
    Budget a frame with exactly its execution cost, or one unit less.

    The exact budget succeeds even though the transaction's intrinsic
    costs dwarf it — they are charged outside frame budgets — and the
    one-below budget runs out of gas although the previous frame left
    most of its own limit unused: unused frame gas never flows
    forward. An out-of-gas frame forfeits its whole limit.
    """
    worker_code = Op.POP(Op.ADD(1, 2)) + Op.STOP
    worker = pre.deploy_contract(code=worker_code)
    sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=FUNDS)
    gas_costs = fork.gas_costs()

    exact = gas_costs.COLD_ACCOUNT_ACCESS + worker_code.gas_cost(fork)
    verify_gas_used = (
        gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    )
    intrinsic_execution, _ = gas_anchors(fork, [b"", b""], [], SCHEME_GAS)
    # An out-of-gas frame forfeits exactly its limit.
    worker_gas_used = exact - gas_shortfall
    gas_used = intrinsic_execution + verify_gas_used + worker_gas_used

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=PRIORITY_FEE,
        frames=[
            verify_frame(),
            default_frame(target=worker, gas_limit=exact - gas_shortfall),
        ],
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=gas_used,
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS, gas_used=verify_gas_used
                ),
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS
                    if gas_shortfall == 0
                    else Spec.STATUS_FAILURE,
                    gas_used=worker_gas_used,
                ),
            ],
        ),
    )

    state_test(
        env=Environment(base_fee_per_gas=BASE_FEE),
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=2)},
    )


SKIPPED_FRAME_GAS = 3_000_000
"""Gas limit of the skipped frame, large enough that charging it
would be unmistakable in the cumulative gas."""


def test_skipped_frame_gas_refund(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Return a skipped frame's entire gas allotment: a failed atomic
    batch skips its remaining frame, whose receipt reports zero gas
    and whose three-million-gas limit is absent from the cumulative
    gas.
    """
    reverter_code = Op.REVERT(0, 0)
    reverter = pre.deploy_contract(code=reverter_code)
    skipped_target = pre.deploy_contract(code=Op.STOP)
    sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=FUNDS)
    gas_costs = fork.gas_costs()

    verify_gas_used = (
        gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    )
    reverter_gas_used = (
        gas_costs.COLD_ACCOUNT_ACCESS + reverter_code.gas_cost(fork)
    )
    intrinsic_execution, _ = gas_anchors(
        fork, [b"", b"", b""], [], SCHEME_GAS
    )
    gas_used = intrinsic_execution + verify_gas_used + reverter_gas_used

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=PRIORITY_FEE,
        frames=[
            verify_frame(),
            default_frame(
                flags=Spec.ATOMIC_BATCH_FLAG,
                target=reverter,
                gas_limit=100_000,
            ),
            default_frame(
                target=skipped_target, gas_limit=SKIPPED_FRAME_GAS
            ),
        ],
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=gas_used,
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS, gas_used=verify_gas_used
                ),
                FrameReceipt(
                    status=Spec.STATUS_FAILURE, gas_used=reverter_gas_used
                ),
                FrameReceipt(status=Spec.STATUS_SKIPPED, gas_used=0),
            ],
        ),
    )

    state_test(
        env=Environment(base_fee_per_gas=BASE_FEE),
        pre=pre,
        tx=tx,
        post={sender: Account(nonce=2)},
    )


SLOT_CLEARED = 0x07
"""Pre-set storage slot the refund tests clear."""


def burner_code(target_gas: int) -> Bytecode:
    """
    Build code consuming exactly `target_gas`: a calldata copy sized
    to cover the bulk of the cost through its memory expansion,
    topped up with single-gas jump destinations.

    Solving for the copy size in-test keeps the expectation valid
    under repricing of the surrounding terms.
    """
    words = 0
    while True:
        base = 12 + 6 * (words + 1) + ((words + 1) * (words + 1)) // 512
        if base > target_gas:
            break
        words += 1
    copy = Op.CALLDATACOPY(
        dest_offset=0,
        offset=0,
        size=words * 32,
        data_size=words * 32,
        new_memory_size=words * 32,
        old_memory_size=0,
    )
    consumed = 12 + 6 * words + (words * words) // 512
    filler = target_gas - consumed
    assert 0 <= filler < 4_096, "filler must stay a small code tail"
    return copy + Op.JUMPDEST * filler + Op.STOP


@pytest.mark.parametrize(
    "quotient_position",
    [
        pytest.param("under", id="refund_below_quotient"),
        pytest.param("exact", id="refund_at_quotient"),
        pytest.param("over", id="refund_above_quotient"),
    ],
)
def test_refund_accounting(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    quotient_position: str,
) -> None:
    """
    Apply the storage clear refund against the gas used before
    refunds, capped at one fifth: the burner frame sizes the total so
    the refund lands below, exactly at, or above the cap.

    Every quantity is recomputed in-test from the fork's costs, so
    the arms keep straddling the quotient under repricing.
    """
    clear_code = Op.SSTORE(
        key=SLOT_CLEARED,
        value=0,
        key_warm=False,
        original_value=1,
        current_value=1,
        new_value=0,
    )
    clearer_code = clear_code + Op.STOP
    clearer = pre.deploy_contract(
        code=clearer_code, storage={SLOT_CLEARED: 1}
    )
    sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=FUNDS)
    gas_costs = fork.gas_costs()

    refund = clear_code.refund(fork)
    verify_gas_used = (
        gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    )
    clearer_gas_used = (
        gas_costs.COLD_ACCOUNT_ACCESS + clearer_code.gas_cost(fork)
    )
    intrinsic_execution, _ = gas_anchors(
        fork, [b"", b"", b""], [], SCHEME_GAS
    )
    fixed_gas_used = (
        intrinsic_execution + verify_gas_used + clearer_gas_used
    )
    # Position the pre-refund gas so that one fifth of it lands below,
    # exactly at, or above the refund counter.
    if quotient_position == "under":
        target_total = 5 * refund + 10_000
    elif quotient_position == "exact":
        # The smallest total whose fifth equals the refund exactly.
        target_total = 5 * refund
    else:
        target_total = 5 * refund - 5_000
    burner_gas = target_total - fixed_gas_used - (
        gas_costs.COLD_ACCOUNT_ACCESS
    )
    burner = pre.deploy_contract(code=burner_code(burner_gas))
    burner_gas_used = gas_costs.COLD_ACCOUNT_ACCESS + burner_gas

    gas_used_before_refund = fixed_gas_used + burner_gas_used
    assert gas_used_before_refund == target_total
    applied_refund = min(refund, gas_used_before_refund // 5)
    if quotient_position == "under":
        assert applied_refund == refund
    else:
        assert applied_refund == gas_used_before_refund // 5
    gas_used = gas_used_before_refund - applied_refund

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=PRIORITY_FEE,
        frames=[
            verify_frame(),
            default_frame(target=clearer, gas_limit=100_000),
            default_frame(target=burner, gas_limit=1_000_000),
        ],
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=gas_used,
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS, gas_used=verify_gas_used
                ),
                # The frame's gross gas keeps the clear cost; only the
                # transaction-level cumulative nets the refund.
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS, gas_used=clearer_gas_used
                ),
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS, gas_used=burner_gas_used
                ),
            ],
        ),
    )

    state_test(
        env=Environment(base_fee_per_gas=BASE_FEE),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=2),
            clearer: Account(storage={SLOT_CLEARED: 0}),
        },
    )


@pytest.mark.parametrize(
    "revert_shape",
    [
        pytest.param("frame_reverts", id="clearing_frame_reverts"),
        pytest.param("child_reverts", id="clearing_child_reverted"),
    ],
)
def test_refund_discarded_with_revert(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    revert_shape: str,
) -> None:
    """
    Discard a reverted frame's refund-counter contribution: a storage
    clear followed by a revert — in the frame itself or in a child
    call the frame rolls back — leaves the cumulative gas without any
    refund term, and the cleared slot restored.
    """
    clear_code = Op.SSTORE(
        key=SLOT_CLEARED,
        value=0,
        key_warm=False,
        original_value=1,
        current_value=1,
        new_value=0,
    )
    sender = pre.deploy_contract(code=APPROVE_ALL_CODE, balance=FUNDS)
    gas_costs = fork.gas_costs()

    if revert_shape == "frame_reverts":
        clearer_code = clear_code + Op.REVERT(0, 0)
        clearer = pre.deploy_contract(
            code=clearer_code, storage={SLOT_CLEARED: 1}
        )
        clearer_gas_used = (
            gas_costs.COLD_ACCOUNT_ACCESS + clearer_code.gas_cost(fork)
        )
        frame_status = Spec.STATUS_FAILURE
        frame_target = clearer
        restored = clearer
    else:
        child_code = clear_code + Op.STOP
        child = pre.deploy_contract(
            code=child_code, storage={SLOT_CLEARED: 1}
        )
        # Call the clearing child with ample gas, then revert the
        # frame, discarding the child's committed clear and refund.
        parent_code = (
            Op.POP(Op.CALL(Op.GAS, child, 0, 0, 0, 0, 0))
            + Op.REVERT(0, 0)
        )
        parent = pre.deploy_contract(code=parent_code)
        # The call's gas model already prices the cold access to the
        # child; only the frame entry access is added on top.
        clearer_gas_used = (
            gas_costs.COLD_ACCOUNT_ACCESS
            + parent_code.gas_cost(fork)
            + child_code.gas_cost(fork)
        )
        frame_status = Spec.STATUS_FAILURE
        frame_target = parent
        restored = child

    verify_gas_used = (
        gas_costs.WARM_ACCESS + APPROVE_ALL_CODE.gas_cost(fork)
    )
    intrinsic_execution, _ = gas_anchors(fork, [b"", b""], [], SCHEME_GAS)
    # No refund term: the revert discarded the counter contribution.
    gas_used = intrinsic_execution + verify_gas_used + clearer_gas_used

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE,
        max_priority_fee_per_gas=PRIORITY_FEE,
        frames=[
            verify_frame(),
            default_frame(target=frame_target, gas_limit=200_000),
        ],
        expected_receipt=TransactionReceipt(
            cumulative_gas_used=gas_used,
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS, gas_used=verify_gas_used
                ),
                FrameReceipt(
                    status=frame_status, gas_used=clearer_gas_used
                ),
            ],
        ),
    )

    state_test(
        env=Environment(base_fee_per_gas=BASE_FEE),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=2),
            restored: Account(storage={SLOT_CLEARED: 1}),
        },
    )
