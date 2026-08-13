"""
Blob composition tests for
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

When the transaction carries blobs, the `gas_pre_charge` parameter
reports the blob fees on top of the execution escrow:
`gas_limit * gas_price + blob_count * GAS_PER_BLOB * blob_base_fee`.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    EIPChecklist,
    Fork,
    FrameReceipt,
    Hash,
    Op,
    StateTestFiller,
    Transaction,
    TransactionReceipt,
    add_kzg_version,
)

from tests.amsterdam.eip8141_frame_transactions.helpers import verify_frame
from tests.amsterdam.eip8141_frame_transactions.spec import Spec as FrameSpec

from .helpers import (
    assert_eq,
    contract_sender_max_cost,
    deploy_approving_sender,
    post_tx_frame,
)
from .spec import Spec, ref_spec_7906

REFERENCE_SPEC_GIT_PATH = ref_spec_7906.git_path
REFERENCE_SPEC_VERSION = ref_spec_7906.version

# EIP-7906 extends EIP-8141, which is slated for the fork after
# Amsterdam, so fixtures are labeled with the pseudo `Bogota` fork.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

MAX_FEE_PER_GAS = 10
"""Fee the execution part of the escrow is priced at."""

BLOB_COMMITMENT_VERSION_KZG = 1
"""Version byte marking a versioned hash as a KZG commitment."""

MINIMUM_BLOB_BASE_FEE = 1
"""
The blob base fee this test relies on, and the maximum blob fee the
transaction offers.

Capping the offer at the minimum makes the transaction invalid — and
the fill loud — if the block's blob base fee were anything else, so
the escrow's blob term cannot be confused with a larger one. The
assertion frame reads `BLOBBASEFEE` back and checks it as well.
"""


@EIPChecklist.Opcode.Test.ExecutionContext.BlockContext()
@pytest.mark.parametrize(
    "blob_count",
    [
        pytest.param(1, id="one_blob"),
        pytest.param(3, id="three_blobs"),
    ],
)
def test_gas_pre_charge_with_blobs(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    blob_count: int,
) -> None:
    """
    Attach blobs to an assertion transaction and check the pre-charge
    the diff reports: the execution escrow plus one blob fee per blob.

    The blob term is derived from the spec's formula with the blob gas
    per blob the fork defines; varying the blob count moves it, so a
    pre-charge that ignores blobs cannot pass both arms.

    Rule: R-043.
    """
    sender = deploy_approving_sender(pre, balance=10**18)

    frames = [verify_frame(), post_tx_frame()]
    blob_fee = blob_count * fork.blob_gas_per_blob() * MINIMUM_BLOB_BASE_FEE
    max_cost = contract_sender_max_cost(frames, MAX_FEE_PER_GAS) + blob_fee

    asserter = pre.deploy_contract(
        code=assert_eq(Op.BLOBBASEFEE, MINIMUM_BLOB_BASE_FEE)
        + assert_eq(Op.TXTRACE(0, Spec.TXTRACE_GAS_PRE_CHARGE), max_cost)
        + Op.STOP
    )
    frames[-1] = post_tx_frame(target=asserter)

    tx = Transaction(
        sender=sender,
        nonce=1,
        max_fee_per_gas=MAX_FEE_PER_GAS,
        max_priority_fee_per_gas=0,
        max_fee_per_blob_gas=MINIMUM_BLOB_BASE_FEE,
        blob_versioned_hashes=add_kzg_version(
            [Hash(index) for index in range(blob_count)],
            BLOB_COMMITMENT_VERSION_KZG,
        ),
        frames=frames,
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
                FrameReceipt(status=FrameSpec.STATUS_SUCCESS),
            ],
        ),
    )

    state_test(pre=pre, tx=tx, post={sender: Account(nonce=2)})
