"""
Block-level access list composition tests for
[EIP-7906: Transaction Assertions via State Diff
Opcode](https://eips.ethereum.org/EIPS/eip-7906).

The TXDIFF parameters that may fall back to live state record their
access like any other state-reading instruction, so the block-level
access list of [EIP-7928] must carry the slots they read. The
parameters answered from the transaction-local diff record nothing,
so an account named only by those must not appear in it.

[EIP-7928]: https://eips.ethereum.org/EIPS/eip-7928
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    BalAccountExpectation,
    Block,
    BlockAccessListExpectation,
    BlockchainTestFiller,
    Op,
    Transaction,
)

from tests.amsterdam.eip8141_frame_transactions.helpers import (
    default_frame,
    verify_frame,
)

from .helpers import (
    BODY_FRAME_GAS,
    SENTINEL_MARKER,
    SENTINEL_SLOT,
    assert_eq,
    deploy_sentinel,
    post_tx_frame,
)
from .spec import Spec, ref_spec_7906

REFERENCE_SPEC_GIT_PATH = ref_spec_7906.git_path
REFERENCE_SPEC_VERSION = ref_spec_7906.version

# EIP-7906 extends EIP-8141, which is slated for the fork after
# Amsterdam, so fixtures are labeled with the pseudo `Bogota` fork.
# Fill these tests with `--fork Bogota`.
pytestmark = pytest.mark.valid_from("Bogota")

READ_SLOT = 0x21
"""Storage key the assertion frame looks up and nothing writes."""

READ_VALUE = 0x99
"""Value stored at `READ_SLOT` before the block."""


def test_txdiff_records_in_block_access_list(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Read one account's slot through TXDIFF and another account's change
    flags, then check the block-level access list: the slot appears as
    a read of the first account, while the second account — named only
    by a parameter served from the transaction-local diff — records
    nothing at all.

    Rule: R-081.
    """
    sentinel = deploy_sentinel(pre)
    read_subject = pre.deploy_contract(
        code=Op.STOP, storage={READ_SLOT: READ_VALUE}
    )
    flat_subject = pre.deploy_contract(code=Op.STOP)

    asserter = pre.deploy_contract(
        code=assert_eq(
            Op.TXDIFF(READ_SLOT, read_subject, Spec.TXDIFF_SLOT_VALUE_BEFORE),
            READ_VALUE,
        )
        + assert_eq(
            Op.TXDIFF(0, flat_subject, Spec.TXDIFF_ACCOUNT_CHANGE_FLAGS), 0
        )
        + Op.STOP
    )

    tx = Transaction(
        sender=pre.fund_eoa(),
        frames=[
            verify_frame(),
            default_frame(target=sentinel, gas_limit=BODY_FRAME_GAS),
            post_tx_frame(target=asserter),
        ],
    )

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations={
                        read_subject: BalAccountExpectation(
                            storage_reads=[READ_SLOT]
                        ),
                        # Named only by a flat parameter, so it must
                        # be absent from the list entirely.
                        flat_subject: None,
                    }
                ),
            )
        ],
        post={sentinel: Account(storage={SENTINEL_SLOT: SENTINEL_MARKER})},
    )
