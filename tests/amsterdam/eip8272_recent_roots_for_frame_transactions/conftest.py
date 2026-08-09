"""Shared fixtures for the EIP-8272 recent root reference tests."""

from typing import List

import pytest
from execution_testing import (
    Address,
    Alloc,
    Bytecode,
    Environment,
    Frame,
    Hash,
    Op,
    RecentRootReference,
)

from .helpers import (
    CANARY_VALUE,
    EXECUTED_VALUE,
    SLOT_EXECUTED,
    compute_source_id,
    verify_and_sender_frames,
)
from .spec import CanonicalVector


@pytest.fixture
def current_slot() -> int:
    """
    Consensus slot of the block that executes the transaction.

    Large enough that the whole usable window lies at non-negative slots,
    so a test can name a reference `RECENT_ROOT_USABLE_WINDOW` slots back
    without clamping.
    """
    return 20_000


@pytest.fixture
def env(current_slot: int) -> Environment:
    """
    Block environment carrying the consensus slot number.

    EIP-8272 reads `current_slot` from EIP-7843's `slotNumber` header
    field, which the environment sets independently of the timestamp.
    """
    return Environment(slot_number=current_slot)


@pytest.fixture
def source_address() -> Address:
    """Address of the root source, an input to `source_id`."""
    return CanonicalVector.SOURCE_ADDRESS


@pytest.fixture
def salt() -> int:
    """Salt the root source writes under."""
    return CanonicalVector.SALT


@pytest.fixture
def source_id(source_address: Address, salt: int) -> Hash:
    """Identifier of the root source used by tests that do not vary it."""
    return compute_source_id(source_address, salt)


@pytest.fixture
def root() -> Hash:
    """
    Root the references name.

    Every byte is non-zero so that a reference's encoded form is
    distinguishable from a zero-filled one, and so that a `root` read
    back through introspection cannot be confused with an unset word.
    """
    return CanonicalVector.ROOT


@pytest.fixture
def reference_slot(current_slot: int) -> int:
    """Slot the reference names: the most recent referenceable one."""
    return current_slot - 1


@pytest.fixture
def references(
    source_id: Hash, reference_slot: int, root: Hash
) -> List[RecentRootReference]:
    """Reference set the transaction declares."""
    return [
        RecentRootReference(
            source_id=source_id, slot=reference_slot, root=root
        )
    ]


@pytest.fixture
def target_code() -> Bytecode:
    """Code of the contract the transaction's `SENDER` frame calls."""
    return Op.SSTORE(SLOT_EXECUTED, EXECUTED_VALUE) + Op.STOP


@pytest.fixture
def target(pre: Alloc, target_code: Bytecode) -> Address:
    """Contract the transaction's `SENDER` frame calls."""
    return pre.deploy_contract(
        code=target_code, storage={SLOT_EXECUTED: CANARY_VALUE}
    )


@pytest.fixture
def frames(target: Address) -> List[Frame]:
    """Frame list of the transaction under test."""
    return verify_and_sender_frames(target)
