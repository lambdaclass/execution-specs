"""
Nonce-domain selection tests for EIP-8250 keyed nonces.

The EIP defines two disjoint places a selected nonce domain can live:

```python
def current_nonce_seq(sender, nonce_key):
    if nonce_key == 0:
        return state[sender].nonce
    return uint256(state[NONCE_MANAGER].storage[slot(sender, nonce_key)])
```

Every other module in this directory exercises one branch at a time against
a state in which the other branch is empty, so an implementation that read
or wrote the wrong account would still agree with them. These tests seed
the branch that must be ignored with a value that disagrees, which is what
makes the choice of domain observable.
"""

from typing import Dict

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    EIPChecklist,
    Environment,
    Frame,
    FrameReceipt,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
)

from .spec import Spec, ref_spec_8250

REFERENCE_SPEC_GIT_PATH = ref_spec_8250.git_path
REFERENCE_SPEC_VERSION = ref_spec_8250.version

pytestmark = pytest.mark.valid_from("Bogota")

# The sender's legacy account nonce, and a different value planted at
# `slot(sender, 0)`. No conformant run can ever write that slot -- key zero
# never reaches `NONCE_MANAGER` -- so its only role is to disagree with the
# account nonce loudly enough that reading it instead is visible.
LEGACY_NONCE = 4
DECOY_SLOT_VALUE = 9


@pytest.mark.parametrize(
    "nonce_seq,error",
    [
        pytest.param(LEGACY_NONCE, None, id="sequence_matches_account_nonce"),
        pytest.param(
            DECOY_SLOT_VALUE,
            TransactionException.NONCE_MISMATCH_TOO_HIGH,
            id="sequence_matches_decoy_slot",
            marks=pytest.mark.exception_test,
        ),
    ],
)
@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.IntrinsicValidity.NonceExact()
def test_key_zero_domain_ignores_manager_slot_zero(
    state_test: StateTestFiller,
    pre: Alloc,
    nonce_seq: int,
    error: TransactionException | None,
) -> None:
    """
    Pin R-032 and the zero branch of R-044: for `nonce_keys == [0]` the
    selected domain is `state[sender].nonce`, never `slot(sender, 0)`.

    `slot(sender, 0)` is seeded with 9 while the sender's account nonce is 4.
    The two expected outcomes are read straight off `current_nonce_seq`, whose
    key-zero branch returns the account nonce: sequence 4 equals it and is
    valid, and sequence 9 is four above it and so is rejected as too high.
    The arms are each other's discriminator -- an implementation that hashed
    key zero into a manager slot like any other key would invert both
    verdicts -- and neither arm may disturb the seeded slot, because the
    consumption branch for `[0]` writes only the account nonce.

    The valid arm's frame is given zero gas. `[0]` skips the first-use read
    and surcharge entirely (steps 1 through 3 apply only when
    `nonce_keys != [0]`), so any charge at all would exhaust the frame.
    """
    sender = pre.fund_eoa(nonce=LEGACY_NONCE)
    decoy_slot = Spec.nonce_slot(sender, 0)
    pre[Spec.NONCE_MANAGER] = Account(
        nonce=1,
        code=Spec.NONCE_MANAGER_CODE,
        storage={decoy_slot: DECOY_SLOT_VALUE},
    )

    tx = Transaction(
        sender=sender,
        nonce_keys=[0],
        nonce_seq=nonce_seq,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=0,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(status=Spec.STATUS_SUCCESS, gas_used=0, logs=[])
            ],
        )
        if error is None
        else None,
        error=error,
    )

    expected_nonce = LEGACY_NONCE + 1 if error is None else LEGACY_NONCE
    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(nonce=expected_nonce),
            Spec.NONCE_MANAGER: Account(
                storage={decoy_slot: DECOY_SLOT_VALUE}
            ),
        },
    )


def test_keyed_consumption_writes_only_manager_storage(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin the non-zero branch of R-044: the consumed slot lives in
    `state[NONCE_MANAGER].storage`, not in the sender's own storage.

    `slot(sender, nonce_key)` is a function of the sender, so an
    implementation that kept keyed sequences on the sender account would
    derive the same slot key and satisfy every assertion the rest of this
    directory makes -- all of which look only at `NONCE_MANAGER`. The
    expectation here is the pair of statements the EIP makes about where the
    write lands: the manager slot holds `nonce_seq + 1 == 1`, and the sender
    holds nothing at that slot or any other, since a keyed transaction gives
    an externally owned sender no way to acquire storage.
    """
    nonce_key = 5
    sender = pre.fund_eoa()
    slot = Spec.nonce_slot(sender, nonce_key)

    tx = Transaction(
        sender=sender,
        nonce_keys=[nonce_key],
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_VERIFY,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                gas_limit=Spec.KEYED_NONCE_FIRST_USE_GAS,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[
                FrameReceipt(
                    status=Spec.STATUS_SUCCESS,
                    gas_used=Spec.KEYED_NONCE_FIRST_USE_GAS,
                    logs=[],
                )
            ],
        ),
    )

    # Keyed by Address: the sender is an EOA, the manager a plain address.
    post: Dict[Address, Account] = {
        sender: Account(nonce=0, storage={}),
        Spec.NONCE_MANAGER: Account(storage={slot: 1}),
    }
    state_test(env=Environment(), pre=pre, tx=tx, post=post)
