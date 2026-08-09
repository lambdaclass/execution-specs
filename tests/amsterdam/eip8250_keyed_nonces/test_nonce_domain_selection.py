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
    Block,
    BlockchainTestFiller,
    EIPChecklist,
    Environment,
    Frame,
    FrameReceipt,
    Hash,
    StateTestFiller,
    Transaction,
    TransactionException,
    TransactionReceipt,
    keccak256,
)

from .helpers import approve_bytecode
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

# The narrowest and the widest key, so that the thirty-two-byte big-endian
# conversion of `nonce_key` is exercised at both ends of `uint256`.
DISCRIMINATED_KEYS = [1, 2**256 - 1]

# Pre-seeded storage on the contract sender. Its only job is to prove the
# sender's storage is really being compared: without it, an absent account
# would read zero at every slot and the "not on the sender" half of R-033
# would hold vacuously.
SENDER_SENTINEL_SLOT = 0xC0
SENDER_SENTINEL_VALUE = 0xC0DE
CONTRACT_SENDER_NONCE = 1


def wrong_layout_slots(sender: Address, nonce_key: int) -> Dict[str, Hash]:
    """
    Return the slots a misderived `slot(sender, nonce_key)` would occupy.

    R-034 fixes the preimage as `A || K` with `A = left_pad_32(sender)` and
    `K = uint256_to_bytes32(nonce_key)`. Each entry is one plausible
    misreading of that sentence: exchanging the two operands, padding the
    address on the wrong side, concatenating the twenty-byte address
    unpadded, treating the pair as a Solidity nested mapping rooted at slot
    zero, and skipping the hash altogether.
    """
    address_bytes = bytes(sender)
    padded_sender = address_bytes.rjust(32, b"\x00")
    key_bytes = nonce_key.to_bytes(32, byteorder="big")
    return {
        "swapped_operands": keccak256(key_bytes + padded_sender),
        "right_padded_sender": keccak256(
            address_bytes.ljust(32, b"\x00") + key_bytes
        ),
        "unpadded_sender": keccak256(address_bytes + key_bytes),
        "nested_mapping": keccak256(
            key_bytes + bytes(keccak256(padded_sender + bytes(32)))
        ),
        "unhashed_key": Hash(nonce_key),
    }


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


@EIPChecklist.TransactionType.Test.SenderAccount.Nonce()
def test_slot_preimage_order_discriminator(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-033 and R-034 positionally: the exact slot, in the exact account.

    `test_keyed_consumption_writes_only_manager_storage` above shows the
    write lands on `NONCE_MANAGER` rather than on the sender, and
    `test_vectors.py::test_storage_slot_hardcoded_vectors` pins
    `Spec.nonce_slot` against hard-coded Keccak results. Neither closes the
    gap between them: every assertion in this directory names its expected
    slot through `Spec.nonce_slot`, so an implementation that derived slots
    by some other rule would disagree with the whole suite in the same
    direction and be caught only by the value 1 going missing, never by
    where it went instead.

    This test names the alternatives. `wrong_layout_slots` enumerates five
    misderivations of R-034's `keccak256(left_pad_32(sender) ||
    uint256_to_bytes32(nonce_key))` preimage, and each of the resulting
    slots must read zero while the R-034 slot reads `nonce_seq + 1 == 1`.
    Both directions are asserted for the narrowest key, 1, and the widest,
    `2**256 - 1`, and the slots are checked for pairwise distinctness first,
    so no arm can be satisfied by two derivations landing on one slot.

    The sender is a contract carrying pre-seeded storage, which is what
    makes the R-033 half load-bearing. `slot(sender, nonce_key)` is a
    function of the sender alone, so an implementation keeping keyed
    sequences on the sender account would derive the same slot key; against
    the externally owned sender used elsewhere in this directory that is
    unobservable, because an EOA has no storage to inspect. Here both
    selected slots are asserted zero in the sender's own storage while the
    sentinel at slot `0xC0` still holds `0xC0DE`, so the comparison is
    against real storage and not against an absent account. The sender's
    account nonce staying at one is the keyed half of R-043: payment
    approval writes manager slots and nothing else.
    """
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION_AND_PAYMENT),
        storage={SENDER_SENTINEL_SLOT: SENDER_SENTINEL_VALUE},
        balance=10**19,
        nonce=CONTRACT_SENDER_NONCE,
    )

    selected_slots = [
        Spec.nonce_slot(sender, key) for key in DISCRIMINATED_KEYS
    ]
    rejected_slots = {
        f"{layout}_key_{index}": slot
        for index, key in enumerate(DISCRIMINATED_KEYS)
        for layout, slot in wrong_layout_slots(sender, key).items()
    }
    # A collision would make one of the two expectations below unreachable.
    assert len(set(selected_slots) | set(rejected_slots.values())) == len(
        selected_slots
    ) + len(rejected_slots)

    manager_storage: Dict[Hash, int] = dict.fromkeys(selected_slots, 1)
    manager_storage.update(dict.fromkeys(rejected_slots.values(), 0))
    sender_storage: Dict[Hash, int] = {
        Hash(SENDER_SENTINEL_SLOT): SENDER_SENTINEL_VALUE
    }
    sender_storage.update(dict.fromkeys(selected_slots, 0))

    tx = Transaction(
        sender=sender,
        nonce_keys=DISCRIMINATED_KEYS,
        nonce_seq=0,
        frames=[
            Frame(
                mode=Spec.MODE_DEFAULT,
                flags=Spec.APPROVE_EXECUTION_AND_PAYMENT,
                # Both keys are fresh, so the frame pays
                # `KEYED_NONCE_FIRST_USE_GAS` twice on top of its own
                # `APPROVE`; sized well above that rather than tuned to it,
                # because a starved frame would leave every slot zero and
                # satisfy the rejected-layout half by accident.
                gas_limit=300_000,
            )
        ],
        expected_receipt=TransactionReceipt(
            payer=sender,
            frame_receipts=[FrameReceipt(status=Spec.STATUS_SUCCESS)],
        ),
    )

    # Keyed by Address: the sender is a contract, the manager a plain address.
    post: Dict[Address, Account] = {
        sender: Account(nonce=CONTRACT_SENDER_NONCE, storage=sender_storage),
        Spec.NONCE_MANAGER: Account(storage=manager_storage),
    }
    state_test(env=Environment(), pre=pre, tx=tx, post=post)


@pytest.mark.pre_alloc_mutable
@EIPChecklist.TransactionType.Test.SenderAccount.Nonce()
def test_payment_approval_nonce_effect_follows_selected_domain(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Pin R-043: payment approval's nonce effect is dispatched on the domain.

    R-043 replaces EIP-8141's unconditional sender-nonce increment with
    `consume_nonce_set`, so post-fork the increment must happen for
    `nonce_keys == [0]` and must not happen otherwise. One sender exercises
    both halves in one block, which is what makes the replacement itself
    observable: the suite's other cross-domain tests give each domain its
    own sender, so nothing there can see one sender's two domains failing to
    interfere.

    Every expected number is re-derived from the spec pseudocode. The sender
    starts at account nonce 7 with `slot(sender, 6)` seeded to 3, three
    values kept pairwise distinct so no confusion of domains can produce the
    same state twice.

    - Transaction one selects `[6]` at `nonce_seq == 3`. `current_nonce_seq`
      takes the non-zero branch and reads 3 from the manager slot, so it is
      valid, and `consume_nonce_set` takes its `else` branch and stores
      `3 + 1 == 4`. The account nonce is not an output of that branch and
      stays 7.
    - Transaction two selects `[0]` at `nonce_seq == 7`. Its validity is the
      assertion that carries transaction one's non-increment: the key-zero
      branch of `current_nonce_seq` compares against `state[sender].nonce`
      at this block position, so `nonce_seq == 7` is accepted only if
      approval left the account nonce alone. `increment_account_nonce` then
      takes it to 8, and the key-zero branch writes no manager slot, so
      `slot(sender, 0)` must still read 0.

    A client that kept the EIP-8141 increment alongside keyed consumption
    would reach nonce 8 in transaction one and reject transaction two at 7;
    one that assigned `nonce_seq + 1` on the keyed branch would reach 4 and
    reject it likewise; one that consumed the keyed sequence out of the
    account nonce would leave the seeded slot at 3.

    Both frames are given zero gas, which is the arms' canary. The seeded
    slot is non-zero so `first_use_count` is 0 for transaction one, and
    `[0]` skips the first-use read and surcharge entirely for transaction
    two, so any charge at all exhausts the frame and fails the block
    instead of quietly reporting a no-op post-state.
    """
    legacy_nonce = 7
    seeded_sequence = 3
    nonce_key = 6

    sender = pre.fund_eoa(nonce=legacy_nonce)
    keyed_slot = Spec.nonce_slot(sender, nonce_key)
    zero_slot = Spec.nonce_slot(sender, 0)
    pre[Spec.NONCE_MANAGER] = Account(
        nonce=1,
        code=Spec.NONCE_MANAGER_CODE,
        storage={keyed_slot: seeded_sequence},
    )

    def approval_transaction(
        nonce_keys: list[int], nonce_seq: int
    ) -> Transaction:
        """Return a transaction whose only frame takes the approval."""
        return Transaction(
            sender=sender,
            nonce_keys=nonce_keys,
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
                    FrameReceipt(
                        status=Spec.STATUS_SUCCESS, gas_used=0, logs=[]
                    )
                ],
            ),
        )

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[
                    approval_transaction([nonce_key], seeded_sequence),
                    approval_transaction([0], legacy_nonce),
                ]
            )
        ],
        post={
            sender: Account(nonce=legacy_nonce + 1),
            Spec.NONCE_MANAGER: Account(
                storage={
                    keyed_slot: seeded_sequence + 1,
                    # The key-zero domain never touches manager storage.
                    zero_slot: 0,
                }
            ),
        },
    )
