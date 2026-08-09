"""
Frame transaction processing.

The block-level flow for [EIP-8141] frame transactions, separate from
the regular flow in `fork.py` from admission onwards: a frame
transaction has no single top-level call to dispatch — it executes a
list of frames — and no upfront sender payment: the sender's nonce
increment and the collection of the transaction's maximum cost are
effects of the `APPROVE` instruction, during execution.

[EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
"""

from typing import Set, Tuple

from ethereum_rlp import rlp
from ethereum_types.bytes import Bytes, Bytes32
from ethereum_types.numeric import U256, Uint

from ethereum.merkle_patricia_trie import trie_set
from ethereum.state import Address

from . import vm
from .blocks import (
    FrameReceipt,
    FrameTransactionReceipt,
    Receipt,
    encode_receipt,
)
from .exceptions import (
    InvalidRecentRootReferenceError,
    MaxCostOverflowError,
)
from .fork_types import ExecutionGas, StateGas
from .state_tracker import (
    TransactionState,
    clear_account_preserving_balance,
    create_ether,
    get_account,
    get_storage_original,
    incorporate_tx_into_block,
)
from .transactions import (
    calculate_effective_gas_price,
    check_nonce,
    encode_transaction,
    get_transaction_hash,
)
from .transactions.frame_transaction import (
    RECENT_ROOT_ADDRESS,
    RECENT_ROOT_USABLE_WINDOW,
    FrameTransaction,
    compute_recent_root_entry_hash,
    compute_recent_root_storage_key,
    validate_frame_transaction,
)
from .vm.frame_interpreter import process_frames
from .vm.gas import (
    TransactionGasSettlement,
    calculate_blob_gas_price,
    calculate_total_blob_gas,
    check_block_gas_capacity,
    check_max_fee_per_blob_gas,
    settle_transaction_gas,
)


def check_recent_root_references(
    block_env: vm.BlockEnvironment,
    tx_state: TransactionState,
    tx: FrameTransaction,
) -> Set[Tuple[Address, Bytes32]]:
    """
    Check the recent root references a frame transaction declares, and
    return the storage keys they name.

    A reference is satisfied when the root source it names stored, for
    the slot it names, an entry committing to that same source, slot, and
    root. The named slot must also be over — a block cannot reference a
    root written in its own slot, which is precisely what keeps such a
    write from displacing an entry the block still relies on — and recent
    enough that the source has not yet overwritten its entry.

    An unsatisfied reference invalidates the transaction, and with it any
    block including it; no frame executes. The block's slot number comes
    from the consensus layer, and never from its timestamp: two blocks
    proposed for different slots may reference different roots even when
    their timestamps agree.

    The entries are read from the transaction's pre-state, which is the
    state left by the preceding transactions of the block. Reading it
    leaves the transaction's own state accesses untouched; the references
    bear on execution only through the warm and cold status of what the
    frames go on to touch.

    Duplicate references are checked, and charged for, one by one, but
    they name one storage key between them — which is why the returned
    set may be smaller than the declared list.
    """
    storage_keys: Set[Tuple[Address, Bytes32]] = set()
    for reference in tx.recent_root_references:
        if (
            reference.slot >= block_env.slot_number
            or block_env.slot_number - reference.slot
            > RECENT_ROOT_USABLE_WINDOW
        ):
            raise InvalidRecentRootReferenceError(
                f"recent root reference to slot {reference.slot} is not "
                f"referenceable in slot {block_env.slot_number}"
            )

        storage_key = compute_recent_root_storage_key(
            reference.source_id, reference.slot
        )
        stored_entry = get_storage_original(
            tx_state, RECENT_ROOT_ADDRESS, storage_key
        )
        entry_hash = compute_recent_root_entry_hash(reference)
        if stored_entry != U256.from_be_bytes(entry_hash):
            raise InvalidRecentRootReferenceError(
                f"recent root reference to slot {reference.slot} names a "
                "root its source did not write in that slot"
            )

        storage_keys.add((RECENT_ROOT_ADDRESS, storage_key))

    return storage_keys


def check_frame_transaction(
    block_env: vm.BlockEnvironment,
    block_output: vm.BlockOutput,
    tx: FrameTransaction,
    index: Uint,
) -> vm.TransactionEnvironment:
    """
    Admit a raw frame transaction and build its execution environment.

    Statically validate the transaction and check that it is includable
    in the block, in that order, so that a transaction invalid in
    several ways reports the earliest failure.

    Unlike the regular flow, the sender needs no recovery — it is an
    explicit field, authenticated by the signature entries during
    static validation — and no balance or EOA check: payment is
    collected from the payer during execution, when a frame `APPROVE`s
    it.

    Parameters
    ----------
    block_env :
        The block scoped environment.
    block_output :
        The block output for the current block.
    tx :
        The frame transaction.
    index :
        The index of the current transaction.

    Returns
    -------
    tx_env :
        The environment for executing the transaction.

    Raises
    ------
    InvalidBlock :
        If the transaction is not includable.
    InvalidSignatureError :
        If a signature entry is cryptographically invalid.
    InvalidFrameError :
        If the frames or signature entries violate a structural
        constraint.
    TransactionGasLimitExceededError :
        If the derived gas limit exceeds the maximum allowed for a
        transaction.
    GasUsedExceedsLimitError :
        If the gas used by the transaction exceeds the block's gas limit.
    NonceMismatchError :
        If the nonce of the transaction is not equal to the sender's nonce.
    InvalidRecentRootReferenceError :
        If a declared recent root reference is not satisfied by the
        transaction's pre-state.
    InsufficientMaxFeePerGasError :
        If the maximum fee per gas is insufficient for the transaction.
    InsufficientMaxFeePerBlobGasError :
        If the maximum fee per blob gas is insufficient for the transaction.
    BlobGasLimitExceededError :
        If the blob gas used by the transaction exceeds the block's blob gas
        limit.
    MaxCostOverflowError :
        If the maximum wei cost the transaction can incur is not
        representable in 256 bits.

    """
    validation = validate_frame_transaction(tx)
    tx_state = TransactionState(parent=block_env.state)

    check_block_gas_capacity(
        block_env,
        block_output,
        validation.max_gas,
        calculate_total_blob_gas(tx),
    )

    sender_account = get_account(tx_state, tx.sender)

    effective_gas_price = calculate_effective_gas_price(
        tx, block_env.base_fee_per_gas
    )

    check_max_fee_per_blob_gas(
        tx.blob_versioned_hashes,
        tx.max_fee_per_blob_gas,
        block_env.excess_blob_gas,
    )

    check_nonce(tx, sender_account.nonce)

    recent_root_storage_keys = check_recent_root_references(
        block_env, tx_state, tx
    )

    # A state gas reservoir holds only gas above `TX_MAX_GAS_LIMIT`,
    # and the derived `max_gas` never exceeds that cap: a frame
    # transaction's reservoir is always empty, and state gas spills
    # from execution gas instead.
    execution_gas_grant = validation.standard_gas_limit - Uint(
        validation.intrinsic.execution
    )

    max_cost = validation.max_gas * tx.max_fee_per_gas + Uint(
        calculate_total_blob_gas(tx)
    ) * calculate_blob_gas_price(block_env.excess_blob_gas)
    if max_cost > Uint(U256.MAX_VALUE):
        raise MaxCostOverflowError("Max cost too high")

    return vm.TransactionEnvironment(
        origin=tx.sender,
        gas_limit=validation.max_gas,
        effective_gas_price=effective_gas_price,
        execution_gas_grant=ExecutionGas(execution_gas_grant),
        state_gas_reservoir=StateGas(Uint(0)),
        calldata_floor=validation.intrinsic.calldata_floor,
        access_list_addresses=(
            {RECENT_ROOT_ADDRESS} if recent_root_storage_keys else set()
        ),
        access_list_storage_keys=recent_root_storage_keys,
        accounts_with_paid_writes={tx.sender},
        state=tx_state,
        blob_versioned_hashes=tx.blob_versioned_hashes,
        authorizations=(),
        index_in_block=index,
        tx_hash=get_transaction_hash(encode_transaction(tx)),
        top_level_context=None,
        frame_context=vm.FrameContext(
            tx=tx,
            signature_hash=validation.signature_hash,
            resolved_signers=validation.resolved_signers,
            standard_gas_limit=validation.standard_gas_limit,
            max_cost=max_cost,
            current_frame_index=Uint(0),
            frame_receipts=[],
            payer=None,
            sender_approved=False,
        ),
    )


def disburse_frame_gas_fees(
    block_env: vm.BlockEnvironment,
    tx_env: vm.TransactionEnvironment,
    settlement: TransactionGasSettlement,
) -> None:
    """
    Refund the payer's unspent escrow and pay the priority fee.

    At `APPROVE` the payer escrowed the transaction's maximum cost,
    priced at the maximum fee per gas; the refund is that escrow less
    the charged fee — the gas used priced at the effective gas price,
    plus the blob gas fee, which appears in both terms and cancels.
    Refunding the unused gas at the effective gas price, as the
    regular flow does, would under-refund the escrowed difference
    between the two prices.

    The coinbase's priority fee on the gas used is unchanged from the
    regular flow.
    """
    frame_context = tx_env.frame_context
    assert frame_context is not None
    # `process_frames` invalidates the transaction unless a frame
    # approved payment.
    payer = frame_context.payer
    assert payer is not None

    blob_gas_fee = Uint(
        calculate_total_blob_gas(frame_context.tx)
    ) * calculate_blob_gas_price(block_env.excess_blob_gas)
    charged_fee = (
        settlement.gas_used * tx_env.effective_gas_price + blob_gas_fee
    )
    payer_refund = frame_context.max_cost - charged_fee

    priority_fee_per_gas = (
        tx_env.effective_gas_price - block_env.base_fee_per_gas
    )
    transaction_fee = settlement.gas_used * priority_fee_per_gas

    create_ether(tx_env.state, payer, U256(payer_refund))
    create_ether(tx_env.state, block_env.coinbase, U256(transaction_fee))


def make_frame_receipt(
    tx: FrameTransaction,
    payer: Address,
    cumulative_gas_used: Uint,
    frame_receipts: Tuple[FrameReceipt, ...],
) -> Bytes | Receipt:
    """
    Make the receipt for a frame transaction that was executed.

    Unlike a regular receipt there is no transaction-level status and
    no bloom filter: outcomes are reported per frame, and the frames'
    logs reach the block's log bloom through the block accumulator.
    """
    receipt = FrameTransactionReceipt(
        cumulative_gas_used=cumulative_gas_used,
        payer=payer,
        frame_receipts=frame_receipts,
    )

    return encode_receipt(tx, receipt)


def process_frame_transaction(
    block_env: vm.BlockEnvironment,
    block_output: vm.BlockOutput,
    tx: FrameTransaction,
    index: Uint,
) -> None:
    """
    Execute a frame transaction against the provided environment.

    Admit the transaction, execute its frames in order, settle the gas,
    disburse the fees, and write the frame transaction receipt.
    """
    tx_env = check_frame_transaction(block_env, block_output, tx, index)

    tx_output = process_frames(block_env, tx_env)

    frame_context = tx_env.frame_context
    assert frame_context is not None
    # `process_frames` invalidates the transaction unless a frame
    # approved payment.
    payer = frame_context.payer
    assert payer is not None

    # Settlement anchors on the standard gas limit; the floor headroom
    # above it — nonzero only when the calldata floor exceeds the
    # standard gas limit — is never executable and counts as unused
    # gas, so the floor can bind without an underflow in the gas
    # accounting.
    floor_headroom = tx_env.gas_limit - frame_context.standard_gas_limit
    settlement = settle_transaction_gas(
        tx_env.gas_limit,
        tx_env.calldata_floor,
        ExecutionGas(tx_output.gas_left + floor_headroom),
        tx_output.state_gas_left,
        tx_output.refund_counter,
        tx_output.state_gas_used,
    )

    disburse_frame_gas_fees(block_env, tx_env, settlement)

    block_output.block_gas_used += settlement.execution_gas_used
    block_output.block_state_gas_used += settlement.state_gas_used
    block_output.blob_gas_used += calculate_total_blob_gas(tx)

    block_output.cumulative_gas_used += settlement.gas_used
    receipt = make_frame_receipt(
        tx,
        payer,
        block_output.cumulative_gas_used,
        tuple(frame_context.frame_receipts),
    )

    receipt_key = rlp.encode(Uint(index))
    block_output.receipt_keys += (receipt_key,)

    trie_set(
        block_output.receipts_trie,
        receipt_key,
        receipt,
    )

    block_output.block_logs += tx_output.logs

    for address in tx_output.accounts_to_delete:
        clear_account_preserving_balance(tx_env.state, address)

    incorporate_tx_into_block(
        tx_env.state, block_env.block_access_list_builder
    )
