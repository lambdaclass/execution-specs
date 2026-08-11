"""
Implementations of the EVM instructions that expose the transaction's
state diff to post-transaction assertion frames, introduced in
[EIP-7906].

The instructions in this module are defined only inside a
[`POST_TX`][pt] frame of an [EIP-8141] frame transaction — including
any call made from within such a frame — and executing any of them in
any other context results in an exceptional halt.

[EIP-7906]: https://eips.ethereum.org/EIPS/eip-7906
[EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
[pt]: ref:ethereum.forks.amsterdam.transactions.frame_transaction.FrameMode.POST_TX
"""  # noqa: E501

from typing import Tuple

from ethereum_types.bytes import Bytes
from ethereum_types.numeric import U256, Uint, ulen

from ethereum.state import Address
from ethereum.utils.numeric import ceil32

from ...blocks import Log
from ...fork_types import ExecutionGas
from ...state_tracker import (
    TransactionState,
    get_account,
    get_pre_state_account,
    get_pre_state_storage,
    get_storage,
    transaction_account_change_flags,
    transaction_balance_changes,
    transaction_deployed_contracts,
    transaction_storage_changes,
)
from ...transactions.frame_transaction import FrameMode
from ...utils.address import to_address_masked
from ...vm.memory import memory_write
from .. import Evm, FrameContext
from ..exceptions import InvalidParameter
from ..gas import GasCosts, calculate_gas_extend_memory, charge_gas
from ..stack import pop, push


def post_transaction_frame_context(evm: Evm) -> FrameContext:
    """
    Return the executing frame transaction's context, or exceptionally
    halt when the executing top-level frame is not a `POST_TX` frame —
    including when the current transaction is not a frame transaction
    at all.

    The gate keys on the mode of the enclosing top-level frame, so any
    contract called from within a `POST_TX` frame's call subtree may
    execute the state-diff instructions.
    """
    frame_context = evm.tx_env.frame_context
    if frame_context is None:
        raise InvalidParameter("not a frame transaction")
    tx = frame_context.tx
    frame = tx.frames[int(frame_context.current_frame_index)]
    if frame.mode != FrameMode.POST_TX:
        raise InvalidParameter("not a post-transaction frame")
    return frame_context


def transaction_events(frame_context: FrameContext) -> Tuple[Log, ...]:
    """
    Return every event the transaction has emitted, in emission order —
    each event's position is its global log index within the
    transaction.

    Completed frames report their logs through their receipts, and the
    receipts of unrolled or reverted frames have had their logs
    emptied, so the concatenation is exactly the transaction's live
    log set. The executing `POST_TX` frame is static and can add none.
    """
    events: Tuple[Log, ...] = ()
    for receipt in frame_context.frame_receipts:
        events += receipt.logs
    return events


def event_view(events: Tuple[Log, ...], address: Address) -> Tuple[Uint, ...]:
    """
    Return the per-address view over the event enumeration: the global
    indexes of the events the address emitted, in emission order.
    """
    return tuple(
        Uint(index)
        for index, event in enumerate(events)
        if event.address == address
    )


def storage_change_view(
    tx_state: TransactionState, address: Address
) -> Tuple[Uint, ...]:
    """
    Return the per-address view over the storage change enumeration:
    the global indexes of the entries whose account matches, in
    enumeration order.
    """
    return tuple(
        Uint(index)
        for index, change in enumerate(transaction_storage_changes(tx_state))
        if change.address == address
    )


def check_reserved_input(value: U256) -> None:
    """
    Exceptionally halt when an input the parameter tables mark as
    *must be 0* carries a non-zero value.
    """
    if value != U256(0):
        raise InvalidParameter("reserved input must be zero")


def txtrace(evm: Evm) -> None:
    """
    Push one parameter of the transaction's collapsed state diff onto
    the stack, selected by the parameter operand.

    The diff compares the transaction prestate with the state as of
    this call: intermediary writes are not observable, and entries
    whose value was restored do not appear. Balance and storage
    changes are enumerated ascending by address (and storage key),
    contract deployments ascending by address, and events in emission
    order. An indexed parameter exceptionally halts when its index is
    out of bounds, as does a topic parameter at or beyond the event's
    topic count. The diff is answered from data the transaction has
    already recorded, adding no new state accesses.
    """
    # STACK
    index = pop(evm.stack)
    param = pop(evm.stack)

    # GAS
    charge_gas(evm, GasCosts.OPCODE_TXTRACE)

    # OPERATION
    frame_context = post_transaction_frame_context(evm)
    tx_state = evm.tx_env.state

    if param == U256(0x00):
        check_reserved_input(index)
        value = U256(len(transaction_balance_changes(tx_state)))
    elif param == U256(0x01):
        check_reserved_input(index)
        value = U256(len(transaction_storage_changes(tx_state)))
    elif param == U256(0x02):
        check_reserved_input(index)
        value = U256(len(transaction_deployed_contracts(tx_state)))
    elif param <= U256(0x05):
        balance_changes = transaction_balance_changes(tx_state)
        if index >= U256(len(balance_changes)):
            raise InvalidParameter("balance change index out of bounds")
        balance_change = balance_changes[int(index)]
        if param == U256(0x03):
            value = U256.from_be_bytes(balance_change.address)
        elif param == U256(0x04):
            value = balance_change.balance_before
        else:
            value = balance_change.balance_after
    elif param <= U256(0x09):
        storage_changes = transaction_storage_changes(tx_state)
        if index >= U256(len(storage_changes)):
            raise InvalidParameter("storage change index out of bounds")
        storage_change = storage_changes[int(index)]
        if param == U256(0x06):
            value = U256.from_be_bytes(storage_change.address)
        elif param == U256(0x07):
            value = U256.from_be_bytes(storage_change.key)
        elif param == U256(0x08):
            value = storage_change.value_before
        else:
            value = storage_change.value_after
    elif param <= U256(0x0B):
        deployed_contracts = transaction_deployed_contracts(tx_state)
        if index >= U256(len(deployed_contracts)):
            raise InvalidParameter("deployed contract index out of bounds")
        deployed_contract = deployed_contracts[int(index)]
        if param == U256(0x0A):
            value = U256.from_be_bytes(deployed_contract.address)
        else:
            value = U256.from_be_bytes(deployed_contract.code_hash_after)
    elif param == U256(0x0C):
        check_reserved_input(index)
        value = U256(len(transaction_events(frame_context)))
    elif param <= U256(0x13):
        events = transaction_events(frame_context)
        if index >= U256(len(events)):
            raise InvalidParameter("event index out of bounds")
        event = events[int(index)]
        if param == U256(0x0D):
            value = U256.from_be_bytes(event.address)
        elif param == U256(0x0E):
            value = U256(len(event.topics))
        elif param <= U256(0x12):
            topic_index = int(param) - 0x0F
            if topic_index >= len(event.topics):
                raise InvalidParameter("event topic index out of bounds")
            value = U256.from_be_bytes(event.topics[topic_index])
        else:
            value = U256(len(event.data))
    elif param == U256(0x14):
        check_reserved_input(index)
        if frame_context.payer is None:
            value = U256(0)
        else:
            value = U256(frame_context.max_cost)
    elif param == U256(0x15):
        check_reserved_input(index)
        if frame_context.payer is None:
            value = U256(0)
        else:
            value = U256.from_be_bytes(frame_context.payer)
    else:
        raise InvalidParameter("undefined TXTRACE parameter")

    push(evm.stack, value)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def txdiff(evm: Evm) -> None:
    """
    Push one keyed lookup into the transaction's state diff onto the
    stack, selected by the parameter operand and keyed by the address
    operand — and, for the storage slot parameters, the slot key.

    The parameters that may fall back to reading live state — the
    before and after values of a storage slot, balance, or code hash —
    are priced with the warm and cold access costs, and record the
    access like any other state-reading instruction; when the queried
    key was never modified, both the before and the after variants
    return the current live value. The per-address view and flags
    parameters are answered entirely from the transaction-local diff
    at a flat cost and touch no access sets.
    """
    # STACK
    in3 = pop(evm.stack)
    address = to_address_masked(pop(evm.stack))
    param = pop(evm.stack)

    # GAS
    if param <= U256(0x01):
        if (address, in3.to_be_bytes32()) in evm.accessed_storage_keys:
            charge_gas(evm, GasCosts.WARM_ACCESS)
        else:
            evm.accessed_storage_keys.add((address, in3.to_be_bytes32()))
            charge_gas(evm, GasCosts.COLD_STORAGE_ACCESS)
    elif param <= U256(0x05):
        if address in evm.accessed_addresses:
            charge_gas(evm, GasCosts.WARM_ACCESS)
        else:
            evm.accessed_addresses.add(address)
            charge_gas(evm, GasCosts.COLD_ACCOUNT_ACCESS)
    else:
        charge_gas(evm, GasCosts.OPCODE_TXTRACE)

    # OPERATION
    frame_context = post_transaction_frame_context(evm)
    tx_state = evm.tx_env.state

    if param == U256(0x00):
        value = get_pre_state_storage(tx_state, address, in3.to_be_bytes32())
    elif param == U256(0x01):
        value = get_storage(tx_state, address, in3.to_be_bytes32())
    elif param == U256(0x02):
        check_reserved_input(in3)
        value = get_pre_state_account(tx_state, address).balance
    elif param == U256(0x03):
        check_reserved_input(in3)
        value = get_account(tx_state, address).balance
    elif param == U256(0x04):
        check_reserved_input(in3)
        code_hash = get_pre_state_account(tx_state, address).code_hash
        value = U256.from_be_bytes(code_hash)
    elif param == U256(0x05):
        check_reserved_input(in3)
        value = U256.from_be_bytes(get_account(tx_state, address).code_hash)
    elif param == U256(0x06):
        check_reserved_input(in3)
        value = U256(len(storage_change_view(tx_state, address)))
    elif param == U256(0x07):
        view = storage_change_view(tx_state, address)
        if in3 >= U256(len(view)):
            raise InvalidParameter("storage view index out of bounds")
        value = U256(view[int(in3)])
    elif param == U256(0x08):
        check_reserved_input(in3)
        events = transaction_events(frame_context)
        value = U256(len(event_view(events, address)))
    elif param == U256(0x09):
        view = event_view(transaction_events(frame_context), address)
        if in3 >= U256(len(view)):
            raise InvalidParameter("event view index out of bounds")
        value = U256(view[int(in3)])
    elif param == U256(0x0A):
        check_reserved_input(in3)
        value = U256(transaction_account_change_flags(tx_state, address))
    else:
        raise InvalidParameter("undefined TXDIFF parameter")

    push(evm.stack, value)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def eventdatacopy(evm: Evm) -> None:
    """
    Copy a portion of the chosen event's non-indexed data to memory.

    The gas matches `CALLDATACOPY` — a fixed cost plus the copy and
    memory expansion costs — but the bounds do not: a read reaching
    past the end of the event's data exceptionally halts instead of
    copying zeroes, as does an event index at or beyond the event
    count.
    """
    # STACK
    event_index = pop(evm.stack)
    memory_offset = pop(evm.stack)
    data_offset = pop(evm.stack)
    length = pop(evm.stack)

    # GAS
    words = ceil32(Uint(length)) // Uint(32)
    copy_gas_cost = ExecutionGas(GasCosts.OPCODE_COPY_PER_WORD * words)
    extend_memory = calculate_gas_extend_memory(
        evm.memory, [(memory_offset, length)]
    )
    charge_gas(
        evm,
        GasCosts.OPCODE_EVENTDATACOPY_BASE
        + copy_gas_cost
        + extend_memory.cost,
    )

    # OPERATION
    frame_context = post_transaction_frame_context(evm)
    events = transaction_events(frame_context)
    if event_index >= U256(len(events)):
        raise InvalidParameter("event index out of bounds")
    data = events[int(event_index)].data
    if Uint(data_offset) + Uint(length) > ulen(data):
        raise InvalidParameter("event data read out of bounds")

    evm.memory += b"\x00" * extend_memory.expand_by
    value = Bytes(data[int(data_offset) : int(data_offset) + int(length)])
    memory_write(evm.memory, memory_offset, value)

    # PROGRAM COUNTER
    evm.pc += Uint(1)
