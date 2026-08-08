"""Unit vectors for EIP-8250 transaction encoding and helpers."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from ethereum.forks.amsterdam.fork import BlockChain, apply_fork
from ethereum.forks.amsterdam.frame_processing import check_frame_transaction
from ethereum.forks.amsterdam.transactions import decode_transaction
from ethereum.forks.amsterdam.transactions.frame_transaction import (
    NONCE_MANAGER,
    NONCE_MANAGER_CODE,
    FrameTransaction,
    calculate_frame_transaction_intrinsic_cost,
    compute_frame_signature_hash,
    nonce_calldata,
    nonce_keys_hash,
    nonce_manager_slot,
)
from ethereum.forks.amsterdam.vm import attempt_approval
from ethereum.forks.amsterdam.vm.exceptions import OutOfGasError
from ethereum.forks.amsterdam.vm.gas import GasMeter
from ethereum.state import EMPTY_CODE_HASH
from ethereum.state import Account as SpecAccount
from ethereum.state_mpt import State, set_account
from ethereum_rlp import rlp
from ethereum_rlp.exceptions import DecodingError
from ethereum_types.bytes import Bytes as SpecBytes
from ethereum_types.bytes import Bytes32
from ethereum_types.numeric import U64, U256, Uint

from execution_testing.base_types import Hash
from execution_testing.forks import Amsterdam, Bogota
from execution_testing.forks.forks.eips import EIP8250

from ..account_types import EOA
from ..transaction_types import Frame, Transaction


def frame_transaction(
    nonce_keys: list[int], nonce_seq: int
) -> FrameTransaction:
    """Build and decode a minimal signature-free frame transaction."""
    tx = Transaction(
        sender=EOA(key=Hash(3)),
        nonce_keys=nonce_keys,
        nonce_seq=nonce_seq,
        frames=[Frame()],
        signatures=[],
    )
    decoded = decode_transaction(SpecBytes(tx.rlp()))
    assert isinstance(decoded, FrameTransaction)
    return decoded


def test_bogota_enables_eip8250_after_eip8141() -> None:
    """The testing fork composes both EIPs and installs the nonce manager."""
    assert Bogota.is_eip_enabled(8141, 8250)
    manager = Bogota.pre_allocation()[int.from_bytes(NONCE_MANAGER, "big")]
    assert manager == {"nonce": 1, "code": NONCE_MANAGER_CODE}


@pytest.mark.parametrize(
    "nonce_keys,nonce_seq,expected_calldata,expected_execution,expected_floor",
    [
        pytest.param(
            [0],
            0,
            "c18080",
            15_523,
            15_595,
            id="legacy_key",
            marks=pytest.mark.xfail(
                reason="implementation uses EIP-7976 uniform floor weighting"
            ),
        ),
        pytest.param(
            [1, 2],
            0,
            "c2010280",
            15_539,
            15_635,
            id="two_keys",
            marks=pytest.mark.xfail(
                reason="implementation uses EIP-7976 uniform floor weighting"
            ),
        ),
        pytest.param(
            [256],
            1,
            "c382010001",
            15_543,
            15_645,
            id="zero_byte",
            marks=pytest.mark.xfail(
                reason="implementation uses EIP-7976 uniform floor weighting"
            ),
        ),
    ],
)
def test_nonce_calldata_and_intrinsic_vectors(
    nonce_keys: list[int],
    nonce_seq: int,
    expected_calldata: str,
    expected_execution: int,
    expected_floor: int,
) -> None:
    """
    Pin R-029--R-039 using production gas calculation.

    Raw nonce calldata is hard-coded RLP. Execution cost is hand-counted with
    4/16 byte pricing; the floor is re-derived as 15,475 plus ten times the
    zero=1/nonzero=4 token count. Known implementation divergence is xfailed.
    """
    tx = frame_transaction(nonce_keys, nonce_seq)
    intrinsic = calculate_frame_transaction_intrinsic_cost(tx)

    assert nonce_calldata(tx).hex() == expected_calldata
    assert intrinsic.execution == expected_execution
    assert intrinsic.calldata_floor == expected_floor


@pytest.mark.parametrize(
    "nonce_keys,nonce_seq,expected_hash",
    [
        pytest.param(
            [0],
            0,
            "7d8c66c19f560f0dd3970444375ffab49f8ab30c3396f6b5b31c3c0abcc2bd90",
            id="legacy_key",
        ),
        pytest.param(
            [1, 2],
            0,
            "81b1d7b6c4b05ea4c11eaf6cafcdab8c15ef3e68fe5e8fd7b18bf449c3afe47c",
            id="key_set",
        ),
        pytest.param(
            [256],
            1,
            "090ae13482fd879ac23b563d57730cd2465ee95bf80e42f8d38bf0a82c6b8918",
            id="sequence_and_zero_byte",
        ),
    ],
)
def test_frame_signature_hash_vectors(
    nonce_keys: list[int], nonce_seq: int, expected_hash: str
) -> None:
    """Pin canonical signature hashes containing the replaced nonce fields."""
    tx = frame_transaction(nonce_keys, nonce_seq)
    assert compute_frame_signature_hash(tx).hex() == expected_hash


def test_nonce_slot_and_key_hash_vectors() -> None:
    """Pin the fixed-width hashing definitions independently of state."""
    sender = frame_transaction([1], 0).sender.__class__(b"\x11" * 20)

    assert nonce_manager_slot(sender, U256(1)).hex() == (
        "8eec1c9afb183a84aac7003cf8e730bfb6385f6e43761d6425fba4265de3a9eb"
    )
    assert nonce_manager_slot(sender, U256(2)).hex() == (
        "06bb1b9bc4293ba066a12274418b7ea4df183c2e4e6b39591987369520ca3956"
    )
    assert nonce_keys_hash(frame_transaction([0], 0)).hex() == (
        "ada5013122d395ba3c54772283fb069b10426056ef8ca54750cb9bb552a59e7d"
    )
    assert nonce_keys_hash(frame_transaction([1, 2], 0)).hex() == (
        "cfeb0db821474afb76aa0658ea10c92266529072ed43958d6e2fa8274443b0f4"
    )


@pytest.mark.parametrize(
    "existing_nonce,balance,expected_nonce",
    [
        pytest.param(None, 0, 1, id="absent"),
        pytest.param(0, 7, 1, id="existing_nonce_zero"),
        pytest.param(1, 7, 1, id="existing_nonce_one"),
        pytest.param(2, 7, 2, id="existing_nonce_two"),
    ],
)
def test_nonce_manager_activation(
    existing_nonce: int | None,
    balance: int,
    expected_nonce: int,
) -> None:
    """Activation installs code, preserves balance, and floors the nonce."""
    state = State()
    if existing_nonce is not None:
        set_account(
            state,
            NONCE_MANAGER,
            SpecAccount(
                nonce=Uint(existing_nonce),
                balance=U256(balance),
                code_hash=EMPTY_CODE_HASH,
            ),
        )

    chain = BlockChain(blocks=[], state=state, chain_id=U64(1))
    assert apply_fork(chain) is chain

    manager = state.get_account_optional(NONCE_MANAGER)
    assert manager is not None
    assert manager.nonce == expected_nonce
    assert manager.balance == balance
    assert state.get_code(manager.code_hash) == NONCE_MANAGER_CODE
    assert not state.account_has_storage(NONCE_MANAGER)


@pytest.mark.parametrize(
    "occupancy",
    [
        pytest.param(
            "code",
            marks=pytest.mark.xfail(
                reason="fork configuration has no occupied-address validator"
            ),
        ),
        pytest.param(
            "storage",
            marks=pytest.mark.xfail(
                reason="fork configuration has no occupied-address validator"
            ),
        ),
        pytest.param(
            "code_and_storage",
            marks=pytest.mark.xfail(
                reason="fork configuration has no occupied-address validator"
            ),
        ),
        pytest.param("empty", id="empty_control"),
    ],
)
def test_activation_config_occupied_address_rejected(occupancy: str) -> None:
    """
    Pin R-012--R-016 at fork-configuration finalization.

    A synthetic intended network contributes a hand-built manager account
    containing code, storage, both, or neither.  Every non-empty case must
    raise the specific configuration error before any activation block is
    attempted; the empty account is the independently enumerated control.
    """
    account: dict[str, Any] = {"nonce": 0, "balance": 1}
    if occupancy in ("code", "code_and_storage"):
        account["code"] = b"\x00"
    if occupancy in ("storage", "code_and_storage"):
        account["storage"] = {0: 1}

    class IntendedNetwork(Amsterdam, ignore=True):
        @classmethod
        def pre_allocation(cls) -> dict[int, dict[str, Any]]:
            return {int.from_bytes(NONCE_MANAGER, "big"): account}

    class CandidateConfiguration(EIP8250, IntendedNetwork, ignore=True):
        pass

    if occupancy == "empty":
        CandidateConfiguration.pre_allocation()
    else:
        with pytest.raises(ValueError, match="occupied nonce manager address"):
            CandidateConfiguration.pre_allocation()


def valid_payload() -> list[Any]:
    """Return the raw ten-field payload of a minimal valid transaction."""
    tx = frame_transaction([1], 0)
    payload = rlp.decode(rlp.encode(tx))
    assert not isinstance(payload, bytes)
    return list(payload)


@pytest.mark.parametrize(
    "case,error_match",
    [
        pytest.param("missing_field", "needs 10 field", id="missing_field"),
        pytest.param("extra_field", "needs 10 field", id="extra_field"),
        pytest.param("old_scalar", "needs 10 field", id="old_scalar_shape"),
        pytest.param(
            "sequence_list", "invalid.*int", id="sequence_not_scalar"
        ),
        pytest.param("keys_scalar", "invalid tuple", id="keys_not_list"),
        pytest.param(
            "key_too_wide", "expected at most 32", id="uint256_overflow"
        ),
        pytest.param(
            "seq_too_wide", "expected at most 8", id="uint64_overflow"
        ),
        pytest.param("key_zero_byte", "non-canonical", id="key_zero_byte"),
        pytest.param(
            "key_leading_zero", "non-canonical", id="key_leading_zero"
        ),
        pytest.param("seq_zero_byte", "non-canonical", id="seq_zero_byte"),
        pytest.param(
            "seq_leading_zero", "non-canonical", id="seq_leading_zero"
        ),
    ],
)
def test_raw_payload_rejections(case: str, error_match: str) -> None:
    """Reject malformed schemas, widths, and non-canonical integers."""
    payload = valid_payload()
    if case == "missing_field":
        payload.pop()
    elif case == "extra_field":
        payload.append(b"")
    elif case == "old_scalar":
        payload.pop(1)
    elif case == "sequence_list":
        payload[2] = [payload[2]]
    elif case == "keys_scalar":
        payload[1] = b"\x01"
    elif case == "key_too_wide":
        payload[1] = [b"\x01" + b"\x00" * 32]
    elif case == "seq_too_wide":
        payload[2] = (2**64).to_bytes(9, byteorder="big")
    elif case == "key_zero_byte":
        payload[1] = [b"\x00"]
    elif case == "key_leading_zero":
        payload[1] = [b"\x00\x01"]
    elif case == "seq_zero_byte":
        payload[2] = b"\x00"
    elif case == "seq_leading_zero":
        payload[2] = b"\x00\x01"
    else:
        raise AssertionError(case)

    raw = SpecBytes(b"\x06" + rlp.encode(payload))
    with pytest.raises(DecodingError, match=error_match):
        decode_transaction(raw)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("empty", id="empty"),
        pytest.param("too_many", id="too_many"),
        pytest.param("duplicate", id="duplicate"),
        pytest.param("descending", id="descending"),
        pytest.param("mixed_zero", id="mixed_zero"),
    ],
)
@pytest.mark.xfail(
    reason="decoder currently defers semantic nonce-key checks to validation"
)
def test_raw_payload_semantic_rejections(case: str) -> None:
    """
    Pin R-040 and R-042--R-049 at the decoder boundary.

    Each payload mutation is a hand-enumerated violation of the pinned EIP's
    decoder rules. The expected DecodingError is xfailed for the recorded
    implementation divergence rather than weakened to later validation.
    """
    payload = valid_payload()
    if case == "empty":
        payload[1] = []
    elif case == "too_many":
        payload[1] = [bytes([key]) for key in range(1, 18)]
    elif case == "duplicate":
        payload[1] = [b"\x01", b"\x01"]
    elif case == "descending":
        payload[1] = [b"\x02", b"\x01"]
    elif case == "mixed_zero":
        payload[1] = [b"", b"\x01"]
    else:
        raise AssertionError(case)

    with pytest.raises(DecodingError):
        decode_transaction(SpecBytes(b"\x06" + rlp.encode(payload)))


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("maximum_key", id="uint256_at"),
        pytest.param("key_overflow", id="uint256_at_plus_one"),
        pytest.param("highest_usable_sequence", id="uint64_at_minus_one"),
        pytest.param(
            "reserved_sequence",
            id="uint64_at",
            marks=pytest.mark.xfail(
                reason="MAX sequence is rejected during static validation"
            ),
        ),
        pytest.param("sequence_overflow", id="uint64_at_plus_one"),
    ],
)
def test_integer_width_phase_boundaries(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """
    Pin R-005, R-022, R-046--R-047, R-065, and R-177--R-178.

    The five raw integers are hard-coded at-1/at/at+1 width vectors. Decoder
    outcomes follow their byte lengths. For both in-range sequence values,
    the production admission entry point is instrumented immediately after
    static validation: MAX-1 reaches stateful validity, while MAX is expected
    to reach that same phase before its reserved-value rejection.
    """
    payload = valid_payload()
    if case == "maximum_key":
        payload[1] = [(2**256 - 1).to_bytes(32, "big")]
        decoded = decode_transaction(SpecBytes(b"\x06" + rlp.encode(payload)))
        assert isinstance(decoded, FrameTransaction)
        assert decoded.nonce_keys == (U256(2**256 - 1),)
        return
    if case == "key_overflow":
        payload[1] = [(2**256).to_bytes(33, "big")]
        with pytest.raises(DecodingError, match="expected at most 32"):
            decode_transaction(SpecBytes(b"\x06" + rlp.encode(payload)))
        return
    if case == "sequence_overflow":
        payload[2] = (2**64).to_bytes(9, "big")
        with pytest.raises(DecodingError, match="expected at most 8"):
            decode_transaction(SpecBytes(b"\x06" + rlp.encode(payload)))
        return

    sequence = 2**64 - (2 if case == "highest_usable_sequence" else 1)
    payload[2] = sequence.to_bytes(8, "big")
    tx = decode_transaction(SpecBytes(b"\x06" + rlp.encode(payload)))
    assert isinstance(tx, FrameTransaction)
    assert tx.nonce_seq == U64(sequence)

    class StatefulValidityEnteredError(Exception):
        pass

    def enter_stateful_validity(*_args: Any, **_kwargs: Any) -> None:
        raise StatefulValidityEnteredError

    monkeypatch.setattr(
        "ethereum.forks.amsterdam.frame_processing.TransactionState",
        enter_stateful_validity,
    )
    with pytest.raises(StatefulValidityEnteredError):
        check_frame_transaction(
            SimpleNamespace(state=object()),  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            tx,
            Uint(0),
        )


@pytest.mark.parametrize("key_count", [1, 16])
def test_approval_read_order_oog_and_unmetered_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
    key_count: int,
) -> None:
    """
    Pin R-073--R-075, R-084, R-086--R-089, and R-104--R-108.

    For one and sixteen keys, instrumented production calls show every
    hand-derived nonce slot is read before any write. With `count*20000-1`
    gas the aggregate raises the specific OutOfGasError and performs no write;
    at equality all writes occur while EVM access sets and the refund counter
    remain exactly empty. A repeated payment approval returns false before
    any keyed read or write, independently proving exactly-once consumption.
    """
    import ethereum.forks.amsterdam.vm as vm

    encoded = Transaction(
        sender=EOA(key=Hash(3)),
        nonce_keys=list(range(1, key_count + 1)),
        nonce_seq=0,
        frames=[Frame(flags=3)],
        signatures=[],
    )
    tx = decode_transaction(SpecBytes(encoded.rlp()))
    assert isinstance(tx, FrameTransaction)

    events: list[tuple[str, Bytes32, U256 | None]] = []
    expected_slots = [
        nonce_manager_slot(tx.sender, U256(key))
        for key in range(1, key_count + 1)
    ]
    account = SimpleNamespace(balance=U256(10**30))
    monkeypatch.setattr(vm, "get_account", lambda _state, _address: account)

    def _record_read(_state: Any, _address: Any, slot: Bytes32) -> U256:
        """Record the read and report the slot as unset."""
        events.append(("read", slot, None))
        return U256(0)

    monkeypatch.setattr(vm, "get_protocol_storage", _record_read)
    monkeypatch.setattr(
        vm,
        "set_protocol_storage",
        lambda _state, _address, slot, value: events.append(
            ("write", slot, value)
        ),
    )
    monkeypatch.setattr(vm, "set_account_balance", lambda *_args: None)

    def environment() -> Any:
        context = SimpleNamespace(
            tx=tx,
            current_frame_index=Uint(0),
            payer=None,
            sender_approved=False,
            max_cost=Uint(0),
            approval_payer_balance=None,
            approval_sender_nonce=None,
            payment_approved_execution=False,
        )
        return cast(
            Any,
            SimpleNamespace(
                frame_context=context,
                state=object(),
                access_list_addresses=set(),
                access_list_storage_keys=set(),
            ),
        )

    failing_env = environment()
    required_gas = 20_000 * key_count
    failing_meter = GasMeter(Uint(required_gas - 1), Uint(0), Uint(0))
    with pytest.raises(OutOfGasError):
        attempt_approval(failing_env, tx.frames[0].flags, failing_meter)
    assert events == [("read", slot, None) for slot in expected_slots]
    assert failing_env.frame_context.payer is None
    assert failing_meter.refund_counter == 0

    events.clear()
    successful_env = environment()
    successful_meter = GasMeter(Uint(required_gas), Uint(0), Uint(0))
    assert attempt_approval(
        successful_env, tx.frames[0].flags, successful_meter
    )
    assert events == (
        [("read", slot, None) for slot in expected_slots]
        + [("write", slot, U256(1)) for slot in expected_slots]
    )
    assert successful_meter.gas_left == 0
    assert successful_meter.refund_counter == 0
    assert successful_env.access_list_addresses == set()
    assert successful_env.access_list_storage_keys == set()

    events.clear()
    assert not attempt_approval(
        successful_env, tx.frames[0].flags, successful_meter
    )
    assert events == []


def test_disjoint_keys_do_not_imply_shared_state_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Pin R-159 and R-160 with a shared-payer solvency conflict.

    Slots 20 and 21 are independently seeded absent, so both sequence-zero
    transactions are nonce-valid by direct comparison. The payer has exactly
    the first transaction's hard-coded maximum cost of five: approval one
    writes only slot 20 and debits five to zero; approval two then returns
    false for insufficient balance before reading or writing slot 21.
    """
    import ethereum.forks.amsterdam.vm as vm

    sender = EOA(key=Hash(3))
    transactions = [
        decode_transaction(
            SpecBytes(
                Transaction(
                    sender=sender,
                    nonce_keys=[key],
                    nonce_seq=0,
                    frames=[Frame(flags=3)],
                    signatures=[],
                ).rlp()
            )
        )
        for key in (20, 21)
    ]
    frame_transactions = []
    for decoded in transactions:
        assert isinstance(decoded, FrameTransaction)
        frame_transactions.append(decoded)
    slots = [
        nonce_manager_slot(tx.sender, tx.nonce_keys[0])
        for tx in frame_transactions
    ]
    storage: dict[Bytes32, U256] = {}
    payer = SimpleNamespace(balance=U256(5))
    reads: list[Bytes32] = []

    monkeypatch.setattr(vm, "get_account", lambda _state, _address: payer)

    def get_slot(_state: object, _address: object, slot: Bytes32) -> U256:
        reads.append(slot)
        return storage.get(slot, U256(0))

    def set_slot(
        _state: object, _address: object, slot: Bytes32, value: U256
    ) -> None:
        storage[slot] = value

    def set_balance(_state: object, _address: object, value: U256) -> None:
        payer.balance = value

    monkeypatch.setattr(vm, "get_protocol_storage", get_slot)
    monkeypatch.setattr(vm, "set_protocol_storage", set_slot)
    monkeypatch.setattr(vm, "set_account_balance", set_balance)

    def environment(tx: FrameTransaction) -> Any:
        return cast(
            Any,
            SimpleNamespace(
                frame_context=SimpleNamespace(
                    tx=tx,
                    current_frame_index=Uint(0),
                    payer=None,
                    sender_approved=False,
                    max_cost=Uint(5),
                    approval_payer_balance=None,
                    approval_sender_nonce=None,
                    payment_approved_execution=False,
                ),
                state=object(),
            ),
        )

    assert [storage.get(slot, U256(0)) for slot in slots] == [U256(0), U256(0)]
    first_meter = GasMeter(Uint(20_000), Uint(0), Uint(0))
    assert attempt_approval(
        environment(frame_transactions[0]),
        frame_transactions[0].frames[0].flags,
        first_meter,
    )
    assert storage == {slots[0]: U256(1)}
    assert payer.balance == U256(0)

    reads.clear()
    second_meter = GasMeter(Uint(20_000), Uint(0), Uint(0))
    assert not attempt_approval(
        environment(frame_transactions[1]),
        frame_transactions[1].frames[0].flags,
        second_meter,
    )
    assert reads == []
    assert storage.get(slots[1], U256(0)) == U256(0)
    assert second_meter.gas_left == Uint(20_000)
