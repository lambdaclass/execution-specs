"""Defines EIP-7906 specification constants and types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceSpec:
    """Defines the reference spec version and git path."""

    git_path: str
    version: str


ref_spec_7906 = ReferenceSpec(
    "EIPS/eip-7906.md", "690fc1d2f7a765dae9d388d51548cafe5cb1da40"
)


@dataclass(frozen=True)
class Spec:
    """
    Parameters from the EIP-7906 specification as defined at
    https://eips.ethereum.org/EIPS/eip-7906.
    """

    MODE_POST_TX = 3
    FIRST_UNDEFINED_MODE = 4

    # Opcode byte assignments: the next free slots after EIP-8141's
    # 0xB0-0xB4. The pinned spec revision assigns no bytes; these are
    # the reference implementation's choice.
    TXTRACE_OPCODE_BYTE = 0xB5
    EVENTDATACOPY_OPCODE_BYTE = 0xB6
    TXDIFF_OPCODE_BYTE = 0xB7

    # TXTRACE parameter selectors
    TXTRACE_BALANCES_CHANGED = 0x00
    TXTRACE_SLOTS_CHANGED = 0x01
    TXTRACE_CONTRACTS_DEPLOYED = 0x02
    TXTRACE_BALANCE_ADDRESS = 0x03
    TXTRACE_BALANCE_BEFORE = 0x04
    TXTRACE_BALANCE_AFTER = 0x05
    TXTRACE_SLOT_ADDRESS = 0x06
    TXTRACE_SLOT_KEY = 0x07
    TXTRACE_SLOT_VALUE_BEFORE = 0x08
    TXTRACE_SLOT_VALUE_AFTER = 0x09
    TXTRACE_DEPLOYED_ADDRESS = 0x0A
    TXTRACE_DEPLOYED_CODEHASH = 0x0B
    TXTRACE_EVENTS_COUNT = 0x0C
    TXTRACE_EVENT_ADDRESS = 0x0D
    TXTRACE_EVENT_TOPIC_COUNT = 0x0E
    TXTRACE_EVENT_TOPIC0 = 0x0F
    TXTRACE_EVENT_TOPIC1 = 0x10
    TXTRACE_EVENT_TOPIC2 = 0x11
    TXTRACE_EVENT_TOPIC3 = 0x12
    TXTRACE_EVENT_DATA_LENGTH = 0x13
    TXTRACE_GAS_PRE_CHARGE = 0x14
    TXTRACE_GAS_PAYER_ADDRESS = 0x15
    FIRST_UNDEFINED_TXTRACE_PARAM = 0x16

    # TXDIFF parameter selectors
    TXDIFF_SLOT_VALUE_BEFORE = 0x00
    TXDIFF_SLOT_VALUE_AFTER = 0x01
    TXDIFF_BALANCE_BEFORE = 0x02
    TXDIFF_BALANCE_AFTER = 0x03
    TXDIFF_CODEHASH_BEFORE = 0x04
    TXDIFF_CODEHASH_AFTER = 0x05
    TXDIFF_ADDRESS_SLOTS_COUNT = 0x06
    TXDIFF_ADDRESS_SLOT_INDEX = 0x07
    TXDIFF_ADDRESS_EVENTS_COUNT = 0x08
    TXDIFF_ADDRESS_EVENT_INDEX = 0x09
    TXDIFF_ACCOUNT_CHANGE_FLAGS = 0x0A
    FIRST_UNDEFINED_TXDIFF_PARAM = 0x0B

    # Account change flag bits, in account tuple field order.
    FLAG_NONCE_CHANGED = 0b0001
    FLAG_BALANCE_CHANGED = 0b0010
    FLAG_STORAGE_CHANGED = 0b0100
    FLAG_CODE_CHANGED = 0b1000

    # The spec leaves `TXTRACE_GAS_COST` TBD; this is the reference
    # implementation's choice, exercised only through behavior (flat
    # pricing of TXDIFF params 0x06-0x0A, TXTRACE's own per-call cost,
    # and the exact/out-of-gas boundary arms).
    TXTRACE_GAS_COST = 100

    # EVENTDATACOPY is priced exactly like CALLDATACOPY: the spec's
    # `EVENTDATACOPY_GAS_COST` constant is superseded by its normative
    # prose ("a fixed cost of 3 and a variable cost that accounts for
    # the memory expansion and copying").
    EVENTDATACOPY_BASE_GAS_COST = 3
    COPY_GAS_COST_PER_WORD = 3

    # keccak256 of empty code: what TXDIFF's codehash params report
    # for undeployed contracts, diverging from EXTCODEHASH's zero.
    EMPTY_CODE_HASH = int.from_bytes(
        bytes.fromhex(
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        ),
        "big",
    )
