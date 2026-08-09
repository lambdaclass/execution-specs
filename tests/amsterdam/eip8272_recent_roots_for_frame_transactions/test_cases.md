# EIP-8272 Recent Roots for Frame Transactions — Test Cases

Reference spec: `EIPS/eip-8272.md` at blob
`1ff890ed8ce03be2cf25f7528c78733d35a3dfe3`.

The write path — `RECENT_ROOT_CODE` and everything reachable only through
it — is not covered here: the specification still lists that constant as
`TBD`, so there is no bytecode to install and no call to make. The read
path is covered in full by seeding entries directly into the contract's
pre-state, which is exactly the storage the reference check reads.

Two further areas are out of reach here, and both are declared in
`eip_checklist_not_applicable.txt` rather than left silent:

- **The activation rules.** Bogota is not reachable through a transition
  fork in this repository — the only Amsterdam-adjacent transition is
  `BPO2ToAmsterdamAtTime15k` — so no fixture can place a block on either
  side of the fork, and the empty-code/empty-storage precondition on the
  parent state has nothing to assert against.
- **Anything that needs a frame transaction's gas limit field**, which
  EIP-8141 does not give it. The equivalent boundaries are reached
  through EIP-7825's cap (`test_reference_intrinsic_gas_boundary`) and
  through the calldata floor (`test_reference_calldata_floor_is_charged`).

One checklist row stays blank for a template reason rather than a
coverage one: `transaction_type/test/intrinsic_validity/data_floor_above
_intrinsic_gas_cost` appears twice under one ID, once for the valid case
and once for the invalid one, and only the valid case exists for a
transaction with no gas limit field.

One expectation in this directory is **contested rather than settled**.
`test_reference_check_adds_no_block_access_list_entry` asserts that the
reference check contributes nothing to the EIP-7928 block access list,
which is the literal reading of "This affects warm/cold gas accounting
only". The check does read `RECENT_ROOT_ADDRESS[storage_key]`, and a
client that records that read commits a different list — and therefore a
different block hash — so the two readings are a consensus split rather
than a cosmetic one. The negative arm is paired with a control,
`test_frame_read_of_the_recent_root_contract_is_recorded`, which shows
the same key appearing when a frame really loads it.

| Function Name | Goal | Setup | Expectation | Status |
|---------------|------|-------|-------------|--------|
| `test_reference_window_boundaries` | Pin `1 <= current_slot - slot <= RECENT_ROOT_USABLE_WINDOW` | One reference, entry always seeded, slot swept at deltas 0, 1, 2, 8190, 8191, 8192, 8193 from the block's slot | Deltas 1 through 8191 accepted; delta 0 and deltas at or beyond 8192 rejected as invalid references | ✅ Completed |
| `test_reference_future_slot` | Reject slots after the block's own | Entry seeded for a slot one and one thousand slots ahead | Rejected as an invalid reference | ✅ Completed |
| `test_reference_slot_underflows_window` | Catch a client that subtracts before it compares | Future slots against blocks at slots 0, 4095 and 8190, each asserted to wrap to a distance inside `[1, 8191]` — the edge arm wraps to exactly 8191 | Rejected; unsigned wraparound must not make a future slot look in-window | ✅ Completed |
| `test_reference_slot_zero` | A root written in slot `S` is referenceable from `S + 1`, at `S = 0` | Reference to slot 0, block at slot 0 and at slot 1 | Rejected at slot 0, accepted at slot 1 | ✅ Completed |
| `test_reference_missing_entry` | All entries start zero | In-window reference, nothing seeded | Rejected; recent root contract storage stays empty | ✅ Completed |
| `test_one_invalid_reference_in_a_full_list` | Every reference is checked, not a prefix of the list | Sixteen references, fifteen in-window and seeded, one defective — a missing entry or the current slot — placed first and last | Rejected from either position; a checker that stopped after the first few admits the last-position arm | ✅ Completed |
| `test_reference_entry_hash_binding` | The entry commits to source, slot and root | Entry seeded for one tuple; the declared reference differs in exactly one field | All three arms rejected; the wrong-slot arm reaches the same cell through the shared window index | ✅ Completed |
| `test_reference_entry_hash_near_miss` | The entry-hash equality is over all 32 bytes | Correct key seeded with the entry hash exclusive-ored with `1 << b` for `b` in 0, 127, 255 — one flipped bit, one changed byte, 31 bytes still agreeing | All three arms rejected; a comparison over the leading byte admits the bit-0 arm, one over the trailing byte admits bit 255, and one over both ends admits bit 127 | ✅ Completed |
| `test_reference_check_ignores_unrelated_storage` | The check reads one cell of one account | Correct entry placed one key over, under the expiry verifier instead, or surrounded by twenty unrelated values | All arms rejected | ✅ Completed |
| `test_reference_count_boundary` | `MAX_RECENT_ROOT_REFERENCES` | 15, 16 and 17 references, each naming a distinct in-window slot and all seeded | 15 and 16 accepted; 17 rejected for exceeding the cap | ✅ Completed |
| `test_duplicate_references_accepted` | Duplicates are valid and preserved | Sixteen byte-identical references naming one storage key | Accepted | ✅ Completed |
| `test_reference_count_cap_counts_duplicates` | The cap counts declared references, not distinct ones | Seventeen byte-identical references over one seeded entry | Rejected for exceeding the cap; `test_duplicate_references_accepted` is the accepted control one reference shorter | ✅ Completed |
| `test_reference_opaque_root_values` | `root` is opaque to consensus | All-zero and all-ones roots, correctly seeded | Both accepted; a zero root is not an unset cell | ✅ Completed |
| `test_ring_index_collision` | `i = slot mod RECENT_ROOT_LENGTH`, and why the current slot is excluded | Slots `C` and `C - 8192` asserted to share a key, then referenced | Both rejected — one not over yet, one expired | ✅ Completed |
| `test_reference_derivation_matches_hand_vector` | `entry_hash` and `storage_key` derivations | Entry seeded from hand-computed literals for `(0x…aa, salt 1, slot 8, 0x42…42)` | Accepted; the client must reproduce both derivations byte for byte | ✅ Completed |
| `test_one_address_addresses_many_root_sources` | `source_id = keccak256(source_address || salt)` — one address owns as many sources as it picks salts | Three references sharing an address and a slot, differing only in salt (1, 2, `2**256 - 1`), each naming a different root; distinctness of the three identifiers and their three keys asserted at fill time | Accepted; dropping the salt from the derivation collapses all three onto one cell, where at most one entry can live, so at least two references would fail | ✅ Completed |
| `test_reference_intrinsic_gas_boundary` | The whole intrinsic cost, at 0, 1, 2 and 16 references and two byte profiles | Contract sender (no signature entries); frame gas limits set to `TX_MAX_GAS_LIMIT - intrinsic` and one more | At the cap accepted, one over rejected — pinning the calldata term, the once-only address charge and the per-reference charge exactly | ✅ Completed |
| `test_zero_references_are_charged_for_the_empty_list` | `rlp([]) = 0xc0` is still charged | No references, frame gas limits at the cap for the intrinsic including the one-byte encoding | Accepted; skipping the empty-list calldata term overshoots the cap | ✅ Completed |
| `test_duplicate_references_are_charged_independently` | Duplicates are charged per declaration while their keys deduplicate | Sixteen byte-identical references asserted to name one key; frame gas limits at `TX_MAX_GAS_LIMIT - intrinsic` and one more | At the cap accepted, one over rejected — charging per distinct reference is short by fifteen per-reference charges | ✅ Completed |
| `test_reference_calldata_floor_is_charged` | The two floor bullets: the per-reference charge anchors the floor and the encoded list joins the bytes it counts | Contract sender, cheap frames, 8 and 16 references at two byte profiles; the derived floor asserted to exceed the standard branch | `cumulative_gas_used` is exactly the derived floor. Both byte profiles of a count expect the same number, because the floor counts bytes while the standard branch counts tokens | ✅ Completed |
| `test_reference_identifier_field_sizes` | `source_id` and `root` are exactly 32 bytes | `transaction_test` arms carrying the well-formed value with its last byte dropped, and with a zero byte prepended, on each field | All four rejected by the decoder | ✅ Completed |
| `test_reference_slot_encoding` | `slot` is a canonical integer below `2**64` | Value 8 encoded in two bytes; slot `2**64`; slot `2**256 - 1` | All rejected — the first non-canonical, the others out of range | ✅ Completed |
| `test_reference_item_shape` | Each element is an RLP list of exactly three items | The same three values as one concatenated byte string, as a two-item list, and as a four-item list | All three rejected by the decoder | ✅ Completed |
| `test_payload_element_count` | The payload has exactly ten elements after the insertion | Reference list dropped from the payload (EIP-8141's nine-element shape) and repeated (eleven) | Rejected with too few and too many elements respectively | ✅ Completed |
| `test_transaction_serialization_length` | The serialization ends where the payload does | The well-formed encoding with its last byte dropped, and with one byte appended | Rejected as truncated and as carrying trailing bytes | ✅ Completed |
| `test_recentrootrefload_stack_underflow` | The opcode needs both operands | Bare instruction, and one preceded by a single push | Both halt; the caller's sentinel still lands | ✅ Completed |
| `test_reference_warms_recent_root_address` | The contract starts warm | `BALANCE` probe on the contract and on an adjacent address, with and without a reference | Warm price with a reference, cold without; the neighbour stays cold in both | ✅ Completed |
| `test_reference_warms_only_declared_storage_keys` | Only the declared key is warm | Probe substituted at the recent root contract, loading the declared key and a sibling key | Declared key warm, sibling key cold | ✅ Completed |
| `test_every_key_of_a_full_reference_list_is_warm` | Each declared reference warms its own key | Probe substituted at the recent root contract, measuring all sixteen declared keys and one sibling | All sixteen warm, sibling cold; warming only the head of the list leaves the later keys cold | ✅ Completed |
| `test_prewarm_survives_frame_revert` | Pre-warming predates every frame | Frame two probes and reverts; frame three probes again | Frame three still reads the warm price | ✅ Completed |
| `test_prewarm_does_not_leak_to_next_transaction` | Pre-warming is transaction-scoped | Referencing transaction followed by a reference-free one in the same block | The second reads the cold price | ✅ Completed |
| `test_recentrootrefload_reads_every_field` | The three field selectors | Three references differing in every field, all nine values read back | Every stored word is the envelope value, slot zero-extended | ✅ Completed |
| `test_recentrootrefload_reads_the_whole_reference_list` | Every index of a full list is reachable and answers for itself | Sixteen references agreeing on no field; all forty-eight words read back | Each stored word is its own envelope value, up to index 15 | ✅ Completed |
| `test_recentrootrefload_stack_order` | `field` on top, `index` below | Mirrored operand pairs `(index=0, field=2)` and `(index=2, field=0)` | Both words correct; transposed operands break both | ✅ Completed |
| `test_recentrootrefload_slot_zero_extension` | Slot as a zero-extended 256-bit word | Reference at slot `2**64 - 2` in a block at slot `2**64 - 1` | Full word asserted | ✅ Completed |
| `test_recentrootrefload_out_of_range_operands_halt` | Index and field bounds | `index = len`, `2**64`, `2**256-1`; `field = 3`, `2**256-1`; and an empty reference list | Every arm exceptionally halts; the caller's sentinel still lands | ✅ Completed |
| `test_recentrootrefload_gas` | `RECENTROOTREFLOAD_GAS` | Measured around the bare opcode with the operand pushes subtracted out | Exactly three gas | ✅ Completed |
| `test_recentrootrefload_out_of_gas` | The charge is enforced | Probe called with exactly its cost and with one gas less | Succeeds, then halts | ✅ Completed |
| `test_recentrootrefload_reads_envelope_not_storage` | Envelope, never contract storage | Duplicate references over a cell holding the entry hash | Every field read returns the envelope value, never the stored hash | ✅ Completed |
| `test_recentrootrefload_does_not_displace_sigparam` | `0xB5` is free, `0xB4` is `SIGPARAM` | Both opcodes in one contract in one transaction | Each returns its own, unrelated value | ✅ Completed |
| `test_txparam_reference_count` | `TXPARAM(0x0F)` | Reference counts 0, 1, 2 and 16 | Returns the count; the zero arm stores count plus one so zero is distinguishable from unexecuted | ✅ Completed |
| `test_txparam_counts_declared_not_distinct_references` | The count is `len`, and duplicates keep their positions | Sixteen references, two distinct, declared eight times each | Returns 16 rather than 2; index 7 reads the first reference and index 15 the second | ✅ Completed |
| `test_txparam_opens_exactly_one_index` | Only `0x0F` is defined | Indices `0x0C`, `0x0D`, `0x0E`, `0x0F`, `0x10` | All but `0x0F` exceptionally halt | ✅ Completed |
| `test_current_slot_comes_from_the_header` | `current_slot` is EIP-7843's `slotNumber` | Mirrored arms with the header slot and the timestamp-implied slot swapped, same reference tuple | Header reading accepts one and rejects the other; a timestamp reading gets both backwards | ✅ Completed |
| `test_slotnum_agrees_with_the_accepted_window` | One slot number for validation and for the EVM | Frame stores `SLOTNUM - RECENTROOTREFLOAD(index, slot)` at both window ends | The stored difference is the distance the check accepted | ✅ Completed |
| `test_verify_frame_binds_reference` | Application-side binding through introspection | Contract sender approves only when all three fields match constants in its code; both arms declare a seeded, in-window reference | Matching tuple valid; a genuinely published root from another source leaves the transaction unapproved | ✅ Completed |
| `test_reference_set_is_signed` | The field is inside the signature hash | Signature bytes made over one reference set, transmitted with another that is itself valid and seeded | Rejected on the signature, not on the reference | ✅ Completed |
| `test_reference_set_is_immutable_across_frames` | Frame data cannot change the set | Middle frame's data is a well-formed encoding of a different set; frames either side read the count and all fields | Both readers see the declared set | ✅ Completed |
| `test_recentrootrefload_in_default_and_sender_frames` | Usable in any frame mode | `DEFAULT` and `SENDER` frames read the same reference | Identical words stored | ✅ Completed |
| `test_invalid_reference_prevents_frame_execution` | One bad reference stops everything | Two references, the first seeded and the second not | Storage at its pre-value, sender nonce unchanged, contract storage untouched | ✅ Completed |
| `test_nonce_is_checked_before_the_references` | References are checked after EIP-8141's nonce check | Sender pre-set to nonce 1, transaction declaring nonce 0, with the reference unsatisfied and — as the control — satisfied | Both arms rejected for the nonce, not the reference; a client checking references first parts company on the unsatisfied arm alone | ✅ Completed |
| `test_references_survive_an_atomic_batch_rollback` | The set is not state a rollback can undo | Atomic batch reads the reference and writes it, terminator reverts, a later frame reads again | The batch's write is unrolled; the later read still returns the reference | ✅ Completed |
| `test_reference_validity_follows_the_block_prestate` | Validity is a property of the block, not of the tuple | Byte-identical transactions in blocks at the same slot, entry seeded in one and absent in the other | Accepted in the first, rejected in the second | ✅ Completed |
| `test_reference_to_an_entry_written_earlier_in_the_block` | The pre-state includes prior transactions of the same block | Writer substituted at the recent root contract; first transaction stores the test-computed key/entry pair, second references it. Control drops the writing transaction | Accepted with the earlier write, rejected without it | ✅ Completed |
| `test_current_slot_write_cannot_invalidate_a_reference` | A current-slot write cannot invalidate a valid reference | First transaction overwrites the cell the current slot maps to; second references an entry one slot old, in a different cell | The reference still validates; both cells asserted in the post-state | ✅ Completed |
| `test_reference_free_frame_tx_matches_baseline` | The frame layout and frame execution rules are otherwise unchanged | Identical three-frame transactions declaring no references and sixteen; the middle frame reads the frame count, its own index and four `FRAMEPARAM` values | Every layout word is the value the transaction declared, in both arms; the two frames that execute nothing report zero gas and the target's cold access charge, in both arms | ✅ Completed |
| `test_txparam_reference_count_gas` | `TXPARAM(0x0F)` costs the standard `TXPARAM` gas | The new index and EIP-8141's frame-count index measured side by side in one transaction | Both measure the same value, and that value is EIP-8141's two | ✅ Completed |
| `test_other_transaction_types_pay_no_reference_gas` | No other transaction type gains a reference charge | Legacy, access-list and dynamic-fee transactions with empty calldata to a `STOP` recipient, gas limit at the fork's intrinsic cost | `cumulative_gas_used` is exactly that intrinsic cost — sixteen gas more would be the empty-list charge leaking across | ✅ Completed |
| `test_delegation_is_unaffected_by_a_referencing_transaction` | EIP-7702 is untouched, and pre-warming does not leak into another transaction type | A reference-carrying frame transaction followed in the same block by a set-code transaction whose delegate probes the recent root contract | The delegation designator lands, the delegated code runs, and the probe reads the cold price | ✅ Completed |
| `test_reference_check_adds_no_block_access_list_entry` | The check contributes nothing to the EIP-7928 block access list | One valid reference, no frame near the recent root contract | The contract is absent from the list. **Contested** — see the note above | ✅ Completed |
| `test_frame_read_of_the_recent_root_contract_is_recorded` | Control for the row above | A prober substituted at the recent root contract `SLOAD`s the declared key from inside a frame | The account appears in the list with that key under `storage_reads` | ✅ Completed |
