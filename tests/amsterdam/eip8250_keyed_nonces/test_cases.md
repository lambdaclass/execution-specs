# EIP-8250 Keyed Nonces — Test Plan

Reference spec: `EIPS/eip-8250.md` @ `81b976ac01591fed2eecb73fa574f27cd18db2e8`.
All tests are gated on `valid_from("Bogota")`, the pseudo-fork that registers
the `EIP8250` mixin on top of `EIP8141`.

## Constants and wire format — `test_vectors.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_pinned_constant_table` | Pin the constant table against the implementation | Import the production constants | `NONCE_MANAGER` is `0x8250`, code `60006000fd`, first-use gas 20,000, `MAX_NONCE_SEQ` `2**64-1`, `MAX_NONCE_KEYS` 16 | Implemented |
| `test_payload_exact_rlp_and_signature_vector` | Pin the ten-field payload layout and signing hash | Fixed sender `0x11..11`, one empty frame, `[0]`/`0` | Hand-assembled RLP payload byte string and a hard-coded Keccak signing hash | Implemented |
| `test_nonce_calldata_pricing_vectors` | Pin `nonce_calldata` encoding and its token counts | Three key/sequence shapes | Hand-encoded RLP plus independently counted tokens, standard gas, and floor gas | Implemented |
| `test_storage_slot_hardcoded_vectors` | Pin the slot preimage | Sender `0x1234`, keys 1 and 2 | Two hard-coded Keccak slot vectors that differ from each other | Implemented |
| `test_erc4337_key_width_subset_vector` | Show EIP-8250 keys are wider than ERC-4337 keys | Keys `2**192-1`, `2**192`, `2**256-1` | All three are valid and consume their slot to 1 | Implemented |
| `test_signature_commits_entire_visible_keyset` | Prove the signature commits every key | Key sets `[1,2]` and `[1,3]` | Two hard-coded, unequal signing hashes | Implemented |

## Static validity — `test_validation.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_nonce_key_collection_bounds` | Pin the 1..16 key-count bound | Key counts 0, 1, 16, 17 | 0 and 17 rejected as invalid frame format; 1 and 16 consume every selected key | Implemented |
| `test_strict_numeric_order_and_zero_singleton` | Pin strict numeric ordering and the zero singleton | Duplicate, descending, mixed-zero, `[0]`, ascending, `[255,256]` | The first three are rejected; `[255,256]` proves numeric rather than lexicographic order | Implemented |
| `test_invalid_nonce_fields` | Sweep every malformed field shape in one place | Five malformed key lists plus `nonce_seq == MAX_NONCE_SEQ` on a slot pre-advanced to `MAX_NONCE_SEQ` | Each is rejected; the reserved sequence isolates the `nonce_seq < MAX_NONCE_SEQ` rule from the per-key equality rule | Implemented |

## State transition and gas — `test_keyed_nonces.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_first_use_consumes_key_set` | Pin first use of a fresh key set | 1, 2, max-width, and 16 keys | Every slot becomes `nonce_seq+1 == 1`, the sender account nonce stays 0, gas is 20,000 per key | Implemented |
| `test_legacy_zero_alias_never_manager_slot` | Pin the key-zero alias | Initial nonces 0, 1, `2**64-2` | The account nonce advances by one and no manager slot is written | Implemented |
| `test_subsequent_key_increment_has_zero_surcharge` | Prove only absent slots are surcharged | Two slots pre-seeded to 4 | Slots advance to 5 in a zero-gas frame | Implemented |
| `test_highest_keyed_sequence_becomes_exhausted` | Pin the final permitted advance | Slot pre-seeded to `MAX_NONCE_SEQ-1` | Slot reaches `MAX_NONCE_SEQ`, the reserved exhausted state | Implemented |
| `test_first_use_surcharge_one_key_gas_triptych` | Boundary the one-key surcharge | Approval gas 19,999 / 20,000 / 20,001 | Below fails and writes nothing; equal and above succeed and write 1 | Implemented |
| `test_first_use_surcharge_sixteen_key_aggregate` | Boundary the aggregate surcharge | Approval gas 319,999 / 320,000 / 320,001 | The aggregate check is all-or-nothing across all sixteen slots | Implemented |
| `test_keyed_sequence_mismatch` | Pin the per-key sequence equality | Stored 0 vs tx 1, stored 2 vs tx 1 | Rejected as nonce too high and too low respectively | Implemented |
| `test_keyset_requires_one_shared_sequence` | Prove every selected key must match | Two slots seeded 4/4 and 4/5 | The shared case succeeds; one mismatched domain rejects the whole transaction | Implemented |
| `test_payment_approval_consumes_only_once` | Pin exactly-once consumption | Two payment-approving payers | The second payment approval fails and the slot stays at 1 | Implemented |
| `test_approval_survives_later_frame_revert` | Pin durability against a later revert | Approving frame then a reverting frame | The slot keeps `nonce_seq+1` | Implemented |
| `test_approval_survives_revert_of_approving_frame` | Pin durability against the approving frame's own revert | `DELEGATECALL` to an approver, then `REVERT` | Approval effects and the slot survive | Implemented |
| `test_protocol_bookkeeping_remains_cold_unmetered` | Prove bookkeeping does not warm the manager | A later frame reads `BALANCE(NONCE_MANAGER)` | The access is charged the full cold price | Implemented |
| `test_approval_survives_failed_atomic_batch` | Pin durability against a batch rollback | Batch with a failing member and a skipped successor | Statuses success/success/failure/skipped; the slot survives the rollback | Implemented |
| `test_payment_approval_preserves_transiently_funded_payer` | Pin paymaster payment | Payer funded inside a batch that later rolls back | Payer balance equals the transient credit minus hand-summed gas | Implemented |
| `test_execution_only_approval_reverts_outside_payment_scope` | Prove execution-only approval is not payment | Separate execution and payment delegate calls, then a revert | The transaction is invalid for want of a durable approval | Implemented |
| `test_key_zero_preserves_live_nonce_across_batch_rollback` | Pin the key-zero increment semantics | Contract sender bumped by `CREATE` then by approval | Final nonce is 3: the increment is of the live nonce, not `nonce_seq+1` | Implemented |
| `test_key_zero_approval_rejects_live_nonce_overflow` | Pin key-zero overflow | Sender nonce at `MAX_NONCE_SEQ-1`, bumped by `CREATE` | Approval fails with no effects | Implemented |

## Introspection — `test_introspection.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_manager_direct_call_empty_revert` | Pin the manager's revert-only code | `CALL`, `STATICCALL`, and value-bearing `CALL` | All fail with zero returndata; the forced balance is unchanged | Implemented |
| `test_keyed_nonce_txparams` | Pin the four new TXPARAM indices | Sender nonce 7, keys `[1,2]` | Sequence 0, legacy nonce 7, count 2, hard-coded key-set hash, first key 1 | Implemented |
| `test_legacy_nonce_snapshot_survives_key_zero_approval` | Pin the pre-state legacy nonce snapshot | Key zero, sender nonce 7 | `0x0C` reports 7 while the live account nonce is already 8 | Implemented |

## Block ordering — `test_block_order.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_block_order_overlap_and_disjoint_domains` | Pin per-position sequence evaluation | Overlapping vs disjoint key sets from one sender | The overlapping second transaction is invalid at its block position; disjoint sets both commit | Implemented |

## TXPARAM index gap and scoping — `test_txparam_scoping.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_undefined_txparam_index_halts` | Prove `0x0F` and `0x11` stay undefined | One bytecode, three indices, identical gas | The defined `0x0D` control stores the canary and the count 2; the undefined indices halt exceptionally, discard both writes, and burn the whole frame gas limit | Implemented |
| `test_txparam_values_are_transaction_scoped` | Prove the parameters never move mid-transaction | Read all five, `CREATE`, read all five again | Both slot ranges hold the same values even though approval and `CREATE` advanced two different account nonces | Implemented |

## Block composition — `test_block_interactions.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_mixed_transaction_types_share_block` | Prove non-frame types are unaffected | Plain, keyed frame, and key-zero frame transactions in one block | Each domain advances independently and the plain target records its canary | Implemented |
| `test_keyed_validity_independent_of_legacy_nonce_advance` | Defeat the legacy-nonce cancellation strategy | One sender sends a plain transaction then a keyed frame transaction | Final account nonce is exactly 1 — the plain transaction's increment only — and the keyed transaction still commits | Implemented |

## VERIFY durability boundary — `test_verify_frame_durability.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_verify_revert_unrolls_keyed_consumption` | Bound the durability rule | Second `VERIFY` frame reverts vs completes | The reverting arm makes the whole transaction invalid and writes no slot; the completing arm writes `nonce_seq+1` | Implemented |

## Nonce manager call contexts — `test_manager_call_contexts_and_gas.py`

`test_introspection.py` covers a contract calling the manager. These cover the
contexts a contract call cannot reach and the consequences of the predeploy's
code being exactly five bytes. Replacing that code with a non-reverting
equivalent fails all three tests and nothing else in the suite.

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_manager_as_tx_entry_point` | Pin the revert at the transaction entry point and the code's exact cost | Type-2 transaction with `to = NONCE_MANAGER` and one wei | `cumulative_gas_used` equals the value-bearing intrinsic plus exactly 6; the manager keeps balance 0, so the wei was returned | Implemented |
| `test_manager_as_frame_target` | Pin the revert for a frame target beside a successful protocol consume | Approving `VERIFY` frame plus a `DEFAULT` frame targeting the manager | Frame fails having spent the fork's cold-account access plus exactly 6; the consumed slot still reads 1 and manager balance stays 0 | Implemented |
| `test_postfork_force_send_balance_inert` | Pin that a force-sent balance is unrecoverable and does not perturb keyed state | `SELFDESTRUCT` sends one wei to the manager, then the same frame calls it to recover | Recovery status and returndata length are both zero (each stored incremented, so a skipped probe fails), manager balance is 1 and its consumed slot reads 1 | Implemented |

## Decoder-phase wire format — `test_wire_encoding.py`

Emitted in the `transaction_test` fixture format, which ships the raw
envelope to the consuming client's decoder. Every arm is a single-byte-level
mutation of an accepted envelope, and each accepted envelope asserts that the
hand-written encoder reproduces the transaction's own serialization exactly,
so each rejection differs from a known-good envelope only in the named
defect.

Overriding the wire bytes is confined to rejection arms. The signing preimage
covers the whole payload, so an accepted arm assembled that way would ship a
signature over the superseded payload — valid-looking to the fixture,
rejected by every conformant client, and invisible to a fill that runs no
transition tool. Accepted arms vary the field through the structured
transaction instead and let the framework sign what it ships; the `accept`
helper refuses an override.

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_canonical_encoding_accepted` | Accepted control for the whole module | Canonical ten-field envelope, one key, sequence 0 | Decodes; the hand-assembled payload equals `tx.rlp()` byte for byte | Implemented |
| `test_non_canonical_integer_encoding_rejected` | Pin the canonical-RLP-integer clause | Leading zero byte and `0x00`-for-zero, applied to a key and to `nonce_seq` | All four rejected as invalid frame format | Implemented |
| `test_integer_field_width_bounds` | Pin the `2**256` and `2**64` field bounds | Transaction signed over the widest admitted value in the structured field, then that one field widened by a byte on the wire: 32- and 33-byte keys, 8- and 9-byte sequences | The wider member of each pair is rejected, the narrower accepted; each pair's serialization equals the hand-assembled envelope, so the wide value provably reaches the wire under a signature that covers it | Implemented |
| `test_nonce_keys_wrong_rlp_type_rejected` | Pin "`nonce_keys` is not an RLP list" | `nonce_keys` as a byte string, and a key wrapped in an inner list | Both rejected | Implemented |
| `test_payload_field_count_mismatch_rejected` | Pin the ten-field schema | `nonce_seq` dropped (nine fields); one item spliced in (eleven fields) | Both rejected | Implemented |
| `test_envelope_byte_length_mismatch_rejected` | Pin exact envelope consumption | Last byte removed; one byte appended | Both rejected | Implemented |

## Transaction-data pricing — `test_calldata_pricing.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_nonce_calldata_priced_into_transaction_gas` | Pin `nonce_calldata_cost` into `standard_gas_limit` | Six key/sequence shapes, an approving frame sized to the first-use surcharge and a trailing `INVALID` frame that burns its whole allocation | Hand-written `nonce_calldata` bytes per arm, and a `cumulative_gas_used` re-derived from the EIP-8141 formula; frame receipts pin that no gas went unused | Implemented |
| `test_nonce_calldata_counted_in_floor_above_standard_cost` | Pin `nonce_calldata_tokens` into `calldata_tokens` | One narrow and one 32-byte key, 512 bytes of frame data, floor asserted above standard | `cumulative_gas_used` equals the floor; the arms differ by the 32-byte key-width delta alone | Implemented |
| `test_calldata_floor_above_standard_cost_exceeds_gas_allowance` | Pin the same two rules on the rejecting side: `nonce_calldata` in the floor is what makes a transaction not fit | Widest key, 512 bytes of frame data, block gas allowance set in turn to the standard cost, the floor recomputed without `nonce_calldata`, one below the floor, and the floor | The first three are rejected for exceeding the gas allowance and the fourth is accepted; the second is the discriminating arm, accepted by any implementation that leaves `nonce_calldata` out of the floor | Implemented |

## Nonce-domain selection — `test_nonce_domain_selection.py`

Every other module exercises one branch of `current_nonce_seq` against a
state in which the other branch is empty, so an implementation reading or
writing the wrong account would still satisfy them. These seed the branch
that must be ignored with a value that disagrees.

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_key_zero_domain_ignores_manager_slot_zero` | Prove `[0]` selects the account nonce and never `slot(sender, 0)` | Account nonce 4 with a decoy 9 planted at `slot(sender, 0)`; sequences 4 and 9 | Sequence 4 is valid and advances the account nonce to 5; sequence 9 is rejected as too high; the decoy slot is untouched in both | Implemented |
| `test_keyed_consumption_writes_only_manager_storage` | Prove the consumed slot lives in the manager, not on the sender | One fresh key, externally owned sender | The manager slot holds 1 and the sender holds nothing at any slot | Implemented |

## Replay scope — `test_replay_scope.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_same_key_different_senders_are_independent` | Pin the sender component of the replay triple | Two senders, one shared key, sequence 0 for both, in one block | Two distinct slots each reach 1 and each transaction pays the full first-use surcharge; the slots are asserted unequal before use | Implemented |
| `test_consumed_slots_persist_and_replay_tuple_is_rejected` | Pin cross-block persistence and replay by tuple rather than by hash | Three blocks: keys 1/2, then disjoint key 3, then the first tuple again under a different priority fee | The disjoint block commits while keys 1/2 stay at 1; the replay is rejected as too low and its block commits nothing; all three slots end at 1 | Implemented |

## Stateful validity stage — `test_stateful_validity_stage.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_legacy_domain_tracks_preceding_transaction_in_block` | Prove `tx_legacy_nonce` is read at the transaction's block position | One block: an ordinary transaction then a `[0]` frame transaction; sequences 1 and 0 | Sequence 1 is valid, the sender ends at nonce 2, and both `0x01` and `0x0C` report the mid-block value 1; the block-start sequence 0 is rejected as too low | Implemented |
| `test_sequence_mismatch_precedes_all_frame_execution` | Prove the sequence check precedes every frame on both domains | Keyed slot planted at 5 and account nonce planted at 3, each with its exact and its `+1` sequence, plus a sentinel-writing frame | The exact arms write the sentinel and advance their domain; the `+1` arms are rejected and leave the sentinel slot zero | Implemented |

## Payload layout — `test_payload_layout.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_full_field_payload_layout_vector` | Pin the whole ten-field schema and the frame layout with nothing left at a default | Two keys of different widths, an eight-byte sequence, two frames covering all six frame subfields, a populated signature entry, distinct fee fields | The hand-assembled envelope equals the transaction's serialization, both keys advance to `nonce_seq + 1`, and the `SENDER` frame's target records its calldata and value | Implemented |
| `test_nonce_field_position_swap_rejected` | Prove `nonce_seq` is read at payload index 2 | `2**64` in `max_priority_fee_per_gas`, then that item exchanged with `nonce_seq` on the wire | Rejected: the value is legal in a `uint256` fee field and above the `uint64` sequence bound | Implemented |

## Key-set introspection at the maximum count — `test_key_set_introspection.py`

| Function Name | Goal | Setup | Expectation | Status |
| --- | --- | --- | --- | --- |
| `test_key_set_introspection_at_maximum_count` | Pin `0x0D` and `0x0E` at `MAX_NONCE_KEYS` | Sixteen keys: fifteen single-byte keys and `2**256-1` | Count 16, a hard-coded hash over the 544-byte preimage, first key 1, a derived non-zero sequence read, a canary, and sixteen slots at 1 | Implemented |

## Notes on fixture formats and checklist rendering

`test_wire_encoding.py` emits `transaction_test` fixtures. That format does
not invoke a transition tool while filling, so its rejection arms cannot
fail locally however wrong a decoder is; they are artifacts for consuming
clients. The module's accepted control does carry one live assertion --
that the hand-built canonical envelope equals the framework's own
serialization -- which pins the payload field order at fill time. Every
rule in that module that is also observable after decoding is pinned
against the transition tool elsewhere: the semantic key-list rules in
`test_validation.py`, and the reserved `nonce_seq` in
`test_invalid_nonce_fields`.

The generated checklist reports 30 of 30 applicable items covered, and one
template row renders blank.
`transaction_type/test/intrinsic_validity/data_floor_above_intrinsic_gas_cost`
appears on two consecutive template rows, once for the "invalid" variant
and once for the "valid" variant. The checklist tool indexes template items
by id into a dictionary, so only the second of the two rows is ever
addressable; no test marker, not-applicable line, or external-coverage line
can reach the first. The blank row is therefore a tool limitation and not a
coverage gap: both variants are covered, the valid one by
`test_nonce_calldata_counted_in_floor_above_standard_cost` and the invalid
one by `test_calldata_floor_above_standard_cost_exceeds_gas_allowance`.
