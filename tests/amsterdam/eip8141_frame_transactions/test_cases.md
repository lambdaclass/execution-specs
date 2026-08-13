# Frame transaction test cases

Test plan for [EIP-8141](https://eips.ethereum.org/EIPS/eip-8141).

Consensus rules only: the EIP's mempool sections (validation-prefix
shapes, banned opcodes during validation, paymaster admission,
replacement and eviction) and networking sections (blob sidecar
wrappers, receipt message mirroring) govern gossip admission, not
block validity, and cannot be expressed as fill fixtures. Consensus
cousins are covered instead — e.g. the payer set after non-`VERIFY`
frames and a `VERIFY` frame after `SENDER` frames are consensus-legal
and deliberately exercised, proving consensus does not enforce
mempool policy. The runtime-unreachable defense-in-depth arm of the
`APPROVE` target restriction is enforced statically (the execution
flag rule already forces the target), so only the static arm is
observable.

| Function Name | Goal | Setup | Expectation | Status |
| - | - | - | - | - |
| `test_invalid_tx_fields` | Transaction-level static bounds | One field varied per arm | Reject, or accept at the exact boundary | ✅ |
| `test_frame_constraints` | Per-frame static constraints | One frame field varied per arm | Reject, or accept the nearest-valid variant | ✅ |
| `test_expiry_verifier_constraints` | Expiry frame shape rules | Expiry frame varied per arm | Reject; shape rules bind only VERIFY frames at the predeploy | ✅ |
| `test_signature_constraints` | Entry structural and crypto validation | One appended entry varied per arm | Reject with range vs recovery class split | ✅ |
| `test_gas_limit_cap_from_frame_gas` | Gas cap on the frame-gas anchor | Frame gas at cap and above | Accept at cap, reject one above | ✅ |
| `test_gas_limit_cap_from_calldata_floor` | Gas cap on the floor anchor | Zero-byte data across the cap | Cap binds the larger anchor | ✅ |
| `test_nonce_at_maximum` | Highest usable nonce | Sender at overflow bound minus one | Accepted, incremented | ✅ |
| `test_transfer_with_default_code` | Default-code transfer flow | VERIFY approval plus SENDER value | Payer, statuses, balances | ✅ |
| `test_contract_sender_approves` | Contract sender approval | Sender code calls APPROVE | Executes SENDER frame | ✅ |
| `test_eoa_paymaster` | Sponsored fees via default code | Split execution and payment approvals | Payer is the sponsor, sender balance untouched | ✅ |
| `test_atomic_batch_rollback` | Batch rollback both directions | Failing first or last batch member | Unrolled writes, statuses, skipped frame | ✅ |
| `test_sender_frame_before_approval` | Execution approval precondition | SENDER frame first | Invalid transaction | ✅ |
| `test_verify_frame_reverts` | VERIFY failure invalidates | Reverting VERIFY frame | Invalid transaction | ✅ |
| `test_txparam` | Transaction selector readbacks | Probe stores each selector | Literal field values | ✅ |
| `test_txparam_max_cost` | Escrow equals reported max cost | Balance comparison in-frame | Equality flag stored | ✅ |
| `test_txparam_sender_and_sig_hash` | Sender and hash readbacks | Probe stores selectors | Sender exact, hash nonzero, digest exact | ✅ |
| `test_frameparam` | Frame selector readbacks | Probe stores each selector | Literal frame fields | ✅ |
| `test_frameparam_resolved_target` | Resolved target readback | Null and explicit targets | Sender and target addresses | ✅ |
| `test_frameparam_atomic_batch_set` | Atomic bit readback | Flagged frame | Bit and raw flags | ✅ |
| `test_frameparam_halts` | Frame selector halts | Current status, OOB, undefined | Marker rolled back | ✅ |
| `test_introspection_halts` | Selector halts across opcodes | Undefined and OOB reads | Marker rolled back | ✅ |
| `test_framedataload` | Frame data word reads | Offsets across the end | Zero-padded words | ✅ |
| `test_framedatacopy` | Frame data copies | Straddling the end | Zero-filled memory | ✅ |
| `test_sigparam` | Signature metadata readbacks | Protocol and arbitrary entries | Scheme, length, msg | ✅ |
| `test_sigparam_resolved_signer` | Resolved signer readback | Protocol vs arbitrary entry | Signer, or halt | ✅ |
| `test_sigparam_copy_arbitrary` | Raw witness copy | Arbitrary vs protocol entry | Bytes copied, or halt | ✅ |
| `test_sender_is_warm` | Sender seeds the warm set | BALANCE probe on sender | Warm access measured | ✅ |
| `test_coinbase_is_warm` | Coinbase warm per frame | BALANCE probe on coinbase | Warm access measured | ✅ |
| `test_frame_target_entry_charge` | Entry access charge | Fresh target, repeat frames | Cold then warm within frame gas | ✅ |
| `test_warmth_carry_to_next_frame` | Warm journal carry and unwind | Toucher outcome varied | Warm iff toucher succeeded | ✅ |
| `test_warmth_from_inner_call` | Depth-two warmth propagation | Child and frame outcomes varied | Warm iff both succeeded | ✅ |
| `test_expiry_verifier_frame` | Expiry boundary timestamps | Deadline around block time | Valid at, invalid past | ✅ |
| `test_nonce_mismatch` | State nonce equality | Contract sender, nonce ±1 | Exact accepted, high and low classes | ✅ |
| `test_nonce_and_signature_both_invalid` | Nonce vs signature check order | Both violated at once | Invalid; either class (order divergence recorded) | ✅ |
| `test_wrong_chain_id` | Chain id equality | Chain id off by one | Rejected with chain id class | ✅ |
| `test_max_fee_below_base_fee` | Inherited base-fee gate | Cap at and below base fee | Accept at, reject below | ✅ |
| `test_max_cost_overflow` | Max cost word bound | Product at 2**256 and one fee below | No balance covers; one below approves | ✅ |
| `test_frame_caller_and_origin` | Caller and origin per mode | Probes at depths one and two | Entry point or sender everywhere | ✅ |
| `test_caller_and_origin_in_verify_frame` | VERIFY context identity | Conditional OOG checker | Acceptance is the oracle | ✅ |
| `test_origin_in_non_frame_transaction` | Origin negative control | Type-two transaction | Signing sender at both slots | ✅ |
| `test_callvalue_in_sender_frame` | Frame value visibility | SENDER value, DEFAULT zero | Value plus one stored, transferred | ✅ |
| `test_transient_storage_cleared_between_frames` | Transient isolation | TSTORE then next-frame TLOAD | Same-frame visible, cross-frame cleared | ✅ |
| `test_sender_value_balance_boundary` | Escrow-then-transfer boundary | Value at remaining balance and one above | Exact succeeds, one more reverts frame | ✅ |
| `test_payer_never_set` | Payment approval required | No or execution-only approval | Invalid after all frames ran | ✅ |
| `test_payment_before_execution_approval` | Payment needs execution first | Payment-only first frame | Invalid transaction | ✅ |
| `test_approve_scope_refusals` | Scope mask and preconditions | Probing frame per refusal shape | Frame reverts, marker discarded, tx valid | ✅ |
| `test_approve_strict_subset_scope` | Subset scopes allowed | Execution-only under both flags | Approved; later payment completes | ✅ |
| `test_approve_duplicate_scopes` | Duplicate approvals refused | Both, execution, third-party payment again | Frames revert; originals survive two SENDER frames | ✅ |
| `test_approve_return_data` | APPROVE return region | Child self-call with marker | Size and bytes observed by parent | ✅ |
| `test_approve_from_child_call_reverts_callee` | Foreign-address APPROVE | Called contract approves | Callee reverts, nothing leaks | ✅ |
| `test_approve_from_delegatecall` | Delegated-context APPROVE | Helper via DELEGATECALL | Approval effective from depth two | ✅ |
| `test_approve_in_initcode_reverts_create` | Initcode APPROVE | CREATE with approving initcode | Creation reverts, canonical approval unaffected | ✅ |
| `test_approve_memory_expansion_gas` | APPROVE gas | Return regions of varied size | Warm entry plus code plus expansion only | ✅ |
| `test_verify_frame_static_restrictions` | VERIFY static context | One write op per arm | Invalid; read-only control passes | ✅ |
| `test_verify_frame_after_sender_frame_unwinds` | Late VERIFY voids execution | Trailing reverting VERIFY | Writes, nonce, and balance fully restored | ✅ |
| `test_payment_balance_boundary` | Payer balance boundary | Balance at max cost and one below | Exact approves, one below invalidates | ✅ |
| `test_opcodes_undefined_in_other_tx_types` | Type-scoped instructions | Each opcode in types 0, 2, 4 | Halt, marker rolled back, full gas | ✅ |
| `test_exact_gas_accounting_standard` | Standard gas path terms | Mixed-byte frame data | Cumulative, per-frame gas, balances | ✅ |
| `test_exact_gas_accounting_floor_dominant` | Floor as final gas | Zero-heavy frame data | Cumulative equals composed floor | ✅ |
| `test_signature_gas_constants` | Per-scheme verification gas | One and three embedded entries | Cumulative solves the constants linearly | ✅ |
| `test_frame_oog_isolation` | Frame budget isolation | Exact and exact-minus-one budgets | Success and forfeited limit | ✅ |
| `test_skipped_frame_gas_refund` | Skipped gas returned | Huge-limit skipped frame | Cumulative excludes the allotment | ✅ |
| `test_refund_accounting` | Refund quotient cap | Burner sizes total under, at, over | Applied refund clamped correctly | ✅ |
| `test_refund_discarded_with_revert` | Refund discard on revert | Clear then frame or child revert | No refund term, slot restored | ✅ |
| `test_multiple_atomic_batches` | Disjoint batch extents | Two batches, second fails | First survives, second unrolled | ✅ |
| `test_large_batch_unroll` | Batch unroll at scale | Sixty-two frame batch | Statuses kept, logs and write dropped | ✅ |
| `test_atomic_batch_unwinds_warmth` | Journal rollback | Toucher inside failed batch | Cold access measured after | ✅ |
| `test_introspection_gas_costs` | Constant instruction gas | Runtime measurement per opcode | Spec literals | ✅ |
| `test_framedatacopy_gas` | Copy gas formula | Lengths across word boundaries | Base, per-word, expansion | ✅ |
| `test_framedatacopy_huge_length_out_of_gas` | Copy expansion OOG | Unpayable length | Frame fails, marker rolled back | ✅ |
| `test_framedataload_future_frame_and_max_offset` | Static frame data reads | Later frame, maximal offset | Word read, zero at max offset | ✅ |
| `test_introspection_stack_underflow` | Underflow halts | One-short stack per opcode | Marker rolled back | ✅ |
| `test_introspection_in_subcontexts` | Depth-two introspection | Every call kind | Identical values | ✅ |
| `test_introspection_in_initcode` | Initcode introspection | Creating frame | Created storage holds index | ✅ |
| `test_frameparam_status_of_unrolled_and_skipped_frames` | Failure and skipped statuses | Probe after failed batch | Statuses plus one stored | ✅ |
| `test_sig_hash_pinned` | Canonical hash preimage | Static payload, in-test RLP | Reported hash equals hand-computed | ✅ |
| `test_sig_hash_elides_empty_msg_bytes` | Empty-msg byte elision | Entry signed over elided preimage in-test | Valid regardless of witness size | ✅ |
| `test_expiry_runtime_length_check` | Runtime length and deadline | DEFAULT frames to the predeploy | Statuses per arm, tx valid | ✅ |
| `test_expiry_verifier_from_child_call` | Predeploy from depth two | Every call kind | Success, empty return data | ✅ |
| `test_expiry_verifier_all_zero_deadline` | Zero deadline | All-zero expiry data | Frame reverts | ✅ |
| `test_expiry_verifier_receives_value_from_sender_frame` | Value to the predeploy | SENDER frame with value | Balance sticks | ✅ |
| `test_expiry_verifier_behind_delegation` | Predeploy code via delegation | Delegated target frame | Succeeds unexpired | ✅ |
| `test_expiry_verifier_exact_gas` | Hand-decoded success gas | Exact and exact-minus-one budgets | Pinned gas; short budget invalidates | ✅ |
| `test_blob_frame_transaction_acceptance` | Blob counts accepted | One and maximum blobs | Valid | ✅ |
| `test_blob_fee_inclusion_gate` | Blob fee cap gate | Cap at and below solved base fee | Accept at, reject below | ✅ |
| `test_blob_fee_settlement` | Blob fee charged at base | Cap far above base fee | Payer debit exact | ✅ |
| `test_blobhash_across_frames` | Transaction-level hashes | Two frames read all indices | Same hashes, zero out of range | ✅ |
| `test_delegated_sender_bypasses_default_code` | Delegate replaces default code | Approving and reverting delegates | Valid and invalid respectively | ✅ |
| `test_delegated_target_in_frames` | Delegated frame targets | DEFAULT and SENDER frames | Delegate runs on the account's storage | ✅ |
| `test_default_code_rejects_empty_scope` | Empty allowed scope | VERIFY frame with zero flags | Invalid transaction | ✅ |
| `test_default_code_signature_requirements` | Entry requirements at the index | No, wrong-scheme, wrong-form entries | Invalid transaction per arm | ✅ |
| `test_default_code_signature_index_selection` | Scope-selected index | Sender and payer entries swapped | Invalid; lookup is by index | ✅ |
| `test_signature_component_edges` | Component bound widths | One component substituted per arm | Range vs recovery class split | ✅ |
| `test_mixed_transaction_block` | Mixed-type receipts | Legacy, fee-market, frame txs | Cumulative gas chains | ✅ |
| `test_invalid_frame_transaction_invalidates_block` | Block-level invalidity | Invalid frame tx last | Whole block rejected | ✅ |
| `test_code_bearing_sender_exemption_is_type_scoped` | Origin ban exemption | Same account, both types | Frame accepted, fee-market block rejected | ✅ |
| `test_logs_concatenation` | Frame-ordered logs | Three loggers, middle reverts | Surviving logs in order | ✅ |
| `test_deploy_then_use` | Cross-frame deployment | Factory frame then call frame | Hand-derived address executes | ✅ |
| `test_block_gas_pool_returns_unused` | Unused gas returns to the pool | Follow-up tx sized to the remainder | Fits at the exact limit, not one below | ✅ |
| `test_admission_constraints` | Rules needing block or account context | Maximum cost across the word bound | Reject, or accept within the bound | ✅ |
| `test_atomic_batch_restores_prior_warmth` | Pre-batch journal restored on unroll | Probe frame before and after a failing batch | Both accesses measure warm afterwards | ✅ |
| `test_invalid_tx_fields_transaction` | Same field bounds, transaction level | Transaction fixtures of the field arms | Same verdicts without a block | ✅ |
| `test_frame_constraints_transaction` | Same frame rules, transaction level | Transaction fixtures of the frame arms | Same verdicts without a block | ✅ |
| `test_expiry_verifier_constraints_transaction` | Same expiry rules, transaction level | Transaction fixtures of the expiry arms | Same verdicts without a block | ✅ |
| `test_signature_constraints_transaction` | Same entry rules, transaction level | Transaction fixtures of the entry arms | Same verdicts without a block | ✅ |
| `test_gas_limit_cap_from_frame_gas_transaction` | Same cap rule, transaction level | Transaction fixtures of the cap arms | Same verdicts without a block | ✅ |
| `test_gas_limit_cap_from_calldata_floor_transaction` | Same floor cap rule, transaction level | Transaction fixtures of the floor arms | Same verdicts without a block | ✅ |
| `test_payload_structure` | Payload field list | Raw payloads with wrong element counts | Decoder rejects each shape | ✅ |
| `test_payload_field_encoding` | Payload field bounds | One field encoded out of range per arm | Decoder rejects each field | ✅ |
| `test_frame_item_encoding` | Frame tuple field list | Raw frame items with wrong shapes | Decoder rejects each shape | ✅ |
| `test_signature_item_encoding` | Signature tuple field list | Raw entry items with wrong shapes | Decoder rejects each shape | ✅ |
| `test_precompile_target` | Precompile as a frame target | Frame targeting a precompile | Precompile runs, input-priced gas | ✅ |
| `test_precompile_target_rejecting_its_input` | Precompile input rejection | Malformed precompile input | Frame halts, whole limit forfeited | ✅ |
| `test_verify_frame_precompile_target` | VERIFY frame at a precompile | Precompile as the verify target | Default code runs and reverts | ✅ |
| `test_delegated_target_entry_charge` | Designation access at entry | Delegated target, warm and cold | Both accesses charged at entry | ✅ |
| `test_dead_target_entry_charge` | Reviving a dead target | Value transfer to a non-alive target | New-account state gas at entry | ✅ |
| `test_delegated_to_precompile_target` | Designation to a precompile | Target designating a precompile | Designation disables dispatch | ✅ |
| `test_verify_frame_delegated_to_precompile_target` | VERIFY frame designating a precompile | Delegated verify target | Empty resolved code, no default code | ✅ |
| `test_bal_atomic_batch_write` | BAL entry of a batch write | Committed and unrolled batches | Write recorded only when it commits | ✅ |
| `test_bal_atomic_batch_skipped_frame_absent` | BAL omits skipped frames | Failed batch with a skipped member | Skipped target absent from the BAL | ✅ |
| `test_bal_frame_revert_write_dropped` | BAL entry of a reverted write | Reverting non-batch frame | Slot re-filed as a bare access | ✅ |
| `test_bal_unaffordable_designation_absent` | BAL omits unreached designations | Frame gas below the designation access | Designated address absent | ✅ |
| `test_bal_sponsored_payer_and_sender` | BAL attribution when sponsored | Non-sender payer | Fee change and nonce bump on distinct accounts | ✅ |
| `test_expiry_verifier_from_initcode` | Predeploy called from initcode | `CREATE` in a frame, and a creating tx | Succeeds in both creation contexts | ✅ |
| `test_framedatacopy_memory_bounds` | Copy size and offset bounds | Zero size at the max offset, huge sizes | No expansion, or halt on expansion | ✅ |
| `test_introspection_gas_boundary` | Introspection gas requirement | Exact, one over, one under | Exact usage twice, halt below | ✅ |
| `test_high_s_range_bound` | First `s` above the low-half bound | Half plus one, and above the order | Rejected on the range check | ✅ |
| `test_refund_discarded_with_halt` | Refund lost to an exceptional halt | Clear then out-of-gas or invalid opcode | No refund term, whole limit charged | ✅ |
| `test_refund_floored_by_calldata_cost` | Refund below the calldata floor | Data sized to straddle the floor | Floor is the final gas | ✅ |

## Deliberately unpinned

The fate of the `payer` and `sender_approved` approval fields when an
atomic batch unrolls is unstated by the pinned spec — it says only
that the *state* is rolled back to the condition before the batch —
so no test here pins whether an approval granted inside a failing
batch survives its unroll. `SPEC_FEEDBACK.md` records the question
(R-101/R-106/R-161); the adjacent, settled half of the rule — that
the shared warm journal is restored rather than emptied — is pinned
by `test_atomic_batch_restores_prior_warmth`.
