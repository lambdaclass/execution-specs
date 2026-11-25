"""
Tests for
[EIP-8504: Proof Frames for leanSPHINCS Signatures](https://eips.ethereum.org/EIPS/eip-8504).

The EIP is a delta against EIP-8141 frame transactions: a frame mode whose
frame executes no code and declares the signatures the transaction depends
on, and a list of those signatures in the block, committed to by the header.
"""
