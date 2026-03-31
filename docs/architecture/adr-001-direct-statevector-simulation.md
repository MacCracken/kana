# ADR-001: Direct Statevector Simulation

**Status**: Accepted

## Context

Quantum circuit simulation traditionally expands each gate into a full 2^n × 2^n matrix via tensor products, then multiplies by the state vector. For n qubits, each gate application costs O(4^n) — prohibitively expensive beyond ~10 qubits.

## Decision

Apply gates directly to state amplitudes by iterating over bit-indexed pairs. A single-qubit gate on target qubit q modifies pairs (i, i|bit_q) where bit_q = 1 << (n-1-q). Each pair is a 2×2 matrix-vector multiply. Cost: O(2^n) per gate.

Extended to 2-qubit (4-element groups) and 3-qubit (8-element groups) gates.

## Key Choices

- **Branchless iteration attempted, reverted**: Bit manipulation to avoid the `if i & bit != 0 { continue }` branch was slower at small qubit counts because the branch predictor handles the alternating pattern perfectly. Branchless only wins at 12+ qubits.
- **Parallel threshold at 1024 (10 qubits)**: Below this, rayon overhead exceeds the parallelism benefit.
- **Matrix expansion kept as fallback**: Gates with >3 target qubits still use the legacy O(4^n) path.

## Consequences

- 10-58x speedup on circuit benchmarks (QFT 3q: 10.5μs → 182ns)
- No allocations per gate (in-place amplitude modification)
- The `expand_gate` / `Operator::apply` path remains for the hisab bridge and testing
