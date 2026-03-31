# ADR-002: Compile-Time No-Cloning via Rust Ownership

**Status**: Accepted

## Context

The quantum no-cloning theorem states that an arbitrary unknown quantum state cannot be copied. Most quantum libraries enforce this only at runtime (or not at all). Rust's ownership model can enforce it at compile time — a qubit consumed by a gate cannot be reused.

## Decision

Added `safe::QuantumBuilder` with move-only `Qubit` handles. Gates consume input qubits and return output qubits. The borrow checker prevents reuse of consumed qubits.

## Key Choices

- **Parallel API, not replacement**: The existing index-based `Circuit` API remains for algorithm builders (Grover, QFT) where qubit indices are computed dynamically. The safe builder is the recommended API for user-facing circuit construction.
- **Const-generic `build()`**: `build::<N>([q0, q1, ...])` requires returning all N qubits, proving none were lost or duplicated.
- **SWAP returns swapped handles**: `swap(a, b)` returns `(b, a)` — the logical qubit identity follows the physical swap.
- **Measurement returns both**: `measure(q)` returns `(ClassicalBit, Qubit)` — the qubit persists (collapsed) post-measurement, matching physical reality.

## Consequences

- Compile-time no-cloning: `let _copy = b.h(q0); let _again = b.x(q0);` is a compiler error
- Zero runtime overhead: `Qubit` is a newtype over `usize`, same as index-based API
- API is more verbose than index-based (must thread qubits through every gate)
