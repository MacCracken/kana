# Kana — Threat Model

## Trust Boundaries

Kana is a library crate — all inputs come from calling code (joshua, kiran, or direct consumers). No network I/O in core modules. The `ai` module makes HTTP requests (feature-gated).

## Attack Surface

### Input Validation

- **StateVector::new**: Validates dimension (power of 2), normalization (within NORM_TOLERANCE)
- **Operator::new**: Validates dim > 0, element count = dim²
- **DensityMatrix::new**: Validates element count. Optional `validate()` checks Hermiticity, trace-1, PSD
- **NoiseChannel::new**: Validates Kraus completeness (Σ E†E = I)
- **Circuit gate methods**: Validate qubit indices, target distinctness, operator dimensions
- **Measurement**: Validates random values in [0, 1)
- **try_zero**: Validates num_qubits in 1..=MAX_QUBITS, handles OOM gracefully

### Numerical Stability

- **Norm drift**: Periodic renormalization every 100 gates in circuit execution
- **Eigenvalue accuracy**: 2n×2n real embedding handles complex Hermitian matrices
- **Lindblad integration**: RK4 (4th order) instead of Euler for stability
- **Tolerance**: NORM_TOLERANCE = 1e-10 used consistently across all checks
- **State fidelity**: Exact for pure states, documented as approximate for mixed-mixed

### Memory Safety

- No `unsafe` code in the entire crate
- MAX_QUBITS = 28 prevents overflow in 2^n dimension calculations
- `try_reserve_exact` used for large state vector allocation (OOM → Result, not panic)
- `tensor_product` uses `checked_mul` for dimension overflow protection
- SparseOperator bounds-checks all (row, col) entries

### Denial of Service

- Large qubit counts (>20) create multi-GB state vectors — bounded by MAX_QUBITS
- Jacobi iteration bounded at n²×100 iterations
- Taylor series for matrix exponential bounded at 30 terms
- No recursion in hot paths

### Dependency Risk

| Dependency | Type | Unsafe | I/O | Risk |
|------------|------|--------|-----|------|
| serde | Serialization | No | No | Low |
| thiserror | Error derive | No | No | Low |
| tracing | Logging | No | No | Low |
| hisab | Math (optional) | No | No | Low |
| rayon | Parallelism (optional) | Yes (internal) | No | Low |
| reqwest | HTTP (optional) | Yes (TLS) | Yes | Medium |
| tokio | Async (optional) | Yes (internal) | Yes | Medium |

### AI Module (feature-gated)

- HTTP requests to configurable endpoints — no hardcoded URLs
- API key sent as Bearer token (not logged)
- Response JSON validated before field access
- HTTP status checked before deserialization
- 30-second timeout on client
