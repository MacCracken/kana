# Kana — Dependency Watch

## Core Dependencies

### serde (1.x)
- **Status**: Pinned to ^1, default features enabled
- **Note**: Used for Serialize/Deserialize on all core types. Hard dependency.

### thiserror (2.x)
- **Status**: Pinned to ^2
- **Note**: Error derive macro only. Zero runtime cost.

### tracing (0.1.x)
- **Status**: Pinned to ^0.1
- **Note**: Unconditional dependency but compiles to no-ops without a subscriber. Structured logging when `logging` feature enables `tracing-subscriber`.

## Optional Dependencies

### hisab (1.4.x)
- **Status**: Optional, gated behind `hisab-bridge` feature
- **Note**: AGNOS math library. Provides ComplexMatrix, eigendecomposition, kronecker, matrix_exp. Only `num` feature used.
- **AGNOS owned**: Yes

### rayon (1.x)
- **Status**: Optional, gated behind `parallel` feature
- **Note**: Work-stealing parallelism for gate application on large statevectors. Auto-threshold at 1024 amplitudes (10 qubits).

### reqwest (0.12.x)
- **Status**: Optional, gated behind `ai` feature
- **Note**: HTTP client for daimon/hoosh registration. JSON feature enabled.

### tokio (1.x)
- **Status**: Optional, gated behind `ai` feature
- **Note**: Async runtime. Only `rt-multi-thread` and `macros` features — NOT `full`.

### serde_json (1.x)
- **Status**: Optional (ai feature) + dev-dependency
- **Note**: JSON parsing for AI responses and test serialization.

### tracing-subscriber (0.3.x)
- **Status**: Optional, gated behind `logging` feature
- **Note**: Subscriber with `env-filter` and `fmt` features.

## Dev Dependencies

### criterion (0.5.x)
- **Note**: Benchmarking with HTML reports. Not shipped to consumers.

### serde_json (1.x)
- **Note**: Test serialization roundtrips. Not shipped to consumers.
