.PHONY: check fmt clippy test audit deny vet semver bench coverage build build-pgo doc clean all

# Run all CI checks locally
check: fmt clippy test audit

# Full check including supply-chain
all: check deny doc

# Format check
fmt:
	cargo fmt --all -- --check

# Lint (zero warnings)
clippy:
	cargo clippy --all-features --all-targets -- -D warnings

# Run test suite
test:
	cargo test --all-features

# Security audit
audit:
	cargo audit

# Supply-chain checks (cargo-deny)
deny:
	cargo deny check

# SemVer compatibility check
semver:
	cargo semver-checks check-release

# Run benchmarks with history
bench:
	./scripts/bench-history.sh

# Generate coverage report
coverage:
	cargo llvm-cov --all-features --html --output-dir coverage/
	@echo "Coverage report: coverage/html/index.html"

# Build release
build:
	cargo build --release --all-features

# Build with Profile-Guided Optimization (PGO)
# Step 1: build instrumented, Step 2: run benchmarks, Step 3: build optimized
build-pgo:
	@echo "=== PGO Step 1: Instrumented build ==="
	RUSTFLAGS="-Cprofile-generate=/tmp/kana-pgo" cargo build --release --all-features
	@echo "=== PGO Step 2: Gathering profile data from benchmarks ==="
	RUSTFLAGS="-Cprofile-generate=/tmp/kana-pgo" cargo bench --all-features -- --quick
	@echo "=== PGO Step 3: Merging profile data ==="
	llvm-profdata merge -o /tmp/kana-pgo/merged.profdata /tmp/kana-pgo/
	@echo "=== PGO Step 4: Optimized build ==="
	RUSTFLAGS="-Cprofile-use=/tmp/kana-pgo/merged.profdata -Cllvm-args=-pgo-warn-missing-function" cargo build --release --all-features
	@echo "=== PGO build complete ==="

# Generate documentation
doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# Clean build artifacts
clean:
	cargo clean
	rm -rf coverage/
