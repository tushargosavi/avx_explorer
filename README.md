# AVX2/AVX512 Simulator

A Rust-based REPL simulator for AVX2 and AVX512 instruction sets with BMI2 extensions. This project provides an interactive environment to experiment with SIMD (Single Instruction, Multiple Data) operations without requiring actual hardware support.

## Features

- **Interactive REPL**: Test AVX2/AVX512 instructions in real-time
- **Comprehensive Instruction Support**: Implements common AVX2 operations including:
  - Set and broadcast operations (`_mm256_set1_epi8`, `_mm256_set1_epi32`)
  - Load and store operations
  - Arithmetic operations (addition, subtraction)
  - Logical operations (AND, OR, XOR)
  - Comparison operations
  - Permutation and shuffle operations
- **BMI2 Extensions**: Includes advanced bit manipulation instructions like `pdep`
- **Memory Management**: Variable storage and retrieval
- **Error Handling**: Clear error messages for invalid operations

## Dependencies

- `chumsky` - Parser combinator library
- `ariadne` - Error reporting and diagnostics
- `rustyline` - Enhanced readline functionality
- `bitvec` - Bit-level manipulation utilities

## Installation

```bash
git clone <repository-url>
cd avx_sim
cargo build --release
```

## Usage

Run the interactive REPL:

```bash
cargo run
```

The simulator will start with a prompt where you can enter AVX2 instructions:

```
AVX2/AVX512 Simulator REPL
Type 'exit' to quit
prompt>
```

### Example Usage

```rust
// Set all elements to a specific value
prompt> _mm256_set1_epi8(42)
[42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]

// Store result in a variable
prompt> let x = _mm256_set1_epi32(100)

// Use stored variable
prompt> _mm256_add_epi32(x, x)
[200, 200, 200, 200, 200, 200, 200, 200]
```

## Supported Instructions

### AVX2 Operations
- `_mm256_set1_epi8` - Broadcast 8-bit integer to all elements
- `_mm256_set1_epi32` - Broadcast 32-bit integer to all elements
- `_mm256_add_epi32` - Add packed 32-bit integers
- `_mm256_sub_epi32` - Subtract packed 32-bit integers
- `_mm256_and_si256` - Bitwise AND
- `_mm256_or_si256` - Bitwise OR
- `_mm256_xor_si256` - Bitwise XOR
- `_mm256_cmpeq_epi32` - Compare packed 32-bit integers for equality
- `_mm256_permutevar8x32_epi32` - Permute 32-bit integers
- `_mm256_shuffle_epi8` - Shuffle bytes using control mask

### BMI2 Operations
- `pdep` - Parallel bits deposit

## Project Structure

```
src/
├── main.rs         - REPL interface and main entry point
├── ast.rs          - Abstract Syntax Tree definitions
├── parser.rs       - Instruction parsing logic
├── interpreter.rs  - Instruction execution engine
├── avx2.rs         - AVX2 instruction implementations
├── bmi2.rs         - BMI2 instruction implementations
├── display.rs      - Result formatting and display
└── tests.rs        - Unit tests
```

## Requirements

- Rust 2024 edition or later
- CPU with AVX2 support (for runtime detection)
- 64-bit system

## Testing

Run the test suite:

```bash
cargo test
```

## License

This project is licensed under the MIT License.