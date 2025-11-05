# PowerPC MMA Dialect for MLIR - Project Specification

**Version**: 2.0 (Integrated)  
**Owner**: Tony Varghese  
**Date**: 2025-01-XX  
**Status**: Phase 0 - Investigation in Progress

---
## Executive Summary

This project introduces a PowerPC-specific MLIR dialect that provides first-class support for IBM POWER10 Matrix Multiply Assist (MMA) instructions, following the spirit of ARM's SME dialect. The goal is to enable a **compiler-only, layered optimization strategy** where C/C++/Fortran code (via Polygeist) is lifted to high-level MLIR dialects (`scf`/`affine`/`linalg`), optimized using existing MLIR infrastructure (tiling, packing, fusion), and then lowered through a new `ppc_mma` dialect to LLVM IR with PowerPC MMA intrinsics.

**Key Insight**: Rather than building everything from scratch, we **reuse existing MLIR components** (`tensor.pack`, `linalg.tile`, fusion, bufferization) and add only the minimal target-specific dialect needed for MMA-aware lowering.

**Target Outcome**: Measurable performance uplift for matrix kernels (GEMM, BRGEMM, fused epilogues) on POWER10+, achieving within 90-96% of hand-tuned BLAS performance while remaining fully compiler-based.

---

## 1. Project Goals

### Primary Goals
1. Create a **`ppc_mma` MLIR dialect** with operations representing MMA hardware capabilities
2. Implement **conversion passes** to lower from high-level dialects to `ppc_mma` to LLVM IR
3. Enable **compiler-driven optimization** without manual tuning or library dependencies
4. Achieve **performance competitive** with OpenBLAS/ESSL (target: 90-96% of peak)
5. Provide **extensible framework** for future PowerPC variants and additional BLAS operations

### Secondary Goals
- Demonstrate compiler-only approach viability for modern matrix engines
- Contribute upstream to LLVM/MLIR community
- Establish patterns reusable for other architectures (Intel AMX, future designs)
- Enable cross-function and cross-module matrix optimizations

---

## 2. Scope

### 2.1 In Scope (Phase 1: MVP)

**Dialect Components**:
- `ppc_mma` dialect with types, operations, and attributes
- Accumulator (ACC) types representing MMA register state
- Operations for rank-k updates, accumulator lifecycle, and data movement
- Layout attributes for packed matrix tiles

**Target Operations**:
- **GEMM** (General Matrix Multiply): `C = α·(A·B) + β·C`
- **BRGEMM** (Batched/Reduced GEMM): Multiple k-iterations with accumulation
- **Fused epilogues**: bias addition, ReLU activation, scaling

**Data Types** (Phase 1):
- **f32** (single-precision): Primary target
- **bf16** (brain float 16): With f32 accumulation
- Hooks for f16, f64, integers (deferred to Phase 2)

**Compilation Pipeline**:
- Polygeist (C/C++ → MLIR)
- High-level optimizations (tiling, packing, fusion) using existing MLIR
- Conversion to `ppc_mma` dialect
- Lowering to LLVM IR with MMA intrinsics
- PowerPC backend code generation

**Infrastructure**:
- Conversion passes: `ConvertLinalgToPPCMMA`, `LowerPPCMMAToLLVM`
- Heuristics for tile size selection
- Unit tests and integration tests
- Microbenchmark harness

### 2.2 Out of Scope (Phase 1)

**Deferred to Later Phases**:
- Full BLAS coverage (SYRK, SYR2K, TRMM, etc.) - Phase 2
- Additional data types (f16, f64, i8, i16) - Phase 2
- GPU lowering and offloading - Future
- Sophisticated auto-tuner - Future
- Multi-threading and OpenMP (basic support only in Phase 1)
- Cross-platform simulation framework - Future

### 2.3 Non-Goals

- Replacing existing BLAS libraries (complementary approach)
- Supporting non-POWER architectures (design should be portable though)
- Runtime tuning or dynamic dispatch
- Automatic parallelization (manual hints accepted)

---

## 3. Background and Motivation

### 3.1 Why a Compiler-Only Approach?

**Library Limitations**:
- Requires manual installation and maintenance
- Architecture-specific hand-written assembly for each target
- Function call overhead significant for small matrices
- No cross-module optimization opportunities
- Time lag between new hardware and library support
- Manual code changes needed to use libraries

**Generic Compiler Limitations**:
- Auto-vectorization doesn't understand MMA semantics
- Misses accumulator-centric optimization opportunities
- Sub-optimal data layouts and tiling strategies
- Polyhedral optimizers (PLuTo, Polly) 5-20× slower than optimal

**Research Evidence**:
The paper "Fast Matrix Multiplication via Compiler-only Layered Data Reorganization" demonstrates:
- 96% of BLAS peak performance achievable
- 2.6× faster than VSX vectorization
- 6-22× faster than polyhedral approaches
- Within 10% of hand-tuned OpenBLAS for large matrices
- Better than Eigen for most cases

### 3.2 Why MLIR?

**Existing Infrastructure**:
- **`tensor.pack/unpack`**: Data layout transformation
- **Linalg tiling**: Cache-aware loop restructuring
- **Fusion**: Combine operations to reduce memory traffic
- **One-shot bufferization**: Efficient tensor→memref lowering
- **Multiple dialects**: Clean separation of concerns

**Benefits**:
- Reuse proven transformations instead of rebuilding
- Clean abstraction layers
- Upstream community support
- Extensible to other targets

### 3.3 The Layered Strategy

Following Goto & Van de Geijn's classical BLAS optimization:

**Macro Kernel** (Target-Independent):
- Cache-aware blocking (L1, L2, L3)
- Data packing for locality
- Loop restructuring for reuse
- Epilogue fusion

**Micro Kernel** (Target-Specific):
- Accumulator management
- MMA instruction emission
- Register allocation and scheduling
- Hardware constraint satisfaction

**Key Principle**: Clean separation between portable high-level optimizations and hardware-specific low-level code generation.

---

## 4. Technical Architecture

### 4.1 Compilation Pipeline
```
┌──────────────────────┐
│   C/C++/Fortran      │
│   (Polygeist)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  SCF / Affine        │  ◄── Polyhedral representation
│  (optional: Linalg)  │      Loop-based IR
└──────────┬───────────┘
           │
           │ Existing MLIR Optimizations
           ▼
┌──────────────────────┐
│  Linalg (tiled)      │  ◄── linalg.tile
│  + Tensor.pack       │      tensor.pack/unpack
│  + Fused epilogues   │      linalg-fuse-elementwise-ops
└──────────┬───────────┘
           │
           │ one-shot-bufferize
           ▼
┌──────────────────────┐
│  Linalg (memref)     │  ◄── Bufferized form
│  + Packed layouts    │
└──────────┬───────────┘
           │
           │ convert-linalg-to-ppc-mma (NEW)
           ▼
┌──────────────────────┐
│  PPC MMA Dialect     │  ◄── NEW: This project
│  (ACC ops)           │      Hardware-aware representation
└──────────┬───────────┘
           │
           │ lower-ppc-mma-to-llvm (NEW)
           ▼
┌──────────────────────┐
│  LLVM Dialect        │  ◄── MMA intrinsics
│  (MMA builtins)      │      llvm.ppc.mma.*
└──────────┬───────────┘
           │
           │ LLVM Backend (existing)
           ▼
┌──────────────────────┐
│  PowerPC Assembly    │
│  (MMA instructions)  │
└──────────────────────┘
```

### 4.2 Proposed Pass Pipeline

**MLIR Pipeline** (as command-line flags):
```bash
mlir-opt input.mlir \
  # Standard cleanup
  -canonicalize -cse \
  
  # Linalg tiling (cache-aware)
  -linalg-tile="tile-sizes=16,16,128" \
  
  # Epilogue fusion (before packing)
  -linalg-fuse-elementwise-ops \
  
  # Packing transformation
  -tensor-pack-pipelines \
  -fold-tensor-pack-unpack \
  
  # Bufferization (tensor → memref)
  -one-shot-bufferize \
  
  # NEW: Convert to PPC MMA
  -convert-linalg-to-ppc-mma \
  
  # NEW: Lower to LLVM intrinsics
  -lower-ppc-mma-to-llvm \
  
  # Optional: OpenMP parallelization
  -convert-scf-to-openmp \
  
  # LLVM dialect finalization
  -llvm-legalize-for-opaque-ptr \
  -finalize-memref-to-llvm \
  
  -o output.mlir
```

**Key Insight**: Only the last two MMA-specific passes are new. Everything before uses existing, well-tested MLIR infrastructure.

### 4.3 Why This Order?

**Early**: `canonicalize`, `cse`
- Simplify IR before transformations
- Remove redundancy

**Middle**: Tiling → Fusion → Packing
- **Tile first**: Create BRGEMM structure (k-loop with accumulation)
- **Fuse next**: Combine epilogues before packing
- **Pack last**: Apply final data layout transformations
- This order minimizes packing overhead and maximizes fusion opportunities

**Late**: Bufferization → MMA Lowering
- Bufferize once, after all tensor-level optimizations
- PPC MMA operates on memref (hardware addresses)
- LLVM lowering is mechanical translation to intrinsics

---

## 5. Dialect Design: `ppc_mma`

### 5.1 Design Philosophy

**Explicit State Management**:
- Accumulators (ACCs) are first-class types
- Lifetime and dataflow explicitly represented
- Hardware constraints encoded in type system

**Granularity**:
- Operations at the level of hardware instructions
- Not too high-level (loses control)
- Not too low-level (loses optimization opportunity)

**Clean Interface**:
- Clear contract between macro and micro kernels
- Verifiable invariants
- Minimal surface area

### 5.2 Type System

#### Accumulator Type
```mlir
!ppc_mma.acc<element_type, shape>

Examples:
  !ppc_mma.acc<f32, 4x4>   // Single-precision 4×4 accumulator
  !ppc_mma.acc<f32, 4x2>   // Double-precision (f64 input) result
  !ppc_mma.acc<i32, 4x4>   // Integer accumulator

Properties:
  - Represents 512-bit ACC register state
  - Tracks element type and shape
  - Immutable (SSA form)
  - Blocks 4 VSRs when materialized
```

#### Vector Fragment Type
```mlir
!ppc_mma.vec<element_type, length>

Examples:
  !ppc_mma.vec<f32, 4>    // Single VSR with 4 f32 elements
  !ppc_mma.vec<bf16, 8>   // Single VSR with 8 bf16 elements
  !ppc_mma.vec<f64, 2>    // Single VSR with 2 f64 elements

Properties:
  - Represents VSR contents for MMA input
  - Sized and typed for specific MMA operations
  - Layout constraints encoded
```

#### Layout Attributes
```mlir
#ppc_mma.layout<direction, tile_shape>

Examples:
  #ppc_mma.layout<"column", 8x128>  // A-matrix packed panel
  #ppc_mma.layout<"row", 128x16>    // B-matrix packed panel
  
Purpose:
  - Guide packing transformation
  - Optimize memory access patterns
  - Match MMA hardware expectations
```

### 5.3 Operation Set (Phase 1)

#### Accumulator Lifecycle

**Create and Zero**:
```mlir
%acc = ppc_mma.acc.create : !ppc_mma.acc<f32, 4x4>
%zero_acc = ppc_mma.acc.zero %acc : !ppc_mma.acc<f32, 4x4>
```

**Convert** (for mixed precision):
```mlir
// bf16 → f32 accumulator type conversion if needed
%f32_acc = ppc_mma.acc.convert %bf16_acc 
  : !ppc_mma.acc<bf16, 4x4> to !ppc_mma.acc<f32, 4x4>
```

#### Data Movement

**Load from Packed Panels**:
```mlir
// Load A-panel (column-major packed)
%a_vec = ppc_mma.load.panel_a %A_packed[%i, %k] 
  {layout = #ppc_mma.layout<"column", 8x128>}
  : memref<?x?xf32> -> !ppc_mma.vec<f32, 4>

// Load B-panel (row-major packed)
%b_vec = ppc_mma.load.panel_b %B_packed[%k, %j]
  {layout = #ppc_mma.layout<"row", 128x16>}
  : memref<?x?xf32> -> !ppc_mma.vec<f32, 4>
```

**Store to C**:
```mlir
ppc_mma.store %acc, %C[%i, %j] : !ppc_mma.acc<f32, 4x4>, memref<?x?xf32>
```

#### Compute Operations

**Rank-k Update** (outer product with accumulation):
```mlir
// Single-precision rank-1: ACC = A_vec ⊗ B_vec + ACC
%new_acc = ppc_mma.rankk %acc, %a_vec, %b_vec
  : !ppc_mma.acc<f32, 4x4>, !ppc_mma.vec<f32, 4>, !ppc_mma.vec<f32, 4>
  -> !ppc_mma.acc<f32, 4x4>

// Brain-float rank-2 (2 outer products per instruction)
%new_acc = ppc_mma.rankk %acc, %a_vec, %b_vec {rank = 2}
  : !ppc_mma.acc<f32, 4x4>, !ppc_mma.vec<bf16, 8>, !ppc_mma.vec<bf16, 8>
  -> !ppc_mma.acc<f32, 4x4>
```

**Attributes**:
- `rank`: Number of outer products per instruction (1, 2, 4, or 8)
- `alpha`: Scaling factor (for GEMM `α` parameter)
- `accumulate`: Whether to add to existing ACC or overwrite

#### Epilogue Operations (Fused)

**Scaled Store**:
```mlir
// Store with scaling: C = α·ACC + β·C
ppc_mma.store_scaled %acc, %C[%i, %j] {alpha = 1.0, beta = 0.5}
  : !ppc_mma.acc<f32, 4x4>, memref<?x?xf32>
```

**Bias Addition**:
```mlir
// Store with bias: C = ACC + bias
ppc_mma.store_add_bias %acc, %bias, %C[%i, %j]
  : !ppc_mma.acc<f32, 4x4>, memref<?xf32>, memref<?x?xf32>
```

**Activation**:
```mlir
// Store with ReLU: C = relu(ACC)
ppc_mma.store_relu %acc, %C[%i, %j]
  : !ppc_mma.acc<f32, 4x4>, memref<?x?xf32>
```

### 5.4 Operation Semantics

**Verification Requirements**:
- ACC shapes must match MMA hardware (4×4 for f32/bf16/i8/i16, 4×2 for f64)
- Vector lengths must match rank and data type
- Memory accesses must respect alignment (16-byte for VSRs)
- Accumulator dataflow must be valid (no use-before-init)

**Lowering Guarantees**:
- Each `ppc_mma` operation maps to one or more LLVM PPC MMA intrinsics
- No implicit state or side effects
- Predictable performance model

---

## 6. Transformation Passes

### 6.1 ConvertLinalgToPPCMMA

**Purpose**: Lower tiled, packed Linalg operations to PPC MMA dialect.

**Input**: Bufferized Linalg with:
- Tiled matmul operations (BRGEMM structure)
- Packed tensor layouts
- Fused epilogues

**Output**: PPC MMA operations with:
- Explicit ACC management
- Panel load operations
- Rank-k update loops
- Store operations

**Pattern Matching Strategy**:

**Recognize GEMM Pattern**:
```mlir
// Input: Tiled matmul
scf.for %i = %c0 to %M step %c16 {
  scf.for %j = %c0 to %N step %c16 {
    scf.for %k = %c0 to %K step %c128 {
      %a = memref.load %A[%i, %k]
      %b = memref.load %B[%k, %j]
      %c = memref.load %C[%i, %j]
      %prod = arith.mulf %a, %b
      %sum = arith.addf %c, %prod
      memref.store %sum, %C[%i, %j]
    }
  }
}
```

**Generate PPC MMA**:
```mlir
scf.for %i = %c0 to %M step %c8 {
  scf.for %j = %c0 to %N step %c16 {
    // Initialize accumulator
    %acc = ppc_mma.acc.create : !ppc_mma.acc<f32, 4x4>
    %acc_zero = ppc_mma.acc.zero %acc
    
    // K-loop with rank-k updates
    %acc_final = scf.for %k = %c0 to %K step %c128 
                   iter_args(%acc_iter = %acc_zero) {
      %a_vec = ppc_mma.load.panel_a %A_packed[%i, %k]
      %b_vec = ppc_mma.load.panel_b %B_packed[%k, %j]
      %acc_new = ppc_mma.rankk %acc_iter, %a_vec, %b_vec
      scf.yield %acc_new
    }
    
    // Store result
    ppc_mma.store %acc_final, %C[%i, %j]
  }
}
```

**Heuristics**:

**Tile Size Selection** (`mr, nr, kr`):
```cpp
// For f32 (4×4 ACC, rank-1)
constexpr int mr = 8;   // 2 rows of accumulators
constexpr int nr = 16;  // 4 columns of accumulators
constexpr int kr = 128; // K-dimension stride

// For bf16 (4×4 ACC, rank-2)
constexpr int mr = 8;
constexpr int nr = 16;
constexpr int kr = 256; // Double because rank-2

// Rationale:
// - Use all 8 ACCs (2×4 layout)
// - Each ACC computes 4×4 tile
// - Total output: 8×16 per micro-kernel
// - Maximize operand reuse
```

**Cache Blocking** (`mc, nc, kc`):
```cpp
// L1 blocking (kc)
int kc = (L1_size / 2) / elem_size / 4;  // VL=4 for f32
kc = roundDown(kc, kr * 2);               // Constraint: kc % (2*kr) = 0

// L2 blocking (mc)
int mc = (L2_size - L1_size) / elem_size / kc;
mc = roundDown(mc, mr * 2);               // Constraint: mc % (2*mr) = 0

// L3 blocking (nc)
int nc = (L3_size - L2_size) / elem_size / kc;
nc = roundDown(nc, nr * 2);               // Constraint: nc % (2*nr) = 0
```

**Fusion Handling**:
- Recognize fused elementwise operations from `linalg-fuse-elementwise-ops`
- Map to appropriate `ppc_mma.store_*` variant
- Preserve fusion benefit (reduce memory traffic)

**Packing Decision**:
```cpp
bool shouldPackA = (M >= mc * 2);
bool shouldPackB = (N >= nc * 2);
bool amortized = (K >= kc);  // K-loop reuses packed data

if (shouldPackA && amortized) {
  // Generate packing code for A
}
if (shouldPackB && amortized) {
  // Generate packing code for B
}
```

### 6.2 LowerPPCMMAToLLVM

**Purpose**: Lower PPC MMA operations to LLVM IR with PowerPC MMA intrinsics.

**Input**: PPC MMA dialect operations
**Output**: LLVM dialect with intrinsic calls

**Lowering Strategy**:

**ACC Operations** → **LLVM Intrinsics**:
```mlir
// Before: Create and zero ACC
%acc = ppc_mma.acc.create : !ppc_mma.acc<f32, 4x4>
%zero = ppc_mma.acc.zero %acc

// After: LLVM intrinsic
%acc_ptr = llvm.alloca 1 x !llvm.array<512 x i1>
llvm.call @llvm.ppc.mma.xxsetaccz(%acc_ptr) : (!llvm.ptr) -> ()
```

**Rank-k Update** → **MMA Instruction**:
```mlir
// Before: PPC MMA rank-k
%new_acc = ppc_mma.rankk %acc, %a_vec, %b_vec
  : !ppc_mma.acc<f32, 4x4>, !ppc_mma.vec<f32, 4>, !ppc_mma.vec<f32, 4>

// After: LLVM MMA intrinsic
%a_llvm = llvm.bitcast %a_vec : vector<4xf32> to !llvm.vec<4 x f32>
%b_llvm = llvm.bitcast %b_vec : vector<4xf32> to !llvm.vec<4 x f32>
llvm.call @llvm.ppc.mma.xvf32ger(%acc_ptr, %a_llvm, %b_llvm)
  : (!llvm.ptr, !llvm.vec<4 x f32>, !llvm.vec<4 x f32>) -> ()
```

**Store** → **Disassemble + Store**:
```mlir
// Before: PPC MMA store
ppc_mma.store %acc, %C[%i, %j]

// After: Disassemble and store
%result_vec = llvm.call @llvm.ppc.mma.disassemble_acc(%acc_ptr)
  : (!llvm.ptr) -> !llvm.array<4 x vec<4 x f32>>
// ... generate stores for each element
```

**Mapping Table** (stored in pass):

| PPC MMA Op | Data Type | LLVM Intrinsic | Notes |
|------------|-----------|----------------|-------|
| rankk | f32 (rank-1) | llvm.ppc.mma.xvf32ger | Single-precision |
| rankk | bf16 (rank-2) | llvm.ppc.mma.xvbf16ger2 | Brain float |
| rankk | f64 (rank-1) | llvm.ppc.mma.xvf64ger | Double-precision |
| acc.zero | all | llvm.ppc.mma.xxsetaccz | Initialize |
| load.panel_a | all | vector.load | Strided load |
| load.panel_b | all | vector.load | Strided load |
| store | all | llvm.ppc.mma.disassemble_acc | Extract + store |

**Verification**:
- Check intrinsic signatures match LLVM definitions
- Validate alignment requirements
- Ensure ACC lifetime correctness

---

## 7. Optimization Strategy

### 7.1 Packing Transformation

**Why Pack?**
- Improve cache locality
- Enable unit-stride access
- Amortize packing cost over K-loop iterations
- Apply layout transformations once

**When to Pack?**
```
Pack A if:
  1. M ≥ mc × 2  (amortize cost)
  2. K ≥ kc      (reuse packed data)
  3. Layout conversion needed

Pack B if:
  1. N ≥ nc × 2
  2. K ≥ kc
  3. Layout conversion needed
```

**Packing Layout**:
```
A-matrix: Column-major panels
  - Tiles stored by columns
  - Unit-stride access in rank-k loop
  - Match MMA outer-product pattern

B-matrix: Row-major panels
  - Tiles stored by rows
  - Unit-stride access in rank-k loop
  - Match MMA outer-product pattern

C-matrix: Not packed (written once)
```

**Implementation**: Use existing `tensor.pack` with appropriate inner tile sizes and permutations.

### 7.2 Fusion Strategy

**Epilogue Fusion**:
```
Fuse before conversion to ppc_mma:

Pattern 1: Bias addition
  C = (A·B) + bias
  → linalg.matmul + linalg.generic(add)
  → fused to single ppc_mma.store_add_bias

Pattern 2: Activation
  C = relu(A·B)
  → linalg.matmul + linalg.generic(max)
  → fused to single ppc_mma.store_relu

Pattern 3: Combined
  C = relu((A·B) + bias)
  → fully fused epilogue
  → ppc_mma.store_add_bias_relu
```

**Benefits**:
- Reduce memory traffic (no intermediate result)
- Keep data hot in ACCs
- Single store operation

**Limitation**: Fusion happens at Linalg level, before MMA lowering. This keeps lowering pass simple.

### 7.3 Parallelization

**Strategy**: 2D decomposition across output tiles
```mlir
// Parallel over i,j (output tiles)
scf.forall (%i, %j) in (%M_tiles, %N_tiles) {
  // Sequential over k (accumulation)
  scf.for %k = %c0 to %K step %c128 {
    // Rank-k update
    %acc = ppc_mma.rankk ...
  }
}

// Lower to OpenMP
// #pragma omp parallel for collapse(2)
```

**Benefits**:
- No data races (each thread writes different C tile)
- Good load balance for square or near-square problems
- Scales with core count

**Consideration**: OpenMP overhead only worthwhile for large problems (M,N ≥ 512).

---

## 8. Validation and Testing

### 8.1 Correctness Testing

**Unit Tests** (MLIR LIT):
```
test/Dialect/PPCMMA/
  ├── ops.mlir                    # Parse/print/verify
  ├── type-inference.mlir         # Type checking
  ├── canonicalization.mlir       # Simplifications
  └── invalid.mlir                # Error cases

test/Conversion/LinalgToPPCMMA/
  ├── gemm-f32.mlir              # Basic GEMM
  ├── gemm-bf16.mlir             # BFloat16
  ├── brgemm.mlir                # Batched/reduced
  ├── fused-bias.mlir            # Epilogue fusion
  └── tiled-packing.mlir         # With packing

test/Conversion/PPCMMAToLLVM/
  ├── acc-lifecycle.mlir         # Init/zero/extract
  ├── rankk-f32.mlir             # Outer product
  ├── store-variants.mlir        # Different stores
  └── intrinsic-mapping.mlir     # Correct intrinsics
```

**Integration Tests** (MLIR + LLVM):
```
test/Integration/PPCMMA/
  ├── e2e-gemm-f32.mlir          # End-to-end GEMM
  ├── e2e-brgemm-bf16.mlir       # BRGEMM example
  └── compare-reference.mlir     # Numerical accuracy
```

**Numerical Validation**:
- Generate random test matrices
- Compare results against reference (naive implementation)
- Check for IEEE compliance (NaN, Inf handling)
- Validate with known matrix products (identity, etc.)

### 8.2 Performance Benchmarking

**Microbenchmark Suite**:
```
benchmarks/
  ├── harness.cpp                # Main driver
  ├── configs/
  │   ├── small.yaml            # 16-128 (favor MMA over VSX)
  │   ├── medium.yaml           # 128-512 (test caching)
  │   └── large.yaml            # 512-4096 (peak performance)
  ├── kernels/
  │   ├── gemm_naive.cpp        # Triple-loop baseline
  │   ├── gemm_vsx.cpp          # VSX vectorized
  │   ├── gemm_mma_mlir.cpp     # Generated from MLIR
  │   └── gemm_blas.cpp         # OpenBLAS reference
  └── utils/
      ├── matrix_gen.cpp        # Test data
      ├── timer.cpp             # Cycle counter
      └── validator.cpp         # Check correctness
```

**Test Matrix**:
```yaml
Sizes: [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
Shapes:
  - Square: M=N=K
  - Tall: M=4K, N=K, K=K
  - Wide: M=K, N=4K, K=K
Types: [f32, bf16]
Layouts: [RowMajor, ColMajor]
Options:
  - pack: [on, off]
  - fuse: [on, off]
  - parallel: [on, off]
```

**Metrics**:
- **GFLOPS**: (2MNK) / time / 1e9
- **Cycles**: Hardware counter
- **% of Peak**: GFLOPS / theoretical_peak
- **Efficiency**: vs OpenBLAS

**Baselines**:
1. **Naive**: Triple-loop C code
2. **VSX**: Auto-vectorized (`-O3 -mno-mma`)
3. **OpenBLAS**: SGEMM/DGEMM
4. **This**: MLIR → PPC MMA

**Success Criteria**:
- Small (16-128): Better than VSX, competitive with BLAS
- Medium (128-512): Within 20% of BLAS
- Large (1024+): Within 10% of BLAS, target 90-96%
- All: Faster than naive and polyhedral approaches

### 8.3 Profiling and Analysis

**Tools**:
- `perf stat`: Cycle counts, cache misses, instructions
- `perf record`: Hotspot analysis
- LLVM profiling: `-fprofile-instr-generate`
- Manual instrumentation: Trace key operations

**Analysis**:
- Accumulator utilization (should approach 8)
- VSR spills (should be zero or minimal)
- Cache miss rates (L1, L2, L3)
- Instruction mix (MMA vs scalar)
- Vectorization effectiveness

---

## 9. Implementation Plan

### Phase 0: Investigation (Weeks 1-3)

**Goals**:
- Complete intrinsic catalog
- Validate MLIR infrastructure reuse
- Study ARM SME patterns
- Finalize heuristics

**Deliverables**:
- `docs/ppc-mma-intrinsics.csv`
- `docs/sme-analysis.md`
- Test cases: `pack-fuse-fold.mlir`, `tile-brgemm.mlir`
- Updated spec and investigation notes

### Phase 1: MVP Implementation (Weeks 4-9)

**Week 4-5: Dialect Foundation**
- Define `ppc_mma` dialect
- Implement types (ACC, vec)
- Define operations (subset for f32)
- Add verifiers and builders
- Basic lit tests

**Week 6-7: Conversion Passes**
- Implement `ConvertLinalgToPPCMMA` (f32 only)
- Implement `LowerPPCMMAToLLVM` (f32 only)
- Integration tests
- End-to-end compilation working

**Week 8-9: Initial Optimization**
- Add heuristics for tile sizes
- Test with packing
- Basic parallelization
- Microbenchmark setup

**Milestone**: f32 GEMM compiles and runs, within 2× of VSX

### Phase 2: Performance Iteration (Weeks 10-15)

**Week 10-11: bf16 Support**
- Add bf16 operations
- Test rank-2 updates
- Validate mixed precision

**Week 12-13: Fusion**
- Implement epilogue fusion
- Test fused bias/relu
- Measure fusion benefit

**Week 14-15: Tuning**
- Refine heuristics
- Profile and optimize
- Cache behavior analysis
- Approach BLAS performance

**Milestone**: f32 within 10% of OpenBLAS, bf16 working

### Phase 3: Upstreaming (Weeks 16-20)

**Week 16-17: Code Quality**
- Follow MLIR style guide
- Comprehensive documentation
- Code review preparation
- Test coverage >90%

**Week 18-19: RFC and Patches**
- Post RFC to Discourse
- Split into reviewable patches
- Address community feedback
- Iterate on design

**Week 20: Integration**
- Merge approved patches
- Update documentation
- Announce availability

**Milestone**: Upstreamed to LLVM main

---

## 10. Repository Structure

Following LLVM community conventions:
```
mlir/
├── include/mlir/Dialect/PPCMMA/
│   ├── IR/
│   │   ├── PPCMMA.h             # Dialect declaration
│   │   ├── PPCMMA.td            # Dialect definition
│   │   ├── PPCMMAOps.td         # Operation definitions
│   │   ├── PPCMMATypes.td       # Type definitions
│   │   ├── PPCMMAAttrs.td       # Attribute definitions
│   │   └── PPCMMADialect.h      # Public interface
│   └── Transforms/
│       ├── Passes.h             # Pass declarations
│       └── Passes.td            # Pass definitions
│
├── lib/Dialect/PPCMMA/
│   ├── IR/
│   │   ├── PPCMMADialect.cpp    # Dialect registration
│   │   ├── PPCMMAOps.cpp        # Operation implementations
│   │   ├── PPCMMATypes.cpp      # Type implementations
│   │   └── PPCMMAAttrs.cpp      # Attribute implementations
│   └── Transforms/
│       └── PPCMMAPasses.cpp     # Pass registration
│
├── lib/Conversion/
│   ├── LinalgToPPCMMA/
│   │   └── LinalgToPPCMMA.cpp   # Linalg → PPC MMA
│   └── PPCMMAToLLVM/
│       └── PPCMMAToLLVM.cpp     # PPC MMA → LLVM
│
├── test/Dialect/PPCMMA/
│   ├── ops.mlir                 # Operation tests
│   ├── types.mlir               # Type tests
│   └── invalid.mlir             # Negative tests
│
├── test/Conversion/
│   ├── LinalgToPPCMMA/
│   │   └── gemm.mlir            # Conversion tests
│   └── PPCMMAToLLVM/
│       └── lower.mlir           # Lowering tests
│
├── test/Integration/PPCMMA/
│   └── e2e.mlir                 # End-to-end tests
│
└── docs/Dialects/
    └── PPCMMA.md                # Dialect documentation
```

**Registration Touch Points**:
```cpp
// mlir/lib/InitAllDialects.cpp
#include "mlir/Dialect/PPCMMA/IR/PPCMMA.h"
registry.insert<ppc_mma::PPCMMADialect>();

// mlir/lib/InitAllPasses.cpp
#include "mlir/Dialect/PPCMMA/Transforms/Passes.h"
mlir::ppc_mma::registerPPCMMAPasses();

// mlir/include/mlir/Conversion/Passes.td
include "mlir/Dialect/PPCMMA/Transforms/Passes.td"
```

---

## 11. Success Criteria

### Phase 1 (MVP)

**Functional**:
- [ ] f32 and bf16 GEMM compile end-to-end
- [ ] Correct numerical results (validated against reference)
- [ ] Generate valid PowerPC MMA instructions
- [ ] All lit tests pass
- [ ] Integration tests pass

**Performance**:
- [ ] 2× faster than VSX baseline
- [ ] Clear speedup visible in microbenchmarks
- [ ] No catastrophic regressions

**Quality**:
- [ ] Code follows MLIR style
- [ ] >80% test coverage
- [ ] Documentation complete

### Phase 2 (Performance)

**Performance**:
- [ ] Small matrices: Competitive with BLAS
- [ ] Medium matrices: Within 20% of BLAS
- [ ] Large matrices: Within 10% of BLAS
- [ ] Better than Eigen for most cases
- [ ] >90% peak achievable for best cases

**Functionality**:
- [ ] Fused epilogues working
- [ ] Packing showing benefit
- [ ] Parallelization scaling

### Phase 3 (Upstream)

**Community**:
- [ ] RFC accepted
- [ ] Patches reviewed and merged
- [ ] Documentation published
- [ ] Examples available

---

## 12. Risks and Mitigation

### Technical Risks

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Intrinsic ABI complications | Medium | Medium | Isolate in PPCMMAToLLVM; table-driven |
| Heuristic mismatch | Medium | Medium | Inspectable rules; trace flags; tuning |
| Fusion side effects | Low | Low | Fuse pre-MMA; keep lowering simple |
| Hardware access | High | Medium | Simulator; IBM partnership; cloud |
| Performance target miss | Medium | Medium | Iterative; profiling; research guidance |
| Upstream rejection | Low | Low | Follow guidelines; early engagement |

### Resource Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Limited POWER10 access | High | Simulator; partnership; cloud |
| Learning curve | Medium | Study existing code; community |
| Time constraints | Medium | Phased approach; MVP first |

---

## 13. Future Work

### Phase 2 Extensions

**Additional Data Types**:
- f16, f64 (full precision range)
- i8, i16 (quantized inference)
- Mixed precision combinations

**Additional Operations**:
- SYRK (symmetric rank-k)
- SYR2K (symmetric rank-2k)
- TRMM (triangular matrix multiply)

**Optimizations**:
- Advanced cost model
- Auto-tuning infrastructure
- Profile-guided optimization

### Long-term Vision

**Compiler Infrastructure**:
- Generalized matrix engine framework
- Shared optimization components
- Cross-architecture abstractions

**Research**:
- Automatic pattern discovery
- ML-based optimization
- Novel transformation strategies

**Ecosystem**:
- Integration with ML frameworks (PyTorch, TensorFlow)
- Scientific computing libraries
- Industry adoption

---

## 14. Open Questions

### To Be Resolved in Phase 0

- [ ] Exact LLVM intrinsic availability in upstream Clang?
- [ ] Any differences between OpenXL and upstream LLVM?
- [ ] Full bf16 support details in POWER10?
- [ ] Optimal (mr,nr,kr) for bf16 vs f32?
- [ ] ABI considerations for Linux vs AIX?
- [ ] Polygeist limitations for MMA builtins?

### Design Decisions

- [ ] Operation granularity: Higher vs lower level?
- [ ] Fusion strategy: At Linalg or PPC MMA level?
- [ ] Type system: How much hardware detail to expose?
- [ ] Cost model: Simple heuristics vs sophisticated?

---

## 15. References

### Research Papers
1. Kuzma et al., "Fast Matrix Multiplication via Compiler-only Layered Data Reorganization and Intrinsic Lowering", Software: Practice and Experience, 2023
2. Goto & Van de Geijn, "Anatomy of High-Performance Matrix Multiplication", ACM TOMS, 2008
3. Bondhugula, "High Performance Code Generation in MLIR", 2020

### Technical Specifications
1. Power ISA Version 3.1, IBM Corporation, 2020
2. POWER10 Processor User's Manual, IBM
3. MLIR Language Reference, https://mlir.llvm.org/
4. LLVM PowerPC Backend Documentation

### Code Repositories
1. LLVM Project: https://github.com/llvm/llvm-project
2. Polygeist: https://github.com/llvm/Polygeist
3. OpenBLAS: https://github.com/xianyi/OpenBLAS

---

## Appendix A: Command Reference

### Build and Test
```bash
# Build MLIR with PPCMMA dialect
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="PowerPC" \
  -DCMAKE_BUILD_TYPE=Release

ninja check-mlir-dialect-ppcmma

# Run specific test
llvm-lit -v test/Dialect/PPCMMA/ops.mlir
```

### Compilation Pipeline
```bash
# Full pipeline
mlir-opt input.mlir -ppc-mma-full-pipeline -o output.mlir

# Step-by-step
mlir-opt input.mlir -linalg-tile="tile-sizes=16,16,128" -o tiled.mlir
mlir-opt tiled.mlir -convert-linalg-to-ppc-mma -o mma.mlir
mlir-opt mma.mlir -lower-ppc-mma-to-llvm -o llvm.mlir

# To LLVM IR
mlir-translate --mlir-to-llvmir llvm.mlir -o output.ll

# To assembly
llc -mcpu=power10 output.ll -o output.s
```

### Benchmarking
```bash
# Build benchmark
make benchmark

# Run sweep
./benchmark --sizes=16,32,64,128,256,512,1024 \
            --types=f32,bf16 \
            --baseline=vsx,blas

# Profile
perf stat -e cycles,instructions,cache-misses ./benchmark
perf record -g ./benchmark
perf report
```

---

**Document Status**: Living Document  
**Last Updated**: 2025-01-XX  
**Next Review**: After Phase 0 completion  

---

**End of Specification Document**