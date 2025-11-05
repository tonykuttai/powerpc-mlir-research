# PowerPC MMA in MLIR - Investigation Notes

**Project**: PowerPC MMA Dialect for MLIR  
**Owner**: Tony Varghese  
**Last Updated**: 2025-01-XX  
**Status**: Phase 0 - Investigation in Progress

---

## Investigation Objectives

1. Catalog PowerPC MMA intrinsics and builtins in LLVM/Clang
2. Understand existing MLIR infrastructure we can reuse (packing, tiling, fusion)
3. Study similar implementations (ARM SME) for design patterns
4. Define concrete pass pipeline with existing MLIR components
5. Establish heuristics for tiling parameters and packing thresholds
6. Create microbenchmark harness for validation

---

## Phase 1: PowerPC MMA Hardware Analysis

### 1.1 Architecture Overview

**Status**: ✅ COMPLETED (from research paper analysis)
#### Hardware Characteristics
- **Processor**: IBM POWER10 (PowerISA 3.1C)
- **Extension**: Matrix Multiply Assist (MMA) - extension of VSX
- **Register Architecture**:
  - 512-bit Accumulator (ACC) registers
  - Each ACC maps to 4×128-bit Vector-Scalar Registers (VSRs)
  - 8 ACCs maximum (blocks 32 of 64 VSRs when in use)
  - 32 VSRs remain available for vector operations
#### Computational Model
- **Primary Operation**: Rank-k updates (outer products)
- **Key Difference**: Uses outer products vs inner products (Intel AMX)
- **Performance Characteristics**:
  - 2 multiply-accumulate outer-product instructions per cycle
  - 4-cycle issue-to-issue latency for same accumulator
  - Expensive accumulator spills to memory
  - High computational density (2D operations)
#### Supported Data Types
| Data Type | Input Size | Computation | Output Shape | Rank | Priority |
|-----------|-----------|-------------|--------------|------|----------|
| f32 | 32-bit | 4×1 · 1×4 | 4×4 f32 | 1 | **Phase 1** |
| bf16 | 16-bit | 4×2 · 2×4 | 4×4 f32 | 2 | **Phase 1** |
| f16 | 16-bit | 4×2 · 2×4 | 4×4 f32 | 2 | Phase 2 |
| f64 | 64-bit | 4×1 · 1×2 | 4×2 f64 | 1 | Phase 2 |
| i8 | 8-bit | 4×4 · 4×4 | 4×4 i32 | 4 | Phase 2 |
| i16 | 16-bit | 4×2 · 2×4 | 4×4 i32 | 2 | Phase 2 |
| i4 | 4-bit | 4×8 · 8×4 | 4×4 i32 | 8 | Future |

#### Cache Hierarchy (Typical POWER10)
- L1 Data Cache: 48 KiB
- L2 Cache: 1024 KiB  
- L3 Cache: 4 MiB (per core in POWER10)
- Cache line: 128 bytes (2× larger than x86)

**Performance Results** (from research paper):
- **vs Vector Extensions (VSX)**: 2.6× faster
- **vs BLAS (OpenBLAS)**: Up to 96% peak performance
- **vs Eigen**: 83% faster for large matrices
- **vs Polyhedral (PLuTo)**: 6-22× faster
---
## Phase 2: LLVM/Clang Intrinsics Inventory

### 2.1 Intrinsics Discovery

**Status**: 🔄 IN PROGRESS

**Objective**: Catalog all PowerPC MMA intrinsics and builtins with complete signatures, constraints, and usage patterns.
#### Search Locations

**LLVM Intrinsics**:
```bash
# In llvm-project repository
cd llvm-project

# Find MMA intrinsic definitions
rg -n --hidden -S "mma|matrix|acc" llvm/include/llvm/IR/IntrinsicsPowerPC.td

# Example patterns to search:
# - int_ppc_mma_*
# - def int_ppc_mma_xvf32ger
# - def int_ppc_mma_xvf64ger
# - def int_ppc_mma_xxsetaccz
```

**Clang Builtins**:
```bash
# Find builtin definitions
rg -n --hidden -S "mma|__builtin" clang/include/clang/Basic/BuiltinsPPC.def
rg -n --hidden -S "mma" clang/lib/Headers/

# Look for patterns like:
# - __builtin_mma_*
# - __builtin_mma_xvf32ger
# - __builtin_mma_assemble_acc
```

**PowerPC Backend**:
```bash
# Backend implementation
rg -n --hidden -S "mma|ACC" llvm/lib/Target/PowerPC/

# Key files to check:
# - PPCInstrMMA.td (instruction definitions)
# - PPCISelLowering.cpp (intrinsic lowering)
# - PPCRegisterInfo.td (ACC register definitions)
```

#### Test Program for IR Probing

**test_mma.c**:
```c
#include <altivec.h>

void test_f32_mma(float *a, float *b, float *c) {
    __vector_quad acc;
    vector float va, vb;
    
    // Zero accumulator
    __builtin_mma_xxsetaccz(&acc);
    
    // Load vectors
    va = vec_xl(0, a);
    vb = vec_xl(0, b);
    
    // Outer product
    __builtin_mma_xvf32ger(&acc, va, vb);
    
    // Disassemble and store
    __builtin_mma_disassemble_acc((void *)c, &acc);
}
```

**Compile and inspect**:
```bash
# Generate LLVM IR
clang -O3 -mcpu=power10 -target powerpc64le-unknown-linux-gnu \
      -S -emit-llvm test_mma.c -o test_mma.ll

# Check intrinsic signatures
grep "@llvm.ppc.mma" test_mma.ll

# Generate assembly
clang -O3 -mcpu=power10 -target powerpc64le-unknown-linux-gnu \
      -S test_mma.c -o test_mma.s
```

#### Artifact to Produce

**docs/ppc-mma-intrinsics.csv**:

| Intrinsic Name | Clang Builtin | Input Types | Output Type | K-Granularity | ACC I/O | Alignment | Notes |
|----------------|---------------|-------------|-------------|---------------|---------|-----------|-------|
| llvm.ppc.mma.xvf32ger | __builtin_mma_xvf32ger | v4f32, v4f32 | acc (4×4×f32) | k=1 | inout | 16-byte | Single precision rank-1 |
| llvm.ppc.mma.xvf64ger | __builtin_mma_xvf64ger | v2f64, v2f64 | acc (4×2×f64) | k=1 | inout | 16-byte | Double precision |
| llvm.ppc.mma.xvbf16ger2 | __builtin_mma_xvbf16ger2 | v8bf16, v8bf16 | acc (4×4×f32) | k=2 | inout | 16-byte | BFloat16 rank-2 |
| llvm.ppc.mma.xxsetaccz | __builtin_mma_xxsetaccz | none | acc (zeroed) | - | out | - | Zero accumulator |
| llvm.ppc.mma.assemble_acc | __builtin_mma_assemble_acc | 4×v4f32 | acc | - | out | - | Assemble from VSRs |
| llvm.ppc.mma.disassemble_acc | __builtin_mma_disassemble_acc | acc | 4×v4f32 | - | in | - | Extract to VSRs |

**TODO**:
- [ ] Complete full intrinsic catalog
- [ ] Document precision and rounding modes
- [ ] Identify performance-critical intrinsics
- [ ] Test each intrinsic with probe programs
- [ ] Document ABI requirements (AIX vs Linux)
- [ ] Note any differences between OpenXL and upstream Clang

### 2.2 Findings from Code Search

**Status**: ⏳ PENDING

**Results** (to be filled):
```
[Intrinsic signatures found]
[Register constraints documented]
[Alignment requirements noted]
[Performance characteristics measured]
```

---

## Phase 3: MLIR Infrastructure Survey

### 3.1 Existing MLIR Components to Reuse

**Status**: 🔄 IN PROGRESS

**Objective**: Identify and validate existing MLIR infrastructure that we can leverage rather than building from scratch.

#### 3.1.1 Tensor Packing Infrastructure

**Key Components**:
- `tensor.pack` / `tensor.unpack` operations
- Pack propagation passes
- Fold redundant pack/unpack sequences

**Investigation**:
```bash
cd llvm-project/mlir

# Find tensor packing infrastructure
find . -name "*Pack*" -o -name "*Unpack*" | grep -i tensor
grep -r "tensor.pack" include/mlir/Dialect/Tensor/

# Look for transformation passes
grep -r "PackPipeline\|FoldPackUnpack" lib/Dialect/Tensor/Transforms/
```

**Test Case to Create**: `mlir/testdata/pack-fuse-fold.mlir`
```mlir
// Example: pack → fuse → fold pattern
func.func @pack_matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) 
    -> tensor<128x128xf32> {
  // Pack A and B
  %A_packed = tensor.pack %A : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  %B_packed = tensor.pack %B : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  
  // Matmul on packed tensors
  %C_packed = linalg.matmul ins(%A_packed, %B_packed: ...) -> ...
  
  // Unpack result
  %C = tensor.unpack %C_packed : tensor<16x16x8x8xf32> -> tensor<128x128xf32>
  return %C : tensor<128x128xf32>
}

// Goal: Demonstrate that packing can be propagated and folded
```

**TODO**:
- [ ] Create pack/fuse/fold test case
- [ ] Document pass order dependencies
- [ ] Understand pack layout attributes
- [ ] Test with matmul operations

#### 3.1.2 Linalg Tiling Infrastructure

**Key Components**:
- `linalg.tile` transformation
- `linalg-fuse-elementwise-ops` pass
- Tiling attributes and strategies

**Investigation**:
```bash
# Find tiling infrastructure
grep -r "TileOp\|linalg-tile" mlir/lib/Dialect/Linalg/Transforms/

# Look for fusion passes
grep -r "FuseElementwise" mlir/lib/Dialect/Linalg/Transforms/
```

**Test Case**: `mlir/testdata/tile-brgemm.mlir`
```mlir
// Example: Tiled matmul creating BRGEMM-like structure
func.func @tiled_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>) 
    -> tensor<?x?xf32> {
  %C = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C_init: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %C : tensor<?x?xf32>
}

// After tiling (tile-sizes=16,16,128):
// Should create nested loops with 16×16 output tiles and k-stride of 128
```

**TODO**:
- [ ] Create tiling example with BRGEMM structure
- [ ] Test fusion of elementwise epilogues
- [ ] Document tile size selection
- [ ] Understand interaction with packing

#### 3.1.3 Bufferization

**Key Component**: One-shot bufferization

**Investigation**:
```bash
# Find bufferization infrastructure
grep -r "one-shot-bufferize" mlir/lib/Dialect/Bufferization/
```

**TODO**:
- [ ] Test bufferization of packed tensors
- [ ] Understand memory layout after bufferization
- [ ] Document copy elimination strategies

#### 3.1.4 Parallel Distribution

**Key Components**:
- `scf.forall` operations
- OpenMP lowering passes

**TODO**:
- [ ] Test 2D tile distribution
- [ ] Understand OpenMP mapping
- [ ] Document parallelization strategy

### 3.2 Integration Points

**Pass Pipeline Order** (draft):
```bash
mlir-opt input.mlir \
  -canonicalize -cse \
  -linalg-tile="tile-sizes=16,16,128" \
  -linalg-fuse-elementwise-ops \
  -tensor-pack-pipelines \
  -fold-tensor-pack-unpack \
  -one-shot-bufferize \
  -convert-linalg-to-ppc-mma \
  -lower-ppc-mma-to-llvm \
  -convert-scf-to-openmp \
  -llvm-legalize-for-opaque-ptr \
  -finalize-memref-to-llvm
```

**TODO**:
- [ ] Test each pass independently
- [ ] Identify critical pass ordering constraints
- [ ] Document IR at each stage
- [ ] Measure compilation time and code size

---

## Phase 4: Analog Study - ARM SME

### 4.1 ARM SME Dialect Analysis

**Status**: 🔄 IN PROGRESS

**Objective**: Study ARM SME dialect for design patterns we can reuse.

**Search Commands**:
```bash
cd llvm-project/mlir

# Find ARM SME dialect
find . -path "*ArmSME*" -type f
grep -r "arm_sme" include/mlir/Dialect/

# Key files to study:
# - ArmSME.td (dialect definition)
# - ArmSMEOps.td (operation definitions)
# - ArmSMEToLLVM (lowering pass)
```

**Key Patterns to Extract**:

1. **Type System**:
   - How are tile/accumulator types represented?
   - What attributes encode hardware constraints?

2. **Operation Design**:
   - How are matrix operations structured?
   - What granularity of operations?
   - How is state management handled?

3. **Lowering Strategy**:
   - Pattern-based conversion?
   - How are hardware constraints verified?

4. **Integration**:
   - How does it integrate with Vector dialect?
   - How does it interact with Linalg?

**Artifact**: `docs/sme-analysis.md`

**Comparison Table**:

| Aspect | ARM SME | PowerPC MMA | Design Decision |
|--------|---------|-------------|-----------------|
| Register Model | Vector + ZA state | VSR + ACC | Explicit ACC types |
| Operation Size | Scalable tiles | Fixed 4×4/4×2 | Fixed size initially |
| Instruction Style | 2×2 ops | Outer products | Rank-k update ops |
| Integration | Tightly coupled | Tightly coupled | Same approach |
| Type System | Tile types | ACC types | ACC as first-class |

**TODO**:
- [ ] Complete ARM SME code review
- [ ] Extract reusable patterns
- [ ] Document key design decisions
- [ ] Create conceptual mapping to PPC MMA

---

## Phase 5: Heuristics and Cost Model

### 5.1 Tiling Parameters

**Status**: 🔄 IN PROGRESS

**Objective**: Define initial heuristics for `(mr, nr, kr)` and cache blocking parameters.

**POWER10 Hardware Facts** (to verify):
- L1 Data: 48 KiB per core
- L2: 1024 KiB per core
- L3: 4 MiB per core (not shared on POWER10)
- Cache line: 128 bytes
- VSRs: 64 total (32 available when 8 ACCs used)
- ACCs: 8 maximum

**Heuristic Framework**:

#### Register-Level Tiling (mr, nr, kr)

**For f32** (4×4 accumulator, rank-1):
```
Initial values:
  mr = 8  (2 rows of accumulators)
  nr = 16 (4 columns of accumulators, 2×4 layout)
  kr = 128 (maximize reuse, fit in registers)

Rationale:
  - 8 accumulators used (full utilization)
  - Each ACC computes 4×4 tile
  - Layout: 2 vertical × 4 horizontal = 8×16 total
  - kr=128 means 128 iterations of rank-1 updates
  - Operand reuse: each A operand used 4 times (nr/4)
  - Operand reuse: each B operand used 2 times (mr/4)
```

**For bf16** (4×4 accumulator, rank-2):
```
Initial values:
  mr = 8
  nr = 16  
  kr = 256 (2× because rank-2)

Rationale:
  - Same accumulator layout as f32
  - kr doubles because each op processes 2 k elements
```

#### Cache-Level Blocking (mc, nc, kc)

**Formula** (from research paper):
```
L1 level (kc):
  kc ≤ (L1_size / 2) / elem_size / VL
  For f32: kc ≤ (48KB / 2) / 4 / 4 = ~1536
  Round to multiple of kr: kc = 1536 (12 × 128)

L2 level (mc):
  mc ≤ (L2_size - L1_size) / elem_size / kc
  For f32: mc ≤ (1024KB - 48KB) / 4 / 1536 = ~163
  Round to multiple of mr: mc = 160 (20 × 8)

L3 level (nc):
  nc ≤ (L3_size - L2_size) / elem_size / kc  
  For f32: nc ≤ (4096KB - 1024KB) / 4 / 1536 = ~500
  Round to multiple of nr: nc = 496 (31 × 16)

Constraints:
  kc mod (2×kr) = 0
  mc mod (2×mr) = 0
  nc mod (2×nr) = 0
```

**TODO**:
- [ ] Validate cache size assumptions on actual POWER10
- [ ] Implement heuristic selection in pass
- [ ] Add command-line overrides for tuning
- [ ] Test sensitivity to tile size variations

### 5.2 Packing Thresholds

**Packing Decision**:
- Pack when dimension size > threshold
- Pack when tiles are reused multiple times
- Pack when layout conversion is needed

**Initial Threshold** (heuristic):
```
Pack A if: M ≥ mc × 2  (amortize packing cost)
Pack B if: N ≥ nc × 2
Pack both if: K ≥ kc    (K-loop reuses packed data)
```

**Layout Decisions**:
```
A: Column-major packing (for outer product)
B: Row-major packing (for outer product)  
C: Row-major storage (conventional)
```

**TODO**:
- [ ] Implement packing threshold logic
- [ ] Measure packing overhead
- [ ] Add toggle for packing enable/disable
- [ ] Profile packing benefit vs cost

### 5.3 Fusion Strategy

**Epilogue Fusion** (before store to C):
```
Supported patterns:
  C = α·(A·B) + β·C  (GEMM scaling)
  C = (A·B) + bias    (bias addition)
  C = relu(A·B)       (activation)
  C = relu((A·B) + bias) (fused bias+relu)
```

**Implementation Approach**:
- Fuse at Linalg level (before conversion to ppc_mma)
- Use `linalg-fuse-elementwise-ops` pass
- Convert fused pattern to single ppc_mma operation
- Generate combined load-compute-store sequence

**TODO**:
- [ ] Define fusion patterns
- [ ] Test with various epilogues
- [ ] Measure fusion benefit

---

## Phase 6: Microbenchmark Infrastructure

### 6.1 Benchmark Harness

**Status**: ⏳ PENDING

**Objective**: Create standalone benchmark harness for performance validation.

**Structure**:
```
benchmarks/
  ├── harness.cpp          # Main driver
  ├── kernels/
  │   ├── gemm_naive.cpp   # Baseline (no MMA)
  │   ├── gemm_vsx.cpp     # VSX vectorized
  │   ├── gemm_mma.cpp     # Generated from MLIR
  │   └── gemm_blas.cpp    # Library baseline
  ├── utils/
  │   ├── matrix_gen.cpp   # Test data generation
  │   └── timer.cpp        # Cycle counter
  └── configs/
      ├── small.yaml       # 16-128 sizes
      ├── medium.yaml      # 128-512 sizes
      └── large.yaml       # 512-4096 sizes
```

**Harness Features**:
- Sweep M, N, K parameters
- Multiple data types (f32, bf16)
- Toggle options:
  - Pack on/off
  - Fused epilogue on/off
  - Parallel on/off
- Output: GFLOPS, cycles, % of peak

**Test Matrix**:
```
Sizes: {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
Shapes: Square (M=N=K), Tall (M>>N), Wide (N>>M)
Types: f32, bf16
Layouts: Row-major, Column-major
```

**TODO**:
- [ ] Implement benchmark harness
- [ ] Add timing infrastructure
- [ ] Create test data generators
- [ ] Add result validation
- [ ] Set up automated runs

### 6.2 Baseline Comparisons

**Baselines to Include**:

1. **VSX Vectorized** (no MMA):
```bash
   clang -O3 -mcpu=power10 -mno-mma gemm.c -o gemm_vsx
```

2. **OpenBLAS** (if available):
```bash
   # Link against system BLAS
```

3. **BLIS** (if available)

4. **This Project** (MLIR → MMA):
```bash
   mlir-opt --ppc-mma-pipeline input.mlir | \
     mlir-translate --mlir-to-llvmir | \
     llc -mcpu=power10 -o output.s
```

**Success Metrics**:
- Clear speedup over VSX baseline (target: 2×+)
- Within 10% of BLAS for large sizes
- Better than VSX for all sizes
- Packing overhead < 5% for large problems

**TODO**:
- [ ] Implement all baseline versions
- [ ] Automate comparison runs
- [ ] Generate performance graphs
- [ ] Document methodology

---

## Phase 7: Open Questions and Risks

### 7.1 Technical Questions

**Status**: ⏳ PENDING

**Questions**:

1. **Intrinsic Availability**:
   - Q: Are all MMA intrinsics available in upstream Clang?
   - Q: Any differences between OpenXL and upstream?
   - Action: Test with probe programs

2. **Data Type Support**:
   - Q: Full bf16 support in POWER10 MMA?
   - Q: Mixed precision accumulation modes?
   - Q: Rounding and saturation controls?
   - Action: Check ISA spec and test

3. **ABI Considerations**:
   - Q: ACC register calling conventions?
   - Q: Linux vs AIX differences?
   - Q: OpenMP interaction with vector calls?
   - Action: Study ABI docs

4. **Optimal Tiling**:
   - Q: Best (mr,nr,kr) for bf16 vs f32?
   - Q: Sensitivity to cache size variations?
   - Q: Impact of hyperthreading on tile sizes?
   - Action: Empirical testing

5. **Integration**:
   - Q: Polygeist support for MMA builtins?
   - Q: Interaction with existing PPC optimizations?
   - Action: Test end-to-end

### 7.2 Risks

**Risk 1**: Intrinsic/ABI complications
- **Severity**: Medium
- **Mitigation**: Isolate in PPCMMAToLLVM; table-driven approach

**Risk 2**: Heuristic mismatch for tile sizes
- **Severity**: Medium
- **Mitigation**: Small, inspectable rules; trace flags; empirical tuning

**Risk 3**: Fusion side effects
- **Severity**: Low
- **Mitigation**: Fuse before ppc_mma lowering; keep lowering simple

**Risk 4**: Hardware access limitations
- **Severity**: High (if no access)
- **Mitigation**: Use simulator; seek IBM partnership; cloud instances

**Risk 5**: Performance below targets
- **Severity**: Medium
- **Mitigation**: Iterative optimization; profiling; learn from paper

---

## Phase 8: Concrete Next Steps (Checklist)

### Week 1-2: Investigation Completion

- [ ] **Intrinsic Catalog**
  - [ ] Extract full intrinsic list from IntrinsicsPowerPC.td
  - [ ] Document all Clang builtins
  - [ ] Create ppc-mma-intrinsics.csv
  - [ ] Test each with probe programs
  - [ ] Note ABI requirements

- [ ] **MLIR Infrastructure**
  - [ ] Create pack-fuse-fold.mlir test
  - [ ] Create tile-brgemm.mlir test
  - [ ] Test pass pipeline ordering
  - [ ] Document IR at each stage

- [ ] **SME Study**
  - [ ] Review ARM SME dialect code
  - [ ] Extract design patterns
  - [ ] Create sme-analysis.md
  - [ ] Map to PPC MMA design

### Week 3-4: Initial Implementation

- [ ] **Dialect Foundation**
  - [ ] Draft PPCMMA.td (dialect definition)
  - [ ] Draft PPCMMAOps.td (operation definitions)
  - [ ] Define ACC types
  - [ ] Define tile types
  - [ ] Add basic verifiers

- [ ] **Stub Passes**
  - [ ] ConvertLinalgToPPCMMA.cpp (one f32 pattern)
  - [ ] PPCMMAToLLVM.cpp (one op variant)
  - [ ] Basic lit tests
  - [ ] Parse/print tests

### Week 5-6: Microbenchmark Setup

- [ ] **Benchmark Infrastructure**
  - [ ] Create benchmark harness
  - [ ] Implement baseline kernels
  - [ ] Add timing utilities
  - [ ] Set up test matrix

- [ ] **Initial Measurements**
  - [ ] Run f32 baseline comparisons
  - [ ] Measure VSX vs MMA potential
  - [ ] Validate against paper results
  - [ ] Document findings

---

## Appendix: Useful Commands

### Intrinsics Discovery
```bash
# Search for MMA intrinsics in LLVM
cd llvm-project
rg -n --hidden -S "mma|matrix|acc" llvm/include/llvm/IR/IntrinsicsPowerPC.td

# Search for Clang builtins
rg -n --hidden -S "mma|__builtin" clang/include/clang/Basic/BuiltinsPPC.def clang/lib/Headers

# Search backend implementation
rg -n --hidden -S "mma|ACC" llvm/lib/Target/PowerPC/
```

### IR Generation and Inspection
```bash
# Generate LLVM IR from C
clang -O3 -mcpu=power10 -target powerpc64le-unknown-linux-gnu \
      -S -emit-llvm test.c -o test.ll

# View intrinsic calls
grep "@llvm.ppc.mma" test.ll

# Generate assembly
clang -O3 -mcpu=power10 -target powerpc64le-unknown-linux-gnu \
      -S test.c -o test.s

# Disassemble object
llvm-objdump -d test.o
```

### MLIR Pass Pipeline
```bash
# Run proposed pipeline (adjust pass names as implemented)
mlir-opt input.mlir \
  -canonicalize -cse \
  -linalg-tile="tile-sizes=16,16,128" \
  -linalg-fuse-elementwise-ops \
  -tensor-pack-pipelines \
  -fold-tensor-pack-unpack \
  -one-shot-bufferize \
  -convert-linalg-to-ppc-mma \
  -lower-ppc-mma-to-llvm \
  -convert-scf-to-openmp \
  -o output.mlir

# Save IR at each stage
mlir-opt input.mlir -canonicalize -o 1_canon.mlir
mlir-opt 1_canon.mlir -linalg-tile="tile-sizes=16,16,128" -o 2_tiled.mlir
# ... etc
```

### Performance Measurement
```bash
# Compile with MMA
clang -O3 -mcpu=power10 -fuse-ld=lld gemm_mma.c -o gemm_mma

# Compile baseline (no MMA)
clang -O3 -mcpu=power10 -mno-mma gemm_baseline.c -o gemm_baseline

# Run benchmark
./benchmark --size=1024 --dtype=f32 --iterations=100

# Profile with perf
perf stat -e cycles,instructions,cache-misses ./gemm_mma
```

### Testing
```bash
# Run MLIR tests
cd llvm-project/build
ninja check-mlir-dialect-ppcmma

# Run specific test
llvm-lit -v ../mlir/test/Dialect/PPCMMA/ops.mlir

# Run integration tests
ninja check-mlir-integration-ppcmma
```

---

## Investigation Log

### 2025-01-XX: Initial Setup
- Investigation framework created
- Hardware analysis completed from paper
- Search strategies defined
- Next: Begin intrinsic catalog

### [Future Dates]
[Log updates as investigation progresses]

---

## References

### Academic Papers
1. Kuzma et al., "Fast Matrix Multiplication via Compiler-only...", 2023
2. Goto & Van de Geijn, "Anatomy of High-Performance Matrix Multiplication", 2008

### Technical Specifications
1. Power ISA Version 3.1, IBM, 2020
2. POWER10 Processor User's Manual
3. MLIR Documentation, https://mlir.llvm.org/

### Code Repositories
1. LLVM Project: https://github.com/llvm/llvm-project
2. Polygeist: https://github.com/llvm/Polygeist

---

**Document Status**: Living Document - Updated as investigation progresses  
**Next Review**: After intrinsic catalog completion  

---

**End of Investigation Notes**