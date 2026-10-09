# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103200  
**Allocated GPU Node:** Tesla T4  
**CUDA Compute Capability:** 7.5  
**Official Verification Token:** 3BDFBA70846A2E8D6CF8  

---

## Task 1 Benchmarks: Warp Divergence Impact

| Kernel Variant | Execution Time (ms) | Description |
| :--- | :--- | :--- |
| **Kernel A** | **18.769 ms** | Uniform Path (No Divergence) |
| **Kernel B** | **93.463 ms** | Interleaved Divergence (`idx % 2`) |
| **Kernel C** | **46.810 ms** | Warp-Aligned Branching (`warp_id % 2`) |

### Key Architectural Takeaways
1. **Intra-warp serialization (Kernel B):** Interleaved branching (`idx % 2`) forces adjacent threads within the same 32-thread warp to take different execution paths. The hardware SIMT execution units must mask off inactive threads and execute the paths sequentially, leading to an almost 5x latency penalty compared to the uniform baseline.
2. **Warp-aligned efficiency (Kernel C):** Aligning conditional branches to warp boundaries (`warp_id % 2`) guarantees that all 32 threads in a given warp follow the exact same execution path simultaneously. This completely eliminates intra-warp serialization, cutting the execution time in half compared to Kernel B.