---
layout: post
title: "nvcc 13.2 quietly miscompiles IQ quants on sm_120, and it lasted exactly two releases"
categories: [cuda, llama.cpp, compilers]
---

Short version: llama.cpp's CUDA kernels for the `IQ1_S`, `IQ2_S` and `IQ3_S`
quantization formats produce wrong numbers on consumer Blackwell when built with
nvcc 13.2.51 or 13.2.78. No crash, no warning - just different values. The
symptom is gone in nvcc 13.2.86 and stays gone in 13.4.92, so the affected window
is exactly CUDA 13.2.0 and 13.2.1.

| nvcc | CUDA release | llama.cpp master | master + [PR #28784][pr] |
|---|---|---|---|
| 13.2.51 | 13.2.0 | 44 / 22 fail | 0 / 0 |
| 13.2.78 | 13.2.1 | 44 / 22 fail | - |
| 13.2.86 | 13.2.2 | 0 / 0 | - |
| 13.4.92 | 13.4.2 | 0 / 0 | - |

If you are running a `IQ*_S` quant on an RTX 50 card and your output degraded
without an obvious cause, check `nvcc --version` on whoever built your binary.

## Why this card collects its own bugs

Consumer Blackwell (`sm_120`, the RTX 50 series) is not a smaller B200. There is
no TMEM, no `tcgen05`, no WGMMA, and shared memory per block tops out around
99-101 KB. The people who maintain the kernels I depend on develop on H100 and
B200. They are not being careless - they simply do not have this card on the
desk, while several million users do.

That gap is the whole reason this post exists. I have the card.

## The bug

[PR #28784][pr] by ForeverYoung1208 reports it: reading a packed `int` byte by
byte through a `uint8_t` pointer compiles incorrectly. The mask gets dropped and
the index into the dequantization grid table runs out of range. The pattern looks
like this, in `mmq-load-tiles.cuh` and `vecdotq.cuh`:

```cuda
const int       qs_packed = get_int_b2(bxi->qs, kqsx);
const uint8_t * qs        = (const uint8_t *) &qs_packed;
// ...
const int grid = iq1s_grid_gpu[qs[l] | (((qh >> (3*l)) & 0x07) << 8)];
```

The fix replaces the byte read with `__byte_perm`, which extracts the byte as a
single instruction and leaves the compiler no room to be clever:

```cuda
const int grid = iq1s_grid_gpu[__byte_perm(qs_packed, 0, 0x4440 | l)
                               | (((qh >> (3*l)) & 0x07) << 8)];
```

The PR sat for eleven days with no comments and one tested card (an RTX 5060 Ti),
which is what got me interested. A miscompilation report with a sample size of
one is hard for a maintainer to act on.

## The rig

RTX 5080, Ubuntu 26.04, driver 595.91.07, gcc 15.2, llama.cpp master at
`9655061`. Five CUDA toolkits installed side by side.

The oracle is llama.cpp's own `test-backend-ops`, which runs every operation on
both the CUDA backend and the CPU reference and compares the results. For a
silent-wrong-number bug this is exactly the right tool, and it is already in the
tree - no fuzzing harness required.

```bash
GGML_CUDA_FORCE_MMQ=1 ./bin/test-backend-ops test -o MUL_MAT
GGML_CUDA_FORCE_MMQ=1 ./bin/test-backend-ops test -o MUL_MAT_ID
```

`GGML_CUDA_FORCE_MMQ=1` matters. Without it, llama.cpp picks the kernel by shape,
and at the sizes `test-backend-ops` generates it takes the vector-dot path, which
never enters the `load_tiles_*` functions where half the bug lives. My first run
was clean for exactly this reason, and I nearly wrote the whole thing off.

## Results

1682 `MUL_MAT` cases and 937 `MUL_MAT_ID` cases per configuration. Failures land
only on `iq1_s`, `iq2_s` and `iq3_s` - no other type moved.

For the patched column I cherry-picked the PR onto current master rather than
building its branch, which is based on an older commit with a different test set.
Same case counts on both sides, so the patch is the only variable.

I did not test the CUDA 12.x branch: 12.8 refuses gcc 15 (`unsupported GNU
version!`, the ceiling there is gcc 14), and `-allow-unsupported-compiler` would
have introduced the exact variable I was measuring.

## The two runs that found nothing

These were more informative than the ones that worked.

**A minimal reproducer does not reproduce.** The obvious next step after
confirming a codegen bug is to shrink it to twenty lines and send it to NVIDIA. I
wrote the pattern in isolation - a local `int`, a `uint8_t` cast, an unrolled
loop, `-O3`, `sm_120` - and it compiled correctly every time. Whatever trips the
optimizer needs the surrounding kernel: the grid table indirection, the register
pressure, the real loop body. So "same pattern" is not the trigger. Something
about the context is.

**The other sites with the same pattern are fine.** Before measuring anything I
grepped for every `(const uint8_t *) &packed` in the CUDA sources and found eleven
more: `iq1_m`, `iq2_xxs`, `iq3_xxs`, the `signs_packed_8` reads sitting *inside
the very functions the PR patches*, and the `q4_K`/`q5_K` scale reads. That looked
like an obvious gap in the fix, and I was ready to say so in review.

All eleven pass on the broken compiler. The PR's scope is correct and my
hypothesis was wrong. Had I posted the observation instead of measuring it, I
would have sent a maintainer chasing eleven non-bugs.

This is the part I want to remember: pattern matching generates candidates, not
findings.

## Installing five toolkits without touching the driver

Worth writing down, because the usual advice (runfile installers, `sudo`, one
version at a time) is not what I ended up doing.

NVIDIA publishes every toolkit component as a separate tarball under
`developer.download.nvidia.com/compute/cuda/redist/`, with a JSON manifest per
release. You can extract them into `$HOME`, merge the components into one root,
and have as many versions side by side as you like. No root, no package manager,
and the driver is never a participant.

Two things cost me time:

**In CUDA 13, `cicc` is not in `cuda_nvcc`.** The NVVM compiler moved to a
separate `libnvvm` component. Miss it and nvcc dies with `nvvm/bin/cicc: not
found` while looking perfectly installed. In CUDA 12 it is still bundled.

**CMake finds the toolkit but not its libraries.** With this layout
`find_package(CUDAToolkit)` locates the headers and reports success, then linking
fails with `undefined reference to cudaMalloc@libcudart.so.13`. Passing
`-L$CUDA_ROOT/lib -Wl,-rpath,$CUDA_ROOT/lib` through `CMAKE_EXE_LINKER_FLAGS` and
`CMAKE_SHARED_LINKER_FLAGS` sorts it out.

## What I still do not know

I could not find this in any NVIDIA release note, so I cannot say whether 13.2.2
fixed it deliberately or incidentally. I also cannot produce a standalone
reproducer, which means there is nothing clean to file upstream - and with the
symptom gone in current toolkits, there is not much point.

What is left is a decision for llama.cpp, and I posted the numbers
[on the PR][comment]: keep a workaround for the two affected releases, or require
CUDA >= 13.2.2 and delete the problem. Either is defensible. What was missing was
knowing that the window is two releases wide.

[pr]: https://github.com/ggml-org/llama.cpp/pull/28784
[comment]: https://github.com/ggml-org/llama.cpp/pull/28784#issuecomment-5773138673
