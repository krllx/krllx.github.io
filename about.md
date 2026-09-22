---
layout: page
title: About
permalink: /about/
---

I am a backend engineer who got curious about what happens below the API, and
ended up writing CUDA kernels and chasing compilers that produce the wrong
number without telling anyone.

Most of what is here concerns consumer Blackwell - the RTX 50 series, compute
capability `sm_120`. It is not a smaller B200: no TMEM, no `tcgen05`, no WGMMA,
and 99-101 KB of shared memory per block. Libraries get tuned on H100 and B200,
so the consumer part tends to collect its own set of bugs that nobody is
positioned to reproduce.

I have one of those cards, so I reproduce them.

- GitHub: [krllx](https://github.com/krllx)
