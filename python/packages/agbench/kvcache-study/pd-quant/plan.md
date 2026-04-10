# Design doc

## Experiments
1. Uniform BF16, single instance — quality ceiling, throughput floor.
2. Uniform FP8, single instance — the obvious deployment baseline. This is the one you have to beat.
3. Uniform BF16 with PD disaggregation — isolates the disagg gain from the precision gain.
4. Uniform FP8 with PD disaggregation — this is the real baseline. If your mixed-precision PD doesn't beat uniform-FP8 PD, you have nothing. Most reviewers will zero in on this comparison.
5. Mixed: BF16 prefill + FP8 decode (PD disagg) — our proposal.

## Hardware config
Sanity baseline: single gpu, time-shared
Multiple nodes but homogeneous hardware
Heterogeneous hardware

