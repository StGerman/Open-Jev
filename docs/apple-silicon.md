# Open-Jev on Apple Silicon (MPS)

Open-Jev runs on the Apple GPU through PyTorch's MPS backend. Everything below
is opt-in: with no flag or variable set, CUDA and CPU behave exactly as before.

**Status.** The released 2B checkpoint runs entirely on Metal with no CPU fallback
(M4 MacBook Air, see [Released checkpoint results](#released-checkpoint-results)). On
that machine, MPS at the released bfloat16 is faster than the faster CPU dtype on every
measured workload, and makes the same decisions as CPU. Earlier checks on the random
test fixture (M4 Pro) are under [Fixture results](#fixture-results).

Docker cannot use the Apple GPU: containers on macOS run in a Linux VM without
Metal. Run natively.

## Setup

Apple's requirements for [PyTorch on Metal](https://developer.apple.com/metal/pytorch/):
Apple silicon, macOS 14.0 or later, Python 3.10 or later, Xcode command-line tools.

```bash
uv venv --python 3.12 .venv            # or python3.12 -m venv .venv
uv pip install --python .venv/bin/python -e '.[train]'
.venv/bin/python -c "import torch; print(torch.ones(1, device='mps'))"   # tensor([1.], device='mps:0')
```

Fetch the pinned checkpoint package and its base with the same verified fetcher the
Docker image uses; it checks every package file against the published manifest and
refuses a base revision other than the one `checkpoint/model.json` names:

```bash
export JEV_MODEL_ROOT="$HOME/models/open-jev"
JEV_PACKAGE_REPO=ZefanCai/Open-Jev-2B JEV_PACKAGE_REVISION=0c7aa498b1627be8da4acf34c863ff0ee0a92785 \
JEV_BASE_MODEL=Qwen/Qwen3.5-2B JEV_BASE_REVISION=15852e8c16360a2fea060d615a32b45270f8a8fc \
  .venv/bin/python docker/fetch_models.py
export HF_HUB_CACHE="$JEV_MODEL_ROOT/hub" HF_HUB_OFFLINE=1
export CKPT="$JEV_MODEL_ROOT/Open-Jev-2B/package/checkpoint"
```

## Running

```bash
.venv/bin/python -m jev.server --checkpoint "$CKPT" --device mps
.venv/bin/python -m jev.predict --checkpoint "$CKPT" --device mps --request configs/example-request.json
```

| Setting | Effect on MPS |
| --- | --- |
| `--device mps` | Required; the default stays `cuda:0` |
| `JEV_TORCH_DTYPE=bfloat16\|float16\|float32` | Backbone dtype; unset keeps the released `bfloat16`. The head stays float32. Any other dtype changes numerics against the saved calibration, so its quality is not comparable to the released numbers. Responses record the loaded dtypes in `metadata.backbone_parameter_dtypes` |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Leave unset. Set to `1`, it silently runs unsupported ops on the CPU; unset, they raise |
| `JEV_LOAD_8BIT`, `JEV_LOAD_4BIT` | Refused on MPS: bitsandbytes has no Apple backend |
| `JEV_PROFILE=1` | Per-phase cached-path timings, now synchronizing Metal instead of calling CUDA |

## Proving the GPU is used

A working request does not show that the GPU did the work. This check exits non-zero
unless every parameter is resident on MPS, Metal holds allocated memory, no
CPU-fallback warning is raised and MPSProfiler logs no CPU fallback. It also reports
Apple GPU `Device Utilization %` (from `ioreg`, no sudo) idle and while scoring:

```bash
env -u PYTORCH_ENABLE_MPS_FALLBACK .venv/bin/python -m scripts.check_mps_engagement \
  --checkpoint "$CKPT" --profile-log reports/mps-profile.log --output reports/mps-engagement.json
```

`--profile-log` re-runs the scoring in a child process with MPSProfiler statistics
(`PYTORCH_MPS_LOG_PROFILE_INFO`: operation, copy and CPU-fallback tables with GPU
time), following the [PyTorch MPS Backend wiki](https://github.com/pytorch/pytorch/wiki/MPS-Backend#pytorch-performance-profiling-using-mps-profiler).
For a timeline, `--signposts` wraps the scoring in `torch.mps.profiler.profile(mode="interval")`;
record it with Instruments (`Logging` template for the OS Signposts, `Metal System Trace`
for the GPU):

```bash
xcrun xctrace record --template Logging --output mps.trace --launch -- \
  .venv/bin/python -m scripts.check_mps_engagement --checkpoint "$CKPT" --signposts
```

Profiling slows execution; never profile inside a timed benchmark run.

## Measuring MPS against CPU on the same Mac

A same-machine comparison is a stronger claim than one against the published H100
rows. `scripts/benchmark_inference_latency.py` synchronizes Metal around every timed
call; without that, asynchronous dispatch would make every MPS time an understatement.
MPS has no peak-memory counter, so samples record point-in-time allocator readings
(`mps_allocated_bytes_at_start/_at_end`, `mps_driver_allocated_bytes_at_end`), never a peak.

1. **Prepare identical inputs once.** Every later run replays this file; the harness
   refuses a changed request by checksum.

   ```bash
   B=reports/inference-latency; R="$B/2b-m4pro-prepare/requests.json"
   .venv/bin/python -m scripts.benchmark_inference_latency --checkpoint "$CKPT" --prepare-only \
     --request examples/community/drone.json --request examples/workflows/customer_service.json \
     --contexts 128 512 --candidates 2 8 --output "$B/2b-m4pro-prepare"
   ```

   Six workloads instead of the published eleven: CPU would exceed the time budget on the
   1,024-token, 32-candidate points. Run those separately (`--contexts 1024 --candidates 32`,
   fewer repetitions) if needed, and state the reduced matrix with any result.

2. **Pilot each device** (`--warmup 1 --repetitions 1`) and size the real runs from its
   per-call times; `--max-seconds` is capped at 3600.
3. **Measure**, plugged in, High Power energy mode, under `caffeinate -dimsu`, with
   `pmset -g therm` recorded before and after. Alternate the order (CPU then MPS, later
   MPS then CPU) with a cooldown between runs, and run each configuration at least twice:

   ```bash
   caffeinate -dimsu .venv/bin/python -m scripts.benchmark_inference_latency --checkpoint "$CKPT" \
     --requests "$R" --device mps --warmup 3 --repetitions 20 --output "$B/2b-m4pro-mps-run1"
   caffeinate -dimsu .venv/bin/python -m scripts.benchmark_inference_latency --checkpoint "$CKPT" \
     --requests "$R" --device cpu --warmup 3 --repetitions 20 --output "$B/2b-m4pro-cpu-run1"
   ```

   Measure CPU at its default `bfloat16` and again with `JEV_TORCH_DTYPE=float32`, and
   compare MPS against the faster of the two, so a slow CPU dtype path does not inflate
   the GPU speedup. A single uncached CPU `bfloat16` request can exceed the harness's
   default 120 s loopback timeout; pass `--http-timeout 1800` for CPU runs.

   On a fanless Mac (MacBook Air), MPS throttles within minutes of sustained load and a
   5-minute cooldown does not bring it back: report short cool runs and long heat-soaked
   runs separately, and compare knobs back to back at the same thermal state.
   `pmset -g therm` records nothing on Apple Silicon.
4. **Compare answers across devices.** Drift between backends is expected; changed
   decisions are what matter, and are listed per request:

   ```bash
   .venv/bin/python -m scripts.compare_device_reports "$B/2b-m4pro-cpu-run1" "$B/2b-m4pro-mps-run1" \
     --output "$B/2b-m4pro-cpu-vs-mps.json"
   ```

5. **Try one knob at a time on MPS**, reporting speed and parity together: `JEV_RAGGED_SUFFIX=1`,
   `JEV_PREFILL_CHUNK=128` or `256`, `JEV_NO_QPREFILL=1`, `--batch-size 8` or `64`,
   `JEV_TORCH_DTYPE=float16`. The first three change only the cached path, so the uncached
   rows of the same run are a built-in control. Do not sweep `JEV_PREFIX_CACHE`: every run
   already measures cached and uncached paths. A knob that changes a decision is not a free
   win; on CUDA, `JEV_NO_QPREFILL=1` was 11% faster but cost 2 of 420 answers.

Report with the numbers: the power mode and thermal state, the reduced workload matrix,
the dtype on each device, and that no hardware-matched claim against the H100 rows is
made (different kernels: the fused linear-attention kernels are CUDA-only, so Metal runs
the pure-PyTorch reference path).

## Released checkpoint results

`ZefanCai/Open-Jev-2B` at `0c7aa498`, base `Qwen/Qwen3.5-2B` at `15852e8c`, fetched and
verified with `docker/fetch_models.py`. MacBook Air (Mac16,12): Apple M4, 10-core CPU
(4 performance + 6 efficiency), 10-core GPU, 24 GB; macOS 26.6.2, Python 3.12.13,
torch 2.14.0, transformers 5.10.2, peft 0.19.1. On AC power with Low Power Mode off;
this model has no High Power mode and no fan. Every run used `caffeinate -dimsu` and
left `PYTORCH_ENABLE_MPS_FALLBACK` unset. CPU runs use torch's default of 4 threads.
Raw reports are in `reports/inference-latency/2b-m4air-*`.

### GPU engagement

`scripts/check_mps_engagement.py` passes every check: all 442 parameters are on MPS,
Metal holds 3.8 GB (5.5 GB driver) after loading, no fallback warning is raised, and
MPSProfiler logs no CPU fallback. Apple GPU `Device Utilization %` has a median of 12%
when idle and 100% while scoring. `configs/example-request.json` takes a median of
0.97 s per request.

MPSProfiler recorded about 10,000 kernel dispatches per request: 41% `copy_identity`
(layout copies), then float32 elementwise `mul`, `add` and `sum` reductions from the
linear-attention reference path. Matrix multiplies (`mps_linear`, `bmm`) are about 5%
of dispatches. The profiler's per-kernel GPU-time column is attributed per command
buffer and is not additive (it sums to about 150× wall time), so rank by dispatch
count, not by that column. Loading moved 4.1 GiB CPU→MPS in 757 copies.

### Latency

Predictor path, median seconds per request (`p50`). Six workloads, not the published
eleven: the 1,024-token and 32-candidate points are left out.

- **MPS sustained:** two runs of 20 measured repetitions each, heat-soaked; each ran after 40–60 minutes of CPU load and a 5-minute cooldown.
- **MPS cool:** three repetitions on an idle machine.
- **CPU fp32:** 10 repetitions per run.
- **CPU bf16:** 2 repetitions per run, cut off by the 3,600 s budget. `context-512-choice-8` got 1 repetition.
- **Speedup:** CPU fp32 ÷ MPS sustained, using the mean of each pair of runs.

| Workload | Path | MPS sustained, run 1 / 2 | MPS cool | CPU fp32, run 1 / 2 | CPU bf16, run 1 / 2 | MPS speedup over CPU fp32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `drone` | uncached | 1.71 / 1.68 | 1.57 | 4.98 / 5.31 | 38.93 / 42.54 | 3.0× |
| `drone` | cached | 0.96 / 0.95 | 0.91 | 13.36 / 13.15 | 20.58 / 23.38 | 13.9× |
| `customer_service` | uncached | 10.61 / 11.34 | 7.11 | 16.57 / 18.09 | 210.17 / 205.00 | 1.6× |
| `customer_service` | cached | 2.68 / 3.07 | 1.93 | 14.16 / 14.23 | 47.01 / 47.00 | 4.9× |
| `context-128-choice-2` | uncached | 0.94 / 0.83 | 0.56 | 3.07 / 3.09 | 15.72 / 15.25 | 3.5× |
| `context-128-choice-2` | cached | 0.60 / 0.65 | 0.41 | 4.19 / 4.00 | 12.39 / 10.93 | 6.6× |
| `context-128-choice-8` | uncached | 3.93 / 2.97 | 2.27 | 6.31 / 7.08 | 57.69 / 51.95 | 1.9× |
| `context-128-choice-8` | cached | 1.25 / 0.88 | 0.77 | 4.28 / 5.61 | 17.08 / 16.46 | 4.6× |
| `context-512-choice-2` | uncached | 3.24 / 2.36 | 1.97 | 5.42 / 5.26 | 42.61 / 40.48 | 1.9× |
| `context-512-choice-2` | cached | 1.78 / 1.32 | 1.09 | 5.68 / 4.90 | 24.36 / 23.20 | 3.4× |
| `context-512-choice-8` | uncached | 13.07 / 11.20 | 12.17 | 15.42 / 15.80 | 191.87 / 178.27 | 1.3× |
| `context-512-choice-8` | cached | 2.18 / 2.01 | 2.34 | 5.39 / 5.70 | 29.99 / 29.57 | 2.6× |

- **MPS against CPU:** MPS bfloat16 beats CPU float32 on every row: 1.3–3.5× uncached and 2.6–13.9× cached. The CPU baseline is float32 because CPU bfloat16 is 4–12× slower still.
- **Cached path on CPU:** the request-local prefix cache is slower than uncached on CPU for `drone` (13.3 s against 5.1 s) and `context-128-choice-2`. On MPS it is faster on every row.
- **Throttling:** the fanless Air throttles. Cool MPS runs are up to 1.7× faster than heat-soaked ones (`customer_service` uncached: 7.1 s against 10.6–11.3 s). The heat-soaked medians also vary by up to 1.4× between runs. Within a cool three-repetition run, `context-512-choice-8` had already slowed to heat-soaked speed. Treat the sustained columns as this machine's steady state, not as a property of Metal.

### Answer parity

- **Cached against uncached, within a run:** every MPS run reports `parity_failed` at the harness's 1e-4 bar. No decision changes; the maximum probability error is 5.3e-3. CPU bfloat16 fails it the same way (up to 4.3e-3), and so does the published H100 bfloat16 run (up to 5.7e-3), so this is bfloat16 drift between the cached and uncached computations, not a Metal defect. CPU float32 runs pass the bar (2.2e-6).
- **CPU against MPS** (`scripts/compare_device_reports.py`, first measured response per request, both cache modes): all decisions match on all six requests. The maximum probability error is 4.5e-3 against CPU float32 and 7.8e-3 against CPU bfloat16.
- **Repeat runs:** two runs on the same device give identical responses (error 0.0) on both CPU and MPS.

### Knobs on MPS

Twelve back-to-back runs on the heat-soaked machine, 8 measured repetitions each, each
knob run sitting between baseline runs (`tune-*`). The table gives the sum of per-workload
medians over the six workloads. It also gives the ratio of cached to uncached time within
each run: this is the control for the three cached-path knobs, because their uncached rows
are unchanged. Decisions are compared against the neighbouring baseline in both cache modes.

| Run | Uncached sum, s | Cached sum, s | Cached / uncached | Decisions vs baseline | Max probability difference vs baseline |
| --- | ---: | ---: | ---: | --- | ---: |
| baseline (5 runs) | 28.4, 31.9, 32.7, 31.7, 32.4 | 8.1, 9.9, 9.3, 9.1, 9.7 | 0.285 – 0.311 | — | — |
| `JEV_RAGGED_SUFFIX=1` | 30.2 | 8.6 | 0.284 | equal | 3.7e-3 |
| `JEV_PREFILL_CHUNK=128` | 33.7 | 9.5 | 0.283 | equal | 2.4e-3 |
| `JEV_PREFILL_CHUNK=256` | 33.9 | 9.5 | 0.281 | equal | 3.6e-4 |
| `JEV_NO_QPREFILL=1` | 34.5 | 10.5 | 0.304 | equal | 3.2e-3 |
| `--batch-size 8` | 34.0 | 9.9 | 0.292 | identical | 0 |
| `--batch-size 64` | 32.5 | 9.4 | 0.290 | identical | 0 |
| `JEV_TORCH_DTYPE=float16` | 34.4 | 9.8 | 0.286 | equal | 4.1e-3 |

The first baseline ran on a machine that had been idle for an hour and is faster. The other four,
all heat-soaked, agree within 3% uncached and 9% cached.

- **Batch size is a no-op here, which measures noise.** Every workload has at most 8 candidates, so batch sizes 8, 32 and 64 all run one batch and produce identical responses. Their timings still differ from the neighbouring baselines by up to about 8%, so differences of that size are noise on this machine.
- **Cached-path knobs:** none of `JEV_RAGGED_SUFFIX`, `JEV_PREFILL_CHUNK=128/256` or `JEV_NO_QPREFILL` lowers the cached/uncached ratio below the baseline range. `JEV_NO_QPREFILL=1`, 11% faster on CUDA, is the slowest cached run here. It also halves the within-run cached/uncached drift (3.1e-3 against 5.3e-3).
- **float16:** no faster than bfloat16 (+7% uncached, within noise to slightly slower). Its answers are closer to CPU float32 (at most 9.7e-4, against 4.5e-3 for bfloat16), but the saved calibration was fit at bfloat16, so its quality is not comparable to the released numbers.

Keep the defaults on MPS. The profile above suggests why none of these knobs help. Most
dispatches are layout copies and float32 elementwise work from the linear-attention
reference path, not matrix multiplies, so the time goes to dispatch and memory traffic.
Knobs that change how prefill is batched or chunked barely touch that. A code-level gain
would have to cut the copies and dispatches in that path, for example by fusing or
compiling it for Metal. That is a hypothesis to profile first, not a measured result.

### Caveats

- **Workloads:** the matrix is reduced to six workloads (no 1,024-token or 32-candidate points), and the CPU bfloat16 rows have 1–2 repetitions.
- **Machine:** these numbers come from an M4 MacBook Air, which has no High Power mode and no fan. The 20-repetition MPS numbers are heat-soaked, and thermal state was not recorded by `pmset`.
- **Dtypes:** MPS runs at the released bfloat16 backbone; CPU runs at float32 (faster) and bfloat16. The head is float32 on both.
- **Not comparable to H100:** no hardware-matched claim is made against the published H100 rows. The fused linear-attention kernels are CUDA-only, so Metal and CPU both run the pure-PyTorch reference path.

## Fixture results

Random hybrid-Qwen fixture from `tests/test_prefix_cache.py`, all three model profiles
with and without LoRA, `PYTORCH_ENABLE_MPS_FALLBACK` unset; M4 Pro (16-core GPU, 24 GB),
macOS 26.5.1, Python 3.12.12, torch 2.14.0, transformers 5.10.2, peft 0.19.1. These test
execution and numerics, not model quality.

| Backbone dtype | MPS vs CPU, max logit error | Decisions differing (6 configs) | Error vs float32 reference, MPS / CPU |
| --- | ---: | ---: | --- |
| float32 | 4.8e-7 – 1.3e-6 | 0 | same as CPU |
| bfloat16 | 6.7e-3 – 2.3e-2 | 2 | 4.8e-3 – 2.3e-2 / 3.6e-3 – 2.3e-2 |
| float16 | 8.7e-4 – 3.7e-3 | 0 | 8.7e-4 – 7.9e-3 / 8.7e-4 – 5.7e-3 |

bfloat16 is as noisy on CPU as on Metal; the two differing decisions are near-tied
candidates of a random model disagreeing between two equally noisy computations. With
the backbone in bfloat16, MPSProfiler recorded 97 graphs and 80 kernels for uncached,
cached and ragged scoring and logged no CPU fallback. Most dispatches were small copy,
elementwise and reduction kernels from the linear-attention reference path; matrix
multiplies were a small share. Re-profile on the released checkpoint before tuning.

`tests/test_mps_support.py` holds these checks and runs wherever MPS is available:
`env -u PYTORCH_ENABLE_MPS_FALLBACK .venv/bin/python -m unittest tests.test_mps_support -v`.
