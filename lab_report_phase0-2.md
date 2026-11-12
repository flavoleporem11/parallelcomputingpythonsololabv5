# Parallel Computing Lab: Phases 0-2 (Code + Analysis Scaffold)

**How to use:** Save each code block as the named file in `parallel_lab/`, set `SEED`, `PPHYS` (from your OS commands), run, and paste the output into the `[PASTE OUTPUT]` slots. All numbers must come from your own machine. Tables below are blank on purpose.

---

## 0. Hardware Baseline

Run the OS command from the lab sheet, then fill in:

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| CPU model | [fill] | L1d (per core) | [fill] KB |
| Physical cores (Pphys) | [fill] | L2 | [fill] |
| Logical threads (Plog) | [fill] | L3 (shared) | [fill] MB |
| Cache line (bytes) | [fill] (64 x86, 128 Apple M) | Python / OS | [fill] |

### harness.py (as given in the lab sheet)

```python
from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Callable

@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float

def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def run_benchmark(fn: Callable[[], None], warmup_runs: int = 3, timed_runs: int = 9) -> BenchStats:
    for _ in range(warmup_runs):
        fn()
    samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    samples.sort()
    mid = timed_runs // 2
    med = samples[mid] if timed_runs % 2 == 1 else 0.5 * (samples[mid - 1] + samples[mid])
    q25 = samples[timed_runs // 4]
    q75 = samples[(3 * timed_runs) // 4]
    return BenchStats(samples[0], med, q75 - q25, samples[-1])

def emit_verify_tag(task_id: str, student_seed: int,
                    primary_metric: float, secondary_metric: float) -> str:
    hw = mp.cpu_count()
    payload = f"{task_id}|{student_seed}|{hw}|{primary_metric:.4f}|{secondary_metric:.4f}"
    sig = fnv1a_hash(payload)
    return f"VERIFY_HASH[{task_id}]: {sig:016X} (P_log={hw})"
```

### task0_cal.py

The lab does not say which two metrics go into PHASE0. I use `(Plog, avg_ns_per_call)`; adjust if your instructor specifies otherwise.

```python
import time
import multiprocessing as mp
from harness import emit_verify_tag

SEED = 0  # <-- your Student ID (integer)

if __name__ == "__main__":
    plog = mp.cpu_count()
    N = 2_000_000
    pc = time.perf_counter_ns
    t0 = pc()
    for _ in range(N):
        pc()
    t1 = pc()
    avg_ns = (t1 - t0) / N
    print(f"mp.cpu_count() = {plog}")
    print(f"avg perf_counter_ns call latency = {avg_ns:.2f} ns over {N} calls")
    print(emit_verify_tag("PHASE0", SEED, float(plog), avg_ns))
```

**[PASTE OUTPUT]**

Note: the loop overhead itself is included in `avg_ns`, so this is an upper bound on pure clock-read cost. Any timed region much shorter than a few microseconds is below your noise floor.

---

## 1. Phase 1: False Sharing and Karp-Flatt

### task1_falsesharing.py

```python
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag

SEED = 0      # your Student ID
PPHYS = 4     # from Phase 0
M = 24_000_000

def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc
    del arr
    shm.close()

def make_runner(shm, p, stride):
    def run():
        shm.buf[:4096] = bytes(4096)          # reset slots
        iters = M // p
        procs = [mp.Process(target=fs_worker, args=(shm.name, t * stride, iters))
                 for t in range(p)]
        for pr in procs: pr.start()
        for pr in procs: pr.join()
    return run

def plist(plog):
    ps, p = [], 1
    while p <= plog:
        ps.append(p); p *= 2
    for extra in (PPHYS, plog):
        if extra not in ps: ps.append(extra)
    return sorted(ps)

if __name__ == "__main__":
    plog = mp.cpu_count()
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        # Table 1A
        print("### Table 1A: stride sweep at p = PPHYS")
        print("| stride | bytes | median ms | IQR ms | Mops/s |")
        print("|---|---|---|---|---|")
        for stride in (1, 2, 4, 8, 16, 32):
            s = run_benchmark(make_runner(shm, PPHYS, stride), 2, 7)
            mops = M / (s.median_ms / 1000) / 1e6
            print(f"| {stride} | {stride*8} | {s.median_ms:.1f} | {s.iqr_ms:.1f} | {mops:.2f} |")

        # Table 1B
        med_at_pphys = {}
        for stride in (1, 16):
            print(f"\n### Table 1B: stride = {stride}")
            print("| p | median ms | S_p | E_p | e(p) |")
            print("|---|---|---|---|---|")
            t1 = None
            for p in plist(plog):
                s = run_benchmark(make_runner(shm, p, stride), 2, 7)
                if p == 1: t1 = s.median_ms
                sp = t1 / s.median_ms
                ep = sp / p
                kf = ((1/sp - 1/p) / (1 - 1/p)) if p >= 2 else float("nan")
                if p == PPHYS: med_at_pphys[stride] = s.median_ms
                print(f"| {p} | {s.median_ms:.1f} | {sp:.2f} | {ep:.2f} | {kf:.4f} |")
        print()
        print(emit_verify_tag("PHASE1", SEED, med_at_pphys[1], med_at_pphys[16]))
    finally:
        shm.close()
        shm.unlink()
```

**[PASTE OUTPUT: Table 1A, Table 1B x2, VERIFY_HASH[PHASE1]]**

### Analysis guide (write your conclusions using your numbers)

**Cache-line boundary.** Find the smallest stride after which the median time drops sharply and stays flat. Byte offset = stride x 8. Expected: 64 B (stride 8) on most x86-64, 128 B (stride 16) on Apple Silicon. State what you measured, even if it differs.

**Caveat specific to CPython (include this in your report).** Each loop iteration runs roughly 100+ ns of interpreter work, while a coherence invalidation costs on the order of tens to ~100 ns. The store is therefore only part of the iteration, so the false-sharing penalty is diluted compared with C. You may see a modest effect (tens of percent) rather than several-fold. If the effect is within your IQR, say so honestly and compare medians against the noise.

**Karp-Flatt interpretation.** `e(p) = (1/S_p - 1/p) / (1 - 1/p)`.
- If e(p) stays roughly constant as p grows, the loss is a genuine serial fraction (e.g., process-start overhead, fixed per-run cost).
- If e(p) rises with p, the loss grows with parallelism: overhead that scales with the number of workers. Under stride = 1, every added process adds another writer invalidating the same line, so coherence traffic per store grows with p, which is why e(p) climbs.
- Under stride = 16 each worker has a private line, so e(p) should stay low and flat, apart from process spawn cost and SMT effects once p > Pphys.
- Spawn/join cost is included in the timing, so e(p) has a non-zero floor even for stride = 16. Mention it.

---

## 2. Phase 2: Strong and Weak Scaling

### task2_scaling.py

Runtime note: with N = 1.6e6 and 35 iterations per element, p = 1 is slow (tens of seconds per run). I use `warmup=1, timed=5`; change `WARM`/`TIMED` if your time budget allows.

```python
import math
import multiprocessing as mp
from harness import run_benchmark, emit_verify_tag

SEED = 0
PPHYS = 4
WARM, TIMED = 1, 5

def stage_a(n: int) -> float:
    """Serial prefix: 12 transcendental ops per element over first 5%."""
    acc = 0.0
    for i in range(int(0.05 * n)):
        x = i * 1e-3 + 1.0
        for _ in range(3):  # 3 x (sin, sqrt, sin, sqrt) = 12 ops
            x = math.sqrt(math.sin(x) ** 2 + x) + math.sin(x * 0.5) + math.sqrt(x)
        acc += x
    return acc

def kernel(rng):
    lo, hi = rng
    acc = 0.0
    for i in range(lo, hi):
        x = i * 1e-6
        for _ in range(35):
            x = x * 1.0000001 + 0.5          # multiply-add
            x = math.sin(x) + 1e-3 * x       # trig
        acc += x
    return acc

def chunks(n, p):
    step = n // p
    return [(t * step, n if t == p - 1 else (t + 1) * step) for t in range(p)]

def trial(pool, n, p, a_times):
    import time
    t0 = time.perf_counter_ns()
    stage_a(n)
    a_times.append((time.perf_counter_ns() - t0) / 1e6)
    pool.map(kernel, chunks(n, p))

def median(xs):
    xs = sorted(xs); m = len(xs)
    return xs[m // 2] if m % 2 else 0.5 * (xs[m // 2 - 1] + xs[m // 2])

def measure(n, p):
    a_times = []
    pool = mp.Pool(processes=p)          # created OUTSIDE timed region
    try:
        s = run_benchmark(lambda: trial(pool, n, p, a_times), WARM, TIMED)
    finally:
        pool.close(); pool.join()
    return s.median_ms, median(a_times[-TIMED:])

if __name__ == "__main__":
    plog = mp.cpu_count()
    ps = list(range(1, plog + 1))

    # ---- 2A strong scaling
    N = 1_600_000
    strong = {}
    for p in ps:
        strong[p] = measure(N, p)
    t1, a1 = strong[1]
    print("### Table 2A: strong scaling (N = 1.6e6)")
    print("| p | T ms | S_p | E_p | marginal dS |")
    print("|---|---|---|---|---|")
    prev = 0.0
    for p in ps:
        S = t1 / strong[p][0]
        print(f"| {p} | {strong[p][0]:.1f} | {S:.2f} | {S/p:.2f} | {S-prev:+.2f} |")
        prev = S
    s_frac = a1 / t1
    s_inf = 1 / s_frac
    s_amdahl_plog = 1 / (s_frac + (1 - s_frac) / plog)
    s_obs = t1 / strong[plog][0]
    print(f"\nserial fraction s = T_A(1)/T_1 = {s_frac:.4f}")
    print(f"Amdahl S_inf = {s_inf:.2f}; theoretical S(Plog={plog}) = {s_amdahl_plog:.2f}; observed = {s_obs:.2f}")

    # ---- 2B weak scaling
    N0 = 250_000
    weak = {}
    for p in ps:
        weak[p] = measure(p * N0, p)
    tw1 = weak[1][0]
    print("\n### Table 2B: weak scaling (N0 = 2.5e5 per worker)")
    print("| p | T ms | E_weak | alpha | S_Gustafson |")
    print("|---|---|---|---|---|")
    g_plog = None
    for p in ps:
        T, A = weak[p]
        alpha = A / T
        g = p - alpha * (p - 1)
        if p == plog: g_plog = g
        print(f"| {p} | {T:.1f} | {tw1/T:.2f} | {alpha:.4f} | {g:.2f} |")
    print()
    print(emit_verify_tag("PHASE2", SEED, s_obs, g_plog))
```

**[PASTE OUTPUT: Tables 2A, 2B, Amdahl numbers, VERIFY_HASH[PHASE2]]**

### Analysis guide

**Amdahl ceiling.** `s = T_StageA(1)/T_1`; `S_inf = 1/s`; `S(p) = 1 / (s + (1-s)/p)`. Compare `S(Plog)` with your observed value. Expect observed < theoretical, because Amdahl's formula ignores `pool.map` pickling/IPC cost, load imbalance (the last chunk takes the remainder), turbo-frequency drop under full load, and shared L3/memory contention. The gap `S_theory - S_obs` is your unmodeled overhead. Report it as a number.

Also note that the serial fraction here is *small* (about 5% of elements x 12 ops, versus 100% x 35 iterations of heavier work), so `S_inf` will be large and the real ceiling at your core count comes from hardware, not from `s`.

**Boundary at p = Pphys to Pphys + 1 (SMT or hybrid cores).** From the `marginal dS` column:
- `avg_below = mean(dS for p = 2..Pphys)`
- `drop% = (avg_below - dS at p=Pphys+1) / avg_below x 100`
- Repeat for each p > Pphys and report the per-worker drop.

Typical result with SMT: below Pphys each added worker adds about 0.8-1.0 of a core; above it, SMT siblings add roughly 0.1-0.3 (a drop of 60-90%). On a hybrid P-core/E-core chip the knee comes at the P-core count instead, and E-cores add a smaller increment (often 30-60% of a P-core). Identify which case you are in from your Phase 0 table, and state the exact percentage you computed. If the Python interpreter is the bottleneck (it is mostly integer/branchy code), SMT yield is often better than for dense FP code, so don't assume the textbook figure.

**Strong vs. weak.** Strong scaling E_p falls as p rises (fixed N, fixed serial part). Weak E_weak stays higher because per-worker work is constant. One subtlety in this lab: Stage A scales with N(p), so under weak scaling the *serial work grows with p*. That makes `alpha` roughly constant (about `s`-sized) and Gustafson `S = p - alpha(p-1)` close to p minus a small term. If your E_weak is much lower than S_Gustafson/p suggests, the difference is overhead the Gustafson model does not capture (IPC, memory bandwidth, SMT).

---

## Notes / things that commonly go wrong

- **spawn on macOS/Windows:** all worker functions at module top level, entry code under `if __name__ == "__main__":`. The `lambda` passed to `run_benchmark` is fine because it runs only in the parent.
- **`BufferError` on `shm.close()`:** you must `del arr` before closing, as the worker does.
- **Leaked shared memory:** always `unlink()` in a `finally`.
- **Noisy results:** close other apps, plug in power, avoid thermal throttling (the first runs are often fastest), and report the IQR.
- **Pphys:** there is no stdlib call for it; read it from your Phase 0 OS output and set `PPHYS` by hand.
