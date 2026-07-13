# Native ARM64 PyTorch on Windows on ARM (Snapdragon X Elite): what works, what doesn't, and the path that actually trains

*Tested 2026-07-13 on a Snapdragon X Elite (X1E80100), Windows 11 25H2 build 26200, 64 GB RAM. Every command and number below comes from a logged session; the scripts are included so you can reproduce each claim on your own box.*

---

## TL;DR

**The official native Windows ARM64 PyTorch wheel is slower than running the x86 wheel under Prism emulation**: 2–3.5× slower on training-shaped workloads on a Snapdragon X Elite, with a bf16 path that effectively does not run. The working native path for training today is WSL2. Data, scripts, and reproduction below.

The details:

1. **Official native ARM64 Windows wheels exist** (announced April 2025; `torch-2.12.1+cpu` ships `win_arm64` on the stable index, **Python 3.11 only**):

   ```powershell
   py -3.11-arm64 -m venv .venv-arm64
   .venv-arm64\Scripts\python -m pip install --no-cache-dir torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
   ```

2. **The wheel is genuinely native** (verified at the binary level, see the verification script below), but on Snapdragon X Elite it loses to its own x86 twin under emulation: 2.882s vs 1.365s on a 4096² fp32 GEMM, 44.9s vs 13.0s per iteration on a training-shaped fp32 loop, and bf16 never completes on either Windows substrate. It also **cannot install `datasets`/`trl`** because pyarrow publishes no `win_arm64` wheel on any index (tracked upstream in [apache/arrow#47195](https://github.com/apache/arrow/issues/47195)) — **gap since closed by a native from-source build: [Windows_ARM64_PyArrow](https://github.com/LucRoot/Windows_ARM64_PyArrow); see §6 update**.
3. **WSL2 (Linux aarch64) is the only substrate where the full stack trains.** Measured 1.64× faster than x86-emulated torch on fp32 GEMM, bf16 completes at 9.1s/iter, and a real 1.83B-parameter LoRA dry-run passed end to end. Same box, no new hardware.
4. Run the 30-second diagnostic in §1 on your own machine first. Most Windows-on-ARM PyTorch installs are silently x86-emulated.

---

## 1. Why this matters: the emulation tax, and how to tell you're paying it

Windows on ARM runs x86_64 binaries through the Prism translation layer. If your Python interpreter is an x86_64 build (the default from python.org until recently, and what most tooling still installs), then **everything** in your ML stack runs translated: Python itself, torch, numpy, all native extensions.

On this machine, the training pipeline's authoritative wall-clock convention applies a **~3× per-cycle multiplier** for emulation; its full-pipeline estimate is **~40h native versus ~200h emulated** across 11 cycles. Those estimates assume a competent native BLAS; §5 measures what the official native wheel actually delivers. Translation cost is worst on workloads made of many small kernel launches, Python-level dispatch, and I/O, which is exactly the shape of a training loop.

### The 30-second diagnostic

```powershell
python -c "import platform, torch; print(platform.machine(), torch.__version__, torch.backends.cpu.get_cpu_capability())"
```

| Output | Meaning |
|---|---|
| `AMD64 2.12.1+cpu AVX2` | **x86-emulated.** Interpreter is an x86_64 binary; `AVX2` is x86 SIMD that Prism is translating to NEON. You are paying the tax. |
| `ARM64 2.12.1+cpu DEFAULT` | Native ARM64. (`DEFAULT` is what an MSVC ARM64 build reports; see §6 caveat.) |

Second check, for numpy (a common second source of emulated binaries):

```powershell
python -c "import numpy; numpy.show_config()"
```

x86 build: `cp311-win_amd64`, `SkylakeX`, `X86_V3 SIMD`. ARM64 build: `cp311-win_arm64`, `ARMV8`, `ASIMD` (NEON).

---

## 2. Environment

| Item | Value |
|---|---|
| CPU | Snapdragon X Elite X1E80100, 12 Oryon cores (ARMv8, NEON, no SVE/SVE2, no BF16 SIMD) |
| OS | Windows 11 25H2, build 10.0.26200 |
| RAM | 64 GB |
| Native Python used | CPython 3.11.15 ARM64 (via `uv python install cpython-3.11-windows-aarch64-none`) |
| Also present | CPython 3.12.10 ARM64 from python.org |
| WSL2 | Ubuntu 24.04, Python 3.12.3 aarch64 |
| Torch tested | 2.12.1+cpu, three ways: win_arm64 wheel, x86_64 wheel under Prism, manylinux_aarch64 wheel in WSL2. All three builds are the same upstream commit (`7269437d`, the v2.12.1 tag). |

One warning about `uv`: the `uv` binary on this box is itself x86_64, and `uv python install 3.11` without an explicit architecture triple will fetch an **x86_64** interpreter and silently recreate the emulation problem. Always pin it:

```powershell
uv python install cpython-3.11-windows-aarch64-none
uv python find cpython-3.11-windows-aarch64-none   # locate it later; uv installs do not register with the py launcher
```

---

## 3. Step-by-step: native ARM64 torch on Windows

```powershell
# 1. ARM64 Python 3.11 (the 2.12.1+cpu win_arm64 wheel is cp311-only)
uv python install cpython-3.11-windows-aarch64-none
#    or: python.org 3.11.x ARM64 installer

# 2. Isolated venv
& "<path-from-step-1>\python.exe" -m venv .venv-arm64

# 3. Pinned install from the official index (the +cpu local version matters)
.venv-arm64\Scripts\python -m pip install --upgrade pip
.venv-arm64\Scripts\python -m pip install --no-cache-dir torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv-arm64\Scripts\python -m pip install numpy
```

Expected pip resolution line:

```
Downloading torch-2.12.1%2Bcpu-cp311-cp311-win_arm64.whl (73.4 MB)
```

Notes:

- `uv` is optional convenience, not a requirement: `pip install uv` gets it, or skip it and use the python.org 3.11.x ARM64 installer instead.
- `--no-cache-dir` forces a fresh download so the wheel filename in the log proves provenance. pip's cache is keyed by name, not architecture.
- numpy resolves to a `win_arm64` wheel automatically on an ARM64 interpreter (2.4.6 at time of writing, ASIMD/ARMV8 OpenBLAS).
- Pure-Python packages install fine. Rust-based `tokenizers` and `safetensors` publish `cp310-abi3-win_arm64` wheels and work (as of 2026-07-13).
- Small C extensions without ARM64 wheels (e.g. MarkupSafe) compile locally if you have the MSVC ARM64 build tools (Visual Studio Build Tools, "MSVC v14x C++ ARM64" component).

---

## 4. Verification battery

`verify_arm64_torch.py` (in this repo) runs six checks and exits non-zero on any failure. On an x86-emulated install it fails checks 1/3/5 by design, which makes it double as the "am I emulated?" diagnostic.

```
$ .venv-arm64\Scripts\python verify_arm64_torch.py --expect-version 2.12 --bench
[1] platform.machine() = 'ARM64' (C:\...\Scripts\python.exe)
    PASS: native ARM64 interpreter
[2] torch.__version__ = '2.12.1+cpu'
    PASS: version OK
[3] get_cpu_capability() = 'DEFAULT'
    PASS: non-x86 capability
[4] 2048x2048 matmul OK (0.399s, threads=12)
    PASS: matmul
[5] torch_cpu.dll PE machine = 0xAA64 (ARM64)
    PASS: ARM64 PE binary
[6] softmax/norm/backward OK (loss=16.0000)
    PASS: autograd
[bench] 4096x4096 matmul x10: mean=2.882s min=1.989s max=3.862s (threads=12)
```

Why check 5 exists: `platform.machine()` and `get_cpu_capability()` are both strings an emulated stack could in principle fake. The PE machine type is read directly from the DLL's header bytes (0xAA64 = ARM64, 0x8664 = x64). Prism translates x86 instructions at execution time; a file read returns the real on-disk bytes, so the header check sees the true binary.

Negative control (same script against an x86_64 venv on the same machine):

```
[1] platform.machine() = 'AMD64' (C:\...\old-x86-venv\Scripts\python.exe)
FAIL: interpreter is AMD64, not ARM64 - this Python is an x86_64 binary running under Prism emulation
```

`--allow-emulated` downgrades the architecture checks to warnings so you can benchmark your emulated install as the "before" side of a bake-off.

**Caveat on check 3:** `get_cpu_capability()` is a compile-time string (what the wheel was built for), not a runtime probe of the CPU. Treat it as a build-provenance signal. The runtime proof is a wall-clock comparison (§5); the binary proof is check 5.

---

## 5. Benchmarks: native wheel vs x86-emulated vs WSL2

Same machine, same day, 12 threads, torch 2.12.1+cpu on all three. GEMM = 4096×4096 fp32 matmul ×10. Training-shaped = 4-layer MLP (Linear 1024→4096→1024, GELU, LayerNorm, bs=16, seq=512) forward + backward + AdamW step (`bench_training_shaped.py` in this repo).

| Workload | win_arm64 wheel (native) | x86_64 wheel under Prism | WSL2 Linux aarch64 |
|---|---|---|---|
| 4096² fp32 GEMM ×10, mean | 2.882s | 1.365s | **0.832s** |
| Training-shaped fp32, mean/iter | 44.9s | 13.0s | 17.3s |
| Training-shaped **bf16**, mean/iter | **>180s, never completed** | **>280s, never completed** | **9.1s** |
| Single bf16 GEMM (16,512,1024)@(1024,4096) | **>150s, did not complete** | — | (completes; dry-run below) |
| GEMM thread scaling | 1T: 3.379s, 12T: 2.882s (flat) | — | — |

Read those numbers carefully:

- **The official win_arm64 wheel is 2–3.5× SLOWER than the x86 wheel under emulation** on these workloads. Its build config shows `BLAS_INFO=apl` (Arm Performance Libraries), no MKL-DNN (Intel-only, expected), and **no XNNPACK in this wheel's binary** (`torch.backends.xnnpack` does not exist at runtime). The x86 wheel has MKL 2026.0 + oneDNN 3.11.2 + XNNPACK. The flat 1→12 thread scaling (3.379s → 2.882s, a 1.17× gain from 12 threads) says GEMM threading is largely ineffective for this PyTorch/APL configuration on Windows ARM64; whether the cause is APL kernel dispatch not recognizing Oryon or an OpenMP affinity problem on Windows-on-ARM is still open (both produce the same symptom).
- **bf16 is broken-grade slow on both Windows substrates.** Oryon has no BF16 SIMD, so torch casts through fp32; on the win_arm64 wheel a single mid-size bf16 GEMM did not return in 150 seconds. The x86 wheel under Prism was not better (>280s for the toy loop). If your training script defaults to bf16 (many do), CPU training on this chip needs a substrate with a working bf16 path.
- **WSL2 is the only substrate where bf16 trains.** 9.1s/iter on the toy workload, and a real 1.7B-parameter LoRA dry-run completed end to end (below). The Linux aarch64 wheel explains why: its build config (verified via `torch.__config__.show()`, same upstream commit `7269437d` as the other two wheels) ships OpenBLAS, oneDNN 3.11.2 (`USE_MKLDNN=1`), XNNPACK, QNNPACK, NNPACK, and SLEEF ARM vector kernels. Its bf16 path runs through mature AArch64 software kernels that the Windows APL build simply does not contain.

### Real-model sanity check (WSL2)

A PEFT LoRA dry-run on a 1.83B-parameter base (r=128, α=256, bf16, 1 real example forward pass), identical script and data on both substrates:

| | x86-emulated Windows (earlier record) | WSL2 Linux aarch64 |
|---|---|---|
| Result | PASS, loss ≈ 2.53 | **PASS, loss = 2.5277** |
| Trainable params | 113.8M (6.2%) | 113,770,496 (6.2041%) — exact parity |
| Wall clock | not recorded | 62s total |

Numerical parity across substrates is a good sign that both stacks compute the same math; the difference is purely throughput.

### Honest measurement limits

Single machine, single session, thermal state uncontrolled between runs (Snapdragon X throttles hard under sustained load; the WSL bf16 band 4.1–16.3s/iter reflects that). The GEMM numbers are medians of 10 iterations and are the most trustworthy; treat per-iteration training numbers as bands. The directional conclusions (win_arm64 slower than emulated; bf16 unusable on both Windows substrates; WSL2 fastest) held across every repetition.

---

## 6. The training-stack gap: pyarrow has no win_arm64 wheel

> **UPDATE 2026-07-13 (late evening): this gap is CLOSED.** Apache Arrow C++ 25.0.0 + pyarrow 25.0.0 were built natively for win_arm64 — replicating Arrow's own `msvc-arm64` CI job (MSVC ARM64 + Ninja + bundled deps) — and published with a full field guide, build scripts, and a ready-to-install cp311 wheel: **[LucRoot/Windows_ARM64_PyArrow](https://github.com/LucRoot/Windows_ARM64_PyArrow)** (release `v25.0.0`; 15,196,861 bytes; sha256 `240c476c26a10e7d83d3f899ad66839b819d3d973ee60673a7af787962ea3a3a`). Verified downstream on native Windows ARM64: `datasets 5.0.0` + `trl 1.8.0` install cleanly, the same downstream training pipeline's Cycle 1 dry-run passes with exact parameter parity (113,770,496 trainable / 6.2041%), and its 230-test suite is green. The index survey below remains accurate for *official* wheels — the published wheel is a community build, not on any index. **What still stands:** native Windows is not the recommended *training* substrate, but the reason is now only this document's §3–§5 finding (the official torch wheel's slow APL math with bf16 non-completing) — no longer the install gap. The WSL2 path below remains the training recommendation; the native env is fully usable for data-prep, dry-runs, datasets ops, and eval.

Native torch on Windows is only half the story. A Hugging Face training stack needs `datasets`, which needs `pyarrow`, and as of 2026-07-13 (gap tracked upstream in [apache/arrow#47195](https://github.com/apache/arrow/issues/47195)):

| Source | pyarrow win_arm64 wheels |
|---|---|
| PyPI, all releases 18.1.0–25.0.0 | **none, at any version** |
| download.pytorch.org (stable + nightly) | win_amd64 and macOS arm64 only |
| conda-forge (subdir `win-arm64`) | **0 builds** |

pip's fallback is a source build of pyarrow, which requires the full Apache Arrow C++ toolchain and ~~fails at CMake configuration on this platform~~ — **superseded by the update above: the source build succeeds** when configured after Arrow's own `msvc-arm64` CI job (the earlier configure failure was a config problem, not a platform wall). `trl` is collateral damage (it requires `datasets>=4.7`). Everything else in the stack has wheels: numpy, transformers, peft, accelerate, tokenizers, safetensors all install natively.

### The WSL2 path (what actually works today)

WSL2 on Windows on ARM is Linux aarch64, where the ecosystem is mature: torch, numpy, pyarrow, everything publishes `manylinux_aarch64` wheels.

```bash
# inside WSL2 Ubuntu 24.04
python3 -m venv ~/gs-venv
~/gs-venv/bin/pip install torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
~/gs-venv/bin/pip install numpy transformers peft trl datasets accelerate safetensors
~/gs-venv/bin/pip check   # no broken requirements
```

Three gotchas we hit so you don't have to:

1. **Pin the version.** A bare `pip install transformers trl ...` from PyPI will pull the latest torch (2.13.0 at time of writing, plus ~4 GB of nvidia/cuda packages you do not need on a CPU box) over whatever you installed. If your project pins a torch version, install it pinned from the PyTorch index **after** the other packages, and verify with `pip freeze | grep torch`. (We caught a silent 2.13.0 substitution exactly this way.)
2. **Don't run training data over `/mnt/c`.** Cross-filesystem I/O costs 2–5× on many small files. Copy datasets and model weights into the WSL filesystem (`~/...`) before training. For a 3.6 GB staging tree this took about a minute.
3. **Windows-hardcoded paths in scripts.** If a script hardcodes `Path(r"C:\Users\you\project")`, Linux pathlib reads that as a single relative path component containing backslashes. Creating a directory literally named `C:\Users\you\project` inside your working directory satisfies it with zero script edits. Ugly, effective, and better than forking the script:

   ```bash
   mkdir -p ~/run && cd ~/run
   mkdir -p 'C:\Users\you\project'
   cp -r /mnt/c/Users/you/project/{tools,data,models} 'C:\Users\you\project'/
   ~/gs-venv/bin/python 'C:\Users\you\project/tools/train.py' --dry-run
   ```

4. **Passing multi-line commands through PowerShell → wsl.exe → bash mangles quoting.** Semicolons and quotes got eaten in transit twice during this session, once badly enough to install the wrong torch. Write a `.sh` file and invoke it by path (`wsl -d Ubuntu-24.04 -- bash /mnt/c/path/to/script.sh`) for anything non-trivial.

---

## 7. Version / availability matrix (as of 2026-07-13)

Stable index (`https://download.pytorch.org/whl/cpu/torch/`):

| torch | win_arm64 wheels | Notes |
|---|---|---|
| 2.12.0+cpu | cp311, cp312, cp313 | |
| 2.12.1+cpu | **cp311 only** | the version this session pinned |

Nightly index (`https://download.pytorch.org/whl/nightly/cpu/torch/`): `2.14.0.dev20260713+cpu` win_arm64 for cp311/cp312/cp313.

Linux aarch64 (`manylinux_2_28_aarch64`): full coverage, 2.12.1+cpu for cp310–cp314, on the stable index.

Re-check anytime:

```bash
curl -s https://download.pytorch.org/whl/cpu/torch/ | grep -o 'torch-[^"#<>]*win_arm64[^"#<>]*\.whl' | sort -u
curl -s https://pypi.org/pypi/pyarrow/json | python -c "import json,sys; d=json.load(sys.stdin); print({v: [f['filename'] for f in fs if 'win_arm64' in f['filename']] for v, fs in d['releases'].items() if any('win_arm64' in f['filename'] for f in fs)})"
```

---

## 8. Building from source (pointer, not executed in this session)

If you need a version with no official win_arm64 wheel (e.g. cp312/cp313 at 2.12.1), the source route is:

- Python 3.11+ ARM64, Visual Studio Build Tools with the **MSVC C++ ARM64** component, CMake, Ninja, git.
- `git clone --recursive --depth 1 --branch v2.12.1 --shallow-submodules https://github.com/pytorch/pytorch`
- Environment: `USE_CUDA=0`, `USE_MKLDNN=0` (Intel-only), `USE_XNNPACK=1`, `USE_DISTRIBUTED=0`, `CMAKE_GENERATOR=Ninja`, `MAX_JOBS=8` (leave thermal headroom on a 12-core X Elite).
- Build shell: `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64`, then `python setup.py bdist_wheel`.
- Expect 6–12 hours on an X Elite and ~25–35 GB of build tree. This chip throttles hard under sustained load; budget idle cool-down windows.

This session did not execute the build (the official wheel made it unnecessary for verification, and the pyarrow gap makes it insufficient for training regardless). A self-built wheel with OpenBLAS + XNNPACK might beat the official wheel's APL configuration; that hypothesis is untested here. ~~Note that even a faster self-built Windows wheel cannot fix the pyarrow gap, so Windows-native training stays blocked until the ecosystem catches up regardless of how you build torch.~~ **Superseded 2026-07-13:** the pyarrow gap is closed (see §6 update) — a self-built torch wheel plus the published pyarrow wheel is now a *complete* native stack, and whether it beats WSL2 for training is the remaining untested hypothesis.

---

## 9. Known limitations

- **`get_cpu_capability()` is compile-time provenance**, not a runtime CPU probe (§4).
- **cp311-only wheel at 2.12.1.** pin your Python accordingly, or build from source.
- **No CUDA.** Snapdragon X has no NVIDIA GPU; CPU-only is the target everywhere in this document. Watch out for PyPI's Linux torch pulling ~4 GB of nvidia packages anyway (§6).
- **Thermal throttling.** Sustained all-core load on the X Elite can silently drop throughput 20–100× until the machine idles. Any benchmark run back-to-back against another is suspect; idle 10 minutes between sides of a comparison.
- **bf16 on CPU without BF16 SIMD is a trap** on every substrate we tested except Linux aarch64, and even there it is the slowest-correct option. If your framework lets you choose, fp32 on WSL2 was 17.3s/iter vs bf16's 9.1s/iter on the toy workload, so bf16 still wins there; on Windows-native nothing wins. Related upstream signal: ARM bf16 configurations ship without PyTorch CI test coverage ([pytorch#142703](https://github.com/pytorch/pytorch/issues/142703)).
- **The win_arm64 wheel's BLAS situation may improve.** These numbers are a snapshot of 2.12.1+cpu on one chip. Re-run `verify_arm64_torch.py --bench` on new releases.

---

## 10. Reproduce this document

Everything above was produced by the two scripts in this repo plus the logged commands inline:

- `verify_arm64_torch.py` — the six-check battery + `--bench` GEMM loop + `--allow-emulated` mode for before/after comparisons.
- `bench_training_shaped.py [bf16|fp32] [iters]` — the forward/backward/optimizer-step workload.

Run them against every Python environment you have (`where.exe python` on Windows; each venv's interpreter directly). If any of them reports `AMD64`/`AVX2`, that environment is emulated and the §1 tax applies to everything you run in it.

---

*License: PolyForm Noncommercial 1.0.0. Measurements: take them as one data point from one machine on one day, and re-run the scripts on yours.*

---

## Appendix A: `verify_arm64_torch.py`

```python
#!/usr/bin/env python3
"""verify_arm64_torch.py — prove a Windows PyTorch install is native ARM64,
not x86-emulated under Snapdragon Prism.

Run it with the interpreter you want to test:

    C:\path\to\your\venv\Scripts\python.exe verify_arm64_torch.py
    C:\path\to\your\venv\Scripts\python.exe verify_arm64_torch.py --bench

Exit code 0 = native ARM64 torch confirmed. Non-zero = emulated / wrong
version / broken install. Works on any interpreter; on an x86_64 interpreter
it FAILS checks 1/3/5 by design (that is your "you are emulated" diagnostic).

Checks:
  1. Interpreter architecture is ARM64 (platform.machine()).
  2. torch version matches --expect-version prefix (default: any 2.x).
  3. torch.backends.cpu.get_cpu_capability() is not an x86 SIMD value.
  4. 2048x2048 matmul executes without error.
  5. torch_cpu.dll is an ARM64 PE binary (machine type 0xAA64), read from
     the file header - not from any API the emulation layer could fake.
  6. Extended smoke: softmax / norm / backward pass through autograd.

--bench adds a 4096x4096 matmul x 10 timing loop and prints mean wall time.
Compare the number between an x86-emulated env and a native env on the same
box: expect roughly 2-3x speedup native (Snapdragon X Elite, 2026-07).
"""

import argparse
import platform
import struct
import sys
import time

X86_SIMD_CAPS = {"AVX2", "AVX512", "AVX", "SSE4.2", "FMA"}
PE_MACHINE = {0x8664: "x64 (AMD64)", 0xAA64: "ARM64", 0x014C: "x86 (I386)"}


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def warn(msg):
    print(f"WARN (allowed emulated): {msg}")


def pe_machine_type(dll_path):
    """Read the PE header machine type from a DLL. No OS APIs, just bytes."""
    with open(dll_path, "rb") as f:
        head = f.read(0x40)
        if head[:2] != b"MZ":
            fail(f"{dll_path}: not a PE file (no MZ magic)")
        (e_lfanew,) = struct.unpack("<I", head[0x3C:0x40])
        f.seek(e_lfanew)
        sig = f.read(4)
        if sig != b"PE\x00\x00":
            fail(f"{dll_path}: bad PE signature {sig!r}")
        (machine,) = struct.unpack("<H", f.read(2))
    return machine


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-version", default=None,
                    help="required torch.__version__ prefix, e.g. 2.12")
    ap.add_argument("--bench", action="store_true",
                    help="run a 4096x4096 matmul x10 timing loop")
    ap.add_argument("--allow-emulated", action="store_true",
                    help="downgrade architecture checks to warnings - for "
                         "benchmarking an x86-emulated install against a "
                         "native one")
    args = ap.parse_args()
    gate = warn if args.allow_emulated else fail

    # Check 1 - interpreter architecture
    machine = platform.machine()
    print(f"[1] platform.machine() = {machine!r} "
          f"({sys.executable})")
    if machine != "ARM64":
        gate(f"interpreter is {machine}, not ARM64 - this Python is "
             f"an x86_64 binary running under Prism emulation")
    else:
        print("    PASS: native ARM64 interpreter")

    import torch  # imported after check 1 so the error message is ours

    # Check 2 - version
    print(f"[2] torch.__version__ = {torch.__version__!r}")
    if args.expect_version and not torch.__version__.startswith(args.expect_version):
        fail(f"expected version prefix {args.expect_version!r}")
    print("    PASS: version OK")

    # Check 3 - CPU capability is not x86 SIMD
    cap = torch.backends.cpu.get_cpu_capability()
    print(f"[3] get_cpu_capability() = {cap!r}")
    if cap in X86_SIMD_CAPS:
        gate(f"capability {cap!r} is x86 SIMD - torch binary is x86_64")
    else:
        print("    PASS: non-x86 capability")

    # Check 4 - matmul sanity
    t0 = time.perf_counter()
    x = torch.randn(2048, 2048)
    (x @ x).sum().item()
    dt = time.perf_counter() - t0
    print(f"[4] 2048x2048 matmul OK ({dt:.3f}s, threads={torch.get_num_threads()})")
    print("    PASS: matmul")

    # Check 5 - PE machine type of the core DLL
    import pathlib
    dll = pathlib.Path(torch.__file__).parent / "lib" / "torch_cpu.dll"
    m = pe_machine_type(dll)
    print(f"[5] {dll.name} PE machine = 0x{m:04X} ({PE_MACHINE.get(m, 'unknown')})")
    if m != 0xAA64:
        gate(f"torch_cpu.dll is {PE_MACHINE.get(m, hex(m))}, not ARM64")
    else:
        print("    PASS: ARM64 PE binary")

    # Check 6 - extended smoke: ops + autograd
    a = torch.randn(256, 256, requires_grad=True)
    b = torch.softmax(a @ a.T, dim=-1)
    loss = torch.linalg.norm(b)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    print(f"[6] softmax/norm/backward OK (loss={loss.item():.4f})")
    print("    PASS: autograd")

    if args.bench:
        import statistics
        big = torch.randn(4096, 4096)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            (big @ big).sum().item()
            times.append(time.perf_counter() - t0)
        print(f"[bench] 4096x4096 matmul x10: mean={statistics.mean(times):.3f}s "
              f"min={min(times):.3f}s max={max(times):.3f}s "
              f"(threads={torch.get_num_threads()})")

    if args.allow_emulated and platform.machine() != "ARM64":
        print("\nCOMPLETED WITH EMULATION WARNINGS - this is an x86-emulated "
              "install. Use these numbers only as the 'before' side of a "
              "bake-off.")
    else:
        print("\nALL CHECKS PASSED - native ARM64 PyTorch confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Appendix B: `bench_training_shaped.py`

```python
import sys, time, statistics, platform
import torch, torch.nn as nn, torch.nn.functional as F

dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[1] if len(sys.argv) > 1 else "bf16"]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20

torch.manual_seed(0)
print(f"interpreter: {platform.machine()} | torch {torch.__version__} | threads={torch.get_num_threads()} | dtype={dtype} iters={iters}")

class Block(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.l1 = nn.Linear(d, d * 4)
        self.l2 = nn.Linear(d * 4, d)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        h = F.gelu(self.l1(x))
        return self.norm(x + self.l2(h))

model = nn.Sequential(*[Block() for _ in range(4)]).to(dtype)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randn(16, 512, 1024, dtype=dtype)

t0 = time.perf_counter()
opt.zero_grad(); loss = model(x).sum(); loss.backward(); opt.step()
print(f"  first iter: {time.perf_counter()-t0:.3f}s (warmup, includes JIT/alloc)")

times = []
for i in range(iters):
    t0 = time.perf_counter()
    opt.zero_grad(); loss = model(x).sum(); loss.backward(); opt.step()
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  iter {i+1}: {dt:.3f}s")

print(f"RESULT {platform.machine()} {dtype}: mean={statistics.mean(times):.3f}s "
      f"min={min(times):.3f}s max={max(times):.3f}s loss={loss.item():.1f}")
```

---

**Author:** Dr. Lucas Root, Ph.D. — [info@lucasroot.com](mailto:info@lucasroot.com)
