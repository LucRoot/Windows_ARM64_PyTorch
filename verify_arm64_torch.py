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
